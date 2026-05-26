module network_community_detection

#   Module Packages
    using DataFrames
    using SparseArrays
    using LinearAlgebra
    using Random
    using Distributions
    using ProgressMeter
    using ..Network_Credible_Intervals  # Parent module that exports all the functions

#	Helper Function for degree calculations: edge list to sparse adjacency matrix
	function _edgelist_to_sparse_matrix(edges::DataFrame; weighted::Bool=true)
		"""
		Args:
			edges::DataFrame: DataFrame with src, dst, and optionally weight columns
			weighted::Bool: use weights if true and available (default = true)
		Returns:
			Tuple{SparseMatrixCSC{Float64,Int64}, Dict{Any,Int}, Vector{Any}}
		Notes:
			Returns (adj_matrix, node_to_idx, idx_to_node).
			Handles arbitrary node identifiers.
		"""
		
		#	Extract unique nodes and create mappings
			all_nodes = unique(vcat(edges.src, edges.dst))
			n = length(all_nodes)
			node_to_idx = Dict(node => i for (i, node) in enumerate(all_nodes))
			idx_to_node = all_nodes
		
		#	Map edges to indices
			src_idx = [node_to_idx[s] for s in edges.src]
			dst_idx = [node_to_idx[d] for d in edges.dst]
		
		#	Determine weights
			if weighted && hasproperty(edges, :weight)
				#	Use provided weights
					weights = Float64.(edges.weight)
			else
				#	Unweighted: use 1.0 for all edges
					weights = ones(Float64, nrow(edges))
			end
		
		#	Build sparse adjacency matrix
			adj_matrix = sparse(src_idx, dst_idx, weights, n, n)
		
		#	Return matrix and mappings
			return (adj_matrix, node_to_idx, idx_to_node)
	end

#	Helper: graph (nodes + edges) to sparse adjacency with fixed node universe
	function _graph_to_sparse_matrix(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
									weighted::Bool=true)
		"""
		Args:
			edges::DataFrame
				Required columns: :src, :dst
				Optional column:  :weight
				src/dst are node IDs (treated as String; supports long IDs)
		
			nodes::Union{Nothing,DataFrame,Vector{<:AbstractString}}
				Nothing  → infer nodes from edges (isolates excluded)
				DataFrame: columns :id and :label (both string vectors). Uses :id as the ID universe.
				Vector   : string vector of node IDs forming the ID universe (includes isolates, if any)
		
			weighted::Bool
				If true and edges has :weight, use it; otherwise use ones.
				If false, ignore any :weight column and use ones.
		
		Returns:
			Tuple{SparseMatrixCSC{Float64,Int64}, Dict{Any,Int}, Vector{Any}}
				(adj_matrix, node_to_idx, idx_to_node)
		
		Notes:
			- When `nodes` is provided, the returned matrix is sized to that universe
			(so isolates are included). All edge endpoints must exist in `nodes`.
			- When `nodes` is not provided, falls back to `_edgelist_to_sparse_matrix`
			which infers the node set from edge endpoints only.
		"""

		#	Basic validation for edge columns
			@assert hasproperty(edges, :src) && hasproperty(edges, :dst) "_graph_to_sparse_matrix: edges must have :src and :dst"

		#	Fallback: no nodes supplied → just delegate to the existing helper
			if nodes === nothing
				return _edgelist_to_sparse_matrix(edges; weighted=weighted)
			end

		#	Build the fixed node universe (idx_to_node) and mapping (node_to_idx)
			ids = String[]
			if nodes isa DataFrame
				#	Nodes as a DataFrame of IDs and Labels (Screen Names)
					ndf = nodes::DataFrame
					@assert hasproperty(ndf, :id) && hasproperty(ndf, :label) "_graph_to_sparse_matrix: nodes DataFrame must have :id and :label"
					ids = String.(ndf.id)
			else
				#	Vector of node IDs
					ids = String.(nodes::AbstractVector{<:AbstractString})
			end

		#	Specifyign Node Specific Return Objects
			n = length(ids)
			node_to_idx = Dict{Any,Int}(id => i for (i, id) in enumerate(ids))
			if(typeof(nodes) == DataFrame)
				idx_to_node = nodes
			else
				idx_to_node = Vector{Any}(ids)  # keep Any to match requested return type
			end

		#	Map edge endpoints to indices (validate all endpoints are known)
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)

			unknown_src = Set{String}(s for s in src_ids if !haskey(node_to_idx, s))
			unknown_dst = Set{String}(d for d in dst_ids if !haskey(node_to_idx, d))
			if !isempty(unknown_src) || !isempty(unknown_dst)
				missing_ids = union(unknown_src, unknown_dst)
				examples = join(collect(Iterators.take(missing_ids, 5)), ", ")
				throw(ArgumentError("_graph_to_sparse_matrix: edges reference IDs not present in supplied nodes (examples: $examples)"))
			end

			src_idx = [node_to_idx[s] for s in src_ids]
			dst_idx = [node_to_idx[d] for d in dst_ids]

		#	Determine edge weights per spec
			use_weights = weighted && hasproperty(edges, :weight)
			weights = use_weights ? Float64.(edges.weight) : ones(Float64, nrow(edges))

		#	Construct sparse adjacency (no symmetrization here; caller decides)
			adj_matrix = sparse(src_idx, dst_idx, weights, n, n)

		#	Return adjacency and mappings
			return (adj_matrix, node_to_idx, idx_to_node)
	end

#	Helper Function for degree calculations: aggregate duplicate edges
	function _aggregate_multi_edges(edges::DataFrame; agg_func::Function=sum)
		"""
		Args:
			edges::DataFrame: DataFrame with src, dst, and optionally weight columns
			agg_func::Function: aggregation function for duplicate edges (default = sum)
		Returns:
			DataFrame: edges with duplicates aggregated
		Notes:
			Handles agg_func even when no weights exist.
			When no weights exist and agg_func=maximum, creates binary presence.
		"""
		
		#	Check if weights exist
			has_weights = hasproperty(edges, :weight)
		
		#	Group and aggregate
			if has_weights
				#	Aggregate weights for duplicate edges
					grouped = combine(groupby(edges, [:src, :dst]), 
					                 :weight => agg_func => :weight)
			else
				#	Handle based on agg_func
					if agg_func == maximum
						#	For maximum without weights: binary presence (any edge = 1)
							grouped = combine(groupby(edges, [:src, :dst])) do _
								DataFrame(weight = 1.0)
							end
					elseif agg_func == sum
						#	For sum without weights: count edges
							grouped = combine(groupby(edges, [:src, :dst]), 
							                 nrow => :weight)
					else
						#	For other functions: apply to ones
							grouped = combine(groupby(edges, [:src, :dst])) do grp
								DataFrame(weight = agg_func(ones(nrow(grp))))
							end
					end
			end
		
		#	Return aggregated edges
			return grouped
	end

#   Helper: symmetric sparse adjacency to undirected, collapsed edge list
    function _symmetric_sparse_to_undirected_edgelist(adj::SparseMatrixCSC{T,Int};
                                                    include_diagonal::Bool = true,
                                                    agg_func::Function = maximum,
                                                    node_map::Union{Nothing,Dict{Any,Int}} = nothing) where {T<:Real}
        """
        Assumes:
            - `adj` is symmetric (adj[i, j] == adj[j, i]).

        Behavior:
            - Treats each unordered pair {i, j} as a single edge.
            - Uses `src = min(i, j)`, `dst = max(i, j)` as a canonical orientation
            in index space.
            - Collapses duplicates by grouping on (src, dst) and aggregating weights
            with `agg_func` (default = maximum, which matches a binarized ORA view).
            - If `node_map` is supplied, maps indices back to original node IDs
            in the returned `:src` and `:dst` columns, and then enforces a
            canonical ordering of labels per edge (string(src) ≤ string(dst)).

        Arguments:
            adj::SparseMatrixCSC{T,Int}
                Symmetric adjacency matrix.

            include_diagonal::Bool
                If true  → keep self-loops (i == j).
                If false → drop self-loops.

            agg_func::Function
                Aggregation function for combining multiple weights for the same
                unordered pair (default = maximum; use `sum` for counts, etc.).

            node_map::Union{Nothing,Dict{Any,Int}}
                Optional mapping from original node IDs → matrix indices, as
                returned by `_graph_to_sparse_matrix` (the `node_to_idx`
                dictionary). If provided, the output `:src` and `:dst` will be
                original node IDs; otherwise they will be Int indices.

        Returns:
            DataFrame with columns:
                :src
                :dst
                :weight :: Float64

            - If `node_map === nothing`, :src and :dst are Int indices.
            - If `node_map !== nothing`, :src and :dst are original node IDs,
            with string(src) ≤ string(dst) for all rows.
        """
        #   Extract all nonzeros
            I, J, V = findnz(adj)

        #   Canonical orientation for unordered pairs (index space)
            src = min.(I, J)
            dst = max.(I, J)

        #   Optionally drop self-loops
            mask = include_diagonal ? trues(length(src)) : (src .!= dst)

            df = DataFrame(
                src    = src[mask],
                dst    = dst[mask],
                weight = Float64.(V[mask]),
            )

        #   Collapse duplicates so each {src, dst} appears once
            df_agg = combine(groupby(df, [:src, :dst]), :weight => agg_func => :weight)

        #   If no node_map is provided, leave indices as-is
            if node_map === nothing
                return df_agg
            end

        #   Invert node_map: index → original node ID
            idx_to_node = Dict{Int,Any}()
            for (node_id, idx) in node_map
                idx_to_node[idx] = node_id
            end

        #   Map indices back to original node IDs
            src_ids = [idx_to_node[i] for i in df_agg.src]
            dst_ids = [idx_to_node[j] for j in df_agg.dst]

            df_agg.src = src_ids
            df_agg.dst = dst_ids

        #   Enforce a canonical ordering on labels per edge:
        #   for undirected graphs, ensure string(src) ≤ string(dst)
            for row in eachrow(df_agg)
                if string(row.src) > string(row.dst)
                    tmp = row.src
                    row.src = row.dst
                    row.dst = tmp
                end
            end

        #   Return Edgelist with original node labels and stable src/dst order
            return df_agg
    end

#	Helper Function: Check Matrix Symmetry
	function _is_symmetric(adj::SparseMatrixCSC{<:Real,Int}; 
	                      directed::Union{Bool,Nothing}=nothing, 
	                      atol::Float64=1e-12)
		"""
		Args:
			adj::SparseMatrixCSC: matrix to check
			directed::Union{Bool,Nothing}: graph type or nothing for pure check
			atol::Float64: absolute tolerance (default = 1e-12)
		Returns:
			Bool: true if symmetric within tolerance
		Notes:
			If directed=nothing: checks actual numerical symmetry
			If directed=false: returns true (undirected assumed symmetric)
			If directed=true: checks actual numerical symmetry
		"""
		
		#	Validation
			if size(adj, 1) != size(adj, 2)
				throw(ArgumentError("Adjacency must be square"))
			end
		
		#	Convention-Based Check
			if directed === false
				return true  # Undirected => symmetric by convention
			end
		
		#	Numerical Symmetry Check
			delta = adj - adj'
			return LinearAlgebra.norm(delta, 1) <= atol
	end

#	Helper Function for leiden_community_detection: Detect Binary Matrix
	function _is_binary_matrix(A::SparseMatrixCSC; directed::Bool, atol::Float64=1e-12)
		"""
		Args:
			A::SparseMatrixCSC: matrix to check
			directed::Bool: expected diagonal convention
			atol::Float64: absolute tolerance (default = 1e-12)
		Returns:
			Bool: true if binary under convention
		Notes:
			Binary means off-diagonal ∈ {0,1}, diagonal ∈ {0,1} if directed
			or {0,2} if undirected.
		"""
		
		#	Extract Non-Zero Elements
			rows, cols, vals = findnz(A)
		
		#	Check Each Non-Zero Value
			for k in eachindex(vals)
				i, j = rows[k], cols[k]
				v = vals[k]
				
				if i == j
					#	Diagonal Elements
						if directed
							valid = abs(v) ≤ atol || abs(v - 1.0) ≤ atol
						else
							valid = abs(v) ≤ atol || abs(v - 2.0) ≤ atol
						end
						if !valid
							return false
						end
				else
					#	Off-Diagonal Elements
						if !(abs(v) ≤ atol || abs(v - 1.0) ≤ atol)
							return false
						end
				end
			end
		
		#	All Values Valid
			return true
	end

#	Helper Function for leiden_community_detection: Binarize Matrix
	function _binarize_matrix(A::SparseMatrixCSC; directed::Bool)
		"""
		Args:
			A::SparseMatrixCSC: matrix to binarize
			directed::Bool: graph type for diagonal convention
		Returns:
			SparseMatrixCSC: binarized copy of matrix
		Notes:
			Off-diagonal → {0,1}, diagonal → {0,1} if directed
			or {0,2} if undirected. Symmetrizes if undirected.
		"""
		
		#	Create Working Copy
			B = copy(A)
		
		#	Binarize Off-Diagonal Elements
			@inbounds for j in 1:size(B, 2)
				for p in B.colptr[j]:(B.colptr[j+1] - 1)
					i = B.rowval[p]
					if i != j
						B.nzval[p] = (B.nzval[p] > 0) ? 1.0 : 0.0
					end
				end
			end
		
		#	Set Diagonal Convention
			if directed
				d = (diag(B) .> 0) .* 1.0
			else
				d = (diag(B) .> 0) .* 2.0
			end
			B = B + spdiagm(0 => (d .- diag(B)))
		
		#	Symmetrize for Undirected
			if !directed
				B = max.(B, B')
			end
		
		#	Return Binarized Matrix
			return B
	end

#	Helper Function for freeman_degree_normalization: bipartite mode counts
	function _bipartite_counts(types::AbstractVector{Bool})
		"""
		Args:
			types::AbstractVector{Bool}: vertex modes; true = first mode, false = second
		Returns:
			Tuple{Int,Int}: (first_mode_count, second_mode_count)
		Notes:
			Aligns with Python reference where counts are derived from V(type).
		"""

		#	Validation
			if isempty(types)
				throw(ArgumentError("types vector must not be empty"))
			end

		#	Count modes
			first_mode = count(types)
			second_mode = length(types) - first_mode

		#	Return counts
			return (first_mode, second_mode)
	end

#	Helper Function for calculate_modularity: Compact Membership Remap
	function _remap_membership(membership::Vector{Int})
		"""
		Args:
			membership::Vector{Int}: community assignments (any integer labels)
		Returns:
			Tuple{Vector{Int}, Int}: (remapped membership in 1..C, number of communities C)
		Notes:
			Remaps arbitrary integer labels to contiguous 1..C indexing without
			allocating a Dict for small label sets. Uses sortperm for stable
			ordering so that downstream code sees deterministic community ids.
		"""

		#	Identify Unique Labels
			labels = sort(unique(membership))
			C = length(labels)

		#	Fast Path: Already in 1..C
			if C == length(membership)
				#	Each node in its own community; trivial mapping
					return (collect(1:C), C)
			end
			if labels[1] == 1 && labels[end] == C
				#	Already 1..C contiguous; no remap needed
					return (membership, C)
			end

		#	Build Label-to-Index Lookup via Sorted Search
			n = length(membership)
			mapped = Vector{Int}(undef, n)
			@inbounds for i in 1:n
				mapped[i] = searchsortedfirst(labels, membership[i])
			end

		#	Return Remapped Membership and Community Count
			return (mapped, C)
	end

#	Helper Function for calculate_modularity: Symmetrize Adjacency for Undirected
	function _symmetrize_for_undirected(adj::SparseMatrixCSC{Float64,Int}, weighted::Bool)
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: adjacency matrix
			weighted::Bool: if true, average forward and reverse; if false, logical OR
		Returns:
			SparseMatrixCSC{Float64,Int}: symmetrized adjacency
		Notes:
			For undirected modularity, igraph convention requires symmetric input.
			Skips the work entirely if adj is already symmetric (cheap to check
			structurally and avoids allocation in the common case).
		"""

		#	Check Symmetry Cheaply
			if adj == adj'
				#	Already symmetric; no work needed
					return adj
			end

		#	Symmetrize
			if weighted
				#	Average forward and reverse weights
					return 0.5 .* (adj + adj')
			else
				#	Logical OR via element-wise max
					return max.(adj, adj')
			end
	end

#	Helper Function for calculate_modularity: Block-Diagonal Internal Edge Weight
	function _internal_edge_weight(adj::SparseMatrixCSC{Float64,Int}, mapped::Vector{Int})
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: adjacency matrix
			mapped::Vector{Int}: membership remapped to 1..C
		Returns:
			Float64: sum of edge weights where source and target are in same community
		Notes:
			Replaces the S' * adj * S matmul plus diagonal extraction. Iterates
			directly over the CSC nonzeros, adding edge weight w_ij to the running
			sum whenever mapped[i] == mapped[j]. O(nnz) with no allocation.
		"""

		#	Extract CSC Internals
			rows = rowvals(adj)
			vals = nonzeros(adj)
			n    = size(adj, 2)

		#	Accumulate Intra-Community Edge Weight
			internal = 0.0
			@inbounds for j in 1:n
				cj = mapped[j]
				for k in nzrange(adj, j)
					i = rows[k]
					if mapped[i] == cj
						internal += vals[k]
					end
				end
			end

		#	Return Internal Weight Sum
			return internal
	end

#	Helper Function for calculate_modularity: Community Degree Sums
	function _community_degree_sums(degree::Vector{Float64}, mapped::Vector{Int}, C::Int)
		"""
		Args:
			degree::Vector{Float64}: per-node degree (or strength) vector
			mapped::Vector{Int}: membership remapped to 1..C
			C::Int: number of communities
		Returns:
			Vector{Float64}: per-community total degree
		Notes:
			Replaces S' * degree as a single linear pass with no sparse allocation.
		"""

		#	Accumulate Per-Community Sum
			K = zeros(Float64, C)
			n = length(degree)
			@inbounds for i in 1:n
				K[mapped[i]] += degree[i]
			end

		#	Return Community Degree Vector
			return K
	end

#	Calculate Modularity for Leiden, CHAMP, and Modularity Vitality Functions
	function calculate_modularity(adj::SparseMatrixCSC, membership::Vector{Int};
	                              weighted::Bool = true,
	                              directed::Bool = false,
	                              γ::Float64     = 1.0)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency matrix (may contain weights)
			membership::Vector{Int}: community assignment for each node
			weighted::Bool: use edge weights if true (default = true)
			directed::Bool: treat as directed graph (default = false)
			γ::Float64: resolution parameter (default = 1.0)
		Returns:
			Float64: modularity score
		Notes:
			Drop-in replacement for the prior calculate_modularity. Matches igraph's
			modularity calculation for weighted/unweighted and directed/undirected
			graphs. Self-loops handled correctly.

			Optimization: replaces the S' * adj * S sparse matmul with direct
			iteration over CSC nonzeros, accumulating into small dense vectors of
			length C (number of communities). Eliminates intermediate allocation
			and is O(nnz + n) per call rather than O(nnz * C).

			For repeated calls in a hot loop with fixed adj, prefer
			calculate_modularity_cached which avoids recomputing degrees and total
			weight on every call.
		"""

		#	Validation
			n = size(adj, 1)
			@assert size(adj, 2) == n "calculate_modularity: adj must be square"
			@assert length(membership) == n "membership length mismatch with adjacency matrix"

		#	Type Conversion for Consistency
			adj = SparseMatrixCSC{Float64,Int}(adj)

		#	Handle Unweighted Case
			if !weighted
				I, J, _ = findnz(adj)
				adj = sparse(I, J, ones(Float64, length(I)), n, n)
			end

		#	Compact Membership to 1..C
			mapped, C = _remap_membership(membership)

		#	Branch on Graph Type
			if directed
				#	Directed Degree Calculation
					k_out = vec(sum(adj, dims = 2))
					k_in  = vec(sum(adj, dims = 1))
					m     = sum(adj)

					if m == 0.0
						return 0.0
					end

				#	Internal Edge Weight (via direct CSC iteration)
					internal = _internal_edge_weight(adj, mapped)

				#	Community Degree Sums
					K_out = _community_degree_sums(k_out, mapped, C)
					K_in  = _community_degree_sums(k_in,  mapped, C)

				#	Expected Edges (Directed Null Model)
					expected = 0.0
					@inbounds for c in 1:C
						expected += (K_out[c] * K_in[c]) / m
					end

				#	Return Directed Modularity
					return (internal - γ * expected) / m

			else
				#	Symmetrize for Undirected
					adj = _symmetrize_for_undirected(adj, weighted)

				#	Undirected Degree Calculation
					k     = vec(sum(adj, dims = 2))
					two_m = sum(adj)

					if two_m == 0.0
						return 0.0
					end

				#	Internal Edge Weight (via direct CSC iteration)
					internal = _internal_edge_weight(adj, mapped)

				#	Community Degree Sums
					K = _community_degree_sums(k, mapped, C)

				#	Expected Weight (Undirected Null Model)
					expected = 0.0
					@inbounds for c in 1:C
						expected += (K[c]^2) / two_m
					end

				#	Return Undirected Modularity
					return (internal - γ * expected) / two_m
			end
	end
	@doc raw"""
	**Description**
	Calculate the modularity of a graph with respect to a given community partition. Supports both weighted/unweighted and directed/undirected graphs, matching igraph's implementation. Handles self-loops correctly and accepts arbitrary integer community labels (not required to be contiguous).

	**Usage**
	`calculate_modularity(adj::SparseMatrixCSC, membership::Vector{Int}; weighted::Bool=true, directed::Bool=false, γ::Float64=1.0)`

	**Arguments**
	- `adj::SparseMatrixCSC`: Adjacency matrix of the graph (may contain edge weights). Must be square; `size(adj, 1) == size(adj, 2)`.
	- `membership::Vector{Int}`: Community assignment for each node. Length must equal `size(adj, 1)`. Labels need not be contiguous; the function remaps internally.
	- `weighted::Bool`: If `true`, use the edge weights stored in `adj`; if `false`, binarize the adjacency matrix before computing modularity (default `true`).
	- `directed::Bool`: If `true`, use the directed-modularity formula with separate in- and out-degree null models. If `false`, symmetrize the adjacency first and use the undirected formula (default `false`).
	- `γ::Float64`: Resolution parameter for the generalized modularity formula (default `1.0`).

	**Details**
	Implements the Newman-Girvan modularity score:

	- **Undirected:** `Q = (1 / 2m) Σ_ij [A_ij − γ · k_i · k_j / (2m)] δ(c_i, c_j)`
	- **Directed:** `Q = (1 / m) Σ_ij [A_ij − γ · k_i^out · k_j^in / m] δ(c_i, c_j)`

	where `A_ij` is the (weighted) adjacency entry, `k_i` is node `i`'s degree (or strength, for weighted graphs), `c_i` is `i`'s community label, and `m` is the total edge weight (or twice the total, in the undirected case).

	The resolution parameter `γ` controls the trade-off between intra-community edge density and community size:
	- `γ < 1.0`: Favors larger communities.
	- `γ = 1.0`: Standard Newman-Girvan modularity.
	- `γ > 1.0`: Favors smaller, denser communities.

	For undirected calculations, the matrix is symmetrized via averaging (`0.5 * (A + A')`) for weighted graphs and element-wise maximum for unweighted. Self-loops are preserved correctly per the igraph convention.

	**Performance Notes**
	This implementation avoids the standard `S' * A * S` block-matrix computation, which allocates a `C × C` sparse intermediate when only the diagonal is needed. Instead, the function iterates directly over the CSC nonzeros of the adjacency matrix and accumulates intra-community edge weights in a single linear pass. Memory allocation is bounded by `O(C)` for the community-degree vector rather than `O(C^2)` for the full block matrix. Operation count is `O(nnz + n)` rather than `O(nnz · C)`.

	For repeated calls in a hot loop with a fixed adjacency matrix (e.g., Leiden's local-move phase), prefer `calculate_modularity_cached`, which accepts a precomputed cache of degrees and total weight.

	**Value**
	Returns a `Float64` in the range `[-1, 1]`. Higher values indicate stronger community structure. A value of `0.0` corresponds to no better-than-random community structure (e.g., when the graph has no edges or when the partition is trivial).

	**Examples**
```julia
		using SparseArrays

		#	Simple unweighted undirected graph
			adj = sparse([1,2,3], [2,3,1], ones(3), 3, 3)
			membership = [1, 1, 2]
			Q = calculate_modularity(adj, membership; weighted = false)

		#	Weighted directed graph
			adj = sparse([1,2], [2,3], [0.5, 1.0], 3, 3)
			Q = calculate_modularity(adj, membership; directed = true)

		#	Resolution sweep for community-size sensitivity
			gammas = collect(0.5:0.1:2.0)
			Qs = [calculate_modularity(adj, membership; γ = γ) for γ in gammas]
```

	**See Also**
	`calculate_modularity_cached`, `leiden_community_detection`, `champ_community_detection`

	**References**
	Newman & Girvan (2004). "Finding and evaluating community structure in networks." Phys. Rev. E 69:026113.

	Reichardt & Bornholdt (2006). "Statistical mechanics of community detection." Phys. Rev. E 74:016110.
	""" calculate_modularity

#	Helper Function for delta_modularity_best!: Undirected Best-Move Evaluation
	function delta_modularity_undirected_best!(adj::SparseMatrixCSC{Float64,Int},
	                                            i::Int,
	                                            c_old::Int,
	                                            membership::Vector{Int},
	                                            k::Vector{Float64},
	                                            K::Vector{Float64},
	                                            two_m::Float64,
	                                            node_to_comm::Vector{Float64};
	                                            γ::Float64 = 1.0)
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: symmetric weighted adjacency (preprocessed)
			i::Int: index of the node being evaluated
			c_old::Int: current community of node i
			membership::Vector{Int}: current community labels (1..C)
			k::Vector{Float64}: per-node degree vector
			K::Vector{Float64}: per-community total degree vector
			two_m::Float64: total weight × 2 (i.e., sum(adj) for a symmetric adj)
			node_to_comm::Vector{Float64}: pre-allocated work vector of length ≥ C; modified in place
			γ::Float64: resolution parameter (default = 1.0)
		Returns:
			Tuple{Int, Float64}: (best_community, best_ΔQ); returns (c_old, 0.0) if no
			move improves modularity
		Notes:
			Walks node i's neighbors once, binning edge weights by neighbor community
			into node_to_comm. Then evaluates ΔQ for each community that received any
			weight and returns the best candidate.

			The work vector is reset via tracked-index zeroing (not a full clear) so
			repeated use is O(degree of i) per call.

			Δ formula (undirected, single move from c_old to c_new), with m = two_m/2:
				ΔQ = (1/m) × [
					(k_{i,c_new} - k_{i,c_old})
					- γ × k_i × (K[c_new] - K[c_old] + k_i) / (2m)
				]
			Equivalently, using two_m directly:
				ΔQ = (2/two_m) × (k_{i,c_new} - k_{i,c_old})
				   - γ × k_i × (K[c_new] - K[c_old] + k_i) × (2 / two_m^2)
			where k_{i,c} = sum of edge weights from i into community c (excluding
			self-loops).
		"""

		#	Adjacency Internals
			rows = rowvals(adj)
			vals = nonzeros(adj)

		#	Walk Neighbors and Bin by Community (Track Touched Indices)
			touched = Int[]
			sizehint!(touched, 16)
			@inbounds for nz in nzrange(adj, i)
				j = rows[nz]
				if j == i
					continue                        # skip self-loops
				end
				c_j = membership[j]
				if node_to_comm[c_j] == 0.0
					push!(touched, c_j)             # first edge into this community
				end
				node_to_comm[c_j] += vals[nz]
			end

		#	Cache Loop-Invariant Quantities
			k_i           = k[i]
			K_old         = K[c_old]
			k_i_old       = node_to_comm[c_old]     # weight from i into its current community
			inv_m         = 2.0 / two_m             # = 1/m  where m = two_m/2
			inv_two_m_sq  = 2.0 / (two_m * two_m)   # = 1/(m * two_m)
			penalty_old   = γ * k_i * (K_old - k_i) * inv_two_m_sq

		#	Evaluate Each Candidate and Track Best
			best_c  = c_old
			best_ΔQ = 0.0
			@inbounds for c_new in touched
				if c_new == c_old
					continue                        # not a move
				end
				k_i_new  = node_to_comm[c_new]
				gain     = (k_i_new - k_i_old) * inv_m
				penalty  = γ * k_i * K[c_new] * inv_two_m_sq - penalty_old
				ΔQ       = gain - penalty
				if ΔQ > best_ΔQ
					best_ΔQ = ΔQ
					best_c  = c_new
				end
			end

		#	Reset Touched Entries in Work Vector
			@inbounds for c in touched
				node_to_comm[c] = 0.0
			end

		#	Return Best Candidate
			return (best_c, best_ΔQ)
	end

#	Helper Function for delta_modularity_best!: Directed Best-Move Evaluation
	function delta_modularity_directed_best!(adj::SparseMatrixCSC{Float64,Int},
	                                          i::Int,
	                                          c_old::Int,
	                                          membership::Vector{Int},
	                                          k_in::Vector{Float64},
	                                          k_out::Vector{Float64},
	                                          K_in::Vector{Float64},
	                                          K_out::Vector{Float64},
	                                          m::Float64,
	                                          node_to_comm_in::Vector{Float64},
	                                          node_to_comm_out::Vector{Float64};
	                                          γ::Float64 = 1.0)
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: directed weighted adjacency
			i::Int: index of the node being evaluated
			c_old::Int: current community of node i
			membership::Vector{Int}: current community labels (1..C)
			k_in::Vector{Float64}: per-node in-degree
			k_out::Vector{Float64}: per-node out-degree
			K_in::Vector{Float64}: per-community total in-degree
			K_out::Vector{Float64}: per-community total out-degree
			m::Float64: total edge weight (sum(adj) for directed)
			node_to_comm_in::Vector{Float64}: work vector for incoming edges; modified in place
			node_to_comm_out::Vector{Float64}: work vector for outgoing edges; modified in place
			γ::Float64: resolution parameter (default = 1.0)
		Returns:
			Tuple{Int, Float64}: (best_community, best_ΔQ); returns (c_old, 0.0) if no
			move improves modularity
		Notes:
			Walks node i's outgoing edges (CSC column) and incoming edges (scan across
			columns for entries with row i) once each, binning edge weights into the
			two work vectors. Then evaluates ΔQ for each community reachable from i
			and returns the best candidate.

			Both work vectors must have length ≥ length(K_in). They are reset via
			tracked-index zeroing at function exit.

			Δ formula (directed, single move from c_old to c_new):
				ΔQ = (1/m) × [
					(k_{i,c_new}^in + k_{i,c_new}^out - k_{i,c_old}^in - k_{i,c_old}^out)
				] - γ × [
					k_i^out × (K_in[c_new]  - K_in[c_old]  + k_i^in) +
					k_i^in  × (K_out[c_new] - K_out[c_old] + k_i^out)
				] / m^2
			where k_{i,c}^in is the weight of edges from community c into i and
			k_{i,c}^out is the weight of edges from i into community c.

			Incoming-edge walk is currently O(nnz) because it scans all columns of
			adj for entries with row i. For improved performance on large graphs,
			a future revision will accept a pre-computed transpose (adj') for CSR-
			style row access via nzrange(adj', i).
		"""

		#	Adjacency Internals
			rows = rowvals(adj)
			vals = nonzeros(adj)

		#	Walk Outgoing Edges (Column i of CSC)
			touched_out = Int[]
			sizehint!(touched_out, 16)
			@inbounds for nz in nzrange(adj, i)
				j = rows[nz]
				if j == i
					continue
				end
				c_j = membership[j]
				if node_to_comm_out[c_j] == 0.0
					push!(touched_out, c_j)
				end
				node_to_comm_out[c_j] += vals[nz]
			end

		#	Walk Incoming Edges (Scan All Columns for Row i)
			touched_in = Int[]
			sizehint!(touched_in, 16)
			n_cols = size(adj, 2)
			@inbounds for col in 1:n_cols
				if col == i
					continue
				end
				for nz in nzrange(adj, col)
					if rows[nz] == i
						c_col = membership[col]
						if node_to_comm_in[c_col] == 0.0
							push!(touched_in, c_col)
						end
						node_to_comm_in[c_col] += vals[nz]
						break                       # at most one entry per (col, i) in CSC
					end
				end
			end

		#	Cache Loop-Invariant Quantities
			k_i_in       = k_in[i]
			k_i_out      = k_out[i]
			K_in_old     = K_in[c_old]
			K_out_old    = K_out[c_old]
			k_i_in_old   = node_to_comm_in[c_old]
			k_i_out_old  = node_to_comm_out[c_old]
			inv_m        = 1.0 / m
			inv_m_sq     = 1.0 / (m * m)

			#	Penalty contribution from leaving c_old (computed with i still in c_old,
			#	so we subtract i's degrees to get the "rest of c_old" community degrees)
				penalty_old = γ * (
					k_i_out * (K_in_old  - k_i_in) +
					k_i_in  * (K_out_old - k_i_out)
				) * inv_m_sq

		#	Union of Touched Communities (Candidates are the Union of In and Out Neighbors)
			candidates = Int[]
			sizehint!(candidates, length(touched_in) + length(touched_out))
			@inbounds for c in touched_in
				push!(candidates, c)
			end
			@inbounds for c in touched_out
				if node_to_comm_in[c] == 0.0          # not already added from touched_in
					push!(candidates, c)
				end
			end

		#	Evaluate Each Candidate and Track Best
			best_c  = c_old
			best_ΔQ = 0.0
			@inbounds for c_new in candidates
				if c_new == c_old
					continue
				end
				k_i_in_new  = node_to_comm_in[c_new]
				k_i_out_new = node_to_comm_out[c_new]

				gain = (k_i_in_new + k_i_out_new - k_i_in_old - k_i_out_old) * inv_m

				penalty_new = γ * (
					k_i_out * K_in[c_new] +
					k_i_in  * K_out[c_new]
				) * inv_m_sq

				ΔQ = gain - (penalty_new - penalty_old)
				if ΔQ > best_ΔQ
					best_ΔQ = ΔQ
					best_c  = c_new
				end
			end

		#	Reset Touched Entries in Work Vectors
			@inbounds for c in touched_in
				node_to_comm_in[c] = 0.0
			end
			@inbounds for c in touched_out
				node_to_comm_out[c] = 0.0
			end

		#	Return Best Candidate
			return (best_c, best_ΔQ)
	end

#	Helper Function: Ensure Connectivity Within Communities
	function _refine_connectivity!(adj::SparseMatrixCSC, membership::Vector{Int}; directed::Bool=false)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency matrix
			membership::Vector{Int}: community labels (modified in-place)
			directed::Bool: whether graph is directed (default = false)

		Returns:
			Nothing (membership updated in-place)

		Notes:
			- Splits disconnected communities into separate components.
			- Undirected: ensures connected subgraphs (standard connectivity).
			- Directed: ensures weakly connected subgraphs (union of in/out edges).
			- Defensive checks ensure |membership| == size(adj,1) and Int labels.
		"""

		#	Defensive checks
			n = size(adj, 1)
			@assert size(adj,1) == size(adj,2) "_refine_connectivity!: adj must be square"
			@assert length(membership) == n "_refine_connectivity!: membership length must match adj"
			@assert eltype(membership) <: Integer "_refine_connectivity!: membership labels must be integers"

		#	Build neighbor lists (use sets to avoid duplicates)
			rows, cols, vals = findnz(adj)
			if directed
				#	Weak connectivity: undirected view of edges
					neighbors_sets = [Set{Int}() for _ in 1:n]
					for k in eachindex(vals)
						i, j = rows[k], cols[k]
						if i != j
							push!(neighbors_sets[i], j)
							push!(neighbors_sets[j], i)
						end
					end
					neighbors = [collect(s) for s in neighbors_sets]
			else
				#	Undirected connectivity: bidirectional neighbors
					neighbors_sets = [Set{Int}() for _ in 1:n]
					for k in eachindex(vals)
						i, j = rows[k], cols[k]
						if i != j
							push!(neighbors_sets[i], j)
							push!(neighbors_sets[j], i)
						end
					end
					neighbors = [collect(s) for s in neighbors_sets]
			end

		#	Process each community; split into components if needed
			current_max = maximum(membership)
			comms = unique(membership)
			for c in comms
				nodes = findall(==(c), membership)
				if length(nodes) ≤ 1
					continue
				end

				unvisited = Set(nodes)
				first_component = true

				while !isempty(unvisited)
					#	Create Iteration Objects
						start = first(unvisited)
						queue = [start]
						component = Int[]
						delete!(unvisited, start)

					#	BFS over the induced subgraph of community c
						while !isempty(queue)
							v = popfirst!(queue)
							push!(component, v)
							for nbr in neighbors[v]
								if (nbr in unvisited) && (membership[nbr] == c)
									delete!(unvisited, nbr)
									push!(queue, nbr)
								end
							end
						end

					#	Label additional components with new community IDs
						if !first_component
							current_max += 1
							for v in component
								membership[v] = current_max
							end
						end
						first_component = false
				end
			end

		#	Memberships Updated in Place
			return nothing
	end

#	Helper Function: Contract Graph by Community Structure
	function _contract_by_membership(adj::SparseMatrixCSC,
									membership::Vector{Int};
									directed::Bool=false,
									weighted::Bool=true)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency matrix
			membership::Vector{Int}: community labels for each node (length == size(adj,1))
			directed::Bool: preserve directionality (default = false)
			weighted::Bool: preserve weights (true=sum weights; false=binarize/OR)

		Returns:
			SparseMatrixCSC{Float64,Int}: contracted adjacency (communities as supernodes)

		Notes:
			- Aggregates edges between communities.
			- Self-loops in the contracted graph represent intra-community edges.
			- For undirected graphs, the result is symmetrized at the end.
			- For weighted=false, nonzero entries are binarized (set to 1.0).
			- Defensive checks ensure |membership| == size(adj,1) and Int labels.
		"""

		#	Defensive checks
			n = size(adj, 1)
			@assert size(adj,1) == size(adj,2) "_contract_by_membership: adj must be square"
			@assert length(membership) == n "_contract_by_membership: membership length must match adj"
			@assert eltype(membership) <: Integer "_contract_by_membership: membership labels must be integers"

		#	Map communities to consecutive indices 1..C
			unique_comms = sort(unique(membership))
			label_map = Dict(old => new for (new, old) in enumerate(unique_comms))
			C = length(unique_comms)

		#	Aggregate edges by community pairs
			rows, cols, vals = findnz(adj)
			edge_dict = Dict{Tuple{Int,Int}, Float64}()

			for k in eachindex(vals)
				ci = label_map[membership[rows[k]]]
				cj = label_map[membership[cols[k]]]
				key = (ci, cj)
				edge_dict[key] = get(edge_dict, key, 0.0) + vals[k]
			end

		#	Build sparse matrix from aggregated pairs
			I = Int[]; J = Int[]; V = Float64[]
			sizehint!(I, length(edge_dict))
			sizehint!(J, length(edge_dict))
			sizehint!(V, length(edge_dict))
			for ((ci, cj), w) in edge_dict
				push!(I, ci); push!(J, cj); push!(V, w)
			end
			S = sparse(I, J, V, C, C)

		#	Binarize if unweighted was requested
			if !weighted
				#	Set all nonzeros to 1.0
					S.nzval .= 1.0
			end

		#	Undirected: enforce symmetry
			if !directed
				if weighted
					#	Average to re-impose exact symmetry without changing total mass
						S = 0.5 .* (S + S')
				else
					#	Logical OR for unweighted undirected
					# 	(since all nonzeros are 1.0 now, max acts as OR)
						S = max.(S, S')
						S.nzval .= 1.0  # keep it strictly binary
				end
			end

		#	Returns Contracted Adjacency Matrix (communities as supernodes)
			return S
	end

#	Helper Function: Single Leiden Run
	function _leiden_single_run_preprocessed(adj::SparseMatrixCSC,
											resolution::Float64,
											n_iterations::Int;
											directed::Bool = false,
											rng::AbstractRNG = Random.default_rng(),
											show_iteration_progress::Bool = false,
											run_description::String = "")
		"""
		Args:
			adj::SparseMatrixCSC: preprocessed adjacency matrix
			resolution::Float64: resolution parameter γ
			n_iterations::Int: maximum iterations per run
			directed::Bool: graph type for modularity (default = false)
			rng::AbstractRNG: random number generator (default = global default RNG)
			show_iteration_progress::Bool: show per-iteration progress (default = false)
			run_description::String: description for progress bar (e.g., "Run 1/5")
		Returns:
			NamedTuple: (membership, modularity, n_communities)
		Notes:
			Assumes adj is already preprocessed (symmetric for undirected, directed
			otherwise) and weighted-typed.

			Phase 1 uses delta_modularity_best! for O(d_i) move evaluation per node
			rather than O(nnz) full modularity recomputation. Per-node and per-
			community degree vectors (k, K) are maintained incrementally as moves
			are accepted.

			RNG is passed in explicitly to support multi-threaded multi-start runs
			without touching the global RNG. Callers wanting reproducibility should
			construct an Xoshiro(seed) or MersenneTwister(seed) and pass it in.
		"""

		#	Store Original Matrix
			@assert issparse(adj) "_leiden_single_run_preprocessed: adj must be SparseMatrixCSC"
			adj_original = adj

		#	Track Original Node Mapping
			n_original = size(adj, 1)
			orig_to_curr = collect(1:n_original)

		#	Initialize Partition
			membership = collect(1:size(adj, 1))
			Q = calculate_modularity(adj, membership; weighted = true, directed = directed, γ = resolution)
			iteration = 0
			improved = true

		#	Optional Iteration Progress Bar
			iter_prog = nothing
			if show_iteration_progress
				desc = isempty(run_description) ? "Iterations" : "$run_description - Iterations"
				iter_prog = Progress(n_iterations, desc = desc, enabled = true)
			end

		#	Main Leiden Loop
			while improved && iteration < n_iterations
				improved = false
				iteration += 1

				if show_iteration_progress && iter_prog !== nothing
					next!(iter_prog)
				end

				#	Compute Degrees and Totals for Current Level
					n = size(adj, 1)
					if directed
						k_out_vec = vec(sum(adj, dims = 2))
						k_in_vec  = vec(sum(adj, dims = 1))
						m_total   = sum(adj)
						two_m     = 2.0 * m_total
					else
						k_vec     = vec(sum(adj, dims = 2))
						two_m     = sum(adj)
						m_total   = two_m / 2.0
					end

				#	Compute Initial Community-Degree Vectors
					C_max = n
					if directed
						K_in_vec  = zeros(Float64, C_max)
						K_out_vec = zeros(Float64, C_max)
						@inbounds for i in 1:n
							c = membership[i]
							K_in_vec[c]  += k_in_vec[i]
							K_out_vec[c] += k_out_vec[i]
						end
					else
						K_vec = zeros(Float64, C_max)
						@inbounds for i in 1:n
							K_vec[membership[i]] += k_vec[i]
						end
					end

				#	Pre-Allocate Work Vectors for Delta Modularity
					if directed
						work_in  = zeros(Float64, C_max)
						work_out = zeros(Float64, C_max)
					else
						work     = zeros(Float64, C_max)
					end

				#	Phase 1: Local Moves via Delta Modularity
					node_order = randperm(rng, n)
					for node in node_order
						c_old = membership[node]

						#	Evaluate Best Move
							if directed
								best_c, best_ΔQ = delta_modularity_directed_best!(
									adj, node, c_old, membership,
									k_in_vec, k_out_vec,
									K_in_vec, K_out_vec, m_total,
									work_in, work_out;
									γ = resolution
								)
							else
								best_c, best_ΔQ = delta_modularity_undirected_best!(
									adj, node, c_old, membership,
									k_vec, K_vec, two_m,
									work;
									γ = resolution
								)
							end

						#	Apply Move If It Improves Modularity
							if best_c != c_old && best_ΔQ > 0.0
								if directed
									K_in_vec[c_old]   -= k_in_vec[node]
									K_out_vec[c_old]  -= k_out_vec[node]
									K_in_vec[best_c]  += k_in_vec[node]
									K_out_vec[best_c] += k_out_vec[node]
								else
									K_vec[c_old]   -= k_vec[node]
									K_vec[best_c]  += k_vec[node]
								end
								membership[node] = best_c
								improved = true
							end
					end

				#	Recompute Q After Phase 1
					Q = calculate_modularity(adj, membership; weighted = true, directed = directed, γ = resolution)

				#	Phase 2: Connectivity Refinement
					_refine_connectivity!(adj, membership; directed = directed)

				#	Phase 3: Graph Contraction
					unique_comms = sort(unique(membership))
					label_map = Dict(old => new for (new, old) in enumerate(unique_comms))

					for i in 1:n_original
						orig_to_curr[i] = label_map[membership[orig_to_curr[i]]]
					end

					adj = _contract_by_membership(adj, membership;
												  directed = directed,
												  weighted = true)
					membership = collect(1:size(adj, 1))
					Q = calculate_modularity(adj, membership;
											weighted = true,
											directed = directed,
											γ = resolution)
			end

		#	Complete Iteration Progress If Early Termination
			if show_iteration_progress && iter_prog !== nothing && iteration < n_iterations
				finish!(iter_prog)
			end

		#	Map Back to Original Nodes
			final_membership = [membership[orig_to_curr[i]] for i in 1:n_original]
			Q_final = calculate_modularity(adj_original, final_membership;
										  weighted = true,
										  directed = directed,
										  γ = resolution)

		#	Return Community Solution
			return (
				membership    = final_membership,
				modularity    = Q_final,
				n_communities = length(unique(final_membership))
			)
	end

#	Leiden Community Detection Main Interface
	function leiden_community_detection(edges::DataFrame;
									   nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
									   n_iterations::Int=10,
									   n_runs::Int=5,
									   resolution::Float64=1.0,
									   weighted::Bool=true,
									   directed::Bool=true,
									   seed::Union{Nothing,Int}=nothing,
									   parallel_runs::Bool=true,
									   test_flag::Bool=false,
									   show_progress::Bool=true,
									   show_iteration_progress::Bool=false)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optional :weight
			nodes::Union{Nothing,DataFrame,Vector}: node universe (optional)
			n_iterations::Int: max iterations per Leiden run (default = 10)
			n_runs::Int: number of multi-start runs (default = 5)
			resolution::Float64: γ resolution parameter (default = 1.0)
			weighted::Bool: treat graph as weighted (default = true)
			directed::Bool: treat graph as directed (default = true)
			seed::Union{Nothing,Int}: base RNG seed (each run uses seed + run - 1)
			parallel_runs::Bool: parallelize n_runs via Threads.@threads (default = true).
			                    Set false when called from an outer threaded context (e.g., CHAMP).
			test_flag::Bool: print diagnostics (default = false)
			show_progress::Bool: show progress bar for runs (default = true)
			show_iteration_progress::Bool: show detailed iteration progress (default = false)
		Returns:
			NamedTuple: (membership, modularity, n_communities, node_names)
		Notes:
			Enforces four preprocessing cases:
			1) unweighted & undirected: binarize, loops={0,2}, symmetrize via max
			2) unweighted & directed: binarize, loops={0,1}, no symmetrization
			3) weighted & undirected: error if binary, symmetrize via 0.5*(A+A')
			4) weighted & directed: error if binary, no symmetrization

			Multi-start runs are parallelized via Threads.@threads when
			parallel_runs=true. Each run gets its own Xoshiro RNG seeded with
			seed + run - 1 (or run - 1 if seed is nothing), so reproducibility is
			preserved across thread counts as long as the same seed is used.

			When called from an outer threaded context (e.g., champ_community_detection),
			pass parallel_runs=false to avoid oversubscription.
		"""

		#	Build Raw Adjacency Matrix
			adj, node_to_idx, idx_to_node = _graph_to_sparse_matrix(edges;
																	nodes = nodes,
																	weighted = true)
			@assert issparse(adj) "Adjacency must be sparse"
			n = size(adj, 1)
			@assert size(adj, 2) == n "Adjacency must be square"

		#	Handle Empty Graph
			if n == 0
				return (
					membership    = Int[],
					modularity    = 0.0,
					n_communities = 0,
					node_names    = idx_to_node
				)
			end

		#	Preprocess Matrix Based on Graph Type
			A_eff = copy(adj)

			if !weighted && !directed
				#	Case 1: Unweighted Undirected
					if !_is_symmetric(A_eff)
						A_eff = max.(A_eff, A_eff')
					end
					A_eff = _binarize_matrix(A_eff; directed = false)

			elseif !weighted && directed
				#	Case 2: Unweighted Directed
					A_eff = _binarize_matrix(A_eff; directed = true)

			elseif weighted && !directed
				#	Case 3: Weighted Undirected
					if _is_binary_matrix(A_eff; directed = false)
						throw(ArgumentError("weighted=true not allowed on binary matrix (undirected)"))
					end
					if !_is_symmetric(A_eff)
						A_eff = 0.5 .* (A_eff + A_eff')
					end

			else
				#	Case 4: Weighted Directed
					if _is_binary_matrix(A_eff; directed = true)
						throw(ArgumentError("weighted=true not allowed on binary matrix (directed)"))
					end
			end

		#	Debug Output
			if test_flag
				println("DEBUG leiden: n=$n  weighted=$weighted  directed=$directed")
				println("DEBUG leiden: symmetric? ", _is_symmetric(A_eff))
				println("DEBUG leiden: nnz(A_eff)=", nnz(A_eff), "  sum(A_eff)=", sum(A_eff))
				println("DEBUG leiden: threads available = ", Threads.nthreads())
				println("DEBUG leiden: parallel_runs = ", parallel_runs)
			end

		#	Pre-Allocate Per-Run Result Storage
			results = Vector{NamedTuple{(:membership, :modularity, :n_communities),
			                            Tuple{Vector{Int}, Float64, Int}}}(undef, n_runs)

		#	Progress Bar Setup
			graph_type = string(weighted ? "Weighted" : "Unweighted", " ",
							   directed ? "Directed" : "Undirected")
			parallel_tag = parallel_runs ? "$(Threads.nthreads()) threads" : "serial"
			desc = "Leiden ($graph_type, γ=$resolution, $parallel_tag)"

			prog = show_progress ? Progress(n_runs, desc = desc, enabled = true) : nothing
			prog_lock = ReentrantLock()

		#	Multi-Start Leiden Optimization
			if parallel_runs
				Threads.@threads for run in 1:n_runs
					local_seed = seed === nothing ? run - 1 : seed + run - 1
					local_rng  = Xoshiro(local_seed)
					run_desc   = "Run $run/$n_runs"
					results[run] = _leiden_single_run_preprocessed(
						A_eff, resolution, n_iterations;
						directed = directed,
						rng      = local_rng,
						show_iteration_progress = show_iteration_progress,
						run_description = run_desc
					)
					if show_progress
						lock(prog_lock) do
							next!(prog)
						end
					end
				end
			else
				for run in 1:n_runs
					local_seed = seed === nothing ? run - 1 : seed + run - 1
					local_rng  = Xoshiro(local_seed)
					run_desc   = "Run $run/$n_runs"
					results[run] = _leiden_single_run_preprocessed(
						A_eff, resolution, n_iterations;
						directed = directed,
						rng      = local_rng,
						show_iteration_progress = show_iteration_progress,
						run_description = run_desc
					)
					if show_progress
						next!(prog)
					end
				end
			end

		#	Reduce: Find Best Result Across Runs
			best_Q = -Inf
			best_m = Vector{Int}()
			for run in 1:n_runs
				if results[run].modularity > best_Q
					best_Q = results[run].modularity
					best_m = results[run].membership
				end
			end

		#	Handle Isolates if Node Universe Provided
			if nodes !== nothing && length(best_m) < size(idx_to_node, 1)
				full_membership = zeros(Int, length(idx_to_node))
				next_comm = maximum(best_m) + 1

				for i in 1:length(idx_to_node)
					if i ≤ length(best_m)
						full_membership[i] = best_m[i]
					else
						full_membership[i] = next_comm
						next_comm += 1
					end
				end

				best_m = full_membership
			end

		#	Return Best Solution with Node Names
			return (
				membership    = best_m,
				modularity    = best_Q,
				n_communities = length(unique(best_m)),
				node_names    = idx_to_node
			)
	end
	@doc raw"""
	**Description**
	Detects communities using the Leiden algorithm with guaranteed well-connected
	communities through local moves, refinement, and multilevel optimization.
	Supports directed and undirected graphs with optional weights, with multi-
	start optimization parallelized across CPU threads.

	**Usage**
	`leiden_community_detection(edges; nodes=nothing, resolution=1.0, n_iterations=10, n_runs=5, weighted=true, directed=true, seed=nothing, test_flag=false, show_progress=true, show_iteration_progress=false)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst` columns, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Node universe (includes isolates if provided).
	  - `Nothing`: Infer from edges (default, excludes isolates).
	  - `DataFrame`: Must have `:id` and `:label` columns.
	  - `Vector`: Node IDs as strings.
	- `resolution::Float64`: Resolution parameter γ (default `1.0`).
	  - γ < 1.0: Larger communities.
	  - γ = 1.0: Standard modularity.
	  - γ > 1.0: Smaller communities.
	- `n_iterations::Int`: Maximum iterations per run (default `10`).
	- `n_runs::Int`: Independent multi-start runs to perform (default `5`). Parallelized across threads.
	- `weighted::Bool`: Use edge weights if present (default `true`).
	- `directed::Bool`: Treat graph as directed (default `true`).
	- `seed::Union{Int,Nothing}`: Base random seed. Run `r` uses seed `seed + r - 1`. If `nothing`, seeds use `r - 1`.
	- `test_flag::Bool`: Print diagnostic output (default `false`).
	- `show_progress::Bool`: Show overall run-level progress bar (default `true`).
	- `show_iteration_progress::Bool`: Show per-iteration progress within each run (default `false`).

	**Details**
	Three-phase optimization per iteration:
	1. **Local Moves**: Greedy node moves to neighboring communities, accelerated via delta-modularity (O(degree) per move evaluation).
	2. **Refinement**: Ensures connectivity within communities (weak connectivity for directed graphs).
	3. **Contraction**: Hierarchical aggregation preserving directionality and weights.

	Optimizes modularity based on graph type:
	- **Undirected**: `Q = (1/2m) Σ[A_ij - γ(k_i · k_j)/(2m)] δ(c_i, c_j)`
	- **Directed**: `Q = (1/m) Σ[A_ij - γ(k_i^out · k_j^in)/m] δ(c_i, c_j)`

	**Threading**
	Multi-start runs are parallelized using `Threads.@threads`. To control the number of threads, launch Julia with `julia --threads N` (or `--threads auto`) or set the `JULIA_NUM_THREADS` environment variable. Each run receives its own `Xoshiro` RNG, so reproducibility is preserved across different thread counts as long as the same `seed` is used.

	**Value**
	`NamedTuple` containing:
	- `membership::Vector{Int}`: Community assignments (1-based).
	- `modularity::Float64`: Best modularity score achieved across all runs.
	- `n_communities::Int`: Number of communities in the best solution.
	- `node_names::Vector`: Original node identifiers in adjacency-matrix order.

	**Examples**
```julia
		#	Undirected weighted (default arguments)
			result = leiden_community_detection(edges)

		#	Directed weighted with node universe and isolates
			result = leiden_community_detection(edges;
											  nodes = node_df,
											  resolution = 0.8,
											  n_runs = 10,
											  seed = 42)

		#	Multiple runs for robustness (parallelized)
			result = leiden_community_detection(edges;
											  n_runs = 20)
```

	**References**
	Traag VA, Waltman L, van Eck NJ (2019) "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports 9(1):5233.
	""" leiden_community_detection

#	Helper Function for champ_community_detection: Calculate Partition Coefficients (igraph-aligned)
	function _calculate_partition_coefficients(adj::SparseMatrixCSC, membership::Vector{Int})
		"""
		Args:
			adj::SparseMatrixCSC: preprocessed adjacency matrix
			membership::Vector{Int}: community assignments
		Returns:
			Tuple{Float64,Float64}: (A, P) coefficients
		Notes:
			Matches igraph's undirected modularity convention.
			For directed graphs, use _calculate_partition_coefficients_directed.
		"""
		#	Validation
			@assert size(adj,1) == size(adj,2) "_calculate_partition_coefficients: adj must be square"
			@assert length(membership) == size(adj,1) "_calculate_partition_coefficients: membership length mismatch"
		
		#	Remap Membership to Contiguous 1..C
			labels = sort(unique(membership))
			label_to_col = Dict{Int,Int}(lab => i for (i, lab) in enumerate(labels))
			n = size(adj, 1)
			C = length(labels)
			mapped = Vector{Int}(undef, n)
			@inbounds for i in 1:n
				mapped[i] = label_to_col[membership[i]]
			end
		
		#	Build Indicator Matrix S (n × C)
			S = sparse(collect(1:n), mapped, ones(Float64, n), n, C)
		
		#	Calculate Effective Totals (igraph-style)
			d = diag(adj)
			two_m_eff = sum(adj) + sum(d)
			if two_m_eff == 0.0
				return (0.0, 0.0)
			end
			m_eff = two_m_eff / 2.0
			k_eff = vec(sum(adj, dims=2)) .+ d
		
		#	Calculate A = E_eff (Internal Weight with Doubled Loops)
			E_blocks = S' * adj * S
			E_diag   = S' * spdiagm(0 => d) * S
			E_eff    = sum(diag(E_blocks)) + sum(diag(E_diag))
		
		#	Calculate P = Expected Edges
			K_eff = vec(S' * k_eff)
			P = sum((K_eff .^ 2) ./ (2.0 * m_eff))
		
		#	Return Coefficients
			return (E_eff, P)
	end

#	Helper Function for champ_community_detection: Calculate Directed Partition Coefficients
	function _calculate_partition_coefficients_directed(adj::SparseMatrixCSC, membership::Vector{Int})
		"""
		Args:
			adj::SparseMatrixCSC: directed adjacency matrix (not symmetrized)
			membership::Vector{Int}: community assignments
		Returns:
			Tuple{Float64,Float64}: (A, P) coefficients for directed graphs
		Notes:
			Uses directed null model: K_out * K_in / m
		"""
		#	Validation
			@assert size(adj,1) == size(adj,2) "_calculate_partition_coefficients_directed: adj must be square"
			@assert length(membership) == size(adj,1) "_calculate_partition_coefficients_directed: membership length mismatch"
		
		#	Remap Membership to Contiguous 1..C
			labels = sort(unique(membership))
			label_to_col = Dict{Int,Int}(lab => i for (i, lab) in enumerate(labels))
			n = size(adj, 1)
			C = length(labels)
			mapped = Vector{Int}(undef, n)
			@inbounds for i in 1:n
				mapped[i] = label_to_col[membership[i]]
			end
		
		#	Build Indicator Matrix S (n × C)
			S = sparse(collect(1:n), mapped, ones(Float64, n), n, C)
		
		#	Calculate Total Weight and Degrees
			m = sum(adj)
			if m == 0.0
				return (0.0, 0.0)
			end
			k_out = vec(sum(adj, dims=2))  # out-degrees
			k_in = vec(sum(adj, dims=1))   # in-degrees
		
		#	Calculate A = Internal Edges
			E_blocks = S' * adj * S
			A = sum(diag(E_blocks))
		
		#	Calculate P = Expected Edges (Directed Null Model)
			K_out = vec(S' * k_out)
			K_in = vec(S' * k_in)
			P = sum((K_out .* K_in) ./ m)
		
		#	Return Coefficients
			return (A, P)
	end

#	Helper Function for champ_community_detection: Curve-Intersection Selection
	function _select_curve_intersection(result::NamedTuple)
		"""
		Args:
			result::NamedTuple: CHAMP sweep result with :gammas, :A_coeffs, :P_coeffs,
				:n_communities_per_gamma, :dominant
		Returns:
			Int: index into result.gammas of the selected partition
		Notes:
			Implements the canonical CHAMP optimization criterion: identify the
			partition at the bend of the upper envelope on the convex-hull plot.
			The bend is where two consecutive admissible partitions have the largest
			difference in slope, because that is where the envelope changes direction
			most sharply.

			Procedure:
			1. Restrict to dominant partitions, sorted by γ.
			2. Exclude trivial single-community partitions, which always dominate at
			   low γ but represent absence of structure rather than structure.
			3. For each consecutive pair (i, i+1) of dominant partitions, compute the
			   absolute slope difference |slope[i+1] - slope[i]| where slope = -P.
			4. Find the pair with the largest slope difference.
			5. Return the partition with the smaller P (i.e., the one on the
			   shallower-slope side of the bend), since this is the partition
			   discovered just after the regime change to consolidated communities.

			On networks with a clear structural scale, the selected γ corresponds
			to the visual elbow where the upper envelope visibly transitions from
			steep (low-γ, many small communities) to shallow (high-γ, fewer large
			communities) line slopes.

			Falls back to the middle dominant partition if too few dominant
			partitions exist to compute a bend (< 2 after filtering trivials).
		"""

		#	Extract Fields
			gammas     = result.gammas
			P_coeffs   = result.P_coeffs
			n_comms_pg = result.n_communities_per_gamma
			dominant   = result.dominant

		#	Identify Dominant Indices Excluding Trivial
			dom_ix = findall(dominant)
			if isempty(dom_ix)
				dom_ix = collect(eachindex(gammas))
			end
			filter!(i -> n_comms_pg[i] > 1, dom_ix)

		#	Fall Back If Insufficient Dominant Partitions
			if isempty(dom_ix)
				return 1
			end
			if length(dom_ix) < 2
				return dom_ix[1]
			end

		#	Sort by γ for Adjacent-Pair Differencing
			sort!(dom_ix; by = i -> gammas[i])

		#	Find Consecutive Pair with Largest Slope Difference
			#	Slope of line i is -P_coeffs[i]. Largest |slope_change| = largest |P_change|.
			#	The bend is the pair (i, i+1) where the line slopes change most sharply.
			best_pair_lo = dom_ix[1]
			best_pair_hi = dom_ix[2]
			best_diff    = abs(P_coeffs[dom_ix[2]] - P_coeffs[dom_ix[1]])

			for k in 2:(length(dom_ix) - 1)
				lo   = dom_ix[k]
				hi   = dom_ix[k + 1]
				diff = abs(P_coeffs[hi] - P_coeffs[lo])
				if diff > best_diff
					best_diff    = diff
					best_pair_lo = lo
					best_pair_hi = hi
				end
			end

		#	Return the Shallower-Slope Side of the Bend
			#	The partition with smaller P is on the shallower side, just after
			#	the regime change to consolidated community structure.
			return P_coeffs[best_pair_hi] < P_coeffs[best_pair_lo] ? best_pair_hi : best_pair_lo
	end

#	CHAMP: Convex Hull of Admissible Modularity Partitions
	function champ_community_detection(edges::DataFrame;
	                                  nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
	                                  resolution::Union{Float64,Nothing}=nothing,
	                                  resolution_range::Tuple{Float64,Float64}=(0.1, 1.8),
	                                  n_resolutions::Int=30,
	                                  n_runs_per_gamma::Int=5,
	                                  n_iterations_per_run::Int=10,
	                                  weighted::Bool=false,
	                                  directed::Bool=false,
	                                  agg_func::Union{Function,Nothing}=nothing,
	                                  seed::Union{Int,Nothing}=nothing,
	                                  show_progress::Bool=true)
		"""
		Args:
			edges::DataFrame: :src, :dst, optional :weight
			nodes::Union{Nothing,DataFrame,Vector}: node universe (optional)
			resolution::Union{Float64,Nothing}: single γ or sweep if nothing
			resolution_range::Tuple: γ range for sweep (default = (0.1, 1.8))
			n_resolutions::Int: number of γ values in sweep (default = 30)
			n_runs_per_gamma::Int: Leiden runs per γ (default = 5)
			n_iterations_per_run::Int: max iterations per run (default = 10)
			weighted::Bool: treat graph as weighted (default = false)
			directed::Bool: treat graph as directed (default = false)
			agg_func::Function: edge aggregation (default = sum if weighted)
			seed::Union{Int,Nothing}: RNG seed
			show_progress::Bool: display progress bars (default = true)
		Returns:
			NamedTuple: (membership, resolution_used, modularity, n_communities, node_names,
			             gammas, A_coeffs, P_coeffs, modularities, n_communities_per_gamma,
			             dominant, best_index)
		Notes:
			The γ sweep is parallelized with Threads.@threads. Each γ task calls
			leiden_community_detection with parallel_runs=false to avoid nested
			threading.

			Selection identifies the partition at the bend of the convex-hull upper
			envelope — the γ at which the steep-slope (low-γ) and shallow-slope
			(high-γ) regimes transition most rapidly. This is the canonical CHAMP
			optimum: the γ value beyond which additional resolution stops yielding
			meaningful structural information.
		"""

		#	Validation
			@assert hasproperty(edges, :src) && hasproperty(edges, :dst) "edges must have :src and :dst"
			if nrow(edges) == 0
				return (membership=Int[], resolution_used=0.0, modularity=0.0,
				       n_communities=0, node_names=String[],
				       gammas=Float64[], A_coeffs=Float64[], P_coeffs=Float64[],
				       modularities=Float64[], n_communities_per_gamma=Int[],
				       dominant=Bool[], best_index=0)
			end

		#	Set Aggregation Strategy
			if isnothing(agg_func)
				agg_func = (weighted && hasproperty(edges, :weight)) ? sum : maximum
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func=agg_func)

		#	Build Base Adjacency
			use_weights = weighted && hasproperty(clean_edges, :weight)
			adj, node_to_idx, idx_to_node = _graph_to_sparse_matrix(clean_edges;
			                                                        nodes=nodes,
			                                                        weighted=use_weights)

		#	Binarize if Unweighted
			if !weighted
				adj = map!(x -> x == 0.0 ? 0.0 : 1.0, copy(adj), adj)
			end

		#	Symmetrize for Undirected
			if !directed
				if weighted
					adj = 0.5 .* (adj + adj')
				else
					adj = max.(adj, adj')
				end
			end

		#	Extract Node Names
			if idx_to_node isa DataFrame
				df = idx_to_node
				cols = names(df)
				if :label in cols
					node_names = string.(df.label)
				elseif :id in cols
					node_names = string.(df.id)
				else
					firstcol = cols[1]
					node_names = string.(df[!, firstcol])
				end
			else
				node_names = string.(idx_to_node)
			end

		#	Define Resolution Grid
			gammas = (resolution === nothing) ?
				collect(range(resolution_range[1], resolution_range[2]; length=n_resolutions)) :
				[resolution]

		#	Storage for Partitions
			all_partitions = Vector{NamedTuple}(undef, length(gammas))
			all_coeffs     = Vector{Tuple{Float64,Float64}}(undef, length(gammas))

		#	Progress Bar Setup
			prog = nothing
			prog_lock = ReentrantLock()
			if show_progress
				desc = "CHAMP γ sweep ($(Threads.nthreads()) threads)"
				prog = Progress(length(gammas), desc = desc, enabled = true)
			end

		#	Phase 1: Resolution Sweep (Threaded)
			Threads.@threads for ix in 1:length(gammas)
				γ = gammas[ix]

				#	Run Leiden at This Resolution (Serial Inside)
					res = leiden_community_detection(clean_edges;
					                                  nodes        = nodes,
					                                  resolution   = γ,
					                                  n_iterations = n_iterations_per_run,
					                                  n_runs       = n_runs_per_gamma,
					                                  weighted     = weighted,
					                                  directed     = directed,
					                                  seed         = seed,
					                                  parallel_runs = false,
					                                  show_progress = false)

				#	Calculate Coefficients
					if directed
						Aeff, Peff = _calculate_partition_coefficients_directed(adj, res.membership)
					else
						Aeff, Peff = _calculate_partition_coefficients(adj, res.membership)
					end

				#	Store Results in Pre-Allocated Slots
					all_partitions[ix] = (
						membership    = res.membership,
						gamma         = γ,
						modularity    = res.modularity,
						n_communities = res.n_communities,
					)
					all_coeffs[ix] = (Aeff, Peff)

				#	Update Progress Bar
					if show_progress
						lock(prog_lock) do
							next!(prog)
						end
					end
			end

		#	Phase 2: Dominance Analysis
			dominant = trues(length(gammas))

			if length(gammas) > 1
				nP = length(all_partitions)
				γmin, γmax = minimum(gammas), maximum(gammas)

				#	Check Dominance Relationships
					for i in 1:nP
						Ai, Pi = all_coeffs[i]
						for j in 1:nP
							i == j && continue
							Aj, Pj = all_coeffs[j]

							if !isapprox(Pi, Pj; atol=1e-12)
								γcross = (Aj - Ai) / (Pi - Pj + 1e-12)
								if γmin - 1e-6 < γcross < γmax + 1e-6
									γtest = clamp((γcross + all_partitions[i].gamma)/2, γmin, γmax)
									if (Aj - γtest*Pj) > (Ai - γtest*Pi) + 1e-12
										dominant[i] = false
										break
									end
								elseif (Aj - all_partitions[i].gamma*Pj) >
								       (Ai - all_partitions[i].gamma*Pi) + 1e-12
									dominant[i] = false
									break
								end
							else
								if Aj > Ai + 1e-12
									dominant[i] = false
									break
								end
							end
						end
					end
			end

		#	Build Intermediate Result for Selection Helper
			intermediate = (
				gammas                  = gammas,
				A_coeffs                = [c[1] for c in all_coeffs],
				P_coeffs                = [c[2] for c in all_coeffs],
				modularities            = [p.modularity for p in all_partitions],
				n_communities_per_gamma = [p.n_communities for p in all_partitions],
				dominant                = dominant
			)

		#	Phase 3: Select Partition at the Convex-Hull Bend
			best_ix = _select_curve_intersection(intermediate)

		#	Return Best Partition with Full Sweep Data
			best = all_partitions[best_ix]
			return (
				membership              = best.membership,
				resolution_used         = best.gamma,
				modularity              = best.modularity,
				n_communities           = best.n_communities,
				node_names              = node_names,
				gammas                  = gammas,
				A_coeffs                = intermediate.A_coeffs,
				P_coeffs                = intermediate.P_coeffs,
				modularities            = intermediate.modularities,
				n_communities_per_gamma = intermediate.n_communities_per_gamma,
				dominant                = dominant,
				best_index              = best_ix
			)
	end
	@doc raw"""
	**Description**
	Implements CHAMP (Convex Hull of Admissible Modularity Partitions) to identify
	the resolution-stable community partition of a graph. Performs a sweep across
	resolution values γ, identifies the admissible set of partitions on the upper
	envelope of the modularity-vs-γ convex hull, and selects the partition at the
	bend of that envelope — the γ value at which the trade-off between
	community-count fineness and modularity quality is optimal.

	Supports directed and undirected graphs with optional edge weights. The γ
	sweep is parallelized across CPU threads for substantial speedup on multi-core
	machines.

	**Usage**
	`champ_community_detection(edges; nodes=nothing, resolution=nothing,
	                          resolution_range=(0.1,1.8), n_resolutions=30,
	                          weighted=false, directed=false, n_runs_per_gamma=5,
	                          n_iterations_per_run=10, seed=nothing,
	                          show_progress=true)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optional `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Node universe (includes isolates if provided).
	- `resolution::Float64|nothing`: If supplied, run Leiden at this single γ and skip the sweep. If `nothing` (default), perform a full resolution sweep.
	- `resolution_range::Tuple`: Range `(γ_min, γ_max)` for the sweep (default `(0.1, 1.8)`).
	- `n_resolutions::Int`: Number of γ values in the sweep grid (default `30`).
	- `n_runs_per_gamma::Int`: Number of independent Leiden multi-starts per γ value (default `5`).
	- `n_iterations_per_run::Int`: Maximum Leiden iterations per run (default `10`).
	- `weighted::Bool`: Use edge weights from the `:weight` column (default `false`).
	- `directed::Bool`: Treat the graph as directed (default `false`).
	- `agg_func::Function`: Edge aggregation when input has multi-edges (default `sum` if weighted, else `maximum`).
	- `seed::Int`: Random seed for reproducibility. Per-thread RNGs are derived from this seed.
	- `show_progress::Bool`: Display progress bar during the γ sweep (default `true`).

	**Details**
	CHAMP performs three phases:

	1. **Sweep**: Run Leiden at each γ in the resolution grid. For each resulting partition, compute the two CHAMP coefficients:
	   - `A` — total edge weight within communities (the modularity "internal" term)
	   - `P` — expected within-community weight under the null model (the modularity "penalty" term)
	   The modularity at any γ for this partition equals `(A - γP) / m_total`, so the partition's modularity is a linear function of γ with intercept `A` and slope `-P`.
	2. **Dominance**: For each partition i, test whether some other partition j has a higher `A - γP` value across the entire resolution range. Partitions that survive (are optimal somewhere in the range) form the *admissible set* on the convex hull.
	3. **Selection at the convex-hull bend**: Identify the partition at the elbow of the upper envelope. Each admissible partition contributes a line with slope `-P`; the steep slopes (large P) come from low-γ partitions with many small communities, and the shallow slopes (small P) come from high-γ partitions with fewer larger communities. The bend of the envelope is the γ at which slope transitions most rapidly from steep to shallow. Trivial single-community partitions are excluded from selection because they always lie on the convex hull at low γ but represent the absence of community structure rather than its presence.

	The γ sweep is parallelized using `Threads.@threads`. Each γ task calls `leiden_community_detection` with `parallel_runs=false` to prevent nested thread oversubscription.

	**Modularity Conventions**

	Coefficients match igraph's modularity calculation:
	- **Undirected**: `Q(γ) = (A - γP) / (2 m_eff)` where `m_eff = sum(adj) / 2 + sum(diag(adj)) / 2` to account for self-loops igraph-style.
	- **Directed**: `Q(γ) = (A - γP) / m` with directed null model `K_out · K_in / m`.

	**Threading**

	The γ sweep is parallelized using `Threads.@threads`. To control thread count, launch Julia with `julia --threads N` or set `JULIA_NUM_THREADS`. Speedup scales approximately linearly with thread count when `n_resolutions ≥ n_threads`.

	**Value**

	NamedTuple containing:
	- `membership::Vector{Int}`: Community assignments for the selected partition.
	- `resolution_used::Float64`: γ value of the selected partition.
	- `modularity::Float64`: Modularity score of the selected partition at its γ.
	- `n_communities::Int`: Number of communities in the selected partition.
	- `node_names::Vector{String}`: Original node identifiers in adjacency-matrix order.
	- `gammas::Vector{Float64}`: All γ values in the sweep.
	- `A_coeffs::Vector{Float64}`: A coefficient for each γ partition.
	- `P_coeffs::Vector{Float64}`: P coefficient for each γ partition.
	- `modularities::Vector{Float64}`: Modularity for each γ partition (computed at each partition's own γ).
	- `n_communities_per_gamma::Vector{Int}`: Community count for each γ partition.
	- `dominant::Vector{Bool}`: `true` for partitions in the admissible set (on the convex hull).
	- `best_index::Int`: Index into `gammas` of the selected partition.

	The sweep data fields enable post-hoc visualization (see `plot_champ_sweep`) for users who want to verify the selected partition against the full sweep.

	**Examples**

```julia
		#	Default analysis on a directed weighted network
			result = champ_community_detection(edges;
											weighted = true,
											directed = true)
			println("Selected γ: $(result.resolution_used)")
			println("Modularity: $(result.modularity)")

		#	Custom resolution range
			result = champ_community_detection(edges;
											resolution_range = (0.2, 2.0),
											n_resolutions    = 40,
											weighted         = true,
											directed         = true)

		#	Single-resolution Leiden via CHAMP (skips sweep, runs Leiden once)
			result = champ_community_detection(edges;
											resolution = 1.0,
											weighted   = true)
```

	**References**
	1. Weir WH, Emmons S, Gibson R, Taylor D, Mucha PJ (2017) "Post-processing partitions to identify domains of modularity optimization." *Algorithms* 10(3):93. doi:10.3390/a10030093
	2. Github implementation: https://github.com/wweir827/CHAMP

	**See Also**
	`leiden_community_detection`, `calculate_modularity`, `plot_champ_sweep`
	""" champ_community_detection

#   Exports (public API)
    export calculate_modularity,
           delta_modularity_undirected_best!,
           delta_modularity_directed_best!,
           delta_modularity_best!,
		   _leiden_single_run_preprocessed,
           leiden_community_detection,
           champ_community_detection

end # module network_community_detection