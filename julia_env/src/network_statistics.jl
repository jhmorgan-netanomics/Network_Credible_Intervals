module network_statistics

#   Module Packages
    using DataFrames
    using SparseArrays
    using LinearAlgebra
    using Printf
    using ProgressMeter
    using Random
    using Statistics

#   Sibling Submodule Helpers
#   These underscore-prefixed helpers live in network_community_detection.jl and
#   are deliberately reach-imported here so this submodule can share their
#   well-tested sparse-matrix machinery without duplication. Both submodules
#   sit inside the same parent package, so this is a within-package import.
    using ..network_community_detection: _edgelist_to_sparse_matrix,
                                         _graph_to_sparse_matrix,
                                         _aggregate_multi_edges,
                                         _is_symmetric

#   ====================================================================
#   network_statistics submodule
#
#   Descriptive structural statistics for the Network_Credible_Intervals
#   package. Implements Phase 0 of the validation roadmap: a corpus-wide
#   table of network properties computed on the pristine (full) networks
#   before any missingness or imputation is applied. The full Phase 0
#   table is produced by build_phase_0_table; this file contains the
#   individual measure functions that the driver calls.
#
#   Measures follow Smith, Morgan, & Moody (2022) conventions:
#       - Centralization is the standard deviation of node-level
#         centralities (SMM's operational definition), not the classical
#         Freeman normalization. A classical Freeman centralization is
#         also exported (freeman_degree_centralization) for users who
#         want it.
#       - Closeness uses the inverse-distance form so disconnected
#         pairs contribute 0 (SMM 2022, p. 11).
#       - Bonacich is computed on the symmetrized adjacency for
#         directed networks (SMM 2022, p. 10).
#       - Transitivity, reciprocity, and triad-based measures are
#         computed on the binarized adjacency.
#       - The tau statistic uses ranked-cluster (RC) weighting and
#         is conditioned on the M/A/N dyad census under U|MAN.
#
#   This file is built up section by section. The current sections are:
#       1. Standalone utilities (gini, centralization, rand index)
#       2. Degree family (ported from Large_Graph_Similarity.jl) plus
#          freeman_degree_centralization (new).
#
#   Future sections will add path-based centralities, Bonacich,
#   topology measures, triad/tau, components, blockmodel, and the
#   Phase 0 drivers.
#   ====================================================================

########################################
#   SECTION 1: STANDALONE UTILITIES    #
########################################

#	Gini Coefficient of a Non-Negative Vector
	function gini_coefficient(values::AbstractVector{<:Real})
		"""
		Args:
			values::AbstractVector{<:Real}: non-negative values (e.g., degrees)
		Returns:
			Float64: Gini coefficient in [0, 1]
		Notes:
			Standard sorted-Lorenz computation:
			G = (sum over i of (2i - n - 1) * x_(i)) / (n * sum(x))
			where x_(i) is the i-th smallest value (i = 1..n).

			Returns 0.0 if all values are zero or if n <= 1.
			Throws ArgumentError if any value is negative — the Gini coefficient
			is undefined on signed quantities.
		"""

		#	Validation
			n = length(values)
			if n == 0
				return 0.0
			end
			if n == 1
				return 0.0
			end
			if any(v -> v < 0, values)
				throw(ArgumentError("gini_coefficient: values must be non-negative"))
			end

		#	Sort Ascending
			x = sort(collect(Float64, values))

		#	Total Sum (Denominator Factor)
			s = sum(x)
			if s == 0.0
				return 0.0
			end

		#	Numerator: sum of (2i - n - 1) * x_(i)
			num = 0.0
			@inbounds for i in 1:n
				num += (2 * i - n - 1) * x[i]
			end

		#	Return Normalized Gini
			return num / (n * s)
	end
	@doc raw"""
	**Description**
	Gini coefficient of a non-negative vector. Standard inequality summary used
	in network analysis as a measure of degree concentration: 0 indicates perfect
	equality (every node has the same degree), 1 indicates maximum inequality
	(one node holds all the mass).

	**Usage**
	`gini_coefficient(values::AbstractVector{<:Real})`

	**Arguments**
	- `values::AbstractVector{<:Real}`: A non-negative numeric vector. Typical
	  inputs are degree sequences or other node-level quantities.

	**Details**
	Computed via the sorted-Lorenz formula:
	$$G = \frac{\sum_{i=1}^{n} (2i - n - 1) \, x_{(i)}}{n \sum_{i=1}^{n} x_i}$$
	where $x_{(i)}$ is the $i$-th smallest value. The result is in $[0, 1]$.

	Returns 0.0 when the input is empty, has length 1, or sums to zero. Negative
	values throw an `ArgumentError`.

	**Value**
	A `Float64` Gini coefficient in $[0, 1]$.

	**Examples**
```julia
	gini_coefficient([1, 1, 1, 1])           # 0.0   — perfect equality
	gini_coefficient([0, 0, 0, 4])           # 0.75  — one-node concentration
	gini_coefficient([1, 2, 3, 4, 5])        # 0.267 — moderate inequality
```

	**References**
	- Gini C. (1912). *Variabilità e mutabilità*. Bologna: C. Cuppini.
	""" gini_coefficient

#	Centralization as Standard Deviation of Centrality Vector
	function centralization(values::AbstractVector{<:Real})
		"""
		Args:
			values::AbstractVector{<:Real}: node-level centrality scores
		Returns:
			Float64: standard deviation of the centrality vector
		Notes:
			Smith, Morgan, & Moody (2022) operationalize "centralization" as
			the simple standard deviation of individual centrality scores
			across nodes. This is the convention used throughout the Phase 0
			table for in-degree, total-degree, Bonacich, closeness, and
			betweenness centralizations.

			This is intentionally NOT the Freeman (1979) normalization
			[sum(C_max - C_i)] / [(N-1)(N-2)]. A classical Freeman
			centralization is provided separately as
			freeman_degree_centralization for users who want it.

			Returns 0.0 if the vector has length 0 or 1.
		"""

		#	Edge Cases
			n = length(values)
			if n <= 1
				return 0.0
			end

		#	Return Standard Deviation
			return std(values)
	end
	@doc raw"""
	**Description**
	Network-level centralization as the standard deviation of a node-level
	centrality vector. Follows the operational definition of Smith, Morgan, &
	Moody (2022): "Our measure of centralization is a simple standard deviation
	of the individual centrality scores."

	**Usage**
	`centralization(values::AbstractVector{<:Real})`

	**Arguments**
	- `values::AbstractVector{<:Real}`: A node-level centrality vector
	  (e.g., the `:in_degree` column of `in_degree(edges)`).

	**Details**
	Returns `std(values)` with no normalization. Larger values indicate greater
	dispersion in the centrality distribution — interpreted as more centralized
	network structure for the underlying centrality measure.

	This is intentionally NOT the Freeman (1979) network-level centralization
	formula $\sum (C_{\max} - C_i) / [(N-1)(N-2)]$. For that, see
	`freeman_degree_centralization`. The two measures answer related questions
	but on different scales and with different theoretical motivations.

	Returns 0.0 for vectors of length 0 or 1.

	**Value**
	A `Float64`, the standard deviation of the input vector.

	**Examples**
```julia
	using DataFrames
	edges = DataFrame(src=[1,2,3,4], dst=[2,3,4,1])
	deg   = total_degree(edges; drop_self_loops=true)
	centralization(deg.total_degree)
```

	**References**
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III:
	  Imputation of missing network data under different network and missing
	  data conditions." *Social Networks* 68: 148–178.

	**See Also**
	`freeman_degree_centralization`, `gini_coefficient`
	""" centralization

#	Rand Index Comparing Two Partitions
	function rand_index(partition_a::AbstractVector, partition_b::AbstractVector)
		"""
		Args:
			partition_a::AbstractVector: cluster labels for each node (any hashable type)
			partition_b::AbstractVector: cluster labels for the same nodes, same length
		Returns:
			Float64: unadjusted Rand index in [0, 1]
		Notes:
			Counts the fraction of unordered pairs (i, j) with i < j on which
			the two partitions agree:
				agreement = (a and b both place i, j together) OR
				            (a and b both place i, j apart)
			Rand = #agreements / #pairs.

			This is the unadjusted form — not chance-corrected. For chance
			correction, use the adjusted Rand index (not implemented here;
			Phase 0 only needs the unadjusted form for the Track A degradation
			measure planned in the validation roadmap).

			Throws ArgumentError if the two partitions have different lengths.
			Returns 1.0 for partitions of length 0 or 1 (vacuously identical).
		"""

		#	Validation
			n = length(partition_a)
			if length(partition_b) != n
				throw(ArgumentError("rand_index: partitions must have the same length"))
			end

		#	Edge Cases
			if n <= 1
				return 1.0
			end

		#	Count Agreements over All Unordered Pairs
			agreements = 0
			total_pairs = 0
			@inbounds for i in 1:(n - 1)
				for j in (i + 1):n
					total_pairs += 1
					same_a = partition_a[i] == partition_a[j]
					same_b = partition_b[i] == partition_b[j]
					if same_a == same_b
						agreements += 1
					end
				end
			end

		#	Return Normalized
			return total_pairs == 0 ? 1.0 : agreements / total_pairs
	end
	@doc raw"""
	**Description**
	Unadjusted Rand index comparing two partitions of the same nodes. Measures
	the proportion of unordered node pairs $(i, j)$ on which the two partitions
	agree — either by placing $i$ and $j$ together in both partitions, or by
	separating them in both.

	**Usage**
	`rand_index(partition_a::AbstractVector, partition_b::AbstractVector)`

	**Arguments**
	- `partition_a::AbstractVector`: Cluster labels for each node. Any hashable
	  label type (Int, Symbol, String, etc.).
	- `partition_b::AbstractVector`: Cluster labels for the same nodes, in the
	  same order. Must have the same length as `partition_a`.

	**Details**
	The unadjusted Rand index is

	$$\text{Rand}(A, B) = \frac{n_{11} + n_{00}}{\binom{n}{2}}$$

	where $n_{11}$ counts pairs together in both partitions and $n_{00}$ counts
	pairs separated in both. Range is $[0, 1]$; 1 indicates identical partitions,
	0 indicates maximally disagreeing partitions (rare in practice).

	This is **not** chance-corrected. For randomly assigned partitions of size
	$k$, the expected Rand index is bounded away from 0 and grows with $k$. The
	adjusted Rand index (Hubert & Arabie 1985) corrects for this but is not
	implemented here; the Phase 0 / Track A validation uses the unadjusted form
	to match Smith, Morgan, & Moody (2022, p. 12).

	**Value**
	A `Float64` in $[0, 1]$. Returns 1.0 for partitions of length 0 or 1.

	**Examples**
```julia
	rand_index([1, 1, 2, 2], [1, 1, 2, 2])     # 1.0 — identical
	rand_index([1, 1, 2, 2], [2, 2, 1, 1])     # 1.0 — same partition, relabeled
	rand_index([1, 1, 2, 2], [1, 2, 1, 2])     # 0.333
```

	**References**
	- Rand WM (1971). "Objective criteria for the evaluation of clustering
	  methods." *Journal of the American Statistical Association* 66: 846–850.
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.
	""" rand_index

#	Distance-to-Strength Weight Transformation
	function transform_distance_weights(edges::DataFrame;
										 method::Symbol = :scaled_reciprocal,
										 tau::Union{Nothing, Float64} = nothing,
										 target_median::Float64 = 10.0,
										 weight_col::Symbol = :weight)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst and a distance-semantic
				weight column (lower = stronger tie).
			method::Symbol: :scaled_reciprocal (default), :max_minus, or :exp_decay.
			tau::Union{Nothing,Float64}: decay constant for :exp_decay (required there).
			target_median::Float64: desired median transformed weight for :exp_decay's
				default scale c (default 10.0).
			weight_col::Symbol: name of the distance column to transform (default :weight).
		Returns:
			NamedTuple (edges::DataFrame, n_dropped::Int, c::Float64, method::Symbol):
				edges carries an integer count-like :weight; n_dropped counts edges that
				mapped to weight 0 and were removed (treated as no edge).
		Notes:
			Maps lower-is-stronger distances onto the non-negative integer count scale
			the reconstruction framework assumes. Intervals from the framework then apply
			to the TRANSFORMED network, not the original distances — nonlinear maps may
			distort interval structure, though rank orderings are often preserved.

			:scaled_reciprocal — w' = round(c / w), c = max(w). Strictly-positive
				distances only; zero-distance edges take the max transformed weight. Never
				drops edges (min count 1).
			:max_minus — w' = round(max(w) - w). Linear inversion; max-distance maps to 0
				(dropped), distance-zero to the max count.
			:exp_decay — w' = round(c * exp(-w / tau)); c defaults so the median
				transformed weight ≈ target_median.
		"""

		#	Validation
			method in (:scaled_reciprocal, :max_minus, :exp_decay) ||
				throw(ArgumentError("method must be :scaled_reciprocal, :max_minus, or :exp_decay; got $method"))
			weight_col in propertynames(edges) ||
				throw(ArgumentError("edges has no column $weight_col"))
			w = Float64.(edges[!, weight_col])
			any(<(0.0), w) &&
				throw(DomainError(minimum(w), "distance weights must be non-negative"))

		#	Compute Transformed Counts
			wmax = maximum(w)
			c = NaN
			if method === :scaled_reciprocal
				#	Reciprocal scaled by max distance; zero-distance -> max count
					c = wmax
					pos = w[w .> 0.0]
					max_count = isempty(pos) ? 1 : round(Int, c / minimum(pos))
					counts = [wi > 0.0 ? round(Int, c / wi) : max_count for wi in w]
			elseif method === :max_minus
				#	Linear inversion; max distance -> 0 (no edge)
					counts = round.(Int, wmax .- w)
			else
				#	Exponential decay; default c places the median count at target_median
					tau === nothing &&
						throw(ArgumentError(":exp_decay requires a positive tau"))
					tau > 0.0 ||
						throw(ArgumentError("tau must be > 0, got $tau"))
					decay = exp.(-w ./ tau)
					med = median(decay)
					c = med > 0.0 ? target_median / med : target_median
					counts = round.(Int, c .* decay)
			end

		#	Drop Zero-Weight Edges (treated as no edge)
			keep = counts .> 0
			n_dropped = count(!, keep)
			out = edges[keep, :]
			out[!, :weight] = counts[keep]

		#	Return
			return (edges = out, n_dropped = n_dropped, c = c, method = method)
	end
	@doc raw"""
	**Description**
	Transform a distance-semantic edge weight column (where a *smaller* weight means a
	*stronger* tie) onto the non-negative integer count scale the reconstruction
	framework assumes (where a *larger* weight means a stronger tie). Networks whose
	weights are travel times, transit costs, dissimilarities, or any other
	"lower = stronger" quantity must be passed through this step before reconstruction;
	count-semantic networks (interaction counts, co-appearances) need no transformation.

	**Usage**
	`transform_distance_weights(edges; method=:scaled_reciprocal, tau=nothing, target_median=10.0, weight_col=:weight)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, and a distance-semantic weight
	  column. Weights must be non-negative.
	- `method::Symbol`: Transformation to apply. One of `:scaled_reciprocal` (default),
	  `:max_minus`, or `:exp_decay` (see **Details**).
	- `tau::Union{Nothing,Float64}`: Length scale for `:exp_decay`; required and must be
	  `> 0` for that method, ignored otherwise (default `nothing`).
	- `target_median::Float64`: Desired median transformed weight, used only to set the
	  default scale constant for `:exp_decay` (default `10.0`).
	- `weight_col::Symbol`: Name of the distance column to transform (default `:weight`).
	  The transformed counts are always written to a `:weight` column on the output.

	**Details**
	Let $w$ be the observed distances. The three transforms are:

	- **`:scaled_reciprocal`** (default): $w' = \mathrm{round}(c / w)$ with $c = \max(w)$.
	  Strictly-positive distances only; any zero-distance edges take the maximum
	  transformed weight. The longest distance maps to count $1$, so this method never
	  drops edges.
	- **`:max_minus`**: $w' = \mathrm{round}(\max(w) - w)$. A linear inversion suited to
	  distances with a meaningful upper bound. Distance zero maps to the maximum count;
	  the maximum distance maps to $0$ and is dropped (treated as no edge).
	- **`:exp_decay`**: $w' = \mathrm{round}(c \cdot \exp(-w / \tau))$, with $c$ chosen so
	  that the median transformed weight is approximately `target_median`.

	Edges whose transformed weight rounds to $0$ are removed (treated as absent ties);
	`:scaled_reciprocal` removes none, while `:max_minus` and `:exp_decay` may.

	Credible intervals subsequently returned by the framework apply to the **transformed**
	network, not to the original distances. Because the transforms are nonlinear, they
	can distort interval structure; rank-based conclusions (which node is most central)
	are usually preserved across reasonable transforms, but distance-dependent measures
	(closeness, betweenness, mean inverse distance) measure something materially different
	on the transformed scale. Back-transforming interval bounds to the original distance
	scale must be done with that distortion in mind.

	**Value**
	A `NamedTuple` with fields:
	- `edges::DataFrame`: The transformed edge list (`:src`, `:dst`, integer-valued
	  `:weight`), with zero-weight edges removed.
	- `n_dropped::Int`: Number of edges removed because they mapped to weight $0$.
	- `c::Float64`: The scale constant used — $\max(w)$ for `:scaled_reciprocal`, the
	  median-matching constant for `:exp_decay`, and `NaN` for `:max_minus` (which uses
	  no multiplicative scale).
	- `method::Symbol`: The transform that was applied (echoed).

	**Examples**
	```julia
		using DataFrames

		#	Travel-time network: smaller weight = stronger tie
			edges = DataFrame(src = [1, 2, 3], dst = [2, 3, 1], weight = [2.0, 8.0, 4.0])

		#	Default scaled-reciprocal: strongest (shortest) tie gets the largest count
			out = transform_distance_weights(edges)
			out.edges.weight        # => [4, 1, 2]

		#	Linear inversion; the longest tie maps to 0 and is dropped
			transform_distance_weights(edges; method = :max_minus).n_dropped   # => 1

		#	Exponential decay with a chosen length scale
			transform_distance_weights(edges; method = :exp_decay, tau = 3.0)
	```

	**See Also**
	`_transform_weights_for_path_cost`, `closeness_centrality`, `betweenness_centrality`,
	`mean_inverse_distance`, `reconstruct_network`, `build_reconstruction_corpus`
	""" transform_distance_weights

################################
#   SECTION 2: DEGREE FAMILY   #
################################

#	Helper Function for freeman_degree_normalization: Bipartite Mode Counts
	function _bipartite_counts(types::AbstractVector{Bool})
		"""
		Args:
			types::AbstractVector{Bool}: node mode flags (true = first mode)
		Returns:
			Tuple{Int, Int}: (first_mode_count, second_mode_count)
		Notes:
			Helper for Freeman normalization in bipartite graphs. The first-mode
			count is used as R (the denominator's row count) and the second-mode
			count is used as N (the column count / max possible neighbors).
		"""

		#	Count Each Mode
			first_mode  = sum(types)
			second_mode = length(types) - first_mode

		#	Return Pair
			return (first_mode, second_mode)
	end

#	In-Degree (Sum of Incoming Edge Weights per Node)
	function in_degree(edges::DataFrame;
	                  nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                  weighted::Bool = true,
	                  normalize::Bool = false,
	                  agg_func::Function = sum,
	                  n::Union{Nothing, Int} = nothing)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
				When supplied, isolates (nodes with no edges) appear in the
				returned DataFrame with in_degree = 0. When nothing, only
				nodes touched by an edge are returned.
			weighted::Bool: use edge weights if available (default = true)
			normalize::Bool: return Freeman-normalized in-degree (default = false)
			agg_func::Function: aggregation for parallel edges (default = sum)
			n::Union{Nothing, Int}: optional graph order for Freeman normalization
		Returns:
			DataFrame: columns [node, in_degree]
		Notes:
			Unnormalized: column sums of the adjacency.
			Normalized: delegates to freeman_degree_normalization (mode=:in).
			The `nodes` argument is the primary mechanism for including
			isolates in Phase 0 reporting.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return DataFrame(node = [], in_degree = Float64[])
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Normalized Path: Delegate to Freeman
			if normalize
				df = freeman_degree_normalization(clean_edges;
				                                  nodes = nodes,
				                                  mode = :in,
				                                  directed = true,
				                                  bipartite = false,
				                                  weighted = weighted,
				                                  agg_func = agg_func,
				                                  n = n)
				rename!(df, :freeman_degree => :in_degree)
				return df
			end

		#	Unnormalized Path: Build Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(clean_edges;
				                                              nodes = nodes,
				                                              weighted = weighted)
			end
			in_deg_values = vec(sum(adj, dims = 1))

		#	Assembling Result (Use :id if Nodes Is a DataFrame)
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, in_degree = in_deg_values)
	end
	@doc raw"""
	**Description**
	Compute in-degree centrality for each node in a directed network. Returns
	either raw in-degree (sum of incoming edge weights) or Freeman-normalized
	in-degree. Optionally accepts a `nodes` argument to ensure isolates appear
	in the returned DataFrame.

	**Usage**
	`in_degree(edges::DataFrame; nodes=nothing, weighted=true, normalize=false, agg_func=sum, n=nothing)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. When
	  supplied (typically the `nodes` field of a `load_graphml` result),
	  isolates appear in the output with `in_degree = 0`. When `nothing`,
	  only nodes that appear in some edge are returned.
	- `weighted::Bool`: Use edge weights if available (default `true`).
	- `normalize::Bool`: Apply Freeman normalization (default `false`).
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).
	- `n::Union{Nothing,Int}`: Explicit graph order for normalization.

	**Details**
	When unnormalized, in-degree is the sum of weights of incoming edges. When
	normalized, delegates to `freeman_degree_normalization` with `mode=:in`.

	Pass `nodes` whenever isolates matter for downstream computation — most
	importantly for centralization measures, since omitting isolates changes
	$N$ and shifts the standard deviation of the centrality vector.

	**Value**
	A `DataFrame` with columns `:node` and `:in_degree`.

	**Examples**
```julia
	using DataFrames
	edges = DataFrame(src=[1, 2], dst=[2, 3])
	nodes = DataFrame(id=[1, 2, 3, 4], label=["A", "B", "C", "D"])
	in_degree(edges; nodes=nodes)   # 4 rows, node 4 has in_degree = 0
	in_degree(edges)                # 3 rows, no row for node 4
```

	**See Also**
	`out_degree`, `total_degree`, `freeman_degree_normalization`
	""" in_degree

#	Out-Degree (Sum of Outgoing Edge Weights per Node)
	function out_degree(edges::DataFrame;
	                   nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                   weighted::Bool = true,
	                   normalize::Bool = false,
	                   agg_func::Function = sum,
	                   n::Union{Nothing, Int} = nothing)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: use edge weights if available (default = true)
			normalize::Bool: return Freeman-normalized out-degree (default = false)
			agg_func::Function: aggregation for parallel edges (default = sum)
			n::Union{Nothing, Int}: optional graph order for normalization
		Returns:
			DataFrame: columns [node, out_degree]
		Notes:
			Unnormalized: row sums of the adjacency.
			Normalized: delegates to freeman_degree_normalization (mode=:out).
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return DataFrame(node = [], out_degree = Float64[])
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Normalized Path: Delegate to Freeman
			if normalize
				df = freeman_degree_normalization(clean_edges;
				                                  nodes = nodes,
				                                  mode = :out,
				                                  directed = true,
				                                  bipartite = false,
				                                  weighted = weighted,
				                                  agg_func = agg_func,
				                                  n = n)
				rename!(df, :freeman_degree => :out_degree)
				return df
			end

		#	Unnormalized Path: Build Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(clean_edges;
				                                              nodes = nodes,
				                                              weighted = weighted)
			end
			out_deg_values = vec(sum(adj, dims = 2))

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, out_degree = out_deg_values)
	end
	@doc raw"""
	**Description**
	Compute out-degree centrality for each node in a directed network. See
	`in_degree` for the symmetric counterpart.

	**Usage**
	`out_degree(edges::DataFrame; nodes=nothing, weighted=true, normalize=false, agg_func=sum, n=nothing)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe; pass to
	  include isolates in the output.
	- `weighted::Bool`: Use edge weights if available (default `true`).
	- `normalize::Bool`: Apply Freeman normalization (default `false`).
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).
	- `n::Union{Nothing,Int}`: Explicit graph order for normalization.

	**Details**
	Unnormalized: row sums of the adjacency. Normalized: delegates to
	`freeman_degree_normalization` with `mode=:out`.

	**Value**
	A `DataFrame` with columns `:node` and `:out_degree`.

	**See Also**
	`in_degree`, `total_degree`, `freeman_degree_normalization`
	""" out_degree

#	Total Degree (Sum of All Adjacent Edge Weights per Node)
	function total_degree(edges::DataFrame;
	                     nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                     weighted::Bool = true,
	                     normalize::Bool = false,
	                     agg_func::Function = sum,
	                     n::Union{Nothing, Int} = nothing,
	                     drop_self_loops::Bool = false,
	                     directed::Bool = true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: use edge weights if available (default = true)
			normalize::Bool: return Freeman-normalized total degree (default = false)
			agg_func::Function: aggregation for parallel edges (default = sum)
			n::Union{Nothing, Int}: optional graph order for normalization
			drop_self_loops::Bool: remove self-loops before computing (default = false)
			directed::Bool: treat as directed (default = true)
		Returns:
			DataFrame: columns [node, total_degree]
		Notes:
			Directed: total = in + out - diag (self-loops counted once).
			Undirected sym storage: total = (in + out - diag) / 2.
			Undirected upper-triangular: total = in + out - diag.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return DataFrame(node = [], total_degree = Float64[])
			end

		#	Optional Self-Loop Removal
			edges_effective = edges
			if drop_self_loops
				if hasproperty(edges_effective, :weight)
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst,
					                                  [:src, :dst, :weight]]
				else
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst,
					                                  [:src, :dst]]
				end
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges_effective; agg_func = agg_func)

		#	Normalized Path: Delegate to Freeman
			if normalize
				df = freeman_degree_normalization(clean_edges;
				                                  nodes = nodes,
				                                  mode = :all,
				                                  directed = directed,
				                                  bipartite = false,
				                                  weighted = weighted,
				                                  agg_func = agg_func,
				                                  n = n,
				                                  drop_self_loops = drop_self_loops)
				rename!(df, :freeman_degree => :total_degree)
				return df
			end

		#	Unnormalized Path: Build Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(clean_edges;
				                                              nodes = nodes,
				                                              weighted = weighted)
			end
			row_sums = vec(sum(adj, dims = 2))
			col_sums = vec(sum(adj, dims = 1))
			diagonal = collect(diag(adj))

		#	Combine According to Direction Convention
			if directed
				total_deg_values = row_sums .+ col_sums .- diagonal
			else
				is_sym = _is_symmetric(adj; directed = false)
				raw_total = row_sums .+ col_sums .- diagonal
				total_deg_values = is_sym ? (raw_total ./ 2) : raw_total
			end

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, total_degree = total_deg_values)
	end
	@doc raw"""
	**Description**
	Compute total degree centrality for each node. For directed networks this is
	in-degree plus out-degree (self-loops counted once). For undirected networks
	this is simply degree. Pass `nodes` to include isolates in the output.

	**Usage**
	`total_degree(edges::DataFrame; nodes=nothing, weighted=true, normalize=false, agg_func=sum, n=nothing, drop_self_loops=false, directed=true)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe; pass to
	  include isolates.
	- `weighted::Bool`: Use edge weights if available (default `true`).
	- `normalize::Bool`: Apply Freeman normalization (default `false`).
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).
	- `n::Union{Nothing,Int}`: Explicit graph order for normalization.
	- `drop_self_loops::Bool`: Remove self-loops before computing (default `false`).
	- `directed::Bool`: Treat as directed (default `true`).

	**Value**
	A `DataFrame` with columns `:node` and `:total_degree`.

	**See Also**
	`in_degree`, `out_degree`, `freeman_degree_normalization`
	""" total_degree

#	Freeman-Normalized Degree (Node-Level)
	function freeman_degree_normalization(edges::DataFrame;
	                                     nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                                     mode::Symbol = :all,
	                                     directed::Bool = true,
	                                     bipartite::Bool = false,
	                                     types::Union{Nothing, AbstractVector{Bool}} = nothing,
	                                     weighted::Bool = true,
	                                     agg_func::Function = sum,
	                                     n::Union{Nothing, Int} = nothing,
	                                     drop_self_loops::Bool = false,
	                                     atol::Float64 = 1e-12)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			mode::Symbol: :all | :out | :in (default = :all)
			directed::Bool: treat as directed (default = true)
			bipartite::Bool: indicate bipartite (default = false)
			types::Union{Nothing, AbstractVector{Bool}}: node mode flags
			weighted::Bool: use edge weights if available (default = true)
			agg_func::Function: aggregation for parallel edges (default = sum)
			n::Union{Nothing, Int}: optional explicit graph order
			drop_self_loops::Bool: remove self-loops before normalization (default = false)
			atol::Float64: symmetry-test tolerance (default = 1e-12)
		Returns:
			DataFrame: columns [node, freeman_degree]
		Notes:
			Freeman (1979) individual-level degree normalization. When `nodes`
			is supplied, isolates receive freeman_degree = 0 and contribute
			to the N denominator. When `n` is supplied, it overrides the
			adjacency-derived N for the denominator only.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges must have :src and :dst columns"))
			end
			if !(mode in (:all, :out, :in))
				throw(ArgumentError("mode must be :all, :out, or :in"))
			end
			if nrow(edges) == 0
				return DataFrame(node = Any[], freeman_degree = Float64[])
			end

		#	Optional Self-Loop Removal at Edge Level
			edges_effective = edges
			if drop_self_loops
				if hasproperty(edges_effective, :weight)
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst,
					                                  [:src, :dst, :weight]]
				else
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst,
					                                  [:src, :dst]]
				end
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges_effective; agg_func = agg_func)

		#	Build Sparse Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(clean_edges;
				                                              nodes = nodes,
				                                              weighted = weighted)
			end

		#	Compute Marginals and Diagonal
			row_sums = vec(sum(adj, dims = 2))
			col_sums = vec(sum(adj, dims = 1))
			diagonal = collect(diag(adj))

		#	Determine V (Max Edge Weight)
			V = (weighted && hasproperty(clean_edges, :weight) && !isempty(clean_edges.weight)) ?
			    maximum(clean_edges.weight) : 1.0

		#	Determine N for Denominator
			adj_n = size(adj, 1)
			N = isnothing(n) ? adj_n : n
			R = N

		#	Validate Explicit n
			if !isnothing(n) && n < adj_n
				throw(ArgumentError("Supplied n ($n) cannot be less than connected node count ($adj_n)"))
			end

		#	Bipartite Handling
			if bipartite
				if types === nothing
					throw(ArgumentError("bipartite=true requires a types::Vector{Bool}"))
				end
				if length(types) != adj_n
					throw(ArgumentError("length(types) must equal adjacency size ($adj_n)"))
				end
				first_mode, second_mode = _bipartite_counts(types)
				R = first_mode
				if isnothing(n)
					N = second_mode
				end
			end

		#	Edge Case: Insufficient Neighbors
			if N <= 1
				node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
				return DataFrame(node = node_col, freeman_degree = zeros(Float64, adj_n))
			end

		#	Initialize Numerator and Denominator
			numerator = zeros(Float64, adj_n)
			denom = 0.0

		#	Compute by Network Type
			if !directed
				#	Undirected: C_D(i) = degree(i) / (V * (N - 1))
					is_sym_undirected = issymmetric(adj)
					deg_raw = row_sums .+ col_sums .- diagonal
					degree  = is_sym_undirected ? (deg_raw ./ 2) : deg_raw
					numerator .= degree
					denom = V * (N - 1)
			else
				#	Directed: check empirical symmetry for total-degree denominator
					is_sym = _is_symmetric(adj; directed = directed, atol = atol)

					if mode == :all
						numerator .= row_sums .+ col_sums .- diagonal
						denom = is_sym ? (V * (N - 1)) : (2 * V * (N - 1))
					elseif mode == :out
						numerator .= row_sums .- diagonal
						denom = V * (N - 1)
					else
						numerator .= col_sums .- diagonal
						denom = V * (N - 1)
					end
			end

		#	Protect Against Zero Denominator
			if denom == 0.0
				node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
				return DataFrame(node = node_col, freeman_degree = zeros(Float64, adj_n))
			end

		#	Compute Normalized Scores
			scores = numerator ./ denom

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, freeman_degree = scores)
	end
	@doc raw"""
	**Description**
	Compute Freeman (1979) individual-level degree normalization for each node.
	Returns a node-level score in $[0, V]$, with $V = 1$ for unweighted graphs.

	**Usage**
	`freeman_degree_normalization(edges::DataFrame; nodes=nothing, mode=:all, directed=true, bipartite=false, types=nothing, weighted=true, agg_func=sum, n=nothing, drop_self_loops=false, atol=1e-12)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. When
	  supplied, isolates receive `freeman_degree = 0` and contribute to the
	  $N$ denominator. Pass when working with networks that have isolates
	  (e.g., Scotland Interlock has 16).
	- `mode::Symbol`: `:all`, `:out`, or `:in`.
	- `directed::Bool`: Treat as directed (default `true`).
	- `bipartite::Bool`: Bipartite network mode (default `false`).
	- `types::Union{Nothing,Vector{Bool}}`: Mode flags for bipartite case.
	- `weighted::Bool`: Use edge weights (default `true`).
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).
	- `n::Union{Nothing,Int}`: Explicit denominator override. Use sparingly;
	  `nodes` is the cleaner way to control $N$.
	- `drop_self_loops::Bool`: Remove self-loops before normalization (default `false`).
	- `atol::Float64`: Symmetry-test tolerance (default `1e-12`).

	**Details**
	Standard Freeman (1979) normalization with edge-weight scaling $V$:
	- Undirected: $C_D(i) = \text{deg}(i) / (V \cdot (N-1))$
	- Directed in/out: $C(i) / (V \cdot (N-1))$
	- Directed total in symmetric graph: $C(i) / (V \cdot (N-1))$
	- Directed total in asymmetric graph: $C(i) / (2V \cdot (N-1))$

	**Value**
	A `DataFrame` with columns `:node` and `:freeman_degree`.

	**References**
	- Freeman LC (1979). "Centrality in social networks: Conceptual clarification."
	  *Social Networks* 1: 215–239.

	**See Also**
	`freeman_degree_centralization`, `in_degree`, `out_degree`, `total_degree`
	""" freeman_degree_normalization

#	Network-Level Freeman Degree Centralization
	function freeman_degree_centralization(edges::DataFrame;
	                                      nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                                      mode::Symbol = :all,
	                                      directed::Bool = true,
	                                      drop_self_loops::Bool = true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			mode::Symbol: :all | :out | :in (default = :all)
			directed::Bool: treat as directed (default = true)
			drop_self_loops::Bool: drop self-loops before computing (default = true)
		Returns:
			Float64: Freeman centralization in [0, 1]
		Notes:
			Always binarizes the adjacency internally. See @doc for the
			rationale. Pass `nodes` to ensure isolates are counted toward N.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges must have :src and :dst columns"))
			end
			if !(mode in (:all, :out, :in))
				throw(ArgumentError("mode must be :all, :out, or :in"))
			end
			if nrow(edges) == 0
				return 0.0
			end

		#	Binarize Edges (Drop Weight Information for Centralization)
			edges_binary = DataFrame(src = edges.src, dst = edges.dst)
			edges_binary.weight = ones(Float64, nrow(edges_binary))
			edges_binary = _aggregate_multi_edges(edges_binary; agg_func = maximum)

		#	Optional Self-Loop Removal
			if drop_self_loops
				edges_binary = edges_binary[edges_binary.src .!= edges_binary.dst, :]
				if nrow(edges_binary) == 0
					return 0.0
				end
			end

		#	Build Binary Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, _ = _edgelist_to_sparse_matrix(edges_binary; weighted = false)
			else
				adj, _, _ = _graph_to_sparse_matrix(edges_binary;
				                                    nodes = nodes,
				                                    weighted = false)
			end
			N = size(adj, 1)

		#	Edge Case: Undefined Centralization
			if N <= 2
				return 0.0
			end

		#	Compute Degree Counts According to Mode and Direction
			row_sums = vec(sum(adj, dims = 2))
			col_sums = vec(sum(adj, dims = 1))
			diagonal = collect(diag(adj))

			if !directed
				is_sym_undirected = issymmetric(adj)
				raw_total = row_sums .+ col_sums .- diagonal
				degrees = is_sym_undirected ? (raw_total ./ 2) : raw_total
			else
				if mode == :all
					degrees = row_sums .+ col_sums .- diagonal
				elseif mode == :out
					degrees = row_sums .- diagonal
				else
					degrees = col_sums .- diagonal
				end
			end

		#	Compute Freeman Centralization
			C_max = maximum(degrees)
			numerator = sum(C_max .- degrees)
			denominator = (N - 1) * (N - 2)

		#	Return Normalized
			return denominator == 0 ? 0.0 : numerator / denominator
	end

######################################
#   SECTION 3: PATH-BASED MEASURES   #
######################################

# ====================================================================
# Closeness centrality, betweenness centrality, and mean inverse
# distance — all built on the same all-pairs shortest-path foundation
# (BFS on binarized adjacency). For Phase 0 these functions accept
# `edge_interpretation=:ignore` only; weighted path computation via
# Dijkstra is deferred to a later commit.
#
# Internal helpers:
#   _build_neighbor_lists       — CSR-style adjacency for fast BFS
#   _bfs_distances_and_sigma!   — single-source BFS computing distances,
#                                  sigma (#geodesics), and ordered stack
#                                  for Brandes' backward pass
#   _brandes_pass!              — single-source Brandes betweenness pass
#   _all_pairs_inverse_distance_sum — shared computation for closeness
#                                  per-source partial sums and global
#                                  MID accumulator
#
# Public API:
#   closeness_centrality        — SMM-style inverse-distance closeness
#   betweenness_centrality      — Brandes 2001, threaded over sources
#   mean_inverse_distance       — average of 1/d(i,j), optionally / log N
# ====================================================================

#	Helper Function for Path-Based Measures: Build CSR-Style Neighbor Lists
	function _build_neighbor_lists(adj::SparseMatrixCSC{<:Real, <:Integer})
		"""
		Args:
			adj::SparseMatrixCSC: sparse adjacency matrix (binary or weighted)
		Returns:
			Tuple{Vector{Int}, Vector{Int}}: (neighbor_starts, neighbor_data)
				neighbor_starts::Vector{Int}: length n+1; neighbors of node i
					are at indices neighbor_starts[i] : neighbor_starts[i+1]-1
				neighbor_data::Vector{Int}: concatenated out-neighbor IDs
		Notes:
			SparseMatrixCSC stores columns contiguously, but BFS needs row-wise
			access (out-neighbors of source node u). For repeated BFS calls,
			pre-converting to CSR-style row-indexed neighbor lists is much
			faster than iterating CSC columns per query.

			Builds the out-neighbor list. For directed graphs this is
			adj[i, :] (row i = targets of edges from i). For symmetrized
			adjacency it's the same as in-neighbors.

			Treats the matrix as binary — nonzero entries are neighbors,
			zero entries are not. Weight values are discarded.
		"""

		#	Dimensions
			n = size(adj, 1)

		#	First Pass: Count Out-Neighbors per Node
			row_counts = zeros(Int, n)
			rows = rowvals(adj)
			vals = nonzeros(adj)
			@inbounds for j in 1:n
				for ptr in nzrange(adj, j)
					i = rows[ptr]
					if vals[ptr] != 0
						row_counts[i] += 1
					end
				end
			end

		#	Build CSR Start Offsets
			neighbor_starts = Vector{Int}(undef, n + 1)
			neighbor_starts[1] = 1
			@inbounds for i in 1:n
				neighbor_starts[i + 1] = neighbor_starts[i] + row_counts[i]
			end

		#	Second Pass: Populate Neighbor Data
			total_edges = neighbor_starts[n + 1] - 1
			neighbor_data = Vector{Int}(undef, total_edges)
			write_ptr = copy(neighbor_starts)
			@inbounds for j in 1:n
				for ptr in nzrange(adj, j)
					i = rows[ptr]
					if vals[ptr] != 0
						neighbor_data[write_ptr[i]] = j
						write_ptr[i] += 1
					end
				end
			end

		#	Return CSR Representation
			return (neighbor_starts, neighbor_data)
	end

#	Helper Function for Weighted Path-Based Measures: Build CSR with Edge Weights
	function _build_weighted_neighbor_lists(adj::SparseMatrixCSC{<:Real, <:Integer})
		"""
		Args:
			adj::SparseMatrixCSC: sparse adjacency matrix with edge weights
		Returns:
			Tuple{Vector{Int}, Vector{Int}, Vector{Float64}}:
				(neighbor_starts, neighbor_data, neighbor_weights)
				neighbor_starts::Vector{Int}: length n+1; neighbors of node i
					are at indices neighbor_starts[i] : neighbor_starts[i+1]-1
				neighbor_data::Vector{Int}: concatenated out-neighbor IDs
				neighbor_weights::Vector{Float64}: edge weights, parallel to
					neighbor_data
		Notes:
			Weighted analog of _build_neighbor_lists. Preserves edge weights
			as a parallel Vector{Float64} so the Dijkstra forward pass can
			access them in O(1) per neighbor without indexing back into the
			sparse matrix.

			Zero-weight entries are skipped (treated as absent edges) so the
			neighbor list is well-formed for weighted graphs that have
			structural zeros in the sparse representation.

			Caller is responsible for ensuring weights are non-negative.
			Negative weights would invalidate Dijkstra; the helper does not
			check.
		"""

		#	Dimensions
			n = size(adj, 1)

		#	First Pass: Count Out-Neighbors per Node
			row_counts = zeros(Int, n)
			rows = rowvals(adj)
			vals = nonzeros(adj)
			@inbounds for j in 1:n
				for ptr in nzrange(adj, j)
					i = rows[ptr]
					if vals[ptr] != 0
						row_counts[i] += 1
					end
				end
			end

		#	Build CSR Start Offsets
			neighbor_starts = Vector{Int}(undef, n + 1)
			neighbor_starts[1] = 1
			@inbounds for i in 1:n
				neighbor_starts[i + 1] = neighbor_starts[i] + row_counts[i]
			end

		#	Second Pass: Populate Neighbor IDs and Weights
			total_edges      = neighbor_starts[n + 1] - 1
			neighbor_data    = Vector{Int}(undef, total_edges)
			neighbor_weights = Vector{Float64}(undef, total_edges)
			write_ptr = copy(neighbor_starts)
			@inbounds for j in 1:n
				for ptr in nzrange(adj, j)
					i = rows[ptr]
					w = vals[ptr]
					if w != 0
						neighbor_data[write_ptr[i]]    = j
						neighbor_weights[write_ptr[i]] = Float64(w)
						write_ptr[i] += 1
					end
				end
			end

		#	Return CSR Representation with Weights
			return (neighbor_starts, neighbor_data, neighbor_weights)
	end

#	Helper Function for Weighted Path-Based Measures: Transform Weights to Path Cost
	function _transform_weights_for_path_cost(neighbor_weights::Vector{Float64},
												edge_interpretation::Symbol)
		"""
		Args:
			neighbor_weights::Vector{Float64}: raw edge weights from _build_weighted_neighbor_lists
			edge_interpretation::Symbol: :tie_strength or :distance
		Returns:
			Vector{Float64}: transformed costs suitable for Dijkstra
		Notes:
			Maps raw edge weights to Dijkstra-ready costs based on the
			semantic interpretation of the weight.

			:tie_strength — weights represent intensity / frequency of the
				tie; convert to cost via 1/w. Stronger ties produce shorter
				weighted distances. Standard convention for co-appearance,
				interaction-count, and shared-attribute networks.

			:distance — weights are already distances / costs (e.g., travel
				times, transit costs, dissimilarities). Used as-is.

			For :tie_strength, zero or negative weights map to Inf (treated
			as absent edges), since they would otherwise produce undefined
			or negative path costs. For :distance, negative weights also
			map to Inf since Dijkstra requires non-negative costs. Both
			cases result in an edge that exists in the adjacency but cannot
			be used in shortest-path computation.

			Does not mutate the input array; returns a new Vector.
		"""

		#	Validation
			if !(edge_interpretation in (:tie_strength, :distance))
				throw(ArgumentError(
					"_transform_weights_for_path_cost: edge_interpretation=$(edge_interpretation) " *
					"not supported. Use :tie_strength or :distance."
				))
			end

		#	Allocate Output
			n_edges = length(neighbor_weights)
			costs   = Vector{Float64}(undef, n_edges)

		#	Apply Transformation
			if edge_interpretation === :tie_strength
				@inbounds for i in 1:n_edges
					w = neighbor_weights[i]
					costs[i] = (w > 0.0 && isfinite(w)) ? 1.0 / w : Inf
				end
			else  # :distance
				@inbounds for i in 1:n_edges
					w = neighbor_weights[i]
					costs[i] = (w >= 0.0 && isfinite(w)) ? w : Inf
				end
			end

		#	Return Transformed Costs
			return costs
	end

#	Helper Function for Weighted Path-Based Measures: Min-Heap Sift-Up (1-Indexed)
	function _heap_sift_up!(heap::Vector{Tuple{Float64, Int}}, i::Int)
		"""
		Args:
			heap::Vector{Tuple{Float64,Int}}: heap buffer, valid in 1:i
			i::Int: index of the just-inserted element to sift toward the root
		Returns:
			Nothing (heap mutated in place)
		Notes:
			Restores the min-heap property after inserting at position i.
			Tuples compare lexicographically, so (distance, node) orders by
			distance with a stable node tiebreak.
		"""
		@inbounds while i > 1
			parent = i >> 1
			if heap[i] < heap[parent]
				heap[i], heap[parent] = heap[parent], heap[i]
				i = parent
			else
				break
			end
		end
		return nothing
	end

#	Helper Function for Weighted Path-Based Measures: Min-Heap Sift-Down (1-Indexed)
	function _heap_sift_down!(heap::Vector{Tuple{Float64, Int}}, heap_size::Int)
		"""
		Args:
			heap::Vector{Tuple{Float64,Int}}: heap buffer, valid in 1:heap_size
			heap_size::Int: number of valid elements after the root was replaced
		Returns:
			Nothing (heap mutated in place)
		Notes:
			Restores the min-heap property after the root has been overwritten
			by the former last element (standard pop procedure). Operates on
			the first heap_size entries only.
		"""
		i = 1
		@inbounds while true
			l = 2 * i
			r = l + 1
			smallest = i
			if l <= heap_size && heap[l] < heap[smallest]
				smallest = l
			end
			if r <= heap_size && heap[r] < heap[smallest]
				smallest = r
			end
			if smallest != i
				heap[i], heap[smallest] = heap[smallest], heap[i]
				i = smallest
			else
				break
			end
		end
		return nothing
	end

#	Helper Function for Weighted Path-Based Measures: Single-Source Dijkstra
	function _dijkstra_distances_and_sigma!(neighbor_starts::Vector{Int},
											neighbor_data::Vector{Int},
											neighbor_costs::Vector{Float64},
											source::Int,
											distance::Vector{Float64},
											sigma::Vector{Float64},
											stack_order::Vector{Int},
											heap::Vector{Tuple{Float64, Int}},
											finalized::Vector{Bool})
		"""
		Args:
			neighbor_starts::Vector{Int}: CSR row pointers (length n+1)
			neighbor_data::Vector{Int}: CSR neighbor IDs (length total_edges)
			neighbor_costs::Vector{Float64}: parallel edge costs from
				_transform_weights_for_path_cost
			source::Int: source node ID (1-indexed)
			distance::Vector{Float64}: pre-allocated work buffer (reset here);
				filled with shortest weighted distances (Inf if unreachable)
			sigma::Vector{Float64}: pre-allocated work buffer (reset here);
				filled with shortest-path counts
			stack_order::Vector{Int}: pre-allocated; filled with finalization order
			heap::Vector{Tuple{Float64,Int}}: pre-allocated heap buffer of
				capacity >= length(neighbor_data) + 1; used via an explicit
				heap_size counter, never push!/pop!
			finalized::Vector{Bool}: pre-allocated work buffer (reset here)
		Returns:
			Int: number of reachable nodes, including the source
		Notes:
			Weighted analog of _bfs_distances_and_sigma!. Computes:
				distance[v] = shortest weighted distance from source to v
							  (Inf if unreachable)
				sigma[v]    = number of distinct shortest paths from source to v
				stack_order = nodes in non-decreasing finalized-distance order
							  (Dijkstra closes nodes in this order, analogous to
							  BFS layer order)

			All work buffers are passed in by the caller and reset at the top
			of this function, matching the buffer-passing convention used by
			_bfs_distances_and_sigma! and _brandes_pass!. The heap is a
			fixed-capacity Vector indexed by an explicit heap_size counter;
			entries are written by index and the heap property is maintained
			by _heap_sift_up! / _heap_sift_down!. No push!/pop!, no per-source
			allocation, no nested closures.

			Lazy deletion: a node may appear multiple times in the heap with
			decreasing keys (deferred relaxation). Stale entries are skipped
			via the `finalized` flag; only the first finalization of a node
			counts.

			Heap capacity: each strictly-improving relaxation writes one heap
			entry, bounded by the number of finite-cost edges. A buffer of
			length (length(neighbor_data) + 1) is always sufficient. The +1
			covers the initial source seed.

			Shortest-path counting (sigma):
				First discovery (old_dist == Inf): always strictly shorter;
					relax unconditionally, set sigma[w] = sigma[v], insert.
				Subsequent visit, strictly shorter (dist[v]+cost < dist[w]-tol):
					relax, set sigma[w] = sigma[v], insert.
				Subsequent visit, equally short (within tol):
					accumulate sigma[w] += sigma[v].

			The first-discovery case MUST bypass the tolerance arithmetic.
			When old_dist == Inf, the relative tolerance scale = max(|new|,
			|old|, 1) is Inf, so tol = eps * Inf = Inf, and old_dist - tol =
			Inf - Inf = NaN. The comparison new_dist < NaN is then false,
			which would wrongly reject every first relaxation and leave all
			non-source nodes unreachable. Testing old_dist == Inf explicitly
			(an exact comparison; fill!(distance, Inf) sets it precisely)
			avoids the NaN trap. Once a node has a finite tentative distance,
			the relative-tolerance comparison is well-defined.

			Floating-point equality (for the equal-distance sigma branch)
			uses a relative tolerance scaled by the magnitude of the
			distances being compared, since accumulated path costs are
			subject to rounding.

			Inf-cost edges are skipped (treated as absent), supporting the
			:tie_strength transformation that maps zero/non-positive weights
			to Inf.
		"""

		#	Dimensions and Reset
			n = length(distance)
			fill!(distance, Inf)
			fill!(sigma, 0.0)
			fill!(finalized, false)

		#	Initialize Source
			distance[source] = 0.0
			sigma[source]    = 1.0

		#	Tolerance for Equal-Distance Comparison
			_DIJKSTRA_REL_EPS = 1e-12

		#	Seed the Heap (Explicit Size Counter, No push!)
			heap_size = 1
			@inbounds heap[1] = (0.0, source)

		#	Finalization Counter
			n_finalized = 0

		#	Main Dijkstra Loop
			@inbounds while heap_size > 0
				#	Pop Closest Node (Root): Save It, Move Last to Root, Shrink, Sift Down
					(d_v, v) = heap[1]
					heap[1] = heap[heap_size]
					heap_size -= 1
					if heap_size > 0
						_heap_sift_down!(heap, heap_size)
					end

				#	Skip Stale Heap Entries
					if finalized[v]
						continue
					end

				#	Finalize v
					finalized[v] = true
					n_finalized += 1
					stack_order[n_finalized] = v

				#	Relax Outgoing Edges
					sigma_v   = sigma[v]
					start_ptr = neighbor_starts[v]
					end_ptr   = neighbor_starts[v + 1] - 1
					for ptr in start_ptr:end_ptr
						w    = neighbor_data[ptr]
						cost = neighbor_costs[ptr]

						#	Skip Inf-Cost Edges (Treated as Absent)
							if !isfinite(cost)
								continue
							end

						new_dist = d_v + cost
						old_dist = distance[w]

						if old_dist == Inf
							#	First Discovery: Any Finite new_dist Is Strictly
							#	Shorter. Bypass the tolerance arithmetic, which
							#	would otherwise yield Inf - Inf = NaN and reject
							#	the relaxation.
								distance[w]     = new_dist
								sigma[w]        = sigma_v
								heap_size      += 1
								heap[heap_size] = (new_dist, w)
								_heap_sift_up!(heap, heap_size)
						else
							#	Subsequent Visit: Relative-Tolerance Comparison
							#	(both distances finite, so tol is well-defined)
								scale = max(abs(new_dist), abs(old_dist), 1.0)
								tol   = _DIJKSTRA_REL_EPS * scale

								if new_dist < old_dist - tol
									#	Strictly Shorter Path: Relax and Insert
										distance[w]     = new_dist
										sigma[w]        = sigma_v
										heap_size      += 1
										heap[heap_size] = (new_dist, w)
										_heap_sift_up!(heap, heap_size)
								elseif new_dist < old_dist + tol
									#	Equally Short Path: Accumulate Path Count
										sigma[w] += sigma_v
								end
								#	Otherwise: strictly longer, ignore
						end
					end
			end

		#	Return Number of Reachable Nodes
			return n_finalized
	end

#	Helper Function for Path-Based Measures: Single-Source BFS
	function _bfs_distances_and_sigma!(neighbor_starts::Vector{Int},
	                                  neighbor_data::Vector{Int},
	                                  source::Int,
	                                  distance::Vector{Int},
	                                  sigma::Vector{Float64},
	                                  stack_order::Vector{Int})
		"""
		Args:
			neighbor_starts::Vector{Int}: CSR row pointers (length n+1)
			neighbor_data::Vector{Int}: CSR neighbor IDs (length total_edges)
			source::Int: source node ID (1-indexed)
			distance::Vector{Int}: pre-allocated, will be filled (-1 = unreachable)
			sigma::Vector{Float64}: pre-allocated, will be filled (#geodesics)
			stack_order::Vector{Int}: pre-allocated, will be filled with BFS order
		Returns:
			Int: number of reachable nodes, including the source
		Notes:
			Computes:
				distance[v] = shortest-path distance from source to v (or -1)
				sigma[v]    = number of distinct shortest paths from source to v
				stack_order = nodes in non-decreasing distance order (BFS order)

			This is the forward pass used both by Brandes' betweenness
			accumulation and by closeness / MID summation. The caller is
			responsible for pre-allocating the three output vectors and
			resetting them between calls.

			Float64 sigma is used rather than Int because in large dense
			networks sigma can exceed typemax(Int) — Newman & Brandes both
			recommend Float64 for robustness.

			Self-loops on `source` are ignored (sigma[source] = 1, the empty
			path). Self-loops on other nodes are also ignored — they cannot
			shorten a path.

			Uses a FIFO queue implemented as a pre-allocated Vector{Int} with
			read/write pointers, avoiding any allocations during the BFS.
		"""

		#	Dimensions and Reset
			n = length(distance)
			fill!(distance, -1)
			fill!(sigma, 0.0)

		#	Initialize Source
			distance[source] = 0
			sigma[source]    = 1.0

		#	Use stack_order as Both Queue (Front..Back) and Result Stack
		#	Queue indices: q_head (next to pop), q_tail (next to push)
		#	The same array doubles as the BFS-order stack since we visit each
		#	node exactly once and the BFS order is exactly the popping order.
			stack_order[1] = source
			q_head = 1
			q_tail = 2  # next write position

		#	Forward BFS
			@inbounds while q_head < q_tail
				#	Pop Next Node from Queue
					v = stack_order[q_head]
					q_head += 1
					d_v = distance[v]
					sigma_v = sigma[v]

				#	Enumerate Out-Neighbors
					start_ptr = neighbor_starts[v]
					end_ptr   = neighbor_starts[v + 1] - 1
					for ptr in start_ptr:end_ptr
						w = neighbor_data[ptr]

						if distance[w] == -1
							#	First Discovery of w
								distance[w] = d_v + 1
								stack_order[q_tail] = w
								q_tail += 1
						end

						if distance[w] == d_v + 1
							#	w Reached via a Shortest Path Through v
								sigma[w] += sigma_v
						end
					end
			end

		#	Return Number of Reachable Nodes (q_tail - 1 = stack_order positions used)
			return q_tail - 1
	end

#	Helper Function for betweenness_centrality: Brandes Single-Source Pass
	function _brandes_pass!(neighbor_starts::Vector{Int},
	                       neighbor_data::Vector{Int},
	                       in_neighbor_starts::Vector{Int},
	                       in_neighbor_data::Vector{Int},
	                       source::Int,
	                       betweenness::Vector{Float64},
	                       distance::Vector{Int},
	                       sigma::Vector{Float64},
	                       stack_order::Vector{Int},
	                       delta::Vector{Float64})
		"""
		Args:
			neighbor_starts::Vector{Int}: forward CSR row pointers
			neighbor_data::Vector{Int}: forward CSR neighbor IDs
			in_neighbor_starts::Vector{Int}: reverse CSR row pointers (predecessors)
			in_neighbor_data::Vector{Int}: reverse CSR neighbor IDs
			source::Int: source node ID (1-indexed)
			betweenness::Vector{Float64}: per-thread accumulator (added to in place)
			distance, sigma, stack_order, delta: pre-allocated work buffers
		Returns:
			Nothing (betweenness accumulator is updated in place)
		Notes:
			Single-source Brandes pass on a binarized adjacency. Adds the
			source's contribution to the betweenness accumulator.

			Implements the dependency accumulation:
				delta[v] = sum over w in successors(v) of
				           (sigma[v]/sigma[w]) * (1 + delta[w])
			processed in reverse BFS order. Then betweenness[w] += delta[w]
			for all w != source.

			Requires both forward and reverse CSR representations. Forward is
			used to discover shortest paths; reverse is used to identify
			predecessors during the backward accumulation. For undirected
			(symmetrized) graphs the two representations are identical and
			a single one can be passed for both.

			This is the standard Brandes (2001) algorithm with the
			classical correction that, for any predecessor v of w in the
			shortest-path DAG (i.e., distance[v] + 1 == distance[w] and
			v is in the in-neighbors of w), the contribution to delta[v]
			is (sigma[v]/sigma[w]) * (1 + delta[w]).
		"""

		#	Forward BFS
			n_reached = _bfs_distances_and_sigma!(neighbor_starts, neighbor_data,
			                                     source, distance, sigma, stack_order)

		#	Reset Delta
			n = length(delta)
			fill!(delta, 0.0)

		#	Backward Accumulation in Reverse BFS Order
			@inbounds for idx in n_reached:-1:1
				w = stack_order[idx]
				if w == source
					continue
				end
				#	Iterate over Predecessors of w (Nodes v with v → w and d[v] + 1 == d[w])
					d_w = distance[w]
					sigma_w = sigma[w]
					coef = (1.0 + delta[w]) / sigma_w
					start_ptr = in_neighbor_starts[w]
					end_ptr   = in_neighbor_starts[w + 1] - 1
					for ptr in start_ptr:end_ptr
						v = in_neighbor_data[ptr]
						if distance[v] == d_w - 1
							delta[v] += sigma[v] * coef
						end
					end
				#	Accumulate into Result
					betweenness[w] += delta[w]
			end

		#	Return (Mutates betweenness)
			return nothing
	end

#	Helper Function for betweenness_centrality: Weighted Brandes Single-Source Pass
	function _brandes_pass_weighted!(neighbor_starts::Vector{Int},
									neighbor_data::Vector{Int},
									neighbor_costs::Vector{Float64},
									in_neighbor_starts::Vector{Int},
									in_neighbor_data::Vector{Int},
									in_neighbor_costs::Vector{Float64},
									source::Int,
									betweenness::Vector{Float64},
									distance::Vector{Float64},
									sigma::Vector{Float64},
									stack_order::Vector{Int},
									delta::Vector{Float64},
									heap::Vector{Tuple{Float64, Int}},
									finalized::Vector{Bool})
		"""
		Args:
			neighbor_starts, neighbor_data, neighbor_costs: forward weighted CSR
				(row pointers, neighbor IDs, parallel Dijkstra costs). Used for
				the forward shortest-path pass.
			in_neighbor_starts, in_neighbor_data, in_neighbor_costs: reverse
				weighted CSR. in_neighbor_costs[ptr] is the cost of the edge
				FROM in_neighbor_data[ptr] TO the node whose in-list is being
				scanned. Used to identify shortest-path predecessors in the
				backward accumulation.
			source::Int: source node ID (1-indexed)
			betweenness::Vector{Float64}: per-thread accumulator (added to in place)
			distance, sigma, stack_order, delta: pre-allocated work buffers
			heap::Vector{Tuple{Float64,Int}}: pre-allocated Dijkstra heap buffer
			finalized::Vector{Bool}: pre-allocated Dijkstra finalization buffer
		Returns:
			Nothing (betweenness accumulator is updated in place)
		Notes:
			Weighted analog of _brandes_pass!. Uses Dijkstra
			(_dijkstra_distances_and_sigma!) for the forward shortest-path
			pass instead of BFS, then performs the standard Brandes
			dependency accumulation in reverse finalization order.

			Predecessor identification. In binary Brandes a node v is a
			shortest-path predecessor of w iff distance[v] + 1 == distance[w].
			In the weighted case the analogous test is
				distance[v] + cost(v -> w) ≈ distance[w]
			evaluated with a relative tolerance, since accumulated path costs
			are subject to floating-point rounding. The cost of the edge
			v -> w is read from the reverse weighted CSR (in_neighbor_costs),
			which stores, for each in-neighbor v of w, the weight of v -> w.

			Inf guard. If distance[v] is Inf (v unreachable from source), the
			predecessor sum distance[v] + cost is Inf and cannot match the
			finite distance[w]; the explicit finite check skips it and avoids
			Inf - Inf = NaN in the tolerance arithmetic, mirroring the guard
			in _dijkstra_distances_and_sigma!.

			Dependency accumulation (unchanged in form from binary Brandes):
				delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
			for each predecessor v of w, processed with w in reverse
			finalization (stack_order) order. Then betweenness[w] += delta[w]
			for all w != source.

			For undirected (symmetrized) graphs the forward and reverse
			weighted CSRs are identical and the same triple may be passed for
			both. The caller applies the directed/undirected halving
			convention after reduction, exactly as in the binary path.
		"""

		#	Tolerance for the Predecessor Relation
			_BRANDES_REL_EPS = 1e-12

		#	Forward Dijkstra (Shared Kernel): Fills distance, sigma, stack_order
			n_reached = _dijkstra_distances_and_sigma!(neighbor_starts,
														neighbor_data,
														neighbor_costs,
														source,
														distance,
														sigma,
														stack_order,
														heap,
														finalized)

		#	Reset Delta
			fill!(delta, 0.0)

		#	Backward Accumulation in Reverse Finalization Order
			@inbounds for idx in n_reached:-1:1
				w = stack_order[idx]
				if w == source
					continue
				end

				#	Coefficient (1 + delta[w]) / sigma[w], Shared Across Predecessors
					sigma_w = sigma[w]
					coef    = (1.0 + delta[w]) / sigma_w
					d_w     = distance[w]

				#	Scan In-Neighbors v of w; Keep Those on a Shortest Path to w
					start_ptr = in_neighbor_starts[w]
					end_ptr   = in_neighbor_starts[w + 1] - 1
					for ptr in start_ptr:end_ptr
						v    = in_neighbor_data[ptr]
						cost = in_neighbor_costs[ptr]

						#	Skip Inf-Cost Edges (Treated as Absent)
							if !isfinite(cost)
								continue
							end

						#	Predecessor Test: distance[v] + cost(v->w) ≈ distance[w]
						#	Skip unreachable v (distance Inf) to avoid Inf - Inf = NaN.
							d_v = distance[v]
							if !isfinite(d_v)
								continue
							end
							pred_dist = d_v + cost
							scale     = max(abs(pred_dist), abs(d_w), 1.0)
							tol       = _BRANDES_REL_EPS * scale
							if abs(pred_dist - d_w) <= tol
								#	v Is a Shortest-Path Predecessor of w
									delta[v] += sigma[v] * coef
							end
					end

				#	Accumulate Dependency into Betweenness
					betweenness[w] += delta[w]
			end

		#	Return (Mutates betweenness)
			return nothing
	end

#	Helper Function for closeness_centrality and mean_inverse_distance:
#	Compute Per-Source Inverse-Distance Sums and Total
	function _all_pairs_inverse_distance_sum(adj::SparseMatrixCSC{<:Real, <:Integer})
		"""
		Args:
			adj::SparseMatrixCSC: binary sparse adjacency matrix
		Returns:
			Tuple{Vector{Float64}, Float64}: (per_source_inv_dist_sum, total_inv_dist_sum)
				per_source_inv_dist_sum[i] = sum_{j != i} 1/d(i, j)
				total_inv_dist_sum         = sum over all ordered pairs of 1/d(i, j)
		Notes:
			Single all-pairs BFS pass producing both the per-source sum
			(for closeness) and the global total (for mean inverse distance).
			Unreachable pairs contribute 0.

			Threaded over source nodes. Each thread maintains its own work
			buffers; results are combined into the final vectors.

			Treats `adj` as binary — nonzero entries are edges, weights
			ignored. Caller is responsible for binarizing first if desired.

			For directed graphs, the per-source sum reflects the out-direction
			(d(i, j) = directed shortest path from i to j). For symmetrized
			adjacency the per-source sum is the same as in-direction.
		"""

		#	Dimensions
			n = size(adj, 1)

		#	Build CSR Neighbor Lists (Forward Direction)
			neighbor_starts, neighbor_data = _build_neighbor_lists(adj)

		#	Per-Source Output and Thread-Local Total Accumulators
			per_source_sum = zeros(Float64, n)
			n_threads = Threads.nthreads()
			thread_totals = zeros(Float64, n_threads)

		#	Threaded Loop Over Sources
			Threads.@threads for s in 1:n
				#	Per-Thread Work Buffers
					tid = Threads.threadid()
					distance    = Vector{Int}(undef, n)
					sigma       = Vector{Float64}(undef, n)
					stack_order = Vector{Int}(undef, n)

				#	Single-Source BFS
					n_reached = _bfs_distances_and_sigma!(neighbor_starts,
					                                     neighbor_data,
					                                     s,
					                                     distance,
					                                     sigma,
					                                     stack_order)

				#	Sum 1/d Over Reachable Non-Source Nodes
					local_sum = 0.0
					@inbounds for idx in 2:n_reached
						v = stack_order[idx]
						d = distance[v]
						local_sum += 1.0 / d
					end

				#	Write per-Source Result and Update Thread Total
					per_source_sum[s] = local_sum
					thread_totals[tid] += local_sum
			end

		#	Combine Thread Totals
			total_sum = sum(thread_totals)

		#	Return Both
			return (per_source_sum, total_sum)
	end

#	Helper Function for Weighted closeness_centrality and mean_inverse_distance:
#	Compute Per-Source Inverse-Distance Sums and Total (Dijkstra)
	function _all_pairs_inverse_distance_sum_weighted(adj::SparseMatrixCSC{<:Real, <:Integer},
														edge_interpretation::Symbol)
		"""
		Args:
			adj::SparseMatrixCSC: weighted sparse adjacency matrix
			edge_interpretation::Symbol: :tie_strength or :distance
		Returns:
			Tuple{Vector{Float64}, Float64}: (per_source_inv_dist_sum, total_inv_dist_sum)
				per_source_inv_dist_sum[i] = sum_{j != i} 1/d(i, j)
				total_inv_dist_sum         = sum over all ordered pairs of 1/d(i, j)
		Notes:
			Weighted analog of _all_pairs_inverse_distance_sum. Single
			all-pairs Dijkstra pass producing both the per-source sum
			(for closeness) and the global total (for mean inverse distance).
			Unreachable pairs contribute 0.

			Threaded over source nodes. Each source iteration allocates its
			own work buffers (distance, sigma, stack_order, heap, finalized)
			and passes them to _dijkstra_distances_and_sigma!. The heap is a
			fixed-capacity buffer (length(neighbor_data) + 1) used via an
			explicit size counter, never grown. The weighted CSR conversion
			and cost transform happen once before the threaded loop and are
			shared (read-only) across threads.

			For directed graphs, the per-source sum reflects the out-direction
			(d(i, j) = directed weighted shortest path from i to j). The
			caller is responsible for applying the desired direction
			convention (symmetrize, transpose, etc.) before passing adj.
		"""

		#	Dimensions
			n = size(adj, 1)

		#	Build Weighted CSR Once (Shared, Read-Only Across Threads)
			neighbor_starts, neighbor_data, neighbor_weights = _build_weighted_neighbor_lists(adj)

		#	Transform Weights to Dijkstra Costs (Shared, Read-Only)
			neighbor_costs = _transform_weights_for_path_cost(neighbor_weights, edge_interpretation)

		#	Heap Capacity: One Entry per Finite-Cost Relaxation, Bounded by Edge Count
		#	The +1 covers the initial source seed. Sufficient for the lazy-deletion
		#	heap, which inserts at most once per strictly-improving relaxation.
			heap_capacity = length(neighbor_data) + 1

		#	Per-Source Output and Thread-Local Total Accumulators
			per_source_sum = zeros(Float64, n)
			n_threads      = Threads.nthreads()
			thread_totals  = zeros(Float64, n_threads)

		#	Threaded Loop Over Sources
			Threads.@threads for s in 1:n
				#	Per-Thread Work Buffers (Allocated per Source Iteration)
				#	Allocate-per-iteration rather than threadid()-indexed pools,
				#	since Threads.threadid() is not stable across yields. Matches
				#	the buffer-allocation pattern of _all_pairs_inverse_distance_sum.
					tid         = Threads.threadid()
					distance    = Vector{Float64}(undef, n)
					sigma       = Vector{Float64}(undef, n)
					stack_order = Vector{Int}(undef, n)
					heap        = Vector{Tuple{Float64, Int}}(undef, heap_capacity)
					finalized   = Vector{Bool}(undef, n)

				#	Single-Source Dijkstra
					n_reached = _dijkstra_distances_and_sigma!(neighbor_starts,
																neighbor_data,
																neighbor_costs,
																s,
																distance,
																sigma,
																stack_order,
																heap,
																finalized)

				#	Sum 1/d Over Reachable Non-Source Nodes
					local_sum = 0.0
					@inbounds for idx in 2:n_reached
						v = stack_order[idx]
						d = distance[v]
						if isfinite(d) && d > 0.0
							local_sum += 1.0 / d
						end
					end

				#	Write per-Source Result and Update Thread Total
					per_source_sum[s]   = local_sum
					thread_totals[tid] += local_sum
			end

		#	Combine Thread Totals
			total_sum = sum(thread_totals)

		#	Return Both
			return (per_source_sum, total_sum)
	end

#	Closeness Centrality (SMM Inverse-Distance Form, Binary or Weighted)
	function closeness_centrality(edges::DataFrame;
									nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
									weighted::Bool = false,
									directed::Bool = true,
									direction::Symbol = :symmetric,
									edge_interpretation::Symbol = :tie_strength,
									normalize::Bool = true,
									agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: API symmetry; ignored — actual weight handling
				is determined by edge_interpretation
			directed::Bool: treat graph as directed (default = true)
			direction::Symbol: :out | :in | :symmetric (default = :symmetric)
			edge_interpretation::Symbol: how to interpret edge weights
				:tie_strength (default) — weights are intensity; cost = 1/w
					(Dijkstra on inverted weights)
				:distance     — weights are distances; cost = w
					(Dijkstra on raw weights)
				:ignore       — binarize before path computation (BFS)
			normalize::Bool: divide by N-1 (default = true)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			DataFrame: columns [node, closeness]
		Notes:
			Smith, Morgan, & Moody (2022) inverse-distance closeness:
				C_C(i) = (1/(N-1)) * sum_{j != i} 1/d(i, j)
			with 1/inf = 0 for unreachable pairs and 1/0 = 0 for self-pairs.

			Three edge-weight semantics:
				:tie_strength (default) — weights interpreted as intensity
					or frequency of the tie. Each edge cost is 1/w, so
					stronger ties produce shorter weighted distances.
					Standard for co-appearance, interaction, shared-attribute
					networks (Marvel, Balikatan, Scotland, Moreno).
				:distance — weights interpreted as costs or geographic
					distances. Used as-is. Standard for transit and road
					networks.
				:ignore — binarize the graph before computing paths. Matches
					Smith, Morgan, & Moody (2022) Table 1 convention. Used
					for cross-binarization comparability with SMM and for
					networks where weight semantics are unclear.

			The default is :tie_strength because most networks in this
			corpus encode weights as tie strength. To match SMM 2022's
			binarized Table 1 convention exactly, pass
			edge_interpretation=:ignore.

			Performance:
				:ignore — BFS, O(N + E) per source, fastest path
				:tie_strength / :distance — Dijkstra, O((N + E) log N)
					per source, ~2-3x slower than BFS on the same graph

			For directed graphs:
				:out       — d(i, j) is the directed shortest path i → j
				:in        — d(i, j) is the directed shortest path j → i
				:symmetric — compute on max(A, A^T), the symmetrized adjacency
					(this matches how Bonacich is computed; consistent with
					the SMM Phase 0 convention)

			For undirected graphs the `direction` argument is ignored; the
			adjacency is symmetrized via max(A, A^T) since GraphML storage
			may use asymmetric edge lists even for conceptually undirected
			networks.

			For weighted symmetric mode, max(A, A^T) takes the higher of
			the two directed weights on a mutual pair. This is consistent
			with the binary case where max(A, A^T) yields the union of
			edges.

			Pass `nodes` to include isolates (closeness = 0) in the output.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !(direction in (:out, :in, :symmetric))
				throw(ArgumentError("direction must be :out, :in, or :symmetric"))
			end
			if !(edge_interpretation in (:ignore, :tie_strength, :distance))
				throw(ArgumentError(
					"closeness_centrality: edge_interpretation=$(edge_interpretation) not supported. " *
					"Use :ignore (binary), :tie_strength (weights as intensity), or :distance (weights as cost)."
				))
			end

		#	Edge Interpretation Requires Weight Column (Unless :ignore)
			if edge_interpretation !== :ignore && !hasproperty(edges, :weight)
				throw(ArgumentError(
					"closeness_centrality: edge_interpretation=$(edge_interpretation) requires a :weight column on edges. " *
					"Either add weights or pass edge_interpretation=:ignore for binary computation."
				))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					if nodes isa DataFrame
						return DataFrame(node = nodes.id, closeness = zeros(Float64, nrow(nodes)))
					else
						return DataFrame(node = collect(nodes), closeness = zeros(Float64, length(nodes)))
					end
				else
					return DataFrame(node = [], closeness = Float64[])
				end
			end

		#	Prepare Edges and Build Sparse Adjacency
		#	For :ignore, binarize. For :tie_strength and :distance, preserve weights.
			if edge_interpretation === :ignore
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst)
				edges_for_adj.weight = ones(Float64, nrow(edges_for_adj))
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = maximum)
				build_weighted = false
			else
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst, weight = edges.weight)
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = agg_func)
				build_weighted = true
			end

			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(edges_for_adj; weighted = build_weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(edges_for_adj;
																nodes    = nodes,
																weighted = build_weighted)
			end

		#	Apply Direction Convention
			if !directed
				adj = max.(adj, adj')
			else
				if direction == :out
					#	Use Adjacency As-Is
				elseif direction == :in
					adj = sparse(adj')
				else  # :symmetric
					adj = max.(adj, adj')
				end
			end

		#	Compute Per-Source Inverse-Distance Sums (BFS or Dijkstra)
			if edge_interpretation === :ignore
				per_source_sum, _ = _all_pairs_inverse_distance_sum(adj)
			else
				per_source_sum, _ = _all_pairs_inverse_distance_sum_weighted(adj, edge_interpretation)
			end

		#	Apply Normalization
			N = size(adj, 1)
			scale = normalize ? (N > 1 ? 1.0 / (N - 1) : 0.0) : 1.0
			closeness = per_source_sum .* scale

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, closeness = closeness)
	end
	@doc raw"""
	**Description**
	Compute closeness centrality using Smith, Morgan, & Moody's (2022)
	inverse-distance formulation: each node's score is the (normalized) sum
	of the reciprocals of its shortest-path distances to all other nodes,
	with unreachable pairs contributing 0. Supports three weight semantics:
	tie strength (default), distance, or ignored (binary).

	**Usage**
	`closeness_centrality(edges; nodes=nothing, weighted=false, directed=true, direction=:symmetric, edge_interpretation=:tie_strength, normalize=true, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, and (for non-`:ignore`
	  interpretations) `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. Pass
	  to include isolates.
	- `weighted::Bool`: API symmetry only; actual weight handling is
	  controlled by `edge_interpretation`.
	- `directed::Bool`: Treat as directed (default `true`).
	- `direction::Symbol`: For directed graphs: `:out`, `:in`, or
	  `:symmetric` (default `:symmetric`).
	- `edge_interpretation::Symbol`: How to interpret edge weights.
	  - `:tie_strength` (default) — weights are intensity; edge cost = $1/w$
	    (Dijkstra on inverted weights). Standard for co-appearance,
	    interaction-count, and shared-attribute networks.
	  - `:distance` — weights are distances/costs; used as-is (Dijkstra on
	    raw weights). Standard for road and transit networks.
	  - `:ignore` — binarize before computing paths (BFS). Matches the SMM
	    (2022) Table 1 convention; required to reproduce that table's values.
	- `normalize::Bool`: Divide by $N-1$ (default `true`). A node connected
	  at distance 1 to all others scores 1.0; an isolate scores 0.0.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).

	**Details**
	The score is $C_C(i) = \frac{1}{N-1} \sum_{j \neq i} \frac{1}{d(i, j)}$
	with $\frac{1}{\infty} = 0$ for unreachable pairs.

	When `edge_interpretation = :ignore`, distances are integer hop counts
	from BFS on the binarized graph; the result reproduces SMM (2022).
	When `:tie_strength`, edges with weight $w$ contribute cost $1/w$ to
	any path through them — stronger ties give shorter weighted distances.
	When `:distance`, edges contribute their weight directly. Both weighted
	modes use Dijkstra's algorithm; threaded over source nodes.

	The default is `:tie_strength` because most networks in this corpus
	encode weights as tie strength rather than as geographic or cost
	distances. To match the binarized SMM (2022) convention exactly, pass
	`edge_interpretation = :ignore`.

	**Performance**
	`:ignore` uses BFS with cost $O(N + E)$ per source. `:tie_strength` and
	`:distance` use Dijkstra with cost $O((N + E) \log N)$ per source, about
	2–3× slower on the same graph. All three paths are threaded over source
	nodes.

	**Value**
	A `DataFrame` with columns `:node` and `:closeness`.

	**Examples**
	```julia
		using DataFrames

		#	Default: weights interpreted as tie strength
			edges = DataFrame(src=[1,1,2,3], dst=[2,3,3,2], weight=[3.0, 1.0, 2.0, 4.0])
			nodes = DataFrame(id=string.(1:5), label=string.(1:5))
			closeness_centrality(edges; nodes=nodes, directed=true, direction=:symmetric)

		#	SMM-style binarized closeness (matches SMM 2022 Table 1)
			closeness_centrality(edges; nodes=nodes, edge_interpretation=:ignore)

		#	Distance semantics (e.g., road network with travel times)
			closeness_centrality(edges; nodes=nodes, edge_interpretation=:distance)
	```

	**References**
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.
	- Dijkstra EW (1959). "A note on two problems in connexion with graphs."
	  *Numerische Mathematik* 1: 269–271.

	**See Also**
	`betweenness_centrality`, `mean_inverse_distance`
	""" closeness_centrality

#	Betweenness Centrality (Brandes 2001, Binary or Weighted)
	function betweenness_centrality(edges::DataFrame;
									nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
									weighted::Bool = false,
									directed::Bool = true,
									edge_interpretation::Symbol = :tie_strength,
									normalize::Bool = false,
									agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: API symmetry; actual weight handling is determined
				by edge_interpretation
			directed::Bool: treat graph as directed (default = true)
			edge_interpretation::Symbol: how to interpret edge weights
				:tie_strength (default) — weights are intensity; cost = 1/w
					(weighted Brandes via Dijkstra on inverted weights)
				:distance     — weights are distances; cost = w
					(weighted Brandes via Dijkstra on raw weights)
				:ignore       — binarize before path computation (binary Brandes/BFS)
			normalize::Bool: normalize to [0, 1] (default = false; SMM uses raw)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			DataFrame: columns [node, betweenness]
		Notes:
			Brandes (2001) algorithm with thread-parallel single-source passes.
			Each source contributes its dependency-accumulated geodesic counts
			to a thread-local betweenness vector; the per-thread vectors are
			reduced at the end.

			Three edge-weight semantics:
				:tie_strength (default) — weights interpreted as intensity;
					edge cost = 1/w. Shortest paths are weighted geodesics, so
					betweenness counts weighted-shortest-path traversals.
					Standard for co-appearance, interaction, shared-attribute
					networks (Marvel, Balikatan, Scotland, Moreno).
				:distance — weights interpreted as costs or geographic
					distances; edge cost = w. Standard for transit and road
					networks.
				:ignore — binarize the graph before computing geodesics, which
					become hop-count shortest paths. Matches the Smith, Morgan,
					& Moody (2022) Table 1 convention.

			The default is :tie_strength for consistency with
			closeness_centrality and mean_inverse_distance, and because most
			networks in this corpus encode weights as tie strength. To match
			the binarized SMM (2022) convention exactly, pass
			edge_interpretation=:ignore.

			For directed graphs, betweenness is the number of geodesics
			s -> v -> t passing through v, summed over ordered (s, t) pairs
			with s != v != t. For undirected graphs, the adjacency is first
			symmetrized via max(A, A^T) so that asymmetric storage of
			conceptually undirected edges is handled correctly; the standard
			directed-Brandes-then-halve convention is then applied (each
			unordered pair would otherwise be counted twice).

			Implementation. The :ignore path uses BFS-based Brandes
			(_brandes_pass!) on the binarized adjacency. The weighted paths
			use Dijkstra-based Brandes (_brandes_pass_weighted!): the forward
			pass is the shared _dijkstra_distances_and_sigma! kernel, and the
			backward dependency accumulation identifies shortest-path
			predecessors via distance[v] + cost(v -> w) ≈ distance[w] using a
			relative tolerance with an Inf guard (an unreachable predecessor
			with distance Inf is skipped to avoid Inf - Inf = NaN). The cost
			of edge v -> w is read from a reverse weighted CSR built from
			sparse(adj').

			For weighted symmetric/undirected mode, max(A, A^T) takes the
			higher of the two directed weights on a mutual pair, consistent
			with the closeness_centrality convention.

			Normalization (when normalize=true):
				directed:   B(v) / ((N-1)(N-2))
				undirected: B(v) / ((N-1)(N-2)/2)
			These factors are the maximum possible betweenness on a star
			graph for the respective network type.

			Performance:
				:ignore — BFS Brandes, O(N(N+E)) total
				:tie_strength / :distance — Dijkstra Brandes,
					O(N(N+E) log N) total
			All paths are thread-parallel over source nodes.

			Endpoints are excluded from their own paths per the Freeman
			convention. Pass `nodes` to include isolates (betweenness = 0).
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !(edge_interpretation in (:ignore, :tie_strength, :distance))
				throw(ArgumentError(
					"betweenness_centrality: edge_interpretation=$(edge_interpretation) not supported. " *
					"Use :ignore (binary), :tie_strength (weights as intensity), or :distance (weights as cost)."
				))
			end
			if edge_interpretation !== :ignore && !hasproperty(edges, :weight)
				throw(ArgumentError(
					"betweenness_centrality: edge_interpretation=$(edge_interpretation) requires a :weight column on edges. " *
					"Either add weights or pass edge_interpretation=:ignore for binary computation."
				))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					if nodes isa DataFrame
						return DataFrame(node = nodes.id, betweenness = zeros(Float64, nrow(nodes)))
					else
						return DataFrame(node = collect(nodes), betweenness = zeros(Float64, length(nodes)))
					end
				else
					return DataFrame(node = [], betweenness = Float64[])
				end
			end

		#	Prepare Edges and Build Sparse Adjacency
		#	For :ignore, binarize. For :tie_strength and :distance, preserve weights.
			if edge_interpretation === :ignore
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst)
				edges_for_adj.weight = ones(Float64, nrow(edges_for_adj))
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = maximum)
				build_weighted = false
			else
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst, weight = edges.weight)
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = agg_func)
				build_weighted = true
			end

			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(edges_for_adj; weighted = build_weighted)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(edges_for_adj;
																nodes    = nodes,
																weighted = build_weighted)
			end

		#	Apply Direction Convention (Symmetrize for Undirected)
			if !directed
				adj = max.(adj, adj')
			end

		#	Dimensions
			N = size(adj, 1)

		#	Thread-Local Betweenness Accumulators
			n_threads          = Threads.nthreads()
			thread_betweenness = [zeros(Float64, N) for _ in 1:n_threads]

		#	Dispatch on edge_interpretation
			if edge_interpretation === :ignore
				#	Binary Brandes (BFS-Based)
				#	Build Forward and Reverse (Binary) CSR Representations
					fwd_starts, fwd_data = _build_neighbor_lists(adj)
					rev_starts, rev_data = _build_neighbor_lists(sparse(adj'))

				#	Threaded Single-Source Passes
					Threads.@threads for s in 1:N
						tid         = Threads.threadid()
						distance    = Vector{Int}(undef, N)
						sigma       = Vector{Float64}(undef, N)
						stack_order = Vector{Int}(undef, N)
						delta       = Vector{Float64}(undef, N)
						_brandes_pass!(fwd_starts, fwd_data,
									  rev_starts, rev_data,
									  s,
									  thread_betweenness[tid],
									  distance, sigma, stack_order, delta)
					end
			else
				#	Weighted Brandes (Dijkstra-Based)
				#	Build Forward and Reverse WEIGHTED CSR Representations, Then
				#	Transform Weights to Dijkstra Costs (Once, Shared Across Threads).
					fwd_starts, fwd_data, fwd_weights = _build_weighted_neighbor_lists(adj)
					rev_starts, rev_data, rev_weights = _build_weighted_neighbor_lists(sparse(adj'))
					fwd_costs = _transform_weights_for_path_cost(fwd_weights, edge_interpretation)
					rev_costs = _transform_weights_for_path_cost(rev_weights, edge_interpretation)

				#	Heap Capacity (Bounded by Forward Edge Count, +1 for the Seed)
					heap_capacity = length(fwd_data) + 1

				#	Threaded Single-Source Passes
					Threads.@threads for s in 1:N
						tid         = Threads.threadid()
						distance    = Vector{Float64}(undef, N)
						sigma       = Vector{Float64}(undef, N)
						stack_order = Vector{Int}(undef, N)
						delta       = Vector{Float64}(undef, N)
						heap        = Vector{Tuple{Float64, Int}}(undef, heap_capacity)
						finalized   = Vector{Bool}(undef, N)
						_brandes_pass_weighted!(fwd_starts, fwd_data, fwd_costs,
											   rev_starts, rev_data, rev_costs,
											   s,
											   thread_betweenness[tid],
											   distance, sigma, stack_order, delta,
											   heap, finalized)
					end
			end

		#	Reduce Thread-Local Accumulators
			betweenness = zeros(Float64, N)
			for tb in thread_betweenness
				betweenness .+= tb
			end

		#	Undirected Correction (Each Unordered Pair Counted Twice in the DAG)
			if !directed
				betweenness ./= 2.0
			end

		#	Normalization
			if normalize
				if directed
					denom = (N - 1) * (N - 2)
				else
					denom = (N - 1) * (N - 2) / 2.0
				end
				if denom > 0
					betweenness ./= denom
				end
			end

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col, betweenness = betweenness)
	end
	@doc raw"""
	**Description**
	Compute betweenness centrality for each node using Brandes' (2001)
	algorithm. The betweenness of node $v$ is the number of shortest paths
	between other node pairs that pass through $v$. Supports three weight
	semantics: tie strength (default), distance, or ignored (binary).

	**Usage**
	`betweenness_centrality(edges; nodes=nothing, weighted=false, directed=true, edge_interpretation=:tie_strength, normalize=false, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, and (for non-`:ignore`
	  interpretations) `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe; pass to
	  include isolates (`betweenness = 0`).
	- `weighted::Bool`: API symmetry only; actual weight handling is controlled
	  by `edge_interpretation`.
	- `directed::Bool`: Treat as directed (default `true`).
	- `edge_interpretation::Symbol`: How to interpret edge weights.
	  - `:tie_strength` (default) — weights are intensity; edge cost = $1/w$
	    (weighted Brandes via Dijkstra on inverted weights). Standard for
	    co-appearance, interaction-count, and shared-attribute networks.
	  - `:distance` — weights are distances/costs; used as-is (weighted Brandes
	    via Dijkstra on raw weights). Standard for road and transit networks.
	  - `:ignore` — binarize before computing geodesics (binary Brandes).
	    Matches the SMM (2022) Table 1 convention; required to reproduce that
	    table's values.
	- `normalize::Bool`: Normalize to $[0, 1]$ by dividing by $(N-1)(N-2)$
	  (directed) or $(N-1)(N-2)/2$ (undirected). Default `false` to match
	  SMM (2022), which reports raw geodesic counts.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).

	**Details**
	Standard definition:
	$$B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
	where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$ and
	$\sigma_{st}(v)$ is the number of those passing through $v$. Endpoints are
	excluded per the Freeman convention.

	When `edge_interpretation = :ignore`, geodesics are hop-count shortest
	paths found by BFS-based Brandes, reproducing SMM (2022). When
	`:tie_strength`, each edge of weight $w$ contributes cost $1/w$, so
	betweenness counts traversals of weighted-shortest paths in which strong
	ties act as short links. When `:distance`, edges contribute their weight
	directly. Both weighted modes use a Dijkstra forward pass with
	shortest-path counting, followed by the standard backward dependency
	accumulation; shortest-path predecessors are identified by
	$d(s,v) + c(v,w) \approx d(s,w)$ under a relative tolerance, with
	unreachable predecessors skipped to avoid degenerate comparisons.

	Implemented with thread-parallel single-source passes; each thread
	maintains its own betweenness accumulator and the per-thread vectors are
	summed at the end. Complexity is $O(N(N+E))$ for the binary path and
	$O(N(N+E)\log N)$ for the weighted paths.

	The default is `:tie_strength` for consistency with `closeness_centrality`
	and `mean_inverse_distance`. To match the binarized SMM (2022) convention
	exactly, pass `edge_interpretation = :ignore`.

	For undirected networks the directed implementation counts each unordered
	pair twice, so the result is divided by 2 at the end (the standard
	convention).

	**Value**
	A `DataFrame` with columns `:node` and `:betweenness`.

	**Examples**
	```julia
		using DataFrames

		#	Default: weights interpreted as tie strength
			edges = DataFrame(src=["1","1","2","3"], dst=["2","3","4","4"],
							weight=[4.0, 1.0, 4.0, 1.0])
			nodes = DataFrame(id=string.(1:4), label=string.(1:4))
			betweenness_centrality(edges; nodes=nodes, directed=true)

		#	SMM-style binarized betweenness (matches SMM 2022 Table 1)
			betweenness_centrality(edges; nodes=nodes, edge_interpretation=:ignore)

		#	Distance semantics (e.g., road network with travel times)
			betweenness_centrality(edges; nodes=nodes, edge_interpretation=:distance)
	```

	**References**
	- Brandes U (2001). "A faster algorithm for betweenness centrality."
	  *Journal of Mathematical Sociology* 25(2): 163–177.
	- Brandes U (2008). "On variants of shortest-path betweenness centrality
	  and their generic computation." *Social Networks* 30(2): 136–145.
	- Freeman LC (1977). "A set of measures of centrality based on betweenness."
	  *Sociometry* 40(1): 35–41.

	**See Also**
	`closeness_centrality`, `mean_inverse_distance`
	""" betweenness_centrality

#	Mean Inverse Distance, Optionally Scaled by Log N (Binary or Weighted)
	function mean_inverse_distance(edges::DataFrame;
									nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
									weighted::Bool = false,
									directed::Bool = true,
									direction::Symbol = :symmetric,
									edge_interpretation::Symbol = :tie_strength,
									scale_by_log_n::Bool = true,
									agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: API symmetry; ignored
			directed::Bool: treat graph as directed (default = true)
			direction::Symbol: :out | :in | :symmetric (default = :symmetric)
			edge_interpretation::Symbol: how to interpret edge weights
				:tie_strength (default) — weights as intensity, cost = 1/w (Dijkstra)
				:distance     — weights as cost, used as-is (Dijkstra)
				:ignore       — binarize before path computation (BFS)
			scale_by_log_n::Bool: divide by log(N) per SMM footnote 6 (default = true)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			Float64: mean inverse distance, optionally scaled
		Notes:
			Smith, Morgan, & Moody (2022) third topology measure:
				MID = (1 / (N(N-1))) * sum_{i != j} 1/d(i, j)
			with 1/inf = 0 for unreachable pairs.

			The edge_interpretation argument controls how edge weights enter
			into d(i, j):
				:tie_strength — d(i, j) is the weighted shortest path under
					cost = 1/w. Stronger ties give shorter distances.
				:distance — d(i, j) is the weighted shortest path under
					cost = w.
				:ignore — d(i, j) is the unweighted hop count via BFS;
					matches the SMM (2022) Table 1 convention.

			The default is :tie_strength for consistency with
			closeness_centrality. To match SMM 2022 exactly, pass
			edge_interpretation=:ignore.

			When scale_by_log_n=true, divides by log(N) per SMM footnote 6.
			The rationale (larger networks have proportionally more
			disconnected pairs, pulling unscaled MID down mechanically) was
			developed for binary graphs; the same intuition applies to
			weighted graphs but the scaling is more aesthetic than principled
			under weighted interpretations.

			Returns 0.0 if N < 2.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !(direction in (:out, :in, :symmetric))
				throw(ArgumentError("direction must be :out, :in, or :symmetric"))
			end
			if !(edge_interpretation in (:ignore, :tie_strength, :distance))
				throw(ArgumentError(
					"mean_inverse_distance: edge_interpretation=$(edge_interpretation) not supported. " *
					"Use :ignore (binary), :tie_strength (weights as intensity), or :distance (weights as cost)."
				))
			end
			if edge_interpretation !== :ignore && !hasproperty(edges, :weight)
				throw(ArgumentError(
					"mean_inverse_distance: edge_interpretation=$(edge_interpretation) requires a :weight column on edges."
				))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return 0.0
			end

		#	Prepare Edges and Build Sparse Adjacency
			if edge_interpretation === :ignore
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst)
				edges_for_adj.weight = ones(Float64, nrow(edges_for_adj))
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = maximum)
				build_weighted = false
			else
				edges_for_adj = DataFrame(src = edges.src, dst = edges.dst, weight = edges.weight)
				edges_for_adj = _aggregate_multi_edges(edges_for_adj; agg_func = agg_func)
				build_weighted = true
			end

			if nodes === nothing
				adj, _, _ = _edgelist_to_sparse_matrix(edges_for_adj; weighted = build_weighted)
			else
				adj, _, _ = _graph_to_sparse_matrix(edges_for_adj;
													nodes    = nodes,
													weighted = build_weighted)
			end

		#	Apply Direction Convention
			if !directed
				if !_is_symmetric(adj; directed = false)
					adj = max.(adj, adj')
				end
			else
				if direction == :in
					adj = sparse(adj')
				elseif direction == :symmetric
					adj = max.(adj, adj')
				end
			end

		#	Get Total Inverse-Distance Sum (BFS or Dijkstra)
			if edge_interpretation === :ignore
				_, total_inv_dist = _all_pairs_inverse_distance_sum(adj)
			else
				_, total_inv_dist = _all_pairs_inverse_distance_sum_weighted(adj, edge_interpretation)
			end

		#	Dimensions
			N = size(adj, 1)
			if N < 2
				return 0.0
			end

		#	Divide by Pair Count
			mid = total_inv_dist / (N * (N - 1))

		#	Scale by Log N
			if scale_by_log_n
				mid /= log(N)
			end

		#	Return
			return mid
	end
	@doc raw"""
	**Description**
	Compute the mean inverse distance between node pairs in the network,
	optionally scaled by $\log N$ to mitigate the mechanical decline of the
	raw measure with network size (per Smith, Morgan, & Moody 2022, footnote 6).
	Supports binary (BFS), tie-strength-weighted (Dijkstra on $1/w$), or
	distance-weighted (Dijkstra on $w$) path computation.

	**Usage**
	`mean_inverse_distance(edges; nodes=nothing, weighted=false, directed=true, direction=:symmetric, edge_interpretation=:tie_strength, scale_by_log_n=true, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, and (for non-`:ignore`
	  interpretations) `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe.
	- `weighted::Bool`: API symmetry; ignored.
	- `directed::Bool`: Treat as directed (default `true`).
	- `direction::Symbol`: `:out`, `:in`, or `:symmetric` (default `:symmetric`).
	- `edge_interpretation::Symbol`: How to interpret edge weights:
	  - `:tie_strength` (default) — Dijkstra on $1/w$, stronger ties give
	    shorter paths.
	  - `:distance` — Dijkstra on $w$, used as-is.
	  - `:ignore` — BFS on binarized graph. Matches SMM (2022) Table 1.
	- `scale_by_log_n::Bool`: Divide by $\log N$ (default `true`).
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).

	**Details**
	Defined as
	$$\overline{d^{-1}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d(i, j)}$$
	with $1/\infty = 0$ for unreachable pairs. The choice of $d(i,j)$ is
	controlled by `edge_interpretation`: BFS hop count for `:ignore`,
	weighted Dijkstra distance for `:tie_strength` and `:distance`.

	The default `:tie_strength` matches `closeness_centrality` and reflects
	the dominant weight semantics in this corpus. To match SMM (2022)
	exactly, pass `:ignore`.

	**Value**
	A `Float64` summary statistic for the whole network.

	**Examples**
```julia
	using DataFrames

	#	Default: weights as tie strength
		edges = DataFrame(src=[1,1,2,3], dst=[2,3,3,2], weight=[3.0, 1.0, 2.0, 4.0])
		mean_inverse_distance(edges; directed=true, direction=:symmetric)

	#	SMM-compatible binarized form
		mean_inverse_distance(edges; edge_interpretation=:ignore)
```

	**References**
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.

	**See Also**
	`closeness_centrality`, `betweenness_centrality`
	""" mean_inverse_distance

######################################
#   SECTION 4: BONACICH CENTRALITY   #
######################################

# ====================================================================
# Bonacich (1987) power centrality:
#
#     c(β) = α (I - β A)^{-1} A · 1
#
# A node's centrality is its (weighted) sum of neighbors' centralities,
# attenuated by β. When β > 0, well-connected neighbors raise a node's
# score (degree-weighted); when β < 0, structurally-isolated neighbors
# raise it (structural-holes). When β = 0, c reduces to scaled degree.
#
# For directed networks, Smith/Morgan/Moody (2022) symmetrize the
# adjacency before computing Bonacich (via max(A, A^T)). This module
# follows that convention with `direction=:symmetric` as the default;
# `direction=:out` or `:in` provides asymmetric forms on request.
#
# Convergence of (I - β A)^{-1} requires |β| < 1/λ_max(A). The default
# β = 0.5 / λ_max provides a comfortable safety margin in the standard
# positive-power regime. λ_max is computed via power iteration on the
# symmetrized adjacency, with no extra package dependencies.
#
# Internal helpers:
#   _power_iteration_lambda_max — power iteration on a sparse symmetric
#                                  matrix; returns the largest eigenvalue
#
# Public API:
#   bonacich_centrality          — Bonacich (1987) power centrality
# ====================================================================

#	Helper Function for bonacich_centrality: Power Iteration for λ_max
	function _power_iteration_lambda_max(adj::SparseMatrixCSC{<:Real, <:Integer};
	                                    max_iter::Int = 200,
	                                    tol::Float64 = 1e-10,
	                                    seed::Int = 20260101)
		"""
		Args:
			adj::SparseMatrixCSC: square sparse adjacency matrix
			max_iter::Int: maximum number of iterations (default = 200)
			tol::Float64: convergence tolerance on ||x_new - x|| (default = 1e-10)
			seed::Int: RNG seed for initial random vector (default = 20260101)
		Returns:
			Float64: estimated largest eigenvalue (by magnitude)
		Notes:
			Standard power iteration:
				x ← A x; λ ← ||x||; x ← x / λ
			Stops when ||x_new - x|| < tol or after max_iter iterations.
			For symmetric non-negative matrices (the case of interest for
			Bonacich on a symmetrized adjacency) this converges geometrically
			at rate |λ_2 / λ_1| to the dominant eigenvalue.

			Returns 0.0 for the empty adjacency or for a matrix whose largest
			eigenvalue is degenerate at zero (e.g., a graph of pure isolates).

			Uses Xoshiro(seed) for the initial vector, matching the seed
			convention used elsewhere in the package.
		"""

		#	Dimensions
			n = size(adj, 1)
			if n == 0
				return 0.0
			end

		#	Initial Random Unit Vector
			rng = Random.Xoshiro(seed)
			x = randn(rng, n)
			nx = norm(x)
			if nx == 0
				return 0.0
			end
			x ./= nx

		#	Iterate
			lambda = 0.0
			@inbounds for iter in 1:max_iter
				#	Apply A
					y = adj * x

				#	Magnitude (Estimate of λ)
					nx = norm(y)
					if nx == 0
						#	Vector Collapsed: All Mass on Subspace with λ = 0
							return 0.0
					end

				#	Normalize
					y ./= nx

				#	Check Convergence
					#	Use the smaller of ||y - x|| and ||y + x|| to be sign-agnostic.
					#	For non-negative adjacencies the dominant eigenvector has
					#	non-negative entries, so ||y - x|| typically converges.
						diff1 = norm(y .- x)
						diff2 = norm(y .+ x)
						diff = min(diff1, diff2)

				#	Update and Possibly Terminate
					lambda = nx
					x = y
					if diff < tol
						break
					end
			end

		#	Return Largest Eigenvalue Estimate
			return lambda
	end

#	Bonacich Power Centrality
	function bonacich_centrality(edges::DataFrame;
	                            nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                            weighted::Bool = false,
	                            directed::Bool = true,
	                            direction::Symbol = :symmetric,
	                            beta::Union{Nothing, Float64} = nothing,
	                            normalize::Symbol = :none,
	                            edge_interpretation::Symbol = :ignore,
	                            agg_func::Function = sum,
	                            power_iter_max::Int = 200,
	                            power_iter_tol::Float64 = 1e-10,
	                            seed::Int = 20260101)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			weighted::Bool: kept for API symmetry; ignored when
				edge_interpretation=:ignore (default = false)
			directed::Bool: treat graph as directed (default = true)
			direction::Symbol: :out | :in | :symmetric (default = :symmetric)
				For directed graphs only; SMM 2022 specifies :symmetric.
			beta::Union{Nothing, Float64}: attenuation parameter. If nothing,
				defaults to 0.5 / λ_max(A_sym). Pass an explicit value to
				override; the caller is then responsible for ensuring
				|beta| < 1 / λ_max for convergence.
			normalize::Symbol: :none | :l2 | :max (default = :none)
				:none — raw values, α = 1 (SMM convention)
				:l2   — divide by sqrt(sum(c_i^2) / N) (igraph convention)
				:max  — divide by max, mapping to [0, 1]
			edge_interpretation::Symbol: :ignore only in this commit
			agg_func::Function: aggregation for parallel edges (default = sum)
			power_iter_max::Int: max iterations for spectral default (default = 200)
			power_iter_tol::Float64: convergence tolerance (default = 1e-10)
			seed::Int: RNG seed for spectral default's initial vector
		Returns:
			DataFrame: columns [node, bonacich, beta_used]
				beta_used is the same Float64 for every row, included so the
				caller can recover the (possibly auto-selected) β value
				without making a separate function call.
		Notes:
			Bonacich (1987) power centrality:
				c = α (I - β A)^{-1} A · 1
			Computed as a linear solve, not by matrix inversion. The result
			vector c gives each node's centrality as a weighted combination
			of its degree and its neighbors' centralities, attenuated by β.

			For directed networks, Smith/Morgan/Moody (2022) symmetrize the
			adjacency first (max(A, A^T)). This is the default direction.
			Use direction=:out or :in for asymmetric directed forms.

			The default β = 0.5 / λ_max(A_sym) puts β safely below the
			convergence bound 1/λ_max. λ_max is computed via power iteration
			on the symmetrized adjacency. For binarized non-negative
			adjacencies (the Phase 0 case) this is fast and reliable.

			Pass `nodes` to include isolates (Bonacich = 0) in the output.
			Without `nodes`, only nodes appearing in some edge are returned.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !(direction in (:out, :in, :symmetric))
				throw(ArgumentError("direction must be :out, :in, or :symmetric"))
			end
			if !(normalize in (:none, :l2, :max))
				throw(ArgumentError("normalize must be :none, :l2, or :max"))
			end
			if edge_interpretation != :ignore
				throw(ArgumentError(
					"bonacich_centrality: edge_interpretation=$(edge_interpretation) not yet implemented. " *
					"Pass edge_interpretation=:ignore to binarize before centrality computation."
				))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					if nodes isa DataFrame
						return DataFrame(node = nodes.id,
						                bonacich = zeros(Float64, nrow(nodes)),
						                beta_used = fill(0.0, nrow(nodes)))
					else
						return DataFrame(node = collect(nodes),
						                bonacich = zeros(Float64, length(nodes)),
						                beta_used = fill(0.0, length(nodes)))
					end
				else
					return DataFrame(node = [], bonacich = Float64[], beta_used = Float64[])
				end
			end

		#	Binarize Edges (edge_interpretation = :ignore)
			edges_binary = DataFrame(src = edges.src, dst = edges.dst)
			edges_binary.weight = ones(Float64, nrow(edges_binary))
			edges_binary = _aggregate_multi_edges(edges_binary; agg_func = maximum)

		#	Build Sparse Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(edges_binary; weighted = false)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(edges_binary;
				                                              nodes = nodes,
				                                              weighted = false)
			end

		#	Apply Direction Convention
			if !directed
				#	Always Symmetrize (Same Pattern as Closeness/Betweenness)
					adj = max.(adj, adj')
			else
				if direction == :out
					#	Use Adjacency As-Is
				elseif direction == :in
					adj = sparse(adj')
				else  # :symmetric
					adj = max.(adj, adj')
				end
			end

		#	Dimensions
			N = size(adj, 1)
			if N == 0
				return DataFrame(node = [], bonacich = Float64[], beta_used = Float64[])
			end

		#	Compute β (Spectral Default or Use Supplied Value)
			if beta === nothing
				#	Spectral Default: β = 0.5 / λ_max
					lambda_max = _power_iteration_lambda_max(adj;
					                                       max_iter = power_iter_max,
					                                       tol = power_iter_tol,
					                                       seed = seed)
					if lambda_max == 0.0
						#	Adjacency Is All Zeros (e.g., All Isolates): Centrality Is All Zeros
							node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
							return DataFrame(node = node_col,
							                bonacich = zeros(Float64, N),
							                beta_used = zeros(Float64, N))
					end
					beta_used = 0.5 / lambda_max
			else
				beta_used = Float64(beta)
			end

		#	Solve (I - β A) c = A · 1 for c
			#	Convert to Dense for the Solve. For N up to a few thousand
			#	this is straightforward; for larger networks consider a sparse
			#	iterative solver, but Phase 0 corpus is well within range.
				A_dense = Matrix(adj)
				ones_vec = ones(Float64, N)
				rhs = A_dense * ones_vec
				M = I - beta_used .* A_dense
				bonacich = M \ rhs

		#	Apply Optional Normalization
			if normalize == :l2
				#	igraph Convention: Divide by sqrt(sum(c^2) / N) so that
				#	the result has L2 norm sqrt(N). Equivalent to scaling
				#	the vector to have RMS = 1.
					rms = sqrt(sum(bonacich .^ 2) / N)
					if rms > 0
						bonacich = bonacich ./ rms
					end
			elseif normalize == :max
				#	Map to [0, 1] by Dividing by the Maximum
					max_val = maximum(bonacich)
					if max_val > 0
						bonacich = bonacich ./ max_val
					end
			end
			# :none — leave values as-is

		#	Assembling Result
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			return DataFrame(node = node_col,
			                bonacich = bonacich,
			                beta_used = fill(beta_used, N))
	end
	@doc raw"""
	**Description**
	Compute Bonacich (1987) power centrality for each node:

	$$c(\beta) = \alpha (I - \beta A)^{-1} A \mathbf{1}$$

	A node's centrality is a weighted combination of its degree and the
	centralities of its neighbors, attenuated by the parameter $\beta$. When
	$\beta > 0$, well-connected neighbors increase a node's score; when
	$\beta < 0$, structurally-isolated neighbors do. The default $\beta$ is
	chosen as half the reciprocal of $\lambda_{\max}(A)$, well inside the
	convergence bound.

	**Usage**
	`bonacich_centrality(edges::DataFrame; nodes=nothing, weighted=false, directed=true, direction=:symmetric, beta=nothing, normalize=:none, edge_interpretation=:ignore, agg_func=sum, power_iter_max=200, power_iter_tol=1e-10, seed=20260101)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. Pass to
	  include isolates (`bonacich = 0`) in the output.
	- `weighted::Bool`: API symmetry only; ignored when `edge_interpretation=:ignore`.
	- `directed::Bool`: Treat as directed (default `true`).
	- `direction::Symbol`: For directed graphs, one of `:out`, `:in`, or
	  `:symmetric` (default `:symmetric`). Smith/Morgan/Moody (2022) specify
	  symmetrization for Bonacich on directed networks; this is our default.
	- `beta::Union{Nothing,Float64}`: Attenuation parameter. If `nothing`,
	  defaults to $0.5 / \lambda_{\max}(A_{\text{sym}})$ computed via power
	  iteration. Pass an explicit value to override; the caller is then
	  responsible for ensuring $|\beta| < 1/\lambda_{\max}$ for convergence.
	- `normalize::Symbol`: `:none` (default — raw values, $\alpha = 1$),
	  `:l2` (RMS = 1, igraph convention), or `:max` (peak normalized to 1).
	- `edge_interpretation::Symbol`: Only `:ignore` currently supported.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).
	- `power_iter_max::Int`: Max iterations for spectral default (default 200).
	- `power_iter_tol::Float64`: Convergence tolerance (default `1e-10`).
	- `seed::Int`: RNG seed for the power iteration's initial vector.

	**Details**
	Computed as a linear solve $(I - \beta A) c = A \mathbf{1}$ for $c$, not by
	matrix inversion. For Phase 0 corpus sizes (up to N = 1347) this is a
	straightforward dense solve completing in a fraction of a second.

	Returns a DataFrame with an extra `:beta_used` column so that callers can
	recover the auto-selected $\beta$ without a separate function call. This is
	particularly useful for reporting and for downstream replication.

	For directed networks, the default symmetrization via $\max(A, A^T)$
	matches Smith, Morgan, & Moody (2022, p. 11): "Bonacich power centrality
	is calculated on a symmetrized version of the network (for the directed
	networks)."

	**Value**
	A `DataFrame` with columns `:node`, `:bonacich`, and `:beta_used`.

	**Examples**
	```julia
	using DataFrames

	#	Star graph K_{1,3}: hub gets higher Bonacich than leaves
	edges = DataFrame(src=[1,1,1], dst=[2,3,4])
	nodes = DataFrame(id=string.(1:4), label=string.(1:4))
	bonacich_centrality(edges; nodes=nodes, directed=false)
	```

	**References**
	- Bonacich P (1987). "Power and centrality: A family of measures."
	  *American Journal of Sociology* 92(5): 1170–1182.
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.

	**See Also**
	`closeness_centrality`, `betweenness_centrality`, `centralization`
	""" bonacich_centrality

##########################################
#   SECTION 5: BASIC TOPOLOGY MEASURES   #
##########################################

# ====================================================================
# Topology measures ported from Large_Graph_Similarity.jl so that this
# package is self-contained for Phase 0 reporting. Reviewers cloning
# the repository do not need a separate package install.
#
# Public API:
#   largest_component_proportion  — fraction of nodes in the largest WCC
#   reciprocity                   — directed reciprocity (arc/dyad, binary/weighted)
#   local_weighted_reciprocity    — node-level Squartini reciprocity
#   local_clustering_coefficient  — Watts-Strogatz / ORA / transitivity per node
#   global_clustering_coefficient — Newman transitivity or mean local clustering
#   triad_census                  — Batagelj-Mrvar 16-class directed,
#                                   4-class undirected mapped to DL order
#
# Internal helpers (not exported):
#   _weak_components, _strong_components, _directed_neighbors
#   _extract_ego_network, _count_triplets_directed_binary
#   _dl_labels, _dl_lookup, _make_directed_simple!,
#   _bm_union_neighbors_excluding, _bm_is_neighbor,
#   _triad_census_bm_directed, _triad_census_bm_undirected
#
# All measures binarize internally for Phase 0. The weighted/layered
# tau-thresholding path from Large_Graph_Similarity is NOT ported;
# pass weighted=false (the default) to use the binary BM census.
# ====================================================================

#	Helper Function for component analysis: Weakly Connected Components
	function _weak_components(adj::SparseMatrixCSC{<:Real, Int})
		"""
		Args:
			adj::SparseMatrixCSC{<:Real,Int}: adjacency matrix (directed or undirected)
		Returns:
			NamedTuple: (membership::Vector{Int}, sizes::Vector{Int})
		Notes:
			Components are weakly connected: directions ignored.
			Self-loops do not create connectivity beyond the node itself.
			BFS/DFS over symmetrized neighbor lists.
		"""

		#	Basic Checks
			n = size(adj, 1)
			@assert size(adj, 2) == n "adj must be square"

		#	Build Undirected Neighbor Lists (Binary, Drop Self-Loops)
			rows = rowvals(adj)
			neighbors = [Int[] for _ in 1:n]

			@inbounds for j in 1:n
				for idx in nzrange(adj, j)
					i = rows[idx]          # edge i → j
					i == j && continue     # ignore self-loops
					push!(neighbors[i], j)
					push!(neighbors[j], i)
				end
			end

		#	BFS/DFS over Undirected Neighbors
			membership = zeros(Int, n)
			sizes      = Int[]
			visited    = falses(n)
			comp_id    = 0
			stack      = Int[]

			@inbounds for v in 1:n
				if visited[v]
					continue
				end
				comp_id += 1
				comp_size = 0
				empty!(stack)
				push!(stack, v)
				visited[v] = true

				while !isempty(stack)
					u = pop!(stack)
					membership[u] = comp_id
					comp_size += 1
					for w in neighbors[u]
						if !visited[w]
							visited[w] = true
							push!(stack, w)
						end
					end
				end

				push!(sizes, comp_size)
			end

		#	Return Weak Components
			return (membership = membership, sizes = sizes)
	end

#	Helper Function for component analysis: Directed Neighbor Lists
	function _directed_neighbors(adj::SparseMatrixCSC{<:Real, Int})
		"""
		Args:
			adj::SparseMatrixCSC{<:Real,Int}: adjacency matrix (interpreted as directed)
		Returns:
			NamedTuple: (out_neighbors::Vector{Vector{Int}}, in_neighbors::Vector{Vector{Int}})
		Notes:
			Self-loops are ignored. Used by _strong_components.
		"""

		#	Basic Checks
			n = size(adj, 1)
			@assert size(adj, 2) == n "adj must be square"

		#	Allocate Neighbor Lists
			rows = rowvals(adj)
			out_neighbors = [Int[] for _ in 1:n]
			in_neighbors  = [Int[] for _ in 1:n]

		#	Populate (column j is destination, row i is source)
			@inbounds for j in 1:n
				for idx in nzrange(adj, j)
					i = rows[idx]
					i == j && continue
					push!(out_neighbors[i], j)
					push!(in_neighbors[j],  i)
				end
			end

		#	Return Both Sides
			return (out_neighbors = out_neighbors, in_neighbors = in_neighbors)
	end

#	Helper Function for component analysis: Strongly Connected Components
	function _strong_components(adj::SparseMatrixCSC{<:Real, Int})
		"""
		Args:
			adj::SparseMatrixCSC{<:Real,Int}: adjacency matrix (directed)
		Returns:
			NamedTuple: (membership::Vector{Int}, sizes::Vector{Int})
		Notes:
			Kosaraju two-pass algorithm on adjacency + transpose.
			Self-loops do not affect SCC structure.
		"""

		#	Basic Checks
			n = size(adj, 1)
			@assert size(adj, 2) == n "adj must be square"

		#	Directed Neighbors (Loopless)
			neigh = _directed_neighbors(adj)
			out_neighbors = neigh.out_neighbors
			in_neighbors  = neigh.in_neighbors

		#	First Pass: DFS Order on Original Graph
			visited = falses(n)
			order   = Int[]
			stack   = Vector{Tuple{Int, Bool}}()

			@inbounds for v in 1:n
				if visited[v]
					continue
				end
				push!(stack, (v, false))
				while !isempty(stack)
					(u, expanded) = pop!(stack)
					if !expanded
						if visited[u]
							continue
						end
						visited[u] = true
						push!(stack, (u, true))
						for w in out_neighbors[u]
							if !visited[w]
								push!(stack, (w, false))
							end
						end
					else
						push!(order, u)
					end
				end
			end

		#	Second Pass: DFS on Transpose Graph in Reverse Finish Order
			membership = zeros(Int, n)
			sizes      = Int[]
			fill!(visited, false)
			comp_id = 0
			stack_v = Int[]

			@inbounds for v in Iterators.reverse(order)
				if visited[v]
					continue
				end
				comp_id += 1
				comp_size = 0
				empty!(stack_v)
				push!(stack_v, v)
				visited[v] = true

				while !isempty(stack_v)
					u = pop!(stack_v)
					membership[u] = comp_id
					comp_size += 1
					for w in in_neighbors[u]
						if !visited[w]
							visited[w] = true
							push!(stack_v, w)
						end
					end
				end

				push!(sizes, comp_size)
			end

		#	Return Strong Components
			return (membership = membership, sizes = sizes)
	end

#	LARGEST WEAKLY CONNECTED COMPONENT AS A PROPORTION OF N
	function largest_component_proportion(edges::DataFrame;
	                                     nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                                     directed::Bool = true,
	                                     agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight (ignored)
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			directed::Bool: kept for API symmetry; component computation is on
				the undirected/symmetrized adjacency regardless (default = true)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			Float64: size of the largest weakly connected component, divided by N
		Notes:
			Smith, Morgan, & Moody (2022) topology measure: proportion of nodes
			in the largest connected component. The denominator N is taken from
			`nodes` when supplied (so isolates count toward N), or from the
			edge list otherwise (in which case isolates are missing).

			Components are always weakly connected: edge direction is ignored
			for connectivity. This matches SMM's "largest set of actors
			connected by at least one path" definition.

			Returns 0.0 for an empty edge list and 1.0 for a network that is
			fully connected (largest component = entire network).
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					n = nodes isa DataFrame ? nrow(nodes) : length(nodes)
					return n > 0 ? 1.0 / n : 0.0  # each isolate is its own component
				else
					return 0.0
				end
			end

		#	Aggregate Multi-Edges (Defensive)
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Build Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, _ = _edgelist_to_sparse_matrix(clean_edges; weighted = false)
			else
				adj, _, _ = _graph_to_sparse_matrix(clean_edges;
				                                    nodes = nodes,
				                                    weighted = false)
			end

		#	Compute Weak Components
			N = size(adj, 1)
			if N == 0
				return 0.0
			end
			wc = _weak_components(adj)

		#	Largest Component / N
			largest = isempty(wc.sizes) ? 0 : maximum(wc.sizes)
			return largest / N
	end
	@doc raw"""
	**Description**
	Compute the proportion of nodes in the largest weakly connected component
	of the network. One of Smith, Morgan, & Moody's (2022) topology measures.

	**Usage**
	`largest_component_proportion(edges::DataFrame; nodes=nothing, directed=true, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally `:weight`
	  (ignored; only edge presence matters).
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. When
	  supplied, the denominator $N$ includes isolates. Without `nodes`, $N$
	  is taken from the nodes that appear in some edge.
	- `directed::Bool`: Kept for API symmetry; component computation is always
	  on the undirected/symmetrized adjacency, since SMM's measure is about
	  weak connectivity ("largest set of actors connected by at least one
	  path"). Default `true`.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`,
	  immaterial since weights are ignored).

	**Details**
	Computed via flood-fill BFS over the symmetrized adjacency. Self-loops
	are ignored. Returns the size of the largest component divided by $N$.

	Pass `nodes` to ensure isolates are counted in $N$, lowering the
	proportion accordingly — this matches the SMM convention where the
	denominator is the network's true node count, not just the connected
	nodes.

	**Value**
	A `Float64` in $[0, 1]$.

	**Examples**
	```julia
	using DataFrames

	#	One connected component of 4, plus 1 isolate
	edges = DataFrame(src=[1,2,3], dst=[2,3,4])
	nodes = DataFrame(id=string.(1:5), label=string.(1:5))
	largest_component_proportion(edges; nodes=nodes)   # 4/5 = 0.8
	```

	**References**
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.

	**See Also**
	`largest_bicomponent_proportion` (Section 6), `centralization`
	""" largest_component_proportion

#	GLOBAL RECIPROCITY
	function reciprocity(edges::DataFrame;
	                    weighted::Bool = false,
	                    agg_func::Union{Function, Nothing} = nothing,
	                    mode::Symbol = :arc_based,
	                    weighted_method::Symbol = :squartini)
		"""
		Args:
			edges::DataFrame: must contain :src, :dst, optionally :weight
			weighted::Bool: enables weighted reciprocity if :weight exists (default = false)
			agg_func::Union{Function, Nothing}: aggregation for parallel edges
				(default = sum for weighted, maximum for binary)
			mode::Symbol: :arc_based or :dyad_based (default = :arc_based)
			weighted_method::Symbol: for dyad_based weighted only — :squartini or
				:ora_mutual (default = :squartini)
		Returns:
			Float64: reciprocity value based on selected mode
		Notes:
			Arc-based counts directed edges; dyad-based counts unordered pairs.
			Weighted dyad methods differ in how they handle weight asymmetry.
			Undirected graphs are not meaningful for reciprocity — pass directed
			edge lists. Returns 0.0 for empty edge lists.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !(mode in (:arc_based, :dyad_based))
				throw(ArgumentError("mode must be :arc_based or :dyad_based"))
			end
			if mode == :dyad_based && weighted && !(weighted_method in (:squartini, :ora_mutual))
				throw(ArgumentError("weighted_method must be :squartini or :ora_mutual for weighted dyad_based"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return 0.0
			end

		#	Set Default Aggregation Function
			if isnothing(agg_func)
				agg_func = weighted ? sum : maximum
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Build Adjacency Matrix
			use_weights = weighted && hasproperty(clean_edges, :weight)
			adj, _, _ = _edgelist_to_sparse_matrix(clean_edges; weighted = use_weights)
			n = size(adj, 1)

		#	Remove Self-Loops
			for i in 1:n
				adj[i, i] = 0
			end
			dropzeros!(adj)

		#	Calculate Based on Mode
			if mode == :arc_based
				#	Arc-Based: Fraction of Directed Edges with Reverse
					rows, cols, vals = findnz(adj)

					if use_weights
						#	Weighted: sum(w_ij * I{w_ji>0}) / sum(w_ij)
							reciprocal_weight = 0.0
							total_weight = 0.0

							for idx in 1:length(rows)
								i, j, w = rows[idx], cols[idx], vals[idx]
								total_weight += w
								if adj[j, i] > 0
									reciprocal_weight += w
								end
							end

							numerator = reciprocal_weight
							denominator = total_weight
					else
						#	Binary: Count Arcs with Reverse / Total Arcs
							reciprocal_arcs = 0

							for idx in 1:length(rows)
								i, j = rows[idx], cols[idx]
								if adj[j, i] > 0
									reciprocal_arcs += 1
								end
							end

							numerator = Float64(reciprocal_arcs)
							denominator = Float64(length(rows))
					end

			else  # mode == :dyad_based
				#	Dyad-Based: Fraction of Connected Dyads That Are Mutual
					if use_weights
						if weighted_method == :squartini
							#	Squartini: Sum of Reciprocated Weights / Total Weight
								rows, cols, vals = findnz(adj)
								total_reciprocated = 0.0
								total_weight = 0.0

								for idx in 1:length(rows)
									i, j, w = rows[idx], cols[idx], vals[idx]
									total_weight += w
									total_reciprocated += min(w, adj[j, i])
								end

								numerator = total_reciprocated
								denominator = total_weight

						else  # weighted_method == :ora_mutual
							#	ORA Mutual: Exact Weight Matching Required
								exact_match_dyads = 0
								total_connected_dyads = 0

								for i in 1:n
									for j in (i + 1):n
										w_ij = adj[i, j]
										w_ji = adj[j, i]

										if w_ij > 0 || w_ji > 0
											total_connected_dyads += 1
											if w_ij > 0 && w_ji > 0 && w_ij == w_ji
												exact_match_dyads += 1
											end
										end
									end
								end

								numerator = Float64(exact_match_dyads)
								denominator = Float64(total_connected_dyads)
						end
					else
						#	Binary: Mutual Dyads / Connected Dyads
							mutual_dyads = 0
							connected_dyads = 0

							for i in 1:n
								for j in (i + 1):n
									has_ij = adj[i, j] > 0
									has_ji = adj[j, i] > 0
									if has_ij || has_ji
										connected_dyads += 1
										if has_ij && has_ji
											mutual_dyads += 1
										end
									end
								end
							end

							numerator = Float64(mutual_dyads)
							denominator = Float64(connected_dyads)
					end
			end

		#	Calculate Final Reciprocity
			if denominator == 0
				return 0.0
			end
			return numerator / denominator
	end
	@doc raw"""
	**Description**
	Global reciprocity of a directed network. Defined either at the arc level
	(fraction of directed edges whose reverse exists) or the dyad level
	(fraction of connected dyads that are mutual).

	**Usage**
	`reciprocity(edges::DataFrame; weighted=false, mode=:arc_based, weighted_method=:squartini, agg_func=nothing)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`.
	- `weighted::Bool`: Use edge weights if available (default `false`).
	- `mode::Symbol`: `:arc_based` (default) or `:dyad_based`.
	- `weighted_method::Symbol`: For `:dyad_based` weighted only —
	  `:squartini` (default) or `:ora_mutual`.
	- `agg_func::Union{Function,Nothing}`: Aggregation for parallel edges
	  (default `sum` for weighted, `maximum` for binary).

	**Value**
	A `Float64` reciprocity score. Returns `0.0` for empty edge lists.

	**References**
	- Squartini T, Picciolo F, Ruzzenenti F, Garlaschelli D (2013). "Reciprocity
	  of weighted networks." *Scientific Reports* 3: 2729.

	**See Also**
	`local_weighted_reciprocity`
	""" reciprocity

#	LOCAL WEIGHTED RECIPROCITY (Squartini Per-Node)
	function local_weighted_reciprocity(edges::DataFrame;
	                                   weighted::Bool = true,
	                                   agg_func::Union{Function, Nothing} = nothing,
	                                   normalize::Symbol = :none)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optional :weight
			weighted::Bool: use weights if present (default = true)
			agg_func::Union{Function, Nothing}: aggregation for parallel edges
			normalize::Symbol: :none, :zscore, or :rank (default = :none)
		Returns:
			DataFrame: columns [node, r, reciprocated, out_strength, r_norm, normalization]
		Notes:
			Squartini et al. local weighted reciprocity:
				r_i = (Σ_j min(w_ij, w_ji)) / (Σ_j w_ij) for j ≠ i.
			Self-loops excluded. Zero out-strength gives r_i = 0.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges must have :src and :dst columns"))
			end
			if !(normalize in (:none, :zscore, :rank))
				throw(ArgumentError("normalize must be :none, :zscore, or :rank"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return DataFrame(
					node = String[],
					r = Float64[],
					reciprocated = Float64[],
					out_strength = Float64[],
					r_norm = Float64[],
					normalization = Symbol[]
				)
			end

		#	Set Default Aggregation Function
			if isnothing(agg_func)
				agg_func = (weighted && hasproperty(edges, :weight)) ? sum : maximum
			end

		#	Aggregate Parallel Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Build Adjacency Matrix
			use_weights = weighted && hasproperty(clean_edges, :weight)
			adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = use_weights)
			n = size(adj, 1)

		#	Remove Self-Loops
			for i in 1:n
				adj[i, i] = 0
			end
			dropzeros!(adj)

		#	Compute Out-Strength and Reciprocated Weight Per Node
			out_strength = Array{Float64}(undef, n)
			recip = Array{Float64}(undef, n)

			for i in 1:n
				#	Out-Strength: Sum of Outgoing Weights
					out_strength[i] = sum(adj[i, :])

				#	Reciprocated Weight: Sum of min(w_ij, w_ji) Over Out-Neighbors
					acc = 0.0
					cols, vals = findnz(adj[i, :])
					for t in 1:length(cols)
						j   = cols[t]
						wij = vals[t]
						wji = adj[j, i]
						acc += min(wij, wji)
					end
					recip[i] = acc
			end

		#	Calculate Raw Reciprocity r_i
			r = similar(recip)
			for i in 1:n
				r[i] = out_strength[i] > 0 ? recip[i] / out_strength[i] : 0.0
			end

		#	Apply Normalization
			r_norm = copy(r)

			if normalize == :zscore
				μ = mean(r)
				σ = std(r)
				if σ > 0
					for i in 1:n
						r_norm[i] = (r[i] - μ) / σ
					end
				else
					fill!(r_norm, 0.0)
				end

			elseif normalize == :rank
				#	Dense Ranks: Equal Values Share a Rank; Next Distinct Value Gets +1.
				#	Then Scale Ranks to [0, 1] So the Highest Tier Maps to 1.0.
					vals_vec = collect(r)
					uniq = sort(unique(vals_vec))
					rankmap = Dict(v => i for (i, v) in enumerate(uniq))
					ranks = [rankmap[v] for v in vals_vec]
					k = length(uniq)
					if k > 1
						for i in 1:n
							r_norm[i] = (ranks[i] - 1) / (k - 1)
						end
					else
						fill!(r_norm, 0.0)
					end
			end

		#	Assembling Result
			result = DataFrame(
				node = [idx_to_node[i] for i in 1:n],
				r = r,
				reciprocated = recip,
				out_strength = out_strength,
				r_norm = r_norm,
				normalization = fill(normalize, n)
			)
			return result
	end
	@doc raw"""
	**Description**
	Per-node weighted reciprocity following Squartini et al. (2013):

	$$r_i = \frac{\sum_{j \neq i} \min(w_{ij}, w_{ji})}{\sum_{j \neq i} w_{ij}}$$

	**Usage**
	`local_weighted_reciprocity(edges::DataFrame; weighted=true, normalize=:none, agg_func=nothing)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`.
	- `weighted::Bool`: Use weights if present (default `true`).
	- `agg_func`: Aggregation for parallel edges.
	- `normalize::Symbol`: `:none`, `:zscore`, or `:rank`.

	**Value**
	`DataFrame` with columns `:node`, `:r`, `:reciprocated`, `:out_strength`,
	`:r_norm`, `:normalization`.

	**References**
	- Squartini T, Picciolo F, Ruzzenenti F, Garlaschelli D (2013). "Reciprocity
	  of weighted networks." *Scientific Reports* 3: 2729.

	**See Also**
	`reciprocity`
	""" local_weighted_reciprocity

#	Helper Function for clustering: Extract Ego Network Submatrix
	function _extract_ego_network(adj::SparseMatrixCSC{Float64, Int64},
	                             node_idx::Int;
	                             directed::Bool = true)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency matrix
			node_idx::Int: index of ego node (1-based)
			directed::Bool: whether graph is directed (default = true)
		Returns:
			Tuple{Vector{Int}, SparseMatrixCSC}: (neighbor_indices, ego_subnet_adjacency)
		Notes:
			For directed graphs, neighbors include any node with edge to/from ego.
			Self-loops on the ego are excluded from the neighbor list.
		"""

		#	Find Neighbors
			if directed
				out_neighbors = findnz(adj[node_idx, :])[1]
				in_neighbors  = findnz(adj[:, node_idx])[1]
				neighbors = unique(vcat(out_neighbors, in_neighbors))
			else
				neighbors = findnz(adj[node_idx, :])[1]
			end

		#	Remove Self-Loop If Present
			neighbors = filter(n -> n != node_idx, neighbors)

		#	Include Ego in the Network
			ego_nodes = vcat(node_idx, neighbors)

		#	Extract Submatrix
			ego_subnet = adj[ego_nodes, ego_nodes]

		#	Return Neighbor Indices and Submatrix
			return (neighbors, ego_subnet)
	end

#	Helper Function for clustering: Count Directed Triplets (Binary)
	function _count_triplets_directed_binary(adj::SparseMatrixCSC{Float64, Int64},
	                                        node_idx::Int)
		"""
		Args:
			adj::SparseMatrixCSC: binary adjacency. adj[i, j] > 0 means edge i → j.
			node_idx::Int: index of center node (1-based).
		Returns:
			Tuple{Float64, Float64}: (closed_triplets, total_nonvacuous_triplets)
		Notes:
			Directed triplets are non-vacuous 2-paths centered at node_idx:
				j → node_idx → k
				k → node_idx → j
			A triplet is closed if the corresponding j→k (or k→j) edge exists.
			Pure in-stars (j → idx, k → idx) and pure out-stars (idx → j, idx → k)
			are excluded.
		"""

		#	Get In- and Out-Neighbors
			out_neighbors = findnz(adj[node_idx, :])[1]
			in_neighbors  = findnz(adj[:, node_idx])[1]

		#	Remove Self
			out_neighbors = filter(n -> n != node_idx, out_neighbors)
			in_neighbors  = filter(n -> n != node_idx, in_neighbors)

		#	Initialize Counts
			total_triplets  = 0.0
			closed_triplets = 0.0

		#	Type 1: j → node_idx → k
			for j in in_neighbors
				for k in out_neighbors
					if j == k
						continue
					end
					total_triplets += 1.0
					if adj[j, k] > 0
						closed_triplets += 1.0
					end
				end
			end

		#	Type 2: k → node_idx → j
			for k in in_neighbors
				for j in out_neighbors
					if j == k
						continue
					end
					total_triplets += 1.0
					if adj[k, j] > 0
						closed_triplets += 1.0
					end
				end
			end

		#	Return (closed, total)
			return (closed_triplets, total_triplets)
	end

#	LOCAL CLUSTERING COEFFICIENT & EGO DENSITY (Node Level)
	function local_clustering_coefficient(edges::DataFrame;
	                                     directed::Bool = true,
	                                     method::Symbol = :local_clustering,
	                                     agg_func::Function = Base.sum,
	                                     include_selfloops::Union{Bool, Nothing} = nothing)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst columns
			directed::Bool: treat as directed (default = true)
			method::Symbol: :local_clustering | :local_density | :local_transitivity
			agg_func::Function: aggregation for multi-edges (default = sum)
			include_selfloops::Union{Bool, Nothing}: include self-loops in alter count
				nothing → defaults to true for :local_density, false otherwise
		Returns:
			DataFrame: columns [node, ego_density, local_clustering_coefficient]
		Notes:
			Classic :local_clustering uses k(k - 1) denominator, excludes loops.
			ORA :local_density uses combinations-with-replacement if loops included.
			:local_transitivity (directed) uses non-vacuous directed triplets.
			All measures use binary (unweighted) edges.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have :src and :dst columns"))
			end
			if !(method in (:local_clustering, :local_density, :local_transitivity))
				throw(ArgumentError("method must be :local_clustering, :local_density, or :local_transitivity"))
			end

		#	Set Method-Appropriate Defaults for Self-Loop Inclusion
			if isnothing(include_selfloops)
				include_selfloops = (method == :local_density)
			end

		#	Handle Empty Input
			if nrow(edges) == 0
				return DataFrame(node = [], ego_density = Float64[], local_clustering_coefficient = Float64[])
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Undirected Binary Path (Edge-List Based, Unordered Counting)
			if method in (:local_clustering, :local_density) && !directed
				T = promote_type(eltype(clean_edges.src), eltype(clean_edges.dst))

				#	Build Neighbor Sets (Self-Loops Do Not Create Neighbors)
					neighbors = Dict{T, Set{T}}()
					all_nodes = Set{T}()
					for row in eachrow(clean_edges)
						s = T(row.src); d = T(row.dst)
						push!(all_nodes, s); push!(all_nodes, d)
						if s != d
							if !haskey(neighbors, s); neighbors[s] = Set{T}(); end
							if !haskey(neighbors, d); neighbors[d] = Set{T}(); end
							push!(neighbors[s], d); push!(neighbors[d], s)
						end
					end
					nodes_vec = sort(collect(all_nodes))
					n_nodes = length(nodes_vec)

				#	Build Unordered Edge Sets
					undirected_pairs = Set{Tuple{T, T}}()
					loop_nodes = Set{T}()
					for row in eachrow(clean_edges)
						s = T(row.src); d = T(row.dst)
						if s == d
							push!(loop_nodes, s)
						else
							a, b = ifelse(s < d, (s, d), (d, s))
							push!(undirected_pairs, (a, b))
						end
					end

				#	Initialize Results
					ego_density = zeros(Float64, n_nodes)
					local_cc = zeros(Float64, n_nodes)

				#	Process Each Ego
					for (idx, ego) in enumerate(nodes_vec)
						if !haskey(neighbors, ego)
							continue
						end

						#	Get Alters
							alters = collect(neighbors[ego])
							k = length(alters)
							if k < 1
								continue
							end
							ego_nodes = Set{T}([ego; alters...])
							n_ego = length(ego_nodes)

						#	Count Ego-Network Edges (Excluding Loops)
							E_ego = 0
							for (a, b) in undirected_pairs
								if a in ego_nodes && b in ego_nodes
									E_ego += 1
								end
							end
							den_ego = n_ego * (n_ego - 1) ÷ 2
							ego_density[idx] = (den_ego > 0) ? (E_ego / den_ego) : 0.0

						#	Count Alter-Only Edges and Loops
							alter_set = Set{T}(alters)
							E_alter = 0
							for (a, b) in undirected_pairs
								if (a in alter_set) && (b in alter_set)
									E_alter += 1
								end
							end
							L_alter = count(in(alter_set), loop_nodes)

						#	Compute Local Coefficient
							if method == :local_clustering
								#	Classic Watts-Strogatz: No Loops
									den = k * (k - 1) ÷ 2
									num = E_alter
							else  # :local_density
								#	ORA-Style With Optional Loops
									if include_selfloops
										den = (k * (k + 1)) ÷ 2
										num = E_alter + L_alter
									else
										den = k * (k - 1) ÷ 2
										num = E_alter
									end
							end
							local_cc[idx] = (den > 0) ? (num / den) : 0.0
					end

				#	Return Results
					return DataFrame(
						node = nodes_vec,
						ego_density = ego_density,
						local_clustering_coefficient = local_cc
					)
			end

		#	Directed / Transitivity Path (Matrix-Based)
			adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = false)
			n = length(idx_to_node)
			ego_density = zeros(Float64, n)
			local_cc = zeros(Float64, n)

			for i in 1:n
				if method == :local_transitivity
					#	Directed Local Transitivity: Non-Vacuous Triplets Centered at i
						closed_i, triplets_i = _count_triplets_directed_binary(adj, i)
						local_cc[i]   = (triplets_i > 0.0) ? (closed_i / triplets_i) : 0.0
						ego_density[i] = 0.0

				else
					#	Extract Ego Subnet
						_, ego_subnet = _extract_ego_network(adj, i; directed = directed)
						if size(ego_subnet, 1) <= 1
							continue
						end
						k_ego = size(ego_subnet, 1)
						neighbor_indices = 2:k_ego
						neighbor_subnet = ego_subnet[neighbor_indices, neighbor_indices]
						k = length(neighbor_indices)

					#	Count Alter Edges (Binary)
						if directed
							if method == :local_clustering || !include_selfloops
								neighbor_nodiag = copy(neighbor_subnet)
								for d in 1:k; neighbor_nodiag[d, d] = 0; end
								edge_sum = nnz(neighbor_nodiag)
							else
								edge_sum = nnz(neighbor_subnet)
							end
						else
							if method == :local_density && include_selfloops
								loops = 0
								for d in 1:k
									loops += (neighbor_subnet[d, d] != 0)
								end
								edges_count = nnz(triu(neighbor_subnet, 1))
								edge_sum = edges_count + loops
							else
								edge_sum = nnz(triu(neighbor_subnet, 1))
							end
						end

					#	Compute Denominator
						if directed
							if method == :local_clustering
								max_edges = k * (k - 1)
							else
								max_edges = include_selfloops ? (k * k) : (k * (k - 1))
							end
						else
							if method == :local_clustering
								max_edges = k * (k - 1) ÷ 2
							else
								max_edges = include_selfloops ? (k * (k + 1) ÷ 2) : (k * (k - 1) ÷ 2)
							end
						end
						local_cc[i] = (max_edges > 0) ? (edge_sum / max_edges) : 0.0

					#	Compute Ego Density (Binary)
						if directed
							ego_sum = nnz(ego_subnet)
							ego_max = k_ego * (k_ego - 1)
							ego_density[i] = (ego_max > 0) ? (ego_sum / ego_max) : 0.0
						else
							ego_edges = nnz(triu(ego_subnet, 1))
							den_ego = k_ego * (k_ego - 1) ÷ 2
							ego_density[i] = (den_ego > 0) ? (ego_edges / den_ego) : 0.0
						end
				end
			end

		#	Return Results
			return DataFrame(
				node = idx_to_node,
				ego_density = ego_density,
				local_clustering_coefficient = local_cc
			)
	end
	@doc raw"""
	**Description**
	Per-node clustering coefficient (and ego density), with three method
	variants. Classic Watts-Strogatz `:local_clustering` uses $k(k-1)$
	denominator and excludes self-loops. ORA `:local_density` uses
	combinations-with-replacement if loops included. `:local_transitivity`
	(directed) uses non-vacuous directed triplets.

	**Usage**
	`local_clustering_coefficient(edges::DataFrame; directed=true, method=:local_clustering, ...)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`
	  (ignored — binary input).
	- `directed::Bool`: Treat as directed (default `true`).
	- `method::Symbol`: `:local_clustering` (default), `:local_density`,
	  or `:local_transitivity`.
	- `agg_func::Function`: Aggregation for multi-edges (default `sum`).
	- `include_selfloops::Union{Bool, Nothing}`: When `nothing`, defaults to
	  `true` for `:local_density`, `false` otherwise.

	**Value**
	`DataFrame` with columns `:node`, `:ego_density`, `:local_clustering_coefficient`.

	**References**
	- Watts DJ, Strogatz SH (1998). "Collective dynamics of 'small-world'
	  networks." *Nature* 393: 440–442.

	**See Also**
	`global_clustering_coefficient`
	""" local_clustering_coefficient

#	GLOBAL CLUSTERING COEFFICIENT
	function global_clustering_coefficient(edges::DataFrame;
	                                      directed::Bool = true,
	                                      method::Symbol = :average,
	                                      average_mode::Symbol = :local_clustering,
	                                      agg_func::Function = sum,
	                                      drop_self_loops::Bool = true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			directed::Bool: treat as directed (default = true)
			method::Symbol: :average | :transitivity
			average_mode::Symbol: when method = :average, which local measure to
				average over (:local_clustering or :local_density)
			agg_func::Function: aggregation passed to local_clustering_coefficient
			drop_self_loops::Bool: remove self-loops before computing (default = true)
		Returns:
			Float64: global clustering coefficient
		Notes:
			All computations on a binarized (unweighted) graph.
			:average — mean of node-level clustering values.
			:transitivity — global ratio of closed to connected triples.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("global_clustering_coefficient: edges must have :src and :dst columns"))
			end
			if !(method in (:average, :transitivity))
				throw(ArgumentError("global_clustering_coefficient: method must be :average or :transitivity"))
			end
			if method == :average && !(average_mode in (:local_clustering, :local_density))
				throw(ArgumentError("global_clustering_coefficient: average_mode must be :local_clustering or :local_density"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				return 0.0
			end

		#	Binarize Network (Aggregate Multi-Edges to Presence/Absence)
			clean_edges = deepcopy(edges)
			clean_edges.weight = ones(Float64, nrow(clean_edges))
			clean_edges = _aggregate_multi_edges(clean_edges; agg_func = maximum)

		#	Base Edge Set with Optional Self-Loop Removal
			edges_effective = clean_edges
			if drop_self_loops
				if hasproperty(edges_effective, :weight)
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst, [:src, :dst, :weight]]
				else
					edges_effective = edges_effective[edges_effective.src .!= edges_effective.dst, [:src, :dst]]
				end
			end

		#	Average of Locals: Delegate and Take Mean
			if method == :average
				local_df = local_clustering_coefficient(
					edges_effective;
					directed = directed,
					method = average_mode,
					agg_func = agg_func,
					include_selfloops = nothing
				)

				#	Adjust If :local_clustering Selected
					if average_mode == :local_clustering
						#	Watts-Strogatz: Average Only Over Nodes With Degree ≥ 2
							degree_scores = total_degree(edges_effective; drop_self_loops = true)
							leftjoin!(local_df, degree_scores, on = :node)
							local_df.total_degree = convert.(Float64, local_df.total_degree)

							vals = local_df[(local_df.total_degree .>= 2), :local_clustering_coefficient]
					else
						#	ORA Density: Average Over All Nodes
							vals = local_df.local_clustering_coefficient
					end

				return isempty(vals) ? 0.0 : mean(vals)
			end

		#	:transitivity Path
			if directed
				#	Directed: Non-Vacuous Triplets (j → i → k)
					adj, _, _ = _edgelist_to_sparse_matrix(edges_effective; weighted = false)

					total_closed   = 0.0
					total_triplets = 0.0
					n = size(adj, 1)

					for i in 1:n
						closed_i, triplets_i = _count_triplets_directed_binary(adj, i)
						total_closed   += closed_i
						total_triplets += triplets_i
					end

					return total_triplets > 0.0 ? (total_closed / total_triplets) : 0.0

			else
				#	Undirected, Binary, Loopless Classic Transitivity (ORA/NetStat Style)
					edges_canonical = DataFrame(
						src = min.(edges_effective.src, edges_effective.dst),
						dst = max.(edges_effective.src, edges_effective.dst)
					)
					edges_simple = unique(edges_canonical)

				#	Duplicate Both Directions for an Undirected Adjacency
					edges_bidirectional = vcat(
						edges_simple,
						DataFrame(src = edges_simple.dst, dst = edges_simple.src)
					)

					A, _, _ = _edgelist_to_sparse_matrix(edges_bidirectional; weighted = false)

				#	Ensure Strictly Binary, Symmetric, Zero-Diagonal
					A = max.(A, A')
					if drop_self_loops
						A = A .- spdiagm(0 => diag(A))
					end
					A = spzeros(Float64, size(A)...) .+ (A .> 0)

				#	Denominator: Σ k_i (k_i - 1) = 2 * (# connected triples)
					k = vec(sum(A, dims = 2))
					den = sum(k .* (k .- 1))
					if den == 0.0
						return 0.0
					end

				#	Numerator: 6 * (# triangles) via sum((A * A) .* A)
					tri6 = sum((A * A) .* A)

				#	Classic Transitivity
					return tri6 / den
			end
	end
	@doc raw"""
	**Description**
	Global clustering coefficient. Either the mean of node-level local
	clustering values (`method = :average`) or the global transitivity ratio
	of closed to connected triples (`method = :transitivity`).

	**Usage**
	`global_clustering_coefficient(edges::DataFrame; directed=true, method=:average, average_mode=:local_clustering, ...)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`.
	- `directed::Bool`: Treat as directed (default `true`).
	- `method::Symbol`: `:average` (default) or `:transitivity`.
	- `average_mode::Symbol`: For `:average` — `:local_clustering` (default,
	  Watts-Strogatz; averages only over nodes with degree ≥ 2) or
	  `:local_density` (ORA style; averages over all nodes).
	- `agg_func::Function`: Aggregation for multi-edges (default `sum`).
	- `drop_self_loops::Bool`: Remove self-loops before computing (default `true`).

	**Value**
	A `Float64` in $[0, 1]$.

	**References**
	- Watts DJ, Strogatz SH (1998). *Nature* 393: 440–442.
	- Newman MEJ (2003). "The structure and function of complex networks."
	  *SIAM Review* 45(2): 167–256.

	**See Also**
	`local_clustering_coefficient`, `triad_census`
	""" global_clustering_coefficient

# TRIAD CENSUS
# Implements Batagelj-Mrvar (2001) via RSiena's dyad-driven approach.
# Triad classes follow Davis-Leinhardt order:
#   003, 012, 102, 021D, 021U, 021C, 111D, 111U,
#   030T, 030C, 201, 120D, 120U, 120C, 210, 300
#
# The weighted/layered (tau-thresholded) path from the source package
# is NOT ported here. For Phase 0, pass weighted=false (default).

#	Helper Function for triad_census: Davis-Leinhardt Labels
	function _dl_labels()
		"""
		Args:
			None
		Returns:
			Vector{String}: 16 Davis-Leinhardt triad class labels
		Notes:
			Order matches RSiena's 'tc' vector in sienaGOF TriadCensus.
		"""
		return ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
		        "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]
	end

#	Helper Function for triad_census: 4x4x4 Lookup Table
	function _dl_lookup()
		"""
		Args:
			None
		Returns:
			Array{Int, 3}: lookup[t1, t2, t3] → triad class index (1..16)
		Notes:
			t1, t2, t3 ∈ {1: empty, 2: forward, 3: backward, 4: reciprocal}.
			Mapping mirrors the RSiena R code.
		"""
		L = fill(0, (4, 4, 4))

		#	i → j, j → k, i → k (Copying RSiena's Assignments)
			L[1, 1, 1] = 1
			L[2, 1, 1] = L[1, 2, 1] = L[1, 1, 2] = L[3, 1, 1] = L[1, 3, 1] = L[1, 1, 3] = 2
			L[4, 1, 1] = L[1, 4, 1] = L[1, 1, 4] = 3
			L[2, 1, 2] = L[3, 2, 1] = L[1, 3, 3] = 4
			L[2, 3, 1] = L[3, 1, 3] = L[1, 2, 2] = 5
			L[2, 2, 1] = L[3, 3, 1] = L[2, 1, 3] = L[3, 1, 2] = L[1, 2, 3] = L[1, 3, 2] = 6
			L[4, 3, 1] = L[4, 1, 3] = L[2, 4, 1] = L[1, 4, 2] = L[3, 1, 4] = L[1, 2, 4] = 7
			L[4, 2, 1] = L[4, 1, 2] = L[3, 4, 1] = L[1, 4, 3] = L[2, 1, 4] = L[1, 3, 4] = 8
			L[2, 2, 2] = L[2, 3, 3] = L[2, 3, 2] = L[3, 3, 3] = L[3, 2, 2] = L[3, 2, 3] = 9
			L[2, 2, 3] = L[3, 3, 2] = 10
			L[4, 4, 1] = L[4, 1, 4] = L[1, 4, 4] = 11
			L[2, 4, 2] = L[3, 2, 4] = L[4, 3, 3] = 12
			L[2, 3, 4] = L[3, 4, 3] = L[4, 2, 2] = 13
			L[2, 2, 4] = L[3, 3, 4] = L[2, 4, 3] = L[3, 4, 2] = L[4, 2, 3] = L[4, 3, 2] = 14
			L[2, 4, 4] = L[4, 2, 4] = L[4, 4, 2] = L[3, 4, 4] = L[4, 3, 4] = L[4, 4, 3] = 15
			L[4, 4, 4] = 16

		#	Return
			return L
	end

#	Helper Function for triad_census: Make Adjacency Directed Simple (0/1, Loopless)
	function _make_directed_simple!(adj::SparseMatrixCSC{Float64, Int})
		"""
		Args:
			adj::SparseMatrixCSC{Float64, Int}: adjacency to be normalized in place
		Returns:
			SparseMatrixCSC{Float64, Int}: the same (modified) adjacency
		Notes:
			Binarizes per-direction (any positive value → 1, zero/negative → 0)
			and drops self-loops.
		"""

		#	Binarize Per Direction
			vals = nonzeros(adj)
			@inbounds for t in eachindex(vals)
				vals[t] = vals[t] > 0 ? 1.0 : 0.0
			end

		#	Drop Self-Loops
			n = size(adj, 1)
			@inbounds for i in 1:n
				adj[i, i] = 0.0
			end
			dropzeros!(adj)
			return adj
	end

#	Helper Function for triad_census: Union of Neighbor Lists Excluding Two Nodes
	function _bm_union_neighbors_excluding(a::Vector{Int}, b::Vector{Int}, i::Int, j::Int)
		"""
		Args:
			a::Vector{Int}: neighbors of i (any direction)
			b::Vector{Int}: neighbors of j (any direction)
			i::Int, j::Int: indices to exclude
		Returns:
			Vector{Int}: sorted unique union minus {i, j}
		Notes:
			Uses sort + unique for determinism.
		"""
		if isempty(a)
			u = copy(b)
		elseif isempty(b)
			u = copy(a)
		else
			u = vcat(a, b)
		end
		if !isempty(u)
			sort!(u)
			u = unique(u)
			(pos = searchsortedfirst(u, i)) <= length(u) && u[pos] == i && deleteat!(u, pos)
			(pos = searchsortedfirst(u, j)) <= length(u) && u[pos] == j && deleteat!(u, pos)
		end
		return u
	end

#	Helper Function for triad_census: Binary-Search Membership Test
	function _bm_is_neighbor(nbrs::Vector{Int}, k::Int)
		"""
		Args:
			nbrs::Vector{Int}: sorted neighbors
			k::Int: node id
		Returns:
			Bool: true if k ∈ nbrs
		Notes:
			Expects nbrs sorted (we sort in the union helper).
		"""
		if isempty(nbrs); return false; end
		idx = searchsortedfirst(nbrs, k)
		return (idx <= length(nbrs)) && (nbrs[idx] == k)
	end

#	Helper Function for triad_census: BM Triad Census (Directed Binary)
	function _triad_census_bm_directed(adj::SparseMatrixCSC{Float64, Int})
		"""
		Args:
			adj::SparseMatrixCSC{Float64, Int}: directed simple graph (0/1), no self-loops
		Returns:
			NamedTuple: (counts::Vector{Int}, labels::Vector{String})
		Notes:
			Implements Batagelj-Mrvar (2001) via RSiena's dyad-driven approach.
			Assumes binary, loopless adjacency.
		"""

		#	Dimensions & Quick Guards
			n = size(adj, 1)
			@assert size(adj, 2) == n "adj must be square"
			n == 0 && return (counts = zeros(Int, 16), labels = _dl_labels())

		#	Ensure Binary Semantics (Defensive)
			vals = nonzeros(adj)
			@inbounds for t in eachindex(vals)
				vals[t] = vals[t] > 0 ? 1.0 : 0.0
			end

		#	Precompute Transpose (For Dyad Tests)
			adjT = SparseMatrixCSC(transpose(adj))

		#	Local Edge Tests
			@inline has_ij(i::Int, j::Int) = adj[i, j] != 0.0
			@inline has_ji(i::Int, j::Int) = adjT[i, j] != 0.0  # == adj[j, i]

		#	Dyad Code to 1..4: 1 Empty, 2 i→j, 3 j→i, 4 Reciprocal
			@inline function dyad_code(i::Int, j::Int)
				a = has_ij(i, j)
				b = has_ji(i, j)
				return a ? (b ? 4 : 2) : (b ? 3 : 1)
			end

		#	Neighbors for Each Node (Any Direction)
			neighbors_arr = Vector{Vector{Int}}(undef, n)
			@inbounds for i in 1:n
				outs = findnz(adj[i, :])[1]
				ins  = findnz(adj[:, i])[1]
				outs = filter(j -> j != i, outs)
				ins  = filter(j -> j != i, ins)
				neighbors_arr[i] = isempty(outs) ? unique(ins) : isempty(ins) ? unique(outs) : unique(vcat(outs, ins))
			end

		#	Neighbors with Higher Index (i < j)
			neighborsHigher = Vector{Vector{Int}}(undef, n)
			@inbounds for i in 1:n
				neighborsHigher[i] = isempty(neighbors_arr[i]) ? Int[] : [j for j in neighbors_arr[i] if j > i]
			end

		#	Init Results
			labels = _dl_labels()
			tc     = zeros(Int, 16)
			lookup = _dl_lookup()

		#	Main Dyad Loop
			if any(!isempty, neighborsHigher)
				@inbounds for i in 1:n
					for j in neighborsHigher[i]
						third = _bm_union_neighbors_excluding(neighbors_arr[i], neighbors_arr[j], i, j)

						#	Single-Dyad Triads: (i, j) Plus Isolated k
							tc[(dyad_code(i, j) == 4) ? 3 : 2] += n - length(third) - 2

						#	Enumerate Triads with Third Node Present
							for k in third
								if j < k || (i < k && k < j && !_bm_is_neighbor(neighbors_arr[i], k))
									t1 = dyad_code(i, j)
									t2 = dyad_code(j, k)
									t3 = dyad_code(i, k)
									tc[lookup[t1, t2, t3]] += 1
								end
							end
					end
				end
			end

		#	Empty Triads by Residual
			total_triads = (n * (n - 1) * (n - 2)) ÷ 6
			tc[1] = total_triads - sum(tc[2:end])

		#	Return
			return (counts = tc, labels = labels)
	end

#	Helper Function for triad_census: Normalize Undirected BM Output to counts16 Vector
	function _bm_undir_counts16(Au::SparseMatrixCSC{Float64,Int})
		"""
		Args:
			Au::SparseMatrixCSC: symmetric 0/1, zero diagonal (undirected simple graph)
		Returns:
			Vector{Int}: length-16 counts in Davis-Leinhardt order
		Notes:
			Wraps _triad_census_bm_undirected to provide a consistent
			Vector{Int} return regardless of which return-shape variant the
			underlying kernel uses. Handles either:
			  - Vector{Int} (current package convention)
			  - NamedTuple(counts = ::Vector{Int}, labels = ...) (legacy)
			  - NamedTuple(counts16 = ::Vector{Int}) (older legacy from the
			    source Large_Graph_Similarity package)
			Only {003, 102, 201, 300} can be non-zero for undirected graphs;
			all other slots are zero.

			This normalizer exists so that callers (notably _triad_census_layered)
			can treat directed and undirected outputs uniformly: a length-16
			Vector{Int} in DL order regardless of the kernel's choice of
			return shape.
		"""

		#	Call the Undirected BM Kernel
			res = _triad_census_bm_undirected(Au)

		#	Normalize to Vector{Int}
			if res isa AbstractVector
				#	Already a length-16 vector
					return res
			elseif hasproperty(res, :counts)
				#	NamedTuple with :counts (current package convention)
					return res.counts
			elseif hasproperty(res, :counts16)
				#	NamedTuple with :counts16 (older legacy from Large_Graph_Similarity)
					return res.counts16
			else
				throw(ArgumentError("_bm_undir_counts16: unrecognized return shape from _triad_census_bm_undirected: $(typeof(res))"))
			end
	end

#	Helper Function for triad_census: BM Triad Census (Undirected Binary)
	function _triad_census_bm_undirected(Au::SparseMatrixCSC{Float64, Int})
		"""
		Args:
			Au::SparseMatrixCSC: symmetric 0/1, zero diagonal (undirected simple graph)
		Returns:
			NamedTuple: (counts::Vector{Int}, labels::Vector{String})
		Notes:
			Only {003, 102, 201, 300} can be non-zero for undirected graphs.
			Returns the standard length-16 DL-ordered vector with zeros elsewhere.
			Return shape matches _triad_census_bm_directed for API consistency
			(fixes a return-shape inconsistency in the source package).
		"""

		#	Dimensions & Guards
			n = size(Au, 1)
			@assert size(Au, 2) == n "Au must be square"

		#	Indices in DL Vector
			idx003 = 1
			idx102 = 3
			idx201 = 11
			idx300 = 16

		#	Initialize Counts
			counts = zeros(Int, 16)
			if n < 3
				return (counts = counts, labels = _dl_labels())
			end

		#	Iterate All Node Triples i < j < k (Read Upper Triangle Only)
			for i in 1:n - 2
				for j in i + 1:n - 1
					eij = (Au[i, j] != 0.0)
					for k in j + 1:n
						eik = (Au[i, k] != 0.0)
						ejk = (Au[j, k] != 0.0)
						m = (eij ? 1 : 0) + (eik ? 1 : 0) + (ejk ? 1 : 0)
						if m == 0
							counts[idx003] += 1
						elseif m == 1
							counts[idx102] += 1
						elseif m == 2
							counts[idx201] += 1
						else
							counts[idx300] += 1
						end
					end
				end
			end

		#	Return
			return (counts = counts, labels = _dl_labels())
	end

#	Helper Function for triad_census: Compute Log-Spaced τ Grid
	function _tau_grid(weights::Vector{Float64};
						L::Int=40,
						tau_min::Union{Symbol,Float64}=:auto,
						tau_max::Union{Symbol,Float64}=:auto)
		"""
		Args:
			weights::Vector{Float64}: positive edge weights (zeros excluded)
			L::Int: number of τ points (default 40)
			tau_min::Union{:auto,Float64}: lower τ bound (default :auto = max(eps(), q005))
			tau_max::Union{:auto,Float64}: upper τ bound (default :auto = maximum weight)
		Returns:
			Vector{Float64}: log-spaced τ values in [tau_min, tau_max]
		Notes:
			Builds the threshold grid used by _triad_census_layered. The grid is
			log-spaced because triadic structure typically responds to weight
			thresholds multiplicatively, not additively — doubling τ has a
			characteristically different effect than adding a constant.

			:auto defaults are conservative: tau_min uses the 0.5%-quantile of
			positive weights (clipped from below by eps()) to avoid wasting
			grid points on noise; tau_max uses the actual maximum to ensure
			the grid covers the full weight range.

			If all weights are equal (degenerate grid), returns the singleton
			value. If L ≤ 1, returns [tau_max] (single threshold).
		"""

		#	Filter to Positive Weights Only
			wpos = filter(>(0.0), weights)
			if isempty(wpos)
				return [1.0]  # no positive weights; degenerate grid
			end

		#	Compute Bounds
			wmin = minimum(wpos)
			wmax = maximum(wpos)
			tmin = tau_min === :auto ? max(eps(), quantile(wpos, 0.005)) : Float64(tau_min)
			tmax = tau_max === :auto ? wmax : Float64(tau_max)
			tmin = min(max(tmin, wmin), tmax)

		#	Degenerate Grid Cases
			if L <= 1 || tmin ≈ tmax
				return [tmax]
			end

		#	Build Log-Spaced Grid
			log10t = range(log10(tmin), log10(tmax), length=L)
			return 10.0 .^ collect(log10t)
	end

#	Helper Function for triad_census: Threshold Weighted Adjacency at τ (In-Place)
	function _threshold_to_binary!(A::SparseMatrixCSC{Float64,Int}, tau::Float64)
		"""
		Args:
			A::SparseMatrixCSC{Float64,Int}: weighted adjacency (modified in-place)
			tau::Float64: threshold (keep edges with weight ≥ τ)
		Returns:
			SparseMatrixCSC{Float64,Int}: 0/1 per-direction; self-loops removed
		Notes:
			Sets entries < τ to 0; entries ≥ τ to 1. Drops self-loops and
			structural zeros after thresholding. Used by _prepare_binary_for_mode
			to build per-τ binary matrices in the layered census.
		"""

		#	Iterate Over CSC Nonzeros
			vals = nonzeros(A)
			rows = rowvals(A)
			n    = size(A, 1)
			@inbounds for j in 1:size(A, 2)
				for idx in nzrange(A, j)
					i = rows[idx]
					v = vals[idx]
					vals[idx] = (i != j && v >= tau) ? 1.0 : 0.0
				end
			end

		#	Drop Zeros and Return
			dropzeros!(A)
			return A
	end

#	Helper Function for triad_census: Canonicalize Undirected Binary Matrix
	function _canonicalize_undirected_binary(A::SparseMatrixCSC{Float64,Int})
		"""
		Args:
			A::SparseMatrixCSC: intended undirected 0/1 adjacency (may be asymmetric in storage)
		Returns:
			SparseMatrixCSC{Float64,Int}: binary, zero-diagonal, symmetric, canonical sparsity
		Notes:
			Unions A and A', gathers unordered edge set {(i,j), i<j}, then mirrors
			to both sides. Ensures identical nnz / sparsity pattern for logically
			equivalent inputs — important for the undirected census so that two
			matrices representing the same undirected graph produce identical
			triad counts regardless of input storage convention.
		"""

		#	Validation
			n = size(A, 1)
			@assert size(A, 2) == n "A must be square"

		#	Union with Transpose, Zero Diagonal
			U = max.(A, A')
			@inbounds for i in 1:n
				U[i, i] = 0.0
			end
			dropzeros!(U)

		#	Gather Unordered Pairs (i < j) That Are Present
			rows = rowvals(U)
			vals = nonzeros(U)
			pairs_i = Int[]
			pairs_j = Int[]

			@inbounds for j in 1:n
				for idx in nzrange(U, j)
					i = rows[idx]
					if (i < j) && (vals[idx] != 0.0)
						push!(pairs_i, i)
						push!(pairs_j, j)
					end
				end
			end

		#	Rebuild Symmetric 0/1 from Unordered Pairs
			I = Int[]
			J = Int[]
			V = Float64[]
			@inbounds for t in eachindex(pairs_i)
				i = pairs_i[t]
				j = pairs_j[t]
				push!(I, i); push!(J, j); push!(V, 1.0)  # i → j
				push!(I, j); push!(J, i); push!(V, 1.0)  # j → i
			end

		#	Return Canonical Symmetric 0/1
			return sparse(I, J, V, n, n)
	end

#	Helper Function for triad_census: Prepare Per-τ Binary Matrix for Chosen graph_type
	function _prepare_binary_for_mode(Aw::SparseMatrixCSC{Float64,Int},
										tau::Float64,
										graph_type::Symbol,
										reciprocity_collapse::Bool)
		"""
		Args:
			Aw::SparseMatrixCSC{Float64,Int}: weighted adjacency (directed)
			tau::Float64: threshold
			graph_type::Symbol: :directed or :undirected
			reciprocity_collapse::Bool: only used for :directed
		Returns:
			SparseMatrixCSC{Float64,Int}: binary simple matrix prepared for the BM kernel
		Notes:
			:directed → threshold per-direction to 0/1; optionally collapse
				reciprocity by max(A, A').
			:undirected → symmetrize weights by SUM first (Aw .+ Aw'), threshold
				once with a small epsilon, then canonicalize to eliminate
				sparsity-pattern differences from input storage variations.

			The sum-before-threshold convention for undirected aligns with
			s-core semantics: an unordered edge {i,j} carries the combined
			weight from both directions, and the threshold is applied to that
			summed weight.
		"""

		if graph_type === :directed
			#	Threshold Directed Weights to 0/1 (Per-Direction)
				A = copy(Aw)
				_threshold_to_binary!(A, tau)

			#	Optional Compatibility Collapse (Pajek-Like)
				if reciprocity_collapse
					A = max.(A, A')
					_make_directed_simple!(A)
				end

			#	Return Directed Binary
				return A

		elseif graph_type === :undirected
			#	Symmetrize Weights by SUM First (Aligns with s-Core Semantics)
				AU = Aw .+ Aw'
				n  = size(AU, 1)
				@inbounds for i in 1:n
					AU[i, i] = 0.0
				end
				dropzeros!(AU)

			#	Threshold Undirected Weights to 0/1 Once
				#	Use a small ε to tolerate floating-point rounding from the
				#	symmetrization sum; treats w_sum ≥ τ − ε as "keep this edge."
					ε    = max(eps(tau), 1e-12)
					vals = nonzeros(AU)
					rows = rowvals(AU)
					@inbounds for j in 1:n
						for idx in nzrange(AU, j)
							i = rows[idx]
							v = vals[idx]
							vals[idx] = (i != j && v + ε >= tau) ? 1.0 : 0.0
						end
					end
					dropzeros!(AU)

			#	Canonicalize to Eliminate Sparsity-Pattern Differences
				AU = _canonicalize_undirected_binary(AU)

			#	Return Undirected Binary
				return AU

		else
			throw(ArgumentError("graph_type must be :directed or :undirected"))
		end
	end

#	Helper Function for triad_census: Convert 16-Class Count Vector to Density by nC3
	function _to_density_16!(counts::Vector{Int}, n::Int)
		"""
		Args:
			counts::Vector{Int}: 16 triad counts in DL order
			n::Int: number of nodes
		Returns:
			Vector{Float64}: densities (count divided by C(n,3))
		Notes:
			Returns zeros if n < 3 (no triads possible). Otherwise returns
			Float64.(counts) ./ nC3 where nC3 = n*(n-1)*(n-2)/6.

			The trailing ! is a holdover convention from a prior version that
			modified in place; the current implementation allocates a new
			Vector{Float64} and returns it. Kept for caller compatibility.
		"""

		#	Compute Total Possible Triads
			total_triads = n < 3 ? 0 : (n * (n - 1) * (n - 2)) ÷ 6

		#	Handle Trivial Case
			if total_triads == 0
				return zeros(Float64, length(counts))
			end

		#	Compute Densities
			return Float64.(counts) ./ total_triads
	end

#	Helper Function for triad_census: AUMC over log10(τ) via Trapezoid Rule
	function _aumc_logtau(tau::Vector{Float64}, y::Vector{Float64})
		"""
		Args:
			tau::Vector{Float64}: τ grid (strictly increasing)
			y::Vector{Float64}: motif density at each τ
		Returns:
			Float64: area under y vs log10(τ) via trapezoidal integration
		Notes:
			Integrates the motif-density curve in log-τ space. Used by
			_triad_census_layered to produce per-motif summary statistics
			(AUMC_density) that capture the cumulative presence of a triad
			class across the τ grid.

			Returns 0.0 for grids with fewer than 2 points (cannot form a
			trapezoid).
		"""

		#	Validation
			@assert length(tau) == length(y) "tau and y must have equal length"

		#	Trivial Case
			if length(tau) < 2
				return 0.0
			end

		#	Trapezoidal Sum in log10(τ) Space
			xt  = log10.(tau)
			acc = 0.0
			@inbounds for k in 1:length(tau) - 1
				h = xt[k + 1] - xt[k]
				acc += 0.5 * h * (y[k] + y[k + 1])
			end
			return acc
	end

#	Helper Function for recommend_L: Estimate τ-Bounds from Observed Weights (Quantile Fallback)
	function _estimate_tau_bounds(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
									graph_type::Symbol=:directed,
									lo::Float64=0.01,
									hi::Float64=0.99)
		"""
		Args:
			edges::DataFrame: expects :src, :dst, :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			graph_type::Symbol: :directed or :undirected
			lo::Float64: lower quantile for tau_min (default 0.01)
			hi::Float64: upper quantile for tau_max (default 0.99)
		Returns:
			NamedTuple: (tau_min::Float64, tau_max::Float64)
		Notes:
			Quantile-based τ bound estimation. Used by recommend_L as the
			fallback path when the analytic triangle-decay derivation cannot
			produce valid bounds (e.g., degenerate weight distribution or no
			triangles in the graph).

			For :undirected, τ thresholds apply to (W + W'), so the weight
			distribution is computed on the summed symmetric matrix before
			quantiling. For :directed, the distribution is taken over directed
			weights as-is.

			The lower bound is clipped from below by eps() for numerical safety.

			Degenerate cases (no positive weights) return (tau_min=1.0, tau_max=1.0).
		"""

		#	Validation
			@assert graph_type in (:directed, :undirected) "graph_type must be :directed or :undirected"

		#	Build Weighted Adjacency
			Aw, _, _ = isnothing(nodes) ?
				_graph_to_sparse_matrix(edges; weighted=true) :
				_graph_to_sparse_matrix(edges; nodes=nodes, weighted=true)

		#	Extract Weight Distribution Per graph_type
			if graph_type === :undirected
				AU = Aw .+ Aw'
				n  = size(AU, 1)
				@inbounds for i in 1:n
					AU[i, i] = 0.0
				end
				dropzeros!(AU)
				w = collect(nonzeros(AU))
			else
				w = collect(nonzeros(Aw))
			end

		#	Filter to Finite Positive Weights
			w = w[isfinite.(w) .& (w .> 0.0)]

		#	Degenerate Case
			if isempty(w)
				return (tau_min = 1.0, tau_max = 1.0)
			end

		#	Compute Bounds
			return (tau_min = max(eps(), quantile(w, lo)),
					tau_max = quantile(w, hi))
	end

#	Helper Function for recommend_L: Quick Heuristic for L (Points-Per-Decade)
	function _suggest_L_quick(tau_min::Float64,
								tau_max::Float64;
								points_per_decade::Int=8,
								L_min::Int=8,
								L_max::Int=64)
		"""
		Args:
			tau_min::Float64: lower τ bound
			tau_max::Float64: upper τ bound
			points_per_decade::Int: target density in log10-space (default 8)
			L_min::Int: lower clamp for returned L (default 8)
			L_max::Int: upper clamp for returned L (default 64)
		Returns:
			Int: suggested L
		Notes:
			Computes L ≈ ceil(points_per_decade * log10(tau_max / tau_min)),
			clamped to [L_min, L_max]. This gives a τ grid that has roughly
			`points_per_decade` thresholds per order of magnitude in weight,
			which is enough to resolve where motif densities change without
			wasting work on a finer grid.

			Used by recommend_L to compute L from the derived τ bounds.

			Falls back to L_min when tau_max ≤ tau_min (degenerate or single-
			value weight distribution).
		"""

		#	Compute Log10 Span
			ratio   = tau_max <= tau_min ? 1.0 : (tau_max / tau_min)
			decades = log10(ratio)

		#	Map to Suggested L
			L = ceil(Int, points_per_decade * max(decades, 0.0))

		#	Clamp to [L_min, L_max]
			return clamp(max(L, L_min), L_min, L_max)
	end

#	Helper Function for recommend_L: Fast Triangle Counting on a Thresholded Graph
	function _count_triangles_at_tau(Aw::SparseMatrixCSC{Float64,Int},
										tau::Float64,
										graph_type::Symbol,
										reciprocity_collapse::Bool)
		"""
		Args:
			Aw::SparseMatrixCSC{Float64,Int}: weighted adjacency (directed)
			tau::Float64: threshold to apply
			graph_type::Symbol: :directed or :undirected
			reciprocity_collapse::Bool: directed-only; collapse mutual arcs
		Returns:
			Int: number of closed triangles in the thresholded undirected graph
		Notes:
			Threshold the weighted adjacency at tau, symmetrize for triangle
			counting, and run a fast BM-style triangle enumeration. The
			triangle count is intentionally computed on the underlying
			undirected graph regardless of graph_type — for τ-bound estimation
			we care about triadic structure, not the orientation of arcs
			within triads. This is cheap (O(sum d_v^2)) and consistent across
			directed and undirected inputs.

			Returns 0 if the thresholded graph has no edges.
		"""

		#	Threshold to Binary (Per-Direction)
			A = copy(Aw)
			_threshold_to_binary!(A, tau)

		#	Optional Reciprocity Collapse for Directed Graphs
			if graph_type === :directed && reciprocity_collapse
				A = max.(A, A')
				_make_directed_simple!(A)
			end

		#	Symmetrize for Triangle Counting (Both Directions Treated Equally)
			#	Triangle counting cares only about whether there's an edge,
			#	not direction. max(A, A') gives the underlying undirected graph.
				U = max.(A, A')
				n = size(U, 1)
				@inbounds for i in 1:n
					U[i, i] = 0.0
				end
				dropzeros!(U)

		#	Quick Out If No Edges
			if nnz(U) == 0
				return 0
			end

		#	Build Sorted Neighbor Lists (CSC Column Indices Per Row)
			#	For each node, collect the sorted set of neighbors.
			#	This makes the inner intersection step efficient via merge.
				neighbors = Vector{Vector{Int}}(undef, n)
				@inbounds for j in 1:n
					nb = rowvals(U)[nzrange(U, j)]
					neighbors[j] = sort(unique(nb))
				end

		#	Count Triangles via BM-Style Enumeration
			#	For each edge (i, j) with i < j, count common neighbors k with
			#	k > j. Each triangle {i, j, k} with i < j < k is counted exactly
			#	once.
				triangle_count = 0
				@inbounds for i in 1:n - 1
					nbrs_i = neighbors[i]
					for j_idx in eachindex(nbrs_i)
						j = nbrs_i[j_idx]
						if j <= i
							continue
						end
						#	Intersect neighbors[i] and neighbors[j], counting
						#	common k > j via merge on sorted lists.
							nbrs_j = neighbors[j]
							p = 1  # pointer into nbrs_i
							q = 1  # pointer into nbrs_j
							while p <= length(nbrs_i) && q <= length(nbrs_j)
								a = nbrs_i[p]
								b = nbrs_j[q]
								if a == b
									if a > j
										triangle_count += 1
									end
									p += 1
									q += 1
								elseif a < b
									p += 1
								else
									q += 1
								end
							end
					end
				end

		#	Return
			return triangle_count
	end

#	Helper Function for recommend_L: Build (τ, T(τ)) Profile Across Exploratory Grid
	function _triangle_profile(edges::DataFrame;
								nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
								graph_type::Symbol=:directed,
								reciprocity_collapse::Bool=false,
								n_exploratory::Int=16,
								verbose::Bool=false)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			graph_type::Symbol: :directed or :undirected
			reciprocity_collapse::Bool: directed-only
			n_exploratory::Int: number of log-spaced τ points to evaluate (default 16)
			verbose::Bool: print τ values being evaluated (default false)
		Returns:
			NamedTuple: (profile::DataFrame, T_max::Int, weight_min::Float64, weight_max::Float64)
				profile columns: [:tau, :triangle_count]
				T_max: triangle count in the unweighted graph (i.e., at minimum τ)
				weight_min, weight_max: bounds of the positive-weight distribution
		Notes:
			Builds the (τ, T(τ)) exploratory profile that recommend_L uses to
			locate the analytic τ_min and τ_max. The grid is log-spaced over
			[weight_min, weight_max], excluding any zero weights.

			T_max is computed at τ = weight_min, which retains the maximum
			number of edges possible from the weighted adjacency. For graphs
			where the minimum positive weight is the natural floor (e.g.,
			Marvel where weight=2 after thresholding), this matches the
			unweighted triangle count exactly.

			Triangle counting is done via _count_triangles_at_tau, which is
			O(sum d_v^2) per τ. For Marvel-scale (~6.5K nodes, 77K edges),
			each evaluation takes well under a second.

			The grid uses log-spaced points to bracket the triangle decay
			curve efficiently. 16 points typically gives sub-decade resolution
			for any realistic weight distribution.
		"""

		#	Build Weighted Adjacency Once
			Aw, _, _ = isnothing(nodes) ?
				_graph_to_sparse_matrix(edges; weighted=true) :
				_graph_to_sparse_matrix(edges; nodes=nodes, weighted=true)

		#	Identify Positive-Weight Range
			#	For undirected, use summed symmetric weights; matches how
			#	_prepare_binary_for_mode thresholds undirected graphs.
				if graph_type === :undirected
					AU = Aw .+ Aw'
					n  = size(AU, 1)
					@inbounds for i in 1:n
						AU[i, i] = 0.0
					end
					dropzeros!(AU)
					weights_for_grid = collect(nonzeros(AU))
				else
					weights_for_grid = collect(nonzeros(Aw))
				end

			weights_for_grid = weights_for_grid[isfinite.(weights_for_grid) .& (weights_for_grid .> 0.0)]
			if isempty(weights_for_grid)
				#	Degenerate case
					return (profile      = DataFrame(tau = Float64[], triangle_count = Int[]),
							T_max        = 0,
							weight_min   = 1.0,
							weight_max   = 1.0)
			end

			weight_min = minimum(weights_for_grid)
			weight_max = maximum(weights_for_grid)

		#	Build Log-Spaced Exploratory Grid
			if weight_max <= weight_min
				#	All weights equal — degenerate grid, single τ
					tau_grid = [weight_min]
				else
					log_min  = log10(max(eps(), weight_min))
					log_max  = log10(weight_max)
					tau_grid = 10.0 .^ collect(range(log_min, log_max, length=n_exploratory))
				end

		#	Compute T_max at the Floor τ (Unweighted-Equivalent Triangle Count)
			T_max = _count_triangles_at_tau(Aw, tau_grid[1], graph_type, reciprocity_collapse)

		#	Build Profile
			profile_rows = Vector{NamedTuple}(undef, length(tau_grid))
			for (k, τ) in pairs(tau_grid)
				if verbose
					println("    [_triangle_profile] τ = $(round(τ, sigdigits=4))...")
				end
				T_at_tau = _count_triangles_at_tau(Aw, τ, graph_type, reciprocity_collapse)
				profile_rows[k] = (tau = τ, triangle_count = T_at_tau)
			end

		#	Return
			return (profile    = DataFrame(profile_rows),
					T_max      = T_max,
					weight_min = weight_min,
					weight_max = weight_max)
	end

#	Helper Function for recommend_L: Apply Analytic τ-Bound Rules to Triangle Profile
	function _analytic_tau_bounds(profile::DataFrame,
									T_max::Int;
									frac_keep::Float64 = 1.0 / ℯ,
									T_min_floor::Int = 9,
									verbose::Bool = false)
		"""
		Args:
			profile::DataFrame: (tau, triangle_count) from _triangle_profile
			T_max::Int: maximum triangle count (typically at τ_min of the grid)
			frac_keep::Float64: τ_min is the smallest τ where T(τ) <= frac_keep * T_max
				(default 1/e ≈ 0.368, the e-folding decay scale)
			T_min_floor::Int: minimum triangle count for τ_max to be considered
				meaningful (default 9, corresponding to 3σ Poisson detection)
			verbose::Bool: print decision points (default false)
		Returns:
			NamedTuple: (tau_min::Float64, tau_max::Float64, T_at_tau_min::Int,
			             T_at_tau_max::Int, valid::Bool)
				valid: true if tau_max > tau_min (meaningful bounds), false otherwise
		Notes:
			Applies two analytic principles to the triangle decay profile:

			(1) τ_min via e-folding: identifies where the triangle count has
			    decayed by a factor of e ≈ 2.72. Below this τ, the threshold
			    has not yet "bitten" — we're essentially looking at the
			    unweighted graph. Above the e-fold scale, we're in the
			    threshold-sensitive regime where the layered census carries
			    meaningful information.

			(2) τ_max via Poisson detection threshold: triangle counts behave
			    Poisson-like under random thresholding, so √T is the standard
			    deviation. T = 9 gives 3σ signal-to-noise — the minimum for
			    statistical "detection" of triadic structure above noise.

			Both choices are derived rather than heuristic. The frac_keep
			default of 1/e is the natural decay scale; T_min_floor = 9 is the
			3σ detection threshold.

			Returns valid=false if the profile doesn't support meaningful
			bounds (e.g., T_max already too low, or the curve doesn't decay
			enough across the grid). In that case the caller should fall back
			to weight-distribution quantiles or raise an error.
		"""

		#	Validation
			@assert nrow(profile) > 0 "profile must contain at least one row"
			@assert T_max >= 0 "T_max must be non-negative"

		#	Degenerate Case: No Triangles Anywhere
			if T_max == 0 || all(profile.triangle_count .== 0)
				return (tau_min      = profile.tau[1],
						tau_max      = profile.tau[end],
						T_at_tau_min = 0,
						T_at_tau_max = 0,
						valid        = false)
			end

		#	τ_min: Smallest τ Where T(τ) <= T_max * frac_keep
			#	I.e., the smallest τ where we have one e-fold of decay.
			#	If no τ in the profile reaches that decay, set τ_min to the
			#	grid floor (we never see meaningful decay).
				T_threshold_min = frac_keep * T_max
				tau_min_idx     = findfirst(t -> t <= T_threshold_min, profile.triangle_count)
				if tau_min_idx === nothing
					#	No decay below threshold; default to grid floor
						tau_min      = profile.tau[1]
						T_at_tau_min = profile.triangle_count[1]
				else
					tau_min      = profile.tau[tau_min_idx]
					T_at_tau_min = profile.triangle_count[tau_min_idx]
				end

		#	τ_max: Largest τ Where T(τ) >= T_min_floor
			#	I.e., the upper edge of the statistically meaningful range.
				tau_max_idx = findlast(t -> t >= T_min_floor, profile.triangle_count)
				if tau_max_idx === nothing
					#	Profile never exceeds T_min_floor; invalid (no signal)
						return (tau_min      = profile.tau[1],
								tau_max      = profile.tau[end],
								T_at_tau_min = T_at_tau_min,
								T_at_tau_max = 0,
								valid        = false)
				end
				tau_max      = profile.tau[tau_max_idx]
				T_at_tau_max = profile.triangle_count[tau_max_idx]

		#	Sanity Check: tau_max Must Exceed tau_min
			if tau_max <= tau_min
				return (tau_min      = tau_min,
						tau_max      = tau_max,
						T_at_tau_min = T_at_tau_min,
						T_at_tau_max = T_at_tau_max,
						valid        = false)
			end

		#	Verbose Reporting
			if verbose
				println("    _analytic_tau_bounds:")
				println("      T_max:           $T_max")
				println("      frac_keep:       $(round(frac_keep, sigdigits=4)) (1/e)")
				println("      T_min_floor:     $T_min_floor (3σ Poisson detection)")
				println("      τ_min:           $(round(tau_min, sigdigits=4)) (T = $T_at_tau_min)")
				println("      τ_max:           $(round(tau_max, sigdigits=4)) (T = $T_at_tau_max)")
			end

		#	Return
			return (tau_min      = tau_min,
					tau_max      = tau_max,
					T_at_tau_min = T_at_tau_min,
					T_at_tau_max = T_at_tau_max,
					valid        = true)
	end

#	Helper Function for recommend_L: Empirical Stability Scan (AUMC-Based)
	function _select_L_by_stability_empirical(edges::DataFrame;
												nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
												graph_type::Symbol=:directed,
												reciprocity_collapse::Bool=false,
												tau_min::Union{Float64,Symbol}=:auto,
												tau_max::Union{Float64,Symbol}=:auto,
												L_grid::Vector{Int} = [8, 12, 16, 24, 32, 48, 64],
												tol::Float64 = 1e-3,
												parallel::Bool = false,
												verbose::Bool = false,
												show_progress::Bool = false,
												inner_show_progress::Bool = false)
		"""
		Args:
			edges, nodes, graph_type, reciprocity_collapse: graph specification
			tau_min, tau_max: τ bounds or :auto (quantile-derived via _estimate_tau_bounds)
			L_grid: candidate L values to evaluate
			tol: AUMC stability tolerance (default 1e-3)
			parallel: parallelize candidates (default false)
			verbose, show_progress, inner_show_progress: diagnostic flags
		Returns:
			NamedTuple: (L_best::Int, table::DataFrame, census_at_L_best::NamedTuple)
		Notes:
			Empirical AUMC stability scan. Used by recommend_L when the
			auto-selection rule (or user override) chooses the empirical path.
			Runs triad_census(weighted=true) at each candidate L, computes
			the 16-class AUMC vector, and picks the smallest L where AUMC
			stabilizes within tol.

			The empirical method is the appropriate choice when:
			- T_max is below the asymptotic regime (small networks where
			  individual triangles carry semantic meaning), OR
			- The weight distribution is too narrow for the analytic method's
			  triangle-decay derivation to discriminate, OR
			- The triangle decay profile is step-function rather than
			  graduated, leaving the analytic e-fold and 3σ thresholds with
			  no meaningful resolution.

			Cached per-candidate census so the layered result at L_best is
			returned without re-running. For Marvel-scale weighted runs
			(which take the analytic path), this caching saves nothing —
			but for small-to-medium networks taking the empirical path it
			saves a redundant triad_census call downstream.

			Default mode (parallel=false): serial candidates with inner τ-loop
			threading. Each candidate gets all threads on its τ loop, and
			early termination kicks in at the first L meeting the stability
			criterion. Best choice for typical 5-candidate scans on 8-thread
			hosts.

			Parallel-candidate mode (parallel=true): all candidates concurrent
			with serial inner census. Use when L_grid has candidates ≥ thread
			count (rare).
		"""

		#	Validation
			@assert graph_type in (:directed, :undirected) "graph_type must be :directed or :undirected"
			if graph_type == :undirected
				@assert !reciprocity_collapse "reciprocity_collapse applies only when graph_type == :directed"
			end

		#	Resolve τ Bounds (Auto or Explicit)
			local_tau_min = 0.0
			local_tau_max = 0.0
			if tau_min === :auto || tau_max === :auto
				tb            = _estimate_tau_bounds(edges; nodes=nodes, graph_type=graph_type)
				local_tau_min = tau_min === :auto ? tb.tau_min : Float64(tau_min)
				local_tau_max = tau_max === :auto ? tb.tau_max : Float64(tau_max)
			else
				local_tau_min = Float64(tau_min)
				local_tau_max = Float64(tau_max)
			end

		#	Guard
			if !(local_tau_max >= local_tau_min)
				local_tau_max = local_tau_min
			end

		#	DL Labels
			labels = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
			          "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

		#	Pre-Allocate Per-Candidate Storage
			Ncand          = length(L_grid)
			aumc_by_cand   = Vector{Vector{Float64}}(undef, Ncand)
			row_by_cand    = Vector{NamedTuple}(undef, Ncand)
			census_by_cand = Vector{NamedTuple}(undef, Ncand)

		#	Configure Progress Bar
			use_threads = parallel && Threads.nthreads() > 1 && Ncand > 1
			prog        = show_progress ?
			              Progress(Ncand,
			                       desc = "  [empirical] candidates ($(string(graph_type)), " *
			                              (use_threads ? "$(Threads.nthreads()) threads" : "serial") * ")",
			                       enabled = true) :
			              nothing
			prog_lock   = ReentrantLock()

		#	Worker Closure
			function _run_one_candidate(L::Int; inner_parallel::Bool)
				res = triad_census(edges;
									nodes                = nodes,
									weighted             = true,
									graph_type           = graph_type,
									reciprocity_collapse = reciprocity_collapse,
									L                    = L,
									tau_min              = local_tau_min,
									tau_max              = local_tau_max,
									parallel             = inner_parallel,
									show_progress        = inner_show_progress)
				s    = res.summary
				aumc = [begin
							v = s[s.triad .== lab, :AUMC_density]
							isempty(v) ? 0.0 : v[1]
						end for lab in labels]
				v300     = s[s.triad .== "300", :AUMC_density]
				aumc_300 = isempty(v300) ? 0.0 : v300[1]
				v003     = s[s.triad .== "003", :AUMC_density]
				aumc_003 = isempty(v003) ? 0.0 : v003[1]
				row = (L             = L,
						max_abs_delta = NaN,
						aumc_300      = aumc_300,
						aumc_003      = aumc_003,
						aumc_total    = sum(aumc))
				return (aumc = aumc, row = row, census = res)
			end

		#	Verbose Header
			if verbose
				mode_str = use_threads ? "parallel candidates" : "serial w/ inner threading"
				println("    [empirical] candidates = $L_grid, mode = $mode_str, tol = $tol")
				println("    [empirical] τ bounds: [$local_tau_min, $local_tau_max]")
			end

		#	Run Candidate Scan
			if use_threads
				Threads.@threads :static for k in 1:Ncand
					verbose && println("      [empirical, thread $(Threads.threadid())] L = $(L_grid[k])...")
					out                = _run_one_candidate(L_grid[k]; inner_parallel=false)
					aumc_by_cand[k]    = out.aumc
					row_by_cand[k]     = out.row
					census_by_cand[k]  = out.census
					if show_progress
						lock(prog_lock) do; next!(prog); end
					end
				end

				L_best     = last(L_grid)
				L_best_idx = Ncand
				prev_aumc  = nothing
				final_rows = NamedTuple[]
				for k in 1:Ncand
					row_k    = row_by_cand[k]
					aumc_k   = aumc_by_cand[k]
					mad      = prev_aumc === nothing ? Inf : maximum(abs.(aumc_k .- prev_aumc))
					push!(final_rows, (L = row_k.L, max_abs_delta = mad,
										aumc_300 = row_k.aumc_300, aumc_003 = row_k.aumc_003,
										aumc_total = row_k.aumc_total))
					if prev_aumc !== nothing && mad < tol && L_best == last(L_grid)
						L_best     = L_grid[k]
						L_best_idx = k
					end
					prev_aumc = aumc_k
				end
			else
				prev_aumc  = nothing
				final_rows = NamedTuple[]
				L_best     = last(L_grid)
				L_best_idx = Ncand
				for (k, L) in pairs(L_grid)
					verbose && println("      [empirical, serial] L = $L...")
					out                = _run_one_candidate(L; inner_parallel=true)
					aumc_k             = out.aumc
					row_k              = out.row
					census_by_cand[k]  = out.census
					aumc_by_cand[k]    = aumc_k
					row_by_cand[k]     = row_k
					mad                = prev_aumc === nothing ? Inf : maximum(abs.(aumc_k .- prev_aumc))
					push!(final_rows, (L = row_k.L, max_abs_delta = mad,
										aumc_300 = row_k.aumc_300, aumc_003 = row_k.aumc_003,
										aumc_total = row_k.aumc_total))
					if show_progress
						next!(prog)
					end
					if prev_aumc !== nothing && mad < tol
						L_best     = L
						L_best_idx = k
						break
					end
					prev_aumc = aumc_k
				end
			end

			if verbose
				println("    [empirical] L_best = $L_best (candidate idx $L_best_idx)")
			end

			return (L_best           = L_best,
					table            = DataFrame(final_rows),
					census_at_L_best = census_by_cand[L_best_idx])
	end

#	Helper Function for recommend_L: Decide Between Analytic and Empirical Method
	function _select_recommendation_method(profile::DataFrame,
											T_max::Int,
											weight_min::Float64,
											weight_max::Float64;
											T_max_threshold::Int = 10000,
											T_max_floor::Int = 100,
											weight_decades_min::Float64 = 1.0,
											decay_range_points_min::Int = 3,
											verbose::Bool = false)
		"""
		Args:
			profile::DataFrame: (tau, triangle_count) from _triangle_profile
			T_max::Int: maximum triangle count in the unweighted graph
			weight_min, weight_max::Float64: bounds of the weight distribution
			T_max_threshold::Int: T_max at or above which analytic is preferred
				(default 10000, the strong asymptotic regime)
			T_max_floor::Int: T_max below which empirical is forced (default 100,
				the rare-structure regime)
			weight_decades_min::Float64: minimum log10(weight_max/weight_min) for
				analytic to be considered (default 1.0)
			decay_range_points_min::Int: minimum grid points where T(τ) is in
				(1, T_max/2) for analytic to be considered (default 3)
			verbose::Bool: print decision rationale (default false)
		Returns:
			NamedTuple: (method::Symbol, reason::String, diagnostics::NamedTuple)
				method ∈ (:analytic, :empirical)
				reason: human-readable explanation
				diagnostics: (T_max, weight_decades, decay_range_points)
		Notes:
			Selects between the analytic (triangle-decay Poisson detection) and
			empirical (AUMC stability scan with quantile bounds) methods based
			on three criteria computed from the network's triangle profile:

			(1) T_max regime. The analytic method's 3σ Poisson detection
			    threshold assumes triangle counts behave Poisson-like at the
			    τ_max cutoff. This requires T_max in the asymptotic regime
			    (T_max ≥ 10,000 by default). For T_max < 100, the rare-
			    structure regime dominates and the analytic method is too
			    aggressive — every triangle carries semantic weight.

			(2) Weight distribution span. The analytic method needs at least
			    one decade of weight variation to have a meaningful decay
			    profile to operate on. Narrow weight distributions (e.g.,
			    Moreno with weight ∈ {1, 2}) give the analytic method nothing
			    to discriminate.

			(3) Decay profile shape. The analytic method's e-fold and 3σ
			    thresholds need a sufficiently graduated decay. Step-function
			    decays (sharp drop then plateau) leave the analytic method
			    choosing between adjacent grid points with no real signal.
			    The decay_range_points metric counts grid points where T(τ)
			    is in the meaningful decay band (1 < T < T_max/2).

			Decision logic:
			- T_max ≥ T_max_threshold → analytic (asymptotic regime confirmed)
			- T_max < T_max_floor → empirical (rare-structure regime)
			- Otherwise (middle zone): analytic requires BOTH weight_decades ≥
			  weight_decades_min AND decay_range_points ≥ decay_range_points_min;
			  empirical otherwise.

			The criteria are derived from the validation suite results on
			Balikatan (T_max=3015, middle zone, analytic works), Moreno (T_max=307,
			narrow weights, empirical needed), Scotland (T_max=269, step-decay,
			empirical needed), and Marvel (T_max in millions, analytic essential).
		"""

		#	Compute Diagnostics
			weight_decades = weight_max > weight_min ?
			                 log10(weight_max / weight_min) : 0.0

			#	Decay range: grid points where T(τ) is in (1, T_max/2)
			#	Captures the "graduated decay" regime where the analytic
			#	method's thresholds have meaningful resolution.
				upper_band = T_max / 2.0
				decay_range_points = sum(@. (profile.triangle_count > 1) &
											  (profile.triangle_count < upper_band))

		#	Decision Logic
			method = :empirical
			reason = ""
			if T_max >= T_max_threshold
				method = :analytic
				reason = "T_max = $T_max ≥ $T_max_threshold (strong Poisson asymptotic regime)"
			elseif T_max < T_max_floor
				method = :empirical
				reason = "T_max = $T_max < $T_max_floor (rare-structure regime; every triangle semantic)"
			else
				#	Middle Zone: Require Both Secondary Criteria for Analytic
					if weight_decades < weight_decades_min
						method = :empirical
						reason = "T_max in middle zone; weight range = " *
						         "$(round(weight_decades, sigdigits=3)) decades " *
						         "< $weight_decades_min (narrow weight distribution)"
					elseif decay_range_points < decay_range_points_min
						method = :empirical
						reason = "T_max in middle zone; decay range = " *
						         "$decay_range_points grid points " *
						         "< $decay_range_points_min (step-function decay)"
					else
						method = :analytic
						reason = "T_max = $T_max in middle zone; weight span and decay shape " *
						         "support analytic (decades=$(round(weight_decades, sigdigits=3)), " *
						         "decay_points=$decay_range_points)"
					end
			end

		#	Verbose Reporting
			if verbose
				println("    _select_recommendation_method:")
				println("      T_max:                  $T_max")
				println("      weight_decades:         $(round(weight_decades, sigdigits=4))")
				println("      decay_range_points:     $decay_range_points")
				println("      Selected method:        $method")
				println("      Reason:                 $reason")
			end

		return (method      = method,
				reason      = reason,
				diagnostics = (T_max               = T_max,
								weight_decades     = weight_decades,
								decay_range_points = decay_range_points))
	end

#	Triad Census: Recommend L and τ Bounds with Auto-Selected Method (Analytic or Empirical)
	function recommend_L(edges::DataFrame;
							nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
							graph_type::Symbol=:directed,
							reciprocity_collapse::Bool=false,
							tau_min::Union{Float64,Symbol}=:auto,
							tau_max::Union{Float64,Symbol}=:auto,
							method::Symbol=:auto,
							points_per_decade::Int=8,
							L_min::Int=8,
							L_max::Int=64,
							n_exploratory::Int=16,
							frac_keep::Float64=1.0 / ℯ,
							T_min_floor::Int=9,
							tol::Float64=1e-3,
							T_max_threshold::Int=10000,
							T_max_floor::Int=100,
							weight_decades_min::Float64=1.0,
							decay_range_points_min::Int=3,
							parallel::Bool=false,
							verbose::Bool=false,
							show_progress::Bool=true,
							inner_show_progress::Bool=false)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			graph_type::Symbol: :directed or :undirected
			reciprocity_collapse::Bool: directed-only (default false)
			tau_min, tau_max::Union{Float64,Symbol}: τ bounds or :auto (default :auto).
				When both are explicit Floats, both methods are short-circuited
				and L is computed directly from the user's range.
			method::Symbol: :auto, :analytic, or :empirical (default :auto).
				:auto applies the decision rule from _select_recommendation_method;
				:analytic and :empirical force the respective path regardless of
				network characteristics.
			points_per_decade::Int: heuristic for L computation (default 8)
			L_min, L_max::Int: clamps for returned L (defaults 8, 64)
			n_exploratory::Int: log-spaced points in triangle profile (default 16)
			frac_keep::Float64: e-fold threshold for analytic τ_min (default 1/e)
			T_min_floor::Int: 3σ Poisson detection threshold for analytic τ_max
				(default 9)
			tol::Float64: AUMC stability tolerance for empirical method (default 1e-3)
			T_max_threshold::Int: T_max at or above which auto-selects analytic
				(default 10000, strong asymptotic regime)
			T_max_floor::Int: T_max below which auto-selects empirical (default 100,
				rare-structure regime)
			weight_decades_min::Float64: minimum log10(weight span) for analytic
				in the middle zone (default 1.0)
			decay_range_points_min::Int: minimum grid points in (1, T_max/2) for
				analytic in the middle zone (default 3)
			parallel::Bool: parallelize empirical candidate scan (default false)
			verbose::Bool: print derivation steps (default false)
			show_progress::Bool: progress bar for empirical candidates (default true)
			inner_show_progress::Bool: per-τ progress bar within each empirical
				candidate (default false)
		Returns:
			NamedTuple: (L::Int, tau_min::Float64, tau_max::Float64,
			             method::Symbol, method_reason::String, T_max::Int,
			             profile::DataFrame, scan::Union{DataFrame,Nothing},
			             census::Union{NamedTuple,Nothing}, valid::Bool)
				method: which method was used (:analytic, :empirical, or :user_supplied)
				method_reason: human-readable explanation
				T_max: unweighted triangle count (-1 if :user_supplied)
				profile: triangle decay profile (empty if :user_supplied)
				scan: empirical AUMC stability table (nothing if not :empirical)
				census: cached layered census at L_best (nothing if not :empirical)
				valid: true if recommendation was derivable
		Notes:
			Two-method recommend_L with automatic method selection. The two paths:

			Analytic method (triangle-decay Poisson detection):
			  Derives τ bounds from the triangle profile using the e-fold scale
			  (τ_min) and 3σ Poisson detection threshold (τ_max). Fast — seconds
			  even on Marvel-scale. Appropriate when T_max is large enough for
			  Poisson asymptotics to apply cleanly.

			Empirical method (AUMC stability scan):
			  Runs the layered census at multiple candidate L values with
			  quantile-derived τ bounds; picks the smallest L where AUMC
			  stabilizes within tol. Slower but captures the rare-structure
			  tail. Appropriate for small networks or networks with narrow
			  weight distributions.

			Auto-selection (method = :auto, default):
			  - T_max ≥ T_max_threshold → analytic
			  - T_max < T_max_floor → empirical
			  - Middle zone: analytic if both weight span ≥ 1 decade AND
			    decay profile has at least 3 grid points in (1, T_max/2);
			    empirical otherwise.

			User override:
			  - method = :analytic → force analytic
			  - method = :empirical → force empirical

			User-supplied τ bounds (both Floats) short-circuit both methods —
			L is computed via _suggest_L_quick from the user's bounds.

			Cost. Auto-selection computes the triangle profile (n_exploratory
			triangle counts, sub-second each on Marvel). The analytic path adds
			no further census work. The empirical path runs the full candidate
			scan (the main cost on large networks; minor on small networks).

			Cached census. The empirical path returns its layered census at
			L_best in the :census field, so downstream code can use rec.census
			directly without re-running triad_census. The analytic path returns
			:census = nothing (no census was needed); downstream code must
			call triad_census(L=rec.L, tau_min=rec.tau_min, tau_max=rec.tau_max).

			Validation. Calibrated on Balikatan (T_max=3015, middle zone with
			graduated decay → analytic), Moreno (T_max=307, step-function decay
			→ empirical), Scotland (T_max=269, step-function decay → empirical),
			and Marvel (T_max in millions → analytic). AUMC cosine similarity
			between methods ≈ 1.0 across calibration networks.
		"""

		#	Validation
			@assert method in (:auto, :analytic, :empirical) "method must be :auto, :analytic, or :empirical"

		#	Handle User-Supplied Bounds (Short-Circuit Both Methods)
			if !(tau_min === :auto) && !(tau_max === :auto)
				resolved_min = Float64(tau_min)
				resolved_max = Float64(tau_max)
				L            = _suggest_L_quick(resolved_min, resolved_max;
												points_per_decade=points_per_decade,
												L_min=L_min, L_max=L_max)
				return (L              = L,
						tau_min        = resolved_min,
						tau_max        = resolved_max,
						method         = :user_supplied,
						method_reason  = "Both τ bounds supplied by user; L computed from log span",
						T_max          = -1,
						profile        = DataFrame(tau = Float64[], triangle_count = Int[]),
						scan           = nothing,
						census         = nothing,
						valid          = true)
			end

		#	Verbose Header
			if verbose
				println("  recommend_L (method = $method)")
				println("    graph_type           = $graph_type")
				println("    reciprocity_collapse = $reciprocity_collapse")
			end

		#	Build Triangle Profile (Needed for Auto Selection and Analytic Path)
		#	Cheap on all network scales — n_exploratory triangle counts at
		#	O(sum d_v^2) each. Even on Marvel this runs in seconds.
			if verbose
				println("    Building triangle profile (n_exploratory = $n_exploratory)...")
			end
			t0             = time()
			profile_result = _triangle_profile(edges;
											nodes                = nodes,
											graph_type           = graph_type,
											reciprocity_collapse = reciprocity_collapse,
											n_exploratory        = n_exploratory,
											verbose              = false)
			t_profile = time() - t0
			if verbose
				println(@sprintf("    Triangle profile: %.3f s (T_max = %d)",
									t_profile, profile_result.T_max))
			end

		#	Select Method (Auto Mode) or Honor User Override
			selected_method = method
			method_reason   = ""
			if method === :auto
				sel = _select_recommendation_method(profile_result.profile,
													profile_result.T_max,
													profile_result.weight_min,
													profile_result.weight_max;
													T_max_threshold         = T_max_threshold,
													T_max_floor             = T_max_floor,
													weight_decades_min      = weight_decades_min,
													decay_range_points_min  = decay_range_points_min,
													verbose                 = verbose)
				selected_method = sel.method
				method_reason   = sel.reason
			else
				method_reason = "User-forced method = $method"
				if verbose
					println("    Method forced by user: $method")
				end
			end

		#	Dispatch to Selected Method
			if selected_method === :analytic
				#	--- Analytic Path ---
					if verbose
						println("    Running analytic τ-bound derivation...")
					end
					bounds = _analytic_tau_bounds(profile_result.profile,
													profile_result.T_max;
													frac_keep   = frac_keep,
													T_min_floor = T_min_floor,
													verbose     = verbose)

					#	Fall Back to Quantile Bounds if Analytic Derivation Failed
						if !bounds.valid
							if verbose
								println("    Analytic bounds not derivable; falling back to quantile heuristic.")
							end
							tb           = _estimate_tau_bounds(edges; nodes=nodes, graph_type=graph_type)
							resolved_min = tau_min === :auto ? tb.tau_min : Float64(tau_min)
							resolved_max = tau_max === :auto ? tb.tau_max : Float64(tau_max)
						else
							resolved_min = tau_min === :auto ? bounds.tau_min : Float64(tau_min)
							resolved_max = tau_max === :auto ? bounds.tau_max : Float64(tau_max)
						end

					L = _suggest_L_quick(resolved_min, resolved_max;
											points_per_decade=points_per_decade,
											L_min=L_min, L_max=L_max)

					if verbose
						println(@sprintf("    Analytic done: L = %d, τ ∈ [%.4g, %.4g]",
											L, resolved_min, resolved_max))
					end

					return (L              = L,
							tau_min        = resolved_min,
							tau_max        = resolved_max,
							method         = :analytic,
							method_reason  = method_reason,
							T_max          = profile_result.T_max,
							profile        = profile_result.profile,
							scan           = nothing,
							census         = nothing,
							valid          = bounds.valid)

			else
				#	--- Empirical Path ---
					if verbose
						println("    Running empirical AUMC stability scan...")
					end

					#	Resolve τ Bounds (Quantile-Derived if :auto)
						if tau_min === :auto || tau_max === :auto
							tb           = _estimate_tau_bounds(edges; nodes=nodes, graph_type=graph_type)
							resolved_min = tau_min === :auto ? tb.tau_min : Float64(tau_min)
							resolved_max = tau_max === :auto ? tb.tau_max : Float64(tau_max)
						else
							resolved_min = Float64(tau_min)
							resolved_max = Float64(tau_max)
						end

					#	Quick L Guess and Focused Candidate Grid
						L_guess = _suggest_L_quick(resolved_min, resolved_max;
													points_per_decade=points_per_decade,
													L_min=L_min, L_max=L_max)
						L_grid  = unique(sort(Int[max(L_min, div(L_guess, 2)),
												max(L_min, round(Int, 0.75 * L_guess)),
												L_guess,
												min(L_max, round(Int, 1.25 * L_guess)),
												min(L_max, 2 * L_guess)]))

						if verbose
							println("    Empirical: τ bounds = [$resolved_min, $resolved_max], " *
									"L_guess = $L_guess, L_grid = $L_grid")
						end

					#	Run Stability Scan
						sel = _select_L_by_stability_empirical(edges;
													nodes                = nodes,
													graph_type           = graph_type,
													reciprocity_collapse = reciprocity_collapse,
													tau_min              = resolved_min,
													tau_max              = resolved_max,
													L_grid               = L_grid,
													tol                  = tol,
													parallel             = parallel,
													verbose              = verbose,
													show_progress        = show_progress,
													inner_show_progress  = inner_show_progress)

					if verbose
						println(@sprintf("    Empirical done: L = %d, τ ∈ [%.4g, %.4g]",
											sel.L_best, resolved_min, resolved_max))
					end

					return (L              = sel.L_best,
							tau_min        = resolved_min,
							tau_max        = resolved_max,
							method         = :empirical,
							method_reason  = method_reason,
							T_max          = profile_result.T_max,
							profile        = profile_result.profile,
							scan           = sel.table,
							census         = sel.census_at_L_best,
							valid          = true)
			end
	end
	@doc raw"""
	**Description**
	Recommends a log-spaced $\tau$ grid size $L$ and $\tau$ bounds for the layered
	Batagelj–Mrvar triad census on weighted graphs, with **automatic method
	selection** between two principled approaches: an analytic triangle-decay
	method appropriate for large networks in the asymptotic regime, and an
	empirical AUMC stability scan appropriate for small networks or those with
	narrow weight distributions.

	**Usage**
	`recommend_L(edges; method=:auto, ...)`

	Forced selection: `method=:analytic` or `method=:empirical`.

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, `:weight`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe.
	- `graph_type::Symbol`: `:directed` or `:undirected`.
	- `reciprocity_collapse::Bool`: For `:directed` only — collapse mutual arcs.
	- `tau_min`, `tau_max::Union{Float64,Symbol}`: $\tau$ bounds or `:auto`.
	  When both are explicit Floats, both methods are short-circuited.
	- `method::Symbol`: `:auto`, `:analytic`, or `:empirical` (default `:auto`).
	- `points_per_decade::Int`: Heuristic for $L$ computation (default 8).
	- `L_min`, `L_max::Int`: Clamps on returned $L$.
	- `n_exploratory::Int`: Log-spaced points in the triangle profile (default 16).
	- `frac_keep::Float64`: e-fold threshold for analytic $\tau_{\min}$ (default $1/e$).
	- `T_min_floor::Int`: 3σ Poisson detection threshold (default 9).
	- `tol::Float64`: AUMC stability tolerance for empirical (default $10^{-3}$).
	- `T_max_threshold::Int`: Auto-selects analytic at or above (default 10000).
	- `T_max_floor::Int`: Auto-selects empirical below (default 100).
	- `weight_decades_min::Float64`: Minimum weight span for analytic in middle
	  zone (default 1.0).
	- `decay_range_points_min::Int`: Minimum grid points in $(1, T_\text{max}/2)$
	  for analytic in middle zone (default 3).
	- `parallel`, `verbose`, `show_progress`, `inner_show_progress`: control flags.

	**Method Selection Logic** (`method = :auto`)
	1. $T_\text{max} \geq T_\text{max\_threshold}$ (10000): **analytic**.
	   Strong Poisson asymptotic regime.
	2. $T_\text{max} < T_\text{max\_floor}$ (100): **empirical**.
	   Rare-structure regime; individual triangles carry semantic meaning.
	3. Middle zone: **analytic** iff weight span $\geq 1$ decade AND triangle
	   decay profile has $\geq 3$ grid points in $(1, T_\text{max}/2)$;
	   **empirical** otherwise.

	**Validation Evidence**
	Calibrated on Balikatan (directed, $T_\text{max}=3015$, middle zone with
	graduated decay $\to$ analytic), Moreno (directed, $T_\text{max}=307$,
	step-function decay $\to$ empirical), Scotland (undirected, $T_\text{max}=269$,
	step-function decay $\to$ empirical), and Marvel (undirected, $T_\text{max}$
	in millions $\to$ analytic). AUMC cosine similarity between methods $\approx 1.0$
	across all calibration networks.

	**Value**
	A `NamedTuple` with:
	- `L::Int`: Recommended grid size.
	- `tau_min::Float64`, `tau_max::Float64`: Recommended bounds.
	- `method::Symbol`: `:analytic`, `:empirical`, or `:user_supplied`.
	- `method_reason::String`: Human-readable explanation of the method choice.
	- `T_max::Int`: Unweighted triangle count (-1 if user-supplied bounds).
	- `profile::DataFrame`: Triangle decay profile (empty if user-supplied).
	- `scan::Union{DataFrame,Nothing}`: Empirical AUMC stability table
	  (`nothing` if analytic or user-supplied).
	- `census::Union{NamedTuple,Nothing}`: Cached layered census at $L_\text{best}$
	  (`nothing` if analytic or user-supplied; only the empirical path runs a
	  census during recommendation).
	- `valid::Bool`: `true` if the recommendation was derivable.

	**Examples**
	```julia
			using DataFrames

			#	Automatic method selection (default — recommended)
				rec = recommend_L(network.edges;
									nodes      = network.nodes,
									graph_type = :undirected,
									verbose    = true)
				println("Method used: $(rec.method) — $(rec.method_reason)")
				println("L = $(rec.L), τ ∈ [$(rec.tau_min), $(rec.tau_max)]")

			#	Force analytic method (e.g., Marvel-scale runs)
				rec_a = recommend_L(network.edges; method = :analytic)

			#	Force empirical method (small networks, sensitivity analysis)
				rec_e = recommend_L(network.edges; method = :empirical)

			#	User-supplied bounds (comparative studies)
				rec_u = recommend_L(network.edges; tau_min = 0.5, tau_max = 100.0)

			#	Empirical path: use cached census directly
				if rec.method === :empirical
					layered = rec.census   # full triad_census output at L_best
				end
	```

	**See Also**
	`triad_census`, `_triad_census_layered`, `_select_recommendation_method`,
	`_triangle_profile`, `_analytic_tau_bounds`, `_select_L_by_stability_empirical`,
	`_estimate_tau_bounds`, `_suggest_L_quick`

	**References**
	- Batagelj, V., & Mrvar, A. (2001). "A subquadratic triad census algorithm
	  for large sparse networks with small maximum degree." *Social Networks*,
	  23(3), 237–243.
	""" recommend_L

#	Helper Triad Census: Layered BM triad census with log-spaced τ (developer wrapper)
	function _triad_census_layered(edges::DataFrame;
									graph_type::Symbol = :directed,
									reciprocity_collapse::Bool = false,
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
									L::Int = 40,
									tau_min::Union{Symbol,Float64} = :auto,
									tau_max::Union{Symbol,Float64} = :auto,
									parallel::Bool = true,
									show_progress::Bool = false,
									progress_desc::Union{Nothing,String} = nothing)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			graph_type::Symbol: :directed (default) or :undirected
			reciprocity_collapse::Bool: directed-only; collapse mutual arcs (default false)
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe (includes isolates)
			L::Int: number of τ points in the log-spaced grid (default 40)
			tau_min::Union{Symbol,Float64}: lower τ bound (default :auto)
			tau_max::Union{Symbol,Float64}: upper τ bound (default :auto)
			parallel::Bool: parallelize the outer τ loop via Threads.@threads :static
				(default true). Set false when calling from an outer threaded
				context to avoid oversubscription, or to obtain a strictly serial
				reference run.
			show_progress::Bool: show a per-τ progress bar (default false). Useful
				for long-running weighted runs on large graphs where total
				wall-clock can reach minutes. When parallel=true, progress is
				ticked from inside the threaded loop under a ReentrantLock;
				ordering of tick events reflects completion order, not τ-grid
				order, but the final result is still folded in τ order and is
				bit-identical to a serial run.
			progress_desc::Union{Nothing,String}: optional description shown on
				the progress bar (default nothing, which produces "Triad census
				τ-grid (graph_type=..., L=...)"). Set explicitly when displaying
				multiple bars in sequence so they can be distinguished.
		Returns:
			NamedTuple: (per_tau::DataFrame, summary::DataFrame, meta::NamedTuple)
		Notes:
			Threading model. Each τ in the log-spaced grid runs an independent
			BM census on its own thresholded adjacency: no shared mutable state,
			no accumulator races, no false sharing. The outer τ loop is therefore
			the safe and natural threading target. The BM kernel itself
			(_triad_census_bm_directed) stays serial under this scheme.

			Census kernel. Both the directed and undirected per-τ censuses use
			the subquadratic dyad-driven directed BM kernel
			(_triad_census_bm_directed). For the undirected case,
			_prepare_binary_for_mode returns a canonicalized symmetric 0/1
			matrix; on a symmetric adjacency every dyad is reciprocal or empty,
			so the directed kernel populates only the four undirected classes
			{003, 102, 201, 300} and zeros the twelve asymmetric classes —
			identical output to the O(N^3) triple-loop kernel
			(_triad_census_bm_undirected), but subquadratic. The triple-loop
			kernel is retained as the reference implementation for regression
			testing; it is no longer on this path. This is the same
			directed-kernel-on-symmetric-matrix equivalence the binary
			undirected path in triad_census relies on.

			Determinism. Threading uses Threads.@threads :static, which assigns
			loop iterations to threads in a fixed, deterministic partition
			regardless of dynamic scheduling decisions. Results are pre-allocated
			into a per-τ Vector indexed by τ position (not appended to a shared
			Vector), so no cross-thread ordering can occur. Final results are
			folded in τ order, producing bit-identical output regardless of
			Threads.nthreads() and across repeated runs.

			When parallel=false, the function executes the τ loop serially.
			Set this when (a) calling from another threaded context (e.g., a
			Monte Carlo grid that's already parallelized over network instances),
			(b) measuring the serial baseline for performance comparison, or
			(c) debugging.

			meta.threaded reports whether the run actually used threads (requires
			parallel=true AND Threads.nthreads() > 1 AND length(τ-grid) > 1).
			meta.n_threads_used reports Threads.nthreads() if threaded, else 1.
		"""

		#	Build weighted adjacency once (directed)
			Aw, _, _ = isnothing(nodes) ?
				_graph_to_sparse_matrix(edges; weighted=true) :
				_graph_to_sparse_matrix(edges; nodes=nodes, weighted=true)
			n = size(Aw, 1)

		#	Compute τ grid from the weights we will actually threshold
			if graph_type === :undirected
				AU = Aw .+ Aw'
				@inbounds for i in 1:n; AU[i, i] = 0.0; end
				dropzeros!(AU)
				wvec = collect(nonzeros(AU))      # undirected (summed) weights
			else
				wvec = collect(nonzeros(Aw))      # directed weights
			end
			tgrid = _tau_grid(wvec; L=L, tau_min=tau_min, tau_max=tau_max)
			Ntau  = length(tgrid)

		#	Pre-Allocate Per-τ Count Storage (Each Slot Filled Independently)
			#	counts_by_tau[t] holds the length-16 DL count vector for tgrid[t].
			#	Pre-allocating by τ index — rather than appending to a shared
			#	Vector — is what makes the threaded loop deterministic and
			#	race-free: each thread writes only to its own slot(s).
				counts_by_tau = Vector{Vector{Int}}(undef, Ntau)

		#	Configure Progress Bar
			#	Mirrors the leiden_community_detection progress pattern: a
			#	Progress instance plus a ReentrantLock for thread-safe ticks.
				use_threads = parallel && Threads.nthreads() > 1 && Ntau > 1
				desc        = progress_desc !== nothing ? progress_desc :
				              "Triad census τ-grid ($(string(graph_type)), L=$Ntau, " *
				              (use_threads ? "$(Threads.nthreads()) threads" : "serial") * ")"
				prog        = show_progress ? Progress(Ntau, desc = desc, enabled = true) : nothing
				prog_lock   = ReentrantLock()

		#	Run Census Across τ (Threaded or Serial Per `parallel` Flag)
			if use_threads
				#	Threaded: deterministic :static scheduling, per-τ independence
					Threads.@threads :static for t in 1:Ntau
						τ  = tgrid[t]
						Ab = _prepare_binary_for_mode(Aw, τ, graph_type, reciprocity_collapse)
						if graph_type === :directed
							res = _triad_census_bm_directed(Ab)
							counts_by_tau[t] = res.counts
						else
							#	Undirected: Ab is a canonicalized symmetric 0/1 matrix
							#	(see _prepare_binary_for_mode). Run the subquadratic
							#	dyad-driven directed BM kernel on it rather than the
							#	O(N^3) triple-loop kernel. On a symmetric adjacency the
							#	directed kernel populates only {003,102,201,300} and
							#	zeros the twelve asymmetric classes — identical output,
							#	subquadratic cost. Same equivalence the binary
							#	undirected path in triad_census relies on.
								res = _triad_census_bm_directed(Ab)
								counts_by_tau[t] = res.counts
						end
						#	Thread-Safe Progress Tick
							if show_progress
								lock(prog_lock) do
									next!(prog)
								end
							end
					end
			else
				#	Serial path: same logic, no threading overhead
					for t in 1:Ntau
						τ  = tgrid[t]
						Ab = _prepare_binary_for_mode(Aw, τ, graph_type, reciprocity_collapse)
						if graph_type === :directed
							res = _triad_census_bm_directed(Ab)
							counts_by_tau[t] = res.counts
						else
							#	Undirected: subquadratic directed kernel on the
							#	canonicalized symmetric matrix (see threaded branch).
								res = _triad_census_bm_directed(Ab)
								counts_by_tau[t] = res.counts
						end
						#	Serial Progress Tick (No Lock Needed)
							if show_progress
								next!(prog)
							end
					end
			end

		#	Fold Per-τ Counts into Per-τ Rows (Deterministic Order)
			labels16     = _dl_labels()
			per_tau_rows = Vector{NamedTuple}(undef, 0)
			for t in 1:Ntau
				τ      = tgrid[t]
				counts = counts_by_tau[t]
				dens   = _to_density_16!(counts, n)
				@inbounds for k in 1:16
					push!(per_tau_rows, (tau = τ, triad = labels16[k], count = counts[k], density = dens[k]))
				end
			end

		#	Tidy per-τ DataFrame
			per_tau = DataFrame(per_tau_rows)

		#	Summaries: AUMC over log10(τ), peak τ and peak density per triad
			summary_rows = Vector{NamedTuple}(undef, 0)
			for tri in labels16
				sub = per_tau[per_tau.triad .== tri, :]
				if nrow(sub) == 0
					push!(summary_rows, (triad=tri, AUMC_density=0.0, peak_tau=NaN, peak_density=0.0))
				else
					auc = _aumc_logtau(sub.tau, sub.density)
					mx  = argmax(sub.density)
					push!(summary_rows, (triad=tri, AUMC_density=auc, peak_tau=sub.tau[mx], peak_density=sub.density[mx]))
				end
			end
			summary = DataFrame(summary_rows)

		#	Meta
			meta = (n                    = n,
					L                    = Ntau,
					tau_min              = first(tgrid),
					tau_max              = last(tgrid),
					graph_type           = graph_type,
					reciprocity_collapse = reciprocity_collapse,
					threaded             = use_threads,
					n_threads_used       = use_threads ? Threads.nthreads() : 1)

		#	Return
			return (per_tau=per_tau, summary=summary, meta=meta)
	end

#	Triad Census (directed/undirected; binary/weighted via layered τ or single-bin)
	function triad_census(edges::DataFrame;
						nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
						weighted::Bool=false,
						graph_type::Symbol=:directed,
						reciprocity_collapse::Bool=false,
						L::Int=20, tau_min::Float64=1.0, tau_max::Float64=maximum(ones(Float64,1)),
						parallel::Bool=true,
						show_progress::Bool=false)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe (includes isolates)
			weighted::Bool: false → binary BM single-bin; true → layered BM over τ
			graph_type::Symbol: :directed (default) or :undirected
			reciprocity_collapse::Bool: directed-only; collapse mutual arcs (default false)
			L, tau_min, tau_max: layered τ controls (weighted=true only)
			parallel::Bool: weighted=true only; parallelize the outer τ loop via
				Threads.@threads :static (default true). Has no effect on the binary
				path (weighted=false), which is a single kernel call.
			show_progress::Bool: weighted=true only; display a per-τ progress bar
				inside _triad_census_layered (default false). Has no effect on the
				binary path. Useful for long Marvel-scale weighted runs where the
				layered census takes many minutes per call.
		Returns:
			- weighted=false: DataFrame(triad, count) in 16-class DL order
			- weighted=true:  NamedTuple(per_tau, summary, meta) from layered census
		Notes:
			Threading. The binary path (weighted=false) does a single BM kernel
			call and is unaffected by the parallel kwarg. The weighted path
			(weighted=true) runs a τ grid and parallelizes the outer loop;
			results are bit-identical across thread counts thanks to :static
			scheduling and pre-allocated per-τ result storage.

			Undirected binary kernel. The undirected binary census is computed
			by feeding the symmetrized 0/1 adjacency to the subquadratic,
			dyad-driven directed BM kernel (_triad_census_bm_directed), NOT the
			O(N^3) triple-loop kernel (_triad_census_bm_undirected). On a
			symmetric adjacency every dyad is reciprocal or empty, so the
			directed kernel's DL classification populates only the four
			undirected classes {003, 102, 201, 300} and leaves the twelve
			asymmetric classes at zero — identical output to the triple-loop
			kernel, but subquadratic. This is the same reciprocity-collapse
			equivalence the directed path uses for Pajek semantics. The
			triple-loop kernel (_triad_census_bm_undirected) is retained as the
			reference implementation for regression testing (see
			test_undirected_triad_kernel_equivalence) but is no longer on the
			public path. Validated on Scotland (exact match across all four
			classes) and Marvel (300 = 1,028,453 in seconds vs minutes).

			See _triad_census_layered for the full threading contract.
		"""

		#	Validation
			@assert graph_type in (:directed, :undirected) "graph_type must be :directed or :undirected"
			if graph_type == :undirected
				@assert !reciprocity_collapse "reciprocity_collapse applies only when graph_type == :directed"
			end

		#	Route by weighted flag
			if !weighted
				#	— Binary BM paths —
					if graph_type == :directed
						#	Build directed simple 0/1, drop loops
							adj, _, _ = isnothing(nodes) ?
								_graph_to_sparse_matrix(edges; weighted=false) :
								_graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)
							_make_directed_simple!(adj)

						#	Optional compatibility collapse (Pajek-style)
							if reciprocity_collapse
								adj = max.(adj, adj')
								_make_directed_simple!(adj)
							end

						#	Run directed BM
							res = _triad_census_bm_directed(adj)
							return DataFrame(triad = res.labels, count = res.counts)

					else
						#	Undirected binary: symmetrize by max, zero diag, then run the
						#	SUBQUADRATIC directed BM kernel on the symmetric matrix.
						#	On a symmetric 0/1 adjacency every dyad is reciprocal or
						#	empty, so _triad_census_bm_directed populates exactly the
						#	four undirected classes {003, 102, 201, 300} and zeros the
						#	twelve asymmetric classes — identical to the O(N^3)
						#	triple-loop kernel but subquadratic. The triple-loop
						#	kernel (_triad_census_bm_undirected) is kept as the
						#	regression reference, not used here.
							adj, _, _ = isnothing(nodes) ?
								_graph_to_sparse_matrix(edges; weighted=false) :
								_graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)
							_make_directed_simple!(adj)           # binarize per-direction, drop loops
							Au = max.(adj, adj')                  # undirected 0/1
							n = size(Au, 1); @inbounds for i in 1:n; Au[i,i] = 0.0; end
							dropzeros!(Au)

						#	Run directed BM on the symmetric matrix (subquadratic route)
							res = _triad_census_bm_directed(Au)
							return DataFrame(triad = res.labels, count = res.counts)
					end

			else
				#	— Layered weighted BM (log-spaced τ), threading + progress forwarded —
					return _triad_census_layered(edges;
								nodes                = nodes,
								graph_type           = graph_type,
								reciprocity_collapse = reciprocity_collapse,
								L                    = L,
								tau_min              = tau_min,
								tau_max              = tau_max,
								parallel             = parallel,
								show_progress        = show_progress)
			end
	end
	@doc raw"""
	**Description**
	Triad census in the Davis–Leinhardt 16-class order, supporting both binary
	(unweighted) and weighted networks under directed and undirected
	conventions. Counts every unordered triple of nodes by its isomorphism
	class. For directed graphs all 16 classes can be non-zero; for undirected
	graphs only $\{003, 102, 201, 300\}$ — the four undirected isomorphism
	types — can be non-zero. The full 16-vector is always returned for
	consistency.

	Two computational paths:
	- **Binary path** (`weighted=false`): a single Batagelj–Mrvar (BM) census
	  on the binarized graph; subquadratic for sparse graphs.
	- **Weighted path** (`weighted=true`): a layered BM census across a
	  log-spaced grid of edge-weight thresholds $\tau$, with the outer loop
	  parallelized by default and an optional per-$\tau$ progress bar.

	The 16 classes in DL order:

	$$003,\ 012,\ 102,\ 021D,\ 021U,\ 021C,\ 111D,\ 111U,\ 030T,\ 030C,\ 201,\ 120D,\ 120U,\ 120C,\ 210,\ 300$$

	**Usage**
	`triad_census(edges; nodes=nothing, weighted=false, graph_type=:directed, reciprocity_collapse=false, L=20, tau_min=1.0, tau_max=..., parallel=true, show_progress=false)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`
	  (used only when `weighted=true`; ignored for the binary path).
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional fixed node universe;
	  includes isolates in the census.
	- `weighted::Bool`: `false` (default) → single binary BM census; `true` →
	  layered weighted BM across a $\tau$ grid.
	- `graph_type::Symbol`: `:directed` (default) or `:undirected`.
	- `reciprocity_collapse::Bool`: Directed only — when `true`, mutual arcs
	  $i \leftrightarrow j$ are collapsed to single ties by $\max(A, A^\top)$,
	  reproducing Pajek-style triad semantics (suppresses asymmetric classes
	  so only $\{003, 102, 201, 300\}$ are non-zero).
	- `L::Int`, `tau_min::Float64`, `tau_max::Float64`: Layered $\tau$ controls
	  (used only when `weighted=true`). For automatic $\tau$ bound selection
	  based on the observed weight distribution, see `recommend_L` and
	  `_triad_census_layered` (which accepts `:auto` for the bounds).
	- `parallel::Bool`: Weighted path only — parallelize the outer $\tau$ loop
	  via `Threads.@threads :static` (default `true`). No effect on the
	  binary path. See *Threading* below.
	- `show_progress::Bool`: Weighted path only — display a per-$\tau$ progress
	  bar inside the layered census (default `false`). No effect on the binary
	  path. Useful for Marvel-scale runs where the layered census takes many
	  minutes per call.

	**Details**

	*Binary path.* Builds a $0/1$ loopless adjacency. For directed graphs with
	`reciprocity_collapse=true`, mutual arcs are first collapsed to single
	ties. The directed census is fed to `_triad_census_bm_directed`. The
	undirected census symmetrizes by $\max(A, A^\top)$ and feeds the resulting
	symmetric matrix to the *same* directed kernel: on a symmetric adjacency
	every dyad is reciprocal or empty, so the DL classification populates only
	$\{003, 102, 201, 300\}$ and zeros the twelve asymmetric classes. Both
	binary paths therefore use the dyad-driven Batagelj–Mrvar (2001) kernel,
	subquadratic for sparse inputs.

	*Weighted path.* Delegates to `_triad_census_layered`, which thresholds
	the weighted adjacency at each $\tau$ in a log-spaced grid and runs the
	binary BM kernel on each thresholded matrix. Returns per-$\tau$ counts
	and densities plus per-triad AUMC/peak summaries. For directed weighted
	graphs the optional `reciprocity_collapse` flag is honored at each $\tau$.
	For undirected weighted graphs, weights are symmetrized by sum
	($W + W^\top$) before thresholding, aligning with $s$-core semantics.

	*Threading.* The weighted path parallelizes the outer $\tau$ loop. Each
	$\tau$ runs an independent census on its own thresholded matrix, so there
	is no shared mutable state and no risk of accumulator races. `:static`
	scheduling with pre-allocated per-$\tau$ result storage guarantees the
	threaded output is bit-identical to the serial output regardless of
	`Threads.nthreads()`. To verify this contract on a given input, use
	`test_threaded_vs_serial_layered` from the test suite. To obtain a strict
	serial run for benchmarking or when calling from another threaded context,
	pass `parallel=false`.

	*Reciprocity collapse semantics.* When `reciprocity_collapse=true`, the
	directed network is rendered as if undirected for purposes of triad
	classification: every mutual arc $i \leftrightarrow j$ is treated as a
	single tie. The result is that asymmetric DL classes (012, 021D, 021U,
	021C, 111D, 111U, 030T, 030C, 120D, 120U, 120C, 210) all report zero
	counts, leaving only the four classes $\{003, 102, 201, 300\}$. This
	reproduces Pajek's triad-census output on directed inputs. The undirected
	binary path relies on this same equivalence internally to use the
	subquadratic directed kernel.

	**Value**

	*Binary path* (`weighted=false`): `DataFrame` with columns:
	- `:triad::String` — DL class label
	- `:count::Int` — number of triples in that class

	Rows are in DL order.

	*Weighted path* (`weighted=true`): `NamedTuple` with:
	- `per_tau::DataFrame`: long-format table with `:tau`, `:triad`, `:count`,
	  `:density` (count divided by $\binom{n}{3}$). One row per
	  $(\tau, \text{triad})$ pair.
	- `summary::DataFrame`: per-triad summary with `:triad`, `:AUMC_density`,
	  `:peak_tau`, `:peak_density`.
	- `meta::NamedTuple`: $n$, $L$, $\tau_{\min}$, $\tau_{\max}$, `graph_type`,
	  `reciprocity_collapse`, `threaded`, `n_threads_used`.

	**Examples**
	```julia
			using DataFrames

			#	Binary directed triad census
				edges = DataFrame(src = ["a","b","c","a"],
								dst = ["b","c","a","c"])
				result = triad_census(edges; graph_type=:directed)
				result  # DataFrame with 16 rows, counts for each DL class

			#	Pajek-style: collapse mutual arcs first
				result = triad_census(edges; graph_type=:directed,
											reciprocity_collapse=true)

			#	Undirected binary
				undir = DataFrame(src = ["a","a","b"],
								dst = ["b","c","c"])
				result = triad_census(undir; graph_type=:undirected)

			#	Weighted layered census (threaded, with per-τ progress bar)
				w = DataFrame(src = ["a","a","b","b","c"],
							dst = ["b","c","c","a","a"],
							weight = [3.0, 1.0, 5.0, 2.0, 4.0])
				layered = triad_census(w; weighted=true, graph_type=:directed,
											L=20, tau_min=0.5, tau_max=5.0,
											show_progress=true)

			#	Strict serial run
				layered_serial = triad_census(w; weighted=true, L=20,
													parallel=false)
	```
	**Performance Notes**
	Both binary kernels are subquadratic for sparse graphs (dyad-driven BM).
	The undirected binary census reuses the directed kernel on the symmetrized
	$0/1$ adjacency rather than a triple loop over node triples, so it is
	subquadratic at Marvel scale ($N = 6{,}486$): the symmetric-matrix route
	computes the four undirected classes in seconds. The retained triple-loop
	kernel `_triad_census_bm_undirected` is $O(N^3)$ and is kept only as a
	regression reference; it is not on the public path. For the weighted
	layered path, threading gives near-linear speedup in the $\tau$ loop,
	which is the dominant cost when $L \cdot n_{\text{thresholds}}$ is large.

	**See Also**
	`_triad_census_layered`, `_triad_census_bm_directed`,
	`_triad_census_bm_undirected`, `recommend_L`, `test_threaded_vs_serial_layered`

	**References**
	- Batagelj, V., & Mrvar, A. (2001). "A subquadratic triad census algorithm
	  for large sparse networks with small maximum degree." *Social Networks*,
	  23(3), 237–243.
	- Davis, J. A., & Leinhardt, S. (1972). "The structure of positive
	  interpersonal relations in small groups."
	""" triad_census
	
##############################
#   SECTION 6: BICOMPONENTS  #
##############################

# ====================================================================
# Biconnected components ("bicomponents", "blocks") and the related
# topology measure: proportion of nodes in the largest bicomponent.
#
# A biconnected component is a maximal subgraph in which any two
# vertices lie on a common cycle. Equivalently: a maximal subgraph
# that remains connected after removing any single vertex. An
# articulation point (cut vertex) is a vertex whose removal disconnects
# the graph; bicomponents share at most one vertex with each other,
# and that shared vertex is always an articulation point.
#
# Algorithm: Tarjan (1972) DFS-based linear-time approach with an
# iterative explicit stack (no recursion, so behavior on large
# networks is bounded by available heap rather than the Julia call
# stack). Standard maintenance of:
#   disc[v] — DFS discovery time of v
#   low[v]  — minimum discovery time reachable from v's subtree via
#             at most one back-edge
#   An edge stack from which bicomponents are popped when an
#   articulation point is identified.
#
# For SMM Phase 0, bicomponents are computed on the *underlying
# undirected* graph regardless of input direction. Directed inputs are
# symmetrized via max(A, A^T) before the DFS, matching the convention
# used by closeness, betweenness, and Bonacich in earlier sections.
#
# Internal helpers:
#   _biconnected_components(adj) — return Vector{Set{Int}} of vertex
#                                  sets, one per bicomponent (plus
#                                  isolate singletons)
#
# Public API:
#   largest_bicomponent_proportion — size of largest bicomponent / N
# ====================================================================

#	Helper Function for largest_bicomponent_proportion: Tarjan's Bicomponents
	function _biconnected_components(adj::SparseMatrixCSC{<:Real, Int})
		"""
		Args:
			adj::SparseMatrixCSC{<:Real, Int}: adjacency matrix
		Returns:
			Vector{Set{Int}}: vertex sets, one per bicomponent.
				Isolated vertices are returned as singleton sets.
		Notes:
			Implements Tarjan's (1972) algorithm iteratively. The input
			adjacency is assumed to be symmetric (caller's responsibility);
			the algorithm treats edges as undirected.

			An edge stack collects edges as DFS descends. When an articulation
			point is identified (low[child] >= disc[parent]), all edges on
			the stack down to and including the (parent, child) edge form
			one bicomponent; their endpoints constitute one returned vertex
			set.

			Isolates are detected as nodes with no neighbors and returned as
			singleton sets — they are not "bicomponents" in the strictest
			graph-theoretic sense, but for proportional measures it is more
			natural to count each isolated vertex as a degenerate block of
			size 1.

			A connected component consisting of a single edge produces one
			bicomponent of size 2 (the two endpoints). A K_3 triangle is one
			bicomponent of size 3. A path A-B-C-D produces three bicomponents
			of size 2 each: {A,B}, {B,C}, {C,D}.
		"""

		#	Basic Checks
			n = size(adj, 1)
			@assert size(adj, 2) == n "adj must be square"

		#	Quick Returns
			if n == 0
				return Set{Int}[]
			end

		#	Build Undirected Neighbor Lists (Drop Self-Loops)
		#	The caller may have passed a symmetric adjacency, but we walk
		#	the rows defensively so an asymmetric input is also handled
		#	by symmetrization at the neighbor-list level.
			rows = rowvals(adj)
			neighbors = [Int[] for _ in 1:n]
			seen_pairs = Set{Tuple{Int, Int}}()
			@inbounds for j in 1:n
				for ptr in nzrange(adj, j)
					i = rows[ptr]
					i == j && continue
					key = i < j ? (i, j) : (j, i)
					if !(key in seen_pairs)
						push!(seen_pairs, key)
						push!(neighbors[i], j)
						push!(neighbors[j], i)
					end
				end
			end

		#	Iterative Tarjan State
			disc       = zeros(Int, n)        # 0 = unvisited
			low        = zeros(Int, n)
			parent     = zeros(Int, n)        # 0 = root in its DFS tree
			timer      = 0
			edge_stack = Tuple{Int, Int}[]    # stack of (u, v) DFS-tree-or-back edges
			bicomps    = Set{Int}[]           # output

		#	Process Each Unvisited Vertex (Handles Disconnected Graphs)
			for start in 1:n
				if disc[start] != 0
					continue
				end

				#	Isolated Vertex: No Neighbors → Singleton Bicomponent
					if isempty(neighbors[start])
						timer += 1
						disc[start] = timer
						low[start]  = timer
						push!(bicomps, Set{Int}([start]))
						continue
					end

				#	Iterative DFS from `start`
				#	Each stack frame is (vertex, neighbor-index)
				#	where neighbor-index is the position in neighbors[v]
				#	from which the next descendant will be considered.
					dfs_stack = Tuple{Int, Int}[(start, 1)]
					timer += 1
					disc[start] = timer
					low[start]  = timer
					root_children = 0   # for articulation-point test at the DFS-tree root

					while !isempty(dfs_stack)
						(u, idx) = dfs_stack[end]

						if idx <= length(neighbors[u])
							w = neighbors[u][idx]
							#	Advance Iterator for this Frame
								dfs_stack[end] = (u, idx + 1)

							if disc[w] == 0
								#	Tree Edge: Descend into w
									parent[w] = u
									timer += 1
									disc[w] = timer
									low[w]  = timer
									push!(edge_stack, (u, w))
									if u == start
										root_children += 1
									end
									push!(dfs_stack, (w, 1))
							elseif w != parent[u] && disc[w] < disc[u]
								#	Back Edge: Update Low and Stack
									push!(edge_stack, (u, w))
									if disc[w] < low[u]
										low[u] = disc[w]
									end
							end
						else
							#	All Neighbors of u Exhausted: Pop u
								pop!(dfs_stack)

							#	If u Is Not the DFS Root: Propagate Low to Parent
							#	and Check the Articulation-Point Condition.
								if u != start
									p = parent[u]
									if low[u] < low[p]
										low[p] = low[u]
									end

									#	Articulation-Point Test
									#	(non-root): low[u] >= disc[p] ⇒ p is articulation
									#	             and the edge stack down to (p, u)
									#	             is one bicomponent.
									if low[u] >= disc[p]
										#	Pop Edges Down to (p, u) and Collect Vertex Set
											comp_vertices = Set{Int}()
											while !isempty(edge_stack)
												(a, b) = pop!(edge_stack)
												push!(comp_vertices, a)
												push!(comp_vertices, b)
												if (a == p && b == u) || (a == u && b == p)
													break
												end
											end
											push!(bicomps, comp_vertices)
									end
								end
						end
					end

				#	Special Case: DFS-Tree Root Is an Articulation Point Iff It Has > 1 Children
				#	The vertex-set collection for the root's last subtree happens automatically
				#	in the loop above (when low[child] >= disc[root] = 1, which always holds
				#	for tree children). So we don't need extra work here for root-articulation;
				#	we just ensure any residual edges in the stack are popped.
				#	But the test for non-root articulation already pops them, so root_children
				#	is informational only.
					_ = root_children
			end

		#	Return Bicomponent Vertex Sets
			return bicomps
	end

#	Largest Bicomponent Proportion (Public API)
	function largest_bicomponent_proportion(edges::DataFrame;
	                                       nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                                       directed::Bool = true,
	                                       agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight (ignored)
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			directed::Bool: kept for API symmetry; bicomponents are always
				computed on the symmetrized adjacency (default = true)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			Float64: size of the largest bicomponent, divided by N
		Notes:
			Smith, Morgan, & Moody (2022) topology measure: proportion of
			nodes in the largest biconnected component. The denominator N is
			taken from `nodes` when supplied (so isolates count toward N),
			or from the edge list otherwise.

			A biconnected component is a maximal subgraph in which any two
			vertices lie on a common cycle — equivalently, a maximal subgraph
			that stays connected when any single vertex is removed. The
			"bicomponent size" is the number of distinct vertices in that
			subgraph, not the number of edges.

			Direction is always symmetrized first (max(A, A^T)); SMM's measure
			is defined on the underlying undirected graph. Self-loops are
			ignored.

			Returns 0.0 for an empty edge list (with no `nodes` argument),
			or 1/N when only isolates exist.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					n = nodes isa DataFrame ? nrow(nodes) : length(nodes)
					return n > 0 ? 1.0 / n : 0.0
				else
					return 0.0
				end
			end

		#	Aggregate Multi-Edges (Defensive)
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Build Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, _ = _edgelist_to_sparse_matrix(clean_edges; weighted = false)
			else
				adj, _, _ = _graph_to_sparse_matrix(clean_edges;
				                                    nodes = nodes,
				                                    weighted = false)
			end

		#	Symmetrize for Bicomponent Analysis
			adj = max.(adj, adj')

		#	Compute Bicomponents
			N = size(adj, 1)
			if N == 0
				return 0.0
			end
			bicomps = _biconnected_components(adj)

		#	Largest Bicomponent / N
			if isempty(bicomps)
				return 0.0
			end
			largest = maximum(length(b) for b in bicomps)
			return largest / N
	end
	@doc raw"""
	**Description**
	Compute the proportion of nodes in the largest biconnected component
	("bicomponent" or "block") of the network. One of Smith, Morgan, & Moody's
	(2022) topology measures.

	A biconnected component is a maximal subgraph in which any two vertices
	lie on a common cycle. Equivalently, it is a maximal subgraph that
	remains connected after the removal of any single vertex. Articulation
	points (cut vertices) join bicomponents; each pair of bicomponents
	shares at most one vertex.

	**Usage**
	`largest_bicomponent_proportion(edges::DataFrame; nodes=nothing, directed=true, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src` and `:dst`, optionally
	  `:weight` (ignored; only edge presence matters).
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe. When
	  supplied, the denominator $N$ includes isolates.
	- `directed::Bool`: Kept for API symmetry; the adjacency is always
	  symmetrized via $\max(A, A^T)$ before computing bicomponents, since
	  SMM's measure is defined on the underlying undirected graph.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`,
	  immaterial since weights are ignored).

	**Details**
	Implemented via Tarjan's (1972) DFS-based linear-time algorithm with an
	iterative explicit stack. Self-loops are ignored. Each isolated vertex
	is counted as a singleton bicomponent for proportion purposes.

	A path graph $A — B — C — D$ has three bicomponents of size 2 each.
	A cycle on $n$ vertices is a single bicomponent of size $n$. A K_3
	triangle is a single bicomponent of size 3.

	**Value**
	A `Float64` in $[0, 1]$.

	**Examples**
	```julia
	using DataFrames

	#	Path graph: 5 nodes, 4 edges, all of degree ≤ 2
	edges = DataFrame(src=["1","2","3","4"], dst=["2","3","4","5"])
	nodes = DataFrame(id=string.(1:5), label=string.(1:5))
	largest_bicomponent_proportion(edges; nodes=nodes)   # 2/5 = 0.4
	```

	**References**
	- Tarjan RE (1972). "Depth-first search and linear graph algorithms."
	  *SIAM Journal on Computing* 1(2): 146–160.
	- Smith JA, Morgan JH, Moody J (2022). "Network sampling coverage III."
	  *Social Networks* 68: 148–178.

	**See Also**
	`largest_component_proportion`
	""" largest_bicomponent_proportion

################################
#   SECTION 7: TAU STATISTIC   #
################################

# ====================================================================
# Tau statistic for the triad census conditioned on the dyad census
# (U|MAN null model). The tau statistic is the standardized weighted
# deviation of the observed triad distribution from its expectation
# under U|MAN:
#
#     τ = (λ^T T_obs − E[λ^T T | M, A, N]) / SD(λ^T T | M, A, N)
#
# where:
#   T_obs    is the observed length-16 Davis-Leinhardt triad count vector
#   M, A, N  are the counts of mutual, asymmetric, and null dyads
#   λ        is a length-16 weighting vector selecting which substantive
#            theory's permitted triads to test (default: ranked clusters)
#
# The U|MAN null model holds M, A, N fixed; otherwise every directed
# graph with that dyad census is equally likely. Tau measures how far
# the observed graph's weighted triad count deviates from the typical
# weighted count produced under that null.
#
# Implementation note: the variance and covariance terms in the closed
# form of E[λ^T T | M, A, N] and Var(λ^T T | M, A, N) are extensive
# (Holland & Leinhardt 1976; Wasserman & Faust 1994). Rather than
# transcribing approximately 150 lines of dense algebra with high typo
# risk, this module estimates E and Var by Monte Carlo sampling from
# the U|MAN distribution. With the default 500 samples the standard
# error on the variance estimate is approximately 4%, which is more
# than precise enough for the descriptive comparisons SMM (2022) use
# this statistic for.
#
# The U|MAN sampler is direct (no MCMC mixing required): for a graph
# with M mutual, A asymmetric, and N null dyads, randomly permute the
# C(N, 2) unordered pairs, assign the first M to be mutual, the next
# A to be asymmetric with random direction, and the rest to be null.
# Each sample is independent.
#
# Ranked-clusters weighting (the default `:RC`):
#   Davis-Leinhardt permitted triads under the ranked-clusters model
#   are 003, 102, 021D, 021U, 030T, 120D, 120U, 300. These are the
#   triads compatible with a hierarchy of mutually-tied clusters
#   where between-cluster ties go in one direction from lower to higher
#   rank. As an indicator vector over the Davis-Leinhardt order:
#       [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
#
# Transitivity weighting (`:transitivity`):
#   Holland & Leinhardt (1971) permitted triads under the transitivity
#   model add 012 and 210 to the ranked-clusters set. The corresponding
#   indicator is:
#       [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
#
# Public API:
#   tau_statistic — Monte Carlo tau under U|MAN conditioning
#
# Internal helpers:
#   _ranked_clusters_weighting — return the RC indicator vector
#   _transitivity_weighting    — return the transitivity indicator
#   _resolve_weighting         — convert a symbol or vector to Vector{Float64}
#   _dyad_census_counts        — compute (M, A, N) on a sparse adjacency
#   _sample_uman_edges         — direct U|MAN sampler returning an edge list
# ====================================================================

#	Helper Function for tau_statistic: Ranked-Clusters Weighting Vector
	function _ranked_clusters_weighting()
		"""
		Args:
			(none)
		Returns:
			Vector{Float64}: length-16 indicator vector over DL-ordered triads.
		Notes:
			Davis-Leinhardt ranked-clusters permitted triads:
				003, 102, 021D, 021U, 030T, 120D, 120U, 300.
			These are the triads compatible with a hierarchy of mutually-tied
			clusters where between-cluster ties go in one direction from
			lower to higher rank.
		"""
		#	Indicator Aligned to _dl_labels(): 003, 012, 102, 021D, 021U, 021C,
		#	111D, 111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300
			return Float64[1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
	end

#	Helper Function for tau_statistic: Transitivity Weighting Vector
	function _transitivity_weighting()
		"""
		Args:
			(none)
		Returns:
			Vector{Float64}: length-16 indicator vector over DL-ordered triads.
		Notes:
			Holland & Leinhardt (1971) transitivity-permitted triads:
				003, 012, 102, 021D, 021U, 030T, 120D, 120U, 210, 300.
			Adds 012 (single-arc) and 210 (mutual-plus-arc) to the
			ranked-clusters set.
		"""
			return Float64[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
	end

#	Helper Function for tau_statistic: Resolve weighting Argument
	function _resolve_weighting(weighting::Union{Symbol, AbstractVector})
		"""
		Args:
			weighting::Union{Symbol, AbstractVector}: :RC, :transitivity, or
				a length-16 numeric vector
		Returns:
			Vector{Float64}: length-16 weighting vector
		Notes:
			Validates and converts the user-facing argument into the canonical
			numeric form used internally. Symbols are looked up in the named
			set; vectors are length-checked and converted to Float64.
		"""
		if weighting isa Symbol
			if weighting == :RC
				return _ranked_clusters_weighting()
			elseif weighting == :transitivity
				return _transitivity_weighting()
			else
				throw(ArgumentError(
					"weighting symbol must be :RC or :transitivity; got :$(weighting). " *
					"For other weightings, pass a length-16 Vector{Float64} aligned to " *
					"Davis-Leinhardt order (003, 012, 102, 021D, 021U, 021C, 111D, 111U, " *
					"030T, 030C, 201, 120D, 120U, 120C, 210, 300)."
				))
			end
		else
			if length(weighting) != 16
				throw(ArgumentError(
					"custom weighting vector must have length 16 (got length " *
					"$(length(weighting))). Align entries to Davis-Leinhardt order."
				))
			end
			return Float64.(collect(weighting))
		end
	end

#	Helper Function for tau_statistic: Dyad Census Counts (M, A, N)
	function _dyad_census_counts(adj::SparseMatrixCSC{<:Real, Int})
		"""
		Args:
			adj::SparseMatrixCSC: directed adjacency, 0/1, loopless
		Returns:
			NamedTuple: (M, A, N_null) — mutual, asymmetric, null dyad counts
		Notes:
			Walks all upper-triangular pairs (i < j) and classifies each
			by edge presence:
				M:      adj[i, j] > 0 AND adj[j, i] > 0
				A:      exactly one of adj[i, j], adj[j, i] is > 0
				N_null: both are 0
			Sums to C(N, 2). Caller should ensure self-loops are zeroed first.
		"""
		n = size(adj, 1)
		@assert size(adj, 2) == n "adj must be square"

		#	Quick Return for Trivial Cases
			if n < 2
				return (M = 0, A = 0, N_null = 0)
			end

		#	Count by Iterating Over Unordered Pairs
		#	For sparse matrices this is O(N^2) in the worst case, but the
		#	matrix lookups are O(log k) for column-sparse format. For Phase 0
		#	corpus (largest N=1347), this is well under a second.
			M = 0
			A = 0
			N_null = 0
			@inbounds for i in 1:n - 1
				for j in i + 1:n
					eij = (adj[i, j] != 0)
					eji = (adj[j, i] != 0)
					if eij && eji
						M += 1
					elseif eij || eji
						A += 1
					else
						N_null += 1
					end
				end
			end

		#	Return Counts
			return (M = M, A = A, N_null = N_null)
	end

#	Helper Function for tau_statistic: Direct U|MAN Sampler
	function _sample_uman_edges(N::Int, M::Int, A::Int, rng::AbstractRNG)
		"""
		Args:
			N::Int: number of nodes
			M::Int: number of mutual dyads (in the target U|MAN distribution)
			A::Int: number of asymmetric dyads
			rng::AbstractRNG: random number generator
		Returns:
			DataFrame: columns [src, dst] with string node IDs "1" .. "N"
		Notes:
			Direct sampler — no MCMC. Each sample is independent of the others.
			Generates a graph uniformly at random over all directed graphs with
			exactly M mutual dyads, A asymmetric dyads, and C(N,2) - M - A null
			dyads.

			Algorithm:
				1. List all C(N,2) unordered pairs (i, j) with i < j
				2. Permute uniformly at random
				3. First M pairs become mutual (i→j and j→i)
				4. Next A pairs become asymmetric; direction chosen by fair coin
				5. Remaining pairs are null (no edges)

			The node universe is "1", "2", ..., "N" as strings, matching the
			rest of the package's node-ID convention.
		"""

		#	Build the List of Unordered Pairs
			n_pairs = (N * (N - 1)) ÷ 2
			@assert M + A <= n_pairs "M + A must not exceed C(N, 2) = $n_pairs"

			pairs_i = Vector{Int}(undef, n_pairs)
			pairs_j = Vector{Int}(undef, n_pairs)
			idx = 0
			@inbounds for i in 1:N - 1
				for j in i + 1:N
					idx += 1
					pairs_i[idx] = i
					pairs_j[idx] = j
				end
			end

		#	Permute Uniformly at Random
			perm = randperm(rng, n_pairs)

		#	Assign Roles and Build Edge List
		#	Each mutual contributes 2 directed edges; each asymmetric contributes 1.
			n_edges = 2 * M + A
			edge_src = Vector{String}(undef, n_edges)
			edge_dst = Vector{String}(undef, n_edges)
			out_idx = 0

			#	Mutual Pairs (First M of the Permutation)
				@inbounds for k in 1:M
					p = perm[k]
					i = pairs_i[p]; j = pairs_j[p]
					out_idx += 1
					edge_src[out_idx] = string(i)
					edge_dst[out_idx] = string(j)
					out_idx += 1
					edge_src[out_idx] = string(j)
					edge_dst[out_idx] = string(i)
				end

			#	Asymmetric Pairs (Next A of the Permutation, Direction Random)
				@inbounds for k in M + 1:M + A
					p = perm[k]
					i = pairs_i[p]; j = pairs_j[p]
					out_idx += 1
					if rand(rng) < 0.5
						edge_src[out_idx] = string(i)
						edge_dst[out_idx] = string(j)
					else
						edge_src[out_idx] = string(j)
						edge_dst[out_idx] = string(i)
					end
				end

		#	Return Edge DataFrame
			return DataFrame(src = edge_src, dst = edge_dst)
	end

#	Tau Statistic Under U|MAN Null
	function tau_statistic(edges::DataFrame;
	                      nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                      weighting::Union{Symbol, AbstractVector} = :RC,
	                      n_samples::Int = 500,
	                      directed::Bool = true,
	                      seed::Int = 20260101,
	                      agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (weights ignored)
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
				(used to count isolates in N for the dyad census)
			weighting::Union{Symbol, AbstractVector}: :RC (default), :transitivity,
				or a length-16 custom indicator vector aligned to DL order.
			n_samples::Int: number of Monte Carlo samples (default = 500)
			directed::Bool: must be true for tau to be meaningful (default = true).
				Passing false throws ArgumentError — tau requires directed dyads.
			seed::Int: RNG seed for reproducibility (default = 20260101)
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			DataFrame: single row with columns
				[tau, observed_weighted_count, null_mean, null_sd, n_samples,
				 M, A, N_null, weighting, weighting_label]
		Notes:
			Computes the tau statistic for a directed network under the U|MAN
			null distribution conditioned on the observed M, A, N dyad census.

			τ = (λ^T T_obs − null_mean) / null_sd

			where the null mean and SD are estimated from `n_samples` Monte Carlo
			draws from U|MAN. Returns tau = 0.0 with null_sd = 0.0 when the null
			distribution is degenerate (e.g., all-null graph: every sample is
			identical to the observed graph).

			A positive tau means the observed graph has more weighting-permitted
			triads than expected under U|MAN; a negative tau means fewer.

			Reuses Section 5's `triad_census` to count triads on each sample.

			Monte Carlo standard error on tau scales as 1/sqrt(n_samples).
			Default n_samples = 500 gives roughly 4% relative error on the
			variance estimate, which is adequate for SMM-style descriptive
			comparisons. Increase to 2000+ for hypothesis testing.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if !directed
				throw(ArgumentError(
					"tau_statistic requires directed=true. Tau is undefined for " *
					"undirected graphs, which have no concept of asymmetric or mutual " *
					"dyads as distinct from each other."
				))
			end
			if n_samples < 2
				throw(ArgumentError("n_samples must be at least 2 for a meaningful SD estimate; got $n_samples"))
			end

		#	Resolve Weighting (Symbol → Vector)
			lambda = _resolve_weighting(weighting)
			weighting_label = weighting isa Symbol ? string(weighting) : "custom"

		#	Build Binary Adjacency on Correct Node Universe
			if nrow(edges) == 0
				#	Empty Edge List: All-Null Graph
					if nodes !== nothing
						N = nodes isa DataFrame ? nrow(nodes) : length(nodes)
					else
						N = 0
					end
					n_pairs = (N * (N - 1)) ÷ 2
					#	Build a Degenerate Result
						return DataFrame(
							tau                       = [0.0],
							observed_weighted_count   = [0.0],
							null_mean                 = [0.0],
							null_sd                   = [0.0],
							n_samples                 = [n_samples],
							M                         = [0],
							A                         = [0],
							N_null                    = [n_pairs],
							weighting                 = [weighting_label],
							weighting_label           = [weighting_label]
						)
			end

		#	Aggregate Multi-Edges
			clean_edges = _aggregate_multi_edges(edges; agg_func = agg_func)

		#	Build Sparse Adjacency on Correct Node Universe
			if nodes === nothing
				adj, _, _ = _edgelist_to_sparse_matrix(clean_edges; weighted = false)
			else
				adj, _, _ = _graph_to_sparse_matrix(clean_edges;
				                                    nodes = nodes,
				                                    weighted = false)
			end

		#	Binarize and Loop-Strip
			_make_directed_simple!(adj)

		#	Observed Dyad Census
			N = size(adj, 1)
			dyads = _dyad_census_counts(adj)
			M_obs = dyads.M
			A_obs = dyads.A
			N_null = dyads.N_null

		#	Observed Triad Census and Weighted Count
		#	Build a node DataFrame to pass to triad_census so it counts on the
		#	correct N (matching the adjacency dimension).
			node_ids = string.(1:N)
			node_df = DataFrame(id = node_ids, label = node_ids)
			#	Build an Edge Frame in Matching ID-Space for triad_census Re-Use
				edge_pairs = findnz(adj)
				edges_obs_for_tc = DataFrame(
					src = [string(i) for i in edge_pairs[1]],
					dst = [string(j) for j in edge_pairs[2]]
				)
				tc_obs = triad_census(edges_obs_for_tc; nodes = node_df, graph_type = :directed)
				T_obs = Float64.(tc_obs.count)

			observed_weighted_count = dot(lambda, T_obs)

		#	Degenerate Case: U|MAN Distribution Is a Single Graph
		#	When M = 0 and A = 0, the only graph in the U|MAN distribution is
		#	the all-null graph. When the total number of pairs equals M
		#	(complete graph of mutual ties), the only graph is the complete
		#	mutual graph. In both cases the MC variance is exactly 0 and tau
		#	is undefined; return 0.0 by convention.
			n_pairs = (N * (N - 1)) ÷ 2
			if (M_obs == 0 && A_obs == 0) || (M_obs == n_pairs)
				return DataFrame(
					tau                       = [0.0],
					observed_weighted_count   = [observed_weighted_count],
					null_mean                 = [observed_weighted_count],
					null_sd                   = [0.0],
					n_samples                 = [0],
					M                         = [M_obs],
					A                         = [A_obs],
					N_null                    = [N_null],
					weighting                 = [weighting_label],
					weighting_label           = [weighting_label]
				)
			end

		#	Monte Carlo Sampling
			rng = Random.Xoshiro(seed)
			weighted_counts = Vector{Float64}(undef, n_samples)
			@inbounds for s in 1:n_samples
				#	Sample U|MAN Edges
					sample_edges = _sample_uman_edges(N, M_obs, A_obs, rng)

				#	Compute Triad Census on the Sample
				#	Pass the same node_df so the census uses the same N
					tc_sample = triad_census(sample_edges; nodes = node_df, graph_type = :directed)
					T_sample = Float64.(tc_sample.count)

				#	Weighted Count
					weighted_counts[s] = dot(lambda, T_sample)
			end

		#	Compute Null Statistics
			null_mean = mean(weighted_counts)
			null_sd   = std(weighted_counts)

		#	Compute Tau
			tau = null_sd > 0 ? (observed_weighted_count - null_mean) / null_sd : 0.0

		#	Assembling Result
			return DataFrame(
				tau                       = [tau],
				observed_weighted_count   = [observed_weighted_count],
				null_mean                 = [null_mean],
				null_sd                   = [null_sd],
				n_samples                 = [n_samples],
				M                         = [M_obs],
				A                         = [A_obs],
				N_null                    = [N_null],
				weighting                 = [weighting_label],
				weighting_label           = [weighting_label]
			)
	end
	@doc raw"""
	**Description**
	Compute the tau statistic for the triad census conditioned on the observed
	dyad census (U|MAN null model). Tau measures how far a weighted summary of
	the observed triad distribution deviates from its expectation under the
	null, in standardized units:

	$$\tau = \frac{\lambda^T T_{\mathrm{obs}} - E[\lambda^T T \mid M, A, N]}{\sqrt{\mathrm{Var}(\lambda^T T \mid M, A, N)}}$$

	where $T_{\mathrm{obs}}$ is the observed length-16 Davis-Leinhardt triad
	count vector, $M, A, N$ are the counts of mutual, asymmetric, and null
	dyads, and $\lambda$ is a length-16 weighting vector indicating which
	triad classes count toward the substantive theory being tested.

	The default weighting `:RC` (ranked clusters) tests for hierarchy of
	mutually-tied clusters with between-cluster ties pointing in one direction.
	The permitted triads are 003, 102, 021D, 021U, 030T, 120D, 120U, 300.

	**Usage**
	`tau_statistic(edges::DataFrame; nodes=nothing, weighting=:RC, n_samples=500, seed=20260101, directed=true)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`
	  (ignored — binary input).
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe.
	  Pass to include isolates in the dyad census.
	- `weighting::Union{Symbol,AbstractVector}`: `:RC` (default), `:transitivity`,
	  or a custom length-16 vector aligned to Davis-Leinhardt order.
	- `n_samples::Int`: Monte Carlo sample size (default `500`).
	- `directed::Bool`: Must be `true`. Tau is undefined for undirected graphs.
	- `seed::Int`: RNG seed (default `20260101`).

	**Details**
	The U|MAN expectation and variance have closed-form expressions
	(Holland & Leinhardt 1976; Wasserman & Faust 1994, Chapter 14), but
	transcribing them is error-prone. This implementation instead estimates
	$E$ and $\mathrm{Var}$ by Monte Carlo sampling from the U|MAN distribution.

	The sampler is direct: for each of `n_samples` independent draws, the
	$\binom{N}{2}$ unordered pairs are randomly permuted, the first $M$ are
	assigned to be mutual, the next $A$ are assigned to be asymmetric with
	random direction, and the rest are null. This is exactly U|MAN-uniform
	by construction — no MCMC burn-in or autocorrelation.

	Monte Carlo error on tau scales as $1/\sqrt{K}$ where $K$ is `n_samples`.
	Default $K = 500$ gives about 4% relative error on the variance estimate,
	adequate for descriptive use. Increase to 2000+ for hypothesis testing.

	Returns tau = 0.0 with null_sd = 0.0 when the U|MAN distribution is
	degenerate — e.g., an all-null graph (every sample identical to the
	observed graph).

	**Value**
	A single-row `DataFrame` with columns:
	- `:tau` — standardized deviation
	- `:observed_weighted_count` — $\lambda^T T_{\mathrm{obs}}$
	- `:null_mean`, `:null_sd` — MC null statistics
	- `:n_samples` — number of MC samples used (0 in the degenerate case)
	- `:M`, `:A`, `:N_null` — observed dyad census
	- `:weighting`, `:weighting_label` — record of which weighting was used

	**References**
	- Holland PW, Leinhardt S (1976). "Local structure in social networks."
	  *Sociological Methodology* 7: 1–45.
	- Davis JA, Leinhardt S (1972). "The structure of positive interpersonal
	  relations in small groups."
	- Wasserman S, Faust K (1994). *Social Network Analysis: Methods and
	  Applications*, Chapter 14. Cambridge University Press.

	**See Also**
	`triad_census`, `reciprocity`
	""" tau_statistic

####################################################
#   SECTION 8: STRUCTURAL EQUIVALENCE BLOCKMODEL   #
####################################################

# ====================================================================
# Partition the actor set into structural-equivalence blocks via
# hierarchical agglomerative clustering on the row+column profile
# correlation matrix.
#
# Two actors are structurally equivalent if they have the same pattern
# of ties to and from every other actor. We measure (in)equivalence as
# 1 − r(i, j), where r is the Pearson correlation between actors'
# profiles. The profile is:
#   - directed:   [row_i; col_i] of the adjacency, length 2N
#   - undirected: row_i, length N (col is identical for symmetric A)
#
# After building the N×N distance matrix, the actors are clustered
# with Ward's linkage and the dendrogram is cut to produce up to
# `n_blocks` blocks. Isolates (actors with zero ties) are excluded
# from the clustering and assigned block label 0 on return.
#
# Output convention:
#   - block 0      : isolates
#   - block 1..k   : non-isolate blocks, k ≤ n_blocks
# This convention lets the Rand index between two partitions correctly
# count isolate-misclassification as a disagreement.
#
# Public API:
#   structural_equivalence_blockmodel — per-node block assignments
#
# Internal helpers:
#   _build_profile_matrix     — per-actor profile (row or row+col)
#   _profile_correlation_dist — N×N distance matrix from profiles
#   _ward_linkage             — Lance-Williams agglomerative clustering
#   _cut_dendrogram           — cut dendrogram at k clusters → labels
# ====================================================================

#	Helper Function for structural_equivalence_blockmodel: Build Profile Matrix
	function _build_profile_matrix(adj::SparseMatrixCSC{<:Real, Int};
	                              directed::Bool = true)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency (binary recommended)
			directed::Bool: directed → profile is [row; col], length 2N;
				undirected → profile is just row, length N (col is identical)
		Returns:
			Matrix{Float64}: N × profile_length matrix; row i is actor i's profile
		Notes:
			Self-ties (diagonal entries) are zeroed before extracting profiles
			so they cannot bias the correlation downstream. The original
			adjacency is not modified — we work with a dense copy.

			Profile dimensionality is 2N for directed, N for undirected.
			For Phase 0 corpus (largest N = 1347), the 2N case gives a
			profile of length 2694 per actor — comfortable in memory.
		"""

		#	Dense Copy with Diagonal Zeroed
			N = size(adj, 1)
			A = Matrix{Float64}(adj)
			@inbounds for i in 1:N
				A[i, i] = 0.0
			end

		#	Build Profile Matrix
			if directed
				#	Profile = [row; col]
					profile_len = 2 * N
					P = Matrix{Float64}(undef, N, profile_len)
					@inbounds for i in 1:N
						P[i, 1:N]            .= view(A, i, :)
						P[i, N + 1:2 * N]    .= view(A, :, i)
					end
			else
				#	Profile = row only (col is identical for symmetric A)
				#	Defensively symmetrize first in case caller didn't
					A = max.(A, A')
					profile_len = N
					P = Matrix{Float64}(undef, N, profile_len)
					@inbounds for i in 1:N
						P[i, :] .= view(A, i, :)
					end
			end

		#	Return Profile Matrix
			return P
	end

#	Helper Function for structural_equivalence_blockmodel: Correlation Distance
	function _profile_correlation_dist(P::AbstractMatrix{Float64})
		"""
		Args:
			P::AbstractMatrix{Float64}: N × profile_length profile matrix
		Returns:
			Matrix{Float64}: N × N symmetric distance matrix, diagonal = 0
		Notes:
			Pairwise Pearson correlation between rows of P, converted to
			distance via d(i,j) = 1 - r(i,j). Bounded in [0, 2].

			NaN handling: if either actor's profile has zero variance (all
			entries equal to the row mean — typically because the actor has
			no ties), the correlation is undefined. We replace such NaNs
			with the maximum distance 2.0 so undefined-correlation actors
			are treated as maximally different from everyone. In practice
			this helper should only be called on profiles of NON-isolated
			actors (since the public function drops isolates first), but
			the NaN guard is retained as a defensive measure for
			near-isolates whose profile variance is essentially zero.
		"""

		#	Dimensions
			N = size(P, 1)
			if N == 0
				return Matrix{Float64}(undef, 0, 0)
			end

		#	Row-Center the Profile Matrix
			row_means = mean(P, dims = 2)
			Pc = P .- row_means

		#	Row L2-Norms
			row_norms = vec(sqrt.(sum(Pc .^ 2, dims = 2)))

		#	Pairwise Cosine of Centered Profiles == Pearson Correlation
		#	C = Pc * Pc' element-wise divided by outer product of norms
			C = Pc * Pc'
			@inbounds for i in 1:N
				for j in 1:N
					denom = row_norms[i] * row_norms[j]
					if denom == 0
						#	One or Both Profiles Are Constant → Correlation Undefined
							C[i, j] = (i == j) ? 1.0 : NaN
					else
						C[i, j] /= denom
					end
				end
			end

		#	Distance = 1 - r, NaN → 2.0
			D = 1.0 .- C
			@inbounds for idx in eachindex(D)
				if isnan(D[idx])
					D[idx] = 2.0
				end
			end

		#	Force Symmetry and Zero Diagonal (Defensive Against Float Noise)
			D = (D .+ D') ./ 2.0
			@inbounds for i in 1:N
				D[i, i] = 0.0
			end

		#	Clamp to [0, 2] in Case of Numerical Drift
			clamp!(D, 0.0, 2.0)

		#	Return Distance Matrix
			return D
	end

#	Helper Function for structural_equivalence_blockmodel: Ward's Linkage
	function _ward_linkage(D::Matrix{Float64})
		"""
		Args:
			D::Matrix{Float64}: N × N symmetric distance matrix, zero diagonal
		Returns:
			NamedTuple: (merges::Matrix{Int}, heights::Vector{Float64})
				merges has size (N-1) × 2: each row gives the two cluster IDs
					that were merged at that step.
				heights[k] is the Ward distance at which merge k happened.
		Notes:
			Standard agglomerative Ward's linkage using the Lance-Williams
			update formula on squared dissimilarities. Cluster IDs follow
			the SciPy/R hclust convention:
				- 1..N      : original singleton clusters (the N actors)
				- N+1..2N-1 : merged clusters created at each step
			At step k (k = 1..N-1), merges[k, :] are the two IDs being
			combined; the new cluster receives ID N + k.

			Complexity is O(N^3) in time and O(N^2) in space, dominated by
			the distance update at each merge. For Phase 0 corpus sizes
			(N ≤ 1347 for Balikatan) this runs in a few seconds.
		"""

		#	Dimensions
			N = size(D, 1)
			@assert size(D, 2) == N "distance matrix must be square"

		#	Quick Returns for Trivial Cases
			if N <= 1
				return (merges = Matrix{Int}(undef, 0, 2), heights = Float64[])
			end

		#	Working Distance Matrix (Squared Distances for Ward)
		#	Ward's update is exact on squared Euclidean distances. We use
		#	squared correlation-distances as a heuristic — this is the same
		#	convention as R's hclust(method='ward.D2') and scipy's
		#	linkage(method='ward').
			Dsq = D .^ 2

		#	Working State
		#	  active[i]: whether singleton/composite cluster i is still active
		#	  cluster_sizes[i]: number of original points in cluster i
		#	  cluster_ids[i]: cluster identifier (for output)
		#	Cluster IDs 1..N are the initial singletons.
		#	Each merge creates a new cluster with ID = N + step_number.
			max_clusters = 2 * N - 1
			active        = falses(max_clusters)
			cluster_sizes = zeros(Int, max_clusters)
			#	Distance Matrix Indexed by Cluster ID (Up to max_clusters)
			#	Lazy expansion: only store distances between currently active
			#	clusters. We resize as merges happen.
				dist = fill(Inf, max_clusters, max_clusters)

			#	Initialize Singletons
				@inbounds for i in 1:N
					active[i] = true
					cluster_sizes[i] = 1
					for j in 1:N
						if i != j
							dist[i, j] = Dsq[i, j]
						end
					end
				end

		#	Output Storage
			merges  = Matrix{Int}(undef, N - 1, 2)
			heights = Vector{Float64}(undef, N - 1)

		#	Main Agglomeration Loop: N - 1 Merges
			@inbounds for step in 1:N - 1
				#	Find the Pair (a, b) of Active Clusters with Minimum Distance
					best_a = -1
					best_b = -1
					best_d = Inf
					for a in 1:max_clusters
						active[a] || continue
						for b in a + 1:max_clusters
							active[b] || continue
							if dist[a, b] < best_d
								best_d = dist[a, b]
								best_a = a
								best_b = b
							end
						end
					end
					@assert best_a > 0 && best_b > 0 "no active pair found at step $step"

				#	Record the Merge
					new_id = N + step
					merges[step, 1] = best_a
					merges[step, 2] = best_b
					heights[step]   = sqrt(best_d)  # report Ward distance (un-squared)

				#	Apply Lance-Williams Update for Ward
				#	For Ward (squared-distance form):
				#	  d²(a ∪ b, k) = [(n_a + n_k) d²(a, k) + (n_b + n_k) d²(b, k)
				#	                  - n_k d²(a, b)] / (n_a + n_b + n_k)
					n_a = cluster_sizes[best_a]
					n_b = cluster_sizes[best_b]
					new_size = n_a + n_b

					active[new_id]        = true
					cluster_sizes[new_id] = new_size
					active[best_a]        = false
					active[best_b]        = false

					for k in 1:max_clusters
						active[k] || continue
						k == new_id && continue
						n_k = cluster_sizes[k]
						d_ak = dist[best_a, k]
						d_bk = dist[best_b, k]
						d_ab = best_d
						new_d = ((n_a + n_k) * d_ak + (n_b + n_k) * d_bk - n_k * d_ab) /
						        (n_a + n_b + n_k)
						dist[new_id, k] = new_d
						dist[k, new_id] = new_d
					end
			end

		#	Return Merge Sequence and Heights
			return (merges = merges, heights = heights)
	end

#	Helper Function for structural_equivalence_blockmodel: Cut Dendrogram to k Clusters
	function _cut_dendrogram(merges::Matrix{Int}, N::Int, k::Int)
		"""
		Args:
			merges::Matrix{Int}: (N-1) × 2 merge sequence from _ward_linkage
			N::Int: number of original points (singletons with IDs 1..N)
			k::Int: target number of clusters (will be clamped to [1, N])
		Returns:
			Vector{Int}: length-N vector of cluster labels in 1..actual_k
		Notes:
			To get k clusters, we undo the last (N - k) merges and walk the
			merge tree to find which original points belong to each remaining
			cluster.

			If k > N or merges is shorter than expected, actual_k may be less
			than the requested k (e.g., fewer than k structural equivalence
			classes exist). The labels are always in 1..actual_k.

			Labels are assigned in order of the smallest member-point index:
			the cluster containing actor 1 is labeled 1, the cluster
			containing the smallest-indexed actor not in cluster 1 is
			labeled 2, and so on. This gives deterministic, reproducible
			labels for Rand-index comparison.
		"""

		#	Clamp k
			k = max(1, min(k, N))

		#	Number of Merges to Apply: We Want to Undo the Last (N - k)
		#	Equivalently, apply only the first (N - k) merges.
			n_merges_to_apply = N - k

		#	Union-Find on Singletons 1..N Plus Composite IDs N+1..N+steps
		#	We track which singleton belongs to which final cluster ID.
		#	Initialize: singleton i is in its own cluster with rep i.
			max_id = 2 * N - 1
			parent = collect(1:max_id)

			function find_root(x::Int)
				while parent[x] != x
					parent[x] = parent[parent[x]]  # path compression
					x = parent[x]
				end
				return x
			end

		#	Apply Merges 1..n_merges_to_apply
			@inbounds for step in 1:n_merges_to_apply
				a = merges[step, 1]
				b = merges[step, 2]
				new_id = N + step
				#	Point Both a and b to new_id
					parent[find_root(a)] = new_id
					parent[find_root(b)] = new_id
			end

		#	Determine Cluster Membership of Each Singleton
			roots = [find_root(i) for i in 1:N]

		#	Relabel to 1..actual_k in Order of First Appearance
		#	(So Cluster Containing Actor 1 Is Labeled 1)
			label_of_root = Dict{Int, Int}()
			labels = Vector{Int}(undef, N)
			next_label = 1
			@inbounds for i in 1:N
				r = roots[i]
				if !haskey(label_of_root, r)
					label_of_root[r] = next_label
					next_label += 1
				end
				labels[i] = label_of_root[r]
			end

		#	Return Labels
			return labels
	end

#	Structural-Equivalence Blockmodel via Hierarchical Clustering
	function structural_equivalence_blockmodel(edges::DataFrame;
	                                          nodes::Union{Nothing, DataFrame, AbstractVector{<:AbstractString}} = nothing,
	                                          n_blocks::Int = 8,
	                                          directed::Bool = true,
	                                          agg_func::Function = sum)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (weights ignored)
			nodes::Union{Nothing, DataFrame, Vector}: optional node universe
			n_blocks::Int: target number of non-isolate blocks (default = 8,
				matching SMM's depth=3 CONCOR equivalent)
			directed::Bool: profile is [row; col] when true, just row when false
			agg_func::Function: aggregation for parallel edges (default = sum)
		Returns:
			DataFrame: columns [node, block]
				block 0           — isolate (no ties)
				block 1..k        — non-isolate block assignments
				where k = min(n_blocks, number of non-isolate equivalence classes)
		Notes:
			Computes structural equivalence between every pair of non-isolate
			actors via the Pearson correlation of their profiles, then
			hierarchically clusters using Ward's linkage and cuts the
			dendrogram to produce up to `n_blocks` blocks.

			Isolates (actors with no ties in the binarized adjacency) are
			excluded from the clustering computation and assigned block 0
			on output. This convention lets the Rand index between two
			partitions correctly count isolate-misclassification as a
			disagreement: if one partition has actor i as isolate (block 0)
			and the other has i in a tied block, the pair (i, j) with j also
			classified differently is counted in the disagreement.

			The profile of actor i is determined by `directed`:
				directed = true  → [row_i; col_i] of A, length 2N
				directed = false → row_i of A (symmetrized), length N
			where A has its diagonal zeroed to exclude self-ties from the
			profile.

			For Phase 0 corpus sizes (largest N = 1347) this runs in a few
			seconds. Hierarchical clustering is O(N^3); the profile matrix
			and distance matrix are O(N^2) in memory.

			The function may return fewer than `n_blocks` blocks if the
			network has fewer than `n_blocks` distinct structural equivalence
			classes among non-isolates. This is expected and not an error.
		"""

		#	Validation
			if !hasproperty(edges, :src) || !hasproperty(edges, :dst)
				throw(ArgumentError("edges DataFrame must have src and dst columns"))
			end
			if n_blocks < 1
				throw(ArgumentError("n_blocks must be >= 1; got $n_blocks"))
			end

		#	Handle Empty Edge List
			if nrow(edges) == 0
				if nodes !== nothing
					node_ids = nodes isa DataFrame ? nodes.id : collect(nodes)
					return DataFrame(node = node_ids, block = zeros(Int, length(node_ids)))
				else
					return DataFrame(node = String[], block = Int[])
				end
			end

		#	Binarize and Build Adjacency
			clean_edges = DataFrame(src = edges.src, dst = edges.dst)
			clean_edges.weight = ones(Float64, nrow(clean_edges))
			clean_edges = _aggregate_multi_edges(clean_edges; agg_func = agg_func)

			if nodes === nothing
				adj, _, idx_to_node = _edgelist_to_sparse_matrix(clean_edges; weighted = false)
			else
				adj, _, idx_to_node = _graph_to_sparse_matrix(clean_edges;
				                                              nodes = nodes,
				                                              weighted = false)
			end

		#	Drop Self-Loops in Adjacency
			N = size(adj, 1)
			@inbounds for i in 1:N
				adj[i, i] = 0
			end
			dropzeros!(adj)

		#	Identify Isolates (No Ties In or Out)
		#	For undirected, symmetrize first so "tie" means "any incident edge"
			if !directed
				adj_for_isolate_check = max.(adj, adj')
			else
				adj_for_isolate_check = adj .+ adj'
			end
			out_deg = vec(sum(adj_for_isolate_check, dims = 2))
			isolate_mask = (out_deg .== 0)
			n_isolates = count(isolate_mask)
			n_active = N - n_isolates

		#	Extract Node IDs in Adjacency Order
			node_col = idx_to_node isa DataFrame ? idx_to_node.id : idx_to_node
			node_col = collect(node_col)

		#	Edge Case: No Non-Isolates
			if n_active == 0
				return DataFrame(node = node_col, block = zeros(Int, N))
			end

		#	Edge Case: Only One Non-Isolate
			if n_active == 1
				blocks = zeros(Int, N)
				blocks[.!isolate_mask] .= 1
				return DataFrame(node = node_col, block = blocks)
			end

		#	Build Profile Matrix on the FULL Adjacency, Then Subset
		#	The profile of a non-isolate actor still references the
		#	full N-dimensional ambient space (including isolate positions
		#	as zero entries), which is the correct treatment: an actor's
		#	structural position is defined relative to ALL potential alters.
			P_full = _build_profile_matrix(adj; directed = directed)

		#	Restrict Profiles to Non-Isolates for Clustering
			active_indices = findall(.!isolate_mask)
			P_active = P_full[active_indices, :]

		#	Distance Matrix on Active Profiles
			D_active = _profile_correlation_dist(P_active)

		#	Hierarchical Clustering with Ward's Linkage
			link = _ward_linkage(D_active)

		#	Cut Dendrogram to n_blocks Clusters
			cluster_labels = _cut_dendrogram(link.merges, n_active, n_blocks)

		#	Assemble Output: Isolates → 0, Active → 1..k
			blocks = zeros(Int, N)
			for (out_idx, active_idx) in enumerate(active_indices)
				blocks[active_idx] = cluster_labels[out_idx]
			end

		#	Assembling Result
			return DataFrame(node = node_col, block = blocks)
	end
	@doc raw"""
	**Description**
	Partition the actor set into structural-equivalence blocks via
	hierarchical agglomerative clustering on the profile correlation matrix.

	Two actors are structurally equivalent if they have identical patterns of
	ties to and from every other actor. We measure (in)equivalence as
	$d_{ij} = 1 - r_{ij}$ where $r_{ij}$ is the Pearson correlation between
	the two actors' profiles. For directed networks, the profile is the
	concatenation of the actor's row and column in the adjacency. For
	undirected networks, just the row (which equals the column).

	Actors are clustered with Ward's linkage and the dendrogram is cut to
	produce up to `n_blocks` blocks. Isolates (actors with no ties) are
	excluded from the clustering and assigned block `0` on output.

	**Usage**
	`structural_equivalence_blockmodel(edges::DataFrame; nodes=nothing, n_blocks=8, directed=true, agg_func=sum)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optionally `:weight`
	  (ignored — binary input).
	- `nodes::Union{Nothing,DataFrame,Vector}`: Optional node universe.
	- `n_blocks::Int`: Target number of non-isolate blocks. Default `8` to
	  match Smith, Morgan, & Moody (2022)'s depth=3 CONCOR convention.
	- `directed::Bool`: When `true`, profile is `[row; col]`; when `false`,
	  profile is the (symmetrized) row.
	- `agg_func::Function`: Aggregation for parallel edges (default `sum`).

	**Details**
	The profile correlation matrix is converted to a distance matrix via
	$d = 1 - r$, bounded in $[0, 2]$. Ward's linkage is applied via the
	Lance-Williams update formula on squared distances — this matches the
	`method='ward'` convention in SciPy and `ward.D2` in R.

	Output convention:
	- `block = 0` indicates an isolated actor (no ties).
	- `block` $\in \{1, 2, \ldots, k\}$ for non-isolated actors, where
	  $k \leq$ `n_blocks` is the actual number of clusters found.

	The number of returned blocks may be less than `n_blocks` if the network
	has fewer distinct structural equivalence classes among non-isolates.

	The Rand index between partitions from two networks (e.g., true vs.
	imputed) can be computed by passing the `:block` columns directly to
	`rand_index`.

	**Value**
	A `DataFrame` with columns `:node` (actor identifier) and `:block`
	(integer assignment in $\{0, 1, \ldots, k\}$).

	**References**
	- White HC, Boorman SA, Breiger RL (1976). "Social structure from multiple
	  networks. I. Blockmodels of roles and positions." *American Journal of
	  Sociology* 81(4): 730–780.
	- Burt RS (1976). "Positions in networks." *Social Forces* 55(1): 93–122.
	- Ward JH Jr. (1963). "Hierarchical grouping to optimize an objective
	  function." *Journal of the American Statistical Association* 58:236–244.

	**See Also**
	`rand_index`
	""" structural_equivalence_blockmodel

#   Exports (Foundation Section)
    export gini_coefficient,
           centralization,
           rand_index,
           transform_distance_weights,
           in_degree,
           out_degree,
           total_degree,
           freeman_degree_normalization,
           freeman_degree_centralization,
           closeness_centrality,
           betweenness_centrality,
           mean_inverse_distance,
           bonacich_centrality,
           largest_component_proportion,
           reciprocity,
           local_weighted_reciprocity,
           local_clustering_coefficient,
           global_clustering_coefficient,
		   recommend_L,
		   _triad_census_layered,
           triad_census,
           largest_bicomponent_proportion,
           tau_statistic,
           structural_equivalence_blockmodel

end # module network_statistics