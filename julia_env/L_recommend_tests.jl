#	=====================================================================
#	TEST SCRIPT: Validate analytic recommend_L against empirical baseline
#	=====================================================================
#	This file contains the OLD empirical _select_L_by_stability (renamed
#	with _empirical suffix) restored as a test-only comparison tool, plus
#	a side-by-side comparison function and an orchestrator that runs the
#	comparison on the calibration networks.
#	
#	After the analytic method is validated, this file can be archived.
#	Nothing in network_statistics.jl depends on it.
#	=====================================================================

#   Pulling-In Network_Credible_Inteverals & Activating Local Environment
    cd("/mnt/d/GitHub_Repositories/Network_Credible_Intervals")
    using Pkg
    Pkg.activate("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/julia_env")
    Pkg.status()

#   Checking that Multi-Threaded Implementation is Running
    get(ENV, "JULIA_NUM_THREADS", "not set")
    Threads.nthreads() 

#   Packages
    using CairoMakie
    using DataFrames
    using Statistics
    using Printf
    using SparseArrays
    using ProgressMeter
    using Network_Credible_Intervals

#######################
#   IMPORT DATASETS   #
#######################

#	Setting Up Data Directory
	data_dir = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks"

#	Find All GraphML Files in Directory (Sorted for Deterministic Ordering)
	graphml_files = sort(filter(f -> endswith(f, ".graphml"), readdir(data_dir)))
	println("Found $(length(graphml_files)) GraphML files")

#	Pre-Allocate Storage Dictionary
	networks = Dict{String, NamedTuple}()

#	Loop Over Files and Load Each One
	for filename in graphml_files
		#	Build Full Path and Derive Short Network Name
			filepath = joinpath(data_dir, filename)
			network_name = replace(filename, ".graphml" => "")

		#	Load Network
			result = load_graphml(filepath)
			networks[network_name] = result

		#	Report Summary
			n_nodes = nrow(result.nodes)
			n_edges = nrow(result.edges)
			directed_str = result.metadata.directed ? "directed" : "undirected"
			weighted_str = get(result.metadata, :weighted, false) ? "weighted" : "unweighted"
			println("  Loaded $network_name: $n_nodes nodes, $n_edges edges ($directed_str, $weighted_str)")
	end

#	Sanity Check: Total Networks Loaded
	println("\nLoaded $(length(networks)) networks into `networks` dictionary")
	println("Access individual networks via networks[\"<name>\"], e.g.:")
	println("  networks[\"moreno_highschool_weighted\"].edges")
	println("  networks[\"moreno_highschool_weighted\"].nodes")
	println("  networks[\"moreno_highschool_weighted\"].metadata")

#################
#   FUNCTIONS   #
#################

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

#	OLD EMPIRICAL PIPELINE (Restored for Comparison Only)

#	Test Helper: Estimate τ-Bounds from Observed Weights (Quantile Method)
	function _estimate_tau_bounds_empirical(edges::DataFrame;
											nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
											graph_type::Symbol=:directed,
											lo::Float64=0.01,
											hi::Float64=0.99)
		"""
		Args:
			edges::DataFrame: expects :src, :dst, :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			graph_type::Symbol: :directed or :undirected
			lo, hi::Float64: quantile bounds (default 0.01 and 0.99)
		Returns:
			NamedTuple: (tau_min::Float64, tau_max::Float64)
		Notes:
			Quantile-based τ bound estimation. For :undirected, computed on
			summed symmetric weights (W + W'). For :directed, on the directed
			weights as-is. Lower bound clipped by eps() for numerical safety.

			TEST-ONLY function. Mirrors the original _estimate_tau_bounds that
			was used by the empirical recommend_L pipeline before the analytic
			version replaced it.
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

#	Test Helper: Quick Heuristic for L (Points-Per-Decade)
	function _suggest_L_quick_empirical(tau_min::Float64,
										tau_max::Float64;
										points_per_decade::Int=8,
										L_min::Int=8,
										L_max::Int=64)
		"""
		Args:
			tau_min::Float64: lower τ bound
			tau_max::Float64: upper τ bound
			points_per_decade::Int: target log-decade density (default 8)
			L_min::Int, L_max::Int: clamp range
		Returns:
			Int: suggested L
		Notes:
			L = ceil(points_per_decade * log10(τ_max / τ_min)), clamped.
			Falls back to L_min for degenerate input (τ_max ≤ τ_min).

			TEST-ONLY function.
		"""

		ratio   = tau_max <= tau_min ? 1.0 : (tau_max / tau_min)
		decades = log10(ratio)
		L       = ceil(Int, points_per_decade * max(decades, 0.0))
		return clamp(max(L, L_min), L_min, L_max)
	end

#	Test Helper: Empirical Stability Scan (AUMC-Based)
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
			tau_min, tau_max: τ bounds or :auto (quantile-derived)
			L_grid: candidate L values to evaluate
			tol: AUMC stability tolerance (default 1e-3)
			parallel: parallelize candidates (default false)
			verbose, show_progress, inner_show_progress: diagnostic flags
		Returns:
			NamedTuple: (L_best::Int, table::DataFrame, census_at_L_best::NamedTuple)
		Notes:
			TEST-ONLY function. Restored empirical AUMC-stability scan from
			the original recommend_L pipeline. Runs triad_census(weighted=true)
			at each candidate L, computes the 16-class AUMC vector, and picks
			the smallest L where AUMC stabilizes within tol.

			Cached per-candidate census so we can return the layered result
			at L_best without re-running.
		"""

		#	Validation
			@assert graph_type in (:directed, :undirected) "graph_type must be :directed or :undirected"
			if graph_type == :undirected
				@assert !reciprocity_collapse "reciprocity_collapse applies only when graph_type == :directed"
			end

		#	Resolve τ Bounds
			local_tau_min = 0.0
			local_tau_max = 0.0
			if tau_min === :auto || tau_max === :auto
				tb            = _estimate_tau_bounds_empirical(edges; nodes=nodes, graph_type=graph_type)
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

#	Test Helper: Empirical recommend_L
	function recommend_L_empirical(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
									graph_type::Symbol=:directed,
									reciprocity_collapse::Bool=false,
									tau_min::Union{Float64,Symbol}=:auto,
									tau_max::Union{Float64,Symbol}=:auto,
									points_per_decade::Int=8,
									L_min::Int=8,
									L_max::Int=64,
									tol::Float64=1e-3,
									parallel::Bool=false,
									verbose::Bool=false,
									show_progress::Bool=true,
									inner_show_progress::Bool=false)
		"""
		Args:
			Same as the original recommend_L (empirical version).
		Returns:
			NamedTuple: (L::Int, tau_min::Float64, tau_max::Float64,
			             scan::DataFrame, census::NamedTuple)
		Notes:
			TEST-ONLY function. Restored empirical recommend_L using quantile
			τ bounds and AUMC stability scan. Used by test_compare_recommend_L
			to provide a baseline against the analytic method.
		"""

		#	Resolve τ Bounds
			if tau_min === :auto || tau_max === :auto
				tb           = _estimate_tau_bounds_empirical(edges; nodes=nodes, graph_type=graph_type)
				resolved_min = tau_min === :auto ? tb.tau_min : Float64(tau_min)
				resolved_max = tau_max === :auto ? tb.tau_max : Float64(tau_max)
			else
				resolved_min = Float64(tau_min)
				resolved_max = Float64(tau_max)
			end

		#	Quick L Guess
			L_guess = _suggest_L_quick_empirical(resolved_min, resolved_max;
													points_per_decade=points_per_decade,
													L_min=L_min, L_max=L_max)

		#	Build L Grid
			L_grid = unique(sort(Int[max(L_min, div(L_guess, 2)),
									max(L_min, round(Int, 0.75 * L_guess)),
									L_guess,
									min(L_max, round(Int, 1.25 * L_guess)),
									min(L_max, 2 * L_guess)]))

			if verbose
				println("  [empirical] recommend_L: τ bounds = [$resolved_min, $resolved_max], " *
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

		return (L       = sel.L_best,
				tau_min = resolved_min,
				tau_max = resolved_max,
				scan    = sel.table,
				census  = sel.census_at_L_best)
	end

#	COMPARISON FUNCTIONS

#	Compute AUMC Vector from a Layered Census Result
	function _aumc_vector_from_census(census_result::NamedTuple)
		"""
		Args:
			census_result::NamedTuple: output of triad_census(weighted=true)
		Returns:
			Vector{Float64}: 16-class AUMC vector in DL order
		Notes:
			Helper for the AUMC consistency check.
		"""
		labels = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
		          "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]
		s = census_result.summary
		return [begin
					v = s[s.triad .== lab, :AUMC_density]
					isempty(v) ? 0.0 : v[1]
				end for lab in labels]
	end

#	Compare Analytic vs Empirical recommend_L on a Single Network
	function test_compare_recommend_L(edges::DataFrame;
										nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
										graph_type::Symbol=:directed,
										reciprocity_collapse::Bool=false,
										label::String="<unnamed>",
										run_aumc_check::Bool=true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			nodes, graph_type, reciprocity_collapse: graph specification
			label::String: descriptive name for diagnostic output
			run_aumc_check::Bool: also run a layered census at the analytic
				recommendation and compare AUMC vectors against the empirical
				census (default true)
		Returns:
			NamedTuple with:
				- label::String
				- analytic::NamedTuple: full recommend_L output
				- empirical::NamedTuple: full recommend_L_empirical output
				- t_analytic::Float64, t_empirical::Float64: wall-clock seconds
				- tau_min_ratio, tau_max_ratio::Float64: empirical / analytic
				- L_diff::Int: empirical L − analytic L
				- aumc_max_abs_diff::Float64: max |Δ| between AUMC vectors
				  (NaN if run_aumc_check=false)
				- aumc_cosine_sim::Float64: cosine similarity (NaN if skipped)
		Notes:
			The empirical method already runs a layered census at L_best
			(returned as :census). The analytic method does not, so the
			AUMC check runs one extra layered census at the analytic
			recommendation. Total cost is therefore approximately
			(empirical scan time) + (one analytic census).
		"""

		println("=" ^ 70)
		println("Comparing recommend_L: $label")
		println("Graph: $(nrow(edges)) edges, graph_type=$graph_type, collapse=$reciprocity_collapse")
		println("=" ^ 70)

		#	--- Analytic Method ---
			println("  Running analytic recommend_L...")
			t0 = time()
			rec_analytic = recommend_L(edges;
										nodes                = nodes,
										graph_type           = graph_type,
										reciprocity_collapse = reciprocity_collapse,
										verbose              = false)
			t_analytic = time() - t0
			println(@sprintf("    Analytic done: %.3f s", t_analytic))
			println(@sprintf("      L = %d, τ ∈ [%.4g, %.4g], T_max = %d, valid = %s",
								rec_analytic.L, rec_analytic.tau_min, rec_analytic.tau_max,
								rec_analytic.T_max, string(rec_analytic.valid)))

		#	--- Empirical Method ---
			println("  Running empirical recommend_L...")
			t0 = time()
			rec_empirical = recommend_L_empirical(edges;
													nodes                = nodes,
													graph_type           = graph_type,
													reciprocity_collapse = reciprocity_collapse,
													parallel             = false,
													verbose              = false,
													show_progress        = false)
			t_empirical = time() - t0
			println(@sprintf("    Empirical done: %.3f s", t_empirical))
			println(@sprintf("      L = %d, τ ∈ [%.4g, %.4g]",
								rec_empirical.L, rec_empirical.tau_min, rec_empirical.tau_max))

		#	--- Comparison Metrics ---
			tau_min_ratio = rec_empirical.tau_min / rec_analytic.tau_min
			tau_max_ratio = rec_empirical.tau_max / rec_analytic.tau_max
			L_diff        = rec_empirical.L - rec_analytic.L

			println()
			println("  Bound comparison:")
			println(@sprintf("    τ_min: analytic = %.4g, empirical = %.4g, ratio (emp/ana) = %.3f",
								rec_analytic.tau_min, rec_empirical.tau_min, tau_min_ratio))
			println(@sprintf("    τ_max: analytic = %.4g, empirical = %.4g, ratio (emp/ana) = %.3f",
								rec_analytic.tau_max, rec_empirical.tau_max, tau_max_ratio))
			println(@sprintf("    L:     analytic = %d, empirical = %d, diff = %+d",
								rec_analytic.L, rec_empirical.L, L_diff))

		#	--- AUMC Consistency Check ---
			aumc_max_abs_diff = NaN
			aumc_cosine_sim   = NaN
			if run_aumc_check
				println()
				println("  Running AUMC consistency check (layered census at analytic recommendation)...")
				census_analytic = triad_census(edges;
												nodes                = nodes,
												weighted             = true,
												graph_type           = graph_type,
												reciprocity_collapse = reciprocity_collapse,
												L                    = rec_analytic.L,
												tau_min              = rec_analytic.tau_min,
												tau_max              = rec_analytic.tau_max,
												parallel             = true)
				census_empirical = rec_empirical.census  # already computed
				aumc_a = _aumc_vector_from_census(census_analytic)
				aumc_e = _aumc_vector_from_census(census_empirical)
				aumc_max_abs_diff = maximum(abs.(aumc_a .- aumc_e))
				dot_ae = sum(aumc_a .* aumc_e)
				norm_a = sqrt(sum(aumc_a .^ 2))
				norm_e = sqrt(sum(aumc_e .^ 2))
				aumc_cosine_sim = (norm_a > 0 && norm_e > 0) ?
				                  dot_ae / (norm_a * norm_e) : NaN
				println(@sprintf("    AUMC max |Δ|:    %.4g", aumc_max_abs_diff))
				println(@sprintf("    AUMC cosine:     %.4f", aumc_cosine_sim))
			end

			println("=" ^ 70)

		return (label              = label,
				analytic           = rec_analytic,
				empirical          = rec_empirical,
				t_analytic         = t_analytic,
				t_empirical        = t_empirical,
				tau_min_ratio      = tau_min_ratio,
				tau_max_ratio      = tau_max_ratio,
				L_diff             = L_diff,
				aumc_max_abs_diff  = aumc_max_abs_diff,
				aumc_cosine_sim    = aumc_cosine_sim)
	end

#	Run the Validation Suite Across Multiple Networks
	function run_recommend_L_validation_suite(networks::Vector{<:NamedTuple};
												run_aumc_check::Bool=true)
		"""
		Args:
			networks::Vector{<:NamedTuple}: each element has fields
				(edges, nodes, graph_type, reciprocity_collapse, label).
			run_aumc_check::Bool: forward to test_compare_recommend_L (default true)
		Returns:
			DataFrame: summary table with one row per network
		Notes:
			Orchestrator. Runs test_compare_recommend_L on each network and
			aggregates results into a tabular summary.
		"""

		println()
		println("=" ^ 70)
		println("RECOMMEND_L VALIDATION SUITE")
		println("=" ^ 70)
		println("Comparing analytic vs empirical recommend_L on $(length(networks)) networks")
		println()

		results = NamedTuple[]
		for net in networks
			cmp = test_compare_recommend_L(net.edges;
											nodes                = net.nodes,
											graph_type           = net.graph_type,
											reciprocity_collapse = net.reciprocity_collapse,
											label                = net.label,
											run_aumc_check       = run_aumc_check)
			push!(results, cmp)
			println()
		end

		#	Build Summary Table
			summary_rows = NamedTuple[]
			for r in results
				push!(summary_rows, (label               = r.label,
									L_analytic           = r.analytic.L,
									L_empirical          = r.empirical.L,
									L_diff               = r.L_diff,
									tau_min_analytic     = r.analytic.tau_min,
									tau_min_empirical    = r.empirical.tau_min,
									tau_min_ratio        = r.tau_min_ratio,
									tau_max_analytic     = r.analytic.tau_max,
									tau_max_empirical    = r.empirical.tau_max,
									tau_max_ratio        = r.tau_max_ratio,
									t_analytic           = r.t_analytic,
									t_empirical          = r.t_empirical,
									speedup              = r.t_empirical / r.t_analytic,
									aumc_max_abs_diff    = r.aumc_max_abs_diff,
									aumc_cosine_sim      = r.aumc_cosine_sim,
									valid                = r.analytic.valid))
			end
			summary = DataFrame(summary_rows)

		#	Print Summary
			println("=" ^ 70)
			println("VALIDATION SUITE SUMMARY")
			println("=" ^ 70)
			println(summary)
			println("=" ^ 70)

		return summary
	end

#   PLOTTING FUNCTIONS

#	Helper: Build the Union τ Grid Spanning Both Methods' Bounds
	function _build_union_tau_grid(rec_analytic::NamedTuple,
									rec_empirical::NamedTuple;
									L_dense::Int = 40)
		"""
		Args:
			rec_analytic::NamedTuple: analytic recommend_L result
			rec_empirical::NamedTuple: empirical recommend_L_empirical result
			L_dense::Int: density of the union τ grid (default 40)
		Returns:
			NamedTuple: (tau_min::Float64, tau_max::Float64, L::Int)
		Notes:
			Builds a log-spaced τ grid spanning the minimum τ_min and maximum
			τ_max across both methods. Used to run a dense layered census for
			diagnostic visualization — we want to see what happens both inside
			and outside each method's recommended range.
		"""
		tau_min = min(rec_analytic.tau_min, rec_empirical.tau_min)
		tau_max = max(rec_analytic.tau_max, rec_empirical.tau_max)
		return (tau_min = tau_min, tau_max = tau_max, L = L_dense)
	end

#	Helper: Plot Triangle Decay Curve for One Network
	function _plot_triangle_decay!(ax,
									profile::DataFrame,
									T_max::Int,
									rec_analytic::NamedTuple,
									rec_empirical::NamedTuple;
									frac_keep::Float64 = 1.0 / ℯ,
									T_min_floor::Int = 9)
		"""
		Args:
			ax: Makie Axis to plot into
			profile::DataFrame: triangle profile from analytic recommend_L
			T_max::Int: maximum triangle count
			rec_analytic, rec_empirical::NamedTuple: method results for τ bounds
			frac_keep::Float64: e-fold threshold (default 1/e)
			T_min_floor::Int: detection threshold (default 9)
		Returns:
			Nothing (mutates ax)
		Notes:
			Renders the T(τ) decay curve on log-log axes, with reference lines
			at T_max, T_max/e, and T_min_floor. Vertical bands mark the τ
			ranges chosen by each method.
		"""

		#	Filter to Positive Counts (Log Scale Requires)
			pos_mask = profile.triangle_count .> 0
			tau_pos  = profile.tau[pos_mask]
			T_pos    = profile.triangle_count[pos_mask]

		#	Triangle Decay Curve
			scatterlines!(ax, tau_pos, T_pos;
							color = :black, linewidth = 2, markersize = 8,
							label = "T(τ)")

		#	Horizontal Reference Lines
			hlines!(ax, [T_max]; color = :gray60, linestyle = :dot, linewidth = 1.5,
					label = "T_max = $T_max")
			hlines!(ax, [T_max * frac_keep]; color = :steelblue, linestyle = :dash, linewidth = 1.5,
					label = "T_max/e ≈ $(round(Int, T_max * frac_keep))")
			hlines!(ax, [T_min_floor]; color = :crimson, linestyle = :dash, linewidth = 1.5,
					label = "T_min_floor = $T_min_floor (3σ)")

		#	Vertical Method Bounds — Analytic
			vlines!(ax, [rec_analytic.tau_min]; color = :steelblue, linewidth = 2, alpha = 0.7,
					label = "analytic τ_min")
			vlines!(ax, [rec_analytic.tau_max]; color = :steelblue, linewidth = 2, linestyle = :dot, alpha = 0.7,
					label = "analytic τ_max")

		#	Vertical Method Bounds — Empirical
			vlines!(ax, [rec_empirical.tau_min]; color = :darkorange, linewidth = 2, alpha = 0.7,
					label = "empirical τ_min")
			vlines!(ax, [rec_empirical.tau_max]; color = :darkorange, linewidth = 2, linestyle = :dot, alpha = 0.7,
					label = "empirical τ_max")

		return nothing
	end

#	Helper: Plot Motif Density Curves on a Dense τ Grid
	function _plot_motif_densities!(ax,
									edges::DataFrame,
									nodes,
									graph_type::Symbol,
									reciprocity_collapse::Bool,
									union_grid::NamedTuple,
									rec_analytic::NamedTuple,
									rec_empirical::NamedTuple)
		"""
		Args:
			ax: Makie Axis to plot into
			edges, nodes, graph_type, reciprocity_collapse: graph spec
			union_grid::NamedTuple: (tau_min, tau_max, L) for dense census
			rec_analytic, rec_empirical::NamedTuple: method results for bounds
		Returns:
			NamedTuple: (per_tau::DataFrame) of the dense census
		Notes:
			Runs a dense layered census over the union τ range and plots the
			density of each of the four undirected DL classes (003, 102, 201,
			300). Vertical lines show where each method placed its bounds.
		"""

		#	Run Dense Layered Census Over Union Range
			result = triad_census(edges;
									nodes                = nodes,
									weighted             = true,
									graph_type           = graph_type,
									reciprocity_collapse = reciprocity_collapse,
									L                    = union_grid.L,
									tau_min              = union_grid.tau_min,
									tau_max              = union_grid.tau_max,
									parallel             = true)

		#	Plot Each Undirected Class on Its Own
			undir_classes = ["003", "102", "201", "300"]
			class_colors  = [:gray40, :steelblue, :goldenrod, :crimson]
			for (cls, col) in zip(undir_classes, class_colors)
				sub = result.per_tau[result.per_tau.triad .== cls, :]
				if nrow(sub) > 0
					scatterlines!(ax, sub.tau, sub.density;
									color = col, linewidth = 2, markersize = 6, label = cls)
				end
			end

		#	Vertical Method Bounds
			vlines!(ax, [rec_analytic.tau_min, rec_analytic.tau_max];
					color = :steelblue, linewidth = 1.5, alpha = 0.5, linestyle = :dash)
			vlines!(ax, [rec_empirical.tau_min, rec_empirical.tau_max];
					color = :darkorange, linewidth = 1.5, alpha = 0.5, linestyle = :dash)

		return result
	end

#	Build Diagnostic Figure for a Single Network's Comparison
	function diagnose_recommend_L_comparison(comparison::NamedTuple,
												edges::DataFrame,
												nodes,
												graph_type::Symbol,
												reciprocity_collapse::Bool;
												L_dense::Int = 40,
												figsize::Tuple{Int,Int} = (1100, 800))
		"""
		Args:
			comparison::NamedTuple: output of test_compare_recommend_L
			edges, nodes, graph_type, reciprocity_collapse: graph spec
			L_dense::Int: density of union τ grid for the motif-density panel
			figsize::Tuple{Int,Int}: figure pixel dimensions
		Returns:
			NamedTuple: (figure::Figure, dense_result::NamedTuple)
				figure: the CairoMakie Figure for saving or display
				dense_result: the layered census run over the union τ grid
				              (useful for further inspection)
		Notes:
			Two-panel diagnostic figure:
			- Top: triangle decay T(τ) with reference lines and method bounds
			- Bottom: per-τ motif densities for the four undirected DL classes,
				   on the union τ range so you can see what happens both inside
				   and outside each method's recommendation.

			Method bounds appear on both panels as vertical reference lines
			(steelblue = analytic, darkorange = empirical).
		"""

		rec_analytic  = comparison.analytic
		rec_empirical = comparison.empirical

		#	Build Figure with Two Vertically Stacked Panels
			fig = Figure(size = figsize)

		#	Panel 1: Triangle Decay
			ax1 = Axis(fig[1, 1];
						title  = "$(comparison.label) — Triangle Decay T(τ)",
						xlabel = "τ (log scale)",
						ylabel = "T(τ) (log scale)",
						xscale = log10,
						yscale = log10)
			_plot_triangle_decay!(ax1, rec_analytic.profile, rec_analytic.T_max,
									rec_analytic, rec_empirical;
									frac_keep    = rec_analytic.frac_keep,
									T_min_floor  = rec_analytic.T_min_floor)
			axislegend(ax1; position = :lb, labelsize = 9, framevisible = false)

		#	Panel 2: Motif Densities Over Union Range
			union_grid = _build_union_tau_grid(rec_analytic, rec_empirical; L_dense = L_dense)
			ax2 = Axis(fig[2, 1];
						title  = "Motif densities (undirected DL classes) over union τ range",
						xlabel = "τ (log scale)",
						ylabel = "density (per C(n,3) triples)",
						xscale = log10)
			dense_result = _plot_motif_densities!(ax2, edges, nodes, graph_type,
													reciprocity_collapse, union_grid,
													rec_analytic, rec_empirical)
			axislegend(ax2; position = :rt, labelsize = 9, framevisible = false)

		#	Add Method-Bound Legend Annotation to Bottom-Right of Figure
			Label(fig[3, 1],
					"Vertical reference lines: steelblue = analytic τ bounds, darkorange = empirical τ bounds";
					fontsize = 10, color = :gray40, padding = (0, 0, 10, 0))

		return (figure = fig, dense_result = dense_result)
	end

#############
#   TESTS   #
#############

#	Helper to Pull a Network from the networks Dict
	function _build_network_spec(networks_dict::Dict{String,<:Any},
									key::String,
									label::String)
		"""
		Args:
			networks_dict::Dict: the loaded GraphML networks
			key::String: the dict key (filename without .graphml)
			label::String: descriptive label for the validation suite
		Returns:
			NamedTuple: (edges, nodes, graph_type, reciprocity_collapse, label)
		Notes:
			graph_type is derived from metadata.directed. Nodes are passed
			through to preserve isolates. reciprocity_collapse defaults to
			false (natural graph type, not Pajek-style collapse).
		"""

		@assert haskey(networks_dict, key) "key '$key' not found in networks dict. " *
		                                   "Available keys: $(sort(collect(keys(networks_dict))))"
		net = networks_dict[key]
		gt  = net.metadata.directed ? :directed : :undirected
		return (edges                = net.edges,
				nodes                = net.nodes,
				graph_type           = gt,
				reciprocity_collapse = false,
				label                = label)
	end

#	Build the Validation Suite Specs
	validation_networks = [
		_build_network_spec(networks, "balikatan_2022_weighted",
							"Balikatan directed weighted"),
		_build_network_spec(networks, "moreno_highschool_weighted",
							"Moreno highschool weighted"),
		_build_network_spec(networks, "scotland_interlock_weighted",
							"Scotland interlock weighted"),
	]

#	Run the suite
	validation_results = run_recommend_L_validation_suite(validation_networks; run_aumc_check=true)

#   Test Empirical vs. Analytic L_Recommendaitons
    for key in ["balikatan_2022_weighted", "moreno_highschool_weighted",
                "scotland_interlock_weighted"]
        net = networks[key]
        gt  = net.metadata.directed ? :directed : :undirected
        println("\n--- $key ($gt) ---")
        rec = recommend_L(net.edges;
                            nodes      = net.nodes,
                            graph_type = gt,
                            verbose    = true)
        println("  → method: $(rec.method)")
        println("  → L: $(rec.L), τ: [$(rec.tau_min), $(rec.tau_max)]")
    end

#   Testing on the Marvel Universe
    marvel_rec = recommend_L(networks["marvel_universe_weighted"].edges;
                            nodes      = networks["marvel_universe_weighted"].nodes,
                            graph_type = :undirected,
                            verbose    = true)
    println("\n=== Marvel result ===")
    println("Method: $(marvel_rec.method)")
    println("Reason: $(marvel_rec.method_reason)")
    println("L = $(marvel_rec.L), τ ∈ [$(marvel_rec.tau_min), $(marvel_rec.tau_max)]")
    println("T_max = $(marvel_rec.T_max)")

#	Diagnose the Undirected Triad Census Kernel: Correctness + Collapse-Route Speed
	function diagnose_undirected_triad_kernel(networks::Dict;
	                                         run_marvel::Bool = true)
		"""
		Args:
			networks::Dict: corpus keyed by name → (edges, nodes, metadata)
			run_marvel::Bool: whether to run the Marvel collapse-route timing
				(default true)
		Returns:
			NamedTuple: (routes_agree, collapse_seconds)
				routes_agree::Bool: collapse route matches the honest undirected
					census on the four undirected classes {003,102,201,300},
					with all twelve asymmetric classes zero (validated on
					Scotland, N=108, where both paths are cheap)
				collapse_seconds::Float64: Marvel collapse-route wall-clock
					(NaN if run_marvel = false or Marvel absent)
		Notes:
			Decides whether to route the undirected static (binary) triad
			census through the subquadratic directed Batagelj-Mrvar kernel with
			reciprocity_collapse = true, instead of the O(N^3) honest undirected
			triple-loop kernel.

			Deliberately does NOT time the honest undirected census on Marvel:
			that path is the known-slow O(N^3) kernel and timing it would only
			reconfirm what we already know. Correctness of the collapse route is
			established on Scotland (Section 1); speed at scale is established by
			timing ONLY the collapse route on Marvel (Section 2). Correct +
			fast ⇒ route undirected static census through collapse.

			This is shared library code (Phase 0 and Large Graph Similarity), so
			the eventual fix belongs in the library — either making the public
			undirected binary path delegate to the collapse route internally, or
			writing a proper subquadratic undirected BM kernel — rather than in
			caller-side workarounds.
		"""

		println("=" ^ 70)
		println("Undirected triad census kernel diagnostic")
		println("=" ^ 70)

		#	Shared Tiny Graph for JIT Warmup
			tiny_e = DataFrame(src = ["a","b","c"], dst = ["b","c","a"])
			tiny_n = DataFrame(id = ["a","b","c"], label = ["a","b","c"])

		#	================================================================
		#	Section 1: Correctness on Scotland (N=108)
		#	Both paths are cheap at this size, so we can validate the collapse
		#	route against the honest undirected census directly.
		#	================================================================
			println("\n[1] Correctness check on Scotland (N=108)")

			routes_agree = false
			if haskey(networks, "scotland_interlock_unweighted")
				s_e = networks["scotland_interlock_unweighted"].edges
				s_n = networks["scotland_interlock_unweighted"].nodes

				undir    = triad_census(s_e; nodes = s_n, graph_type = :undirected)
				collapse = triad_census(s_e; nodes = s_n, graph_type = :directed,
				                       reciprocity_collapse = true)

				u = Dict(string(r.triad) => Int(r.count) for r in eachrow(undir))
				c = Dict(string(r.triad) => Int(r.count) for r in eachrow(collapse))

				#	Four Undirected Classes Must Match
					undirected_classes = ("003", "102", "201", "300")
					four_match = true
					println("    undirected classes (must match):")
					for cls in undirected_classes
						uv = get(u, cls, 0); cv = get(c, cls, 0)
						m = uv == cv
						four_match &= m
						println("      $cls:  undirected=$uv  collapsed=$cv  match=$m")
					end

				#	Twelve Asymmetric Classes Must Be Zero on Both
					asym_classes = ("012", "021D", "021U", "021C", "111D", "111U",
					               "030T", "030C", "120D", "120U", "120C", "210")
					asym_ok = true
					for cls in asym_classes
						uv = get(u, cls, 0); cv = get(c, cls, 0)
						ok = (uv == 0) && (cv == 0)
						asym_ok &= ok
						ok || println("      $cls:  undirected=$uv  collapsed=$cv  NONZERO!")
					end
					asym_ok && println("    all twelve asymmetric classes zero on both")

				routes_agree = four_match && asym_ok
				println("    verdict: ", routes_agree ? "ROUTES AGREE" : "ROUTES DISAGREE")
			else
				println("    SKIPPED (scotland_interlock_unweighted not in corpus)")
			end

		#	================================================================
		#	Section 2: Collapse-Route Timing on Marvel (N=6486)
		#	We time ONLY the rewired collapse route. The honest undirected
		#	census is the known-slow O(N^3) path and is intentionally not run.
		#	================================================================
			collapse_seconds = NaN

			if run_marvel && haskey(networks, "marvel_universe_unweighted")
				println("\n[2] Collapse-route timing on Marvel (N=6486)")
				mu_e = networks["marvel_universe_unweighted"].edges
				mu_n = networks["marvel_universe_unweighted"].nodes

				#	Warm Up the Collapse Path on the Tiny Graph (Absorb JIT)
					triad_census(tiny_e; nodes = tiny_n, graph_type = :directed,
					            reciprocity_collapse = true)

				#	Time ONLY the Collapse Route
					local tc_collapse
					collapse_seconds = @elapsed begin
						tc_collapse = triad_census(mu_e; nodes = mu_n,
						                          graph_type = :directed,
						                          reciprocity_collapse = true)
					end
					println("    collapse route: $(round(collapse_seconds, digits=3))s")

				#	Sanity Spot-Check: the Four Undirected Classes Are Populated
					c = Dict(string(r.triad) => Int(r.count) for r in eachrow(tc_collapse))
					println("    Marvel collapse counts:  " *
					        join(["$cls=$(get(c, cls, 0))" for cls in ("003","102","201","300")],
					             "   "))
			else
				println("\n[2] Collapse-route Marvel timing — SKIPPED",
				        run_marvel ? " (marvel_universe_unweighted not in corpus)" : " (run_marvel=false)")
			end

		#	Overall Verdict
			println("\n" * "=" ^ 70)
			println("Diagnostic complete: ",
			        routes_agree ? "collapse route VALID" : "collapse route INVALID",
			        isfinite(collapse_seconds) ? " — Marvel $(round(collapse_seconds, digits=2))s" : "")
			println("=" ^ 70)

			return (routes_agree     = routes_agree,
			        collapse_seconds  = collapse_seconds)
	end
	diagnose_undirected_triad_kernel(networks)

################
#   PLOTTING   #
################

diagnostic_figures = NamedTuple[]

for spec in validation_networks
	println("\nGenerating diagnostic figure for: $(spec.label)")
	println("="^70)

	#	Run Comparison (Already Have validation_results, But Need the
	#	Full NamedTuple With profile, T_max, etc. — re-run is cheap.)
		cmp = test_compare_recommend_L(spec.edges; nodes                = spec.nodes,
										graph_type           = spec.graph_type,
										reciprocity_collapse = spec.reciprocity_collapse,
										label                = spec.label,
										run_aumc_check       = false)  # skip AUMC, already done

	#	Generate Diagnostic Figure
		diag = diagnose_recommend_L_comparison(cmp, spec.edges,
												spec.nodes,
												spec.graph_type,
												spec.reciprocity_collapse;
												L_dense = 40)

	#	Save Figure to Disk
	#	Derive a clean filename from the label
		fname = "recommend_L_diagnostic_" *
					replace(replace(spec.label, " " => "_"), "/" => "_") * ".png"
					save(fname, diag.figure; px_per_unit = 2)
					println("  Saved figure to: $fname")

	#	Store for Later Inspection
				push!(diagnostic_figures, (label   = spec.label,
											figure = diag.figure,
											dense  = diag.dense_result,
											cmp    = cmp))
end

println("\nGenerated $(length(diagnostic_figures)) diagnostic figures.")
println("Files saved in current working directory:")
for d in diagnostic_figures
	fname = "recommend_L_diagnostic_" *
			replace(replace(d.label, " " => "_"), "/" => "_") * ".png"
	println("  - $fname")
end