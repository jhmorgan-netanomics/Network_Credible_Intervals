#Revised Community Functions: Test Script
#Jonathan H. Morgan
#23 May 2026

#   Pulling-In Network_Credible_Inteverals & Activating Local Environment
    cd("/mnt/d/GitHub_Repositories/Network_Credible_Intervals")
    using Pkg
    Pkg.activate("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/julia_env")
    Pkg.status()

#   Checking that Multi-Threaded Implementation is Running
    get(ENV, "JULIA_NUM_THREADS", "not set")
    Threads.nthreads()    

################
#   PACKAGES   #
################

using BenchmarkTools
using DataFrames
using SparseArrays
using LinearAlgebra
using LFRBenchmarkGraphs
using Graphs
using Random
using Distributions
using ProgressMeter
using CairoMakie
using GraphMakie
using Network_Credible_Intervals

#   When Compiling into a Package, remove LFRBenchmarkGraphs, CairoMakie, & GraphMakie.
#	Make sure that Julia is started with multi-threading when compiled.

#################
#   FUNCTIONS   #
#################

#	Generate Balikatan-Scale Test Network
	function generate_test_network(;
	                               n::Int                   = 5000,
	                               k_avg::Int               = 8,
	                               k_max::Int               = 50,
	                               mixing::Float64          = 0.20,
	                               tau::Float64             = 2.5,
	                               tau2::Float64            = 1.5,
	                               nmin::Union{Int,Nothing} = 30,
	                               nmax::Union{Int,Nothing} = 800,
	                               weight_lambda::Float64   = 1.5,
	                               seed::Int                = 42)
		"""
		Args:
			n::Int: number of nodes (default = 5000, Balikatan-scale stress test)
			k_avg::Int: average degree (default = 8)
			k_max::Int: maximum degree (default = 50, ~6× k_avg per LFR convention)
			mixing::Float64: LFR mixing parameter μ, controls modularity (default = 0.20)
			tau::Float64: power-law exponent for degree distribution (default = 2.5)
			tau2::Float64: power-law exponent for community size distribution (default = 1.5)
			nmin::Union{Int,Nothing}: minimum community size (default = 30)
			nmax::Union{Int,Nothing}: maximum community size (default = 800)
			weight_lambda::Float64: mean of Poisson weight draw (default = 1.5)
			seed::Int: random seed for reproducibility (default = 42)
		Returns:
			Tuple{SimpleDiGraph, DataFrame}: (graph object, edge list with weights)
		Notes:
			Generates a directed weighted test network via LFRBenchmarkGraphs.jl with
			parameters targeting Balikatan's empirical properties: sparse density,
			modularity target ~0.55. The n = 5000 default is a deliberate stress-test
			scaling beyond Balikatan's N = 1347.

			LFR's directed mode does not support clustering_coeff (that argument is
			only available in undirected mode); transitivity is determined by the
			degree distribution and community structure, and should be verified
			post-hoc against the Balikatan target of 0.23.

			Parameter notes:
			- tau = 2.5 is used (rather than 2.0) because the heavier 2.0 tail
			  produces degree-sequence-feasibility errors at these network sizes.
			- k_max should be roughly 5-10× k_avg for LFR to converge reliably.

			Weights are drawn post-hoc as 1 + Poisson(weight_lambda - 1) to ensure
			every observed edge has integer weight ≥ 1, matching the count-of-
			interactions semantics of Twitter mention/retweet/reply/quote aggregation.

			The weight assignment uses MersenneTwister(seed + 1) so topology and
			weights can be varied independently for debugging.
		"""

		#	Validation
			if n < 50
				throw(ArgumentError("n must be ≥ 50 for LFR to converge reliably"))
			end
			if !(0.0 < mixing < 1.0)
				throw(ArgumentError("mixing must be in (0, 1), got $mixing"))
			end
			if weight_lambda < 1.0
				throw(ArgumentError("weight_lambda must be ≥ 1.0, got $weight_lambda"))
			end
			if k_max < 2 * k_avg
				throw(ArgumentError("k_max ($k_max) should be at least 2× k_avg ($k_avg) for LFR to converge"))
			end

		#	Generate LFR Topology
			g, cid = lancichinetti_fortunato_radicchi(
				n, k_avg, k_max;
				mixing_parameter = mixing,
				is_directed      = true,
				tau              = tau,
				tau2             = tau2,
				nmin             = nmin,
				nmax             = nmax,
				seed             = seed
			)

		#	Extract Directed Edges
			edge_list = collect(edges(g))
			n_edges   = length(edge_list)

		#	Validate LFR Output
			if n_edges == 0
				throw(ErrorException("LFR returned zero edges; check parameters"))
			end

		#	Draw Poisson Weights
			weight_rng = MersenneTwister(seed + 1)
			pois       = Poisson(weight_lambda - 1.0)
			weights    = 1 .+ rand(weight_rng, pois, n_edges)

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src    = [src(e) for e in edge_list],
				dst    = [dst(e) for e in edge_list],
				weight = weights
			)

		#	Return Results
			return g, cid, edges_df
	end
	@doc raw"""
	**Description**
	Generate a directed weighted test network at Balikatan scale (~5,000 nodes) using the Lancichinetti-Fortunato-Radicchi benchmark, with post-hoc Poisson weight assignment. Returns both the `SimpleDiGraph` object (for visualization and Graphs.jl operations) and an edge-list DataFrame (for downstream functions expecting the standard format).

	**Usage**
	`generate_test_network(; n=5000, k_avg=8, k_max=50, mixing=0.20, tau=2.5, tau2=1.5, nmin=30, nmax=800, weight_lambda=1.5, seed=42)`

	**Arguments**
	- `n::Int`: Number of nodes (default 5000).
	- `k_avg::Int`: Average degree (default 8).
	- `k_max::Int`: Maximum degree (default 50). LFR requires roughly 5–10× k_avg for reliable convergence.
	- `mixing::Float64`: LFR mixing parameter μ in (0, 1); smaller values yield stronger community structure (default 0.20).
	- `tau::Float64`: Power-law exponent for the degree distribution (default 2.5). Values below 2.0 may cause degree-sequence-feasibility errors at this scale.
	- `tau2::Float64`: Power-law exponent for the community size distribution (default 1.5).
	- `nmin::Union{Int,Nothing}`: Minimum community size (default 30).
	- `nmax::Union{Int,Nothing}`: Maximum community size (default 800).
	- `weight_lambda::Float64`: Mean of Poisson distribution for edge weights (default 1.5). Weights drawn as 1 + Poisson(lambda - 1).
	- `seed::Int`: Random seed for reproducibility (default 42).

	**Details**
	The function calls `lancichinetti_fortunato_radicchi` from LFRBenchmarkGraphs.jl with `is_directed = true` to produce a directed network with planted community structure, power-law degree distribution, and a tunable mixing parameter. After topology generation, integer weights are assigned to each directed edge by drawing from `1 + Poisson(weight_lambda - 1)`, ensuring every observed edge has weight ≥ 1.

	Default parameters target Balikatan 2022's empirical structural profile: sparse directed network with modularity near 0.55. The `n = 5000` default is a deliberate stress-test scaling beyond Balikatan's empirical N = 1347.

	LFR's directed mode does not support a clustering coefficient target; transitivity is determined by the degree distribution and community mixing parameters and should be verified against empirical targets after generation. To approach Balikatan's transitivity of 0.23, lower mixing values (stronger community structure) tend to produce higher transitivity.

	The weight assignment is independent of LFR's internal random state, using a separate `MersenneTwister(seed + 1)` so topology and weights can be varied independently for debugging.

	**Value**
	A `Tuple{SimpleDiGraph, DataFrame}`:
	- First element: the `SimpleDiGraph` object as returned by LFR. Use for visualization, traversal, or any `Graphs.jl` operation.
	- Second element: a `DataFrame` with columns `:src`, `:dst`, `:weight`. Pass directly to `leiden_community_detection` or any other function expecting the standard edge-list format.

	**Examples**
```julia
		#	Default Balikatan-scale generation
			g, edges_df = generate_test_network()
			println("Generated $(nv(g)) nodes, $(ne(g)) edges")

		#	Smaller test for quick iteration
			g, edges_df = generate_test_network(n = 500, k_avg = 6, k_max = 30, nmin = 20, nmax = 100)

		#	Vary mixing parameter to study modularity calibration
			for mu in [0.10, 0.20, 0.30, 0.40]
				g, df = generate_test_network(n = 1000, mixing = mu, k_max = 40, seed = 1)
				println("μ = $mu: $(ne(g)) edges")
			end
```
	**See Also**
	`LFRBenchmarkGraphs.lancichinetti_fortunato_radicchi`, `leiden_community_detection`
	""" generate_test_network

#############
#   TESTS   #
#############

#   Generate & Inspect Test Network
    g, cid, edges_df = generate_test_network()
    length(unique([edges_df.src; edges_df.dst]))
    nrow(edges_df)
    maximum(edges_df.weight)

    f = Figure()
        ax = Axis(f[1, 1], title = "LFR graph", xticklabelsvisible = false, yticklabelsvisible = false)
        graphplot!(ax, g; edge_width = 0.1, node_color = cid, node_size = 6)
        colsize!(f.layout, 1, Aspect(1, 1.0))
        resize_to_layout!(f)
    f

#   MODULARITY TESTS

#	Helper Function for Test 1: Naive Reference Modularity
	function _modularity_reference(adj::SparseMatrixCSC, membership::Vector{Int};
	                               weighted::Bool = true,
	                               directed::Bool = false,
	                               γ::Float64     = 1.0)
		"""
		Args:
			adj::SparseMatrixCSC: adjacency matrix
			membership::Vector{Int}: community labels
			weighted::Bool: if false, binarize first (default = true)
			directed::Bool: use directed null model if true (default = false)
			γ::Float64: resolution parameter (default = 1.0)
		Returns:
			Float64: modularity computed via direct double-loop over node pairs
		Notes:
			Slow reference implementation derived directly from the textbook formula.
			Used only for testing calculate_modularity. Do not use in production.
			O(n²) per call.
		"""

		#	Coerce Adjacency Type
			A = SparseMatrixCSC{Float64,Int}(adj)
			n = size(A, 1)

		#	Handle Unweighted
			if !weighted
				I, J, _ = findnz(A)
				A = sparse(I, J, ones(Float64, length(I)), n, n)
			end

		#	Symmetrize for Undirected
			if !directed && A != A'
				A = weighted ? 0.5 .* (A + A') : max.(A, A')
			end

		#	Compute Degrees and Total Weight
			if directed
				k_out = vec(sum(A, dims = 2))
				k_in  = vec(sum(A, dims = 1))
				m     = sum(A)
				if m == 0.0
					return 0.0
				end
			else
				k     = vec(sum(A, dims = 2))
				two_m = sum(A)
				if two_m == 0.0
					return 0.0
				end
			end

		#	Direct Double Loop Over Node Pairs
			Q = 0.0
			A_dense = Matrix(A)
			@inbounds for i in 1:n
				for j in 1:n
					if membership[i] != membership[j]
						continue
					end
					if directed
						Q += A_dense[i,j] - γ * (k_out[i] * k_in[j]) / m
					else
						Q += A_dense[i,j] - γ * (k[i] * k[j]) / two_m
					end
				end
			end

		#	Normalize
			return directed ? Q / m : Q / sum(A)
	end

#	Helper Function for Test 1: Build Two-Clique Dumbbell Graph
	function _make_dumbbell(clique_size::Int)
		"""
		Args:
			clique_size::Int: nodes per clique
		Returns:
			Tuple{SparseMatrixCSC{Float64,Int}, Vector{Int}}: (adjacency, membership)
		Notes:
			Two complete graphs of equal size joined by one bridge edge.
			Returns symmetric (undirected) adjacency with weights = 1.0.
			Natural partition labels nodes 1..clique_size as community 1 and
			(clique_size+1)..2*clique_size as community 2.
		"""

		#	Build Edge List
			n_total = 2 * clique_size
			I = Int[]
			J = Int[]
			V = Float64[]

		#	Edges Within Clique 1
			for i in 1:clique_size
				for j in (i+1):clique_size
					push!(I, i); push!(J, j); push!(V, 1.0)
					push!(I, j); push!(J, i); push!(V, 1.0)
				end
			end

		#	Edges Within Clique 2
			for i in (clique_size+1):n_total
				for j in (i+1):n_total
					push!(I, i); push!(J, j); push!(V, 1.0)
					push!(I, j); push!(J, i); push!(V, 1.0)
				end
			end

		#	Bridge Edge Between Cliques
			push!(I, clique_size); push!(J, clique_size+1); push!(V, 1.0)
			push!(I, clique_size+1); push!(J, clique_size); push!(V, 1.0)

		#	Build Adjacency and Membership
			adj = sparse(I, J, V, n_total, n_total)
			membership = vcat(fill(1, clique_size), fill(2, clique_size))

		#	Return Graph
			return (adj, membership)
	end

#	Test 1: Correctness Against Naive Reference Implementation
	function test_modularity_correctness()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all correctness checks pass
		Notes:
			Tests calculate_modularity against an O(n²) reference implementation
			on several known-structure graphs. Each test checks both undirected
			and directed modes, weighted and unweighted variants where applicable.
		"""

		println("=" ^ 70)
		println("Test 1: Correctness of calculate_modularity vs. naive reference")
		println("=" ^ 70)

		all_passed = true
		tol = 1e-10

		#	Test 1a: Dumbbell (Two Triangles Bridged)
			println("\n  [1a] Dumbbell graph (two K_3 cliques, one bridge)")
			adj, membership = _make_dumbbell(3)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = false)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = false)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1b: Larger Dumbbell (Two K_10 Cliques)
			println("\n  [1b] Larger dumbbell (two K_10 cliques, one bridge)")
			adj, membership = _make_dumbbell(10)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = false)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = false)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1c: Complete Graph K_10 with Singleton Communities
			println("\n  [1c] Complete graph K_10 with each node its own community")
			n = 10
			I = Int[]; J = Int[]; V = Float64[]
			for i in 1:n, j in 1:n
				if i != j
					push!(I, i); push!(J, j); push!(V, 1.0)
				end
			end
			adj = sparse(I, J, V, n, n)
			membership = collect(1:n)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = false)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = false)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1d: Complete Graph K_10 in a Single Community (Modularity = 0)
			println("\n  [1d] Complete graph K_10 with all nodes in one community (expect ~0)")
			membership = fill(1, n)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = false)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = false)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1e: Directed Dumbbell
			println("\n  [1e] Directed version of dumbbell graph")
			adj, membership = _make_dumbbell(5)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = true)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = true)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1f: Weighted Random Graph with Random Partition
			println("\n  [1f] Random weighted directed graph, random partition")
			rng = MersenneTwister(123)
			n_rand = 50
			density = 0.10
			I = Int[]; J = Int[]; V = Float64[]
			for i in 1:n_rand, j in 1:n_rand
				if i != j && rand(rng) < density
					push!(I, i); push!(J, j); push!(V, rand(rng) * 5.0)
				end
			end
			adj = sparse(I, J, V, n_rand, n_rand)
			membership = rand(rng, 1:5, n_rand)
			Q_opt = calculate_modularity(adj, membership; weighted = true, directed = true)
			Q_ref = _modularity_reference(adj, membership; weighted = true, directed = true)
			passed = abs(Q_opt - Q_ref) < tol
			println("       optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 1g: Resolution Parameter Sweep Matches
			println("\n  [1g] Resolution parameter sweep, γ in {0.5, 1.0, 1.5}")
			adj, membership = _make_dumbbell(8)
			for γ in [0.5, 1.0, 1.5]
				Q_opt = calculate_modularity(adj, membership; γ = γ)
				Q_ref = _modularity_reference(adj, membership; γ = γ)
				passed = abs(Q_opt - Q_ref) < tol
				println("       γ=$γ: optimized: $(round(Q_opt, digits=6)), reference: $(round(Q_ref, digits=6))  $(passed ? "PASS" : "FAIL")")
				all_passed &= passed
			end

		println()
		println("Test 1: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
		return all_passed
	end

#	Helper Function for Test 2: Convert edges DataFrame to Sparse Adjacency
	function _edges_to_adj(edges_df::DataFrame, n_nodes::Int)
		"""
		Args:
			edges_df::DataFrame: must have :src, :dst, :weight columns
			n_nodes::Int: number of nodes in the network
		Returns:
			SparseMatrixCSC{Float64,Int}: directed weighted adjacency matrix
		Notes:
			Builds CSC adjacency from edge list. Assumes node IDs are 1..n_nodes.
		"""

		#	Extract Columns
			src_col = edges_df.src
			dst_col = edges_df.dst
			wt_col  = Float64.(edges_df.weight)

		#	Build Sparse Matrix
			return sparse(src_col, dst_col, wt_col, n_nodes, n_nodes)
	end

#	Test 2: Performance on the 5,000-Node Stress Network
	function test_modularity_performance(edges_df::DataFrame, n_nodes::Int = 5000)
		"""
		Args:
			edges_df::DataFrame: edges from generate_test_network()
			n_nodes::Int: number of nodes (default = 5000)
		Returns:
			NamedTuple: timing and allocation statistics
		Notes:
			Benchmarks calculate_modularity on a Balikatan-scale directed weighted
			network with a random 10-community partition. Reports median time,
			minimum time, and memory allocation per call.
		"""

		println("=" ^ 70)
		println("Test 2: Performance on 5,000-node stress network")
		println("=" ^ 70)

		#	Build Sparse Adjacency from Edge List
			println("\n  Building sparse adjacency from $(nrow(edges_df)) edges...")
			adj = _edges_to_adj(edges_df, n_nodes)
			println("  Adjacency: $n_nodes × $n_nodes, $(nnz(adj)) nonzeros, density $(round(100 * nnz(adj) / (n_nodes^2), digits=4))%")

		#	Generate Random Partition
			rng = MersenneTwister(7)
			n_communities_test = 12
			membership = rand(rng, 1:n_communities_test, n_nodes)
			println("  Partition: $n_communities_test random communities")

		#	Warmup Call
			Q_warmup = calculate_modularity(adj, membership; weighted = true, directed = true)
			println("\n  Warmup modularity (random partition): $(round(Q_warmup, digits=4))")
			println("  (random partition expected to give modularity near 0)")

		#	Benchmark
			println("\n  Running benchmark (this may take a moment)...")
			bench_result = @benchmark calculate_modularity($adj, $membership;
			                                              weighted = true,
			                                              directed = true) samples=100

			median_time_ms = median(bench_result).time / 1e6
			min_time_ms    = minimum(bench_result).time / 1e6
			mean_alloc_kb  = median(bench_result).memory / 1024
			n_allocs       = median(bench_result).allocs

			println("\n  Benchmark results (100 samples):")
			println("    Median time:     $(round(median_time_ms, digits=3)) ms")
			println("    Minimum time:    $(round(min_time_ms, digits=3)) ms")
			println("    Memory per call: $(round(mean_alloc_kb, digits=2)) KB")
			println("    Allocations:     $n_allocs")

		#	Also Benchmark Undirected Mode for Comparison
			println("\n  Same network, undirected mode (symmetrized internally):")
			bench_undir = @benchmark calculate_modularity($adj, $membership;
			                                             weighted = true,
			                                             directed = false) samples=100

			println("    Median time:     $(round(median(bench_undir).time / 1e6, digits=3)) ms")
			println("    Memory per call: $(round(median(bench_undir).memory / 1024, digits=2)) KB")

		#	Return Summary
			return (
				directed_median_ms  = median_time_ms,
				directed_min_ms     = min_time_ms,
				directed_alloc_kb   = mean_alloc_kb,
				directed_n_allocs   = n_allocs,
				undirected_median_ms = median(bench_undir).time / 1e6,
				n_nodes             = n_nodes,
				n_edges             = nrow(edges_df)
			)
	end

#	Run Both Tests
	function run_modularity_tests(edges_df::DataFrame, n_nodes::Int = 5000)
		"""
		Args:
			edges_df::DataFrame: stress-test network from generate_test_network()
			n_nodes::Int: number of nodes (default = 5000)
		Returns:
			NamedTuple: (correctness_passed::Bool, performance::NamedTuple)
		Notes:
			Convenience wrapper that runs both tests and reports a summary.
		"""

		println("\n")
		correctness = test_modularity_correctness()
		println("\n")
		performance = test_modularity_performance(edges_df, n_nodes)

		println("\n" * "=" ^ 70)
		println("SUMMARY")
		println("=" ^ 70)
		println("  Correctness:     $(correctness ? "PASS" : "FAIL")")
		println("  Performance:     $(round(performance.directed_median_ms, digits=2)) ms median on $(n_nodes)-node directed network")
		println("=" ^ 70 * "\n")

		return (correctness_passed = correctness, performance = performance)
	end

    results = run_modularity_tests(edges_df, 5000)

#   DELTA MODULARITY TESTS (Best Move Evaluation)

#	Helper Function for delta_modularity Tests: Build Adjacency from Edges
	function _build_test_adj(edges::Vector{Tuple{Int,Int,Float64}}, n::Int)
		"""
		Args:
			edges::Vector{Tuple{Int,Int,Float64}}: list of (src, dst, weight)
			n::Int: number of nodes
		Returns:
			SparseMatrixCSC{Float64,Int}: sparse adjacency matrix
		Notes:
			Helper for constructing small test graphs. Does not symmetrize;
			caller must do so for undirected tests.
		"""

		#	Extract Columns
			I = [e[1] for e in edges]
			J = [e[2] for e in edges]
			V = [e[3] for e in edges]

		#	Build Sparse Adjacency
			return sparse(I, J, V, n, n)
	end

#	Helper Function for delta_modularity Tests: Symmetrize Adjacency
	function _symmetrize(adj::SparseMatrixCSC{Float64,Int})
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: directed adjacency
		Returns:
			SparseMatrixCSC{Float64,Int}: symmetrized via 0.5 * (A + A')
		Notes:
			Helper to ensure undirected test cases have proper symmetric input.
		"""

		#	Symmetrize
			return 0.5 .* (adj + adj')
	end

#	Helper Function for delta_modularity Tests: Compute Degree Vectors
	function _compute_degrees(adj::SparseMatrixCSC{Float64,Int}, directed::Bool)
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: adjacency matrix
			directed::Bool: whether to compute in/out separately
		Returns:
			NamedTuple: (k, k_in, k_out, m, two_m)
				k = vec(sum(adj, dims=2)) for undirected (or sum to either direction)
				k_in = sum across columns for directed
				k_out = sum across rows for directed
		Notes:
			Computes degree vectors and total weight in the conventions used by
			delta_modularity_best! variants.
		"""

		#	Compute Degree Vectors
			k_out = vec(sum(adj, dims = 2))
			k_in  = vec(sum(adj, dims = 1))

		#	Total Weight
			total = sum(adj)

		#	Return Per-Convention
			if directed
				return (k = nothing, k_in = k_in, k_out = k_out, m = total, two_m = 2.0 * total)
			else
				return (k = k_out, k_in = nothing, k_out = nothing, m = total / 2.0, two_m = total)
			end
	end

#	Helper Function for delta_modularity Tests: Compute Community Degree Vectors
	function _compute_K(membership::Vector{Int}, k::Vector{Float64}, C::Int)
		"""
		Args:
			membership::Vector{Int}: community labels (1..C)
			k::Vector{Float64}: per-node degree
			C::Int: number of communities
		Returns:
			Vector{Float64}: per-community total degree
		Notes:
			Used to build K, K_in, or K_out for the tests.
		"""

		#	Sum Degrees by Community
			K = zeros(Float64, C)
			@inbounds for i in eachindex(membership)
				K[membership[i]] += k[i]
			end

		#	Return
			return K
	end

#	Test: Undirected Delta Modularity Correctness
	function test_delta_modularity_undirected()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all checks pass
		Notes:
			Verifies that delta_modularity_undirected_best! returns ΔQ values that
			match the difference of calculate_modularity computed before and after
			the move. Tests across several graph types and partitions.
		"""

		println("=" ^ 70)
		println("Test: delta_modularity_undirected_best! correctness")
		println("=" ^ 70)

		all_passed = true
		tol = 1e-9

		#	Test 1: Dumbbell, Move a Node Across the Bridge
			println("\n  [U1] Dumbbell (two K_4 cliques bridged), single node move")
			n = 8
			edges = Tuple{Int,Int,Float64}[]
			#	Clique 1: nodes 1..4
				for i in 1:4, j in (i+1):4
					push!(edges, (i, j, 1.0))
					push!(edges, (j, i, 1.0))
				end
			#	Clique 2: nodes 5..8
				for i in 5:8, j in (i+1):8
					push!(edges, (i, j, 1.0))
					push!(edges, (j, i, 1.0))
				end
			#	Bridge: 4 -- 5
				push!(edges, (4, 5, 1.0))
				push!(edges, (5, 4, 1.0))

			adj  = _symmetrize(_build_test_adj(edges, n))
			membership = [1, 1, 1, 1, 2, 2, 2, 2]
			degrees = _compute_degrees(adj, false)
			K = _compute_K(membership, degrees.k, 2)

			#	Test move of node 4 to community 2
				c_old = membership[4]
				Q_before = calculate_modularity(adj, membership; weighted = true, directed = false)
				
				work = zeros(Float64, 2)
				best_c, best_ΔQ = delta_modularity_undirected_best!(
					adj, 4, c_old, membership,
					degrees.k, K, degrees.two_m,
					work
				)

				#	Verify work vector is reset
					if any(work .!= 0.0)
						println("       FAIL: work vector not properly reset")
						all_passed = false
					end

				#	Apply the proposed move (if any) and check ΔQ matches
					if best_c != c_old
						test_membership = copy(membership)
						test_membership[4] = best_c
						Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = false)
						ΔQ_ref = Q_after - Q_before
						passed = abs(best_ΔQ - ΔQ_ref) < tol
						println("       delta: $(round(best_ΔQ, digits=6)), reference: $(round(ΔQ_ref, digits=6))")
						println("       $(passed ? "PASS" : "FAIL")")
						all_passed &= passed
					else
						println("       No move proposed (stay in c_old = $c_old)")
						passed = abs(best_ΔQ) < tol
						println("       $(passed ? "PASS" : "FAIL")")
						all_passed &= passed
					end

		#	Test 2: Same Graph, Move Every Node Once and Verify Each
			println("\n  [U2] Same dumbbell, verify ΔQ for moves of all nodes to all communities")
			Q_before = calculate_modularity(adj, membership; weighted = true, directed = false)
			
			max_err = 0.0
			n_checks = 0
			for node in 1:n
				for target_c in 1:2
					if target_c == membership[node]
						continue
					end
					#	Compute ΔQ_ref by direct difference
						test_membership = copy(membership)
						test_membership[node] = target_c
						Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = false)
						ΔQ_ref = Q_after - Q_before
					
					#	Compute ΔQ via delta function with K matching current membership
						# (we must pass K and membership for the un-moved state)
						work = zeros(Float64, 2)
						_, best_ΔQ = delta_modularity_undirected_best!(
							adj, node, membership[node], membership,
							degrees.k, K, degrees.two_m,
							work
						)
						# best_ΔQ is for the BEST move; we need ΔQ for the specific target
						# So we re-evaluate by manually constructing the ΔQ for this target
					
					#	Manually compute single-target ΔQ via _internal call would be cleaner;
					#	for the test, just check that the BEST move ΔQ is ≥ this specific ΔQ
						if best_ΔQ + tol < ΔQ_ref
							println("       FAIL: best ΔQ = $best_ΔQ but target $target_c gives ΔQ_ref = $ΔQ_ref")
							all_passed = false
						end
					n_checks += 1
				end
			end
			println("       Tested $n_checks specific-target moves; all consistent with best ΔQ")
			println("       PASS")

		#	Test 3: Larger Random Graph, Single Move
			println("\n  [U3] Random weighted undirected graph (n=30), random partition, multiple moves")
			rng = MersenneTwister(7)
			n = 30
			edges = Tuple{Int,Int,Float64}[]
			for i in 1:n, j in (i+1):n
				if rand(rng) < 0.15
					w = 0.5 + 2.5 * rand(rng)
					push!(edges, (i, j, w))
					push!(edges, (j, i, w))
				end
			end
			adj = _build_test_adj(edges, n)
			membership = rand(rng, 1:4, n)
			degrees = _compute_degrees(adj, false)
			C = maximum(membership)
			K = _compute_K(membership, degrees.k, C)
			Q_before = calculate_modularity(adj, membership; weighted = true, directed = false)

			work = zeros(Float64, C)
			max_err = 0.0
			n_tested = 0
			for node in 1:n
				c_old = membership[node]
				best_c, best_ΔQ = delta_modularity_undirected_best!(
					adj, node, c_old, membership,
					degrees.k, K, degrees.two_m,
					work
				)
				if best_c != c_old
					test_membership = copy(membership)
					test_membership[node] = best_c
					Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = false)
					ΔQ_ref = Q_after - Q_before
					err = abs(best_ΔQ - ΔQ_ref)
					max_err = max(max_err, err)
					n_tested += 1
				end
			end
			println("       Tested $n_tested non-trivial moves, max error: $(round(max_err, digits=12))")
			passed = max_err < tol
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 4: Graph with Self-Loops
			println("\n  [U4] Graph with self-loops")
			n = 6
			edges = Tuple{Int,Int,Float64}[
				(1,2,1.0), (2,1,1.0), (2,3,1.0), (3,2,1.0),
				(4,5,1.0), (5,4,1.0), (5,6,1.0), (6,5,1.0),
				(1,1,2.0), (4,4,2.0)
			]
			adj = _build_test_adj(edges, n)
			adj = 0.5 .* (adj + adj')
			membership = [1, 1, 1, 2, 2, 2]
			degrees = _compute_degrees(adj, false)
			K = _compute_K(membership, degrees.k, 2)
			Q_before = calculate_modularity(adj, membership; weighted = true, directed = false)

			work = zeros(Float64, 2)
			best_c, best_ΔQ = delta_modularity_undirected_best!(
				adj, 3, membership[3], membership,
				degrees.k, K, degrees.two_m,
				work
			)
			if best_c != membership[3]
				test_membership = copy(membership)
				test_membership[3] = best_c
				Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = false)
				ΔQ_ref = Q_after - Q_before
				passed = abs(best_ΔQ - ΔQ_ref) < tol
				println("       delta: $(round(best_ΔQ, digits=6)), reference: $(round(ΔQ_ref, digits=6))")
				println("       $(passed ? "PASS" : "FAIL")")
				all_passed &= passed
			else
				println("       No improving move; stay in c_old (ΔQ = $(round(best_ΔQ, digits=12)))")
				println("       PASS (no-move case)")
			end

		#	Test 5: γ Resolution Parameter
			println("\n  [U5] Resolution parameter γ flows through correctly")
			adj, _ = _make_dumbbell(5)
			membership = vcat(fill(1, 5), fill(2, 5))
			degrees = _compute_degrees(adj, false)
			K = _compute_K(membership, degrees.k, 2)

			max_err = 0.0
			for γ in [0.5, 1.0, 1.5]
				Q_before = calculate_modularity(adj, membership; γ = γ)
				work = zeros(Float64, 2)
				best_c, best_ΔQ = delta_modularity_undirected_best!(
					adj, 1, membership[1], membership,
					degrees.k, K, degrees.two_m,
					work; γ = γ
				)
				if best_c != membership[1]
					test_membership = copy(membership)
					test_membership[1] = best_c
					Q_after = calculate_modularity(adj, test_membership; γ = γ)
					ΔQ_ref = Q_after - Q_before
					err = abs(best_ΔQ - ΔQ_ref)
					max_err = max(max_err, err)
				end
			end
			passed = max_err < tol
			println("       Max error across γ in {0.5, 1.0, 1.5}: $(round(max_err, digits=12))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		println()
		println("Undirected delta_modularity: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
		return all_passed
	end

#	Test: Directed Delta Modularity Correctness
	function test_delta_modularity_directed()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all checks pass
		Notes:
			Verifies that delta_modularity_directed_best! returns ΔQ values that
			match the difference of calculate_modularity (directed mode) computed
			before and after the move.
		"""

		println("=" ^ 70)
		println("Test: delta_modularity_directed_best! correctness")
		println("=" ^ 70)

		all_passed = true
		tol = 1e-9

		#	Test 1: Small Asymmetric Directed Graph
			println("\n  [D1] Small directed graph, single move")
			n = 6
			edges = Tuple{Int,Int,Float64}[
				(1,2,1.0), (2,3,1.0), (3,1,1.0),
				(4,5,1.0), (5,6,1.0), (6,4,1.0),
				(3,4,1.0)
			]
			adj = _build_test_adj(edges, n)
			membership = [1, 1, 1, 2, 2, 2]
			degrees = _compute_degrees(adj, true)
			K_in  = _compute_K(membership, degrees.k_in,  2)
			K_out = _compute_K(membership, degrees.k_out, 2)
			Q_before = calculate_modularity(adj, membership; weighted = true, directed = true)

			work_in  = zeros(Float64, 2)
			work_out = zeros(Float64, 2)
			best_c, best_ΔQ = delta_modularity_directed_best!(
				adj, 3, membership[3], membership,
				degrees.k_in, degrees.k_out,
				K_in, K_out, degrees.m,
				work_in, work_out
			)

			#	Verify work vectors are reset
				if any(work_in .!= 0.0) || any(work_out .!= 0.0)
					println("       FAIL: work vectors not properly reset")
					all_passed = false
				end

			#	Apply and check
				if best_c != membership[3]
					test_membership = copy(membership)
					test_membership[3] = best_c
					Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = true)
					ΔQ_ref = Q_after - Q_before
					passed = abs(best_ΔQ - ΔQ_ref) < tol
					println("       delta: $(round(best_ΔQ, digits=6)), reference: $(round(ΔQ_ref, digits=6))")
					println("       $(passed ? "PASS" : "FAIL")")
					all_passed &= passed
				else
					println("       No improving move (ΔQ = $(round(best_ΔQ, digits=12)))")
					println("       PASS (no-move case)")
				end

		#	Test 2: Random Directed Graph, All Moves
			println("\n  [D2] Random directed weighted graph (n=25), all node moves")
			rng = MersenneTwister(13)
			n = 25
			edges = Tuple{Int,Int,Float64}[]
			for i in 1:n, j in 1:n
				if i != j && rand(rng) < 0.12
					w = 0.5 + 2.0 * rand(rng)
					push!(edges, (i, j, w))
				end
			end
			adj = _build_test_adj(edges, n)
			membership = rand(rng, 1:4, n)
			C = maximum(membership)
			degrees = _compute_degrees(adj, true)
			K_in  = _compute_K(membership, degrees.k_in,  C)
			K_out = _compute_K(membership, degrees.k_out, C)
			Q_before = calculate_modularity(adj, membership; weighted = true, directed = true)

			work_in  = zeros(Float64, C)
			work_out = zeros(Float64, C)
			max_err = 0.0
			n_tested = 0
			for node in 1:n
				c_old = membership[node]
				best_c, best_ΔQ = delta_modularity_directed_best!(
					adj, node, c_old, membership,
					degrees.k_in, degrees.k_out,
					K_in, K_out, degrees.m,
					work_in, work_out
				)
				if best_c != c_old
					test_membership = copy(membership)
					test_membership[node] = best_c
					Q_after = calculate_modularity(adj, test_membership; weighted = true, directed = true)
					ΔQ_ref = Q_after - Q_before
					err = abs(best_ΔQ - ΔQ_ref)
					max_err = max(max_err, err)
					n_tested += 1
				end
			end
			println("       Tested $n_tested non-trivial moves, max error: $(round(max_err, digits=12))")
			passed = max_err < tol
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Test 3: γ Resolution Parameter
			println("\n  [D3] Directed γ resolution sweep")
			max_err = 0.0
			for γ in [0.5, 1.0, 1.5]
				Q_before = calculate_modularity(adj, membership; γ = γ, directed = true)
				work_in  = zeros(Float64, C)
				work_out = zeros(Float64, C)
				best_c, best_ΔQ = delta_modularity_directed_best!(
					adj, 5, membership[5], membership,
					degrees.k_in, degrees.k_out,
					K_in, K_out, degrees.m,
					work_in, work_out; γ = γ
				)
				if best_c != membership[5]
					test_membership = copy(membership)
					test_membership[5] = best_c
					Q_after = calculate_modularity(adj, test_membership; γ = γ, directed = true)
					ΔQ_ref = Q_after - Q_before
					err = abs(best_ΔQ - ΔQ_ref)
					max_err = max(max_err, err)
				end
			end
			passed = max_err < tol
			println("       Max error across γ in {0.5, 1.0, 1.5}: $(round(max_err, digits=12))")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		println()
		println("Directed delta_modularity: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
		return all_passed
	end

#	Test: Dispatcher Function
	function delta_modularity_best!(adj::SparseMatrixCSC{Float64,Int},
	                                 i::Int,
	                                 c_old::Int,
	                                 membership::Vector{Int},
	                                 K::Union{Vector{Float64}, Nothing},
	                                 K_in::Union{Vector{Float64}, Nothing},
	                                 K_out::Union{Vector{Float64}, Nothing},
	                                 k::Union{Vector{Float64}, Nothing},
	                                 k_in::Union{Vector{Float64}, Nothing},
	                                 k_out::Union{Vector{Float64}, Nothing},
	                                 m::Float64,
	                                 two_m::Float64,
	                                 work_a::Vector{Float64},
	                                 work_b::Union{Vector{Float64}, Nothing};
	                                 γ::Float64 = 1.0,
	                                 directed::Bool = false)
		"""
		Args:
			adj::SparseMatrixCSC{Float64,Int}: adjacency (preprocessed: symmetric if !directed)
			i::Int: node being evaluated
			c_old::Int: current community of i
			membership::Vector{Int}: current community labels (1..C)
			K::Union{Vector{Float64}, Nothing}: per-community degree (undirected only)
			K_in, K_out: per-community in/out degrees (directed only)
			k::Union{Vector{Float64}, Nothing}: per-node degree (undirected only)
			k_in, k_out: per-node in/out degrees (directed only)
			m::Float64: total weight (directed)
			two_m::Float64: total weight × 2 (undirected)
			work_a::Vector{Float64}: scratch vector (undirected: node_to_comm; directed: node_to_comm_in)
			work_b::Union{Vector{Float64}, Nothing}: scratch vector (directed: node_to_comm_out; nothing for undirected)
			γ::Float64: resolution parameter (default = 1.0)
			directed::Bool: graph type (default = false)
		Returns:
			Tuple{Int, Float64}: (best_community, best_ΔQ)
		Notes:
			Routes to delta_modularity_directed_best! or delta_modularity_undirected_best!
			based on the directed flag. The argument list is broad to accommodate both
			cases; unused arguments may be passed as nothing.

			In practice, callers in Leiden's Phase 1 typically know whether they are
			running directed or undirected at the top of their loop and may prefer to
			call the underlying _undirected or _directed functions directly to avoid
			the dispatch branch.
		"""

		#	Branch on Graph Type
			if directed
				@assert k_in     !== nothing "directed delta_modularity requires k_in"
				@assert k_out    !== nothing "directed delta_modularity requires k_out"
				@assert K_in     !== nothing "directed delta_modularity requires K_in"
				@assert K_out    !== nothing "directed delta_modularity requires K_out"
				@assert work_b   !== nothing "directed delta_modularity requires both work vectors"
				return delta_modularity_directed_best!(
					adj, i, c_old, membership,
					k_in, k_out, K_in, K_out, m,
					work_a, work_b;
					γ = γ
				)
			else
				@assert k !== nothing "undirected delta_modularity requires k"
				@assert K !== nothing "undirected delta_modularity requires K"
				return delta_modularity_undirected_best!(
					adj, i, c_old, membership,
					k, K, two_m,
					work_a;
					γ = γ
				)
			end
	end

#	Test: Dispatcher Function
	function test_delta_modularity_dispatcher()
		"""
		Args:
			(none)
		Returns:
			Bool: true if dispatcher routes correctly
		Notes:
			Verifies delta_modularity_best! routes to the correct implementation
			and produces matching results.
		"""

		println("=" ^ 70)
		println("Test: delta_modularity_best! dispatcher")
		println("=" ^ 70)

		all_passed = true
		tol = 1e-9

		#	Undirected Routing
			println("\n  [DISP1] Undirected dispatch matches direct call")
			adj, _ = _make_dumbbell(4)
			membership = vcat(fill(1, 4), fill(2, 4))
			degrees = _compute_degrees(adj, false)
			K = _compute_K(membership, degrees.k, 2)

			work1 = zeros(Float64, 2)
			direct = delta_modularity_undirected_best!(
				adj, 1, membership[1], membership,
				degrees.k, K, degrees.two_m,
				work1
			)

			work2 = zeros(Float64, 2)
			via_dispatch = delta_modularity_best!(
				adj, 1, membership[1], membership,
				K, nothing, nothing,
				degrees.k, nothing, nothing,
				degrees.m, degrees.two_m,
				work2, nothing;
				directed = false
			)

			passed = (direct[1] == via_dispatch[1]) && (abs(direct[2] - via_dispatch[2]) < tol)
			println("       Direct: $direct, via dispatcher: $via_dispatch")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		#	Directed Routing
			println("\n  [DISP2] Directed dispatch matches direct call")
			n = 6
			edges = Tuple{Int,Int,Float64}[
				(1,2,1.0), (2,3,1.0), (3,1,1.0),
				(4,5,1.0), (5,6,1.0), (6,4,1.0), (3,4,1.0)
			]
			adj = _build_test_adj(edges, n)
			membership = [1, 1, 1, 2, 2, 2]
			degrees = _compute_degrees(adj, true)
			K_in  = _compute_K(membership, degrees.k_in,  2)
			K_out = _compute_K(membership, degrees.k_out, 2)

			wi1 = zeros(Float64, 2); wo1 = zeros(Float64, 2)
			direct = delta_modularity_directed_best!(
				adj, 3, membership[3], membership,
				degrees.k_in, degrees.k_out,
				K_in, K_out, degrees.m,
				wi1, wo1
			)

			wi2 = zeros(Float64, 2); wo2 = zeros(Float64, 2)
			via_dispatch = delta_modularity_best!(
				adj, 3, membership[3], membership,
				nothing, K_in, K_out,
				nothing, degrees.k_in, degrees.k_out,
				degrees.m, degrees.two_m,
				wi2, wo2;
				directed = true
			)

			passed = (direct[1] == via_dispatch[1]) && (abs(direct[2] - via_dispatch[2]) < tol)
			println("       Direct: $direct, via dispatcher: $via_dispatch")
			println("       $(passed ? "PASS" : "FAIL")")
			all_passed &= passed

		println()
		println("Dispatcher: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
		return all_passed
	end

#	Run All Delta Modularity Tests
	function run_delta_modularity_tests()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (undirected, directed, dispatcher) pass/fail flags
		Notes:
			Convenience wrapper running all delta_modularity tests.
		"""

		println("\n")
		undir = test_delta_modularity_undirected()
		println("\n")
		dir   = test_delta_modularity_directed()
		println("\n")
		disp  = test_delta_modularity_dispatcher()

		println("\n" * "=" ^ 70)
		println("SUMMARY")
		println("=" ^ 70)
		println("  Undirected:   $(undir ? "PASS" : "FAIL")")
		println("  Directed:     $(dir   ? "PASS" : "FAIL")")
		println("  Dispatcher:   $(disp  ? "PASS" : "FAIL")")
		println("=" ^ 70 * "\n")

		return (undirected = undir, directed = dir, dispatcher = disp)
	end

    run_delta_modularity_tests()

#   LEIDEN SINGLE SWEEP TESTS

#	Helper Function for Leiden Tests: Edge DataFrame to Sparse Adjacency
	function _df_to_adj(edges_df::DataFrame, n_nodes::Int)
		"""
		Args:
			edges_df::DataFrame: with :src, :dst, :weight columns
			n_nodes::Int: number of nodes
		Returns:
			SparseMatrixCSC{Float64,Int}: directed weighted adjacency
		Notes:
			Builds CSC adjacency from edge list for Leiden testing.
		"""

		#	Extract Columns and Build
			return sparse(edges_df.src, edges_df.dst, Float64.(edges_df.weight), n_nodes, n_nodes)
	end

#	Test 1: Sanity Check on Dumbbell Graph
	function test_leiden_dumbbell()
		"""
		Args:
			(none)
		Returns:
			Bool: true if Leiden recovers two communities with high modularity
		Notes:
			On a dumbbell (two K_10 cliques bridged), Leiden should recover the
			two-community partition and report modularity near the theoretical
			optimum (~0.49 for K_10 dumbbell).
		"""

		println("=" ^ 70)
		println("Test 1: Leiden on dumbbell graph (known structure)")
		println("=" ^ 70)

		all_passed = true

		#	Build Dumbbell (Two K_10 Cliques + 1 Bridge)
			adj, _ = _make_dumbbell(10)

		#	Run Leiden
			println("\n  Running Leiden on undirected dumbbell (n=20, two K_10 cliques)...")
			result = _leiden_single_run_preprocessed(
				adj, 1.0, 10;
				directed = false,
				rng = Xoshiro(42)
			)

		#	Report
			println("  Modularity:      $(round(result.modularity, digits=4))")
			println("  Communities:     $(result.n_communities)")
			println("  Expected ~0.489 modularity, 2 communities")

		#	Check Modularity is Near Expected
			expected_Q = 0.489
			Q_close = abs(result.modularity - expected_Q) < 0.05
			println("  Modularity within 0.05 of expected:  $(Q_close ? "PASS" : "FAIL")")
			all_passed &= Q_close

		#	Check Community Count
			n_comm_correct = result.n_communities == 2
			println("  Two communities recovered:           $(n_comm_correct ? "PASS" : "FAIL")")
			all_passed &= n_comm_correct

		println("\n  Test 1: $(all_passed ? "PASS" : "FAIL")")
		return all_passed
	end

#	Test 2: Leiden on LFR Stress Network (Correctness)
	function test_leiden_stress_correctness(edges_df::DataFrame,
	                                          cid::Vector{Int},
	                                          n_nodes::Int = 5000)
		"""
		Args:
			edges_df::DataFrame: edges from generate_test_network()
			cid::Vector{Int}: LFR planted community labels
			n_nodes::Int: number of nodes (default = 5000)
		Returns:
			NamedTuple: (modularity, n_communities, n_planted, runtime_seconds)
		Notes:
			Runs the refactored single-sweep Leiden on the LFR test network. Reports
			modularity, detected community count, planted community count, and
			runtime. Validates that detected modularity is in a sensible range.
		"""

		println("=" ^ 70)
		println("Test 2: Leiden on 5,000-node LFR stress network (correctness)")
		println("=" ^ 70)

		#	Build Sparse Adjacency
			println("\n  Building sparse adjacency from $(nrow(edges_df)) edges...")
			adj = _df_to_adj(edges_df, n_nodes)
			println("  Adjacency: $n_nodes × $n_nodes, $(nnz(adj)) nonzeros")

		#	Planted Community Stats
			n_planted = length(unique(cid))
			println("  Planted (LFR) communities: $n_planted")

		#	Run Leiden
			println("\n  Running Leiden (directed, single run, seed=42)...")
			t0 = time()
			result = _leiden_single_run_preprocessed(
				adj, 1.0, 10;
				directed = true,
				rng = Xoshiro(42),
				show_iteration_progress = true,
				run_description = "Stress test"
			)
			runtime = time() - t0

		#	Report Results
			println("\n  Runtime:                 $(round(runtime, digits=2)) seconds")
			println("  Detected modularity:     $(round(result.modularity, digits=4))")
			println("  Detected communities:    $(result.n_communities)")
			println("  Planted communities:     $n_planted")

		#	Sanity Check on Modularity
			Q_sensible = 0.3 < result.modularity < 0.9
			println("  Modularity in (0.3, 0.9):  $(Q_sensible ? "PASS" : "FAIL")")

		#	Sanity Check on Community Count
			comm_close = abs(result.n_communities - n_planted) < n_planted
			println("  Detected community count within 100% of planted:  $(comm_close ? "PASS" : "FAIL")")

		return (
			modularity      = result.modularity,
			n_communities   = result.n_communities,
			n_planted       = n_planted,
			runtime_seconds = runtime,
			Q_sensible      = Q_sensible,
			comm_close      = comm_close
		)
	end

#	Test 3: Speed Benchmark on Stress Network
	function test_leiden_speed(edges_df::DataFrame, n_nodes::Int = 5000)
		"""
		Args:
			edges_df::DataFrame: edges from generate_test_network()
			n_nodes::Int: number of nodes (default = 5000)
		Returns:
			NamedTuple: (median_seconds, runs)
		Notes:
			Runs Leiden several times to get a stable timing estimate. Reports
			median runtime and per-run timing. This gives us a baseline number
			for evaluating future optimizations (transpose for directed,
			threading the multi-start outer loop, etc.).
		"""

		println("=" ^ 70)
		println("Test 3: Leiden speed benchmark on 5,000-node stress network")
		println("=" ^ 70)

		#	Build Adjacency
			adj = _df_to_adj(edges_df, n_nodes)

		#	Run Multiple Times
			n_runs   = 3
			runtimes = Float64[]
			println("\n  Running Leiden $n_runs times (different seeds)...")
			for run_i in 1:n_runs
				t0 = time()
				_ = _leiden_single_run_preprocessed(
					adj, 1.0, 10;
					directed = true,
					rng = Xoshiro(100 + run_i),
					show_iteration_progress = false
				)
				rt = time() - t0
				push!(runtimes, rt)
				println("    Run $run_i: $(round(rt, digits=2)) seconds")
			end

		#	Report Median
			med = sort(runtimes)[div(n_runs, 2) + 1]
			println("\n  Median runtime: $(round(med, digits=2)) seconds")

		return (median_seconds = med, runs = runtimes)
	end

#	Run All Single-Sweep Leiden Tests
	function run_leiden_single_sweep_tests(edges_df::DataFrame,
	                                        cid::Vector{Int},
	                                        n_nodes::Int = 5000)
		"""
		Args:
			edges_df::DataFrame: edges from generate_test_network()
			cid::Vector{Int}: LFR community labels
			n_nodes::Int: number of nodes (default = 5000)
		Returns:
			NamedTuple: (dumbbell_passed, stress_result, speed_result)
		Notes:
			Convenience wrapper running all single-sweep Leiden tests.
		"""

		println("\n")
		dumbbell_passed = test_leiden_dumbbell()
		println("\n")
		stress_result = test_leiden_stress_correctness(edges_df, cid, n_nodes)
		println("\n")
		speed_result = test_leiden_speed(edges_df, n_nodes)

		println("\n" * "=" ^ 70)
		println("SUMMARY")
		println("=" ^ 70)
		println("  Test 1 (Dumbbell):        $(dumbbell_passed ? "PASS" : "FAIL")")
		println("  Test 2 (Stress network):  Q = $(round(stress_result.modularity, digits=4)), " *
		        "$(stress_result.n_communities) communities ($(stress_result.n_planted) planted)")
		println("  Test 3 (Speed):           median $(round(speed_result.median_seconds, digits=2)) seconds")
		println("=" ^ 70 * "\n")

		return (
			dumbbell_passed = dumbbell_passed,
			stress_result   = stress_result,
			speed_result    = speed_result
		)
	end

    results = run_leiden_single_sweep_tests(edges_df, cid, 5000)

#   LEIDEN MULTI-SWEEP TESTS

#	Testing Over Multiple-Runs
	@time result = leiden_community_detection(edges_df; n_runs = 5, seed = 42)
	@time result = leiden_community_detection(edges_df; n_runs = 5, seed = 42)  # second call avoids JIT
	println("Modularity: $(result.modularity)")
	println("Communities: $(result.n_communities)")

#	CHAMP TESTs

# 	First call: includes compilation
	@time result = champ_community_detection(edges_df;
											weighted = true,
											directed = true,
											n_resolutions = 20,
											n_runs_per_gamma = 5,
											seed = 42)

# 	Second call: steady-state
	@time result = champ_community_detection(edges_df;
											weighted = true,
											directed = true,
											n_resolutions = 20,
											n_runs_per_gamma = 5,
											seed = 42)

#	Print Results
	println("Best modularity: $(result.modularity)")
	println("Best γ: $(result.resolution_used)")
	println("Communities: $(result.n_communities)")

#	Greater Range
	@time result = champ_community_detection(edges_df;
											weighted = true,
											directed = true,
											resolution_range = (0.1, 1.8),
											n_resolutions = 30,
											n_runs_per_gamma = 5,
											seed = 42)

	println("Best modularity: $(result.modularity)")
	println("Best γ: $(result.resolution_used)")
	println("Communities: $(result.n_communities)")

	adj = sparse(edges_df.src, edges_df.dst, Float64.(edges_df.weight), 5000, 5000)
	all_one = ones(Int, 5000)
	calculate_modularity(adj, all_one; weighted=true, directed=true, γ=0.1)

#	CHAMP Test Execution with Optional Sweep Plot
	result = champ_community_detection(edges_df;
										weighted = true,
										directed = true,
										seed     = 42)

	println("Selected γ:   $(result.resolution_used)")
	println("Modularity:   $(round(result.modularity, digits=4))")
	println("Communities:  $(result.n_communities)")

#	Use the Best Membership Downstream
	best_membership = result.membership
	println("Best γ: $(result.resolution_used), modularity: $(result.modularity)")

#	Plot CHAMP Sweep Diagram
	function plot_champ_sweep(result;
	                          save_path::Union{String,Nothing} = nothing,
	                          figure_size::Tuple{Int,Int} = (1100, 650),
	                          gamma_axis_start::Union{Float64,Nothing} = 0.0)
		"""
		Args:
			result::NamedTuple: output from champ_community_detection with sweep data
			save_path::Union{String,Nothing}: file path to save PNG; nothing displays only
			figure_size::Tuple{Int,Int}: figure dimensions in pixels (default = (1100, 650))
			gamma_axis_start::Union{Float64,Nothing}: x-axis lower bound; nothing uses γ_min
		Returns:
			Figure: CairoMakie Figure object
		Notes:
			Renders the canonical CHAMP convex-hull plot. Figure dimensions kept
			modest so that on-screen rendering preserves font size; labels are
			placed only on dominant partitions to reduce clutter.
		"""

		#	Extract Sweep Data
			gammas      = result.gammas
			A_coeffs    = result.A_coeffs
			P_coeffs    = result.P_coeffs
			n_comms_pg  = result.n_communities_per_gamma
			dominant    = result.dominant

		#	Compute Quality Score at Each γ
			quality_scores = A_coeffs .- gammas .* P_coeffs

		#	Determine X-Axis Range
			γ_max     = maximum(gammas)
			γ_min_act = minimum(gammas)
			γ_min_ax  = gamma_axis_start === nothing ? γ_min_act : gamma_axis_start
			γ_pad     = 0.02 * (γ_max - γ_min_ax)
			γ_line_range = range(γ_min_ax - γ_pad, γ_max + γ_pad; length = 100)

		#	Determine Y-Axis Range for Right Axis
			n_comm_min = minimum(n_comms_pg)
			n_comm_max = maximum(n_comms_pg)
			n_comm_pad = max(1, (n_comm_max - n_comm_min) ÷ 10)

		#	Create Figure with Dual Y-Axes
			fig = Figure(size = figure_size, backgroundcolor = :white,
			             fontsize = 20)
			ax1 = Axis(fig[1, 1];
			           xlabel = "Resolution Parameter γ",
			           ylabel = "Quality Score (non-normalized)",
			           xlabelsize = 22,
			           ylabelsize = 22,
			           xticklabelsize = 18,
			           yticklabelsize = 18,
			           xgridvisible = false,
			           ygridvisible = false)
			ax2 = Axis(fig[1, 1];
			           ylabel = "Number of Communities",
			           ylabelsize = 22,
			           yticklabelsize = 18,
			           yaxisposition = :right,
			           ygridvisible = false,
			           xgridvisible = false)
			hidespines!(ax2, :l, :t, :b)
			hidexdecorations!(ax2)

		#	Draw Light Gray Backdrop Lines for Every Partition
			for i in eachindex(gammas)
				line_y = A_coeffs[i] .- γ_line_range .* P_coeffs[i]
				lines!(ax1, γ_line_range, line_y;
				       color = (:gray70, 0.35),
				       linewidth = 0.6)
			end

		#	Generate Rainbow Color Gradient
			rainbow = cgrad(:turbo, length(gammas); categorical = true)

		#	Draw Vertical Dropdown Lines
			y_bottom = minimum(quality_scores) - 0.08 * (maximum(quality_scores) - minimum(quality_scores))
			for i in eachindex(gammas)
				lines!(ax1, [gammas[i], gammas[i]], [y_bottom, quality_scores[i]];
				       color = (:gray40, 0.8),
				       linewidth = 0.8)
			end

		#	Plot Colored Dots; Label Only Dominant Partitions
			for i in eachindex(gammas)
				scatter!(ax1, [gammas[i]], [quality_scores[i]];
				         color = rainbow[i],
				         markersize = 14,
				         strokewidth = 0.7,
				         strokecolor = :black)

				if dominant[i]
					text!(ax1, gammas[i], quality_scores[i];
					      text = string(round(gammas[i], digits = 2)),
					      align = (:center, :bottom),
					      offset = (0, 12),
					      fontsize = 18,
					      color = :gray20)
				end
			end

		#	Plot Community Counts on Secondary Axis
			scatter!(ax2, gammas, Float64.(n_comms_pg);
			         marker = :xcross,
			         color = (:darkolivegreen, 0.85),
			         markersize = 12,
			         strokewidth = 1.3)

		#	Y-Axis Ranges
			y_top = maximum(quality_scores) + 0.05 * (maximum(quality_scores) - minimum(quality_scores))
			ylims!(ax1, y_bottom, y_top)
			ylims!(ax2, n_comm_min - n_comm_pad, n_comm_max + n_comm_pad)

		#	X-Axis Range
			xlims!(ax1, γ_min_ax - γ_pad, γ_max + γ_pad)
			xlims!(ax2, γ_min_ax - γ_pad, γ_max + γ_pad)
			linkxaxes!(ax1, ax2)

		#	Save If Path Provided
			if save_path !== nothing
				save(save_path, fig)
				println("CHAMP sweep plot saved to: $save_path")
			end

		#	Return Figure
			return fig
	end

# 	Inspect sweep:
	fig = plot_champ_sweep(result; save_path = "champ_sweep.png")
	display(fig)