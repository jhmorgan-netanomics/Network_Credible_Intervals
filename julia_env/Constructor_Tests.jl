#Testing the two Constructor Functions: Credible Inverals & Bias Scores (Used When a Groundtruth Network is Available)
#Jonathan H. Morgan
#8 June 2026

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
    using Arrow
    using BenchmarkTools
    using CairoMakie
    using CSV
    using DataFrames
    using Graphs
    using GraphMakie
    using Random
    using Statistics
    using Network_Credible_Intervals

    const NDG = Network_Credible_Intervals.network_degeneracy

#################
#   FUNCTIONS   #
#################

#	Knuth Poisson (dependency-free) -- integer edge weights like the Python reference
	function _rand_poisson(rng::AbstractRNG, lambda::Real)
		L = exp(-lambda); k = 0; p = 1.0
		while true
			k += 1
			p *= rand(rng)
			p <= L && return k - 1
		end
	end

#   Random Network Generators
    function make_random_weighted_network(; n::Int = 50, density::Float64 = 0.08,
										   weight_lambda::Float64 = 3.0, seed::Int = 42)
		g   = erdos_renyi(n, density; is_directed = true, seed = seed)
		rng = MersenneTwister(seed + 1)
		src = String[]; dst = String[]; w = Float64[]
		for e in edges(g)
			push!(src, string(Graphs.src(e))); push!(dst, string(Graphs.dst(e)))
			push!(w, Float64(max(1, _rand_poisson(rng, weight_lambda))))
		end
		ids = string.(1:n)
		return (edges = DataFrame(src = src, dst = dst, weight = w),
				nodes = DataFrame(id = ids, label = ids))   # String ids: _graph_to_sparse_matrix does String.(ids)
	end

	function make_hub_weighted_network(; n::Int = 60, m::Int = 2,
									   weight_lambda::Float64 = 3.0, seed::Int = 42)
		g   = barabasi_albert(n, m; seed = seed)
		rng = MersenneTwister(seed + 1)
		src = String[]; dst = String[]; w = Float64[]
		for e in edges(g)
			u = Graphs.src(e); v = Graphs.dst(e)
			s, d = u > v ? (u, v) : (v, u)                # newer (higher idx) -> older hub (lower idx)
			push!(src, string(s)); push!(dst, string(d))
			push!(w, Float64(max(1, _rand_poisson(rng, weight_lambda))))
		end
		ids = string.(1:n)
		return (edges = DataFrame(src = src, dst = dst, weight = w),
				nodes = DataFrame(id = ids, label = ids))
	end

#	Binarize: collapse all weights to 1 (the unit-weight / whole-tie limit). Same
#	skeleton as the weighted net, so the two runs are directly comparable.
	function binarize(net)
		e = copy(net.edges); e.weight .= 1.0
		return (edges = e, nodes = net.nodes)
	end

#	Validation driver: degrade at target_rho with OUR sampler, reconstruct with
#	OUR pipeline (rho = realized), compare to truth across pi_edge. Computes all
#	measures from the SAME degraded sample per rep (one degrade + one reconstruct).
    function run_validation(true_net; measures, pi_edge_grid, target_rho::Float64,
							weighted::Bool, B::Int = 300, n_rep::Int = 15,
							allocation::Symbol = :observed,
							node_tilt::Float64 = 0.10, base_seed::Int = 99,
                            recon_rho::Union{Nothing,Real} = nothing)

		directed    = true
		names       = [String(nm) for (nm, _) in measures]
		metric_dict = Dict(nm => f for (nm, f) in measures)
		true_values = Dict(String(nm) => f(true_net.edges, true_net.nodes) for (nm, f) in measures)

		acc = Dict(nm => (obs = Float64[], med = Float64[], lo = Float64[], hi = Float64[]) for nm in names)
		rho_real = Float64[]; org = Float64[]; nom = Float64[]
		ful = Float64[]; zed = Float64[]; ceil_cnt = Int[]

		for (li, pe) in enumerate(pi_edge_grid)
			per = Dict(nm => (o = Float64[], m = Float64[], l = Float64[], h = Float64[]) for nm in names)
			rr = Float64[]; og = Float64[]; nmv = Float64[]; fl = Float64[]; zd = Float64[]; cf = 0

			for rep in 1:n_rep
				seed = base_seed + 1000 * li + rep

				#	Degrade: emergent edge arm at target_rho. weighted selects the
				#	mechanism (weight removal vs whole-tie removal).
					deg = NDG.generate_missingness_mask(true_net.edges;
							nodes          = true_net.nodes,
							directed       = directed,
							weighted       = weighted,
							node_loss      = :emergent,
							target_pi_node = node_tilt,          # tilt-only calibration in :emergent
							target_pi_edge = pe,
							target_rho     = target_rho,
							seed           = seed,
							return_removed = true)
					obs_edges = deg.observed_edges
					obs_nodes = deg.observed_nodes

				#	Nominated non-respondents kept in the roster (still receive ties):
				#	the Stage 0.5 inputs.
					miss_names  = Set(true_net.nodes.id[v] for v in deg.missing_nodes)
					nominee_idx = [i for (i, id) in enumerate(obs_nodes.id) if id in miss_names]

				#	Reconstruct with the KNOWN realized priors, INCLUDING realized rho
				#	(the tilt the reconstruction needs to mirror removal).
					ci = network_credible_intervals(obs_edges, obs_nodes;
							metrics                  = metric_dict,
							directed                 = directed,
							weighted                 = weighted,
							pi_node                  = deg.realized_pi_node,
							pi_edge                  = deg.realized_pi_edge,
							rho                      = isnothing(recon_rho) ? deg.realized_rho : Float64(recon_rho),
							allocation               = allocation,
							community_method         = :champ,
							K                        = 4,
							partially_observed_nodes = nominee_idx,
							n_replicates             = B,
							prob                     = 0.95)

				for (nm, f) in measures
					key = String(nm); iv = ci.intervals[nm]
					push!(per[key].o, f(obs_edges, obs_nodes))
					push!(per[key].m, iv.median); push!(per[key].l, iv.lower); push!(per[key].h, iv.upper)
				end
				push!(rr, deg.realized_rho); push!(og, deg.n_organic_losses)
				push!(nmv, deg.n_nominations); push!(fl, deg.n_full_removal); push!(zd, deg.n_edges_zeroed)
				(deg.field_status == :ceiling_hit) && (cf += 1)
			end

			for nm in names
				push!(acc[nm].obs, mean(per[nm].o)); push!(acc[nm].med, mean(per[nm].m))
				push!(acc[nm].lo, mean(per[nm].l));  push!(acc[nm].hi, mean(per[nm].h))
			end
			push!(rho_real, mean(rr)); push!(org, mean(og)); push!(nom, mean(nmv))
			push!(ful, mean(fl)); push!(zed, mean(zd)); push!(ceil_cnt, cf)
		end

		#	Degradation diagnostics (did the tilt engage? did the node machinery fire?)
			println("\n=== target rho = ", target_rho, ",  weighted = ", weighted,
					",  allocation = ", allocation, " ===")
			println("Degradation diagnostics (mean over ", n_rep, " reps):")
			println(rpad("pi_edge", 9), rpad("real_rho", 10), rpad("organic", 9),
					rpad("nomin", 8), rpad("full_rem", 10), rpad("zeroed", 9), "ceil/n")
			for i in eachindex(pi_edge_grid)
				println(rpad(round(pi_edge_grid[i]; digits = 2), 9),
						rpad(round(rho_real[i];     digits = 3), 10),
						rpad(round(org[i];          digits = 2), 9),
						rpad(round(nom[i];          digits = 2), 8),
						rpad(round(ful[i];          digits = 2), 10),
						rpad(round(zed[i];          digits = 2), 9),
						string(ceil_cnt[i], "/", n_rep))
			end

		#	Per-measure coverage tables + assembled results
			results = NamedTuple[]
			for (nm, _) in measures
				key = String(nm); tv = true_values[key]; a = acc[key]
				push!(results, (measure_name = key, pi_edge = collect(pi_edge_grid),
								observed = a.obs, post_median = a.med, ci_lo = a.lo, ci_hi = a.hi,
								true_value = tv, realized_rho = rho_real))

				println("\nMeasure: ", key, "   (true value = ", round(tv; digits = 3), ")")
				println(rpad("pi_edge", 9), rpad("observed", 11), rpad("post.med.", 11),
						rpad("95% CI", 22), "covers true")
				for i in eachindex(pi_edge_grid)
					covers = a.lo[i] <= tv <= a.hi[i]
					ci_str = "[" * string(round(a.lo[i]; digits = 2)) * ", " *
								   string(round(a.hi[i]; digits = 2)) * "]"
					println(rpad(round(pi_edge_grid[i]; digits = 2), 9),
							rpad(round(a.obs[i];          digits = 3), 11),
							rpad(round(a.med[i];          digits = 3), 11),
							rpad(ci_str, 22), covers ? "YES" : "no")
				end
			end
		return results
	end

#	Plot (CairoMakie): one panel per measure -- CI band, posterior median,
#	observed (no correction), and the true-value reference line.
	function plot_validation(results, output_path, title, xlabel)
		np  = length(results)
		fig = Figure(size = (600 * np, 540))

		axes_vec = Axis[]
		for (i, res) in enumerate(results)
			ax = Axis(fig[2, i];
					  title  = "$(res.measure_name)  (true = $(round(res.true_value; digits = 2)))",
					  xlabel = xlabel, ylabel = res.measure_name)

			band!(ax, res.pi_edge, res.ci_lo, res.ci_hi;
				  color = (:steelblue, 0.25), label = "95% credible interval")

			lines!(ax, res.pi_edge, res.post_median;
				   color = :steelblue, linewidth = 2, label = "Posterior median")
			scatter!(ax, res.pi_edge, res.post_median; color = :steelblue, marker = :circle)

			lines!(ax, res.pi_edge, res.observed;
				   color = :red, linewidth = 1.5, linestyle = :dash,
				   label = "Observed (no correction)")
			scatter!(ax, res.pi_edge, res.observed; color = :red, marker = :rect)

			hlines!(ax, [res.true_value];
					color = :black, linewidth = 1, linestyle = :dot,
					label = "True value")

			push!(axes_vec, ax)
		end

		#	Supertitle (row 1) and one shared horizontal legend beneath the panels
		#	(row 3). The legend is built from a single axis since every panel carries
		#	the same labeled series; tellheight keeps it to a tight strip and
		#	tellwidth = false stops it from stretching the panel columns.
			Label(fig[1, :], title; fontsize = 16, font = :bold)
			Legend(fig[3, :], axes_vec[1];
				   orientation = :horizontal, framevisible = true,
				   tellheight = true, tellwidth = false, nbanks = 1)

		save(output_path, fig)
		println("\nPlot saved to: ", output_path)
		return output_path
	end

########################################
#   IMPORT DEGENERATE NETWORK CORPUS   #
########################################


#####################################
#   BASIC: STYLIZED NETWORK TESTS   #
#####################################

#   WEIGHT TRANSFORMATION TESTS

#	Unit Test: transform_distance_weights (pure, deterministic, hand-derived)
	function test_transform_distance_weights()
		"""
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Pure unit test on the distance->count weight transform. Every expected value is
			hand-derived from the three method definitions, with weights chosen to avoid .5
			rounding ties. Covers each method, zero-distance handling, edge dropping, the
			exp_decay median-targeting, the guards (bad method / missing column / negative
			distance / missing-or-nonpositive tau), custom weight_col, non-mutation, and the
			return shape. Requires DataFrames, Statistics, and transform_distance_weights.
		"""
		println("─" ^ 70)
		println("Unit test: transform_distance_weights")
		println("─" ^ 70)

		#	Local Helper: did f() throw an exception of type T?
			function caught(T, f)
				try
					f(); return false
				catch err
					return err isa T
				end
			end

		#	scaled_reciprocal (default): c = max = 8; w' = round(8 / w)
		#	round(8/2)=4, round(8/8)=1, round(8/4)=2  ->  [4, 1, 2]; never drops
			base = DataFrame(src = [1, 2, 3], dst = [2, 3, 1], weight = [2.0, 8.0, 4.0])
			r = transform_distance_weights(base)
			c_recip = r.method == :scaled_reciprocal && r.c == 8.0 && r.n_dropped == 0 &&
					  r.edges.weight == [4, 1, 2] && nrow(r.edges) == 3

		#	scaled_reciprocal with a zero-distance edge: zero -> max count = round(8/2) = 4
		#	rest as above  ->  [4, 4, 1, 2]
			z  = DataFrame(src = [1, 2, 3, 4], dst = [2, 3, 4, 1], weight = [0.0, 2.0, 8.0, 4.0])
			rz = transform_distance_weights(z)
			c_zero = rz.edges.weight == [4, 4, 1, 2] && rz.n_dropped == 0 && rz.c == 8.0

		#	max_minus: w' = round(8 - w) = [6, 0, 4]; the max-distance edge -> 0 is dropped
			m = transform_distance_weights(base; method = :max_minus)
			c_maxminus = m.method == :max_minus && isnan(m.c) && m.n_dropped == 1 &&
						 m.edges.weight == [6, 4] && nrow(m.edges) == 2 && m.edges.src == [1, 3]

		#	max_minus, all equal distances -> all map to 0 -> empty result
			eq = DataFrame(src = [1, 2, 3], dst = [2, 3, 1], weight = [5.0, 5.0, 5.0])
			me = transform_distance_weights(eq; method = :max_minus)
			c_empty = me.n_dropped == 3 && nrow(me.edges) == 0 && isnan(me.c)

		#	exp_decay: decay = exp(-[2,8,4]/3); median = exp(-4/3); c = 10/exp(-4/3) ~= 37.937
		#	counts = round(c .* decay) = [19, 3, 10]; median transformed weight = 10
			e = transform_distance_weights(base; method = :exp_decay, tau = 3.0)
			c_exp = e.method == :exp_decay && e.n_dropped == 0 && e.edges.weight == [19, 3, 10] &&
					median(e.edges.weight) == 10.0 &&
					isapprox(e.c, 10.0 / median(exp.(-[2.0, 8.0, 4.0] ./ 3.0)); atol = 1e-9)

		#	Guards: exp_decay needs a positive tau
			c_tau = caught(ArgumentError, () -> transform_distance_weights(base; method = :exp_decay)) &&
					caught(ArgumentError, () -> transform_distance_weights(base; method = :exp_decay, tau = 0.0)) &&
					caught(ArgumentError, () -> transform_distance_weights(base; method = :exp_decay, tau = -1.0))

		#	Guards: bad method / missing column / negative distance
			c_guards = caught(ArgumentError, () -> transform_distance_weights(base; method = :bogus)) &&
					   caught(ArgumentError, () -> transform_distance_weights(base; weight_col = :nope)) &&
					   caught(DomainError,   () -> transform_distance_weights(
							   DataFrame(src = [1, 2], dst = [2, 1], weight = [-1.0, 3.0])))

		#	Custom weight_col: counts written to :weight, source column preserved
			d  = DataFrame(src = [1, 2, 3], dst = [2, 3, 1], distance = [2.0, 8.0, 4.0])
			rd = transform_distance_weights(d; weight_col = :distance)
			c_col = rd.edges.weight == [4, 1, 2] && rd.edges.distance == [2.0, 8.0, 4.0]

		#	Input not mutated
			snapshot = copy(base.weight)
			transform_distance_weights(base)
			c_nomutate = base.weight == snapshot && nrow(base) == 3

		#	Return shape
			c_shape = propertynames(r) == (:edges, :n_dropped, :c, :method) &&
					  r.edges isa DataFrame && r.n_dropped isa Int && r.method isa Symbol

		#	Report
			checks = (("scaled_reciprocal",     c_recip),
					  ("zero-distance",          c_zero),
					  ("max_minus + drop",       c_maxminus),
					  ("max_minus all-equal",    c_empty),
					  ("exp_decay",              c_exp),
					  ("tau guard",              c_tau),
					  ("method/col/sign guards", c_guards),
					  ("custom weight_col",      c_col),
					  ("no mutation",            c_nomutate),
					  ("return shape",           c_shape))
			for (label, ok) in checks
				println("  ", rpad(label, 26), ok ? "YES" : "NO")
			end
			passed = all(last, checks)
			println("  ", rpad("Result", 26), passed ? "PASS ✓" : "FAIL ✗")

		#	Return
			details = join(["$(label)=$(ok)" for (label, ok) in checks], " ")
			return (passed = passed, details = details)
	end
    test_transform_distance_weights()

#   RANDOM NETWORK TESTS (Replicating Belutta Chapter 4)

#   Create Synthetic Network
    true_net_w = make_random_weighted_network(n = 50, density = 0.08, weight_lambda = 3.0, seed = 42)

#	Bellutta: observed allocation, reconstruction assumes MCAR (rho = 0) -- no rho argument,
#	so the tilt is flat and Stage 2 is pure proportional-to-observed.
	res_w_bellutta = run_validation(true_net_w;
								    measures = weighted_measures, pi_edge_grid = pi_edge_grid,
									target_rho = 0.0, weighted = true, B = 300, n_rep = 15,
									allocation = :observed, recon_rho = 0.0)
    plot_validation(res_w_bellutta, "credible_interval_random_weighted_bellutta.png",
					"Credible intervals under MCAR edge-weight missingness (target ρ = 0, random network, weighted, Bellutta)",
					"Proportion of edge weight missing  (π_edge)")

#   Deficit Method on Random
    res_w_deficit  = run_validation(true_net_w;
									measures = weighted_measures, pi_edge_grid = pi_edge_grid,
									target_rho = 0.0, weighted = true, B = 300, n_rep = 15,
									allocation = :deficit, recon_rho = nothing)
	plot_validation(res_w_deficit, "credible_interval_random_weighted_deficit.png",
					"Credible intervals under MCAR edge-weight missingness (target ρ = 0, random network, weighted, deficit)",
					"Proportion of edge weight missing  (π_edge)")

#   HUB & SPOKE NETWORK

#   Parameters
    NET_KIND   = :hub

#   Create Synthetic Network
    true_net_w = NET_KIND === :hub ?
		make_hub_weighted_network(n = 60, m = 2, weight_lambda = 3.0, seed = 42) :
		make_random_weighted_network(n = 50, density = 0.08, weight_lambda = 3.0, seed = 42)

	println("Synthetic network (", NET_KIND, "): N=", nrow(true_net_w.nodes),
			" E=", nrow(true_net_w.edges), " total weight=", sum(true_net_w.edges.weight))

#   Defining Test Parameters
	pi_edge_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

#	Weighted measures: in_degree(...).in_degree is in-STRENGTH when weighted=true.
	weighted_measures = [
		:top_in_strength  => (e, n) -> maximum(in_degree(e; nodes = n, weighted = true).in_degree),
		:in_strength_gini => (e, n) -> gini_coefficient(in_degree(e; nodes = n, weighted = true).in_degree),
	]

#	Binary measures: weighted=false -> in_degree(...).in_degree is the COUNT (degree).
	binary_measures = [
		:top_in_degree  => (e, n) -> maximum(in_degree(e; nodes = n, weighted = false).in_degree),
		:in_degree_gini => (e, n) -> gini_coefficient(in_degree(e; nodes = n, weighted = false).in_degree),
	]

#	(1) Negative-rho, WEIGHTED (Stage 2 active; tilt reshapes where weight lands)
	res_w_deficit  = run_validation(true_net_w;
									measures = weighted_measures, pi_edge_grid = pi_edge_grid,
									target_rho = -0.35, weighted = true, B = 300, n_rep = 15,
									allocation = :deficit, recon_rho = nothing)
	plot_validation(res_w_deficit, "credible_interval_negrho_weighted_deficit.png",
					"Credible intervals under centrality-tilted edge-weight missingness (target ρ = -0.35, weighted, deficit)",
					"Proportion of edge weight missing  (π_edge)")

#	Bellutta: observed allocation, reconstruction assumes MCAR (rho = 0) -- no rho argument,
#	so the tilt is flat and Stage 2 is pure proportional-to-observed.
	res_w_bellutta = run_validation(true_net_w;
								    measures = weighted_measures, pi_edge_grid = pi_edge_grid,
									target_rho = -0.35, weighted = true, B = 300, n_rep = 15,
									allocation = :observed, recon_rho = 0.0)
    plot_validation(res_w_bellutta, "credible_interval_negrho_weighted_bellutta.png",
						"Credible intervals under centrality-tilted edge-weight missingness (target ρ = -0.35, weighted, Bellutta)",
						"Proportion of edge weight missing  (π_edge)")

#	(2) Negative-rho, BINARY (same HUB skeleton; Stage 2 no-op -> topology recovery only)
#	Regenerate the hub net -- true_net_w currently holds the RANDOM net from the rho=0 cell.
	true_net_w = make_hub_weighted_network(n = 60, m = 2, weight_lambda = 3.0, seed = 42)
	true_net_b = binarize(true_net_w)

	res_b = run_validation(true_net_b;
						   measures = binary_measures, pi_edge_grid = pi_edge_grid,
						   target_rho = -0.35, weighted = false, B = 300, n_rep = 15,
						   allocation = :observed)        # vestigial: Stage 2 is off when weighted = false

#   Plot the Credible Intervals
	plot_validation(res_b, "credible_interval_negrho_binary.png",
					"Credible intervals under centrality-tilted edge missingness (target ρ = -0.35, binary)",
					"Proportion of edges missing  (π_edge)")