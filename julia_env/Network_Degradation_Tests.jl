#Network Degradation Tests
#Jonathan H. Morgan
#29 May 2026

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
using CSV
using DataFrames
using Dates
using Printf
using SparseArrays
using Statistics
using StatsBase
using Network_Credible_Intervals
using Network_Credible_Intervals.network_community_detection: _edgelist_to_sparse_matrix,
                                                              _graph_to_sparse_matrix,
                                                              _aggregate_multi_edges,
                                                              _is_symmetric

#################
#   FUNCTIONS   #
#################

#	Constructed Fixture: 50-node 4-regular ring (Test 6: ties)
	function _build_regular_ring_fixture(; n::Int = 50, k::Int = 4)
		"""
		Args:
			n::Int: number of nodes (default 50)
			k::Int: degree (must be even and < n); default 4
		Returns:
			NamedTuple (edges::DataFrame, nodes::DataFrame, metadata::NamedTuple)
				in the corpus format, with undirected, unweighted edges.
		Notes:
			Constructs a k-regular ring graph where each node is connected
			to the k nearest nodes on a circular layout. Every node has
			degree exactly k, so binarized degree is constant across the
			node set — the centrality vector min-max-normalizes to zero,
			triggering the constant-centrality path in the sampler.

			Used as the Test 6 fixture: targeting any non-zero rho on this
			network should produce realized_rho = NaN and bisection_status
			= :failed_other from the sampler's NaN-cor short-circuit.

			Nodes DataFrame has :id and :label columns to match the
			convention enforced by network_community_detection._graph_to_sparse_matrix.
			Both columns hold the same identifier strings ("n1", "n2", ...).
		"""
		k % 2 == 0 || throw(ArgumentError("k must be even, got $k"))
		k < n      || throw(ArgumentError("k must be less than n"))

		#	Each node connects to k/2 neighbors on each side of the ring
			src = String[]
			dst = String[]
			half_k = k ÷ 2
			for i in 1:n
				for offset in 1:half_k
					#	Wrap around the ring
						j = ((i - 1 + offset) % n) + 1
						push!(src, "n$i")
						push!(dst, "n$j")
				end
			end
			edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
			node_ids = ["n$i" for i in 1:n]
			nodes    = DataFrame(id = node_ids, label = node_ids)
			metadata = (directed = false, name = "regular_ring_$(n)x$(k)")

		return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Constructed Fixture: N-node star (Test 3: achievable-rho ceiling)
	function _build_star_fixture(; n::Int = 50, directed::Bool = true)
		"""
		Args:
			n::Int: number of nodes including the hub (default 50)
			directed::Bool: if true, all edges point from leaves to hub
				(in-degree concentrated at hub); if false, undirected.
				Default true to match the sampler's binarized-in-degree
				driver on directed networks.
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata.
		Notes:
			Construction: node 1 is the hub, nodes 2..n are leaves. When
			directed, all arcs are leaf -> hub, giving the hub in-degree
			(n-1) and all leaves in-degree 0. When undirected, the hub has
			degree (n-1) and all leaves have degree 1.

			Either way, the centrality distribution is maximally extreme —
			one node with all the mass, all others equal. This produces
			a low achievable-rho ceiling: even at b=1, the probability
			vector concentrates entirely on the hub, and the realized
			cor(is_dropped, c) is fundamentally limited by the
			(1, 0, 0, ..., 0) shape of the centrality vector. The ceiling
			is well below 0.95 for n = 50, which is what Test 3 exploits.

			Nodes DataFrame has :id and :label columns to match the
			convention enforced by network_community_detection._graph_to_sparse_matrix.
			Both columns hold the same identifier strings ("n1", "n2", ...).
		"""
		n >= 3 || throw(ArgumentError("n must be at least 3"))

		src = String[]
		dst = String[]
		for leaf in 2:n
			push!(src, "n$leaf")
			push!(dst, "n1")
		end
		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		node_ids = ["n$i" for i in 1:n]
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = directed, name = "star_$n")

		return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Draw n_reps replicates for one (network, pi_node, pi_edge, rho) cell; return per-rep records
	function _draw_replicates(net::NamedTuple;
								target_pi_node::Real,
								target_rho::Real,
								target_pi_edge::Real = 0.0,
								n_reps::Int,
								master_seed::Integer = 1,
								weighted::Bool = get(net.metadata, :weighted, false),
								true_community::Union{Nothing,AbstractVector{<:Integer}} = nothing)
		"""
		Args:
			net::NamedTuple: (edges, nodes, metadata) in corpus format
			target_pi_node::Real: nominal missing-node fraction for this cell
			target_rho::Real: nominal Kendall tau-b for this cell
			target_pi_edge::Real: nominal weight-degradation budget (default 0.0;
				forced to 0 internally on unweighted input by the mask)
			n_reps::Int: number of replicates to draw
			master_seed::Integer: seeds the per-replicate seed derivation
			weighted::Bool: whether the edge-degradation stage runs; defaults to
				the network's :weighted metadata (false for the constructed
				star/ring fixtures, which omit that field)
			true_community::Union{Nothing,AbstractVector{<:Integer}}: optional
				ground-truth community labels. When supplied, the mask's prior-3
				survivor-profile check is active; when nothing (the default), the
				mask falls back to a single community and prior 3 is skipped —
				which ISOLATES the basic sampling tests (priors 1 and 2) from
				prior-3 behavior.
		Returns:
			Vector{NamedTuple}: each element is the record returned by
				generate_missingness_mask for that replicate.
		Notes:
			Per-network artifacts (centrality, binarized adjacency, and the
			weighted adjacency when weighted) are computed once and cached across
			the n_reps calls, mirroring the build_degeneration_corpus orchestrator.
			Per-replicate seeds are derived via hash((target_rho, target_pi_node,
			target_pi_edge, rep, master_seed)) so identical inputs reproduce
			identical records (the determinism contract).

			This is the unified-pipeline harness: it drives the two-dial mask
			(pi_node + pi_edge under one rho). It does NOT pass a mechanism — full
			removal vs nomination is composed per node inside the mask.
		"""
		directed = net.metadata.directed

		#	Cache per-network artifacts once
			centrality = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
				net.edges; nodes = net.nodes, directed = directed)
			adj_binary = _graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = false)[1]
			adj_weighted = weighted ?
				_graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = true)[1] : nothing

		#	Draw replicates
			records = Vector{NamedTuple}(undef, n_reps)
			for rep in 1:n_reps
				rep_seed = Int(hash((target_rho, target_pi_node, target_pi_edge, rep, master_seed)) % UInt32)
				records[rep] = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
					net.edges;
					nodes          = net.nodes,
					directed       = directed,
					weighted       = weighted,
					target_pi_node = target_pi_node,
					target_pi_edge = target_pi_edge,
					target_rho     = target_rho,
					seed           = rep_seed,
					centrality     = centrality,
					true_community = true_community,
					adj_binary     = adj_binary,
					adj_weighted   = adj_weighted)
			end
		return records
	end

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

#############
#   TESTS   #
#############

#	Phase 1 Sampler Validation Harness
#
#		Tests the network_degeneracy module's contract on Scotland (undirected)
#		and Moreno (directed) as the small-network default, with an opt-in
#		Marvel stress pass. Each test perturbs a network according to a target
#		(rate, rho) cell and verifies that the perturbation matched its
#		specification — i.e., that the sampler honored its contract for that
#		cell. Two contract families are checked:
#
#			(a) Sampling correctness: rate hit per replicate, realized rho
#			    converges to target rho across replicates, determinism in
#			    master seed, achievable-rho ceiling detection.
#			(b) Topological degeneracy: gc fraction, n_observed, n_edges_observed
#			    flags fire when and only when the recorded continuous quantities
#			    cross their thresholds.
#
#		Post-degradation measure comparison (closeness on perturbed vs original,
#		etc.) is NOT in this harness — those checks belong to the Phase 1 grid
#		evaluation rather than to sampler validation.
#
#	Conventions:
#		- Each sub-test is callable in isolation and returns a NamedTuple
#		  (passed::Bool, details::String) for the driver to aggregate.
#		- Replicate counts: 100 for tests that average realized values across
#		  seeds (Tests 1, 5, 6, 7); 10 or fewer for deterministic checks
#		  (Tests 2, 3, 4).
#		- The driver runs sub-tests in order, prints a one-line status per
#		  test, and returns the conjunction Bool.

#	Test Rho Convergence: analytic field + rejection gate reaches target rho
	function test_rho_convergence_scotland(networks::Dict;
											n_reps::Int = 100,
											target_pi_node::Real = 0.10,
											target_rho::Real = 0.10,
											tol_mean_rho::Real = 0.03,
											min_converged_frac::Real = 0.80)
		"""
		Args:
			networks::Dict: corpus dict; must contain "scotland_interlock_unweighted"
			n_reps::Int: number of replicates (default 100)
			target_pi_node::Real: target missing-node fraction (default 0.10)
			target_rho::Real: target Kendall tau-b (default 0.10 — interior to the
				rate-bounded ceiling 2p(1-p)=0.18 at pi_node=0.10, so the gate can
				accept)
			tol_mean_rho::Real: tolerance on |mean(realized_rho) - target_rho|
				across the CONVERGED replicates (default 0.03)
			min_converged_frac::Real: minimum fraction of replicates that must
				reach gate_status == :converged (default 0.80)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Replaces the old bisection-convergence test. There is no bisection in
			the unified pipeline: the rho-tilt is solved analytically by
			_solve_propensity_field, the missing set is drawn from that field by
			_topup_missing_nodes, and the three-prior end gate accepts a draw only
			when |realized_rho - target_rho| <= rho_tol (the mask's default
			rho_tol = 0.02), retrying up to max_retries on a miss.

			Two consequences reshape what this test asserts:

			(1) For any record with gate_status == :converged, realized_rho is
			    within the gate's rho_tol of target BY CONSTRUCTION — the gate
			    enforced it. So "mean realized_rho ≈ target" is a sanity check on
			    gate semantics, not the core assertion; it cannot fail unless the
			    gate is mislabeling status.

			(2) The core assertion is the CONVERGENCE RATE. The gate does rejection
			    sampling against a single-draw Kendall tau-b whose sampling SD at
			    pi_node=0.10, N=108 is roughly 0.065. With rho_tol = 0.02 the
			    per-attempt acceptance probability is well below 1, so a fraction of
			    cells exhaust max_retries and return :failed_other. We require
			    >= min_converged_frac converged rather than all-converged.

			CALIBRATION CAVEAT: min_converged_frac = 0.80 is an estimate pending the
			first real run. The analytic field's expected realized rho is only
			approximate for degradation (the solve assumes uniform retained; carving
			out the missing set violates that), so the draw cloud may be biased off
			target, lowering acceptance. If the observed converged fraction is far
			below 0.80, that is NOT a signal to loosen this test — it is the signal
			that the gate's fixed rho_tol is too tight relative to the single-draw
			noise floor, i.e. the module-side fix is to make rho_tol rate/N-aware.
			The details string reports the converged fraction so a low rate is
			diagnostic rather than a bare FAIL.

			Prior 3 is intentionally OFF here (no true_community passed), isolating
			the test to priors 1 (proportion) and 2 (correlation). Scotland
			(undirected, N=108, moderate degree skew) runs in seconds.
		"""
		println("─" ^ 70)
		println("Test 1: Rho convergence on Scotland (rho=$target_rho, pi_node=$target_pi_node)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_pi_node = target_pi_node,
									  target_rho = target_rho,
									  n_reps     = n_reps)

		#	Check 1a: convergence rate at/above the floor
			n_converged    = count(r -> r.gate_status == :converged, records)
			converged_frac = n_converged / n_reps
			rate_ok        = converged_frac >= min_converged_frac

		#	Check 1b: among converged replicates, mean realized rho ≈ target
		#	(gate-guaranteed within rho_tol; this is a semantics sanity check)
			conv_rhos = [r.realized_rho for r in records if r.gate_status == :converged]
			mean_rho  = isempty(conv_rhos) ? NaN : mean(conv_rhos)
			rho_delta = isempty(conv_rhos) ? Inf : abs(mean_rho - target_rho)
			rho_in_tol = rho_delta < tol_mean_rho

		#	Report
			println("  Replicates:           $n_reps")
			println("  Converged:            $n_converged / $n_reps  (frac $(round(converged_frac, digits=3)), floor $min_converged_frac)")
			println("  Convergence rate OK:  $(rate_ok ? "YES" : "NO")")
			println("  Mean realized rho:    $(round(mean_rho, digits=4))  (converged subset)")
			println("  Target rho:           $target_rho")
			println("  |Δ|:                  $(round(rho_delta, digits=4))  (tol $tol_mean_rho)")
			println("  Rho in tolerance:     $(rho_in_tol ? "YES" : "NO")")

			passed = rate_ok && rho_in_tol
			println("  Result:               $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "converged=$(round(converged_frac, digits=3)) mean_rho=$(round(mean_rho, digits=4)) Δ=$(round(rho_delta, digits=4))")
	end

#	Test Rate-Bounded Ceiling: unreachable rho is rejected, realized respects sqrt(2p(1-p))
	function test_rate_bounded_ceiling_scotland(networks::Dict;
												n_reps::Int = 20,
												target_pi_node::Real = 0.10,
												target_rho::Real = 0.50)
		"""
		Args:
			networks::Dict: corpus dict; must contain "scotland_interlock_unweighted"
			n_reps::Int: replicates (default 20)
			target_pi_node::Real: missing-node fraction (default 0.10)
			target_rho::Real: target tau (default 0.50 — ABOVE the theoretical
				absolute ceiling sqrt(2*0.1*0.9) = 0.424, so unreachable on any
				network at this rate)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			The unified pipeline has no bisection probe; an unreachable rho is
			rejected by the end gate (it never reports :converged) and, when the
			propensity-field solve self-detects infeasibility, surfaced as
			:ceiling_hit (accept-and-flag, no retry). Because whether the
			reconstruction-side solve flags :ceiling_hit vs. silently best-efforts
			is its own concern, this test asserts the robust contract: NO replicate
			is :converged, and realized_rho never exceeds the theoretical absolute
			ceiling sqrt(2p(1-p)). It reports the :ceiling_hit / :failed_other split
			for diagnosis. The designed up-front clamp is tested at the orchestrator
			level (rho_was_substituted), not here.

			target_rho = 0.50 is chosen above the 0.424 theoretical max so the
			target is unreachable regardless of the field's practical ceiling.
		"""
		println("─" ^ 70)
		println("Test 1b: Rate-bounded ceiling on Scotland (rho=$target_rho at pi_node=$target_pi_node)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_pi_node = target_pi_node,
									  target_rho = target_rho,
									  n_reps     = n_reps)

		theoretical_ceiling = sqrt(2.0 * target_pi_node * (1.0 - target_pi_node))

		n_converged   = count(r -> r.gate_status == :converged,   records)
		n_ceiling     = count(r -> r.gate_status == :ceiling_hit,  records)
		n_failed      = count(r -> r.gate_status == :failed_other, records)
		none_converged = n_converged == 0

		realized_rhos = [r.realized_rho for r in records]
		max_realized  = maximum(abs.(realized_rhos))
		tolerance     = 0.05
		respects_ceiling = max_realized <= theoretical_ceiling + tolerance

		println("  Replicates:                $n_reps")
		println("  Target rho:                $target_rho")
		println("  Theoretical abs ceiling:   $(round(theoretical_ceiling, digits=4))  (sqrt(2p(1-p)))")
		println("  :converged / :ceiling / :failed:  $n_converged / $n_ceiling / $n_failed")
		println("  None converged:            $(none_converged ? "YES" : "NO")")
		println("  Max |realized rho|:        $(round(max_realized, digits=4))")
		println("  Respects abs ceiling:      $(respects_ceiling ? "YES" : "NO")")

		passed = none_converged && respects_ceiling
		println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "conv=$n_converged ceil=$n_ceiling fail=$n_failed max=$(round(max_realized, digits=4))")
	end

#	Test Exact Proportion: every replicate yields exactly round(pi_node*N) missing nodes
	function test_exact_proportion_scotland_moreno(networks::Dict;
													n_reps::Int = 10)
		"""
		Args:
			networks::Dict: corpus dict; needs Scotland and Moreno
			n_reps::Int: replicates per cell (default 10)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Prior-1 contract: with pi_edge = 0 the node-accounting stage fills the
			missing set to exactly k = round(pi_node*N) regardless of gate status
			(topup is pre-gate). Asserts |missing_nodes| == k per replicate across a
			small (rho, pi_node) sub-grid on both Scotland (undirected) and Moreno
			(directed). Independent of gate_status by design.
		"""
		println("─" ^ 70)
		println("Test 2: Exact proportion per replicate (Scotland + Moreno)")
		println("─" ^ 70)

		test_grid = [
			("scotland_interlock_unweighted",  108, [-0.25, 0.0, 0.25], [0.05, 0.10, 0.25, 0.50]),
			("moreno_highschool_unweighted",    70, [-0.25, 0.0, 0.25], [0.05, 0.10, 0.25, 0.50]),
		]

		all_passed = true
		mismatches = 0
		total      = 0

		for (name, n_nodes, rhos, pis) in test_grid
			haskey(networks, name) || (all_passed = false; continue)
			net = networks[name]
			for rho in rhos, pin in pis
				expected_k = Int(round(pin * n_nodes))
				records    = _draw_replicates(net; target_rho = rho,
													target_pi_node = pin,
													n_reps = n_reps)
				for r in records
					total += 1
					if length(r.missing_nodes) != expected_k
						mismatches += 1
						all_passed = false
					end
				end
			end
		end

		println("  Replicates checked:   $total")
		println("  Proportion mismatches:$mismatches")
		println("  Result:               $(all_passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = all_passed,
				details = "$mismatches/$total mismatches")
	end

#	Test Achievable-rho Ceiling on a star: extreme target rejected on extreme centrality
	function test_achievable_rho_ceiling_star(; n_reps::Int = 20,
												target_rho::Real = 0.95,
												target_pi_node::Real = 0.10,
												n_star::Int = 50)
		"""
		Args:
			n_reps::Int: replicates (default 20)
			target_rho::Real: extreme positive target (default 0.95, unreachable)
			target_pi_node::Real: missing-node fraction (default 0.10)
			n_star::Int: star node count (default 50)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Same rejection contract as Test 1b on the extreme star fixture (one
			hub, n-1 leaves). 0.95 exceeds both the theoretical sqrt(2p(1-p)) max
			and any practical field ceiling. Asserts NO replicate :converged and
			realized_rho respects the theoretical ceiling; reports the status split.
		"""
		println("─" ^ 70)
		println("Test 3: Achievable-rho ceiling on star (rho=$target_rho, n=$n_star)")
		println("─" ^ 70)

		net     = _build_star_fixture(n = n_star, directed = true)
		records = _draw_replicates(net; target_rho = target_rho,
									  target_pi_node = target_pi_node,
									  n_reps = n_reps)

		theoretical_ceiling = sqrt(2.0 * target_pi_node * (1.0 - target_pi_node))
		n_converged = count(r -> r.gate_status == :converged,   records)
		n_ceiling   = count(r -> r.gate_status == :ceiling_hit,  records)
		n_failed    = count(r -> r.gate_status == :failed_other, records)
		none_converged = n_converged == 0
		max_realized   = maximum(abs.(r.realized_rho for r in records))
		respects_ceiling = max_realized <= theoretical_ceiling + 0.05

		println("  Replicates:             $n_reps")
		println("  :converged / :ceiling / :failed:  $n_converged / $n_ceiling / $n_failed")
		println("  None converged:         $(none_converged ? "YES" : "NO")")
		println("  Max |realized rho|:     $(round(max_realized, digits=4))  (ceiling $(round(theoretical_ceiling, digits=4)))")

		passed = none_converged && respects_ceiling
		println("  Result:                 $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "conv=$n_converged ceil=$n_ceiling fail=$n_failed max=$(round(max_realized, digits=4))")
	end

#	Test Seed Reproducibility: identical inputs produce identical records
	function test_seed_reproducibility(networks::Dict;
										n_reps::Int = 10,
										target_rho::Real = 0.25,
										target_pi_node::Real = 0.10)
		"""
		Args:
			networks::Dict: corpus dict; uses Moreno (directed)
			n_reps::Int: replicate pairs to verify (default 10)
			target_rho::Real: target rho (default 0.25)
			target_pi_node::Real: missing-node fraction (default 0.10)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Determinism contract: identical (edges, nodes, directed, weighted,
			target_pi_node, target_pi_edge, target_rho, seed) reproduce identical
			records, including the retry loop's deterministic sub-seed sequence.
			Compares missing_nodes, realized_rho, gate_status, and retry_count.
			realized_rho is never NaN in the record (the gate coerces NaN to 0.0),
			so direct equality is safe.
		"""
		println("─" ^ 70)
		println("Test 4: Seed reproducibility on Moreno")
		println("─" ^ 70)

		haskey(networks, "moreno_highschool_unweighted") ||
			return (passed = false, details = "Moreno missing from corpus")

		net = networks["moreno_highschool_unweighted"]
		ndg = Network_Credible_Intervals.network_degeneracy
		c   = ndg._centrality_for_sampler(net.edges; nodes = net.nodes,
											directed = net.metadata.directed)
		adjb = _graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = false)[1]

		mismatches = 0
		for rep in 1:n_reps
			rep_seed = Int(hash((target_rho, target_pi_node, 0.0, rep, 1)) % UInt32)
			args = (net.edges,)
			kw = (nodes = net.nodes, directed = net.metadata.directed, weighted = false,
				  target_pi_node = target_pi_node, target_pi_edge = 0.0,
				  target_rho = target_rho, seed = rep_seed,
				  centrality = c, adj_binary = adjb)
			rec_a = ndg.generate_missingness_mask(args...; kw...)
			rec_b = ndg.generate_missingness_mask(args...; kw...)
			if rec_a.missing_nodes != rec_b.missing_nodes ||
			   rec_a.realized_rho  != rec_b.realized_rho  ||
			   rec_a.gate_status   != rec_b.gate_status   ||
			   rec_a.retry_count   != rec_b.retry_count
				mismatches += 1
			end
		end

		passed = mismatches == 0
		println("  Replicate pairs checked:    $n_reps")
		println("  Reproducibility mismatches: $mismatches")
		println("  Result:                     $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed, details = "$mismatches/$n_reps non-reproducible")
	end

#	Test MCAR Baseline: rho=0 yields a flat field (beta≈0) and unbiased realized rho
	function test_mcar_baseline(networks::Dict;
									n_reps::Int = 100,
									target_pi_node::Real = 0.10,
									tol_mean_rho::Real = 0.04,
									tol_beta::Real = 1e-6)
		"""
		Args:
			networks::Dict: corpus dict; uses Scotland
			n_reps::Int: replicates (default 100)
			target_pi_node::Real: missing-node fraction (default 0.10)
			tol_mean_rho::Real: tolerance on |mean(realized_rho)| over ALL records
			tol_beta::Real: tolerance on |beta| (default 1e-6)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			The Bellutta MCAR corner. At target_rho = 0 the propensity-field solve
			returns beta = 0 (flat field, uniform selection). This test asserts the
			MCAR property DIRECTLY and independently of the gate:
			  (a) beta ≈ 0 for every replicate (the field is flat), and
			  (b) mean realized_rho over ALL records ≈ 0 (unbiased selection).
			Mean is taken over all records, not just converged ones: under the
			strict gate many individual rho=0 draws miss the 0.02 band and end
			:failed_other, but their realized_rho is symmetric about 0, so the
			grand mean is the right unbiasedness statistic and does not depend on
			gate acceptance. Convergence rate is reported for diagnosis only.
		"""
		println("─" ^ 70)
		println("Test 5: MCAR baseline on Scotland (rho=0, pi_node=$target_pi_node)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_rho = 0.0,
									  target_pi_node = target_pi_node,
									  n_reps      = n_reps)

		max_abs_beta = maximum(abs(r.beta) for r in records)
		beta_flat    = max_abs_beta <= tol_beta

		realized_rhos = [r.realized_rho for r in records]
		mean_rho      = mean(realized_rhos)
		mean_in_tol   = abs(mean_rho) < tol_mean_rho

		n_converged = count(r -> r.gate_status == :converged, records)

		println("  Replicates:           $n_reps")
		println("  Max |beta|:           $(max_abs_beta)  (flat if ≤ $tol_beta)")
		println("  Field flat (beta≈0):  $(beta_flat ? "YES" : "NO")")
		println("  Mean realized rho:    $(round(mean_rho, digits=4))  (all records)")
		println("  |Mean| < tol:         $(mean_in_tol ? "YES" : "NO") (tol $tol_mean_rho)")
		println("  (diagnostic) converged: $n_converged / $n_reps")

		passed = beta_flat && mean_in_tol
		println("  Result:               $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed,
				details = "beta_max=$(round(max_abs_beta, digits=8)) mean_rho=$(round(mean_rho, digits=4))")
	end

#	Test Constant Centrality: non-zero rho unreachable; uniform draw, rho coerced to 0
	function test_ties_handled_regular_ring(; n_reps::Int = 10,
												target_rho::Real = 0.25,
												target_pi_node::Real = 0.10,
												n_ring::Int = 50,
												k_ring::Int = 4)
		"""
		Args:
			n_reps::Int: replicates (default 10)
			target_rho::Real: non-zero target (default 0.25)
			target_pi_node::Real: missing-node fraction (default 0.10)
			n_ring, k_ring::Int: regular-ring parameters (default 50, 4)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			On a k-regular ring every node has identical centrality, so no field can
			create a non-zero rho correlation. Behavior under the unified pipeline
			(NOT the old NaN short-circuit): the field solve still returns a beta
			(field_status :converged), the node-accounting stage still fills the
			missing set, but corkendall(indicator, constant) is undefined and the
			gate coerces realized_rho to 0.0; prior 2 then fails for any non-zero
			target, retries exhaust, and gate_status is :failed_other.

			Asserts the new signature of this case:
			  - gate_status == :failed_other for all replicates
			  - realized_rho == 0.0 for all (NaN-coerced, the constant-centrality tell)
			  - missing_nodes NON-empty, size == round(pi_node*N) (selection still ran)
			This inverts the old test's "empty dropped set / NaN rho" expectations.
		"""
		println("─" ^ 70)
		println("Test 6: Constant centrality on $(n_ring)x$(k_ring) ring (rho=$target_rho)")
		println("─" ^ 70)

		net     = _build_regular_ring_fixture(n = n_ring, k = k_ring)
		records = _draw_replicates(net; target_rho = target_rho,
									  target_pi_node = target_pi_node,
									  n_reps = n_reps)

		expected_k = Int(round(target_pi_node * n_ring))
		all_failed   = all(r.gate_status == :failed_other for r in records)
		all_rho_zero = all(r.realized_rho == 0.0 for r in records)
		all_filled   = all(length(r.missing_nodes) == expected_k for r in records)

		println("  Replicates:                $n_reps")
		println("  All :failed_other:         $(all_failed ? "YES" : "NO")")
		println("  All realized_rho == 0:     $(all_rho_zero ? "YES" : "NO")")
		println("  All missing filled to k:   $(all_filled ? "YES ($expected_k)" : "NO")")

		passed = all_failed && all_rho_zero && all_filled
		println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed,
				details = "failed=$all_failed rho0=$all_rho_zero filled=$all_filled")
	end

#	Test Degeneracy Flagging: adversarial pi_node/rho fire the topological flags
	function test_degeneracy_flagging_fires(networks::Dict;
											n_reps::Int = 100,
											target_rho::Real = 0.75,
											target_pi_node::Real = 0.65,
											fire_rate_threshold::Real = 0.50,
											fixture_name::String = "moreno_highschool_unweighted")
		"""
		Args:
			networks::Dict: corpus dict; uses Moreno by default
			n_reps::Int: replicates (default 100)
			target_rho::Real: adversarial rho (default 0.75)
			target_pi_node::Real: adversarial missing fraction (default 0.65)
			fire_rate_threshold::Real: min fraction with any_topo_degenerate (0.50)
			fixture_name::String: network key
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			At adversarial settings the topological flags should fire on a
			meaningful fraction of replicates. Degeneracy is assessed on EVERY
			record, NOT a gate-filtered subset: even :failed_other records carry a
			full-size missing set (topup fills to round(pi_node*N)) and a computed
			sampler_degeneracy, and the collapse contract depends only on the
			missing set's effect on connectivity, not on whether the rho gate
			passed. (rho=0.75 is unreachable at pi_node=0.65, so most/all records
			are :failed_other — filtering them out is what made an earlier version
			report "cannot assess.") At Moreno N=70, pi_node=0.65 leaves ~24 nodes,
			below min_n=25, so too_small fires on essentially every record.
		"""
		println("─" ^ 70)
		println("Test 7: Degeneracy flagging on $fixture_name (rho=$target_rho, pi_node=$target_pi_node)")
		println("─" ^ 70)

		haskey(networks, fixture_name) ||
			return (passed = false, details = "$fixture_name missing from corpus")

		net     = networks[fixture_name]
		records = _draw_replicates(net; target_rho = target_rho,
									  target_pi_node = target_pi_node,
									  n_reps      = n_reps)

		#	Assess degeneracy on EVERY record (see Notes): missing set and
		#	sampler_degeneracy exist regardless of gate_status.
			n_tot = length(records)
			n_any = count(r -> r.sampler_degeneracy.any_topo_degenerate, records)
			n_gc  = count(r -> r.sampler_degeneracy.gc_collapse,          records)
			n_tsm = count(r -> r.sampler_degeneracy.too_small,            records)
			n_ned = count(r -> r.sampler_degeneracy.no_edges,             records)
			any_rate = n_any / n_tot
			passed   = any_rate >= fire_rate_threshold

		println("  Records:                    $n_tot / $n_reps")
		println("  any_topo_degenerate rate:   $(round(any_rate, digits=3))  (threshold $fire_rate_threshold)")
		println("  gc_collapse rate:           $(round(n_gc  / n_tot, digits=3))")
		println("  too_small rate:             $(round(n_tsm / n_tot, digits=3))")
		println("  no_edges rate:              $(round(n_ned / n_tot, digits=3))")
		println("  Result:                     $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed, details = "any_topo=$(round(any_rate, digits=3))")
	end

#	Test Materialize Composition: per-node nomination vs full-removal, auto-decided
	function test_materialize_composition(; )
		"""
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Tests the unified materializer's per-node composition on a tiny
			hand-checkable directed fixture A->B, B->C, A->C, C->D (nodes A,B,C,D in
			canonical order 1..4). Three cases:
			  (1) directed, missing {C}: C has surviving incoming (A->C, B->C) =>
			      NOMINATION. n_nominations=1, n_full_removal=0, C kept in roster,
			      C->D gone, A->C and B->C retained.
			  (2) directed, missing {A}: A is a pure source (no incoming) =>
			      FULL REMOVAL. n_full_removal=1, n_nominations=0, A dropped, no
			      edge with A survives.
			  (3) undirected, missing {C}: no in/out asymmetry => FULL REMOVAL.
			      n_full_removal=1, n_nominations=0, C dropped, no edge touching C.
			The two standalone materializers (apply_missingness,
			apply_missingness_outgoing_only) moved to network_reconstruction; their
			contracts are tested there, not here.
		"""
		println("─" ^ 70)
		println("Test 8: _materialize_missing_nodes composition (nomination vs full removal)")
		println("─" ^ 70)

		ndg   = Network_Credible_Intervals.network_degeneracy
		nodes = DataFrame(id = ["A", "B", "C", "D"])
		edges = DataFrame(src = ["A", "B", "A", "C"],
						  dst = ["B", "C", "C", "D"],
						  weight = ones(Int, 4))

		#	Case 1: directed, missing {C} (index 3) -> nomination
			m1 = ndg._materialize_missing_nodes(edges, [3]; nodes = nodes, directed = true)
			c1_counts = m1.n_nominations == 1 && m1.n_full_removal == 0
			c1_roster = "C" in string.(m1.nodes[!, 1])
			e1 = Set((string(m1.edges.src[r]), string(m1.edges.dst[r])) for r in 1:nrow(m1.edges))
			c1_edges = !(("C", "D") in e1) && (("A", "C") in e1) && (("B", "C") in e1)
			check_1 = c1_counts && c1_roster && c1_edges

		#	Case 2: directed, missing {A} (index 1) -> full removal
			m2 = ndg._materialize_missing_nodes(edges, [1]; nodes = nodes, directed = true)
			c2_counts = m2.n_full_removal == 1 && m2.n_nominations == 0
			c2_roster = !("A" in string.(m2.nodes[!, 1]))
			c2_edges  = !any(string(m2.edges.src[r]) == "A" || string(m2.edges.dst[r]) == "A"
							 for r in 1:nrow(m2.edges))
			check_2 = c2_counts && c2_roster && c2_edges

		#	Case 3: undirected, missing {C} -> full removal
			m3 = ndg._materialize_missing_nodes(edges, [3]; nodes = nodes, directed = false)
			c3_counts = m3.n_full_removal == 1 && m3.n_nominations == 0
			c3_roster = !("C" in string.(m3.nodes[!, 1]))
			c3_edges  = !any(string(m3.edges.src[r]) == "C" || string(m3.edges.dst[r]) == "C"
							 for r in 1:nrow(m3.edges))
			check_3 = c3_counts && c3_roster && c3_edges

		println("  (1) directed nomination:   $(check_1 ? "YES" : "NO")")
		println("  (2) directed full removal: $(check_2 ? "YES" : "NO")")
		println("  (3) undirected full removal:$(check_3 ? "YES" : "NO")")

		passed = check_1 && check_2 && check_3
		println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed, details = "c1=$check_1 c2=$check_2 c3=$check_3")
	end

#	Test Weight Stage: budget hit, edges preserved at zero, organic losses peripheral
	function test_weight_stage_contract(; n_seeds::Int = 40)
		"""
		Args:
			n_seeds::Int: seeds for the organic-loss frequency check (default 40)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Tests _sample_weight_removal + apply_weight_removal directly on a small
			weighted path A-B-C-D-E with weights 10,10,10,1 (E peripheral, total 31)
			under a FLAT field (d = ones => Bellutta MCAR corner). Checks:
			  (a) Budget: W_true == 31, W_removed == round(pi_edge*31) (clamped).
			  (b) Edge preservation: apply_weight_removal keeps every edge ROW
			      (row count invariant; depleted edges retained at weight 0).
			  (c) Weight accounting: sum(orig) - sum(degraded) == W_removed.
			  (d) Organic-loss periphery: across n_seeds, the low-weight node E is
			      the most frequent organic loss (tie-first losses concentrate on
			      the periphery even at rho=0). The frequency threshold is a
			      first-run-calibrated heuristic; E should dominate clearly.
		"""
		println("─" ^ 70)
		println("Test (weighted): weight-removal stage contract")
		println("─" ^ 70)

		ndg   = Network_Credible_Intervals.network_degeneracy
		node_ids = ["A", "B", "C", "D", "E"]
		nodes = DataFrame(id = node_ids, label = node_ids)
		edges = DataFrame(src = ["A", "B", "C", "D"],
						  dst = ["B", "C", "D", "E"],
						  weight = [10, 10, 10, 1])
		adj = _graph_to_sparse_matrix(edges; nodes = nodes, weighted = true)[1]
		n   = size(adj, 1)
		d_flat = ones(Float64, n)

		#	(a)+(b)+(c) at pi_edge = 0.5
			pi_edge = 0.5
			wr = ndg._sample_weight_removal(adj, d_flat, pi_edge, 12345)
			expected_budget = min(Int(round(pi_edge * 31)), 31)
			check_budget = wr.W_true == 31 && wr.W_removed == expected_budget

			degraded = ndg.apply_weight_removal(edges, wr.removed; nodes = nodes)
			check_rows = nrow(degraded) == nrow(edges)
			check_floor = all(degraded.weight .>= 0)
			check_acct  = (sum(edges.weight) - sum(degraded.weight)) == wr.W_removed
			check_b = check_rows && check_floor && check_acct

		#	(d) organic-loss periphery across seeds (E = index 5)
			freq = zeros(Int, n)
			for s in 1:n_seeds
				w = ndg._sample_weight_removal(adj, d_flat, pi_edge, 1000 + s)
				for v in w.organic_losses
					freq[v] += 1
				end
			end
			e_idx = findfirst(==("E"), string.(nodes[!, 1]))
			e_is_top = freq[e_idx] == maximum(freq) && freq[e_idx] > 0
			check_d = e_is_top

		println("  W_true / W_removed:        $(wr.W_true) / $(wr.W_removed)  (expected budget $expected_budget)")
		println("  (a) budget hit:            $(check_budget ? "YES" : "NO")")
		println("  (b) edges preserved:       $(check_b ? "YES" : "NO")  (rows $(nrow(degraded))/$(nrow(edges)))")
		println("  (d) E most-frequent organic:$(check_d ? "YES" : "NO")  (freq E=$(freq[e_idx]) of $n_seeds)")

		passed = check_budget && check_b && check_d
		println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed,
				details = "budget=$check_budget preserve=$check_b periphery=$check_d")
	end

#	Test Prior 3: survivor E/I-by-rank profile distortion (benign low, broker-stripping high)
	function test_prior3_survivor_profile(; ei_tvd_tol::Real = 0.25)
		"""
		Args:
			ei_tvd_tol::Real: the gate's prior-3 tolerance (default 0.25)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Tests the redesigned prior 3 directly via _three_prior_gate on a
			two-community fixture (two 8-node cliques, comm 1 and comm 2) bridged by
			two high-degree BROKER nodes whose ties are heavily cross-community.
			The true profile's high-degree-rank bin is broker-dominated (high
			external E/I). Two removals at the same proportion:
			  benign  = a few low-degree within-clique nodes -> survivors' E/I-by-rank
			            curve ~ true curve -> low distortion -> prior3_ok = true.
			  malign  = the brokers -> the high-rank bin is now clique-internal
			            instead of broker-external -> the curve bends -> high
			            distortion -> prior3_ok = false.
			Asserts ei_tvd(benign) < ei_tvd(malign), prior3_ok(benign) == true, and
			prior3_ok(malign) == false. Isolates prior 3 by reading prior3_ok / ei_tvd
			directly (priors 1 and 2 are not the discriminator here). The tolerance
			and the fixture's bridging strength may need first-run tuning to make the
			malign case clearly exceed ei_tvd_tol.
		"""
		println("─" ^ 70)
		println("Test (prior 3): survivor-profile distortion, benign vs broker-stripping")
		println("─" ^ 70)

		ndg = Network_Credible_Intervals.network_degeneracy

		#	Build two cliques + 2 brokers
			src = String[]; dst = String[]
			c1 = ["a$i" for i in 1:8]
			c2 = ["b$i" for i in 1:8]
			brokers = ["k1", "k2"]
			for grp in (c1, c2)            # internal clique edges
				for i in 1:length(grp), j in (i+1):length(grp)
					push!(src, grp[i]); push!(dst, grp[j])
				end
			end
			for k in brokers              # brokers bridge both communities
				for v in vcat(c1, c2)
					push!(src, k); push!(dst, v)
				end
			end
			node_ids = vcat(c1, c2, brokers)
			nodes = DataFrame(id = node_ids, label = node_ids)
			edges = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))

			adj = _graph_to_sparse_matrix(edges; nodes = nodes, weighted = false)[1]
			n   = size(adj, 1)
			#	true community labels: brokers nominally in community 1
				comm = vcat(fill(1, 8), fill(2, 8), fill(1, 2))
			centrality = ndg._centrality_for_sampler(edges; nodes = nodes, directed = false)

			n_rank_bins = 4; n_ei_bins = 3; min_count = 2
			true_profile, true_occ = ndg._ei_rank_profile(adj, comm, trues(n),
														   n_rank_bins, n_ei_bins)

		#	index helpers
			idx_of(name) = findfirst(==(name), node_ids)
			benign_names = ["a1", "a2", "b1", "b2"]              # low-degree clique members
			malign_names = ["k1", "k2", "a1", "a2"]             # brokers + filler, same count
			benign = sort([idx_of(x) for x in benign_names])
			malign = sort([idx_of(x) for x in malign_names])
			pin = length(benign) / n

		#	Gate each removal; isolate prior 3 by reading prior3_ok / ei_tvd.
		#	target_rho set to the realized rho of each set so prior 2 is not the
		#	discriminator (we only inspect prior3_ok / ei_tvd here).
			gate_b = ndg._three_prior_gate(benign, centrality, comm, adj,
											true_profile, true_occ, pin, 0.0;
											ei_tvd_tol = ei_tvd_tol,
											n_rank_bins = n_rank_bins, n_ei_bins = n_ei_bins,
											min_count = min_count)
			gate_m = ndg._three_prior_gate(malign, centrality, comm, adj,
											true_profile, true_occ, pin, 0.0;
											ei_tvd_tol = ei_tvd_tol,
											n_rank_bins = n_rank_bins, n_ei_bins = n_ei_bins,
											min_count = min_count)

			ord_ok    = !isnan(gate_b.ei_tvd) && !isnan(gate_m.ei_tvd) &&
						gate_b.ei_tvd < gate_m.ei_tvd
			benign_ok = gate_b.prior3_ok == true
			malign_ok = gate_m.prior3_ok == false

		println("  benign ei_tvd:    $(round(gate_b.ei_tvd, digits=4))  prior3_ok=$(gate_b.prior3_ok)")
		println("  malign ei_tvd:    $(round(gate_m.ei_tvd, digits=4))  prior3_ok=$(gate_m.prior3_ok)")
		println("  benign < malign:  $(ord_ok ? "YES" : "NO")")

		passed = ord_ok && benign_ok && malign_ok
		println("  Result:           $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed,
				details = "benign=$(round(gate_b.ei_tvd, digits=3)) malign=$(round(gate_m.ei_tvd, digits=3))")
	end

#	Run the Phase 1 sampler-validation harness
	function run_phase1_sampler_tests(networks::Dict; verbose::Bool = true)
		"""
		Args:
			networks::Dict: corpus dict with at least Scotland and Moreno
			verbose::Bool: print per-test status (default true)
		Returns:
			NamedTuple (passed::Bool, n_passed::Int, n_total::Int, results::Vector)
		Notes:
			Runs the unified-pipeline sampler-contract tests on the small-network
			fixtures plus the constructed star, regular ring, materializer,
			weighted, and prior-3 fixtures. No mechanism axis; the weighted and
			prior-3 tests are new with Spec v3.
		"""
		verbose && println("\n" * "═" ^ 70)
		verbose && println("Phase 1 Sampler Validation Harness — unified pipeline")
		verbose && println("═" ^ 70)

		results = NamedTuple[]
		push!(results, test_rho_convergence_scotland(networks))
		push!(results, test_rate_bounded_ceiling_scotland(networks))
		push!(results, test_exact_proportion_scotland_moreno(networks))
		push!(results, test_achievable_rho_ceiling_star())
		push!(results, test_seed_reproducibility(networks))
		push!(results, test_mcar_baseline(networks))
		push!(results, test_ties_handled_regular_ring())
		push!(results, test_degeneracy_flagging_fires(networks))
		push!(results, test_materialize_composition())
		push!(results, test_weight_stage_contract())
		push!(results, test_prior3_survivor_profile())

		n_passed = count(r -> r.passed, results)
		n_total  = length(results)
		all_ok   = n_passed == n_total

		verbose && println("\n" * "═" ^ 70)
		verbose && println("Result: $n_passed / $n_total tests passed " *
							(all_ok ? "— ALL PASS ✓" : "— FAILURES ✗"))
		verbose && println("═" ^ 70 * "\n")

		return (passed = all_ok, n_passed = n_passed, n_total = n_total, results = results)
	end
	run_phase1_sampler_tests(networks)

#	Smoke Test: orchestrator two-dial grid (no mechanism axis)
	function smoke_test_orchestrator_two_dial(networks::Dict;
												fixture_directed::String   = "moreno_highschool_unweighted",
												fixture_undirected::String = "scotland_interlock_unweighted")
		"""
		Args:
			networks::Dict: corpus dict; needs Moreno (directed) and Scotland (undirected)
			fixture_directed, fixture_undirected::String: fixture keys
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Sanity-checks build_degeneration_corpus on a minimal two-dial grid.
			Verifies, against the Spec v3 schema:
			  (a) Shape: rows == sum over nets of |rhos|*|pi_nodes|*|pi_edges_net|*reps,
			      with pi_edges forced to {0.0} on these unweighted fixtures (no
			      mechanism doubling).
			  (b) No mechanism column; n_full_removal and n_nominations present.
			  (c) Undirected rows have n_nominations == 0 for every row (full removal
			      only); directed rows may have nominations (reported).
			  (d) Feasibility: an over-ceiling rho in the grid yields
			      rho_was_substituted == true on some rows.
			  (e) Determinism: two serial runs at the same master_seed are identical
			      on missing_nodes, realized_rho, gate_status.
			  (f) reverse_edges: a reverse run completes and is itself reproducible.
		"""
		println("─" ^ 70)
		println("Smoke Test: orchestrator two-dial grid")
		println("─" ^ 70)

		haskey(networks, fixture_directed) ||
			return (passed = false, details = "$fixture_directed missing")
		haskey(networks, fixture_undirected) ||
			return (passed = false, details = "$fixture_undirected missing")

		ndg = Network_Credible_Intervals.network_degeneracy
		small = Dict(fixture_directed   => networks[fixture_directed],
					 fixture_undirected => networks[fixture_undirected])

		rhos = [0.0, 0.25, 0.9]; pins = [0.10, 0.25]; reps = 3
		runargs = (; target_rhos = rhos, target_pi_nodes = pins,
					 target_pi_edges = [0.0], n_replicates = reps,
					 master_seed = 7, parallel = false, show_progress = false)

		result = ndg.build_degeneration_corpus(small; runargs...)

		#	(a) shape — both unweighted => pi_edges == {0.0}
			expected = 2 * length(rhos) * length(pins) * 1 * reps
			check_a  = nrow(result) == expected

		#	(b) no mechanism column; composition columns present
			cols = names(result)
			check_b = !("mechanism" in cols) &&
					   ("n_full_removal" in cols) && ("n_nominations" in cols)

		#	(c) undirected => all nominations zero
			und_rows = filter(r -> r.network_name == fixture_undirected, result)
			dir_rows = filter(r -> r.network_name == fixture_directed,   result)
			check_c  = all(und_rows.n_nominations .== 0)
			dir_nom_total = sum(dir_rows.n_nominations)

		#	(d) feasibility substitution fired for the over-ceiling rho
			check_d = any(result.rho_was_substituted)

		#	(e) determinism across two serial runs
			result2 = ndg.build_degeneration_corpus(small; runargs...)
			check_e = result.missing_nodes == result2.missing_nodes &&
					   result.realized_rho == result2.realized_rho &&
					   result.gate_status  == result2.gate_status

		#	(f) reverse_edges completes and is reproducible
			rev1 = ndg.build_degeneration_corpus(small; runargs..., reverse_edges = true)
			rev2 = ndg.build_degeneration_corpus(small; runargs..., reverse_edges = true)
			check_f = nrow(rev1) == expected &&
					   rev1.missing_nodes == rev2.missing_nodes &&
					   rev1.gate_status  == rev2.gate_status

		println("  Rows:                     $(nrow(result)) (expected $expected)")
		println("  (a) shape:                $(check_a ? "YES" : "NO")")
		println("  (b) no mechanism col:     $(check_b ? "YES" : "NO")")
		println("  (c) undirected nom == 0:  $(check_c ? "YES" : "NO")  (directed nom total $dir_nom_total)")
		println("  (d) feasibility fired:    $(check_d ? "YES" : "NO")")
		println("  (e) determinism:          $(check_e ? "YES" : "NO")")
		println("  (f) reverse reproducible: $(check_f ? "YES" : "NO")")

		passed = check_a && check_b && check_c && check_d && check_e && check_f
		println("  Result:                   $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed = passed,
				details = "a=$check_a b=$check_b c=$check_c d=$check_d e=$check_e f=$check_f")
	end
    smoke_test_orchestrator_two_dial(networks)

############################
#   Pre-Launch Smoke Run   #
############################

#	Three networks, full design grid, 10 replicates per cell, both
#	mechanisms. Exercises the threaded orchestrator, the Arrow writer,
#	and the per-network diagnostic readout before launching the
#	production run.

#	Set Parameters (plain variables, not const, so reruns don't error
#	with "cannot redefine constant" warnings)
	master_seed       = 42
	output_dir        = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degenerate_Networks"
	smoke_output_file = joinpath(output_dir, "smoke_degeneration_corpus.arrow")

	target_rhos       = [-0.75, -0.25, 0.0, 0.25, 0.75]
	target_rates      = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50]
	n_replicates      = 10                                        #	reduced from 100 for smoke
	mechanisms_cfg    = [:full_removal, :outgoing_only]

	smoke_network_names = [
		"moreno_highschool_unweighted",
		"scotland_interlock_unweighted",
		"marvel_universe_unweighted",
	]

#	Pre-flight Checks
	println("─" ^ 70)
	println("Phase 1 Pre-Launch Smoke Test")
	println("─" ^ 70)
	println("Started:           ", now())
	println("Julia version:     ", VERSION)
	println("Threads:           ", Threads.nthreads())
	println("Master seed:       ", master_seed)
	println("Output file:       ", smoke_output_file)
	println()

	mkpath(output_dir)
	Threads.nthreads() >= 4 || @warn "Only $(Threads.nthreads()) threads available"

#	Subset Networks
	#	Each value in the smoke dict must be the network NamedTuple
	#	(networks[name]), not the full networks dict.
	test_networks = Dict{String, NamedTuple}(
		name => networks[name]
		for name in smoke_network_names
		if haskey(networks, name)
	)

	length(test_networks) == length(smoke_network_names) ||
		error("Missing networks: ", setdiff(smoke_network_names, keys(test_networks)))

	println("Smoke corpus (", length(test_networks), " networks):")
	for (name, net) in sort(collect(test_networks), by=first)
		println("  $(rpad(name, 40)) N=$(rpad(nrow(net.nodes), 8)) E=$(rpad(nrow(net.edges), 10)) directed=$(net.metadata.directed)")
	end
	println()

#	Run Smoke Grid
	println("Launching build_degeneration_corpus (smoke configuration) ...")
	println()

	@time corpus_df = Network_Credible_Intervals.network_degeneracy.build_degeneration_corpus(
											test_networks;
											target_rhos    = target_rhos,
											target_rates   = target_rates,
											n_replicates   = n_replicates,
											mechanisms     = mechanisms_cfg,
											master_seed    = master_seed,
											parallel       = true,
											show_progress  = true)

#	Diagnostic Readout
	println()
	println("─" ^ 70)
	println("Smoke result")
	println("─" ^ 70)
	println("Rows:              ", nrow(corpus_df))
	println("Networks:          ", length(unique(corpus_df.network_name)))
	println()

	println("Per-network breakdown")
	println("─" ^ 70)
	for grp in groupby(corpus_df, :network_name)
		n_rows  = nrow(grp)
		n_full  = count(==(:full_removal), grp.mechanism)
		n_out   = count(==(:outgoing_only), grp.mechanism)
		n_conv  = count(==(:converged), grp.bisection_status)
		n_ceil  = count(==(:ceiling_hit), grp.bisection_status)
		n_fail  = count(==(:failed_other), grp.bisection_status)
		n_deg   = count(grp.any_topo_degenerate)
		println("  $(rpad(grp.network_name[1], 40)) rows=$(rpad(n_rows, 5)) full=$(rpad(n_full, 4)) out=$(rpad(n_out, 4)) conv=$(rpad(n_conv, 4)) ceil=$(rpad(n_ceil, 3)) fail=$(rpad(n_fail, 3)) deg=$n_deg")
	end
	println()

#	Sanity Checks
	expected_rows = 0
	for name in smoke_network_names
		net = test_networks[name]
		n_mech = net.metadata.directed ? 2 : 1
		expected_rows += length(target_rhos) * length(target_rates) * n_replicates * n_mech
	end
	nrow(corpus_df) == expected_rows ||
		@warn "Row count mismatch: got $(nrow(corpus_df)), expected $expected_rows"

#	Write Arrow
	println("Writing Arrow file ...")
	Arrow.write(smoke_output_file, corpus_df; compress = :zstd)
	println("Wrote: ", smoke_output_file, " (", round(filesize(smoke_output_file) / 1024^2, digits=1), " MB)")
	println()

#	Roundtrip Test
	#	Confirm the Arrow file reads back correctly with the expected
	#	column types — particularly the Vector{Vector{Int}} dropped_nodes
	#	column, which is the column most likely to misbehave.
	println("Roundtrip test ...")
	read_df = DataFrame(Arrow.Table(smoke_output_file))
	nrow(read_df) == nrow(corpus_df) || error("Roundtrip row count mismatch")
	typeof(read_df.dropped_nodes[1]) <: AbstractVector{<:Integer} ||
		@warn "dropped_nodes roundtrip type unexpected: $(typeof(read_df.dropped_nodes[1]))"
	read_df.realized_rho[1] ≈ corpus_df.realized_rho[1] || error("realized_rho roundtrip mismatch")
	println("Roundtrip OK: read back ", nrow(read_df), " rows, dropped_nodes type intact")
	println()
	println("Smoke complete: ", now())
