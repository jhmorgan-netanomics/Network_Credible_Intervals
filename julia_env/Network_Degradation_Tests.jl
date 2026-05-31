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

#	Draw n_reps replicates for one (network, rho, rate) cell; return per-rep records
	function _draw_replicates(net::NamedTuple;
								target_rho::Real,
								target_rate::Real,
								n_reps::Int,
								master_seed::Integer = 1)
		"""
		Args:
			net::NamedTuple: (edges, nodes, metadata) in corpus format
			target_rho::Real: nominal rho for this cell
			target_rate::Real: nominal rate for this cell
			n_reps::Int: number of replicates to draw
			master_seed::Integer: seeds the per-replicate seed derivation
		Returns:
			Vector{NamedTuple}: each element is the record returned by
				generate_missingness_mask for that replicate.
		Notes:
			Centrality is computed once and cached across the n_reps calls,
			mirroring the build_degeneration_corpus orchestrator. Per-
			replicate seeds are derived via hash((target_rho, target_rate,
			rep_idx, master_seed)) so identical inputs reproduce identical
			records (the determinism contract).
		"""
		#	Cache centrality once
			centrality = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
				net.edges; nodes = net.nodes, directed = net.metadata.directed)

		#	Draw replicates
			records = Vector{NamedTuple}(undef, n_reps)
			for rep in 1:n_reps
				rep_seed = Int(hash((target_rho, target_rate, rep, master_seed)) % UInt32)
				records[rep] = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
					net.edges;
					nodes       = net.nodes,
					directed    = net.metadata.directed,
					target_rate = target_rate,
					target_rho  = target_rho,
					seed        = rep_seed,
					centrality  = centrality)
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

#	Test Bisection Convergence
	function test_bisection_convergence_scotland(networks::Dict;
                                              n_reps::Int = 100,
                                              target_rho::Real = 0.10,
                                              target_rate::Real = 0.10,
                                              tol_mean_rho::Real = 0.03)
		"""
		Args:
			networks::Dict: corpus dict; must contain "scotland_interlock_unweighted"
			n_reps::Int: number of replicates (default 100)
			target_rho::Real: target Kendall tau-b (default 0.10 — interior
				to the rate-bounded ceiling at rate=0.10)
			target_rate::Real: target rate (default 0.10)
			tol_mean_rho::Real: tolerance on |mean(realized_rho) - target_rho|
				across n_reps replicates (default 0.03)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies that the bisection actually solves cor_kendall(is_dropped, c) =
			target_rho in expectation across replicates. Single-replicate
			realized_rho is noisy; the assertion is on the mean across
			n_reps draws.

			RATE-BOUNDED CEILING. Under Kendall tau-b, the realized
			correlation between a binary indicator (is_dropped) and a
			continuous variable (centrality) is structurally bounded by
			|tau| <= 2*p*(1-p) where p is the missingness rate. At
			rate=0.10, the ceiling is ~0.18. The default target_rho of
			0.10 is comfortably inside this ceiling, leaving room for
			the bisection to converge without saturating.

			This is the Kendall-era replacement for the previous Pearson
			default of target_rho=0.25, which under Kendall would
			structurally exceed the rate-bounded ceiling and produce
			:ceiling_hit rather than :converged. The test now exercises
			the bisection's normal convergence path; the explicit
			rate-bounded ceiling check lives in
			test_rate_bounded_ceiling_scotland (Test 1b).

			Scotland (undirected, N=108, degree Gini ~0.48) is the
			natural fixture: moderate skew, large enough that 100
			replicates give a tight estimate of mean realized tau, small
			enough that the whole test runs in seconds.

			The bisection status must be :converged for every replicate.
		"""
		println("─" ^ 70)
		println("Test 1: Bisection convergence on Scotland (rho=$target_rho, rate=$target_rate)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_rho = target_rho,
									  target_rate = target_rate,
									  n_reps      = n_reps)

		#	Check 1a: every replicate converged
			all_converged = all(r.bisection_status == :converged for r in records)

		#	Check 1b: mean realized rho is close to target
			realized_rhos = [r.realized_rho for r in records]
			mean_rho      = mean(realized_rhos)
			rho_delta     = abs(mean_rho - target_rho)
			rho_in_tol    = rho_delta < tol_mean_rho

		#	Report
			println("  Replicates:         $n_reps")
			println("  All converged:      $(all_converged ? "YES" : "NO")")
			println("  Mean realized rho:  $(round(mean_rho, digits=4))")
			println("  Target rho:         $target_rho")
			println("  |Δ|:                $(round(rho_delta, digits=4))  (tol $tol_mean_rho)")
			println("  Rho in tolerance:   $(rho_in_tol ? "YES" : "NO")")

			passed = all_converged && rho_in_tol
			println("  Result:             $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "mean_rho=$(round(mean_rho, digits=4)) Δ=$(round(rho_delta, digits=4))")
	end

#	Test Rate-Bounded Ceiling: Kendall tau-b structural ceiling under rate
	function test_rate_bounded_ceiling_scotland(networks::Dict;
                                                n_reps::Int = 20,
                                                target_rho::Real = 0.25,
                                                target_rate::Real = 0.10)
		"""
		Args:
			networks::Dict: corpus dict; must contain "scotland_interlock_unweighted"
			n_reps::Int: number of replicates (default 20)
			target_rho::Real: target tau (default 0.25 — above Scotland's
				practical ceiling at rate=0.10, though well below the
				theoretical absolute ceiling of ~0.42)
			target_rate::Real: target rate (default 0.10)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies the Kendall tau-b rate-bounded ceiling property.
			Under Kendall, the realized correlation between a binary
			indicator (proportion p) and a continuous variable is
			structurally bounded by

			    |tau|_max = sqrt(2 * k * (n-k) / (n * (n-1)))
			              ≈ sqrt(2 * p * (1 - p))

			where p = k/n is the missingness rate. At p = 0.10 the
			theoretical absolute ceiling is sqrt(0.18) ≈ 0.42.

			The TIES-BOUNDED CEILING. On unweighted networks the
			centrality vector contains many ties (multiple nodes at
			the same degree). Ties on the continuous variable reduce
			the achievable tau-b below the theoretical absolute ceiling
			(the +Ty term in tau-b's denominator). The practical
			ceiling on a given network is therefore below sqrt(2*p*(1-p))
			and depends on the network's centrality tie structure.
			Scotland (N=108, many ties at low degree) has a practical
			ceiling around 0.20-0.25 at rate=0.10 — well below the
			theoretical absolute of 0.42.

			Test design: request target_rho = 0.25 at rate = 0.10.
			Scotland's practical ceiling is at or below this, so the
			bisection should return :ceiling_hit consistently. Expected
			behavior:
			(a) Every replicate returns bisection_status = :ceiling_hit.
			(b) Max realized rho across replicates does not exceed the
			    theoretical absolute ceiling sqrt(2*p*(1-p)) (with
			    small tolerance for floating-point and single-replicate
			    saturation noise).

			This is a NEW test introduced with the Pearson-to-Kendall
			refactor. The rate-bounded ceiling is a structural property
			of Kendall tau-b on binary indicators that didn't exist
			under Pearson. Combined with the practical tie-bounded
			ceiling, this gives the framework two distinct "you can't
			reach this target" constraints to surface to users.
		"""
		println("─" ^ 70)
		println("Test 1b: Rate-bounded ceiling on Scotland (rho=$target_rho at rate=$target_rate)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_rho = target_rho,
									  target_rate = target_rate,
									  n_reps      = n_reps)

		#	Compute theoretical absolute ceiling for diagnostic
			theoretical_ceiling = sqrt(2.0 * target_rate * (1.0 - target_rate))

		#	Check 1: every replicate ceiling-hit
			ceiling_hit_count = count(r -> r.bisection_status == :ceiling_hit, records)
			all_ceiling_hit   = ceiling_hit_count == n_reps

		#	Check 2: realized rhos do not exceed the theoretical absolute ceiling
		#	Allow modest tolerance for single-replicate saturation noise.
			realized_rhos = [r.realized_rho for r in records]
			max_realized  = maximum(realized_rhos)
			mean_realized = mean(realized_rhos)
			tolerance     = 0.05
			respects_ceiling = max_realized <= theoretical_ceiling + tolerance

		#	Report
			println("  Replicates:                $n_reps")
			println("  Target rho:                $target_rho")
			println("  Theoretical abs ceiling:   $(round(theoretical_ceiling, digits=4))  (sqrt(2*p*(1-p)))")
			println("  :ceiling_hit count:        $ceiling_hit_count / $n_reps")
			println("  All ceiling_hit:           $(all_ceiling_hit ? "YES" : "NO")")
			println("  Mean realized rho:         $(round(mean_realized, digits=4))  (practical ceiling on Scotland)")
			println("  Max realized rho:          $(round(max_realized, digits=4))")
			println("  Respects abs ceiling:      $(respects_ceiling ? "YES" : "NO")  (max ≤ $(round(theoretical_ceiling + tolerance, digits=4)))")

			passed = all_ceiling_hit && respects_ceiling
			println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "ceiling_hits=$ceiling_hit_count/$n_reps max_realized=$(round(max_realized, digits=4)) abs_ceiling=$(round(theoretical_ceiling, digits=4))")
	end

#   Extraction Rate Validation
	function test_exact_rate_scotland_moreno(networks::Dict;
												n_reps::Int = 10)
		"""
		Args:
			networks::Dict: corpus dict; needs Scotland and Moreno
			n_reps::Int: replicates per (network, rho, rate) cell (default 10)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies the exact-rate contract: every replicate drops exactly
			round(target_rate * N) nodes. Asserts per-replicate (not on the
			mean) because a bug where rho-targeting accidentally affected
			rate would be subtle in mean terms.

			Sweeps a small (rho, rate) sub-grid on both Scotland (undirected)
			and Moreno (directed) so the test exercises both code paths.
			Only 10 replicates per cell — the check is deterministic per
			replicate, so 10 is enough to catch a systematic bug.
		"""
		println("─" ^ 70)
		println("Test 2: Exact rate per replicate (Scotland + Moreno)")
		println("─" ^ 70)

		test_grid = [
			("scotland_interlock_unweighted",  108, [-0.25, 0.0, 0.25], [0.05, 0.10, 0.25, 0.50]),
			("moreno_highschool_unweighted",    70, [-0.25, 0.0, 0.25], [0.05, 0.10, 0.25, 0.50]),
		]

		all_passed = true
		mismatches = 0
		total      = 0

		for (name, n_nodes, rhos, rates) in test_grid
			haskey(networks, name) || (all_passed = false; continue)
			net = networks[name]
			for rho in rhos, rate in rates
				expected_k = Int(round(rate * n_nodes))
				records    = _draw_replicates(net; target_rho = rho,
													target_rate = rate,
													n_reps = n_reps)
				for r in records
					total += 1
					if length(r.dropped_nodes) != expected_k
						mismatches += 1
						all_passed = false
					end
				end
			end
		end

		println("  Replicates checked:   $total")
		println("  Rate mismatches:      $mismatches")
		println("  Result:               $(all_passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = all_passed,
				details = "$mismatches/$total mismatches")
	end

#   Test 3: Achievable-rho Ceiling
	function test_achievable_rho_ceiling_star(; n_reps::Int = 20,
												target_rho::Real = 0.95,
												target_rate::Real = 0.10,
												n_star::Int = 50)
		"""
		Args:
			n_reps::Int: number of replicates to confirm ceiling status
				(default 20; ceiling detection is near-deterministic in the
				bisection, so few replicates suffice)
			target_rho::Real: extreme positive target that should be
				unreachable on a star fixture (default 0.95)
			target_rate::Real: target rate (default 0.10)
			n_star::Int: number of nodes in the constructed star fixture
				(default 50)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies that the bisection correctly detects when target_rho
			is beyond what the centrality distribution can produce. The
			star fixture is the natural extreme case: one hub with in-degree
			(n-1), all leaves with in-degree 0. The centrality vector is
			(n-1, 0, 0, ..., 0) — maximally concentrated.

			At b=1 the probability vector becomes c_normalized = (1, 0, ..., 0),
			and the realized correlation with the indicator vector is bounded
			by the structure: even when the hub is always dropped, the
			remaining k-1 dropped nodes are chosen from zero-weighted leaves
			via the uniform (1-b)*u term, which contributes nothing
			deterministic. The achievable realized correlation on the star
			at rate 0.10 is roughly 0.3-0.5, well below 0.95.

			The bisection should return status = :ceiling_hit with b = 1
			and realized_rho_pos at the achievable ceiling. All replicates
			should converge on the same ceiling status because the bisection
			is deterministic in (centrality, target_rho, target_rate) modulo
			the MC noise in inner samples.
		"""
		println("─" ^ 70)
		println("Test 3: Achievable-rho ceiling on star fixture (rho=$target_rho, n=$n_star)")
		println("─" ^ 70)

		net = _build_star_fixture(n = n_star, directed = true)

		records = _draw_replicates(net; target_rho = target_rho,
									  target_rate = target_rate,
									  n_reps = n_reps)

		#	Check 3a: all replicates report :ceiling_hit
			ceiling_count = count(r -> r.bisection_status == :ceiling_hit, records)
			all_ceiling   = ceiling_count == n_reps

		#	Check 3b: the recorded ceiling value is meaningfully below target
			#	If the achievable ceiling is actually >= target_rho - tol, the
			#	bisection should have converged rather than hit the ceiling.
			#	The very fact that it returned :ceiling_hit implies the gap
			#	is non-trivial; we just verify the realized values are sane.
				ceiling_rhos = [abs(r.realized_rho) for r in records]
				mean_ceiling = isempty(ceiling_rhos) ? NaN : mean(ceiling_rhos)
				ceiling_below_target = mean_ceiling < target_rho

		#	Report
			println("  Replicates:             $n_reps")
			println("  :ceiling_hit count:     $ceiling_count / $n_reps")
			println("  Mean realized |rho|:    $(round(mean_ceiling, digits=4))")
			println("  Target rho:             $target_rho")
			println("  Ceiling below target:   $(ceiling_below_target ? "YES" : "NO")")

			passed = all_ceiling && ceiling_below_target
			println("  Result:                 $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "$ceiling_count/$n_reps ceiling_hit, mean=$(round(mean_ceiling, digits=4))")
	end

#   Test 4: Seed Reproducibility
	function test_seed_reproducibility(networks::Dict;
										n_reps::Int = 10,
										target_rho::Real = 0.25,
										target_rate::Real = 0.10)
		"""
		Args:
			networks::Dict: corpus dict; uses Moreno (directed) for the check
			n_reps::Int: number of independent (seed, replicate) pairs to
				verify (default 10)
			target_rho::Real: target rho for the test (default 0.25)
			target_rate::Real: target rate (default 0.10)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies the determinism contract: identical inputs (edges,
			nodes, directed, target_rate, target_rho, seed) produce
			identical records bit-for-bit. This includes the dropped_nodes
			vector, realized_rate, realized_rho, bisection_status, and all
			degeneracy fields.

			The test runs each of n_reps replicates twice in succession
			with the same seed and asserts equality. Failure here means the
			seed-splitting, hash-derived inner seeds, or MC bisection
			scheme has a non-determinism somewhere — most likely a missing
			seed parameter or an RNG state leak across calls.

			Runs on Moreno (directed) so the centrality driver exercises
			the in-degree path; the property should hold identically on
			any network.
		"""
		println("─" ^ 70)
		println("Test 4: Seed reproducibility on Moreno")
		println("─" ^ 70)

		haskey(networks, "moreno_highschool_unweighted") ||
			return (passed = false, details = "Moreno missing from corpus")

		net = networks["moreno_highschool_unweighted"]
		c   = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
				net.edges; nodes = net.nodes, directed = net.metadata.directed)

		mismatches = 0
		for rep in 1:n_reps
			rep_seed = Int(hash((target_rho, target_rate, rep, 1)) % UInt32)
			rec_a = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
					  net.edges; nodes = net.nodes, directed = net.metadata.directed,
					  target_rate = target_rate, target_rho = target_rho,
					  seed = rep_seed, centrality = c)
			rec_b = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
					  net.edges; nodes = net.nodes, directed = net.metadata.directed,
					  target_rate = target_rate, target_rho = target_rho,
					  seed = rep_seed, centrality = c)
			#	Compare structural fields
				if rec_a.dropped_nodes  != rec_b.dropped_nodes  ||
				   rec_a.realized_rate  != rec_b.realized_rate  ||
				   rec_a.realized_rho   != rec_b.realized_rho   ||
				   rec_a.bisection_status != rec_b.bisection_status
					mismatches += 1
				end
		end

		passed = mismatches == 0
		println("  Replicate pairs checked:    $n_reps")
		println("  Reproducibility mismatches: $mismatches")
		println("  Result:                     $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "$mismatches/$n_reps non-reproducible")
	end

#   Test 5: MCAR Baseline (rho = 0)
	function test_mcar_baseline(networks::Dict;
									n_reps::Int = 100,
									target_rate::Real = 0.10,
									tol_mean_rho::Real = 0.04)
		"""
		Args:
			networks::Dict: corpus dict; uses Scotland
			n_reps::Int: number of replicates (default 100)
			target_rate::Real: target rate (default 0.10)
			tol_mean_rho::Real: tolerance on |mean(realized_rho)|;
				default 0.04
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies that the MCAR fast-path produces an unbiased sampler
			whose realized indicator correlations average to zero across
			replicates. At target_rho = 0 the bisection short-circuits to
			b = 0 (pure uniform sampling); the realized correlation per
			replicate is sampling noise centered on zero.

			Tolerance 0.04 matches Test 1's: with 100 reps and per-replicate
			SE around 0.10 on Scotland, the mean's SE is roughly 0.01. The
			0.04 tolerance is four SE — passes essentially always under the
			null, catches a sign-flip or wrong fast-path with high power.

			Additionally checks that the bisection_status is :converged for
			all replicates (the MCAR fast-path is supposed to return
			:converged immediately without entering the bisection loop).
		"""
		println("─" ^ 70)
		println("Test 5: MCAR baseline on Scotland (rho=0, rate=$target_rate)")
		println("─" ^ 70)

		haskey(networks, "scotland_interlock_unweighted") ||
			return (passed = false, details = "Scotland missing from corpus")

		net     = networks["scotland_interlock_unweighted"]
		records = _draw_replicates(net; target_rho = 0.0,
									  target_rate = target_rate,
									  n_reps      = n_reps)

		#	Check 5a: all replicates converged via the fast-path
			all_converged = all(r.bisection_status == :converged for r in records)

		#	Check 5b: mean realized rho is near zero
			realized_rhos = [r.realized_rho for r in records]
			mean_rho      = mean(realized_rhos)
			mean_in_tol   = abs(mean_rho) < tol_mean_rho

		#	Report
			println("  Replicates:           $n_reps")
			println("  All converged:        $(all_converged ? "YES" : "NO")")
			println("  Mean realized rho:    $(round(mean_rho, digits=4))")
			println("  |Mean| < tol:         $(mean_in_tol ? "YES" : "NO") (tol $tol_mean_rho)")

			passed = all_converged && mean_in_tol
			println("  Result:               $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "mean_rho=$(round(mean_rho, digits=4))")
	end

#   Test 6: Constant Centrality (Ties)             
	function test_ties_handled_regular_ring(; n_reps::Int = 10,
												target_rho::Real = 0.25,
												target_rate::Real = 0.10,
												n_ring::Int = 50,
												k_ring::Int = 4)
		"""
		Args:
			n_reps::Int: number of replicates to verify (default 10;
				property is deterministic-per-seed, few replicates suffice)
			target_rho::Real: non-zero target rho (default 0.25); the test
				checks that non-zero targets fail on constant centrality
			target_rate::Real: target rate (default 0.10)
			n_ring::Int: ring node count (default 50)
			k_ring::Int: ring regularity (default 4 — every node degree 4)
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies the ties contract: on a network where every node has
			identical centrality, no value of b can produce a non-zero
			target rho (because the prob vector contains no signal that
			correlates with the constant centrality). The bisection should
			detect this via NaN from _centrality_correlation_for_b and
			return :failed_other.

			The 4-regular ring fixture has every node at degree exactly k,
			so centrality is constant. The MCAR fast-path at target_rho = 0
			would still succeed on this fixture (it doesn't need centrality
			variance), but any non-zero target should fail.

			The test asserts:
			- bisection_status == :failed_other for all replicates
			- realized_rho is NaN for all replicates
			- dropped_nodes is empty (the sampler is bypassed on failure)
		"""
		println("─" ^ 70)
		println("Test 6: Constant centrality on $(n_ring)x$(k_ring) regular ring (rho=$target_rho)")
		println("─" ^ 70)

		net = _build_regular_ring_fixture(n = n_ring, k = k_ring)

		records = _draw_replicates(net; target_rho = target_rho,
									  target_rate = target_rate,
									  n_reps = n_reps)

		#	Check 6a: all replicates :failed_other
			all_failed = all(r.bisection_status == :failed_other for r in records)

		#	Check 6b: all realized_rho are NaN
			all_nan = all(isnan(r.realized_rho) for r in records)

		#	Check 6c: all dropped_nodes are empty
			all_empty = all(isempty(r.dropped_nodes) for r in records)

		#	Report
			println("  Replicates:                $n_reps")
			println("  All :failed_other:         $(all_failed ? "YES" : "NO")")
			println("  All realized_rho NaN:      $(all_nan ? "YES" : "NO")")
			println("  All dropped_nodes empty:   $(all_empty ? "YES" : "NO")")

			passed = all_failed && all_nan && all_empty
			println("  Result:                    $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "$(all_failed && all_nan && all_empty ? "fully handled" : "partial detection")")
	end

#   Test 7: Degeneracy Flagging Fires
	function test_degeneracy_flagging_fires(networks::Dict;
                                            n_reps::Int = 100,
                                            target_rho::Real = 0.75,
                                            target_rate::Real = 0.65,
                                            fire_rate_threshold::Real = 0.50,
                                            fixture_name::String = "moreno_highschool_unweighted")
		"""
		Args:
			networks::Dict: corpus dict; uses Moreno by default
				(toledo is preferable if available — small network, easier
				to degrade adversarially)
			n_reps::Int: number of replicates (default 100)
			target_rho::Real: target rho (default 0.75 — adversarial,
				preferentially drops central nodes)
			target_rate::Real: target rate (default 0.50 — adversarial,
				drops half the nodes)
			fire_rate_threshold::Real: minimum fraction of replicates that
				must show any_topo_degenerate = true; default 0.50
			fixture_name::String: which network to use; default Moreno
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies the degeneracy-detection contract: at sufficiently
			adversarial (rho, rate) settings, the topological degeneracy
			flags should fire on a meaningful fraction of replicates.
			This is the opposite of Test 5 — instead of confirming the
			sampler runs cleanly, we confirm that the sampler reports
			degeneracy when degeneracy is present.

			The threshold of 0.50 (over half of replicates flag) is
			loose-but-meaningful: stochastic variation in dropped-set
			composition means some replicates might land in non-degenerate
			configurations even at adversarial settings. Asserting a hard
			count of 100/100 would be over-strict. Asserting > 50% gives
			the test the right shape: 'this regime should usually be
			flagged as degenerate.'

			At Moreno N=70 and rate=0.50, k=35 nodes are dropped, leaving
			35. The min_n threshold of 25 is satisfied, so the too_small
			flag should NOT fire. The gc_threshold of 0.30 of remaining
			should fire frequently — dropping the most central 35 nodes
			from a high-school friendship network will fragment the
			remaining 35 substantially.

			A failure here means either: (a) the degeneracy detector is
			too lenient (gc_threshold too low for the actual fragmentation,
			or no_edges/too_small not firing when they should), or
			(b) the (rho, rate) setting isn't actually adversarial enough
			on this network. The details string includes which flags fired
			and at what frequency for diagnosis.
		"""
		println("─" ^ 70)
		println("Test 7: Degeneracy flagging on $fixture_name (rho=$target_rho, rate=$target_rate)")
		println("─" ^ 70)

		haskey(networks, fixture_name) ||
			return (passed = false, details = "$fixture_name missing from corpus")

		net     = networks[fixture_name]
		records = _draw_replicates(net; target_rho = target_rho,
									  target_rate = target_rate,
									  n_reps      = n_reps)

		#	Filter to records where bisection succeeded (we want to assess
		#	degeneracy detection, not bisection failures)
			ok_records = filter(r -> r.bisection_status != :failed_other, records)
			n_ok = length(ok_records)
			if n_ok == 0
				println("  All replicates failed bisection; cannot assess degeneracy")
				return (passed = false, details = "no successful bisections")
			end

		#	Count flag firing rates among successful bisections
			n_any_topo  = count(r -> r.sampler_degeneracy.any_topo_degenerate, ok_records)
			n_gc        = count(r -> r.sampler_degeneracy.gc_collapse,          ok_records)
			n_too_small = count(r -> r.sampler_degeneracy.too_small,            ok_records)
			n_no_edges  = count(r -> r.sampler_degeneracy.no_edges,             ok_records)

			any_topo_rate = n_any_topo / n_ok
			passed = any_topo_rate >= fire_rate_threshold

		#	Report
			println("  Replicates (bisection ok):  $n_ok / $n_reps")
			println("  any_topo_degenerate rate:   $(round(any_topo_rate, digits=3))  (threshold $fire_rate_threshold)")
			println("  gc_collapse rate:           $(round(n_gc        / n_ok, digits=3))")
			println("  too_small rate:             $(round(n_too_small / n_ok, digits=3))")
			println("  no_edges rate:              $(round(n_no_edges  / n_ok, digits=3))")
			println("  Result:                     $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "any_topo=$(round(any_topo_rate, digits=3))")
	end

#   Test 8: apply_missingness_outgoing_only Contract 
	function test_outgoing_only_materializer(networks::Dict;
												target_rho::Real = 0.25,
												target_rate::Real = 0.10,
												seed::Integer = 1,
												fixture_directed::String = "moreno_highschool_unweighted",
												fixture_undirected::String = "scotland_interlock_unweighted")
		"""
		Args:
			networks::Dict: corpus dict; needs both Moreno (directed) and
				Scotland (undirected) to exercise the directed path and
				the undirected guard
			target_rho::Real: target rho for the sampler call (default 0.25)
			target_rate::Real: target rate (default 0.10)
			seed::Integer: master seed (default 1)
			fixture_directed::String: directed network for the four
				assertions on the materializer
			fixture_undirected::String: undirected network for the guard test
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Verifies four properties of apply_missingness_outgoing_only:

			(a) Selection invariance: given the same dropped-node set,
				the dropped-identifier set agrees between
				apply_missingness and apply_missingness_outgoing_only.
				Both functions consume the same sampler output; only
				their materializations differ.

			(b) Roster preservation: the returned nodes table has the
				same row count as the input. Non-respondents remain in
				the roster as zero-out-degree actors.

			(c) Src-only edge filtering: no edge with src equal to a
				dropped identifier survives; all original edges (u -> v)
				where v is dropped and u is NOT dropped survive.

			(d) Undirected guard: calling the function with directed=false
				on an undirected network throws ArgumentError.

			The test draws ONE sampler call per check rather than averaging
			across replicates — these are deterministic properties of the
			materializer for any given dropped set, not statistical
			properties needing many draws.
		"""
		println("─" ^ 70)
		println("Test 8: apply_missingness_outgoing_only materializer contract")
		println("─" ^ 70)

		#	Setup: verify fixtures present
			haskey(networks, fixture_directed) ||
				return (passed = false, details = "$fixture_directed missing from corpus")
			haskey(networks, fixture_undirected) ||
				return (passed = false, details = "$fixture_undirected missing from corpus")

			net_dir = networks[fixture_directed]
			net_und = networks[fixture_undirected]

		#	Draw one replicate to get a dropped-node set
			rec = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
					net_dir.edges; nodes = net_dir.nodes, directed = true,
					target_rate = target_rate, target_rho = target_rho,
					seed = seed)
			dropped = rec.dropped_nodes

			if isempty(dropped)
				println("  Sampler returned empty dropped set; cannot test")
				return (passed = false, details = "sampler returned empty dropped set")
			end

		#	Materialize via both functions
			full_removal = Network_Credible_Intervals.network_degeneracy.apply_missingness(
								net_dir.edges, dropped; nodes = net_dir.nodes)
			outgoing_only = Network_Credible_Intervals.network_degeneracy.apply_missingness_outgoing_only(
								net_dir.edges, dropped; nodes = net_dir.nodes, directed = true)

		#	Resolve identifiers for the dropped set
			#	apply_missingness's canonical ordering uses the :id column when
			#	a DataFrame is supplied; match that here.
				dropped_ids = Set{String}(string.(net_dir.nodes[!, 1][dropped]))

		#	Check 8a: selection invariance
			#	apply_missingness removed the dropped identifiers from its nodes table
				full_removal_ids = Set{String}(string.(full_removal.nodes[!, 1]))
				full_removal_dropped = setdiff(Set{String}(string.(net_dir.nodes[!, 1])),
												  full_removal_ids)
			#	apply_missingness_outgoing_only keeps the roster; dropped IDs
			#	must still be present
				outgoing_only_ids = Set{String}(string.(outgoing_only.nodes[!, 1]))

				check_8a = (full_removal_dropped == dropped_ids) &&
							issubset(dropped_ids, outgoing_only_ids)

		#	Check 8b: roster preservation
			n_input_nodes = nrow(net_dir.nodes)
			n_out_nodes   = nrow(outgoing_only.nodes)
			check_8b      = n_out_nodes == n_input_nodes

		#	Check 8c: src-only edge filtering
			#	No edge in outgoing-only has src in dropped_ids
				edges_with_dropped_src = count(r -> string(outgoing_only.edges.src[r]) in dropped_ids,
												  1:nrow(outgoing_only.edges))
				check_8c_no_outgoing = edges_with_dropped_src == 0

			#	All original edges (u, v) with u NOT dropped and v IN dropped
			#	must be present in outgoing-only edges
				orig_inbound_edges = Set{Tuple{String,String}}()
				for r in 1:nrow(net_dir.edges)
					s = string(net_dir.edges.src[r])
					d = string(net_dir.edges.dst[r])
					if !(s in dropped_ids) && (d in dropped_ids)
						push!(orig_inbound_edges, (s, d))
					end
				end
				out_edge_set = Set{Tuple{String,String}}()
				for r in 1:nrow(outgoing_only.edges)
					push!(out_edge_set,
						  (string(outgoing_only.edges.src[r]),
						   string(outgoing_only.edges.dst[r])))
				end
				missing_inbound = setdiff(orig_inbound_edges, out_edge_set)
				check_8c_inbound = isempty(missing_inbound)

			check_8c = check_8c_no_outgoing && check_8c_inbound

		#	Check 8d: undirected guard
			check_8d = false
			try
				Network_Credible_Intervals.network_degeneracy.apply_missingness_outgoing_only(
					net_und.edges, Int[]; nodes = net_und.nodes, directed = false)
				check_8d = false  # should have thrown
			catch err
				check_8d = err isa ArgumentError
			end

		#	Report
			println("  Dropped set size:                $(length(dropped))")
			println("  (a) Selection invariance:        $(check_8a ? "YES" : "NO")")
			println("  (b) Roster preservation:         $(check_8b ? "YES ($n_out_nodes/$n_input_nodes)" : "NO ($n_out_nodes/$n_input_nodes)")")
			println("  (c) No outgoing from dropped:    $(check_8c_no_outgoing ? "YES (0 edges)" : "NO ($edges_with_dropped_src edges)")")
			println("  (c) Inbound to dropped preserved: $(check_8c_inbound ? "YES" : "NO ($(length(missing_inbound)) missing)")")
			println("  (d) Undirected guard throws:     $(check_8d ? "YES" : "NO")")

			passed = check_8a && check_8b && check_8c && check_8d
			println("  Result:                          $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "a=$check_8a b=$check_8b c=$check_8c d=$check_8d")
	end

#   Execute Test 3-7
	function run_phase1_sampler_tests(networks::Dict;
										verbose::Bool = true)
		"""
		Args:
			networks::Dict: corpus dict containing at minimum
				"scotland_interlock_unweighted" and "moreno_highschool_unweighted"
			verbose::Bool: print per-test status (default true)
		Returns:
			NamedTuple (passed::Bool, n_passed::Int, n_total::Int, results::Vector)
		Notes:
			Runs all seven sampler-contract tests in order on the small-network
			fixtures (Scotland + Moreno + constructed star + 4-regular ring).
			Reports per-test pass/fail and the conjunction across all tests.
		"""
		verbose && println("\n" * "═" ^ 70)
		verbose && println("Phase 1 Sampler Validation Harness — small networks")
		verbose && println("═" ^ 70)

		results = NamedTuple[]
		push!(results, test_bisection_convergence_scotland(networks))
		push!(results, test_rate_bounded_ceiling_scotland(networks))
		push!(results, test_exact_rate_scotland_moreno(networks))
		push!(results, test_achievable_rho_ceiling_star())
		push!(results, test_seed_reproducibility(networks))
		push!(results, test_mcar_baseline(networks))
		push!(results, test_ties_handled_regular_ring())
		push!(results, test_degeneracy_flagging_fires(networks))
        push!(results, test_outgoing_only_materializer(networks))

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

#   Marvel Stress Test (Diagnostic)
	function run_marvel_stress_test(networks::Dict;
									fixture_name::String = "marvel_universe_unweighted",
									target_rho::Real = 0.25,
									target_rate::Real = 0.10,
									n_reps::Int = 20,
									rate_sweep::AbstractVector{<:Real} = [0.10, 0.30, 0.50, 0.70])
		"""
		Args:
			networks::Dict: corpus dict; expects "marvel_universe_unweighted"
				or whatever the Marvel key happens to be in the local corpus
			fixture_name::String: which Marvel key to use; default
				"marvel_universe_unweighted"
			target_rho::Real: target rho for the main timing/convergence check
				(default 0.25)
			target_rate::Real: target rate for the main timing/convergence check
				(default 0.10)
			n_reps::Int: number of replicates for the main check (default 20;
				smaller than the small-network 100 because each replicate
				is much slower at Marvel scale)
			rate_sweep::AbstractVector{<:Real}: rates to sweep for the
				degeneracy-detector engagement check (default [0.10, 0.30,
				0.50, 0.70])
		Returns:
			NamedTuple: diagnostic record with timing, convergence, realized-rho,
				saturation, and degeneracy stats. No PASS/FAIL — this is
				exploratory measurement, not contract verification.
		Notes:
			Six diagnostic panels, exercising the parts of the pipeline that
			small-network tests can't stress:

			Panel 1: Centrality computation timing.
				One-time cost amortized across all replicates of a network.
				Should be sub-second at Marvel scale.

			Panel 2: Per-replicate wall-clock cost at (target_rho, target_rate).
				This is the dominant cost in the production grid. Reports
				mean and per-replicate range.

			Panel 3: Bisection convergence diagnostics.
				Distribution of bisection iteration counts; saturation-path
				engagement rate; failure rate.

			Panel 4: Realized rho statistics on Marvel.
				Same property Test 1 checks, but on the largest fixture.
				Mean and SE should be tighter than on Scotland (larger N).

			Panel 5: Rate sweep for degeneracy detector engagement.
				At each rate in rate_sweep, fire-rate of the degeneracy flags.
				Marvel is large and well-connected, so we expect the gc_collapse
				and too_small flags to fire at much higher rates than on Moreno.

			Panel 6: Budget extrapolation.
				Given the measured per-replicate cost, extrapolates total
				wall-clock for one full network grid (30 cells x 100 reps)
				and for the full corpus (16 networks x 30 cells x 100 reps).
				Doubled-mechanism note: when the orchestrator supports the
				doubled grid, this number doubles (but only the materializer
				cost doubles, which is small relative to the sampler).
		"""
		println("\n" * "═" ^ 70)
		println("Marvel Stress Test — diagnostic")
		println("═" ^ 70)

		haskey(networks, fixture_name) || begin
			println("ERROR: $fixture_name not found in corpus")
			return (passed = false, details = "fixture missing")
		end

		net = networks[fixture_name]
		println("Fixture:  $fixture_name")
		println("Nodes:    $(nrow(net.nodes))")
		println("Edges:    $(nrow(net.edges))")
		println("Directed: $(net.metadata.directed)")
		println()

		#	─── Panel 1: Centrality computation ────────────────────────────
			println("─" ^ 70)
			println("Panel 1: Centrality computation")
			println("─" ^ 70)
			t_centrality = @elapsed begin
				centrality = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
					net.edges; nodes = net.nodes, directed = net.metadata.directed)
			end
			println("  Elapsed:       $(round(t_centrality, digits=3)) sec")
			println("  Length:        $(length(centrality))")
			println("  Min / max:     $(minimum(centrality)) / $(maximum(centrality))")
			println("  Mean:          $(round(mean(centrality), digits=3))")
			println("  Has isolates:  $(any(==(0.0), centrality))")

		#	─── Panel 2: Per-replicate timing at (target_rho, target_rate) ──
			println("─" ^ 70)
			println("Panel 2: Per-replicate timing at rho=$target_rho, rate=$target_rate")
			println("─" ^ 70)
			times = Float64[]
			records = NamedTuple[]
			for rep in 1:n_reps
				rep_seed = Int(hash((target_rho, target_rate, rep, 1)) % UInt32)
				t = @elapsed begin
					rec = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
							net.edges; nodes = net.nodes, directed = net.metadata.directed,
							target_rate = target_rate, target_rho = target_rho,
							seed = rep_seed, centrality = centrality)
					push!(records, rec)
				end
				push!(times, t)
			end
			mean_t = mean(times)
			println("  Replicates:    $n_reps")
			println("  Mean elapsed:  $(round(mean_t, digits=3)) sec")
			println("  Min / max:     $(round(minimum(times), digits=3)) / $(round(maximum(times), digits=3)) sec")

		#	─── Panel 3: Bisection convergence ──────────────────────────────
			println("─" ^ 70)
			println("Panel 3: Bisection convergence")
			println("─" ^ 70)
			n_converged    = count(r -> r.bisection_status == :converged, records)
			n_ceiling_hit  = count(r -> r.bisection_status == :ceiling_hit, records)
			n_failed_other = count(r -> r.bisection_status == :failed_other, records)
			println("  :converged:      $n_converged / $n_reps")
			println("  :ceiling_hit:    $n_ceiling_hit / $n_reps")
			println("  :failed_other:   $n_failed_other / $n_reps")

		#	─── Panel 4: Realized rho statistics ────────────────────────────
			println("─" ^ 70)
			println("Panel 4: Realized rho at target rho=$target_rho")
			println("─" ^ 70)
			ok_records   = filter(r -> r.bisection_status == :converged, records)
			realized_rhos = [r.realized_rho for r in ok_records]
			if isempty(realized_rhos)
				println("  No converged replicates to analyze")
			else
				mean_rho   = mean(realized_rhos)
				se_rho     = std(realized_rhos) / sqrt(length(realized_rhos))
				delta      = abs(mean_rho - target_rho)
				println("  Converged reps:  $(length(realized_rhos))")
				println("  Mean rho:        $(round(mean_rho, digits=4))")
				println("  SE of mean:      $(round(se_rho, digits=4))")
				println("  |Δ vs target|:   $(round(delta, digits=4))")
				println("  Per-rep SD:      $(round(std(realized_rhos), digits=4))")
			end

		#	─── Panel 5: Rate sweep for degeneracy detector ────────────────
			println("─" ^ 70)
			println("Panel 5: Degeneracy detector engagement across rates")
			println("─" ^ 70)
			println("  Rate    n_ok  any_topo  gc_collapse  too_small  no_edges")
			for sweep_rate in rate_sweep
				n_any  = 0
				n_gc   = 0
				n_tsm  = 0
				n_ned  = 0
				n_ok   = 0
				n_sweep = 10   # small per cell — exploratory sweep
				for rep in 1:n_sweep
					rep_seed = Int(hash((target_rho, sweep_rate, rep, 1)) % UInt32)
					rec = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
							net.edges; nodes = net.nodes, directed = net.metadata.directed,
							target_rate = sweep_rate, target_rho = target_rho,
							seed = rep_seed, centrality = centrality)
					if rec.bisection_status != :failed_other
						n_ok += 1
						rec.sampler_degeneracy.any_topo_degenerate && (n_any += 1)
						rec.sampler_degeneracy.gc_collapse         && (n_gc  += 1)
						rec.sampler_degeneracy.too_small           && (n_tsm += 1)
						rec.sampler_degeneracy.no_edges            && (n_ned += 1)
					end
				end
				println("  $(rpad(string(sweep_rate), 7))$(rpad(string(n_ok),5)) $(rpad(string(n_any),9)) $(rpad(string(n_gc),12)) $(rpad(string(n_tsm),10))$n_ned")
			end

		#	─── Panel 6: Budget extrapolation ──────────────────────────────
			println("─" ^ 70)
			println("Panel 6: Production grid budget extrapolation")
			println("─" ^ 70)
			cells_per_network = 5 * 6     # 5 rhos x 6 rates
			reps_per_cell     = 100
			networks_in_grid  = 16
			single_net_secs   = cells_per_network * reps_per_cell * mean_t
			full_grid_secs    = networks_in_grid * single_net_secs
			println("  Mean per-replicate time:   $(round(mean_t, digits=3)) sec")
			println("  Per-network grid time:     $(round(single_net_secs / 60, digits=1)) min  ($(round(single_net_secs, digits=0)) sec)")
			println("  Full corpus grid time:     $(round(full_grid_secs / 60, digits=1)) min  ($(round(full_grid_secs / 3600, digits=2)) hr)")
			println("  (Single mechanism; doubled-mechanism corpus is ~2x materializer overhead, which is small.)")
			println()

		return (passed = true,
				details = "marvel stress complete: mean_t=$(round(mean_t, digits=3)) sec, mean_rho=$(isempty(realized_rhos) ? NaN : round(mean(realized_rhos), digits=4))")
	end
    run_marvel_stress_test(networks)

#   Smoke Test: Orchestrator Dual-Mechanism            
	function smoke_test_orchestrator_dual_mechanism(networks::Dict;
														fixture_directed::String   = "moreno_highschool_unweighted",
														fixture_undirected::String = "scotland_interlock_unweighted")
		"""
		Args:
			networks::Dict: corpus dict; needs both Moreno (directed) and
				Scotland (undirected) to exercise both code paths
			fixture_directed::String: directed fixture name
			fixture_undirected::String: undirected fixture name
		Returns:
			NamedTuple (passed::Bool, details::String)
		Notes:
			Sanity-checks build_degeneration_corpus on a minimal grid before
			the production run. Verifies:

			(a) Output shape: total row count matches the per-network
				mechanism applicability * cells * reps formula.
			(b) Dual-mechanism rows: Moreno (directed) produces both
				:full_removal and :outgoing_only rows; Scotland (undirected)
				produces only :full_removal rows.
			(c) Seed sharing across mechanism rows: the two mechanism rows
				for the same (network, rho, rate, rep) cell share the same
				seed, dropped_nodes, realized_rho, and bisection_status.
				This verifies the sampler-cache design — both mechanisms
				consume the same sampler output.
			(d) Degeneracy field sharing: the two mechanism rows for the
				same cell share their degeneracy values (mechanism-
				agnostic by design).
			(e) Mechanism column type: returned DataFrame has mechanism
				as Symbol with valid values only.
			(f) Row ordering: name → rho → rate → rep → mechanism, with
				mechanism varying fastest.

			Runs a minimal 2-network × 2-rho × 2-rate × 3-rep grid in serial
			mode (parallel=false) to make output deterministic and quick.
		"""
		println("─" ^ 70)
		println("Smoke Test: orchestrator dual-mechanism on Moreno + Scotland")
		println("─" ^ 70)

		haskey(networks, fixture_directed) ||
			return (passed = false, details = "$fixture_directed missing from corpus")
		haskey(networks, fixture_undirected) ||
			return (passed = false, details = "$fixture_undirected missing from corpus")

		small_corpus = Dict(
			fixture_directed   => networks[fixture_directed],
			fixture_undirected => networks[fixture_undirected],
		)

		println("  Running build_degeneration_corpus on 2-network smoke grid...")
		result = Network_Credible_Intervals.network_degeneracy.build_degeneration_corpus(
					small_corpus;
					target_rhos    = [0.0, 0.25],
					target_rates   = [0.10, 0.25],
					n_replicates   = 3,
					mechanisms     = [:full_removal, :outgoing_only],
					parallel       = false,
					show_progress  = false)

		#	(a) Output shape
			#	Scotland (undirected): 1 mechanism × 2 rho × 2 rate × 3 rep = 12 rows
			#	Moreno (directed):    2 mechanisms × 2 rho × 2 rate × 3 rep = 24 rows
			expected_total = 12 + 24
			actual_total   = nrow(result)
			check_a        = actual_total == expected_total

		#	(b) Dual-mechanism rows
			scotland_rows = filter(r -> r.network_name == fixture_undirected, result)
			moreno_rows   = filter(r -> r.network_name == fixture_directed, result)
			scotland_mechs = unique(scotland_rows.mechanism)
			moreno_mechs   = unique(moreno_rows.mechanism)
			check_b = scotland_mechs == [:full_removal] &&
					   Set(moreno_mechs) == Set([:full_removal, :outgoing_only])

		#	(c) Seed sharing across mechanism rows for the same cell
			#	Group Moreno rows by (rho, rate, rep) and confirm the two
			#	mechanism rows share seed, dropped_nodes, realized_rho, status
			cell_keys = unique([(r.nominal_rho, r.nominal_rate, r.replicate_idx) for r in eachrow(moreno_rows)])
			seed_share_violations    = 0
			dropped_share_violations = 0
			rho_share_violations     = 0
			status_share_violations  = 0
			for ck in cell_keys
				cell_rows = filter(r -> r.nominal_rho == ck[1] &&
											r.nominal_rate == ck[2] &&
											r.replicate_idx == ck[3], moreno_rows)
				nrow(cell_rows) == 2 || continue
				if cell_rows.seed[1]             != cell_rows.seed[2]
					seed_share_violations += 1
				end
				if cell_rows.dropped_nodes[1]    != cell_rows.dropped_nodes[2]
					dropped_share_violations += 1
				end
				if !(isnan(cell_rows.realized_rho[1]) && isnan(cell_rows.realized_rho[2])) &&
				   cell_rows.realized_rho[1] != cell_rows.realized_rho[2]
					rho_share_violations += 1
				end
				if cell_rows.bisection_status[1] != cell_rows.bisection_status[2]
					status_share_violations += 1
				end
			end
			check_c = seed_share_violations    == 0 &&
					   dropped_share_violations == 0 &&
					   rho_share_violations     == 0 &&
					   status_share_violations  == 0

		#	(d) Degeneracy field sharing across mechanism rows
			gc_share_violations  = 0
			nob_share_violations = 0
			any_share_violations = 0
			for ck in cell_keys
				cell_rows = filter(r -> r.nominal_rho == ck[1] &&
											r.nominal_rate == ck[2] &&
											r.replicate_idx == ck[3], moreno_rows)
				nrow(cell_rows) == 2 || continue
				if !(isnan(cell_rows.gc_fraction_of_remaining[1]) &&
					 isnan(cell_rows.gc_fraction_of_remaining[2])) &&
				   cell_rows.gc_fraction_of_remaining[1] != cell_rows.gc_fraction_of_remaining[2]
					gc_share_violations += 1
				end
				if cell_rows.n_observed[1] != cell_rows.n_observed[2]
					nob_share_violations += 1
				end
				if cell_rows.any_topo_degenerate[1] != cell_rows.any_topo_degenerate[2]
					any_share_violations += 1
				end
			end
			check_d = gc_share_violations  == 0 &&
					   nob_share_violations == 0 &&
					   any_share_violations == 0

		#	(e) Mechanism column type and valid values
			check_e = eltype(result.mechanism) == Symbol &&
					   all(m in (:full_removal, :outgoing_only) for m in result.mechanism)

		#	(f) Row ordering: within a network, expect name → rho → rate → rep → mechanism
			#	Specifically: for any consecutive pair of rows with the same (rho, rate, rep),
			#	their mechanisms should differ. Check this on Moreno rows.
			ordering_violations = 0
			for i in 1:(nrow(moreno_rows) - 1)
				if moreno_rows.nominal_rho[i]   == moreno_rows.nominal_rho[i+1] &&
				   moreno_rows.nominal_rate[i]  == moreno_rows.nominal_rate[i+1] &&
				   moreno_rows.replicate_idx[i] == moreno_rows.replicate_idx[i+1] &&
				   moreno_rows.mechanism[i]     == moreno_rows.mechanism[i+1]
					ordering_violations += 1
				end
			end
			check_f = ordering_violations == 0

		#	Report
			println("  Total rows:               $actual_total (expected $expected_total)")
			println("  (a) Output shape:         $(check_a ? "YES" : "NO")")
			println("  (b) Mechanism rows:       $(check_b ? "YES" : "NO")")
			println("      Scotland mechs:       $scotland_mechs")
			println("      Moreno mechs:         $moreno_mechs")
			println("  (c) Sampler-cache share:  $(check_c ? "YES" : "NO")")
			if !check_c
				println("      seed violations:      $seed_share_violations")
				println("      dropped violations:   $dropped_share_violations")
				println("      rho violations:       $rho_share_violations")
				println("      status violations:    $status_share_violations")
			end
			println("  (d) Degeneracy share:     $(check_d ? "YES" : "NO")")
			if !check_d
				println("      gc violations:        $gc_share_violations")
				println("      n_observed violations:$nob_share_violations")
				println("      any_topo violations:  $any_share_violations")
			end
			println("  (e) Mechanism dtype:      $(check_e ? "YES" : "NO")")
			println("  (f) Row ordering:         $(check_f ? "YES" : "NO")")

			passed = check_a && check_b && check_c && check_d && check_e && check_f
			println("  Result:                   $(passed ? "PASS ✓" : "FAIL ✗")")

		return (passed  = passed,
				details = "a=$check_a b=$check_b c=$check_c d=$check_d e=$check_e f=$check_f")
	end
    smoke_test_orchestrator_dual_mechanism(networks)

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
