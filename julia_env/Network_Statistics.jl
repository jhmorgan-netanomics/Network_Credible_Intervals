#Calculate Network Statistics for Comparison Table
#Jonathan H. Morgan
#26 May 2026

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

using CSV
using DataFrames
using Printf
using Statistics
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

#############
#   TESTS   #
#############

#	Build the Foundation-Test Synthetic Network (8 Nodes, 2 Isolates)
	function _build_foundation_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for an 8-node directed graph
				designed to exercise every foundation-section function with
				hand-computable expected values.
		Notes:
			Topology (binary):
				1 → 2, 1 → 3            (node 1 is a 2-out hub)
				2 → 3, 3 → 2            (mutual dyad)
				4 → 5, 5 → 6, 6 → 4     (directed triangle)
				Nodes 7 and 8 are isolates

			Weighted version uses edge weights [1, 2, 3, 3, 4, 1, 2] in the
			order above.

			Expected hand-computed values are documented in the test function
			that consumes this network.
		"""

		#	Build Edge List
			edges = DataFrame(
				src    = [1, 1, 2, 3, 4, 5, 6],
				dst    = [2, 3, 3, 2, 5, 6, 4],
				weight = [1.0, 2.0, 3.0, 3.0, 4.0, 1.0, 2.0]
			)

		#	Build Node List (Strings to Match _graph_to_sparse_matrix Convention)
			nodes = DataFrame(
				id    = string.(1:8),
				label = string.(1:8)
			)

		#	Convert Edge IDs to Strings as Well
			edges_str = DataFrame(
				src    = string.(edges.src),
				dst    = string.(edges.dst),
				weight = edges.weight
			)

		#	Metadata
			metadata = (
				network_name  = "Foundation Test Synthetic",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = true,
				n_nodes       = 8,
				n_edges       = 7
			)

		#	Return Triple
			return (edges = edges_str, nodes = nodes, metadata = metadata)
	end

#	Run Foundation Tests on a Synthetic Network with Known Values
	function run_synthetic_foundation_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests every foundation-section function on an 8-node synthetic
			network with hand-computed expected values. Includes two isolates
			to verify the `nodes` argument correctly preserves isolate rows
			and shifts centralization measures.

			Expected values (binary, N=8 including isolates):
				in-degree:    [0, 2, 2, 1, 1, 1, 0, 0]
				out-degree:   [2, 1, 1, 1, 1, 1, 0, 0]
				total-degree: [2, 3, 3, 2, 2, 2, 0, 0]

			Expected values (weighted, N=8 including isolates):
				in-strength:  [0, 4, 5, 2, 4, 1, 0, 0]
				out-strength: [3, 3, 3, 4, 1, 2, 0, 0]
				total:        [3, 7, 8, 6, 5, 3, 0, 0]

			Distribution measures (binary, N=8 with isolates):
				gini(total)        = 0.321429
				freeman_cent(total) = 10/42 = 0.238095
				centralization(total) = std([2,3,3,2,2,2,0,0]) = 1.164965

			Without isolates (only the 6 connected nodes):
				gini(total)        = 0.095238
				freeman_cent(total) = 4/20 = 0.200000
				centralization(total) = std([2,3,3,2,2,2]) = 0.516398
		"""

		println("=" ^ 70)
		println("Synthetic foundation tests for network_statistics.jl")
		println("=" ^ 70)
		println("\nNetwork: 8 nodes, 7 directed edges, 2 isolates (nodes 7, 8)")

		all_passed = true

		#	Build Test Network
			net = _build_foundation_test_network()
			edges = net.edges
			nodes = net.nodes

		#	Expected Values (Binary)
			expected_in_bin  = [0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0]
			expected_out_bin = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
			expected_tot_bin = [2.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.0, 0.0]

		#	Expected Values (Weighted)
			expected_in_wt  = [0.0, 4.0, 5.0, 2.0, 4.0, 1.0, 0.0, 0.0]
			expected_out_wt = [3.0, 3.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0]
			expected_tot_wt = [3.0, 7.0, 8.0, 6.0, 5.0, 3.0, 0.0, 0.0]

		#	Test 1: in_degree Returns Isolates When `nodes` Supplied
			println("\nTest 1: in_degree with nodes argument (preserves isolates)")
			try
				#	Run with nodes (should return 8 rows including isolates as zero)
					in_full = in_degree(edges; nodes = nodes, weighted = false)
					sort!(in_full, :node)
				#	Assertions
					@assert nrow(in_full) == 8 "Expected 8 rows including isolates, got $(nrow(in_full))"
					@assert in_full.in_degree == expected_in_bin "in_degree mismatch: $(in_full.in_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: in_degree Without `nodes` Drops Isolates (Backward Compatibility)
			println("\nTest 2: in_degree without nodes (legacy behavior)")
			try
				in_partial = in_degree(edges; weighted = false)
				@assert nrow(in_partial) == 6 "Expected 6 rows (no isolates), got $(nrow(in_partial))"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: out_degree on Binary Adjacency
			println("\nTest 3: out_degree with nodes argument")
			try
				out_full = sort(out_degree(edges; nodes = nodes, weighted = false), :node)
				@assert nrow(out_full) == 8
				@assert out_full.out_degree == expected_out_bin "out_degree mismatch: $(out_full.out_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: total_degree (Directed, Binary, with Isolates)
			println("\nTest 4: total_degree (directed, binary, with isolates)")
			try
				tot_full = sort(total_degree(edges;
				                            nodes = nodes,
				                            weighted = false,
				                            directed = true), :node)
				@assert nrow(tot_full) == 8
				@assert tot_full.total_degree == expected_tot_bin "total_degree mismatch: $(tot_full.total_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Weighted Total Degree
			println("\nTest 5: total_degree (directed, weighted, with isolates)")
			try
				tot_w = sort(total_degree(edges;
				                         nodes = nodes,
				                         weighted = true,
				                         directed = true), :node)
				@assert nrow(tot_w) == 8
				@assert tot_w.total_degree == expected_tot_wt "weighted total mismatch: $(tot_w.total_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: gini_coefficient on Total-Degree (with Isolates)
			println("\nTest 6: gini_coefficient on total-degree (N = 8 with isolates)")
			try
				tot = sort(total_degree(edges;
				                       nodes = nodes,
				                       weighted = false,
				                       directed = true), :node)
				g = gini_coefficient(tot.total_degree)
				@assert isapprox(g, 0.321429; atol = 1e-5) "Gini with isolates: expected 0.321429, got $g"
				println("  PASSED (= $(round(g, digits = 6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: gini_coefficient on Total-Degree (without Isolates)
			println("\nTest 7: gini_coefficient on total-degree (N = 6, no isolates)")
			try
				tot_no_iso = sort(total_degree(edges;
				                              weighted = false,
				                              directed = true), :node)
				g = gini_coefficient(tot_no_iso.total_degree)
				@assert isapprox(g, 0.095238; atol = 1e-5) "Gini without isolates: expected 0.095238, got $g"
				println("  PASSED (= $(round(g, digits = 6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 8: freeman_degree_centralization with Isolates (N = 8)
			println("\nTest 8: freeman_degree_centralization (N = 8 with isolates)")
			try
				f = freeman_degree_centralization(edges;
				                                  nodes = nodes,
				                                  mode = :all,
				                                  directed = true)
				expected = 10.0 / 42.0
				@assert isapprox(f, expected; atol = 1e-10) "Freeman with isolates: expected $expected, got $f"
				println("  PASSED (= $(round(f, digits = 6)) = 10/42)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 9: freeman_degree_centralization without Isolates (N = 6)
			println("\nTest 9: freeman_degree_centralization (N = 6, no isolates)")
			try
				f_no_iso = freeman_degree_centralization(edges;
				                                         mode = :all,
				                                         directed = true)
				expected = 4.0 / 20.0
				@assert isapprox(f_no_iso, expected; atol = 1e-10) "Freeman no isolates: expected $expected, got $f_no_iso"
				println("  PASSED (= $(round(f_no_iso, digits = 6)) = 4/20)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 10: SMM-Style Centralization (std) with and without Isolates
			println("\nTest 10: centralization (SD) with vs without isolates")
			try
				tot_full = total_degree(edges; nodes = nodes, weighted = false, directed = true)
				tot_iso  = total_degree(edges;                 weighted = false, directed = true)

				c_full = centralization(tot_full.total_degree)
				c_iso  = centralization(tot_iso.total_degree)

				@assert isapprox(c_full, 1.164965; atol = 1e-5) "SD with isolates: expected 1.164965, got $c_full"
				@assert isapprox(c_iso,  0.516398; atol = 1e-5) "SD no isolates: expected 0.516398, got $c_iso"
				println("  PASSED  (with isolates = $(round(c_full, digits = 4)), without = $(round(c_iso, digits = 4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 11: freeman_degree_normalization on the Same Network
			println("\nTest 11: freeman_degree_normalization (mode=:all, directed, asymmetric)")
			try
				#	Expected denominator: 2 * V * (N - 1) = 2 * 1 * 7 = 14 (binary, asymmetric)
				#	Expected scores: total_degree / 14 = [2,3,3,2,2,2,0,0] / 14
					fdn = sort(freeman_degree_normalization(edges;
					                                       nodes = nodes,
					                                       mode = :all,
					                                       directed = true,
					                                       weighted = false), :node)
					expected_scores = [2.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.0, 0.0] ./ 14.0
					@assert isapprox(fdn.freeman_degree, expected_scores; atol = 1e-10) "Got $(fdn.freeman_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic foundation tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_foundation_tests()

#   Looking at Scotland Interlocking Directorate Data
    scotland_nodes = networks["scotland_interlock_unweighted"].nodes
    scotland_edges = networks["scotland_interlock_unweighted"].edges

    scot_tot = total_degree(scotland_edges;
                            nodes    = scotland_nodes,
                            weighted = false,
                            directed = false)

    println("Scotland N: $(nrow(scot_tot))")
    println("Isolates:   $(sum(scot_tot.total_degree .== 0))")
    println("Gini:       $(round(gini_coefficient(scot_tot.total_degree), digits=4))")

#	Run All Foundation Tests for network_statistics.jl
	function run_network_statistics_foundation_tests(networks::Dict)
		"""
		Args:
			networks::Dict: dictionary of loaded GraphML networks, keyed by name,
			               with values as (edges, nodes, metadata) NamedTuples
		Returns:
			Bool: true if all tests passed, false otherwise
		Notes:
			Verifies correctness of the foundation-section functions:
			gini_coefficient, centralization, rand_index, in_degree, out_degree,
			total_degree, freeman_degree_normalization, and
			freeman_degree_centralization. Combines hand-computable assertions
			with smoke tests on loaded corpus networks.

			Expects `networks` to contain at minimum the keys
			"moreno_highschool_weighted" and "scotland_interlock_unweighted".
		"""

		println("=" ^ 70)
		println("Foundation tests for network_statistics.jl")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: gini_coefficient on Known Values
			println("\nTest 1: gini_coefficient")
			try
				@assert isapprox(gini_coefficient([1, 1, 1, 1]), 0.0; atol = 1e-10)
				@assert isapprox(gini_coefficient([0, 0, 0, 4]), 0.75; atol = 1e-10)
				@assert isapprox(gini_coefficient([1, 2, 3, 4, 5]), 0.2667; atol = 1e-3)
				@assert gini_coefficient(Int[]) == 0.0
				@assert gini_coefficient([5]) == 0.0
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: centralization (Standard Deviation)
			println("\nTest 2: centralization")
			try
				@assert centralization([1.0, 1.0, 1.0]) == 0.0
				@assert isapprox(centralization([1.0, 2.0, 3.0]),
				                 std([1.0, 2.0, 3.0]); atol = 1e-10)
				@assert centralization(Float64[]) == 0.0
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: rand_index on Known Cases
			println("\nTest 3: rand_index")
			try
				@assert rand_index([1, 1, 2, 2], [1, 1, 2, 2]) == 1.0
				@assert rand_index([1, 1, 2, 2], [2, 2, 1, 1]) == 1.0
				@assert isapprox(rand_index([1, 1, 2, 2], [1, 2, 1, 2]),
				                 1 / 3; atol = 1e-10)
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: Star Graph K_{1,4} — Freeman Centralization Should Be 1.0
			println("\nTest 4: freeman_degree_centralization on star graph K_{1,4}")
			try
				star = DataFrame(src = [1, 1, 1, 1], dst = [2, 3, 4, 5])
				star_cent = freeman_degree_centralization(star; directed = false)
				@assert isapprox(star_cent, 1.0; atol = 1e-10) "Got $star_cent"
				println("  PASSED (= $star_cent)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Complete Graph K_4 — Freeman Centralization Should Be 0.0
			println("\nTest 5: freeman_degree_centralization on complete graph K_4")
			try
				k4 = DataFrame(src = [1, 1, 1, 2, 2, 3], dst = [2, 3, 4, 3, 4, 4])
				k4_cent = freeman_degree_centralization(k4; directed = false)
				@assert isapprox(k4_cent, 0.0; atol = 1e-10) "Got $k4_cent"
				println("  PASSED (= $k4_cent)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: in_degree and out_degree on a Tiny Directed Graph
		#	Edges: 1→2, 2→3, 3→1, 1→3
		#	Expected: in_degree = [1, 1, 2], out_degree = [2, 1, 1] when sorted by node
			println("\nTest 6: in_degree / out_degree on directed triangle")
			try
				triangle = DataFrame(src = [1, 2, 3, 1], dst = [2, 3, 1, 3])
				in_df  = sort(in_degree(triangle;  weighted = false), :node)
				out_df = sort(out_degree(triangle; weighted = false), :node)
				@assert in_df.in_degree   == [1.0, 1.0, 2.0] "in_degree = $(in_df.in_degree)"
				@assert out_df.out_degree == [2.0, 1.0, 1.0] "out_degree = $(out_df.out_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: total_degree on the Same Triangle
			println("\nTest 7: total_degree on directed triangle")
			try
				triangle = DataFrame(src = [1, 2, 3, 1], dst = [2, 3, 1, 3])
				tot_df = sort(total_degree(triangle;
				                          weighted = false,
				                          directed = true), :node)
				@assert tot_df.total_degree == [3.0, 2.0, 3.0] "total_degree = $(tot_df.total_degree)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 8: Smoke Test on Moreno High School (Weighted Directed)
			println("\nTest 8: Smoke test on Moreno High School (weighted, directed)")
			try
				edges = networks["moreno_highschool_weighted"].edges

				#	Compute Each Measure
					in_deg  = in_degree(edges;  weighted = true)
					out_deg = out_degree(edges; weighted = true)
					tot_deg = total_degree(edges; weighted = true, directed = true)
					gini_val = gini_coefficient(total_degree(edges;
					                                        weighted = false,
					                                        directed = true).total_degree)
					freeman_val = freeman_degree_centralization(edges;
					                                           mode = :all,
					                                           directed = true)
					smm_val = centralization(tot_deg.total_degree)

				#	Report
					println("  in_degree:                          mean = $(round(mean(in_deg.in_degree), digits = 3))")
					println("  out_degree:                         mean = $(round(mean(out_deg.out_degree), digits = 3))")
					println("  total_degree:                       mean = $(round(mean(tot_deg.total_degree), digits = 3))," *
					        " sd = $(round(std(tot_deg.total_degree), digits = 3))")
					println("  Gini of total_degree (binary):      $(round(gini_val, digits = 3))")
					println("  Freeman centralization (binary):    $(round(freeman_val, digits = 3))")
					println("  SMM centralization (SD, weighted):  $(round(smm_val, digits = 3))")
					println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 9: Smoke Test on Scotland Interlock (Unweighted Undirected)
			println("\nTest 9: Smoke test on Scotland Interlock (unweighted, undirected)")
			try
				scotland_edges = networks["scotland_interlock_unweighted"].edges
				scot_tot = total_degree(scotland_edges;
				                       weighted = false,
				                       directed = false)
				gini_val = gini_coefficient(scot_tot.total_degree)
				freeman_val = freeman_degree_centralization(scotland_edges;
				                                           directed = false)

				println("  N = $(nrow(scot_tot)), mean degree = $(round(mean(scot_tot.total_degree), digits = 3))")
				println("  Gini of degree:          $(round(gini_val, digits = 3))")
				println("  Freeman centralization:  $(round(freeman_val, digits = 3))")
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Foundation tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_network_statistics_foundation_tests(networks)

#	Run Path-Based Centrality Tests on the Synthetic 8-Node Network
	function run_synthetic_path_centrality_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Verifies closeness_centrality, betweenness_centrality, and
			mean_inverse_distance on the 8-node synthetic network with
			hand-computed expected values, then exercises the weighted-path
			dispatch (:tie_strength, :distance) on a small varied-weight
			directed path with hand-computed weighted distances.

			Topology recap (foundation test network):
				1 → 2, 1 → 3, 2 → 3, 3 → 2     (component A: 2-out hub + mutual)
				4 → 5, 5 → 6, 6 → 4              (component B: directed triangle)
				7, 8                              (isolates)

			Expected closeness (normalized, symmetric on max(A, A^T), N=8):
				Each of nodes 1..6 sees two neighbors at distance 1 in its
				component, so sum_{j != i} 1/d = 2 and closeness = 2/7.
				Nodes 7, 8 score 0.
				Expected: [2/7, 2/7, 2/7, 2/7, 2/7, 2/7, 0, 0]

			Expected closeness (normalized, out, directed, N=8):
				Node 1: reaches 2 and 3 at d=1 → 2/7
				Node 2: reaches only 3 at d=1 → 1/7
				Node 3: reaches only 2 at d=1 → 1/7
				Nodes 4, 5, 6: each reaches the other two of the triangle at
					d=1 and d=2 → (1 + 0.5) / 7 = 1.5/7
				Nodes 7, 8: 0
				Expected: [2/7, 1/7, 1/7, 1.5/7, 1.5/7, 1.5/7, 0, 0]

			Expected betweenness (directed, unnormalized):
				Triangle component: each node sits on exactly one length-2
				geodesic → B = 1. Mutual dyad: no intermediates. Isolates: 0.
				Expected: [0, 0, 0, 1, 1, 1, 0, 0]

			Expected betweenness (undirected/symmetric, unnormalized):
				Once symmetrized, component B becomes a 3-clique. Every pair
				at distance 1; no intermediates. All zeros.
				Expected: [0, 0, 0, 0, 0, 0, 0, 0]

			Expected mean_inverse_distance:
				Directed out: 8.5/56 ≈ 0.15179, scaled by log(8) ≈ 0.07299
				Symmetric:   12/56 ≈ 0.21429, scaled by log(8) ≈ 0.10305

			Weighted-path fixture (tests 12-15): directed path 1→2→3→4
			with weights [2.0, 4.0, 1.0] on edges (1,2), (2,3), (3,4).
				Under :tie_strength (cost = 1/w):
					1→2: cost 0.5
					2→3: cost 0.25
					3→4: cost 1.0
					Shortest distances from 1: d(1,2)=0.5, d(1,3)=0.75, d(1,4)=1.75
					Sum 1/d from node 1 = 1/0.5 + 1/0.75 + 1/1.75
					                    = 2 + 1.3333... + 0.5714...
					                    ≈ 3.9048
				Under :distance (cost = w):
					Shortest distances from 1: d(1,2)=2, d(1,3)=6, d(1,4)=7
					Sum 1/d from node 1 = 1/2 + 1/6 + 1/7 = 0.5 + 0.1667 + 0.1429
					                    ≈ 0.8095
				Under :ignore (BFS hop counts):
					Shortest distances from 1: d(1,2)=1, d(1,3)=2, d(1,4)=3
					Sum 1/d from node 1 = 1 + 0.5 + 1/3 ≈ 1.8333
		"""

		println("=" ^ 70)
		println("Synthetic path-centrality tests for network_statistics.jl")
		println("=" ^ 70)
		println("\nNetwork: 8 nodes, 7 directed edges, 2 isolates")

		all_passed = true

		#	Build Test Network
			net = _build_foundation_test_network()
			edges = net.edges
			nodes = net.nodes

		#	Test 1: Closeness Centrality, Symmetric Direction, with Isolates
			println("\nTest 1: closeness_centrality (symmetric direction, with isolates)")
			try
				cc = sort(closeness_centrality(edges;
				                              nodes = nodes,
				                              directed = true,
				                              direction = :symmetric,
				                              edge_interpretation = :ignore,
				                              normalize = true), :node)
				expected = [2/7, 2/7, 2/7, 2/7, 2/7, 2/7, 0.0, 0.0]
				@assert nrow(cc) == 8 "Expected 8 rows, got $(nrow(cc))"
				@assert isapprox(cc.closeness, expected; atol = 1e-10) "Got: $(cc.closeness)"
				println("  PASSED (each connected = 2/7 ≈ 0.2857, isolates = 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: Closeness Centrality, Out Direction
			println("\nTest 2: closeness_centrality (out direction, with isolates)")
			try
				cc_out = sort(closeness_centrality(edges;
				                                  nodes = nodes,
				                                  directed = true,
				                                  direction = :out,
				                                  edge_interpretation = :ignore,
				                                  normalize = true), :node)
				expected = [2/7, 1/7, 1/7, 1.5/7, 1.5/7, 1.5/7, 0.0, 0.0]
				@assert isapprox(cc_out.closeness, expected; atol = 1e-10) "Got: $(cc_out.closeness)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Closeness Centrality, In Direction
			println("\nTest 3: closeness_centrality (in direction, with isolates)")
			try
				cc_in = sort(closeness_centrality(edges;
				                                 nodes = nodes,
				                                 directed = true,
				                                 direction = :in,
				                                 edge_interpretation = :ignore,
				                                 normalize = true), :node)
				expected = [0.0, 2/7, 2/7, 1.5/7, 1.5/7, 1.5/7, 0.0, 0.0]
				@assert isapprox(cc_in.closeness, expected; atol = 1e-10) "Got: $(cc_in.closeness)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: Closeness Without Isolates (Backward Compatibility)
			println("\nTest 4: closeness_centrality without nodes argument (6 rows)")
			try
				cc_partial = closeness_centrality(edges;
				                                 directed = true,
				                                 direction = :symmetric,
				                                 edge_interpretation = :ignore,
				                                 normalize = true)
				@assert nrow(cc_partial) == 6 "Expected 6 rows (no isolates), got $(nrow(cc_partial))"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Closeness Centrality, Unnormalized
			println("\nTest 5: closeness_centrality (symmetric, unnormalized)")
			try
				cc_raw = sort(closeness_centrality(edges;
				                                  nodes = nodes,
				                                  directed = true,
				                                  direction = :symmetric,
				                                  edge_interpretation = :ignore,
				                                  normalize = false), :node)
				expected = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
				@assert isapprox(cc_raw.closeness, expected; atol = 1e-10) "Got: $(cc_raw.closeness)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: Betweenness Centrality (Directed)
			println("\nTest 6: betweenness_centrality (directed, unnormalized)")
			try
				bc = sort(betweenness_centrality(edges;
				                                nodes = nodes,
				                                directed = true,
				                                edge_interpretation = :ignore,
				                                normalize = false), :node)
				expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
				@assert nrow(bc) == 8
				@assert isapprox(bc.betweenness, expected; atol = 1e-10) "Got: $(bc.betweenness)"
				println("  PASSED (triangle component → 1 each; rest → 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: Betweenness Centrality (Undirected)
			println("\nTest 7: betweenness_centrality (undirected, unnormalized)")
			try
				bc_undir = sort(betweenness_centrality(edges;
				                                      nodes = nodes,
				                                      directed = false,
				                                      edge_interpretation = :ignore,
				                                      normalize = false), :node)
				expected = zeros(Float64, 8)
				@assert isapprox(bc_undir.betweenness, expected; atol = 1e-10) "Got: $(bc_undir.betweenness)"
				println("  PASSED (symmetrization makes both components 3-cliques, all zero)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 8: Betweenness Normalized to [0, 1]
			println("\nTest 8: betweenness_centrality (directed, normalized)")
			try
				bc_norm = sort(betweenness_centrality(edges;
				                                     nodes = nodes,
				                                     directed = true,
				                                     edge_interpretation = :ignore,
				                                     normalize = true), :node)
				expected = [0.0, 0.0, 0.0, 1/42, 1/42, 1/42, 0.0, 0.0]
				@assert isapprox(bc_norm.betweenness, expected; atol = 1e-12) "Got: $(bc_norm.betweenness)"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 9: Mean Inverse Distance (Directed, Out, Scaled by Log N)
			println("\nTest 9: mean_inverse_distance (directed, out, scaled by log N)")
			try
				mid = mean_inverse_distance(edges;
				                           nodes = nodes,
				                           directed = true,
				                           direction = :out,
				                           edge_interpretation = :ignore,
				                           scale_by_log_n = true)
				expected = 8.5 / 56 / log(8)
				@assert isapprox(mid, expected; atol = 1e-10) "Got: $mid, expected $expected"
				println("  PASSED (= $(round(mid, digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 10: Mean Inverse Distance (Symmetric, Scaled)
			println("\nTest 10: mean_inverse_distance (symmetric, scaled by log N)")
			try
				mid_sym = mean_inverse_distance(edges;
				                               nodes = nodes,
				                               directed = true,
				                               direction = :symmetric,
				                               edge_interpretation = :ignore,
				                               scale_by_log_n = true)
				expected = 12.0 / 56 / log(8)
				@assert isapprox(mid_sym, expected; atol = 1e-10) "Got: $mid_sym, expected $expected"
				println("  PASSED (= $(round(mid_sym, digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 11: Mean Inverse Distance, Unscaled
			println("\nTest 11: mean_inverse_distance (symmetric, NOT scaled)")
			try
				mid_unscaled = mean_inverse_distance(edges;
				                                    nodes = nodes,
				                                    directed = true,
				                                    direction = :symmetric,
				                                    edge_interpretation = :ignore,
				                                    scale_by_log_n = false)
				expected = 12.0 / 56
				@assert isapprox(mid_unscaled, expected; atol = 1e-10) "Got: $mid_unscaled"
				println("  PASSED (= $(round(mid_unscaled, digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	---------------------------------------------------------------
		#	Weighted-Path Tests (Tests 12-15)
		#	---------------------------------------------------------------
		#	Use a small varied-weight directed path: 1→2 (w=2), 2→3 (w=4),
		#	3→4 (w=1). Hand-computed weighted distances under each
		#	edge_interpretation mode. The fixture is inline so it doesn't
		#	affect other tests that might use the foundation network.

		#	Build Varied-Weight Path Fixture
			weighted_edges = DataFrame(src    = ["1", "2", "3"],
			                          dst    = ["2", "3", "4"],
			                          weight = [2.0, 4.0, 1.0])
			weighted_nodes = DataFrame(id    = string.(1:4),
			                          label = string.(1:4))

		#	Test 12: Closeness with edge_interpretation=:tie_strength (Default)
			println("\nTest 12: closeness_centrality (:tie_strength on weighted path)")
			try
				cc_ts = sort(closeness_centrality(weighted_edges;
				                                 nodes = weighted_nodes,
				                                 directed = true,
				                                 direction = :out,
				                                 edge_interpretation = :tie_strength,
				                                 normalize = true), :node)
				#	From node 1 (out direction): reaches 2 (cost 0.5), 3 (cost 0.75), 4 (cost 1.75)
				#	Sum 1/d = 1/0.5 + 1/0.75 + 1/1.75 = 2 + 4/3 + 4/7 = (84+56+24)/42 = 164/42
				#	Normalized: 164/42 / 3 = 164/126 ≈ 1.3016
					exp_node1 = (1.0/0.5 + 1.0/0.75 + 1.0/1.75) / 3
				#	From node 2: reaches 3 (cost 0.25), 4 (cost 1.25)
				#	Sum 1/d = 4 + 0.8 = 4.8, normalized = 4.8/3 = 1.6
					exp_node2 = (1.0/0.25 + 1.0/1.25) / 3
				#	From node 3: reaches 4 (cost 1.0)
				#	Sum 1/d = 1, normalized = 1/3
					exp_node3 = 1.0 / 3
				#	From node 4: reaches nothing → 0
					exp_node4 = 0.0
					expected  = [exp_node1, exp_node2, exp_node3, exp_node4]
					@assert isapprox(cc_ts.closeness, expected; atol = 1e-10) "Got: $(cc_ts.closeness), Expected: $expected"
				println("  PASSED (node 1 closeness = $(round(cc_ts.closeness[1], digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 13: Closeness with edge_interpretation=:distance
			println("\nTest 13: closeness_centrality (:distance on weighted path)")
			try
				cc_dist = sort(closeness_centrality(weighted_edges;
				                                   nodes = weighted_nodes,
				                                   directed = true,
				                                   direction = :out,
				                                   edge_interpretation = :distance,
				                                   normalize = true), :node)
				#	From node 1 (out): reaches 2 (cost 2), 3 (cost 6), 4 (cost 7)
				#	Sum 1/d = 0.5 + 1/6 + 1/7, normalized / 3
					exp_node1 = (1.0/2.0 + 1.0/6.0 + 1.0/7.0) / 3
				#	From node 2: reaches 3 (cost 4), 4 (cost 5)
					exp_node2 = (1.0/4.0 + 1.0/5.0) / 3
				#	From node 3: reaches 4 (cost 1)
					exp_node3 = 1.0 / 3
				#	From node 4: 0
					exp_node4 = 0.0
					expected  = [exp_node1, exp_node2, exp_node3, exp_node4]
					@assert isapprox(cc_dist.closeness, expected; atol = 1e-10) "Got: $(cc_dist.closeness), Expected: $expected"
				println("  PASSED (node 1 closeness = $(round(cc_dist.closeness[1], digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 14: Three Modes Produce Different Results on Same Graph
			println("\nTest 14: :ignore, :tie_strength, :distance produce distinct results")
			try
				#	Compute Closeness Under All Three Modes
					cc_ign  = sort(closeness_centrality(weighted_edges;
					                                   nodes = weighted_nodes,
					                                   directed = true,
					                                   direction = :out,
					                                   edge_interpretation = :ignore), :node)
					cc_ts2  = sort(closeness_centrality(weighted_edges;
					                                   nodes = weighted_nodes,
					                                   directed = true,
					                                   direction = :out,
					                                   edge_interpretation = :tie_strength), :node)
					cc_dst2 = sort(closeness_centrality(weighted_edges;
					                                   nodes = weighted_nodes,
					                                   directed = true,
					                                   direction = :out,
					                                   edge_interpretation = :distance), :node)
				#	Node 1's Closeness Should Differ Across All Three Modes
					ign_v1 = cc_ign.closeness[1]
					ts_v1  = cc_ts2.closeness[1]
					dst_v1 = cc_dst2.closeness[1]
					@assert !isapprox(ign_v1, ts_v1; atol = 1e-6) ":ignore and :tie_strength produced same closeness for node 1: $ign_v1"
					@assert !isapprox(ign_v1, dst_v1; atol = 1e-6) ":ignore and :distance produced same closeness for node 1: $ign_v1"
					@assert !isapprox(ts_v1, dst_v1; atol = 1e-6) ":tie_strength and :distance produced same closeness for node 1: $ts_v1"
				println("  PASSED (:ignore=$(round(ign_v1, digits=4)), :tie_strength=$(round(ts_v1, digits=4)), :distance=$(round(dst_v1, digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 15: Mean Inverse Distance Under Weighted Modes
			println("\nTest 15: mean_inverse_distance under all three modes")
			try
				#	:ignore on Weighted Path → BFS Hop Counts
				#	Ordered pairs in path of length 4: (1,2)d=1, (1,3)d=2, (1,4)d=3,
				#	(2,3)d=1, (2,4)d=2, (3,4)d=1. Sum 1/d = 1 + 0.5 + 1/3 + 1 + 0.5 + 1 = 4.333...
				#	Mean = 4.333.../(4*3) = 0.3611..., scaled by log(4) = 1.3863 → 0.2604...
					mid_ign = mean_inverse_distance(weighted_edges;
					                              nodes = weighted_nodes,
					                              directed = true,
					                              direction = :out,
					                              edge_interpretation = :ignore,
					                              scale_by_log_n = false)
					expected_ign = (1.0 + 0.5 + 1.0/3 + 1.0 + 0.5 + 1.0) / (4 * 3)
					@assert isapprox(mid_ign, expected_ign; atol = 1e-10) ":ignore MID got $mid_ign, expected $expected_ign"

				#	:tie_strength → Dijkstra on 1/w; same ordered pairs, weighted distances
				#	Costs: 1→2 = 0.5, 2→3 = 0.25, 3→4 = 1.0
				#	d(1,2) = 0.5, d(1,3) = 0.75, d(1,4) = 1.75
				#	d(2,3) = 0.25, d(2,4) = 1.25
				#	d(3,4) = 1.0
				#	Sum 1/d = 2 + 4/3 + 4/7 + 4 + 0.8 + 1
					mid_ts = mean_inverse_distance(weighted_edges;
					                              nodes = weighted_nodes,
					                              directed = true,
					                              direction = :out,
					                              edge_interpretation = :tie_strength,
					                              scale_by_log_n = false)
					expected_ts = (1.0/0.5 + 1.0/0.75 + 1.0/1.75 + 1.0/0.25 + 1.0/1.25 + 1.0/1.0) / (4 * 3)
					@assert isapprox(mid_ts, expected_ts; atol = 1e-10) ":tie_strength MID got $mid_ts, expected $expected_ts"

				#	:distance → Dijkstra on raw weights
				#	d(1,2) = 2, d(1,3) = 6, d(1,4) = 7, d(2,3) = 4, d(2,4) = 5, d(3,4) = 1
					mid_dist = mean_inverse_distance(weighted_edges;
					                                nodes = weighted_nodes,
					                                directed = true,
					                                direction = :out,
					                                edge_interpretation = :distance,
					                                scale_by_log_n = false)
					expected_dist = (1.0/2.0 + 1.0/6.0 + 1.0/7.0 + 1.0/4.0 + 1.0/5.0 + 1.0/1.0) / (4 * 3)
					@assert isapprox(mid_dist, expected_dist; atol = 1e-10) ":distance MID got $mid_dist, expected $expected_dist"

				#	All Three Should Differ
					@assert !isapprox(mid_ign, mid_ts; atol = 1e-6) ":ignore and :tie_strength MID matched: $mid_ign"
					@assert !isapprox(mid_ign, mid_dist; atol = 1e-6) ":ignore and :distance MID matched"

				println("  PASSED (:ignore=$(round(mid_ign, digits=4)), :tie_strength=$(round(mid_ts, digits=4)), :distance=$(round(mid_dist, digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 16: Betweenness Still Errors on Weighted Modes (Deferred)
			println("\nTest 16: betweenness_centrality(:tie_strength) raises ArgumentError (deferred)")
			try
				try
					_ = betweenness_centrality(weighted_edges;
					                          nodes = weighted_nodes,
					                          directed = true,
					                          edge_interpretation = :tie_strength)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError && occursin("not yet implemented", e2.msg)
						println("  PASSED (ArgumentError for deferred weighted Brandes)")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Test 17: Unknown edge_interpretation Mode Raises ArgumentError
			println("\nTest 17: edge_interpretation=:bogus raises ArgumentError")
			try
				try
					_ = closeness_centrality(edges;
					                        nodes = nodes,
					                        edge_interpretation = :bogus)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED (ArgumentError for unrecognized mode)")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic path-centrality tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
	run_synthetic_path_centrality_tests()

#   Moreno Tests
	function generate_path_measures_csv(network_name::String,
	                                    networks::Dict,
	                                    output_dir::String;
	                                    direction::Symbol = :symmetric,
	                                    directed_path::Bool = true)
		"""
		Args:
			network_name::String: key into the networks dictionary
				(e.g., "moreno_highschool_weighted")
			networks::Dict: dictionary of loaded GraphML networks
			output_dir::String: directory where CSV files will be written
			direction::Symbol: :symmetric (default), :out, or :in. Controls
				the closeness and MID computations. Betweenness uses the
				explicit `directed`/`undirected` calls below regardless.
			directed_path::Bool: whether to treat the input as directed when
				computing path measures (default = true). Set to false to
				use the undirected branches of the centrality functions.
		Returns:
			NamedTuple: (per_node_path::String, summary_path::String)
				file paths of the two CSVs written
		Notes:
			Computes closeness_centrality (symmetric direction by default),
			betweenness_centrality (both directed and undirected), and
			mean_inverse_distance (scaled and unscaled by log N) for the
			specified network. Writes two CSVs in `output_dir`:

				<network_name>_julia_path_measures.csv
				<network_name>_julia_summary.csv

			The per-node CSV schema matches the igraph reference produced by
			compute_igraph_moreno_reference.py, enabling direct line-by-line
			comparison.

			Pre-allocates the summary DataFrame columns rather than pushing
			rows individually, in keeping with the project's style convention
			for building small heterogeneous tables.

			All path computations use edge_interpretation=:ignore (binarize
			before BFS). Weights in the edges DataFrame are ignored in this
			Phase 0 pipeline.
		"""

		#	Validation
			if !haskey(networks, network_name)
				throw(ArgumentError("Network '$network_name' not found in networks dict"))
			end
			if !isdir(output_dir)
				mkpath(output_dir)
			end

		#	Extract Edges and Nodes from the Loaded Network
			edges  = networks[network_name].edges
			nodes  = networks[network_name].nodes
			n_nodes = nrow(nodes)
			n_edges = nrow(edges)

			println("Computing path-centrality measures on $network_name " *
			        "(N = $n_nodes, E = $n_edges)...")

		#	Compute Closeness (Apples-to-Apples with igraph Harmonic Centrality)
			cc = closeness_centrality(edges;
			                         nodes = nodes,
			                         directed = directed_path,
			                         direction = direction,
			                         edge_interpretation = :ignore,
			                         normalize = true)

		#	Compute Betweenness (Directed, Unnormalized)
			bc_directed = betweenness_centrality(edges;
			                                    nodes = nodes,
			                                    directed = true,
			                                    edge_interpretation = :ignore,
			                                    normalize = false)

		#	Compute Betweenness (Undirected, Unnormalized)
			bc_undirected = betweenness_centrality(edges;
			                                      nodes = nodes,
			                                      directed = false,
			                                      edge_interpretation = :ignore,
			                                      normalize = false)

		#	Compute Mean Inverse Distance (Symmetric Direction)
			mid_scaled = mean_inverse_distance(edges;
			                                  nodes = nodes,
			                                  directed = directed_path,
			                                  direction = direction,
			                                  edge_interpretation = :ignore,
			                                  scale_by_log_n = true)
			mid_unscaled = mean_inverse_distance(edges;
			                                    nodes = nodes,
			                                    directed = directed_path,
			                                    direction = direction,
			                                    edge_interpretation = :ignore,
			                                    scale_by_log_n = false)

		#	Sort Each Result by Node for Predictable Diff Order
			cc_sorted   = sort(cc, :node)
			bc_d_sorted = sort(bc_directed, :node)
			bc_u_sorted = sort(bc_undirected, :node)

		#	Sanity Check: All Three Results Should Be in the Same Node Order
			@assert cc_sorted.node == bc_d_sorted.node == bc_u_sorted.node "Node ordering inconsistent across measures"

		#	Build Per-Node DataFrame
			out = DataFrame(
				node                   = cc_sorted.node,
				closeness_harmonic_sym = cc_sorted.closeness,
				betweenness_directed   = bc_d_sorted.betweenness,
				betweenness_undirected = bc_u_sorted.betweenness
			)

		#	Sort by Numeric Order of Node Label (Treats "1".."70" as Integers)
			out.sort_key = parse.(Int, string.(out.node))
			sort!(out, :sort_key)
			select!(out, Not(:sort_key))

		#	Build Summary DataFrame Directly (Five Heterogeneous Rows)
		#	Three vector-valued measures (mean, sd, max) and two scalar measures
		#	(single value). Pre-allocating column vectors rather than push!-ing
		#	rows matches the project style convention.
			summary_out = DataFrame(
				measure = ["closeness_harmonic_sym",
				           "betweenness_directed",
				           "betweenness_undirected",
				           "mean_inverse_distance_scaled",
				           "mean_inverse_distance_unscaled"],
				mean = Union{Missing, Float64}[
				           mean(out.closeness_harmonic_sym),
				           mean(out.betweenness_directed),
				           mean(out.betweenness_undirected),
				           missing,
				           missing],
				sd = Union{Missing, Float64}[
				           std(out.closeness_harmonic_sym),
				           std(out.betweenness_directed),
				           std(out.betweenness_undirected),
				           missing,
				           missing],
				max = Union{Missing, Float64}[
				           maximum(out.closeness_harmonic_sym),
				           maximum(out.betweenness_directed),
				           maximum(out.betweenness_undirected),
				           missing,
				           missing],
				value = Union{Missing, Float64}[
				           missing,
				           missing,
				           missing,
				           mid_scaled,
				           mid_unscaled]
			)

		#	Write CSVs
			per_node_path = joinpath(output_dir, "$(network_name)_julia_path_measures.csv")
			summary_path  = joinpath(output_dir, "$(network_name)_julia_summary.csv")
			CSV.write(per_node_path, out)
			CSV.write(summary_path, summary_out)

		#	Print Summary to Console
			println("\nJulia summary statistics for $network_name:")
			for row in eachrow(summary_out)
				if ismissing(row.value)
					println("  $(row.measure): mean=$(round(row.mean, digits = 6)), " *
					        "sd=$(round(row.sd, digits = 6)), " *
					        "max=$(round(row.max, digits = 6))")
				else
					println("  $(row.measure): $(round(row.value, digits = 6))")
				end
			end

			println("\nWrote per-node CSV to: $per_node_path")
			println("Wrote summary CSV to:  $summary_path")

		#	Return File Paths
			return (per_node_path = per_node_path, summary_path = summary_path)
	end

    generate_path_measures_csv("moreno_highschool_weighted",
                              networks,
                              "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data";
                              direction = :symmetric,
                              directed_path = true)

#	Helper Function for run_synthetic_bonacich_tests: Build the Star K_{1,3}
	function _build_star_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for K_{1,3} — node 1 is the hub,
				nodes 2, 3, 4 are leaves.
		Notes:
			Stored as a directed edge list (1 → 2, 1 → 3, 1 → 4). When the
			Bonacich call symmetrizes, the result is the undirected star.

			Hand-computed Bonacich values (β = 0.5 / sqrt(3)):
				λ_max(K_{1,3}) = sqrt(3)
				β = 0.5 / sqrt(3) ≈ 0.288675
				c_hub = 3 (β + 1) / (1 - 3 β^2) ≈ 5.154701
				c_leaf = β · c_hub + 1            ≈ 2.488034
		"""

		#	Edges (Directed Hub-to-Leaves; Symmetrization Inside the Function
		#	Will Make It Undirected)
			edges = DataFrame(
				src    = ["1", "1", "1"],
				dst    = ["2", "3", "4"],
				weight = [1.0, 1.0, 1.0]
			)

		#	Nodes
			nodes = DataFrame(
				id    = string.(1:4),
				label = string.(1:4)
			)

		#	Metadata
			metadata = (
				network_name  = "Star K_{1,3} Test",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = false,
				n_nodes       = 4,
				n_edges       = 3
			)

		#	Return Triple
			return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Run Bonacich Centrality Tests
	function run_synthetic_bonacich_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests bonacich_centrality on two hand-computable cases:

			1. The 8-node foundation-test graph, symmetrized.
			   Both components become K_3 triangles, so every connected
			   node has degree 2. With β = 0.5/λ_max = 0.25:
				 Bonacich = [4, 4, 4, 4, 4, 4, 0, 0]
			   This verifies isolate handling and the linear-solve path.

			2. The 4-node star K_{1,3}.
			   Hub has degree 3, leaves have degree 1. With β = 0.5/sqrt(3):
				 c_hub  ≈ 5.154701
				 c_leaf ≈ 2.488034
			   This verifies that asymmetric structure produces asymmetric
			   centrality and that the power iteration computes the right
			   λ_max (= sqrt(3) for K_{1,3}).
		"""

		println("=" ^ 70)
		println("Synthetic Bonacich centrality tests for network_statistics.jl")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: Two K_3 Components + Two Isolates (Foundation Test Network)
			println("\nTest 1: bonacich_centrality on 2x K_3 + 2 isolates (symmetric, default β)")
			try
				net = _build_foundation_test_network()
				bc = sort(bonacich_centrality(net.edges;
				                             nodes = net.nodes,
				                             directed = true,
				                             direction = :symmetric,
				                             edge_interpretation = :ignore,
				                             normalize = :none), :node)

				#	Expected: λ_max = 2, β = 0.25, c = 4 for connected, 0 for isolates
					expected_bonacich = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0]
					expected_beta    = 0.25

				@assert nrow(bc) == 8 "Expected 8 rows, got $(nrow(bc))"
				@assert isapprox(bc.bonacich, expected_bonacich; atol = 1e-8) "Got: $(bc.bonacich)"
				@assert isapprox(bc.beta_used[1], expected_beta; atol = 1e-8) "Got β = $(bc.beta_used[1])"
				println("  PASSED (all connected = 4.0, isolates = 0.0, β = 0.25)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: Star K_{1,3} — Asymmetric Centrality
			println("\nTest 2: bonacich_centrality on star K_{1,3} (hand-computed values)")
			try
				net = _build_star_test_network()
				bc = sort(bonacich_centrality(net.edges;
				                             nodes = net.nodes,
				                             directed = true,
				                             direction = :symmetric,
				                             edge_interpretation = :ignore,
				                             normalize = :none), :node)

				expected_hub  = 3.0 * (0.5 / sqrt(3.0) + 1.0) / (1.0 - 3.0 * (0.5 / sqrt(3.0))^2)
				expected_leaf = (0.5 / sqrt(3.0)) * expected_hub + 1.0
				expected_beta = 0.5 / sqrt(3.0)

				#	Sort Order: Nodes Are "1", "2", "3", "4". Bonacich Values
				#	After Sort: hub at index 1, three leaves at indices 2..4.
					@assert nrow(bc) == 4
					@assert isapprox(bc.bonacich[1], expected_hub;  atol = 1e-8) "Hub: expected $expected_hub, got $(bc.bonacich[1])"
					@assert isapprox(bc.bonacich[2], expected_leaf; atol = 1e-8) "Leaf 2: expected $expected_leaf, got $(bc.bonacich[2])"
					@assert isapprox(bc.bonacich[3], expected_leaf; atol = 1e-8) "Leaf 3: expected $expected_leaf, got $(bc.bonacich[3])"
					@assert isapprox(bc.bonacich[4], expected_leaf; atol = 1e-8) "Leaf 4: expected $expected_leaf, got $(bc.bonacich[4])"
					@assert isapprox(bc.beta_used[1], expected_beta; atol = 1e-8) "β: expected $expected_beta, got $(bc.beta_used[1])"
				println("  PASSED (hub ≈ $(round(expected_hub, digits=4)), " *
				        "leaves ≈ $(round(expected_leaf, digits=4)), β ≈ $(round(expected_beta, digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Explicit β Override
			println("\nTest 3: bonacich_centrality with explicit β = 0.0 (should equal degree)")
			try
				net = _build_star_test_network()
				bc = sort(bonacich_centrality(net.edges;
				                             nodes = net.nodes,
				                             directed = true,
				                             direction = :symmetric,
				                             beta = 0.0,
				                             edge_interpretation = :ignore,
				                             normalize = :none), :node)

				#	With β = 0: c = A · 1 = degree vector
				#	Star K_{1,3}: hub degree 3, leaves degree 1 each
					expected = [3.0, 1.0, 1.0, 1.0]
					@assert isapprox(bc.bonacich, expected; atol = 1e-10) "Got: $(bc.bonacich)"
					@assert bc.beta_used[1] == 0.0 "Got β = $(bc.beta_used[1])"
				println("  PASSED (β = 0 reduces Bonacich to degree)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: Backward Compatibility (No `nodes`, So No Isolate Rows)
			println("\nTest 4: bonacich_centrality without nodes argument (legacy 6 rows)")
			try
				net = _build_foundation_test_network()
				bc = bonacich_centrality(net.edges;
				                        directed = true,
				                        direction = :symmetric,
				                        edge_interpretation = :ignore,
				                        normalize = :none)
				@assert nrow(bc) == 6 "Expected 6 rows (no isolates), got $(nrow(bc))"
				println("  PASSED")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: L2 Normalization
			println("\nTest 5: bonacich_centrality with normalize=:l2 (RMS = 1)")
			try
				net = _build_foundation_test_network()
				bc = bonacich_centrality(net.edges;
				                        nodes = net.nodes,
				                        directed = true,
				                        direction = :symmetric,
				                        edge_interpretation = :ignore,
				                        normalize = :l2)
				#	Check RMS Is 1
					rms = sqrt(sum(bc.bonacich .^ 2) / nrow(bc))
					@assert isapprox(rms, 1.0; atol = 1e-10) "Expected RMS = 1, got $rms"
				println("  PASSED (RMS = $(round(rms, digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: Max Normalization
			println("\nTest 6: bonacich_centrality with normalize=:max (peak = 1)")
			try
				net = _build_star_test_network()
				bc = sort(bonacich_centrality(net.edges;
				                             nodes = net.nodes,
				                             directed = true,
				                             direction = :symmetric,
				                             edge_interpretation = :ignore,
				                             normalize = :max), :node)
				#	Check Max Is 1.0 (Hub Should Be the Max)
					@assert isapprox(maximum(bc.bonacich), 1.0; atol = 1e-10) "Expected max = 1, got $(maximum(bc.bonacich))"
					#	Leaves Should Be Their Original Value Divided by Hub Value
						expected_hub  = 3.0 * (0.5 / sqrt(3.0) + 1.0) / (1.0 - 3.0 * (0.5 / sqrt(3.0))^2)
						expected_leaf = (0.5 / sqrt(3.0)) * expected_hub + 1.0
						expected_ratio = expected_leaf / expected_hub
						@assert isapprox(bc.bonacich[2], expected_ratio; atol = 1e-10) "Got leaf = $(bc.bonacich[2]), expected $expected_ratio"
				println("  PASSED (hub = 1.0, leaves = $(round(bc.bonacich[2], digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: edge_interpretation Unsupported Mode Raises ArgumentError
			println("\nTest 7: edge_interpretation=:tie_strength raises ArgumentError")
			try
				net = _build_foundation_test_network()
				try
					_ = bonacich_centrality(net.edges;
					                       nodes = net.nodes,
					                       edge_interpretation = :tie_strength)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Test 8: Invalid normalize Symbol Raises ArgumentError
			println("\nTest 8: normalize=:invalid raises ArgumentError")
			try
				net = _build_foundation_test_network()
				try
					_ = bonacich_centrality(net.edges;
					                       nodes = net.nodes,
					                       normalize = :invalid)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic Bonacich tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_bonacich_tests()

#   Bonacich Moreno Test
    function generate_bonacich_csv(network_name::String,
	                               networks::Dict,
	                               output_dir::String;
	                               direction::Symbol = :symmetric,
	                               directed_path::Bool = true)
		"""
		Args:
			network_name::String: key into the networks dictionary
				(e.g., "moreno_highschool_weighted")
			networks::Dict: dictionary of loaded GraphML networks
			output_dir::String: directory where CSV files will be written
			direction::Symbol: :symmetric | :out | :in (default = :symmetric)
			directed_path::Bool: whether to treat input as directed (default = true)
		Returns:
			NamedTuple: (per_node_path::String, summary_path::String)
		Notes:
			Computes bonacich_centrality on the specified network using the
			Phase 0 defaults (spectral β, no normalization, edge_interpretation
			binarized). Writes two CSVs in `output_dir`:

				<network_name>_julia_bonacich.csv
				<network_name>_julia_bonacich_summary.csv

			Schema matches the numpy reference produced by
			compute_bonacich_reference.py, enabling line-by-line comparison.
		"""

		#	Validation
			if !haskey(networks, network_name)
				throw(ArgumentError("Network '$network_name' not found in networks dict"))
			end
			if !isdir(output_dir)
				mkpath(output_dir)
			end

		#	Extract Edges and Nodes
			edges  = networks[network_name].edges
			nodes  = networks[network_name].nodes
			n_nodes = nrow(nodes)
			n_edges = nrow(edges)

			println("Computing Bonacich centrality on $network_name " *
			        "(N = $n_nodes, E = $n_edges)...")

		#	Compute Bonacich (Default: Symmetric, Spectral β, No Normalization)
			bc = bonacich_centrality(edges;
			                        nodes = nodes,
			                        directed = directed_path,
			                        direction = direction,
			                        edge_interpretation = :ignore,
			                        normalize = :none)

		#	Sort by Numeric Node Label
			bc.sort_key = parse.(Int, string.(bc.node))
			sort!(bc, :sort_key)
			select!(bc, Not(:sort_key))

		#	Build Per-Node DataFrame (Strip beta_used Column for CSV;
		#	It Is Constant Across Rows and Goes in the Summary)
			out = DataFrame(
				node     = bc.node,
				bonacich = bc.bonacich
			)

		#	Build Summary DataFrame Directly (Pre-Allocated Columns)
			beta_used = bc.beta_used[1]
			summary_out = DataFrame(
				measure = ["beta_used",
				           "bonacich_mean",
				           "bonacich_sd",
				           "bonacich_max",
				           "bonacich_min"],
				value = [beta_used,
				         mean(out.bonacich),
				         std(out.bonacich),
				         maximum(out.bonacich),
				         minimum(out.bonacich)]
			)

		#	Write CSVs
			per_node_path = joinpath(output_dir, "$(network_name)_julia_bonacich.csv")
			summary_path  = joinpath(output_dir, "$(network_name)_julia_bonacich_summary.csv")
			CSV.write(per_node_path, out)
			CSV.write(summary_path, summary_out)

		#	Print Summary
			println("\nJulia Bonacich summary for $network_name:")
			println("  β used:        $(round(beta_used, digits = 10))")
			println("  bonacich mean: $(round(mean(out.bonacich), digits = 6))")
			println("  bonacich sd:   $(round(std(out.bonacich), digits = 6))")
			println("  bonacich max:  $(round(maximum(out.bonacich), digits = 6))")
			println("  bonacich min:  $(round(minimum(out.bonacich), digits = 6))")

			println("\nWrote per-node CSV to: $per_node_path")
			println("Wrote summary CSV to:  $summary_path")

		#	Return Paths
			return (per_node_path = per_node_path, summary_path = summary_path)
	end
    generate_bonacich_csv("moreno_highschool_weighted",
                     networks,
                     "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data")

#	Helper Function for run_synthetic_topology_tests: Build a Tiny Triangle Network
	function _build_triangle_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for a 4-node directed graph with
				one closed triangle (1 → 2 → 3 → 1) plus an extra isolate (4).
		Notes:
			Used to verify clustering and triad census on a known hand-computed
			structure. The triangle is the unique 030C triad (cyclic).
		"""

		#	Edges: Directed Triangle 1 → 2 → 3 → 1
			edges = DataFrame(
				src    = ["1", "2", "3"],
				dst    = ["2", "3", "1"],
				weight = [1.0, 1.0, 1.0]
			)

		#	Nodes: Include One Isolate (4)
			nodes = DataFrame(
				id    = string.(1:4),
				label = string.(1:4)
			)

		#	Metadata
			metadata = (
				network_name  = "Directed Triangle + Isolate",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = false,
				n_nodes       = 4,
				n_edges       = 3
			)

		#	Return Triple
			return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Run Synthetic Topology Tests
	function run_synthetic_topology_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests the seven public Section 5 functions on hand-computable cases:
				1.  largest_component_proportion on K_3 + isolate (= 0.75)
				2.  largest_component_proportion on 8-node foundation graph (= 3/8)
				3.  reciprocity (arc-based) on directed triangle (= 0.0)
				4.  reciprocity on K_3 symmetrized (would be 1.0 if we mutualized)
				5.  local_weighted_reciprocity on partially reciprocated dyads
				6.  local_clustering_coefficient on directed triangle (transitivity = 0)
				7.  global_clustering_coefficient :average on undirected K_3
				    (each node clusters at 1.0; average = 1.0)
				8.  global_clustering_coefficient :transitivity on directed triangle
				    (no non-vacuous triplets centered at any node; ratio undefined → 0.0)
				9.  triad_census on directed triangle: exactly one 030C, rest accounted for
				10. triad_census undirected: only 003, 102, 201, 300 are non-zero
				11. triad_census weighted=true throws ArgumentError
		"""

		println("=" ^ 70)
		println("Synthetic topology tests for network_statistics.jl Section 5")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: largest_component_proportion on K_3 + Isolate
			println("\nTest 1: largest_component_proportion on K_3 + isolate (= 0.75)")
			try
				net = _build_triangle_test_network()
				p = largest_component_proportion(net.edges; nodes = net.nodes)
				@assert isapprox(p, 0.75; atol = 1e-12) "Expected 0.75, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: largest_component_proportion on 8-Node Foundation Network
			println("\nTest 2: largest_component_proportion on foundation network (= 3/8)")
			try
				net = _build_foundation_test_network()
				p = largest_component_proportion(net.edges; nodes = net.nodes)
				#	The foundation network has two 3-node components + 2 isolates,
				#	so the largest component has 3 nodes out of 8 total.
					@assert isapprox(p, 3/8; atol = 1e-12) "Expected 3/8 = 0.375, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: reciprocity (arc-based) on Directed Triangle Is 0.0
			println("\nTest 3: reciprocity (arc-based) on directed triangle 1→2→3→1 (= 0.0)")
			try
				net = _build_triangle_test_network()
				r = reciprocity(net.edges; mode = :arc_based)
				@assert isapprox(r, 0.0; atol = 1e-12) "Expected 0.0, got $r"
				println("  PASSED (= $r)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: reciprocity on Fully Mutualized Triangle = 1.0
			println("\nTest 4: reciprocity (arc-based) on mutualized triangle (= 1.0)")
			try
				#	Build a fully reciprocated triangle: 1↔2, 2↔3, 1↔3
					edges = DataFrame(
						src = ["1", "2", "2", "3", "1", "3"],
						dst = ["2", "1", "3", "2", "3", "1"],
						weight = ones(Float64, 6)
					)
					r = reciprocity(edges; mode = :arc_based)
					@assert isapprox(r, 1.0; atol = 1e-12) "Expected 1.0, got $r"
					println("  PASSED (= $r)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: local_weighted_reciprocity on Partially Reciprocated Dyad
			println("\nTest 5: local_weighted_reciprocity (Squartini per-node)")
			try
				#	Two-Node Dyad with Asymmetric Weights: w(1→2) = 3, w(2→1) = 1
					edges = DataFrame(
						src = ["1", "2"],
						dst = ["2", "1"],
						weight = [3.0, 1.0]
					)
					df = sort(local_weighted_reciprocity(edges; weighted = true), :node)
					#	r_1 = min(3, 1) / 3 = 1/3
					#	r_2 = min(1, 3) / 1 = 1
					@assert isapprox(df.r[1], 1/3; atol = 1e-10) "node 1: expected 1/3, got $(df.r[1])"
					@assert isapprox(df.r[2], 1.0; atol = 1e-10) "node 2: expected 1.0, got $(df.r[2])"
					println("  PASSED (r_1 = $(round(df.r[1], digits=4)), r_2 = $(round(df.r[2], digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: local_clustering_coefficient (:local_transitivity) on Directed Triangle
			println("\nTest 6: local_clustering_coefficient :local_transitivity on triangle")
			try
				net = _build_triangle_test_network()
				df = sort(local_clustering_coefficient(net.edges;
				                                      directed = true,
				                                      method = :local_transitivity), :node)
				#	Each node in 1→2→3→1 has exactly one non-vacuous triplet
				#	centered at it (j → i → k, with j, k the predecessor/successor),
				#	and that triplet IS closed (j → k goes the other way around the cycle).
				#	So local transitivity = 1/1 = 1.0 for each of nodes 1, 2, 3.
				#	Wait — re-check: for node 1, in_neighbors = {3}, out_neighbors = {2}.
				#	Triplet 3 → 1 → 2, closed iff 3 → 2 exists. We have 3 → 1 and 2 → 3,
				#	so 3 → 2 does NOT exist. So local transitivity is 0 / 2 = 0 for each node.
					@assert isapprox(df.local_clustering_coefficient[1], 0.0; atol = 1e-12) "node 1: expected 0.0, got $(df.local_clustering_coefficient[1])"
					@assert isapprox(df.local_clustering_coefficient[2], 0.0; atol = 1e-12) "node 2: expected 0.0, got $(df.local_clustering_coefficient[2])"
					@assert isapprox(df.local_clustering_coefficient[3], 0.0; atol = 1e-12) "node 3: expected 0.0, got $(df.local_clustering_coefficient[3])"
					println("  PASSED (cyclic triangle has no transitive triplets; all = 0.0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: global_clustering_coefficient :average on Undirected K_3
			println("\nTest 7: global_clustering_coefficient :average on undirected K_3 (= 1.0)")
			try
				#	K_3 As an Undirected Graph (Edges Listed Once, Function Symmetrizes)
					edges = DataFrame(
						src = ["1", "2", "3"],
						dst = ["2", "3", "1"],
						weight = ones(Float64, 3)
					)
					gcc = global_clustering_coefficient(edges;
					                                   directed = false,
					                                   method = :average,
					                                   average_mode = :local_clustering)
					@assert isapprox(gcc, 1.0; atol = 1e-10) "Expected 1.0, got $gcc"
					println("  PASSED (= $gcc)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 8: global_clustering_coefficient :transitivity on Directed Triangle
			println("\nTest 8: global_clustering_coefficient :transitivity on directed triangle")
			try
				net = _build_triangle_test_network()
				gcc = global_clustering_coefficient(net.edges;
				                                   directed = true,
				                                   method = :transitivity)
				#	Each node contributes 1 non-vacuous triplet, none closed → 0/3 = 0
					@assert isapprox(gcc, 0.0; atol = 1e-12) "Expected 0.0, got $gcc"
					println("  PASSED (= $gcc)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 9: triad_census on Directed Triangle Should Have Exactly One 030C
			println("\nTest 9: triad_census on directed triangle (one 030C, rest in 003)")
			try
				net = _build_triangle_test_network()
				tc = triad_census(net.edges; nodes = net.nodes, graph_type = :directed)
				#	With N = 4, total triads = C(4,3) = 4.
				#	The cyclic triangle on {1,2,3} is a 030C.
				#	Each of the 3 triads involving node 4 (the isolate) is a 012 or similar.
				#	Specifically: triads (1,2,4), (1,3,4), (2,3,4) each contain exactly
				#	one directed dyad — they fall into class 012.
					counts = tc.count
					labels = tc.triad
					idx_030C = findfirst(==("030C"), labels)
					idx_012  = findfirst(==("012"), labels)
					idx_003  = findfirst(==("003"), labels)

					@assert counts[idx_030C] == 1 "Expected one 030C, got $(counts[idx_030C])"
					@assert counts[idx_012] == 3 "Expected three 012, got $(counts[idx_012])"
					@assert counts[idx_003] == 0 "Expected zero 003, got $(counts[idx_003])"
					@assert sum(counts) == 4 "Total should be C(4,3) = 4, got $(sum(counts))"
					println("  PASSED (030C=1, 012=3, total=4)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 10: triad_census Undirected on Symmetrized Triangle
			println("\nTest 10: triad_census undirected on K_3 + isolate (one 300, three 102)")
			try
				net = _build_triangle_test_network()
				tc = triad_census(net.edges; nodes = net.nodes, graph_type = :undirected)
				#	Symmetrized: K_3 on {1,2,3} is a complete undirected triangle (300).
				#	Triads involving node 4: each has exactly one edge (102) - wait,
				#	actually each triad with the isolate has the K_3 edge between the
				#	two non-isolated nodes, which is 1 edge → 102.
					counts = tc.count
					labels = tc.triad
					idx_300 = findfirst(==("300"), labels)
					idx_201 = findfirst(==("201"), labels)
					idx_102 = findfirst(==("102"), labels)
					idx_003 = findfirst(==("003"), labels)

					@assert counts[idx_300] == 1 "Expected one 300, got $(counts[idx_300])"
					@assert counts[idx_102] == 3 "Expected three 102, got $(counts[idx_102])"
					@assert counts[idx_201] == 0 "Expected zero 201, got $(counts[idx_201])"
					@assert counts[idx_003] == 0 "Expected zero 003, got $(counts[idx_003])"
					#	Sanity Check: Only Undirected Classes Non-Zero
						directed_only_indices = [i for (i, lbl) in enumerate(labels)
						                         if !(lbl in ("003", "102", "201", "300"))]
						for di in directed_only_indices
							@assert counts[di] == 0 "Class $(labels[di]) should be zero for undirected, got $(counts[di])"
						end
					println("  PASSED (300=1, 102=3, all other classes = 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 11: triad_census weighted=true Raises ArgumentError
			println("\nTest 11: triad_census weighted=true raises ArgumentError")
			try
				net = _build_triangle_test_network()
				try
					_ = triad_census(net.edges; weighted = true)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic topology tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_topology_tests()

#	Helper Function for run_synthetic_bicomponent_tests: Build the Bowtie Graph
	function _build_bowtie_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for a 5-node bowtie graph:
				two triangles sharing vertex 3. Edges 1-2, 1-3, 2-3 form the
				first triangle; edges 3-4, 3-5, 4-5 form the second. Vertex 3
				is the unique articulation point.
		Notes:
			Stored as a directed edge list (with each undirected edge appearing
			as one directed arc, in canonical order). The bicomponent measure
			symmetrizes before computing, so direction doesn't affect output.

			Hand-computed bicomponents:
				{1, 2, 3} — first triangle
				{3, 4, 5} — second triangle
			Largest bicomponent size = 3, N = 5, proportion = 0.6.
		"""

		#	Edges (Stored Directed; Symmetrization Happens Inside the Function)
			edges = DataFrame(
				src    = ["1", "1", "2", "3", "3", "4"],
				dst    = ["2", "3", "3", "4", "5", "5"],
				weight = ones(Float64, 6)
			)

		#	Nodes
			nodes = DataFrame(
				id    = string.(1:5),
				label = string.(1:5)
			)

		#	Metadata
			metadata = (
				network_name  = "Bowtie Test",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = false,
				n_nodes       = 5,
				n_edges       = 6
			)

		#	Return Triple
			return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Helper Function for run_synthetic_bicomponent_tests: Build a 5-Node Path
	function _build_path_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for a 5-node path 1-2-3-4-5.
		Notes:
			Path graphs are a classic test for bicomponent algorithms because
			every edge is its own bicomponent. There are 4 bicomponents of size
			2 each: {1,2}, {2,3}, {3,4}, {4,5}. Articulation points are 2, 3, 4.
			Largest bicomponent size = 2, N = 5, proportion = 0.4.
		"""

		edges = DataFrame(
			src    = ["1", "2", "3", "4"],
			dst    = ["2", "3", "4", "5"],
			weight = ones(Float64, 4)
		)

		nodes = DataFrame(
			id    = string.(1:5),
			label = string.(1:5)
		)

		metadata = (
			network_name  = "Path-5 Test",
			source_format = "synthetic_test",
			directed      = true,
			weighted      = false,
			n_nodes       = 5,
			n_edges       = 4
		)

		return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Run Synthetic Bicomponent Tests
	function run_synthetic_bicomponent_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests largest_bicomponent_proportion on six hand-computable cases
			whose expected values were cross-checked against networkx's
			biconnected_components:

			1. Two K_3 triangles + 2 isolates → largest = 3, N = 8 → 3/8 = 0.375
			2. 5-node path 1-2-3-4-5 → four bicomps of size 2 → 2/5 = 0.4
			3. Bowtie (two triangles sharing vertex 3) → two bicomps of size 3 → 3/5 = 0.6
			4. 4-cycle (square) → single bicomp of size 4 → 4/4 = 1.0
			5. Single edge + isolate → bicomps {1,2} and {3} → 2/3
			6. Single K_3 triangle (no isolates) → single bicomp of size 3 → 1.0
			7. Empty edge list with `nodes` argument → 1/N (each isolate alone)
		"""

		println("=" ^ 70)
		println("Synthetic bicomponent tests for network_statistics.jl Section 6")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: Two K_3 Triangles + 2 Isolates → 3/8
			println("\nTest 1: Two K_3 + 2 isolates (foundation network) → 3/8 = 0.375")
			try
				net = _build_foundation_test_network()
				p = largest_bicomponent_proportion(net.edges; nodes = net.nodes)
				@assert isapprox(p, 3/8; atol = 1e-12) "Expected 0.375, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: 5-Node Path → 2/5
			println("\nTest 2: 5-node path 1-2-3-4-5 → 2/5 = 0.4")
			try
				net = _build_path_test_network()
				p = largest_bicomponent_proportion(net.edges; nodes = net.nodes)
				@assert isapprox(p, 0.4; atol = 1e-12) "Expected 0.4, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Bowtie → 3/5
			println("\nTest 3: Bowtie (two triangles sharing vertex 3) → 3/5 = 0.6")
			try
				net = _build_bowtie_test_network()
				p = largest_bicomponent_proportion(net.edges; nodes = net.nodes)
				@assert isapprox(p, 0.6; atol = 1e-12) "Expected 0.6, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: 4-Cycle (Square) → 1.0
			println("\nTest 4: 4-cycle 1-2-3-4-1 → 4/4 = 1.0")
			try
				edges = DataFrame(
					src = ["1", "2", "3", "4"],
					dst = ["2", "3", "4", "1"],
					weight = ones(Float64, 4)
				)
				nodes = DataFrame(id = string.(1:4), label = string.(1:4))
				p = largest_bicomponent_proportion(edges; nodes = nodes)
				@assert isapprox(p, 1.0; atol = 1e-12) "Expected 1.0, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Single Edge + Isolate → 2/3
			println("\nTest 5: Single edge (1-2) + isolate (3) → 2/3 ≈ 0.6667")
			try
				edges = DataFrame(
					src = ["1"],
					dst = ["2"],
					weight = [1.0]
				)
				nodes = DataFrame(id = string.(1:3), label = string.(1:3))
				p = largest_bicomponent_proportion(edges; nodes = nodes)
				@assert isapprox(p, 2/3; atol = 1e-12) "Expected 2/3, got $p"
				println("  PASSED (= $(round(p, digits=6)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: Single K_3 Triangle (No Isolates) → 1.0
			println("\nTest 6: Single K_3 triangle (no isolates) → 3/3 = 1.0")
			try
				edges = DataFrame(
					src = ["1", "1", "2"],
					dst = ["2", "3", "3"],
					weight = ones(Float64, 3)
				)
				nodes = DataFrame(id = string.(1:3), label = string.(1:3))
				p = largest_bicomponent_proportion(edges; nodes = nodes)
				@assert isapprox(p, 1.0; atol = 1e-12) "Expected 1.0, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: Empty Edge List with `nodes` → 1/N (Each Isolate)
			println("\nTest 7: Empty edge list with nodes (N=5) → 1/5 = 0.2")
			try
				edges = DataFrame(src = String[], dst = String[], weight = Float64[])
				nodes = DataFrame(id = string.(1:5), label = string.(1:5))
				p = largest_bicomponent_proportion(edges; nodes = nodes)
				@assert isapprox(p, 0.2; atol = 1e-12) "Expected 0.2, got $p"
				println("  PASSED (= $p)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic bicomponent tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_bicomponent_tests()

#	Run Synthetic Tau Statistic Tests
	function run_synthetic_tau_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests tau_statistic on hand-derivable cases plus a Monte Carlo
			reproducibility check.

			1. Empty graph (no edges, N = 4): U|MAN distribution is degenerate
			   (only the all-null graph); tau should be 0.0 with null_sd = 0.0.
			2. Directed triangle + isolate (the case verified against Python):
			   M = 0, A = 3, N_null = 3. Observed weighted count under RC = 0
			   (the cyclic triangle is 030C, which has RC weight 0; the three
			   012 triads involving the isolate also have RC weight 0). Under
			   the U|MAN null with these dyads, the expected weighted count is
			   roughly 1.25 with SD around 0.96, so tau should be around -1.3.
			   Tolerance allows for Monte Carlo noise at n_samples = 500.
			3. Reproducibility: same seed produces identical tau across runs.
			4. Different seeds produce different tau but within Monte Carlo noise.
			5. Custom 16-vector weighting passes validation and runs.
			6. weighting=:transitivity gives different (broader) permitted set
			   and so a different tau than :RC.
			7. directed=false raises ArgumentError.
			8. Custom weighting vector of wrong length raises ArgumentError.
			9. Invalid weighting symbol raises ArgumentError.
		"""

		println("=" ^ 70)
		println("Synthetic tau statistic tests for network_statistics.jl Section 7")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: Empty Edge List → Degenerate U|MAN, tau = 0
			println("\nTest 1: Empty graph (N=4, no edges) → tau = 0.0, null_sd = 0.0")
			try
				edges = DataFrame(src = String[], dst = String[], weight = Float64[])
				nodes = DataFrame(id = string.(1:4), label = string.(1:4))
				result = tau_statistic(edges; nodes = nodes, n_samples = 100, seed = 20260101)
				@assert nrow(result) == 1 "Expected 1-row result, got $(nrow(result))"
				@assert isapprox(result.tau[1], 0.0; atol = 1e-12) "Expected tau = 0.0, got $(result.tau[1])"
				@assert isapprox(result.null_sd[1], 0.0; atol = 1e-12) "Expected null_sd = 0.0, got $(result.null_sd[1])"
				@assert result.M[1] == 0 "Expected M = 0, got $(result.M[1])"
				@assert result.A[1] == 0 "Expected A = 0, got $(result.A[1])"
				println("  PASSED (tau = 0.0 as expected for degenerate U|MAN)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: Directed Triangle + Isolate → tau ≈ -1.3 (Hand Computed via Python)
			println("\nTest 2: Directed triangle 1→2→3→1 + isolate 4 → tau ≈ -1.3")
			try
				net = _build_triangle_test_network()
				result = tau_statistic(net.edges;
				                      nodes = net.nodes,
				                      weighting = :RC,
				                      n_samples = 500,
				                      seed = 20260101)
				#	Verify Dyad Census
					@assert result.M[1] == 0 "Expected M = 0, got $(result.M[1])"
					@assert result.A[1] == 3 "Expected A = 3, got $(result.A[1])"
					@assert result.N_null[1] == 3 "Expected N_null = 3, got $(result.N_null[1])"
				#	Verify Observed Weighted Count = 0 (Cyclic Triangle Has 030C, RC=0)
					@assert isapprox(result.observed_weighted_count[1], 0.0; atol = 1e-10) "Expected obs = 0.0, got $(result.observed_weighted_count[1])"
				#	Verify Tau in the Expected Range
				#	Hand-computed via Python MC: tau ≈ -1.3 ± noise
				#	Use a wide tolerance because n_samples = 500 means non-trivial MC noise
				#	on a small (N=4) network.
					@assert -2.0 < result.tau[1] < -0.5 "tau = $(result.tau[1]) outside expected range (-2.0, -0.5)"
				println("  PASSED (tau = $(round(result.tau[1], digits=4)), within expected range)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Reproducibility — Same Seed Gives Identical Tau
			println("\nTest 3: Reproducibility (same seed → identical tau)")
			try
				net = _build_triangle_test_network()
				r1 = tau_statistic(net.edges; nodes = net.nodes, n_samples = 200, seed = 42)
				r2 = tau_statistic(net.edges; nodes = net.nodes, n_samples = 200, seed = 42)
				@assert r1.tau[1] == r2.tau[1] "Same seed produced different tau: $(r1.tau[1]) vs $(r2.tau[1])"
				println("  PASSED (tau = $(round(r1.tau[1], digits=6)) reproducible across runs)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: Different Seeds Give Different Tau (Within MC Noise)
			println("\nTest 4: Different seeds give different tau (but similar magnitude)")
			try
				net = _build_triangle_test_network()
				r1 = tau_statistic(net.edges; nodes = net.nodes, n_samples = 500, seed = 42)
				r2 = tau_statistic(net.edges; nodes = net.nodes, n_samples = 500, seed = 99)
				@assert r1.tau[1] != r2.tau[1] "Different seeds gave identical tau (suspicious)"
				#	But Both Should Be in the Same Ballpark (the True tau Is Around -1.3
				#	on This Tiny Network with K=500, MC SE ~ 0.05)
					diff = abs(r1.tau[1] - r2.tau[1])
					@assert diff < 0.5 "Tau values differ too much: $(r1.tau[1]) vs $(r2.tau[1]) (diff = $diff)"
				println("  PASSED (tau1 = $(round(r1.tau[1], digits=4)), tau2 = $(round(r2.tau[1], digits=4)), |diff| = $(round(diff, digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Custom 16-Vector Weighting
			println("\nTest 5: Custom weighting vector (count only 300 triads)")
			try
				net = _build_triangle_test_network()
				#	Only Count 300 Triads (Fully Mutual Triangles)
					custom_lambda = zeros(16)
					custom_lambda[16] = 1.0  # 300
					result = tau_statistic(net.edges;
					                      nodes = net.nodes,
					                      weighting = custom_lambda,
					                      n_samples = 200,
					                      seed = 20260101)
				#	The Triangle 1→2→3→1 + Isolate Has Zero 300 Triads (No Mutuals)
					@assert isapprox(result.observed_weighted_count[1], 0.0; atol = 1e-10) "Expected obs = 0.0, got $(result.observed_weighted_count[1])"
				#	With M = 0, No U|MAN Sample Can Have a 300 Triad Either
				#	(300 requires 3 mutual dyads). So the null is degenerate at 0.
					@assert isapprox(result.null_mean[1], 0.0; atol = 1e-10) "Expected null_mean = 0.0, got $(result.null_mean[1])"
					@assert isapprox(result.null_sd[1], 0.0; atol = 1e-10) "Expected null_sd = 0.0, got $(result.null_sd[1])"
					@assert result.weighting[1] == "custom" "Expected weighting label = 'custom', got '$(result.weighting[1])'"
				println("  PASSED (custom weighting validated; null degenerate at 0 since no mutuals possible)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: :transitivity Weighting Differs from :RC
			println("\nTest 6: :transitivity weighting gives different tau than :RC")
			try
				net = _build_triangle_test_network()
				#	On the Triangle + Isolate Graph: under :RC, observed = 0;
				#	under :transitivity, observed weighted count includes 012 triads.
				#	The 3 triads involving the isolate are 012, with weight 1
				#	in :transitivity but 0 in :RC.
					r_rc = tau_statistic(net.edges; nodes = net.nodes, weighting = :RC, n_samples = 500, seed = 20260101)
					r_tr = tau_statistic(net.edges; nodes = net.nodes, weighting = :transitivity, n_samples = 500, seed = 20260101)
					@assert r_rc.observed_weighted_count[1] != r_tr.observed_weighted_count[1] "RC and transitivity gave same observed count"
					@assert r_tr.observed_weighted_count[1] == 3.0 "Expected 3 (three 012 triads with isolate), got $(r_tr.observed_weighted_count[1])"
					println("  PASSED (obs_RC = $(r_rc.observed_weighted_count[1]), obs_trans = $(r_tr.observed_weighted_count[1]))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: directed=false Raises ArgumentError
			println("\nTest 7: directed=false raises ArgumentError")
			try
				net = _build_triangle_test_network()
				try
					_ = tau_statistic(net.edges; nodes = net.nodes, directed = false)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Test 8: Custom Weighting Vector of Wrong Length Raises ArgumentError
			println("\nTest 8: Wrong-length custom weighting raises ArgumentError")
			try
				net = _build_triangle_test_network()
				try
					_ = tau_statistic(net.edges;
					                 nodes = net.nodes,
					                 weighting = [1.0, 2.0, 3.0])  # wrong length
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Test 9: Invalid Weighting Symbol Raises ArgumentError
			println("\nTest 9: Invalid weighting symbol raises ArgumentError")
			try
				net = _build_triangle_test_network()
				try
					_ = tau_statistic(net.edges; nodes = net.nodes, weighting = :nonsense)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic tau tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_tau_tests()

#	Test: Threaded vs Serial Layered Triad Census Must Match Bit-Exactly
	function test_threaded_vs_serial_layered(edges::DataFrame;
												nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
												graph_type::Symbol=:directed,
												reciprocity_collapse::Bool=false,
												L::Int=20,
												tau_min::Union{Symbol,Float64}=:auto,
												tau_max::Union{Symbol,Float64}=:auto,
												label::String="<unnamed>",
												show_progress::Bool=true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			nodes, graph_type, reciprocity_collapse, L, tau_min, tau_max:
				forwarded to _triad_census_layered
			label::String: descriptive name for diagnostic output
			show_progress::Bool: display a per-τ progress bar inside each of the
				serial and threaded runs (default true). On small/fast graphs
				the bars complete instantly and add no useful information; on
				large graphs (especially Marvel-scale undirected, where the
				kernel is O(N^3) per τ) the bars are essential for visibility
				into a multi-minute run.
		Returns:
			NamedTuple: (passed::Bool, serial_count_sum::Int, threaded_count_sum::Int,
			             ntau::Int, mismatches::Vector{NamedTuple},
			             t_serial::Float64, t_threaded::Float64)
		Notes:
			Runs the layered census twice — once with parallel=false, once with
			parallel=true — and compares the per-τ count tables row-by-row.
			Any mismatch reports the τ index, triad label, and the two count
			values. Bit-exactness is required: threaded scheduling must not
			perturb any count.

			This test exercises the bit-equality determinism contract documented
			on _triad_census_layered.

			Progress bars are labeled with [serial] and [threaded] suffixes so
			the two runs can be distinguished when they appear in sequence.
		"""

		println("=" ^ 70)
		println("Threaded vs Serial Layered Triad Census — $label")
		println("Graph: $(nrow(edges)) edges, graph_type=$graph_type, collapse=$reciprocity_collapse, L=$L")
		println("Threads available: $(Threads.nthreads())")
		println("=" ^ 70)

		#	Run Serial Reference
			println("  Running serial reference...")
			t0 = time()
			ref = _triad_census_layered(edges;
								nodes                = nodes,
								graph_type           = graph_type,
								reciprocity_collapse = reciprocity_collapse,
								L                    = L,
								tau_min              = tau_min,
								tau_max              = tau_max,
								parallel             = false,
								show_progress        = show_progress,
								progress_desc        = "  [serial] $label")
			t_serial = time() - t0
			println(@sprintf("  Serial done (%.2f s)", t_serial))

		#	Run Threaded
			println("  Running threaded version...")
			t0 = time()
			thr = _triad_census_layered(edges;
								nodes                = nodes,
								graph_type           = graph_type,
								reciprocity_collapse = reciprocity_collapse,
								L                    = L,
								tau_min              = tau_min,
								tau_max              = tau_max,
								parallel             = true,
								show_progress        = show_progress,
								progress_desc        = "  [threaded] $label")
			t_threaded = time() - t0
			println(@sprintf("  Threaded done (%.2f s)", t_threaded))

		#	Compare Per-τ Tables
			pt_ref = ref.per_tau
			pt_thr = thr.per_tau
			@assert nrow(pt_ref) == nrow(pt_thr) "row count mismatch: serial $(nrow(pt_ref)), threaded $(nrow(pt_thr))"

			mismatches = NamedTuple[]
			for i in 1:nrow(pt_ref)
				if pt_ref.count[i] != pt_thr.count[i] || !isapprox(pt_ref.tau[i], pt_thr.tau[i]; atol = 0.0, rtol = 0.0)
					push!(mismatches, (row     = i,
										tau     = pt_ref.tau[i],
										triad   = pt_ref.triad[i],
										serial  = pt_ref.count[i],
										threaded = pt_thr.count[i]))
				end
			end

		#	Report
			passed = isempty(mismatches)
			println()
			println("  Serial total count sum:   ", sum(pt_ref.count))
			println("  Threaded total count sum: ", sum(pt_thr.count))
			println("  τ grid points:            ", thr.meta.L)
			println("  Threads actually used:    ", thr.meta.n_threads_used)
			if t_serial > 0
				speedup = t_serial / t_threaded
				println(@sprintf("  Speedup:                  %.2fx (serial %.2f s, threaded %.2f s)",
									speedup, t_serial, t_threaded))
			end
			if passed
				println("  Bit-exact equality:       PASS")
			else
				println("  Bit-exact equality:       FAIL ($(length(mismatches)) mismatches)")
				println("  First 5 mismatches:")
				for m in first(mismatches, 5)
					println("    row=$(m.row) tau=$(round(m.tau, sigdigits=4)) triad=$(m.triad) serial=$(m.serial) threaded=$(m.threaded)")
				end
			end
			println("=" ^ 70)

		return (passed             = passed,
				serial_count_sum   = sum(pt_ref.count),
				threaded_count_sum = sum(pt_thr.count),
				ntau               = thr.meta.L,
				mismatches         = mismatches,
				t_serial           = t_serial,
				t_threaded         = t_threaded)
	end

#	Test: Binary Triad Census vs Pajek Reference (Balikatan Directed w/ Reciprocity Collapse)
	function test_pajek_reference_triad_census(edges::DataFrame,
												observed_vec_path::String;
												nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
												label::String="<unnamed>")
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (weights ignored — binary path)
			observed_vec_path::String: path to a Pajek .vec file of observed counts
				(expected format: "*Vertices 16" header followed by 16 numeric rows
				in Davis-Leinhardt order)
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			label::String: descriptive name for diagnostic output
		Returns:
			NamedTuple: (passed::Bool, total_delta::Float64, per_triad::DataFrame)
		Notes:
			Runs the binary directed-with-reciprocity-collapse census (the only
			variant the existing fixture covers) and compares against Pajek's
			observed counts. Total delta should be exactly zero on a correct run.

			The reciprocity_collapse=true path emits non-zero counts only in
			{003, 102, 201, 300} — the four undirected classes — and zeros
			everywhere else. This is the expected Pajek behavior.
		"""

		println("=" ^ 70)
		println("Pajek Fixture Test — $label")
		println("=" ^ 70)

		triad_types = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
		               "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

		#	Load Pajek Observed Counts
			obs_lines = readlines(observed_vec_path)
			#	First line is "*Vertices 16" header; skip it
				observed_counts = parse.(Float64, obs_lines[2:end])
			@assert length(observed_counts) == 16 ".vec file must contain 16 count rows after the header"

		#	Run Census (Directed, Reciprocity-Collapsed, Binary)
			print("  Running binary directed-collapse census... ")
			t0 = time()
			result = triad_census(edges;
									nodes                = nodes,
									weighted             = false,
									graph_type           = :directed,
									reciprocity_collapse = true)
			t_elapsed = time() - t0
			println(@sprintf("done (%.2f s)", t_elapsed))

		#	Align by Triad Label and Compute Per-Class Delta
			per_triad = DataFrame(triad           = triad_types,
									pajek_observed = observed_counts,
									our_count      = Float64.(result.count))
			per_triad.delta = per_triad.pajek_observed .- per_triad.our_count
			total_delta = sum(per_triad.delta)

		#	Report
			println()
			println("  Per-class comparison (Pajek observed vs ours):")
			for r in eachrow(per_triad)
				marker = abs(r.delta) < 0.5 ? "OK" : "** MISMATCH **"
				println(@sprintf("    %-5s  pajek=%-15.0f  ours=%-15.0f  delta=%-15.0f  %s",
									r.triad, r.pajek_observed, r.our_count, r.delta, marker))
			end
			passed = isapprox(total_delta, 0.0; atol = 0.5)
			println()
			println("  Total delta: ", total_delta)
			println("  Result:      ", passed ? "PASS" : "FAIL")
			println("=" ^ 70)

		return (passed = passed, total_delta = total_delta, per_triad = per_triad)
	end

#	Motif Test Helper: Directed Triad Motif Generator
	function _motif_triad_edges_directed(code::String; weight::Float64=1.0)
		"""
		Args:
			code::String: DL class label (e.g. "003", "012", "030T", "300")
			weight::Float64: edge weight to assign (default 1.0)
		Returns:
			DataFrame: edges DataFrame for a single triad of class `code`,
				with node ids "n1", "n2", "n3"
		Notes:
			Generates a minimal three-node directed graph whose single triad
			falls in the named DL class. Used by motif suite tests to verify
			that the census correctly classifies each motif type.

			The "D" / "U" suffix on classes 111 and 120 distinguishes the
			direction of the asymmetric arc(s) relative to the mutual edge.
			This generator follows the convention encoded by the package's
			BM kernel (verified empirically by the motif suite):
			- "D" (Down): the asymmetric arc points OUT of the dyad — its
			  endpoint inside the mutual edge is the source.
			- "U" (Up): the asymmetric arc points INTO the dyad — its endpoint
			  inside the mutual edge is the destination.

			Edge conventions per DL class (i=n1, j=n2, k=n3):
			- 003:  empty (no edges)
			- 012:  i→j
			- 102:  i↔j (mutual)
			- 021D: i→j, i→k (down — both arcs leave i)
			- 021U: j→i, k→i (up — both arcs enter i)
			- 021C: i→j→k (chain)
			- 111D: i↔j, k→i (mutual + tail entering the dyad from outside ≡ "U" — see below)
			- 111U: i↔j, i→k (mutual + tail leaving the dyad ≡ "D" — see below)
			- 030T: i→j, i→k, j→k (transitive)
			- 030C: i→j→k→i (cyclic)
			- 201:  i↔j, i↔k
			- 120D: i↔j, k→i, k→j (mutual + both asymmetric arcs target the dyad)
			- 120U: i↔j, i→k, j→k (mutual + both asymmetric arcs leave the dyad)
			- 120C: i↔j, i→k→j (cyclic on the asymmetric arcs)
			- 210:  i↔j, i↔k, j→k
			- 300:  i↔j, i↔k, j↔k (complete reciprocal)

			Note that the kernel's D/U labeling for 111 and 120 follows a
			convention that is the reverse of one common reading: "D" means
			"asymmetric tail goes away from the mutual edge" interpreted from
			the kernel's perspective is encoded here as `k→i, k→j` for 120D
			(both outside-arcs enter the mutual pair). Empirically, the motif
			suite verifies this convention is what the kernel produces.
		"""

		#	Storage for Edge Triples
			src = String[]
			dst = String[]
			wts = Float64[]

		#	Add Directed Edge
			@inline function add_dir(i::Int, j::Int)
				push!(src, "n$(i)"); push!(dst, "n$(j)"); push!(wts, weight)
			end

		#	Add Mutual (Reciprocal) Edge
			@inline function add_mut(i::Int, j::Int)
				add_dir(i, j); add_dir(j, i)
			end

		#	Build by DL Class
			if code == "003"
				#	Empty triad
					nothing
			elseif code == "012"
				add_dir(1, 2)
			elseif code == "102"
				add_mut(1, 2)
			elseif code == "021D"
				#	Down: both edges from i
					add_dir(1, 2); add_dir(1, 3)
			elseif code == "021U"
				#	Up: both edges to i
					add_dir(2, 1); add_dir(3, 1)
			elseif code == "021C"
				#	Chain: i→j→k
					add_dir(1, 2); add_dir(2, 3)
			elseif code == "111D"
				#	Mutual + tail INTO the dyad (kernel's "D" convention)
					add_mut(1, 2); add_dir(3, 1)
			elseif code == "111U"
				#	Mutual + tail OUT OF the dyad (kernel's "U" convention)
					add_mut(1, 2); add_dir(1, 3)
			elseif code == "030T"
				#	Transitive: i→j, i→k, j→k
					add_dir(1, 2); add_dir(1, 3); add_dir(2, 3)
			elseif code == "030C"
				#	Cyclic: i→j→k→i
					add_dir(1, 2); add_dir(2, 3); add_dir(3, 1)
			elseif code == "201"
				add_mut(1, 2); add_mut(1, 3)
			elseif code == "120D"
				#	Mutual + both asymmetric arcs target the dyad (kernel's "D")
					add_mut(1, 2); add_dir(3, 1); add_dir(3, 2)
			elseif code == "120U"
				#	Mutual + both asymmetric arcs leave the dyad (kernel's "U")
					add_mut(1, 2); add_dir(1, 3); add_dir(2, 3)
			elseif code == "120C"
				#	Cyclic asymmetric arcs around the mutual edge
					add_mut(1, 2); add_dir(1, 3); add_dir(3, 2)
			elseif code == "210"
				add_mut(1, 2); add_mut(1, 3); add_dir(2, 3)
			elseif code == "300"
				add_mut(1, 2); add_mut(1, 3); add_mut(2, 3)
			else
				throw(ArgumentError("Unknown directed DL class: $code"))
			end

		#	Return
			return DataFrame(; src, dst, weight = wts)
	end

#	Test: Each DL-Class Motif Yields count=1 in Its Own Class, count=0 Elsewhere
	function test_motif_suite_directed()
		"""
		Args:
			None
		Returns:
			NamedTuple: (all_passed::Bool, per_motif::Dict{String,Bool})
		Notes:
			For each of the 16 directed DL classes, builds a single-triad graph
			of that class on three nodes (n1, n2, n3), runs the binary directed
			census (without reciprocity collapse), and asserts that the resulting
			16-class count vector is the unit vector at the matching slot.

			Forces the node universe to {n1, n2, n3} for the empty (003) case,
			which would otherwise have no nodes inferable from edges.
		"""

		println("=" ^ 70)
		println("Directed Motif Suite — Each Class Should Yield count=1 in Its Own Slot")
		println("=" ^ 70)

		dl_codes = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U",
		            "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]

		per_motif = Dict{String, Bool}()
		all_passed = true

		#	Fixed Node Universe for the Empty Triad Case
			fixed_nodes = ["n1", "n2", "n3"]

		for (idx, code) in pairs(dl_codes)
			edges = _motif_triad_edges_directed(code)
			result = triad_census(edges;
									nodes                = fixed_nodes,
									weighted             = false,
									graph_type           = :directed,
									reciprocity_collapse = false)

			#	Build Expected Vector (1.0 at code's slot, 0.0 elsewhere)
				expected_counts = zeros(Int, 16)
				expected_counts[idx] = 1

			#	Compare
				match = all(result.count .== expected_counts)
				per_motif[code] = match
				if !match
					all_passed = false
					actual_nonzero = [(dl_codes[k], result.count[k]) for k in 1:16 if result.count[k] != 0]
					println("  [$idx] $code: FAIL — expected count=1 at slot $idx, got nonzero at $actual_nonzero")
				else
					println("  [$idx] $code: PASS")
				end
		end

		println()
		println("Summary: $(all_passed ? "ALL 16 PASSED" : "SOME FAILED")")
		println("=" ^ 70)

		return (all_passed = all_passed, per_motif = per_motif)
	end

#	Motif Test Helper: Undirected Triad Motif Generator
	function _motif_triad_edges_undirected(kind::Symbol; weight::Float64=1.0)
		"""
		Args:
			kind::Symbol: :empty (003), :single (102), :open (201), :triangle (300)
			weight::Float64: edge weight (default 1.0)
		Returns:
			DataFrame: edges DataFrame with paired arcs (i→j AND j→i)
		Notes:
			Simple undirected motifs returned as paired arcs to match the
			package's symmetrization-handled-upstream convention. Maps the
			four undirected motif kinds to their DL class slots:
			- :empty    → 003 (slot 1)
			- :single   → 102 (slot 3)
			- :open     → 201 (slot 11)
			- :triangle → 300 (slot 16)
		"""

		src = String[]
		dst = String[]
		wts = Float64[]

		@inline function add_und(i::Int, j::Int)
			push!(src, "n$(i)"); push!(dst, "n$(j)"); push!(wts, weight)
			push!(src, "n$(j)"); push!(dst, "n$(i)"); push!(wts, weight)
		end

		if kind === :empty
			nothing
		elseif kind === :single
			add_und(1, 2)
		elseif kind === :open
			add_und(1, 2); add_und(1, 3)
		elseif kind === :triangle
			add_und(1, 2); add_und(2, 3); add_und(1, 3)
		else
			throw(ArgumentError("Unknown undirected kind: $kind"))
		end

		return DataFrame(; src, dst, weight = wts)
	end

#	Test: Undirected Motif Suite — Four Classes, Each Yields count=1 in Its Slot
	function test_motif_suite_undirected()
		"""
		Args:
			None
		Returns:
			NamedTuple: (all_passed::Bool, per_motif::Dict{Symbol,Bool})
		Notes:
			Verifies the undirected binary BM kernel classifies each of the
			four undirected triad types correctly. Slots not in {1, 3, 11, 16}
			should always be zero for undirected output.
		"""

		println("=" ^ 70)
		println("Undirected Motif Suite — 4 Classes Should Yield count=1 in Their Slots")
		println("=" ^ 70)

		#	Map Kinds to (DL Slot Index, DL Label)
			motif_slots = Dict(:empty    => (1, "003"),
			                   :single   => (3, "102"),
			                   :open     => (11, "201"),
			                   :triangle => (16, "300"))

		per_motif  = Dict{Symbol, Bool}()
		all_passed = true

		#	Fixed Node Universe (Covers Empty Case)
			fixed_nodes = ["n1", "n2", "n3"]

		for (kind, (slot, label)) in motif_slots
			edges = _motif_triad_edges_undirected(kind)
			result = triad_census(edges;
									nodes      = fixed_nodes,
									weighted   = false,
									graph_type = :undirected)

			#	Build Expected Vector
				expected_counts = zeros(Int, 16)
				expected_counts[slot] = 1

			#	Compare
				match = all(result.count .== expected_counts)
				per_motif[kind] = match
				if !match
					all_passed = false
					nonzero = [(result.triad[k], result.count[k]) for k in 1:16 if result.count[k] != 0]
					println("  $kind ($label, slot $slot): FAIL — got $nonzero")
				else
					println("  $kind ($label, slot $slot): PASS")
				end
		end

		println()
		println("Summary: $(all_passed ? "ALL 4 PASSED" : "SOME FAILED")")
		println("=" ^ 70)

		return (all_passed = all_passed, per_motif = per_motif)
	end

#	Threaded-Only Timing: Single Layered Triad Census with Optional Auto-Recommendation
	function time_threaded_layered(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}}=nothing,
									graph_type::Symbol=:directed,
									reciprocity_collapse::Bool=false,
									auto_recommend_L::Bool=false,
									method::Symbol=:auto,
									L::Int=8,
									tau_min::Union{Symbol,Float64}=:auto,
									tau_max::Union{Symbol,Float64}=:auto,
									L_min::Int=8,
									L_max::Int=32,
									points_per_decade::Int=8,
									n_exploratory::Int=16,
									tol::Float64=1e-3,
									label::String="<unnamed>",
									show_progress::Bool=true,
									inner_show_progress::Bool=true,
									verbose::Bool=false)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, :weight
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
			graph_type::Symbol: :directed or :undirected
			reciprocity_collapse::Bool: directed-only; collapse mutual arcs (default false)
			auto_recommend_L::Bool: when true, call recommend_L first to derive
				L and τ bounds, then run the layered census at those settings.
				When false, use the explicit L, tau_min, tau_max args below.
				Default false.
			method::Symbol: passed to recommend_L when auto_recommend_L=true.
				:auto (default), :analytic, or :empirical. Ignored when
				auto_recommend_L=false.
			L::Int: τ grid size when auto_recommend_L=false (default 8)
			tau_min, tau_max::Union{Symbol,Float64}: τ bounds when
				auto_recommend_L=false (default :auto, derived from weight
				distribution)
			L_min, L_max::Int: clamps for recommend_L's L when
				auto_recommend_L=true (defaults 8, 32)
			points_per_decade::Int: heuristic for recommend_L's L computation
				when auto_recommend_L=true (default 8)
			n_exploratory::Int: log-spaced points in recommend_L's triangle
				profile when auto_recommend_L=true (default 16)
			tol::Float64: AUMC stability tolerance for recommend_L's empirical
				path only (default 1e-3). Ignored when the analytic path is
				selected.
			label::String: descriptive name for diagnostic output
			show_progress::Bool: display top-level progress bars (default true).
				For the standalone layered census this is the per-τ bar; for
				auto_recommend_L=true with empirical method it's the candidate-
				completion bar inside recommend_L.
			inner_show_progress::Bool: when auto_recommend_L=true with empirical
				method, also display per-τ progress within each candidate's
				layered census (default true). Has no effect on the analytic
				path (no inner census runs) or when auto_recommend_L=false.
			verbose::Bool: print recommend_L's derivation steps when
				auto_recommend_L=true (default false)
		Returns:
			NamedTuple: (result::NamedTuple, t_threaded::Float64,
			             n_threads_used::Int, recommendation::Union{NamedTuple,Nothing},
			             t_recommend::Float64, method_used::Symbol)
				result: the full _triad_census_layered output (per_tau, summary, meta)
				t_threaded: wall-clock for the layered census
				n_threads_used: threads used by the layered census
				recommendation: recommend_L output when auto_recommend_L=true, else nothing
				t_recommend: wall-clock for recommend_L (0.0 if auto_recommend_L=false)
				method_used: recommendation method (:analytic, :empirical, :user_supplied)
					when auto_recommend_L=true, else :none
		Notes:
			Threaded-only performance measurement. The layered census runs with
			parallel=true (τ-loop threading) and reports total wall-clock plus
			per-τ amortized cost. Bit-exactness vs the serial reference has
			been verified on Balikatan and doesn't need re-verification at
			each graph scale.

			Two usage modes:

			1. Explicit L (auto_recommend_L=false, default): run the layered
			   census at the supplied L, tau_min, tau_max. Use when you
			   already know what grid you want.

			2. Auto recommendation (auto_recommend_L=true): call recommend_L
			   first with the specified method (:auto, :analytic, or
			   :empirical), then run the census at the recommended settings.

			Method selection. When method=:auto, recommend_L's auto-selection
			rule chooses analytic or empirical based on T_max, weight span,
			and triangle decay shape. For Marvel-scale networks (T_max in
			millions) the analytic path is essentially instant; for small
			networks (T_max < 1000) the empirical path runs but completes
			quickly anyway.

			Cache reuse note. When the empirical path runs inside recommend_L,
			the layered census at L_best has already been computed and is
			available as recommendation.census. This function still re-runs
			the census for a clean wall-clock measurement; downstream callers
			wanting the cached result should use recommendation.census
			directly. When the analytic path runs, recommendation.census is
			nothing and the census in this function is the only one performed.
		"""

		println("=" ^ 70)
		println("Threaded Layered Triad Census (Timing Only) — $label")
		println("Graph: $(nrow(edges)) edges, graph_type=$graph_type, collapse=$reciprocity_collapse")
		println("Threads available: $(Threads.nthreads())")
		println("Mode: $(auto_recommend_L ? "auto-recommendation via recommend_L (method=$method)" : "explicit L=$L")")
		println("=" ^ 70)

		#	Optionally Run recommend_L to Derive L and τ Bounds
			recommendation = nothing
			used_L         = L
			used_tau_min   = tau_min
			used_tau_max   = tau_max
			t_recommend    = 0.0
			method_used    = :none
			if auto_recommend_L
				println("  Running recommend_L (method = $method)...")
				t0_rec         = time()
				recommendation = recommend_L(edges;
											nodes                = nodes,
											graph_type           = graph_type,
											reciprocity_collapse = reciprocity_collapse,
											tau_min              = tau_min,
											tau_max              = tau_max,
											method               = method,
											points_per_decade    = points_per_decade,
											L_min                = L_min,
											L_max                = L_max,
											n_exploratory        = n_exploratory,
											tol                  = tol,
											parallel             = false,           # serial candidates with inner threading (empirical only)
											verbose              = verbose,
											show_progress        = show_progress,
											inner_show_progress  = inner_show_progress)
				t_recommend  = time() - t0_rec
				used_L       = recommendation.L
				used_tau_min = recommendation.tau_min
				used_tau_max = recommendation.tau_max
				method_used  = recommendation.method
				println()
				println(@sprintf("  recommend_L done in %.2f s (%.2f min)", t_recommend, t_recommend / 60))
				println("    Method used:     $(recommendation.method)")
				println("    Reason:          $(recommendation.method_reason)")
				println("    T_max:           $(recommendation.T_max)")
				println("    Recommended L:   $used_L")
				println("    Recommended τ:   [$(round(used_tau_min, sigdigits=4)), $(round(used_tau_max, sigdigits=4))]")
				println()
			end

		#	Run the Threaded Layered Census at the Resolved (L, τ_min, τ_max)
			println("  Running threaded layered census at L = $used_L...")
			t0 = time()
			result = _triad_census_layered(edges;
								nodes                = nodes,
								graph_type           = graph_type,
								reciprocity_collapse = reciprocity_collapse,
								L                    = used_L,
								tau_min              = used_tau_min,
								tau_max              = used_tau_max,
								parallel             = true,
								show_progress        = show_progress,
								progress_desc        = "  [threaded] $label")
			t_threaded = time() - t0

		#	Report
			println()
			println(@sprintf("  Threaded census done:    %.2f s (%.2f min)", t_threaded, t_threaded / 60))
			println("    L used:                $used_L")
			println("    τ bounds used:          [$(round(Float64(used_tau_min), sigdigits=4)), " *
			        "$(round(Float64(used_tau_max), sigdigits=4))]")
			println("    τ grid points:          $(result.meta.L)")
			println("    Threads actually used:  $(result.meta.n_threads_used)")
			println(@sprintf("    Mean per-τ wall-clock:  %.2f s", t_threaded / result.meta.L))
			println("    Total per-τ count sum:  $(sum(result.per_tau.count))")
			if auto_recommend_L
				println(@sprintf("    Combined wall-clock:    %.2f s (%.2f min) including recommend_L",
									t_recommend + t_threaded, (t_recommend + t_threaded) / 60))
			end
			println("=" ^ 70)

		return (result         = result,
				t_threaded     = t_threaded,
				n_threads_used = result.meta.n_threads_used,
				recommendation = recommendation,
				t_recommend    = t_recommend,
				method_used    = method_used)
	end

#	Motif suites — fast unit-style tests on synthetic micro-graphs
	motif_dir_results   = test_motif_suite_directed()
	motif_undir_results = test_motif_suite_undirected()

#	Pajek fixture — Balikatan binary directed-collapse census
	agent_agent_all_com_edges = networks["balikatan_2022_weighted"].edges
	pajek_results = test_pajek_reference_triad_census(
		agent_agent_all_com_edges,
		"/mnt/d/Dropbox/Netanomics_Resources/Documents/SBP_BRIMS_2025/Large_Graph_Similarity/Test_Data/traid_census_observed.vec";
		label = "Balikatan directed w/ reciprocity collapse")

#	Threaded-vs-Serial bit-equality — Balikatan weighted layered census
#	This is the determinism contract test for the threaded τ loop.
#	Uses L=20 to keep total runtime manageable.
	balikatan_thread_test = test_threaded_vs_serial_layered(
		agent_agent_all_com_edges;
		graph_type           = :directed,
		reciprocity_collapse = false,
		L                    = 20,
		label                = "Balikatan directed weighted (L=20)")

#	Test Tau Bound Recommendation Algorithm 
	balikatan_rec = recommend_L(agent_agent_all_com_edges;
								graph_type = :directed,
								verbose    = true)
	println("Balikatan: L=$(balikatan_rec.L), τ ∈ [$(balikatan_rec.tau_min), $(balikatan_rec.tau_max)]")
	balikatan_rec.scan

#	Tests the undirected layered path on the new large-undirected case.
	marvel_network_edges = networks["marvel_universe_weighted"].edges
	marvel_network_nodes = networks["marvel_universe_weighted"].nodes
	marvel_timing_auto = time_threaded_layered(marvel_network_edges;
											nodes            = marvel_network_nodes,
											graph_type       = :undirected,
											auto_recommend_L = true,
											L_min            = 8,
											L_max            = 32,
											label            = "Marvel undirected weighted (auto-L)")

#	Helper Function for run_synthetic_blockmodel_tests: Build the "Hourglass" Network
	function _build_hourglass_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for a bipartite-like 10-node
				graph with two clear structural-equivalence classes:
				- Actors 1-4 send and receive ties only with actors 5-8
				- Actors 5-8 send and receive ties only with actors 1-4
				- Actors 9, 10 are isolates
		Notes:
			Used to verify that the blockmodel function recovers a clean
			bipartite block structure and correctly handles isolates.

			Expected result: actors 1-4 in one block (e.g., block 1),
			actors 5-8 in another (e.g., block 2), actors 9, 10 in block 0
			(isolates).
		"""

		#	Edges: All-To-All Between {1..4} and {5..8}, Both Directions
			src = String[]
			dst = String[]
			for i in 1:4
				for j in 5:8
					push!(src, string(i)); push!(dst, string(j))
					push!(src, string(j)); push!(dst, string(i))
				end
			end
			edges = DataFrame(src = src, dst = dst, weight = ones(Float64, length(src)))

		#	Nodes: 10 Total (Actors 1-8 + 2 Isolates)
			nodes = DataFrame(
				id    = string.(1:10),
				label = string.(1:10)
			)

		#	Metadata
			metadata = (
				network_name  = "Hourglass + Isolates Test",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = false,
				n_nodes       = 10,
				n_edges       = length(src)
			)

		#	Return Triple
			return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Helper Function for run_synthetic_blockmodel_tests: Build Three-Clique Network
	function _build_three_clique_test_network()
		"""
		Args:
			(none)
		Returns:
			NamedTuple: (edges, nodes, metadata) for a 12-node graph with three
				4-node cliques connected by one cross-edge between consecutive
				cliques (creating a "tri-clique with weak bridges" structure).
		Notes:
			Expected: structural equivalence recovers three blocks of 4 each
			at n_blocks = 3. The three cross-block edges (one per clique)
			barely perturb the clean block structure.
		"""

		#	Edges: Three Cliques (Each Internally Fully Connected, Both Directions)
			src = String[]
			dst = String[]
			for block_start in (1, 5, 9)
				for i in block_start:block_start + 3
					for j in block_start:block_start + 3
						if i != j
							push!(src, string(i))
							push!(dst, string(j))
						end
					end
				end
			end
			#	Cross-Block Edges
				push!(src, "1"); push!(dst, "5")
				push!(src, "5"); push!(dst, "9")
				push!(src, "9"); push!(dst, "1")

			edges = DataFrame(src = src, dst = dst, weight = ones(Float64, length(src)))

		#	Nodes: 12 Total
			nodes = DataFrame(
				id    = string.(1:12),
				label = string.(1:12)
			)

		#	Metadata
			metadata = (
				network_name  = "Three-Clique Test",
				source_format = "synthetic_test",
				directed      = true,
				weighted      = false,
				n_nodes       = 12,
				n_edges       = length(src)
			)

		#	Return Triple
			return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Run Synthetic Blockmodel Tests
	function run_synthetic_blockmodel_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Tests structural_equivalence_blockmodel on hand-derivable cases:

			1. Hourglass + 2 isolates → 2 blocks of 4 each, isolates in block 0
			2. Three cliques → 3 blocks of 4 each at n_blocks = 3
			3. Three cliques at n_blocks = 8 → at most 8 blocks; cluster counts
			   should be consistent with the 3-clique structure subdivided
			4. Empty edge list with nodes → all nodes in block 0 (all isolates)
			5. All-isolate input → all nodes in block 0
			6. Determinism: same input gives same partition across runs
			7. Rand index recovery: blockmodel of full network vs. itself = 1.0
			8. Bad n_blocks argument raises ArgumentError
		"""

		println("=" ^ 70)
		println("Synthetic blockmodel tests for network_statistics.jl Section 8")
		println("=" ^ 70)

		all_passed = true

		#	Test 1: Hourglass + 2 Isolates → Two Blocks of 4 + 2 Isolates
			println("\nTest 1: Hourglass + 2 isolates → 2 blocks of 4 each, 2 isolates")
			try
				net = _build_hourglass_test_network()
				bm = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 2)
				bm_sorted = sort(bm, :node)

				#	Sort by Integer Conversion of Node Label
					bm_sorted.sort_key = parse.(Int, string.(bm_sorted.node))
					sort!(bm_sorted, :sort_key)
					select!(bm_sorted, Not(:sort_key))

				blocks = bm_sorted.block
				#	Isolates (Nodes 9, 10) Should Be Block 0
					@assert blocks[9]  == 0 "Expected node 9 in block 0, got $(blocks[9])"
					@assert blocks[10] == 0 "Expected node 10 in block 0, got $(blocks[10])"
				#	The Other Two Sets Should Each Be Internally Consistent
					b1 = unique(blocks[1:4])
					b2 = unique(blocks[5:8])
					@assert length(b1) == 1 "Nodes 1-4 split across blocks: $(blocks[1:4])"
					@assert length(b2) == 1 "Nodes 5-8 split across blocks: $(blocks[5:8])"
					@assert b1[1] != b2[1] "Nodes 1-4 and 5-8 are in the same block (should be different)"
					@assert b1[1] in (1, 2) "Block label for actors 1-4 is $(b1[1]); expected 1 or 2"
					@assert b2[1] in (1, 2) "Block label for actors 5-8 is $(b2[1]); expected 1 or 2"
				println("  PASSED (block(1..4) = $(b1[1]), block(5..8) = $(b2[1]), isolates = 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: Three Cliques → 3 Blocks of 4 Each
			println("\nTest 2: Three-clique network at n_blocks=3 → 3 blocks of 4")
			try
				net = _build_three_clique_test_network()
				bm = sort(structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 3), :node)

				bm.sort_key = parse.(Int, string.(bm.node))
				sort!(bm, :sort_key)
				select!(bm, Not(:sort_key))

				blocks = bm.block
				#	Should Have Exactly 3 Blocks
					@assert length(unique(blocks)) == 3 "Expected 3 distinct blocks, got $(length(unique(blocks)))"
				#	No Isolates in This Network
					@assert minimum(blocks) >= 1 "Got block 0 (isolate) but no isolates expected; blocks: $blocks"
				#	Each Clique Should Be in Its Own Block
					b_clique_1 = unique(blocks[1:4])
					b_clique_2 = unique(blocks[5:8])
					b_clique_3 = unique(blocks[9:12])
					@assert length(b_clique_1) == 1 "Clique 1 split: $(blocks[1:4])"
					@assert length(b_clique_2) == 1 "Clique 2 split: $(blocks[5:8])"
					@assert length(b_clique_3) == 1 "Clique 3 split: $(blocks[9:12])"
					@assert length(unique([b_clique_1[1], b_clique_2[1], b_clique_3[1]])) == 3 "Two cliques merged into the same block"
				println("  PASSED (clique 1 → $(b_clique_1[1]), clique 2 → $(b_clique_2[1]), clique 3 → $(b_clique_3[1]))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Three Cliques at n_blocks = 8 → Returns at Most 8 Blocks
			println("\nTest 3: Three-clique at n_blocks=8 → at most 8 distinct blocks")
			try
				net = _build_three_clique_test_network()
				bm = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 8)
				n_blocks_actual = length(unique(bm.block))
				@assert n_blocks_actual <= 8 "Got $(n_blocks_actual) blocks, exceeds n_blocks = 8"
				@assert n_blocks_actual >= 3 "Got $(n_blocks_actual) blocks, fewer than the 3 underlying cliques"
				println("  PASSED ($(n_blocks_actual) blocks returned, in [3, 8])")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: Empty Edge List with Nodes → All Block 0
			println("\nTest 4: Empty edge list with nodes (N=5) → all in block 0")
			try
				edges = DataFrame(src = String[], dst = String[], weight = Float64[])
				nodes = DataFrame(id = string.(1:5), label = string.(1:5))
				bm = structural_equivalence_blockmodel(edges; nodes = nodes)
				@assert all(bm.block .== 0) "Expected all block 0, got $(bm.block)"
				@assert nrow(bm) == 5 "Expected 5 rows, got $(nrow(bm))"
				println("  PASSED (all 5 nodes in block 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: All-Isolate Input (Has nodes But No Edges) → All Block 0
			println("\nTest 5: All-isolate network (foundation test, only isolates kept)")
			try
				edges = DataFrame(src = String[], dst = String[], weight = Float64[])
				nodes = DataFrame(id = string.(1:8), label = string.(1:8))
				bm = structural_equivalence_blockmodel(edges; nodes = nodes, n_blocks = 4)
				@assert all(bm.block .== 0) "Expected all block 0, got $(bm.block)"
				@assert nrow(bm) == 8 "Expected 8 rows, got $(nrow(bm))"
				println("  PASSED (all 8 nodes in block 0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: Determinism — Same Input, Same Output
			println("\nTest 6: Determinism (same input → identical partition)")
			try
				net = _build_three_clique_test_network()
				bm1 = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 3)
				bm2 = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 3)
				@assert bm1.block == bm2.block "Same input gave different partitions"
				println("  PASSED (deterministic across runs)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 7: Rand Index of Blockmodel vs. Itself = 1.0
			println("\nTest 7: Self-Rand-index = 1.0 (perfect agreement of partition with itself)")
			try
				net = _build_three_clique_test_network()
				bm = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 3)
				ri = rand_index(bm.block, bm.block)
				@assert isapprox(ri, 1.0; atol = 1e-12) "Expected Rand = 1.0, got $ri"
				println("  PASSED (Rand(P, P) = $ri)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 8: Bad n_blocks Argument
			println("\nTest 8: n_blocks = 0 raises ArgumentError")
			try
				net = _build_three_clique_test_network()
				try
					_ = structural_equivalence_blockmodel(net.edges; nodes = net.nodes, n_blocks = 0)
					all_passed = false
					println("  FAILED: expected ArgumentError, got result")
				catch e2
					if e2 isa ArgumentError
						println("  PASSED")
					else
						rethrow(e2)
					end
				end
			catch e
				println("  FAILED: unexpected exception: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Synthetic blockmodel tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_synthetic_blockmodel_tests()

#######################################
#   CONSTRUCT NETWORK SUMMARY TABLE   #
#######################################

#	Helper Function: Single-Source Mean of a DataFrame Column
#	(Centralizes the std-vs-undefined logic for centralization output)
	function _network_centralization(scores::AbstractVector{<:Real})
		"""
		Args:
			scores::AbstractVector{<:Real}: per-node score values
		Returns:
			Float64: std of the scores (SMM's centralization convention)
				or 0.0 if scores has fewer than 2 elements.
		Notes:
			SMM (2022) report centralization as the sample standard deviation
			of the node-level centrality vector. This helper consolidates the
			minimum-length guard so that single-node or empty networks return
			0.0 rather than NaN.
		"""
		if length(scores) < 2
			return 0.0
		end
		return std(scores)
	end

#	Compute Phase 0 Statistics for a Single Network
	function compute_phase_0_statistics(edges::DataFrame,
	                                   nodes::DataFrame;
	                                   network_name::String = "(unnamed)",
	                                   directed::Bool = true,
	                                   bonacich_normalize::Symbol = :max,
	                                   tau_n_samples::Int = 500,
	                                   tau_seed::Int = 20260101,
	                                   binarize_degrees::Bool = true)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optionally :weight
			nodes::DataFrame: node universe with :id, :label
			network_name::String: identifier for the result row
			directed::Bool: whether to treat the graph as directed (default = true)
			bonacich_normalize::Symbol: Bonacich normalization to apply for
				comparable scales across networks (default = :max maps each
				network's Bonacich to [0, 1] by dividing by its own maximum;
				other options :l2, :none — see bonacich_centrality docs)
			tau_n_samples::Int: Monte Carlo samples for tau (default = 500)
			tau_seed::Int: RNG seed for tau (default = 20260101)
			binarize_degrees::Bool: when true (default), in/out/total degree are
				computed on the binarized adjacency (tie counts, not weight
				sums). This matches the binarization convention of every other
				Phase 0 measure (closeness, betweenness, Bonacich, components,
				transitivity, tau) and ensures that paired weighted/unweighted
				versions of the same network produce identical descriptive
				statistics. Set to false only if weight-summed degree is
				specifically desired.
		Returns:
			DataFrame: single-row data frame matching Phase 0 schema
		Notes:
			Runs the full Phase 0 measure battery on one network:
				- Degree family (in / out / symmetric)
				- Closeness (symmetric direction, normalized)
				- Betweenness (directed, normalized)
				- Bonacich power centrality (symmetric, normalized as specified)
				- Centralization for each centrality (SD of node scores)
				- Largest component proportion
				- Largest bicomponent proportion
				- Mean inverse distance (unscaled)
				- Global transitivity
				- Tau statistic with ranked-clusters weighting (directed only;
				  returns 0.0 for undirected networks since tau is undefined)

			For undirected networks, in-degree and out-degree are identical to
			the symmetric degree by construction.
		"""

		#	Network Dimensions
			n_nodes = nrow(nodes)
			n_edges = nrow(edges)

		#	Empty-Edge Guard: Return All-Zeros Row
			if n_edges == 0 || n_nodes == 0
				return DataFrame(
					network_name                       = [network_name],
					n_nodes                            = [n_nodes],
					n_edges                            = [n_edges],
					directed                           = [directed],
					mean_indegree                      = [0.0],
					sd_indegree                        = [0.0],
					mean_outdegree                     = [0.0],
					sd_outdegree                       = [0.0],
					mean_symmetric_degree              = [0.0],
					sd_symmetric_degree                = [0.0],
					mean_closeness                     = [0.0],
					sd_closeness                       = [0.0],
					mean_betweenness                   = [0.0],
					sd_betweenness                     = [0.0],
					mean_bonacich                      = [0.0],
					sd_bonacich                        = [0.0],
					centralization_indegree            = [0.0],
					centralization_outdegree           = [0.0],
					centralization_symmetric_degree    = [0.0],
					centralization_closeness           = [0.0],
					centralization_betweenness         = [0.0],
					centralization_bonacich            = [0.0],
					largest_component_proportion       = [n_nodes > 0 ? 1.0 / n_nodes : 0.0],
					largest_bicomponent_proportion     = [n_nodes > 0 ? 1.0 / n_nodes : 0.0],
					mean_inverse_distance              = [0.0],
					global_transitivity                = [0.0],
					tau_rc                             = [0.0]
				)
			end

		#	--- Degree Family ---
		#	Binarize by default so degree statistics match the convention of
		#	the other Phase 0 measures (which all use edge_interpretation = :ignore).
		#	Without binarization, paired weighted/unweighted versions of the same
		#	network produce different rows in the degree columns, breaking
		#	cross-network comparability.
			use_weights = !binarize_degrees
			in_deg  = in_degree(edges;    nodes = nodes, weighted = use_weights)
			out_deg = out_degree(edges;   nodes = nodes, weighted = use_weights)
			sym_deg = total_degree(edges; nodes = nodes, weighted = use_weights, directed = directed)

			in_vals  = Float64.(in_deg.in_degree)
			out_vals = Float64.(out_deg.out_degree)
			sym_vals = Float64.(sym_deg.total_degree)

		#	--- Closeness (Symmetric, Normalized) ---
			cc_df = closeness_centrality(edges;
			                            nodes = nodes,
			                            directed = directed,
			                            direction = :symmetric,
			                            edge_interpretation = :ignore,
			                            normalize = true)
			cc_vals = Float64.(cc_df.closeness)

		#	--- Betweenness (Normalized to [0, 1] for Cross-Network Comparability) ---
			bc_df = betweenness_centrality(edges;
			                              nodes = nodes,
			                              directed = directed,
			                              edge_interpretation = :ignore,
			                              normalize = true)
			bc_vals = Float64.(bc_df.betweenness)

		#	--- Bonacich (Symmetric, Normalized for Cross-Network Comparability) ---
			bn_df = bonacich_centrality(edges;
			                           nodes = nodes,
			                           directed = directed,
			                           direction = :symmetric,
			                           edge_interpretation = :ignore,
			                           normalize = bonacich_normalize)
			bn_vals = Float64.(bn_df.bonacich)

		#	--- Topology Measures ---
			lcp = largest_component_proportion(edges; nodes = nodes, directed = directed)
			lbp = largest_bicomponent_proportion(edges; nodes = nodes, directed = directed)
			mid_val = mean_inverse_distance(edges;
			                               nodes = nodes,
			                               directed = directed,
			                               direction = :symmetric,
			                               edge_interpretation = :ignore,
			                               scale_by_log_n = false)
			if directed
				gcc = global_clustering_coefficient(edges;
				                                  directed = true,
				                                  method = :transitivity)
			else
				gcc = global_clustering_coefficient(edges;
				                                  directed = false,
				                                  method = :average,
				                                  average_mode = :local_clustering)
			end

		#	--- Tau (Directed Only) ---
			if directed
				tau_df = tau_statistic(edges;
				                      nodes = nodes,
				                      weighting = :RC,
				                      n_samples = tau_n_samples,
				                      directed = true,
				                      seed = tau_seed)
				tau_val = tau_df.tau[1]
			else
				tau_val = 0.0
			end

		#	Assembling Result Row
			return DataFrame(
				network_name                       = [network_name],
				n_nodes                            = [n_nodes],
				n_edges                            = [n_edges],
				directed                           = [directed],
				mean_indegree                      = [mean(in_vals)],
				sd_indegree                        = [length(in_vals)  > 1 ? std(in_vals)  : 0.0],
				mean_outdegree                     = [mean(out_vals)],
				sd_outdegree                       = [length(out_vals) > 1 ? std(out_vals) : 0.0],
				mean_symmetric_degree              = [mean(sym_vals)],
				sd_symmetric_degree                = [length(sym_vals) > 1 ? std(sym_vals) : 0.0],
				mean_closeness                     = [mean(cc_vals)],
				sd_closeness                       = [length(cc_vals)  > 1 ? std(cc_vals)  : 0.0],
				mean_betweenness                   = [mean(bc_vals)],
				sd_betweenness                     = [length(bc_vals)  > 1 ? std(bc_vals)  : 0.0],
				mean_bonacich                      = [mean(bn_vals)],
				sd_bonacich                        = [length(bn_vals)  > 1 ? std(bn_vals)  : 0.0],
				centralization_indegree            = [_network_centralization(in_vals)],
				centralization_outdegree           = [_network_centralization(out_vals)],
				centralization_symmetric_degree    = [_network_centralization(sym_vals)],
				centralization_closeness           = [_network_centralization(cc_vals)],
				centralization_betweenness         = [_network_centralization(bc_vals)],
				centralization_bonacich            = [_network_centralization(bn_vals)],
				largest_component_proportion       = [lcp],
				largest_bicomponent_proportion     = [lbp],
				mean_inverse_distance              = [mid_val],
				global_transitivity                = [gcc],
				tau_rc                             = [tau_val]
			)
	end

#	Build Phase 0 Table by Iterating Over All Networks in a Corpus
	function build_phase_0_table(networks::Dict;
	                            bonacich_normalize::Symbol = :max,
	                            tau_n_samples::Int = 500,
	                            tau_seed::Int = 20260101,
	                            verbose::Bool = true)
		"""
		Args:
			networks::Dict: dictionary keyed by network name with values
				having .edges (DataFrame), .nodes (DataFrame), .metadata
				(NamedTuple with at least :directed)
			bonacich_normalize::Symbol: passed through to per-network call
			tau_n_samples::Int: Monte Carlo samples for tau (default = 500)
			tau_seed::Int: RNG seed for tau (default = 20260101)
			verbose::Bool: print progress as each network is processed
		Returns:
			DataFrame: one row per network, columns matching Phase 0 schema
		Notes:
			Iterates over `networks` in sorted-key order so the result row
			order is deterministic and easy to compare across runs. Each
			network's `metadata.directed` flag determines the directed
			argument; if metadata lacks that field, defaults to true.

			Progress reporting (when verbose=true) includes elapsed time per
			network — useful when computing tau on large networks like
			Balikatan (typically ~30-60s for n_samples=500).
		"""

		#	Validation
			if isempty(networks)
				return DataFrame()
			end

		#	Sort Keys for Deterministic Row Order
			network_names = sort(collect(keys(networks)))

		#	Accumulate Rows
			rows = DataFrame[]
			for (i, name) in enumerate(network_names)
				net = networks[name]
				if verbose
					println("[$(i)/$(length(network_names))] Computing Phase 0 stats for: $name")
				end
				t0 = time()

				#	Determine Directedness from Metadata If Available
					is_directed = true
					if hasproperty(net, :metadata) && hasproperty(net.metadata, :directed)
						is_directed = Bool(net.metadata.directed)
					end

				#	Compute Single-Row Result
					row = compute_phase_0_statistics(net.edges, net.nodes;
					                                network_name = name,
					                                directed = is_directed,
					                                bonacich_normalize = bonacich_normalize,
					                                tau_n_samples = tau_n_samples,
					                                tau_seed = tau_seed)
					push!(rows, row)

				if verbose
					elapsed = time() - t0
					println("    Done in $(round(elapsed, digits=2))s " *
					        "(N = $(nrow(net.nodes)), E = $(nrow(net.edges)))")
				end
			end

		#	Vertical Concatenation
			return reduce(vcat, rows)
	end
	@doc raw"""
	**Description**
	Compute the full Phase 0 descriptive-statistics table for a corpus of
	networks, matching Smith, Morgan, & Moody (2022) Table 1.

	**Usage**
	`build_phase_0_table(networks::Dict; bonacich_normalize=:max, tau_n_samples=500, tau_seed=20260101, verbose=true)`

	**Arguments**
	- `networks::Dict`: Dictionary keyed by network name (String), with
	  values having `.edges`, `.nodes`, `.metadata` fields. The `.metadata`
	  NamedTuple should include `:directed` (Bool); defaults to `true` if
	  missing.
	- `bonacich_normalize::Symbol`: Normalization applied to Bonacich power
	  centrality for cross-network comparability. Default `:max` divides
	  each network's Bonacich vector by its maximum, producing values in
	  $[0, 1]$. See `bonacich_centrality` for other options.
	- `tau_n_samples::Int`: Monte Carlo samples for the tau statistic
	  (default 500; gives roughly 4% relative error on the variance).
	- `tau_seed::Int`: RNG seed for tau reproducibility.
	- `verbose::Bool`: Print per-network progress with elapsed time.

	**Details**
	Runs the full Phase 0 measure battery on every network in the corpus:
	degree family, closeness, betweenness, Bonacich, their centralizations,
	largest component proportion, largest bicomponent proportion, mean
	inverse distance, global transitivity, and the tau statistic with
	ranked-clusters weighting.

	Tau is undefined for undirected networks (no asymmetric/mutual dyad
	distinction) and is reported as `0.0` in that case.

	Networks are processed in sorted-name order for reproducible output.

	**Value**
	A `DataFrame` with one row per network and columns matching the Phase 0
	schema (see module-level documentation).

	**See Also**
	`compute_phase_0_statistics`
	""" build_phase_0_table
    table = build_phase_0_table(networks)

#   Build & Evaluate Summary Table
	function run_phase_0_driver_tests()
		"""
		Args:
			(none)
		Returns:
			Bool: true if all tests pass, false otherwise
		Notes:
			Sanity-tests compute_phase_0_statistics and build_phase_0_table.
			These tests are not checking specific numeric values (those are
			covered by each measure's own synthetic tests). Instead they
			verify that:

			1. The driver returns a 1-row DataFrame with the expected schema
			2. All Phase 0 columns are present
			3. n_nodes and n_edges fields match the input
			4. Centralization values equal the SD of the corresponding
			   centrality column (cross-check our centralization convention)
			5. build_phase_0_table on a 2-network corpus returns 2 rows
			6. Empty edge list produces a degenerate but valid row
		"""

		println("=" ^ 70)
		println("Phase 0 driver sanity tests")
		println("=" ^ 70)

		all_passed = true

		#	Expected Schema Columns
			expected_cols = [
				:network_name, :n_nodes, :n_edges, :directed,
				:mean_indegree, :sd_indegree,
				:mean_outdegree, :sd_outdegree,
				:mean_symmetric_degree, :sd_symmetric_degree,
				:mean_closeness, :sd_closeness,
				:mean_betweenness, :sd_betweenness,
				:mean_bonacich, :sd_bonacich,
				:centralization_indegree,
				:centralization_outdegree,
				:centralization_symmetric_degree,
				:centralization_closeness,
				:centralization_betweenness,
				:centralization_bonacich,
				:largest_component_proportion,
				:largest_bicomponent_proportion,
				:mean_inverse_distance,
				:global_transitivity,
				:tau_rc
			]

		#	Test 1: Schema and Single-Row Output
			println("\nTest 1: compute_phase_0_statistics returns 1-row DataFrame with full schema")
			try
				net = _build_foundation_test_network()
				df = compute_phase_0_statistics(net.edges, net.nodes;
				                                network_name = "foundation_test",
				                                directed = true,
				                                tau_n_samples = 50)
				@assert nrow(df) == 1 "Expected 1 row, got $(nrow(df))"
				for col in expected_cols
					@assert col in propertynames(df) "Missing column: $col"
				end
				@assert df.network_name[1] == "foundation_test" "Wrong network_name"
				@assert df.n_nodes[1] == 8 "Expected n_nodes = 8, got $(df.n_nodes[1])"
				@assert df.n_edges[1] == 7 "Expected n_edges = 7, got $(df.n_edges[1])"
				@assert df.directed[1] == true "Expected directed = true"
				println("  PASSED ($(length(expected_cols)) columns present)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 2: Centralization Equals SD of Centrality Column
			println("\nTest 2: Centralization values equal std of node-level scores")
			try
				net = _build_foundation_test_network()
				df = compute_phase_0_statistics(net.edges, net.nodes;
				                                network_name = "foundation_test",
				                                directed = true,
				                                tau_n_samples = 50)
				#	Recompute Independently and Verify Match
				#	Driver binarizes degrees by default (binarize_degrees=true),
				#	so the reference computation here must also binarize.
					in_deg = in_degree(net.edges; nodes = net.nodes, weighted = false)
					expected_centralization = std(Float64.(in_deg.in_degree))
					@assert isapprox(df.centralization_indegree[1], expected_centralization; atol = 1e-12) "centralization_indegree mismatch: got $(df.centralization_indegree[1]), expected $expected_centralization"
				println("  PASSED (centralization_indegree = $(round(df.centralization_indegree[1], digits=4)))")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 3: Component and Bicomponent Proportions Match Known Values
			println("\nTest 3: Component / bicomponent proportions on foundation network")
			try
				net = _build_foundation_test_network()
				df = compute_phase_0_statistics(net.edges, net.nodes;
				                                network_name = "foundation_test",
				                                directed = true,
				                                tau_n_samples = 50)
				#	Largest Component Has 3 Nodes (One of the Triangles), N = 8
				#	→ Proportion = 3/8 = 0.375
				#	Same for the Largest Bicomponent
					@assert isapprox(df.largest_component_proportion[1], 3/8; atol = 1e-12) "Component proportion mismatch"
					@assert isapprox(df.largest_bicomponent_proportion[1], 3/8; atol = 1e-12) "Bicomponent proportion mismatch"
				println("  PASSED (both = 3/8 = 0.375)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 4: build_phase_0_table on 2-Network Corpus
			println("\nTest 4: build_phase_0_table on 2-network corpus returns 2 rows")
			try
				net1 = _build_foundation_test_network()
				net2 = _build_triangle_test_network()
				corpus = Dict("a_foundation" => net1, "b_triangle" => net2)
				table = build_phase_0_table(corpus;
				                           tau_n_samples = 50,
				                           verbose = false)
				@assert nrow(table) == 2 "Expected 2 rows, got $(nrow(table))"
				#	Sorted by Key: "a_foundation" Comes Before "b_triangle"
					@assert table.network_name[1] == "a_foundation" "Row order is not alphabetical by key"
					@assert table.network_name[2] == "b_triangle" "Row order is not alphabetical by key"
				println("  PASSED (2 rows in alphabetical order)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 5: Empty Edge List Produces Valid Row (No Errors)
			println("\nTest 5: Empty edge list produces a valid degenerate row")
			try
				edges_empty = DataFrame(src = String[], dst = String[], weight = Float64[])
				nodes_empty = DataFrame(id = string.(1:5), label = string.(1:5))
				df = compute_phase_0_statistics(edges_empty, nodes_empty;
				                                network_name = "empty_test",
				                                directed = true,
				                                tau_n_samples = 50)
				@assert nrow(df) == 1 "Expected 1 row"
				@assert df.n_nodes[1] == 5 "Expected n_nodes = 5"
				@assert df.n_edges[1] == 0 "Expected n_edges = 0"
				@assert df.mean_indegree[1] == 0.0 "Expected mean_indegree = 0"
				@assert df.tau_rc[1] == 0.0 "Expected tau_rc = 0"
				#	Schema Should Still Be Complete
					for col in expected_cols
						@assert col in propertynames(df) "Missing column on empty input: $col"
					end
				println("  PASSED (empty input produces well-formed row)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Test 6: Undirected Network Returns Tau = 0
			println("\nTest 6: Undirected network has tau_rc = 0 (tau undefined for undirected)")
			try
				net = _build_foundation_test_network()
				df = compute_phase_0_statistics(net.edges, net.nodes;
				                                network_name = "undirected_test",
				                                directed = false,
				                                tau_n_samples = 50)
				@assert df.tau_rc[1] == 0.0 "Expected tau_rc = 0.0 for undirected, got $(df.tau_rc[1])"
				@assert df.directed[1] == false "Expected directed = false"
				println("  PASSED (undirected → tau_rc = 0.0)")
			catch e
				println("  FAILED: $e")
				all_passed = false
			end

		#	Report Overall Result
			println("\n" * "=" ^ 70)
			println("Phase 0 driver tests: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("=" ^ 70)

			return all_passed
	end
    run_phase_0_driver_tests()

#   Write-Out CSV. 
    CSV.write("Data/phase_0_table.csv", table)

#	Iterate Over Networks in Sorted-Key Order (Matches Phase 0 Table Order)
	gini_rows = DataFrame[]
	for name in sort(collect(keys(networks)))
		net = networks[name]

		#	Determine Directedness
			is_directed = true
			if hasproperty(net, :metadata) && hasproperty(net.metadata, :directed)
				is_directed = Bool(net.metadata.directed)
			end

		#	Compute the Binarized Symmetric Degree Distribution
		#	weighted=false matches the binarization convention used elsewhere
		#	in the Phase 0 driver.
			sym_deg = total_degree(net.edges;
			                      nodes = net.nodes,
			                      weighted = false,
			                      directed = is_directed)
			sym_vals = Float64.(sym_deg.total_degree)

		#	Compute Gini (Guard Against Trivially-Small Networks)
			g = length(sym_vals) > 1 ? gini_coefficient(sym_vals) : 0.0

		#	Accumulate Row
			push!(gini_rows, DataFrame(network_name = [name],
			                          gini_symmetric_degree = [round(g, digits = 4)]))
	end

#	Combine and Print
	gini_table = reduce(vcat, gini_rows)
	println(gini_table)
