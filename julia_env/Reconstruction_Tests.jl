#Network Reconstruction Tests
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

using DataFrames
using Printf
using Random
using Statistics
using StatsBase
using Network_Credible_Intervals
using Network_Credible_Intervals.network_reconstruction: _compute_observed_centrality,
                                                         _compute_ei_values,
                                                         _detect_community_structure,
                                                         _bin_observed_nodes,
                                                         _compute_p_matrix,
                                                         _compute_r_matrix,
                                                         _determine_n_add,
                                                         _realized_rho_for_beta,
                                                         _solve_bin_distribution,
                                                         _compute_ei_conditional

#################
#   FUNCTIONS   #
#################

#	Constructed Fixture: N-node star (Phase 2 unit tests)
	function _build_star_fixture(; n::Int = 11, directed::Bool = true)
		"""
		Args:
			n::Int: number of nodes including the hub (default 11)
			directed::Bool: if true, all edges point from leaves to hub
				(in-degree concentrated at hub); if false, undirected.
				Default true.
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata.
		Notes:
			Construction: node 1 is the hub ("n1"), nodes 2..n are leaves
			("n2" through "nN"). When directed, all arcs are leaf -> hub,
			giving the hub in-degree (n-1) and all leaves in-degree 0.
			When undirected, the hub has degree (n-1) and all leaves have
			degree 1.

			For Phase 2 _compute_observed_centrality testing, the directed
			star produces hub centrality = (n-1) (from in-degree, since
			out-degree is 0) and leaf centrality = 1 each (from out-degree).
			The undirected star produces hub centrality = (n-1) and leaf
			centrality = 1 each via undirected degree.

			Nodes DataFrame has :id and :label columns to match the
			convention enforced by _graph_to_sparse_matrix. Both columns
			hold the same identifier strings ("n1", "n2", ...).
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
		metadata = (directed = directed, name = "star_$n", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Constructed Fixture: Asymmetric directed block (Test 2)
	function _build_asymmetric_block_fixture()
		"""
		Args: none
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata.
		Notes:
			Builds a small directed network where each node has a known,
			asymmetric in-degree and out-degree. Hand-computed totals:

			Node  in-deg  out-deg  total (cent)
			n1     0       3        3
			n2     1       2        3
			n3     2       1        3
			n4     3       0        3
			n5     2       2        4
			n6     2       3        5

			Edge list (directed):
				n1 -> n2,  n1 -> n3,  n1 -> n4
				n2 -> n3,  n2 -> n4
				n3 -> n4
				n5 -> n3,  n5 -> n6
				n6 -> n2,  n6 -> n4,  n6 -> n5
				          (and n5 -> n6 already, so n6 also gets one inbound)

			Total edges = 11. Total in-degree sum = total out-degree sum
			= 11 (sanity check).

			This fixture exists specifically to verify that
			_compute_observed_centrality on directed networks SUMS in-degree
			and out-degree rather than returning just one. If the function
			were returning only in-degree, n1 would show centrality 0;
			if only out-degree, n4 would show 0. The sum-based design
			gives n1 = 3, n4 = 3, both nonzero and identical, with
			n5 = 4 and n6 = 5 as further discriminating values.
		"""
		src = ["n1", "n1", "n1", "n2", "n2", "n3", "n5", "n5", "n6", "n6", "n6"]
		dst = ["n2", "n3", "n4", "n3", "n4", "n4", "n3", "n6", "n2", "n4", "n5"]
		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		node_ids = ["n1", "n2", "n3", "n4", "n5", "n6"]
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = true, name = "asym_block_6", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata)
	end

#	Constructed Fixture: All-internal partition (Test 3)
	function _build_all_internal_fixture(; n_per_community::Int = 5,
											n_communities::Int = 2)
		"""
		Args:
			n_per_community::Int: nodes per community (default 5)
			n_communities::Int: number of communities (default 2)
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata, plus a
			:community_labels field with the hand-set partition.
		Notes:
			Builds a network where every edge is internal to a community.
			Each community is a complete graph (clique) among its members;
			no edges cross between communities. The community labels are
			set such that nodes 1..k are in community 1, nodes k+1..2k
			are in community 2, etc.

			Expected E/I per node: -1 (all internal, zero external).

			With defaults (5 nodes × 2 communities), the network has:
			- 10 nodes total
			- 2 cliques of 5 nodes each
			- Each node has degree 4 (4 internal edges to clique-mates)
			- Total edges = 2 * C(5, 2) = 20 edges
			(For directed treatment, each unordered pair contributes one
			edge; this fixture is logically undirected but stored in the
			edge-list convention we use everywhere.)

			The fixture returns community_labels so tests can pass them
			directly to _compute_ei_values without having to recompute.
		"""
		n_per_community >= 2 || throw(ArgumentError("n_per_community must be >= 2"))
		n_communities >= 1 || throw(ArgumentError("n_communities must be >= 1"))

		n_total = n_per_community * n_communities
		node_ids = ["n$i" for i in 1:n_total]

		src = String[]
		dst = String[]
		community_labels = Int[]

		#	Build cliques and labels
			for c in 1:n_communities
				start_idx = (c - 1) * n_per_community + 1
				end_idx   = c * n_per_community
				#	Within-clique edges (unordered pairs, stored as src -> dst with src < dst)
					for i in start_idx:end_idx
						for j in (i + 1):end_idx
							push!(src, node_ids[i])
							push!(dst, node_ids[j])
						end
					end
				#	Community labels for these nodes
					for _ in start_idx:end_idx
						push!(community_labels, c)
					end
			end

		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = false, name = "all_internal", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata,
				community_labels = community_labels)
	end

#	Constructed Fixture: All-external partition (Test 4)
	function _build_all_external_fixture(; n_group_a::Int = 4, n_group_b::Int = 5)
		"""
		Args:
			n_group_a::Int: nodes in community 1 (default 4)
			n_group_b::Int: nodes in community 2 (default 5)
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata, plus a
			:community_labels field with the hand-set partition.
		Notes:
			Builds a complete bipartite graph K(n_group_a, n_group_b) where
			every edge crosses between communities. Group A nodes are in
			community 1 (labels 1..1), Group B nodes are in community 2
			(labels 2..2). No within-group edges exist.

			Expected E/I per node:
				EI = (E - I) / (E + I) = (E - 0) / (E + 0) = +1
			where E = (n_group_b) for Group A nodes and E = (n_group_a)
			for Group B nodes.

			With defaults (4 × 5):
			- 9 nodes total: 4 in group A, 5 in group B
			- 4 * 5 = 20 edges, all crossing
			- Group A nodes have degree 5; Group B nodes have degree 4
			- All nodes return E/I = +1

			Edges are stored as (a-node, b-node) pairs once per pair,
			consistent with the unordered convention used in
			_build_all_internal_fixture.
		"""
		n_group_a >= 1 || throw(ArgumentError("n_group_a must be >= 1"))
		n_group_b >= 1 || throw(ArgumentError("n_group_b must be >= 1"))

		n_total = n_group_a + n_group_b
		node_ids = ["n$i" for i in 1:n_total]

		#	Build complete-bipartite edges: group A indices 1..n_group_a,
		#	group B indices (n_group_a+1)..n_total
			src = String[]
			dst = String[]
			for a in 1:n_group_a
				for b in (n_group_a + 1):n_total
					push!(src, node_ids[a])
					push!(dst, node_ids[b])
				end
			end

		#	Community labels: 1 for group A nodes, 2 for group B nodes
			community_labels = vcat(fill(1, n_group_a), fill(2, n_group_b))

		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = false, name = "all_external", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata,
				community_labels = community_labels)
	end

#	Constructed Fixture: Block-with-bridges (Test 5)
	function _build_block_with_bridges_fixture()
		"""
		Args: none
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata, plus a
			:community_labels field with the hand-set partition and an
			:expected_ei field with hand-computed E/I values.
		Notes:
			Builds a 6-node network with two 3-node cliques bridged by
			two cross-cutting edges. Hand-computed E/I values verify the
			function on a non-trivial mixed case.

			Edges (10 total):
			  Community 1 internal (clique on n1, n2, n3): 3 edges
				n1-n2, n1-n3, n2-n3
			  Community 2 internal (clique on n4, n5, n6): 3 edges
				n4-n5, n4-n6, n5-n6
			  Bridge edges (cross-community): 4 edges
				n1-n4 (bridges via n1 and n4)
				n1-n5 (bridges via n1 and n5)
				n2-n4 (bridges via n2 and n4)
				n3-n6 (bridges via n3 and n6)

			Community labels:
				n1: 1, n2: 1, n3: 1, n4: 2, n5: 2, n6: 2

			Hand-computed per-node E and I counts:
				n1: I = 2 (to n2, n3),    E = 2 (to n4, n5)    -> E/I = (2-2)/4 =  0.0
				n2: I = 2 (to n1, n3),    E = 1 (to n4)        -> E/I = (1-2)/3 = -0.333...
				n3: I = 2 (to n1, n2),    E = 1 (to n6)        -> E/I = (1-2)/3 = -0.333...
				n4: I = 2 (to n5, n6),    E = 2 (to n1, n2)    -> E/I = (2-2)/4 =  0.0
				n5: I = 2 (to n4, n6),    E = 1 (to n1)        -> E/I = (1-2)/3 = -0.333...
				n6: I = 2 (to n4, n5),    E = 1 (to n3)        -> E/I = (1-2)/3 = -0.333...

			Sanity check: total external-incidences should equal 2 * (number
			of bridge edges) = 8. Sum E across nodes: 2+1+1+2+1+1 = 8. OK
			Sum I across nodes: 2+2+2+2+2+2 = 12 = 2 * (6 internal edges). OK

			This fixture exercises:
			- The case where E and I are both nonzero (mixed)
			- Both 0 and negative E/I values in the same fixture
			- Asymmetric distributions: n1 and n4 are bridge-balanced
			  (E/I = 0), while the other four nodes lean internal
		"""
		src = String[
			"n1", "n1", "n2",                      # community 1 clique
			"n4", "n4", "n5",                      # community 2 clique
			"n1", "n1", "n2", "n3"                 # bridges
		]
		dst = String[
			"n2", "n3", "n3",
			"n5", "n6", "n6",
			"n4", "n5", "n4", "n6"
		]
		node_ids = ["n1", "n2", "n3", "n4", "n5", "n6"]
		community_labels = [1, 1, 1, 2, 2, 2]

		#	Hand-computed expected E/I values
			one_third = -1.0 / 3.0
			expected_ei = [0.0, one_third, one_third, 0.0, one_third, one_third]

		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = false, name = "block_with_bridges", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata,
				community_labels = community_labels,
				expected_ei = expected_ei)
	end

#	Constructed Fixture: Mostly-isolates with small cluster (Test 7)
	function _build_mostly_isolates_fixture(; n_total::Int = 12, n_connected::Int = 4)
		"""
		Args:
			n_total::Int: total number of nodes (default 12)
			n_connected::Int: number of connected nodes; rest are isolates
				(default 4; must be < n_total)
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata, plus a
			:community_labels field with the hand-set partition.
		Notes:
			Builds a network where most nodes are isolates (no edges) and
			a small handful form a connected subnetwork split across two
			communities.

			With defaults (12 total, 4 connected):
			- Nodes n1, n2 are in community 1 with an edge between them
			- Nodes n3, n4 are in community 2 with an edge between them
			- One cross-community edge: n2 - n3
			- Nodes n5..n12 are isolates with no edges, labeled in
			  community 1 (arbitrary for isolates)

			Expected E/I values:
				n1: I = 1 (to n2),         E = 0           -> EI = -1.0
				n2: I = 1 (to n1),         E = 1 (to n3)   -> EI = 0.0
				n3: I = 1 (to n4),         E = 1 (to n2)   -> EI = 0.0
				n4: I = 1 (to n3),         E = 0           -> EI = -1.0
				n5..n12 (8 isolates):                       -> EI = NaN

			Number of nodes with defined E/I: 4.
			Threshold at J=3, min_nodes_per_ei_bin=3: 9.
			Since 4 < 9, the function should return :too_few_nodes.

			Two communities are present, so the :single_community fallback
			does NOT fire. The :too_few_nodes fallback fires before any
			E/I-variance check is reached.

			The community labels for isolates are arbitrary (set to 1 here),
			but it does mean the unique-label count is 2, not 1.
		"""
		n_total >= 5 || throw(ArgumentError("n_total must be >= 5"))
		n_connected >= 4 || throw(ArgumentError("n_connected must be >= 4"))
		n_connected < n_total || throw(ArgumentError("n_connected must be < n_total"))

		node_ids = ["n$i" for i in 1:n_total]

		#	Build the connected subnetwork (always uses first 4 nodes)
			src = ["n1", "n3", "n2"]
			dst = ["n2", "n4", "n3"]

		#	Build community labels: n1, n2 -> community 1; n3, n4 -> community 2;
		#	isolates (n5..n_total) -> community 1 (arbitrary)
			community_labels = vcat([1, 1, 2, 2], fill(1, n_total - 4))

		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = false, name = "mostly_isolates", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata,
				community_labels = community_labels)
	end

#	Constructed Fixture: Healthy 2D-supportable network (Test 9)
	function _build_healthy_2d_fixture()
		"""
		Args: none
		Returns:
			NamedTuple in corpus format with edges/nodes/metadata, plus a
			:community_labels field and an :expected_ei field with hand-
			computed E/I values.
		Notes:
			Builds a 12-node, two-community network where the connected
			structure is engineered to produce E/I values spanning all three
			semantic bins:
			- At least 3 nodes with E/I <= -0.33 (low bin: internal)
			- At least 3 nodes with -0.33 < E/I < 0.33 (mid bin)
			- At least 3 nodes with E/I >= +0.33 (high bin: broker)

			Construction:
			Community 1: nodes n1, n2, n3, n4, n5, n6 (labels = 1)
			Community 2: nodes n7, n8, n9, n10, n11, n12 (labels = 2)

			Edges, engineered to produce the target E/I distribution:
				n1-n2  (within C1)
				n1-n3  (within C1)
				n2-n3  (within C1)
					-> n1, n2, n3 are pure-internal members (each has 2
					   internal edges, 0 external; E/I = -1)

				n10-n11 (within C2)
				n10-n12 (within C2)
				n11-n12 (within C2)
					-> n10, n11, n12 are pure-internal members
					   (E/I = -1)

				n4-n7  (cross)
				n4-n8  (cross)
				n4-n9  (cross)
					-> n4 has 0 internal, 3 external; E/I = +1
					-> n7, n8, n9 each have 1 external incidence and
					   so far 0 internal

				n7-n8  (within C2)
				n8-n9  (within C2)
					-> n7 has 1 internal, 1 external; E/I = 0
					-> n8 has 2 internal, 1 external; E/I = -1/3
					-> n9 has 1 internal, 1 external; E/I = 0

				n5-n7  (cross)
				n6-n9  (cross)
					-> n5 has 0 internal, 1 external; E/I = +1
					-> n6 has 0 internal, 1 external; E/I = +1
					-> n7 updated: 1 internal, 2 external; E/I = +1/3
					-> n9 updated: 1 internal, 2 external; E/I = +1/3

			Final hand-computed E/I per node:
				n1:  -1.0    (3 internal: n2, n3; 0 external)
				n2:  -1.0    (2 internal: n1, n3; 0 external)
				n3:  -1.0    (2 internal: n1, n2; 0 external)
				n4:  +1.0    (0 internal; 3 external: n7, n8, n9)
				n5:  +1.0    (0 internal; 1 external: n7)
				n6:  +1.0    (0 internal; 1 external: n9)
				n7:  +1/3    (1 internal: n8; 2 external: n4, n5)
				n8:  -1/3    (2 internal: n7, n9; 1 external: n4)
				n9:  +1/3    (1 internal: n8; 2 external: n4, n6)
				n10: -1.0    (2 internal: n11, n12; 0 external)
				n11: -1.0    (2 internal: n10, n12; 0 external)
				n12: -1.0    (2 internal: n10, n11; 0 external)

			Wait, let me recount n1: edges n1-n2, n1-n3, plus... no others
			that include n1. So n1 has 2 incident edges (to n2 and n3), both
			internal. E/I = -1.

			Bin populations under J=3 semantic thresholds (-0.33, +0.33):
				E/I <= -0.33: n1, n2, n3, n8, n10, n11, n12 = 7 nodes (low bin)
				-0.33 < E/I < 0.33: n7, n9 = 2 nodes (mid bin)
				E/I >= +0.33: n4, n5, n6 = 3 nodes (high bin)

			Hmm — mid bin has 2 nodes, which is < min_nodes_per_ei_bin = 3.
			Need to add one more node to the mid range. Let me add:
				n5-n6 (within C1)
					-> n5 updated: 1 internal, 1 external; E/I = 0
					-> n6 updated: 1 internal, 1 external; E/I = 0

			Updated E/I per node:
				n1:  -1.0
				n2:  -1.0
				n3:  -1.0
				n4:  +1.0
				n5:  0.0      (1 internal: n6; 1 external: n7)
				n6:  0.0      (1 internal: n5; 1 external: n9)
				n7:  +1/3
				n8:  -1/3
				n9:  +1/3
				n10: -1.0
				n11: -1.0
				n12: -1.0

			Bin populations:
				E/I <= -0.33: n1, n2, n3, n8, n10, n11, n12 = 7 (low)
				-0.33 < E/I < 0.33: n5, n6, n7, n9 = 4 (mid)
				E/I >= +0.33: n4 = 1 (high)

			Still no good — high bin has only 1 node. Need to add more
			high-EI nodes. Add cross-community edges making n7 and n9
			have more external than internal so they tip into the high bin.
			Actually they're at +1/3 currently which is just above the
			threshold. Let me check: +1/3 ≈ 0.333 which is >= +0.33, so
			n7 and n9 DO fall in the high bin.

			Recheck: E/I >= +0.33: n4, n7, n9 = 3 (high) — passes!
			And: -0.33 < E/I < 0.33: n5, n6, n8 = 3 (mid) — passes!
				(n8 at -1/3 ≈ -0.333; is this exactly the boundary?
				 -1/3 = -0.333... which is < -0.33 (since -0.333 < -0.33
				 because more negative). So n8 falls in the low bin.)

			Actually -0.33 is the threshold. -1/3 = -0.333... and
			-0.333... < -0.33 is TRUE (more negative). So n8 is in low bin.

			Recheck final:
				Low bin (EI <= -0.33): n1, n2, n3, n8, n10, n11, n12 = 7
				Mid bin (-0.33 < EI < +0.33): n5, n6 = 2 (only!)
				High bin (EI >= +0.33): n4, n7, n9 = 3

			Mid bin still has only 2. Need one more in mid. Add:
				n6-n7 (cross)
					-> n6 updated: 1 internal (n5), 2 external (n9, n7)
					-> E/I = (2-1)/3 = +1/3 -> high bin, not mid! 

			Hmm. Let me try yet another approach. The problem is that
			small modifications keep tipping nodes between bins. Let me
			just systematically add more nodes that fall in the mid bin.

			Add an extra cross-edge n5-n8:
				n5 updated: 1 internal (n6), 2 external (n7, n8)
					E/I = (2-1)/3 = +1/3 -> high bin (not mid)

			This isn't working. The semantic-threshold cuts at ±1/3 are
			naturally producing rare "mid" values. Let me build a different
			fixture: use a small denser subnetwork where mid values are
			natural.

			Simpler approach: build the network with 5 internal edges per
			pure node and 1 external edge per pure-mid node, giving
			E/I = (1 - 5) / 6 = -2/3 ≈ -0.667 which falls in the low bin.

			Or: nodes with 1 internal, 1 external gives E/I = 0. Three
			such nodes are enough for the mid bin. Let me redesign.

			REVISED FIXTURE: build a network where some nodes have exactly
			1 internal and 1 external edge (E/I = 0, mid bin).

			Community 1: n1..n6 with labels 1
			Community 2: n7..n12 with labels 2

			Internal edges:
				n1-n2, n1-n3 (n1 internal degree 2)
				n2-n3 (n2 internal degree 2, n3 internal degree 2)
				n10-n11, n10-n12, n11-n12 (n10, n11, n12 each internal degree 2)

			Cross edges:
				n4-n7  (n4 external 1, n7 external 1)
				n5-n8  (n5 external 1, n8 external 1)
				n6-n9  (n6 external 1, n9 external 1)

			Internal-only edges for nodes 4, 5, 6, 7, 8, 9 (to give them
			internal degree 1 each):
				n4-n5  (n4 internal 1, n5 internal 1)
				n5-n6  (n5 internal 2, n6 internal 1)
				... wait, this is getting complicated.

			Cleanest version: give each of n4, n5, n6, n7, n8, n9 exactly
			one internal and one external edge:
				Edges:
					n1-n2, n1-n3, n2-n3 (cluster in C1)
					n10-n11, n10-n12, n11-n12 (cluster in C2)
					n4-n5 (internal C1)
					n4-n7 (cross)
					n5-n6 (internal C1) -- but then n5 has internal degree 2, not 1
				Hmm.

				Let me just pair up: n4-n5 (internal C1), n6 alone in C1
				gets no internal. Then n4 has 1 internal (to n5), n5 has 1
				internal (to n4), n6 has 0 internal.

				Pair up n7-n8 (internal C2), and n9 alone gets 0 internal.

				Cross edges: n4-n7 (n4 ext 1, n7 ext 1), n5-n8 (n5 ext 1,
				n8 ext 1), n6-n9 (n6 ext 1, n9 ext 1).

				Per-node E/I:
					n1, n2, n3: each 2 internal, 0 external -> -1
					n4: 1 internal (n5), 1 external (n7) -> 0
					n5: 1 internal (n4), 1 external (n8) -> 0
					n6: 0 internal, 1 external (n9) -> +1
					n7: 1 internal (n8), 1 external (n4) -> 0
					n8: 1 internal (n7), 1 external (n5) -> 0
					n9: 0 internal, 1 external (n6) -> +1
					n10, n11, n12: each 2 internal, 0 external -> -1

				Bin populations:
					Low (<= -0.33): n1, n2, n3, n10, n11, n12 = 6
					Mid (-0.33 to 0.33): n4, n5, n7, n8 = 4
					High (>= +0.33): n6, n9 = 2 (only!)

				Still high bin under 3. Need one more in high.

				Add n6 a second external edge: n6-n7 (cross).
					n6 updated: 0 internal, 2 external -> +1 (still high)
					n7 updated: 1 internal (n8), 2 external (n4, n6) -> +1/3 (now high!)

				Updated bin populations:
					Low (<= -0.33): n1, n2, n3, n10, n11, n12 = 6
					Mid (-0.33 to 0.33): n4, n5, n8 = 3 (just makes it)
					High (>= +0.33): n6, n7, n9 = 3 (just makes it)

			Final fixture edges (12 total):
				n1-n2, n1-n3, n2-n3            (C1 cluster)
				n10-n11, n10-n12, n11-n12       (C2 cluster)
				n4-n5                            (C1 internal pair)
				n7-n8                            (C2 internal pair)
				n4-n7, n5-n8, n6-n7, n6-n9       (cross-community edges)

			That's 11 edges. Wait, I had n4-n7 and added n6-n7. Total:
			3 + 3 + 1 + 1 + 4 = 12 edges. Let me recount:
				Cluster C1: n1-n2, n1-n3, n2-n3 = 3 edges
				Cluster C2: n10-n11, n10-n12, n11-n12 = 3 edges
				Internal pairs: n4-n5, n7-n8 = 2 edges
				Cross: n4-n7, n5-n8, n6-n7, n6-n9 = 4 edges
				Total = 12 edges

			Final expected E/I:
				n1: 2 internal (n2, n3), 0 external                  -> -1
				n2: 2 internal (n1, n3), 0 external                  -> -1
				n3: 2 internal (n1, n2), 0 external                  -> -1
				n4: 1 internal (n5), 1 external (n7)                 -> 0
				n5: 1 internal (n4), 1 external (n8)                 -> 0
				n6: 0 internal, 2 external (n7, n9)                  -> +1
				n7: 1 internal (n8), 2 external (n4, n6)             -> +1/3
				n8: 1 internal (n7), 1 external (n5)                 -> 0
				n9: 0 internal, 1 external (n6)                      -> +1
				n10: 2 internal (n11, n12), 0 external               -> -1
				n11: 2 internal (n10, n12), 0 external               -> -1
				n12: 2 internal (n10, n11), 0 external               -> -1

			Bin populations:
				Low (EI <= -0.33): n1, n2, n3, n10, n11, n12 = 6 (>= 3) OK
				Mid (-0.33 < EI < +0.33): n4, n5, n8 = 3 (>= 3) OK
				High (EI >= +0.33): n6, n7, n9 = 3 (>= 3) OK
				All bins satisfy min_nodes_per_ei_bin = 3 -> NO fallback fires

			This is the healthy 2D-supportable fixture for Test 9.
		"""
		src = String[
			"n1", "n1", "n2",                    # C1 cluster
			"n10", "n10", "n11",                 # C2 cluster
			"n4",                                 # C1 internal pair
			"n7",                                 # C2 internal pair
			"n4", "n5", "n6", "n6"               # cross edges
		]
		dst = String[
			"n2", "n3", "n3",
			"n11", "n12", "n12",
			"n5",
			"n8",
			"n7", "n8", "n7", "n9"
		]
		node_ids = ["n$i" for i in 1:12]
		community_labels = vcat(fill(1, 6), fill(2, 6))   # n1..n6 -> C1, n7..n12 -> C2

		#	Hand-computed expected E/I
			one_third = 1.0 / 3.0
			expected_ei = [
				-1.0, -1.0, -1.0,    # n1, n2, n3
				 0.0,  0.0,          # n4, n5
				 1.0,                # n6
				 one_third,          # n7
				 0.0,                # n8
				 1.0,                # n9
				-1.0, -1.0, -1.0     # n10, n11, n12
			]

		edges    = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
		nodes    = DataFrame(id = node_ids, label = node_ids)
		metadata = (directed = false, name = "healthy_2d", weighted = false)

		return (edges = edges, nodes = nodes, metadata = metadata,
				community_labels = community_labels,
				expected_ei = expected_ei)
	end

#	Total Variation distance between two probability distributions
	function _total_variation_distance(p::Vector{Float64}, q::Vector{Float64})
		"""
		Args:
			p::Vector{Float64}: first probability distribution (sums to 1)
			q::Vector{Float64}: second probability distribution (sums to 1)
		Returns:
			Float64: TV distance in [0, 1]
		Notes:
			TV(p, q) = 0.5 * sum_i |p_i - q_i|
			Returns 0 when distributions match exactly, 1 when they share
			no support. Symmetric: TV(p, q) = TV(q, p).
		"""
		length(p) == length(q) || throw(ArgumentError("distributions must have same length"))
		return 0.5 * sum(abs.(p .- q))
	end

#	Bin a vector of node indices under a centrality-based equal-rank K-binning
	function _bin_nodes_by_centrality_ranks(node_indices::Vector{Int},
											 centrality::Vector{Float64},
											 K::Int)
		"""
		Args:
			node_indices::Vector{Int}: indices into the full node set
				whose bin assignments we want
			centrality::Vector{Float64}: per-node centrality (full set)
			K::Int: number of bins
		Returns:
			Vector{Int}: bin assignment (1..K) for each input index
		Notes:
			Replicates the equal-rank binning used by _bin_observed_nodes.
			Used here to bin G_true nodes under their own centrality for
			the calibration baseline.
		"""
		ranks = StatsBase.tiedrank(centrality)
		n = length(centrality)
		return [clamp(Int(ceil((ranks[i] / n) * K)), 1, K) for i in node_indices]
	end

#	Compute the empirical distribution of dropped nodes across K bins
	function _empirical_dropped_distribution(dropped_nodes::Vector{Int},
											  centrality::Vector{Float64},
											  K::Int)
		"""
		Args:
			dropped_nodes::Vector{Int}: indices of dropped nodes
			centrality::Vector{Float64}: G_true centrality (full node set)
			K::Int: number of bins
		Returns:
			Vector{Float64}: length K, sums to 1. Entry b is the fraction
				of dropped nodes that fell in degree bin b under G_true's
				equal-rank binning.
		Notes:
			This is the empirical histogram of where the dropped nodes
			actually came from. The framework's predicted q should
			approximate this distribution when the framework is well-
			calibrated to the missingness mechanism.
		"""
		dropped_bins = _bin_nodes_by_centrality_ranks(dropped_nodes, centrality, K)
		counts = zeros(Float64, K)
		for b in dropped_bins
			counts[b] += 1.0
		end
		n_dropped = length(dropped_nodes)
		return n_dropped > 0 ? counts ./ n_dropped : counts
	end

#	Run R calibration replicates for one (network, rho, rate) cell
	function _calibration_cell(net::NamedTuple,
								target_rho::Float64,
								target_rate::Float64;
								R::Int = 20,
								K::Int = 4,
								J::Int = 3,
								master_seed::Int = 42)
		"""
		Args:
			net::NamedTuple: (edges, nodes, metadata) in corpus format
			target_rho::Float64: nominal centrality-missingness correlation
			target_rate::Float64: nominal missingness rate
			R::Int: number of replicates (default 20)
			K::Int: degree bins for binning (default 4)
			J::Int: E/I bins for binning (default 3)
			master_seed::Int: master seed for the Phase 1 calls
		Returns:
			NamedTuple with fields:
				:cell_id            human-readable identifier
				:per_replicate_tv   Vector{Float64} length R of TV distances
				:realized_rhos      Vector{Float64} length R
				:realized_rates    Vector{Float64} length R
				:n_dropped         Vector{Int} length R
				:n_add_predicted   Vector{Int} length R
				:empirical_pooled  Vector{Float64} length K, pooled empirical
				:q_predicted_mean  Vector{Float64} length K, mean of q across replicates
				:pooled_tv         Float64, TV(empirical_pooled, q_predicted_mean)
				:bin_rank_corr     Float64, Spearman rank corr between distributions
				:n_skipped         Int, replicates skipped due to ceiling_hit / fail
		Notes:
			Implements the calibration experiment for one (rho, rate) cell.
			Per replicate:
			  1. Phase 1: generate_missingness_mask -> dropped_nodes
			  2. Empirical: bin dropped_nodes under G_true centrality
			  3. Materialize G_obs via apply_missingness
			  4. CHAMP on G_obs -> community labels
			  5. Phase 2: compute_setup -> q_predicted
			  6. Compare: TV(empirical_dropped_dist, q_predicted)

			Skips replicates where Phase 1 returns :failed_other.
			Records :ceiling_hit replicates but counts them separately.

			Pooled-empirical computation aggregates all dropped nodes
			across all non-skipped replicates into a single histogram.
			This is the marginal empirical distribution Phase 2 should
			recover on average.
		"""

		#	Pre-compute G_true centrality and binning (reused across replicates)
			centrality_true = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
				net.edges; nodes = net.nodes, directed = net.metadata.directed)

		#	Storage
			per_replicate_tv  = Float64[]
			realized_rhos     = Float64[]
			realized_rates    = Float64[]
			n_dropped         = Int[]
			n_add_predicted   = Int[]
			q_predicted_all   = Vector{Vector{Float64}}()
			all_dropped_bins  = Int[]   # for pooled empirical
			n_skipped         = 0

		#	Per-replicate experiment
			for rep in 1:R
				#	Phase 1: generate dropped-node set
					rep_seed = Int(hash((target_rho, target_rate, rep, master_seed)) % UInt32)
					record = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
						net.edges;
						nodes       = net.nodes,
						directed    = net.metadata.directed,
						target_rate = target_rate,
						target_rho  = target_rho,
						seed        = rep_seed,
						centrality  = centrality_true)

					#	Skip on failure
						if record.bisection_status == :failed_other
							n_skipped += 1
							continue
						end

				#	Empirical: bin dropped nodes under G_true
					dropped = collect(record.dropped_nodes)
					empirical_dist = _empirical_dropped_distribution(
						dropped, centrality_true, K)

					#	Accumulate raw bins for pooled computation
						dropped_bins = _bin_nodes_by_centrality_ranks(
							dropped, centrality_true, K)
						append!(all_dropped_bins, dropped_bins)

				#	Materialize G_obs
					materialized = Network_Credible_Intervals.network_degeneracy.apply_missingness(
						net.edges, dropped; nodes = net.nodes)

				#	CHAMP on G_obs
					champ_result = Network_Credible_Intervals.network_community_detection.champ_community_detection(
						materialized.edges;
						nodes         = materialized.nodes,
						weighted      = net.metadata.weighted,
						directed      = net.metadata.directed,
						seed          = rep_seed + 1000,    # decoupled from Phase 1 seed
						show_progress = false)

				#	Phase 2: compute_setup
					setup = Network_Credible_Intervals.network_reconstruction.compute_setup(
						materialized.edges, materialized.nodes,
						champ_result.membership;
						directed = net.metadata.directed,
						weighted = net.metadata.weighted,
						pi_node  = record.realized_rate,
						pi_edge  = 0.0,
						rho      = record.realized_rho,
						K        = K,
						J        = J)

				#	Compare
					q_predicted = setup.q
					tv = _total_variation_distance(empirical_dist, q_predicted)

				#	Record
					push!(per_replicate_tv, tv)
					push!(realized_rhos, record.realized_rho)
					push!(realized_rates, record.realized_rate)
					push!(n_dropped, length(dropped))
					push!(n_add_predicted, setup.N_add)
					push!(q_predicted_all, q_predicted)
			end

		#	Pooled empirical: aggregate all dropped bins across replicates
			pooled_counts = zeros(Float64, K)
			for b in all_dropped_bins
				pooled_counts[b] += 1.0
			end
			n_total_dropped = length(all_dropped_bins)
			empirical_pooled = n_total_dropped > 0 ?
				pooled_counts ./ n_total_dropped : pooled_counts

		#	Mean predicted across replicates
			if isempty(q_predicted_all)
				q_predicted_mean = zeros(Float64, K)
			else
				q_stack = hcat(q_predicted_all...)   # K x n_reps
				q_predicted_mean = vec(mean(q_stack, dims = 2))
			end

		#	Pooled TV
			pooled_tv = _total_variation_distance(empirical_pooled, q_predicted_mean)

		#	Spearman rank correlation between distributions
			#	Both vectors have length K, so we correlate the bin-wise
			#	probability rankings
				if all(empirical_pooled .== 0.0) || all(q_predicted_mean .== 0.0)
					bin_rank_corr = NaN
				else
					ranks_emp  = StatsBase.tiedrank(empirical_pooled)
					ranks_pred = StatsBase.tiedrank(q_predicted_mean)
					bin_rank_corr = cor(Float64.(ranks_emp), Float64.(ranks_pred))
				end

		#	Return cell results
			cell_id = "rho=$(round(target_rho, digits=2)), rate=$(round(target_rate, digits=2))"
			return (
				cell_id           = cell_id,
				per_replicate_tv  = per_replicate_tv,
				realized_rhos     = realized_rhos,
				realized_rates    = realized_rates,
				n_dropped         = n_dropped,
				n_add_predicted   = n_add_predicted,
				empirical_pooled  = empirical_pooled,
				q_predicted_mean  = q_predicted_mean,
				pooled_tv         = pooled_tv,
				bin_rank_corr     = bin_rank_corr,
				n_skipped         = n_skipped,
			)
	end

#	Print a formatted cell report
	function _print_calibration_cell(cell::NamedTuple, K::Int)
		"""
		Args:
			cell::NamedTuple: result from _calibration_cell
			K::Int: number of bins (for display)
		Returns:
			Nothing
		Notes:
			Formats and prints the per-cell calibration report. Shows
			per-replicate TV summary, the pooled empirical and predicted
			distributions side-by-side, and the pooled TV and rank
			correlation.
		"""
		println("  Cell: $(cell.cell_id)")
		println("  ─" ^ 35)
		n_completed = length(cell.per_replicate_tv)
		println("  Replicates completed: $n_completed (skipped: $(cell.n_skipped))")

		if n_completed == 0
			println("  No completed replicates; no calibration data.")
			return nothing
		end

		#	Per-replicate TV summary
			println()
			println("  Per-replicate TV distances:")
			println(@sprintf("    Mean:   %.4f", mean(cell.per_replicate_tv)))
			println(@sprintf("    Median: %.4f", median(cell.per_replicate_tv)))
			println(@sprintf("    Min:    %.4f", minimum(cell.per_replicate_tv)))
			println(@sprintf("    Max:    %.4f", maximum(cell.per_replicate_tv)))

		#	Realized ρ summary
			println()
			println("  Realized ρ across replicates:")
			println(@sprintf("    Mean:   %.4f", mean(cell.realized_rhos)))
			println(@sprintf("    Range:  [%.4f, %.4f]",
							  minimum(cell.realized_rhos),
							  maximum(cell.realized_rhos)))

		#	N_dropped and N_add comparison
			println()
			println("  N_dropped vs N_add prediction:")
			println(@sprintf("    N_dropped: mean=%.1f, range=[%d, %d]",
							  mean(cell.n_dropped),
							  minimum(cell.n_dropped),
							  maximum(cell.n_dropped)))
			println(@sprintf("    N_add (predicted): mean=%.1f, range=[%d, %d]",
							  mean(cell.n_add_predicted),
							  minimum(cell.n_add_predicted),
							  maximum(cell.n_add_predicted)))

		#	Pooled distribution comparison
			println()
			println("  Pooled bin distribution comparison (K=$K bins):")
			println("    Bin  Empirical  Predicted  Diff")
			for b in 1:K
				emp  = cell.empirical_pooled[b]
				pred = cell.q_predicted_mean[b]
				diff = pred - emp
				println(@sprintf("    %-4d %-10.4f %-10.4f %+.4f",
								  b, emp, pred, diff))
			end

		#	Aggregate metrics
			println()
			println("  Aggregate metrics:")
			println(@sprintf("    Pooled TV distance:  %.4f", cell.pooled_tv))
			rank_str = isnan(cell.bin_rank_corr) ? "NaN (degenerate dist)" :
				@sprintf("%.4f", cell.bin_rank_corr)
			println("    Bin-rank correlation: $rank_str")
		println()

		return nothing
	end

#	Run R replicates for one algorithmic-contract cell
	function _algorithmic_contract_cell(net::NamedTuple,
										  rho_input::Float64,
										  rate_input::Float64;
										  R::Int = 20,
										  K::Int = 4,
										  J::Int = 3,
										  master_seed::Int = 42)
		"""
		Args:
			net::NamedTuple: (edges, nodes, metadata) in corpus format
			rho_input::Float64: nominal target rho for Phase 1
			rate_input::Float64: nominal target rate for Phase 1
			R::Int: number of replicates
			K::Int: degree bins
			J::Int: E/I bins
			master_seed::Int: seed for Phase 1 calls
		Returns:
			NamedTuple with the three prior gates and diagnostic detail.
		Notes:
			Runs R replicates of (Phase 1 -> materialize -> Phase 2 setup).
			Per replicate, checks all three algorithmic contracts and
			records whether they were honored.

			Returns:
			- cell_id: identifier
			- prior_1_pass: all reps satisfied Prior 1 within tolerance
			- prior_2_pass: all reps satisfied Prior 2 within tolerance
			  (or status was :ceiling_hit when the target was out of range)
			- prior_3_pass: all reps satisfied Prior 3 row-stochasticity
			  and matched-empirical
			- per-rep details for diagnostic printing
		"""
		#	Pre-compute G_true centrality and binning (reused across replicates)
			centrality_true = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
				net.edges; nodes = net.nodes, directed = net.metadata.directed)

		#	Storage
			rep_realized_rhos    = Float64[]
			rep_realized_rates   = Float64[]
			rep_n_dropped        = Int[]
			rep_n_add            = Int[]
			rep_beta             = Float64[]
			rep_beta_realized    = Float64[]
			rep_beta_status      = Symbol[]
			rep_prior_1          = Bool[]
			rep_prior_2          = Bool[]
			rep_prior_3          = Bool[]
			rep_ei_row_stochastic = Bool[]
			rep_ei_matches_emp    = Bool[]
			n_skipped             = 0

		#	Tolerances
			tol_prior_1 = 0.02   # |observed_rate - input_rate| < tol
			tol_prior_2 = 1e-3   # |realized_rho_at_beta - input_rho| < tol
			tol_prior_3 = 1e-10  # row stochasticity and empirical match

		#	Per-replicate loop
			for rep in 1:R
				rep_seed = Int(hash((rho_input, rate_input, rep, master_seed)) % UInt32)

				#	Phase 1: generate dropped-node set
					record = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
						net.edges;
						nodes       = net.nodes,
						directed    = net.metadata.directed,
						target_rate = rate_input,
						target_rho  = rho_input,
						seed        = rep_seed,
						centrality  = centrality_true)

					if record.bisection_status == :failed_other
						n_skipped += 1
						continue
					end

				#	Materialize G_obs
					dropped = collect(record.dropped_nodes)
					materialized = Network_Credible_Intervals.network_degeneracy.apply_missingness(
						net.edges, dropped; nodes = net.nodes)

				#	CHAMP on G_obs
					champ_result = Network_Credible_Intervals.network_community_detection.champ_community_detection(
						materialized.edges;
						nodes         = materialized.nodes,
						weighted      = net.metadata.weighted,
						directed      = net.metadata.directed,
						seed          = rep_seed + 1000,
						show_progress = false)

				#	Phase 2: compute_setup with Phase 1's realized values
				#	as inputs (this is the user-specified priors, after
				#	Phase 1 produces them)
					rho_input_to_phase2  = record.realized_rho
					rate_input_to_phase2 = record.realized_rate

					setup = Network_Credible_Intervals.network_reconstruction.compute_setup(
						materialized.edges, materialized.nodes,
						champ_result.membership;
						directed = net.metadata.directed,
						weighted = net.metadata.weighted,
						pi_node  = rate_input_to_phase2,
						pi_edge  = 0.0,
						rho      = rho_input_to_phase2,
						K        = K,
						J        = J)

				#	Prior 1: N_add must satisfy N_add / (N_obs + N_add) ≈ rate_input
					N_obs = nrow(setup.nodes) - length(setup.partially_observed)
					N_add = setup.N_add
					observed_rate = N_add / (N_obs + N_add)
					prior_1_ok = abs(observed_rate - rate_input_to_phase2) < tol_prior_1

				#	Prior 2: re-evaluate _realized_rho_for_beta at solved beta
				#	and compare to rho_input. Ceiling_hit is also acceptable
				#	if rho is outside achievable range.
					realized_at_beta = Network_Credible_Intervals.network_reconstruction._realized_rho_for_beta(
						setup.beta, setup.K, N_obs, N_add)
					if setup.beta_status == :converged
						prior_2_ok = abs(realized_at_beta - rho_input_to_phase2) < tol_prior_2
					elseif setup.beta_status == :ceiling_hit
						#	Ceiling-hit is acceptable: framework correctly
						#	detected target was unreachable.
						prior_2_ok = true
					else
						prior_2_ok = false
					end

				#	Prior 3: ei_conditional must be row-stochastic and match
				#	the empirical conditional distribution in G_obs.
					ei_cond = setup.ei_conditional
					row_sums = sum(ei_cond, dims = 2)
					ei_row_stochastic = all(isapprox.(row_sums, 1.0; atol = tol_prior_3))

				#	Compute empirical conditional from setup.degree_bins and setup.ei_bins
                    empirical_cond = zeros(Float64, K, size(ei_cond, 2))
                    for i in 1:length(setup.degree_bins)
                        b = setup.degree_bins[i]
                        j = setup.ei_bins[i]
                        empirical_cond[b, j] += 1.0
                    end
                    J_eff = size(ei_cond, 2)
                    for b in 1:K
                        rs = sum(empirical_cond[b, :])
                        if rs > 0
                            empirical_cond[b, :] ./= rs
                        else
                            #	Match framework's empty-bin fallback: uniform over J_eff
                                empirical_cond[b, :] .= 1.0 / J_eff
                        end
                    end

					ei_matches_emp = isapprox(ei_cond, empirical_cond; atol = tol_prior_3)
					prior_3_ok = ei_row_stochastic && ei_matches_emp

				#	Record
					push!(rep_realized_rhos, record.realized_rho)
					push!(rep_realized_rates, record.realized_rate)
					push!(rep_n_dropped, length(dropped))
					push!(rep_n_add, N_add)
					push!(rep_beta, setup.beta)
					push!(rep_beta_realized, realized_at_beta)
					push!(rep_beta_status, setup.beta_status)
					push!(rep_prior_1, prior_1_ok)
					push!(rep_prior_2, prior_2_ok)
					push!(rep_prior_3, prior_3_ok)
					push!(rep_ei_row_stochastic, ei_row_stochastic)
					push!(rep_ei_matches_emp, ei_matches_emp)
			end

		#	Aggregate gates: all replicates must honor all priors
			prior_1_pass = !isempty(rep_prior_1) && all(rep_prior_1)
			prior_2_pass = !isempty(rep_prior_2) && all(rep_prior_2)
			prior_3_pass = !isempty(rep_prior_3) && all(rep_prior_3)

			cell_id = "rho=$(round(rho_input, digits=2)), rate=$(round(rate_input, digits=2))"
			return (
				cell_id              = cell_id,
				n_completed          = length(rep_prior_1),
				n_skipped            = n_skipped,
				prior_1_pass         = prior_1_pass,
				prior_2_pass         = prior_2_pass,
				prior_3_pass         = prior_3_pass,
				rep_realized_rhos    = rep_realized_rhos,
				rep_realized_rates   = rep_realized_rates,
				rep_n_dropped        = rep_n_dropped,
				rep_n_add            = rep_n_add,
				rep_beta             = rep_beta,
				rep_beta_realized    = rep_beta_realized,
				rep_beta_status      = rep_beta_status,
				rep_prior_1          = rep_prior_1,
				rep_prior_2          = rep_prior_2,
				rep_prior_3          = rep_prior_3,
				rep_ei_row_stochastic = rep_ei_row_stochastic,
				rep_ei_matches_emp    = rep_ei_matches_emp,
			)
	end

#	Print a formatted contract-cell report
	function _print_contract_cell(cell::NamedTuple)
		println("  Cell: $(cell.cell_id)")
		println("  ─" ^ 35)
		n = cell.n_completed
		println("  Replicates completed: $n (skipped: $(cell.n_skipped))")

		if n == 0
			println("  No completed replicates; no contract data.")
			return nothing
		end

		println()
		println("  Phase 1 realized values across replicates:")
		println(@sprintf("    Mean realized_rho:   %.4f", mean(cell.rep_realized_rhos)))
		println(@sprintf("    Range realized_rho:  [%.4f, %.4f]",
						  minimum(cell.rep_realized_rhos),
						  maximum(cell.rep_realized_rhos)))
		println(@sprintf("    Mean realized_rate:  %.4f", mean(cell.rep_realized_rates)))

		println()
		println("  Per-prior contract check (must hold for ALL replicates):")
		n_pass_1 = count(cell.rep_prior_1)
		n_pass_2 = count(cell.rep_prior_2)
		n_pass_3 = count(cell.rep_prior_3)
		println("    Prior 1 (proportion missing): $n_pass_1 / $n reps  $(cell.prior_1_pass ? "OK" : "FAIL")")
		println("    Prior 2 (centrality corr):    $n_pass_2 / $n reps  $(cell.prior_2_pass ? "OK" : "FAIL")")
		println("    Prior 3 (E/I-given-degree):   $n_pass_3 / $n reps  $(cell.prior_3_pass ? "OK" : "FAIL")")

		println()
		println("  Beta solver diagnostic:")
		println(@sprintf("    Mean solved beta:        %.4f", mean(cell.rep_beta)))
		println(@sprintf("    Mean realized at beta:   %.4f (vs input rho %.4f)",
						  mean(cell.rep_beta_realized),
						  mean(cell.rep_realized_rhos)))
		println("    Status distribution: $(StatsBase.countmap(cell.rep_beta_status))")
		println()

		return nothing
	end

######################
#   FUNCTION TESTS   #
######################

#	Test 1: _compute_observed_centrality on directed star fixture
	function test_compute_centrality_star()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_observed_centrality on the directed Star fixture
			(n=11: 1 hub + 10 leaves, all arcs leaf -> hub).

			Expected on directed star (in-degree + out-degree per design Step 1):
			- Hub (node 1): in-degree = 10, out-degree = 0, total = 10
			- Each leaf (nodes 2-11): in-degree = 0, out-degree = 1, total = 1

			Verifies the function:
			(a) returns a vector of correct length
			(b) returns Float64 (matching the design spec)
			(c) sums in-degree and out-degree for directed networks
			(d) is indexed in nodes-DataFrame order (hub at position 1)
			(e) on undirected star, returns the same per-node degree
				values (sum of incidences regardless of direction)
		"""
		println("─" ^ 70)
		println("Test 1: _compute_observed_centrality on Star fixture (n=11)")
		println("─" ^ 70)

		#	Build directed star fixture
			star_directed = _build_star_fixture(n = 11, directed = true)

		#	Compute centrality (directed)
			cent_directed = _compute_observed_centrality(
				star_directed.edges,
				star_directed.nodes,
				true   # directed
			)

		#	Expected values
			expected_hub = 10.0   # 10 leaves each pointing in
			expected_leaf = 1.0   # 1 outgoing arc each, 0 incoming

		#	Validate directed result
			length_ok = length(cent_directed) == 11
			hub_ok    = cent_directed[1] == expected_hub
			leaves_ok = all(cent_directed[2:end] .== expected_leaf)
			type_ok   = eltype(cent_directed) == Float64

			println("  Directed star (n=11):")
			println("    Length:       $(length(cent_directed)) (expected 11)             $(length_ok ? "OK" : "FAIL")")
			println("    Element type: $(eltype(cent_directed)) (expected Float64)       $(type_ok ? "OK" : "FAIL")")
			println("    Hub cent:     $(cent_directed[1]) (expected $expected_hub)       $(hub_ok ? "OK" : "FAIL")")
			println("    Leaf cent:    all $(expected_leaf)?                              $(leaves_ok ? "OK" : "FAIL")")
			println("    Full vector:  ", cent_directed)

		#	Now do undirected star
			star_undirected = _build_star_fixture(n = 11, directed = false)
			cent_undirected = _compute_observed_centrality(
				star_undirected.edges,
				star_undirected.nodes,
				false   # undirected
			)

		#	On undirected: each edge contributes to BOTH endpoints regardless
		#	of direction, so the totals are identical to the directed case
			und_length_ok = length(cent_undirected) == 11
			und_hub_ok    = cent_undirected[1] == expected_hub
			und_leaves_ok = all(cent_undirected[2:end] .== expected_leaf)

			println()
			println("  Undirected star (n=11):")
			println("    Length:       $(length(cent_undirected)) (expected 11)            $(und_length_ok ? "OK" : "FAIL")")
			println("    Hub cent:     $(cent_undirected[1]) (expected $expected_hub)      $(und_hub_ok ? "OK" : "FAIL")")
			println("    Leaf cent:    all $(expected_leaf)?                               $(und_leaves_ok ? "OK" : "FAIL")")

		#	Final aggregate result
			all_pass = length_ok && hub_ok && leaves_ok && type_ok &&
					   und_length_ok && und_hub_ok && und_leaves_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Test 1
	function run_test_1()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()

		println("=" ^ 70)
		println("Test 1 result: ", t1 ? "PASS" : "FAIL")
		println("=" ^ 70)

		return t1
	end
	run_test_1()

#	Test 2: _compute_observed_centrality on asymmetric directed block
	function test_compute_centrality_asymmetric_block()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_observed_centrality on a directed network
			where in-degree and out-degree are genuinely asymmetric per
			node. Critical for distinguishing the design's "in + out"
			behavior from "in-degree only" or "out-degree only" failure
			modes that would pass the Star test (where one of them is
			zero for every node).

			Expected per-node centrality (in-degree + out-degree):
				n1: 0 + 3 = 3
				n2: 1 + 2 = 3
				n3: 2 + 1 = 3
				n4: 3 + 0 = 3
				n5: 2 + 2 = 4
				n6: 2 + 3 = 5

			n1 and n4 are the diagnostic nodes: if the function returned
			only in-degree, n1 would be 0; if only out-degree, n4 would
			be 0. Both being 3 confirms the sum is happening.

			Sum-of-centralities sanity check: total in + total out =
			2 * n_edges = 22. The sum of the expected vector is
			3+3+3+3+4+5 = 21... let me re-check.
				Sum in:  0+1+2+3+2+2 = 10
				Sum out: 3+2+1+0+2+3 = 11
				Discrepancy: total in (10) != total out (11)
			This is the expected behavior: my edge list has 11 entries,
			so total out = 11 and total in = 11 must both equal 11.
			Re-counting in-degrees from the edge list:
				n1: 0 incoming
				n2: from n1, n6                  -> 2
				n3: from n1, n2, n5              -> 3
				n4: from n1, n2, n3, n6          -> 4
				n5: from n6                       -> 1
				n6: from n5                       -> 1
				Total in: 0+2+3+4+1+1 = 11      OK matches total out=11
			And out-degrees:
				n1: 3 (-> n2, n3, n4)
				n2: 2 (-> n3, n4)
				n3: 1 (-> n4)
				n5: 2 (-> n3, n6)
				n6: 3 (-> n2, n4, n5)
				n4: 0
				Total out: 3+2+1+0+2+3 = 11      OK
			So the corrected expected centralities (in + out):
				n1: 0 + 3 = 3
				n2: 2 + 2 = 4
				n3: 3 + 1 = 4
				n4: 4 + 0 = 4
				n5: 1 + 2 = 3
				n6: 1 + 3 = 4
			Sum: 3+4+4+4+3+4 = 22 = 2 * 11 edges. Sanity check passes.
		"""
		println("─" ^ 70)
		println("Test 2: _compute_observed_centrality on asymmetric directed block")
		println("─" ^ 70)

		#	Build fixture
			fixture = _build_asymmetric_block_fixture()

		#	Compute centrality (directed)
			cent = _compute_observed_centrality(
				fixture.edges,
				fixture.nodes,
				true   # directed
			)

		#	Expected per-node values: in-degree + out-degree
			expected = [3.0, 4.0, 4.0, 4.0, 3.0, 4.0]

		#	Pointwise comparison
			length_ok = length(cent) == 6
			values_ok = cent == expected

		#	Sanity check: sum of centralities should equal 2 * n_edges
			sum_cent = sum(cent)
			n_edges = nrow(fixture.edges)
			sum_ok = sum_cent == 2 * n_edges

		#	Per-node diagnostic output
			println("  Per-node centrality:")
			println("    Node  Got      Expected")
			for (i, id) in enumerate(fixture.nodes.id)
				ok_str = cent[i] == expected[i] ? "OK" : "FAIL"
				println(@sprintf("    %-5s %-8.1f %-8.1f   %s", id, cent[i], expected[i], ok_str))
			end
			println()
			println("    Sum of centralities: $sum_cent (expected $(2 * n_edges) = 2 * $n_edges edges)  $(sum_ok ? "OK" : "FAIL")")

		#	Diagnostic check on n1 and n4 specifically
			#	n1 has only outgoing edges; n4 has only incoming. If the
			#	function were one-directional only, one of them would be
			#	zero. Both being 3 confirms the sum.
				n1_diagnostic = cent[1] > 0   # n1 has no in-degree; only nonzero if out included
				n4_diagnostic = cent[4] > 0   # n4 has no out-degree; only nonzero if in included

				println()
				println("  Diagnostic (verifies in + out, not just one):")
				println("    n1 nonzero (verifies out-degree counted): $(n1_diagnostic ? "OK" : "FAIL")")
				println("    n4 nonzero (verifies in-degree counted):  $(n4_diagnostic ? "OK" : "FAIL")")

		#	Final result
			all_pass = length_ok && values_ok && sum_ok && n1_diagnostic && n4_diagnostic

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Test 1 + Test 2
	function run_tests_through_2()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2])) / 2 tests passed")
		println("=" ^ 70)

		return t1 && t2
	end
	run_tests_through_2()

#	Test 3: _compute_ei_values on all-internal partition
	function test_compute_ei_all_internal()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_ei_values on a network where every edge is
			internal to a community. Built from two 5-node cliques with
			no inter-community edges.

			Expected E/I per node:
				EI = (E - I) / (E + I) = (0 - 4) / (0 + 4) = -1

			Every node has 4 internal edges (to its clique-mates) and 0
			external edges (no inter-clique edges exist). The E/I value
			should be exactly -1.0 for every node.

			Failure modes this test catches:
			- Confusion between "internal" and "external": if the function
			  inverted the E and I counts, every node would return +1.0
			- Off-by-one in counter increment (e.g., only counting from
			  source's perspective, not both endpoints): would still give
			  -1 because the ratio doesn't change with per-edge double-
			  counting as long as it's consistent
			- Misalignment between community_labels indexing and edge
			  endpoint indexing: would produce mixed values, not uniform -1
		"""
		println("─" ^ 70)
		println("Test 3: _compute_ei_values on all-internal partition")
		println("─" ^ 70)

		#	Build fixture
			fixture = _build_all_internal_fixture(n_per_community = 5, n_communities = 2)

		#	Compute E/I
			ei = _compute_ei_values(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels
			)

		#	Expected: all values exactly -1.0
			expected = fill(-1.0, nrow(fixture.nodes))

		#	Validate
			length_ok = length(ei) == 10
			values_ok = all(ei .== -1.0)
			no_nan    = !any(isnan, ei)

		#	Per-node diagnostic output
			println("  Per-node E/I values:")
			println("    Node  Community  Got        Expected")
			for (i, id) in enumerate(fixture.nodes.id)
				ok_str = ei[i] == -1.0 ? "OK" : "FAIL"
				println(@sprintf("    %-5s %-10d %-10.4f %-10.4f   %s",
								  id, fixture.community_labels[i], ei[i], -1.0, ok_str))
			end

		#	Aggregate diagnostics
			println()
			println("  Summary:")
			println("    Length: $(length(ei)) (expected 10)             $(length_ok ? "OK" : "FAIL")")
			println("    All == -1.0:                                    $(values_ok ? "OK" : "FAIL")")
			println("    No NaN entries:                                 $(no_nan ? "OK" : "FAIL")")
			println("    Min: $(minimum(ei)), Max: $(maximum(ei))")

		#	Final result
			all_pass = length_ok && values_ok && no_nan

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-3
	function run_tests_through_3()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3])) / 3 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3
	end
	run_tests_through_3()

#	Test 4: _compute_ei_values on all-external partition
	function test_compute_ei_all_external()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_ei_values on a complete bipartite graph where
			every edge crosses between two communities. The mirror of Test 3.

			Expected E/I per node:
				EI = (E - I) / (E + I) = (E - 0) / (E + 0) = +1

			Every node has E > 0 (all edges go to other community) and
			I = 0 (no within-community edges). E/I = +1 exactly.

			Together with Test 3, this verifies both extremes of the E/I
			scale and catches:
			- Sign inversion (would produce -1 here)
			- Wrong assignment of edges to internal/external counters
			  (would produce mixed values, not uniform +1)
			- Both endpoints not being counted (would still give +1 since
			  the ratio is preserved under consistent double-counting)
		"""
		println("─" ^ 70)
		println("Test 4: _compute_ei_values on all-external partition")
		println("─" ^ 70)

		#	Build fixture (4 × 5 complete bipartite)
			fixture = _build_all_external_fixture(n_group_a = 4, n_group_b = 5)

		#	Compute E/I
			ei = _compute_ei_values(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels
			)

		#	Expected: all values exactly +1.0
			expected = fill(1.0, nrow(fixture.nodes))

		#	Validate
			length_ok = length(ei) == 9
			values_ok = all(ei .== 1.0)
			no_nan    = !any(isnan, ei)

		#	Per-node diagnostic output
			println("  Per-node E/I values:")
			println("    Node  Community  Got        Expected")
			for (i, id) in enumerate(fixture.nodes.id)
				ok_str = ei[i] == 1.0 ? "OK" : "FAIL"
				println(@sprintf("    %-5s %-10d %-10.4f %-10.4f   %s",
								  id, fixture.community_labels[i], ei[i], 1.0, ok_str))
			end

		#	Aggregate diagnostics
			println()
			println("  Summary:")
			println("    Length: $(length(ei)) (expected 9)              $(length_ok ? "OK" : "FAIL")")
			println("    All == +1.0:                                    $(values_ok ? "OK" : "FAIL")")
			println("    No NaN entries:                                 $(no_nan ? "OK" : "FAIL")")
			println("    Min: $(minimum(ei)), Max: $(maximum(ei))")

		#	Final result
			all_pass = length_ok && values_ok && no_nan

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-4
	function run_tests_through_4()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4])) / 4 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4
	end
	run_tests_through_4()

#	Test 5: _compute_ei_values on block-with-bridges mixed partition
	function test_compute_ei_mixed()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_ei_values on a 6-node network with two 3-node
			cliques connected by 4 bridge edges. Different nodes have
			different E/I values, all hand-computed.

			Expected per-node E/I:
				n1 (bridge-balanced): 0.000
				n2 (mostly internal):  -1/3 = -0.333...
				n3 (mostly internal):  -1/3 = -0.333...
				n4 (bridge-balanced): 0.000
				n5 (mostly internal):  -1/3 = -0.333...
				n6 (mostly internal):  -1/3 = -0.333...

			This test is the critical one for E/I: Tests 3 and 4 verified
			the extremes (-1 and +1), but a function could be correct at
			extremes and wrong in the interior. Test 5 verifies arithmetic
			on a mixed case with three distinct values (0, -1/3, and the
			implicit verification that not all values are -1 or +1).

			Tolerance: floating-point equality is unsafe for -1/3, so we
			use isapprox with atol=1e-12 (well below any reasonable
			arithmetic precision concern).
		"""
		println("─" ^ 70)
		println("Test 5: _compute_ei_values on block-with-bridges (mixed)")
		println("─" ^ 70)

		#	Build fixture
			fixture = _build_block_with_bridges_fixture()

		#	Compute E/I
			ei = _compute_ei_values(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels
			)

		#	Validate
			length_ok = length(ei) == 6
			values_ok = all(isapprox.(ei, fixture.expected_ei; atol = 1e-12))
			no_nan    = !any(isnan, ei)

		#	Per-node diagnostic output
			println("  Per-node E/I values:")
			println("    Node  Community  Got        Expected   Match")
			for (i, id) in enumerate(fixture.nodes.id)
				match = isapprox(ei[i], fixture.expected_ei[i]; atol = 1e-12)
				ok_str = match ? "OK" : "FAIL"
				println(@sprintf("    %-5s %-10d %-10.6f %-10.6f %s",
								  id, fixture.community_labels[i],
								  ei[i], fixture.expected_ei[i], ok_str))
			end

		#	Distinct-values check: verify the function returns values that
		#	cover both 0 and -1/3, not all the same value
			unique_ei = unique(round.(ei; digits = 10))
			values_distinct = length(unique_ei) >= 2

		#	Aggregate diagnostics
			println()
			println("  Summary:")
			println("    Length: $(length(ei)) (expected 6)               $(length_ok ? "OK" : "FAIL")")
			println("    All values match expected:                       $(values_ok ? "OK" : "FAIL")")
			println("    At least 2 distinct values (not all same):       $(values_distinct ? "OK" : "FAIL")")
			println("    No NaN entries:                                  $(no_nan ? "OK" : "FAIL")")
			println("    Min: $(round(minimum(ei); digits=4)), Max: $(round(maximum(ei); digits=4))")

		#	Final result
			all_pass = length_ok && values_ok && values_distinct && no_nan

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-5
	function run_tests_through_5()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()
		t5 = test_compute_ei_mixed()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4, t5])) / 5 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4 && t5
	end
	run_tests_through_5()

#	Test 6: _detect_community_structure triggers :single_community fallback
	function test_detect_single_community()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _detect_community_structure correctly identifies the
			single-community case and falls back to degree-only binning.

			Construction: the all-internal fixture from Test 3 (two cliques)
			but with every node labeled as community 1 — i.e., a single
			community covering all 10 nodes. Since there is no community
			structure to leverage, the function must:
			- Return binning_mode = :degree_only
			- Return fallback_reason = :single_community
			- Still compute valid E/I values (uniform -1 for all nodes,
			  since every edge is now technically "internal" to the
			  single community)

			This tests the first of the three fallback paths in the
			function: single_community, too_few_nodes (next test), and
			insufficient_ei_variance.
		"""
		println("─" ^ 70)
		println("Test 6: _detect_community_structure -> :single_community fallback")
		println("─" ^ 70)

		#	Build the all-internal fixture but override labels to all 1s
			fixture = _build_all_internal_fixture(n_per_community = 5, n_communities = 2)
			single_community_labels = fill(1, nrow(fixture.nodes))

		#	Call the function
			result = _detect_community_structure(
				fixture.edges,
				fixture.nodes,
				single_community_labels;
				J = 3,
				min_nodes_per_ei_bin = 3
			)

		#	Expected:
			#	- binning_mode == :degree_only
			#	- fallback_reason == :single_community
			#	- ei_values is a vector of -1.0 (all edges are within
			#	  the single community, so all edges are technically
			#	  internal). It's computed even though the fallback
			#	  fires, because the function needs to return E/I for
			#	  diagnostic purposes.
				mode_ok    = result.binning_mode == :degree_only
				reason_ok  = result.fallback_reason == :single_community
				ei_length  = length(result.ei_values) == 10
				ei_value   = all(result.ei_values .== -1.0)

		#	Diagnostic output
			println("  Returned binning_mode:    $(result.binning_mode)")
			println("  Expected binning_mode:    :degree_only")
			println("  Mode match:               $(mode_ok ? "OK" : "FAIL")")
			println()
			println("  Returned fallback_reason: $(result.fallback_reason)")
			println("  Expected fallback_reason: :single_community")
			println("  Reason match:             $(reason_ok ? "OK" : "FAIL")")
			println()
			println("  E/I vector length:        $(length(result.ei_values)) (expected 10)  $(ei_length ? "OK" : "FAIL")")
			println("  E/I all -1.0:             $(ei_value ? "OK" : "FAIL")")
			println("  E/I min/max:              $(minimum(result.ei_values)) / $(maximum(result.ei_values))")

		#	Final result
			all_pass = mode_ok && reason_ok && ei_length && ei_value

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-6
	function run_tests_through_6()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()
		t5 = test_compute_ei_mixed()
		t6 = test_detect_single_community()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4, t5, t6])) / 6 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4 && t5 && t6
	end
	run_tests_through_6()
 
#	Test 7: _detect_community_structure triggers :too_few_nodes fallback
	function test_detect_too_few_nodes()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _detect_community_structure correctly identifies the
			too-few-defined-EI case and falls back to degree-only binning.

			Construction: 12 nodes, only 4 connected (in a small two-
			community subnetwork). The 8 isolates have NaN E/I values.

			With J=3 and min_nodes_per_ei_bin=3, the threshold for
			supportable 2D binning is J * min_nodes = 9 defined nodes.
			With only 4 defined nodes, the function must:
			- Return binning_mode = :degree_only
			- Return fallback_reason = :too_few_nodes
			- Still compute E/I values (4 defined, 8 NaN)

			This tests the second of the three fallback paths. The fixture
			has two communities (so :single_community does not fire) but
			too few connected nodes (so :too_few_nodes fires before the
			:insufficient_ei_variance check is reached).
		"""
		println("─" ^ 70)
		println("Test 7: _detect_community_structure -> :too_few_nodes fallback")
		println("─" ^ 70)

		#	Build fixture
			fixture = _build_mostly_isolates_fixture(n_total = 12, n_connected = 4)

		#	Call the function
			result = _detect_community_structure(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels;
				J = 3,
				min_nodes_per_ei_bin = 3
			)

		#	Expected behavior
			mode_ok       = result.binning_mode == :degree_only
			reason_ok     = result.fallback_reason == :too_few_nodes
			ei_length     = length(result.ei_values) == 12
			n_defined     = count(!isnan, result.ei_values)
			n_defined_ok  = n_defined == 4
			n_nan         = count(isnan, result.ei_values)
			n_nan_ok      = n_nan == 8

		#	Diagnostic output
			println("  Returned binning_mode:    $(result.binning_mode)")
			println("  Expected binning_mode:    :degree_only")
			println("  Mode match:               $(mode_ok ? "OK" : "FAIL")")
			println()
			println("  Returned fallback_reason: $(result.fallback_reason)")
			println("  Expected fallback_reason: :too_few_nodes")
			println("  Reason match:             $(reason_ok ? "OK" : "FAIL")")
			println()
			println("  E/I vector length:        $(length(result.ei_values)) (expected 12)         $(ei_length ? "OK" : "FAIL")")
			println("  E/I defined (non-NaN):    $n_defined (expected 4)                $(n_defined_ok ? "OK" : "FAIL")")
			println("  E/I NaN (isolates):       $n_nan (expected 8)                    $(n_nan_ok ? "OK" : "FAIL")")
			println()
			println("  Per-node E/I:")
			for (i, id) in enumerate(fixture.nodes.id)
				val = result.ei_values[i]
				val_str = isnan(val) ? "NaN" : @sprintf("%.4f", val)
				println(@sprintf("    %-5s community=%d  EI=%s", id, fixture.community_labels[i], val_str))
			end

		#	Final result
			all_pass = mode_ok && reason_ok && ei_length && n_defined_ok && n_nan_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-7
	function run_tests_through_7()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()
		t5 = test_compute_ei_mixed()
		t6 = test_detect_single_community()
		t7 = test_detect_too_few_nodes()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4, t5, t6, t7])) / 7 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4 && t5 && t6 && t7
	end
	run_tests_through_7()

#	Test 8: _detect_community_structure triggers :insufficient_ei_variance fallback
	function test_detect_insufficient_ei_variance()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _detect_community_structure correctly identifies the
			insufficient-EI-variance case and falls back to degree-only
			binning.

			Construction: reuses the all-internal fixture from Test 3 (two
			5-node cliques, 0 bridge edges). All 10 nodes have:
			- Defined E/I (every node has 4 internal edges; no isolates)
			- E/I = -1.0 (no external edges)
			- Community labels indicating two distinct communities

			Under default J=3 with semantic thresholds (-0.33, +0.33):
			- E/I <= -0.33: all 10 nodes
			- -0.33 < E/I < 0.33: 0 nodes
			- E/I >= +0.33: 0 nodes
			- min across bins: 0, which is < min_nodes_per_ei_bin = 3

			So the function must:
			- Pass the :single_community check (2 communities exist)
			- Pass the :too_few_nodes check (10 defined > 9 threshold)
			- Fail the :insufficient_ei_variance check (middle and high
			  bins are empty)
			- Return binning_mode = :degree_only
			- Return fallback_reason = :insufficient_ei_variance

			This tests the third and final fallback path, completing
			coverage of all three branches in the fallback decision.

			Real-world relevance: this is the fallback condition that
			fires most often on the corpus. Networks like Synthetic 1 (SBM)
			are constructed to be modular, meaning most nodes are pure
			internal members and the broker / mixed bins are sparsely
			populated. The fallback ensures the framework handles these
			cases gracefully rather than producing degenerate 2D matrices
			where some E/I bins have zero nodes.
		"""
		println("─" ^ 70)
		println("Test 8: _detect_community_structure -> :insufficient_ei_variance fallback")
		println("─" ^ 70)

		#	Reuse the all-internal fixture from Test 3
			fixture = _build_all_internal_fixture(n_per_community = 5, n_communities = 2)

		#	Call the function with default J=3 and min_nodes_per_ei_bin=3
			result = _detect_community_structure(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels;
				J = 3,
				min_nodes_per_ei_bin = 3
			)

		#	Expected behavior
			mode_ok       = result.binning_mode == :degree_only
			reason_ok     = result.fallback_reason == :insufficient_ei_variance
			ei_length     = length(result.ei_values) == 10
			ei_all_neg1   = all(result.ei_values .== -1.0)
			no_nan        = !any(isnan, result.ei_values)

		#	Manually compute the per-bin counts to verify the fallback
		#	condition is what we think it is
			ei = result.ei_values
			n_lo  = count(x -> !isnan(x) && x <= -0.33, ei)
			n_mid = count(x -> !isnan(x) && -0.33 < x < 0.33, ei)
			n_hi  = count(x -> !isnan(x) && x >= 0.33, ei)
			bins_correct = n_lo == 10 && n_mid == 0 && n_hi == 0

		#	Diagnostic output
			println("  Returned binning_mode:    $(result.binning_mode)")
			println("  Expected binning_mode:    :degree_only")
			println("  Mode match:               $(mode_ok ? "OK" : "FAIL")")
			println()
			println("  Returned fallback_reason: $(result.fallback_reason)")
			println("  Expected fallback_reason: :insufficient_ei_variance")
			println("  Reason match:             $(reason_ok ? "OK" : "FAIL")")
			println()
			println("  E/I vector length:        $(length(result.ei_values)) (expected 10)  $(ei_length ? "OK" : "FAIL")")
			println("  E/I all -1.0:             $(ei_all_neg1 ? "OK" : "FAIL")")
			println("  No NaN entries:           $(no_nan ? "OK" : "FAIL")")
			println()
			println("  Manual bin populations (J=3 semantic thresholds):")
			println("    E/I <= -0.33:  $n_lo nodes  (expected 10)")
			println("    -0.33 < E/I < 0.33:  $n_mid nodes (expected 0)")
			println("    E/I >= 0.33:  $n_hi nodes  (expected 0)")
			println("    Bin populations match expected: $(bins_correct ? "OK" : "FAIL")")
			println("    Min across bins: $(min(n_lo, n_mid, n_hi)) (must be < min_nodes_per_ei_bin = 3 to trigger fallback)")

		#	Final result
			all_pass = mode_ok && reason_ok && ei_length && ei_all_neg1 && no_nan && bins_correct

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-8
	function run_tests_through_8()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()
		t5 = test_compute_ei_mixed()
		t6 = test_detect_single_community()
		t7 = test_detect_too_few_nodes()
		t8 = test_detect_insufficient_ei_variance()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4, t5, t6, t7, t8])) / 8 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8
	end
	run_tests_through_8()

#	Test 9: _detect_community_structure stays in :two_dimensional on healthy fixture
	function test_detect_two_dimensional_active()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _detect_community_structure correctly identifies a
			network with healthy structure and returns :two_dimensional
			binning mode (no fallback).

			Construction: 12 nodes, two 6-node communities, 12 edges
			engineered to produce E/I values populating all three
			semantic bins.

			Expected bin populations (J=3, min_nodes_per_ei_bin=3):
				Low (EI <= -0.33): n1, n2, n3, n10, n11, n12 = 6 nodes
				Mid (-0.33 < EI < +0.33): n4, n5, n8 = 3 nodes
				High (EI >= +0.33): n6, n7, n9 = 3 nodes

			All three bins meet the threshold of 3 minimum, so 2D mode
			should remain active. The function must:
			- Pass the :single_community check (2 communities present)
			- Pass the :too_few_nodes check (12 defined E/I values, >= 9)
			- Pass the :insufficient_ei_variance check (min bin = 3)
			- Return binning_mode = :two_dimensional
			- Return fallback_reason = nothing

			This is the positive complement to Tests 6, 7, and 8.
			Without it, the decision logic could in principle always
			fall back (a buggy function that always returns
			:degree_only would pass Tests 6, 7, 8). This test verifies
			the function returns the active 2D mode when conditions
			support it.
		"""
		println("─" ^ 70)
		println("Test 9: _detect_community_structure -> :two_dimensional (no fallback)")
		println("─" ^ 70)

		#	Build fixture
			fixture = _build_healthy_2d_fixture()

		#	Call the function
			result = _detect_community_structure(
				fixture.edges,
				fixture.nodes,
				fixture.community_labels;
				J = 3,
				min_nodes_per_ei_bin = 3
			)

		#	Expected behavior
			mode_ok       = result.binning_mode == :two_dimensional
			reason_ok     = result.fallback_reason === nothing
			ei_length     = length(result.ei_values) == 12
			no_nan        = !any(isnan, result.ei_values)
			ei_values_ok  = all(isapprox.(result.ei_values, fixture.expected_ei; atol = 1e-12))

		#	Manually verify the bin populations are what we engineered
			ei = result.ei_values
			n_lo  = count(x -> !isnan(x) && x <= -0.33, ei)
			n_mid = count(x -> !isnan(x) && -0.33 < x < 0.33, ei)
			n_hi  = count(x -> !isnan(x) && x >= 0.33, ei)
			bins_correct = n_lo >= 3 && n_mid >= 3 && n_hi >= 3

		#	Diagnostic output
			println("  Returned binning_mode:    $(result.binning_mode)")
			println("  Expected binning_mode:    :two_dimensional")
			println("  Mode match:               $(mode_ok ? "OK" : "FAIL")")
			println()
			println("  Returned fallback_reason: $(result.fallback_reason)")
			println("  Expected fallback_reason: nothing")
			println("  Reason match:             $(reason_ok ? "OK" : "FAIL")")
			println()
			println("  E/I vector length:        $(length(result.ei_values)) (expected 12)  $(ei_length ? "OK" : "FAIL")")
			println("  E/I values match expected: $(ei_values_ok ? "OK" : "FAIL")")
			println("  No NaN entries:           $(no_nan ? "OK" : "FAIL")")
			println()
			println("  Per-node E/I:")
			for (i, id) in enumerate(fixture.nodes.id)
				ok_str = isapprox(result.ei_values[i], fixture.expected_ei[i]; atol = 1e-12) ? "OK" : "FAIL"
				println(@sprintf("    %-5s community=%d  got=%.4f  expected=%.4f  %s",
								  id, fixture.community_labels[i],
								  result.ei_values[i], fixture.expected_ei[i], ok_str))
			end
			println()
			println("  Manual bin populations (J=3 semantic thresholds):")
			println("    E/I <= -0.33:  $n_lo nodes  (need >= 3 to stay 2D)")
			println("    -0.33 < E/I < 0.33:  $n_mid nodes  (need >= 3 to stay 2D)")
			println("    E/I >= 0.33:  $n_hi nodes  (need >= 3 to stay 2D)")
			println("    All bins meet threshold: $(bins_correct ? "OK" : "FAIL")")

		#	Final result
			all_pass = mode_ok && reason_ok && ei_length && ei_values_ok && no_nan && bins_correct

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-9
	function run_tests_through_9()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1 = test_compute_centrality_star()
		t2 = test_compute_centrality_asymmetric_block()
		t3 = test_compute_ei_all_internal()
		t4 = test_compute_ei_all_external()
		t5 = test_compute_ei_mixed()
		t6 = test_detect_single_community()
		t7 = test_detect_too_few_nodes()
		t8 = test_detect_insufficient_ei_variance()
		t9 = test_detect_two_dimensional_active()

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, [t1, t2, t3, t4, t5, t6, t7, t8, t9])) / 9 tests passed")
		println("=" ^ 70)

		return t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9
	end
	run_tests_through_9()

#	Test 10: _bin_observed_nodes produces equal-rank K degree bins
	function test_bin_degree_equal_rank()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _bin_observed_nodes partitions nodes into K equal-rank
			degree bins, with bin 1 the most peripheral and bin K the most
			central.

			Construction: 20 nodes with centralities 1.0..20.0 (strictly
			increasing, no ties). With K=4, each bin should contain
			exactly 5 nodes:
				Bin 1: ranks 1..5  -> centrality values 1..5
				Bin 2: ranks 6..10 -> centrality values 6..10
				Bin 3: ranks 11..15 -> centrality values 11..15
				Bin 4: ranks 16..20 -> centrality values 16..20

			Since binning_mode is :degree_only in this test, the E/I
			dimension collapses and J_effective should be 1. All ei_bins
			should be 1.

			The function uses tiedrank for rank computation, so:
			- No-tie input produces ranks 1, 2, ..., N
			- The ceiling-division logic maps rank r to bin ceil(r/N * K)
			- With N=20, K=4: ranks 1..5 -> bin 1, 6..10 -> bin 2, etc.

			Failure modes this test catches:
			- Wrong bin direction (bin 1 most central instead of most
			  peripheral): would produce reversed mapping
			- Off-by-one in ceiling division: would produce, e.g., 4
			  nodes in bin 1 and 6 in bin K
			- Wrong J_effective: would return J=3 instead of J=1 in
			  degree-only mode
		"""
		println("─" ^ 70)
		println("Test 10: _bin_observed_nodes -> equal-rank K degree bins")
		println("─" ^ 70)

		#	Construct centrality vector with no ties: 1.0, 2.0, ..., 20.0
			n = 20
			K = 4
			J = 3   # passed but should collapse to J_effective = 1 in degree-only mode
			centrality = Float64.(1:n)
			ei_values  = fill(NaN, n)   # arbitrary; in degree-only mode they aren't used

		#	Call the function
			result = _bin_observed_nodes(centrality, ei_values, K, J, :degree_only)

		#	Expected bin assignments
			expected_degree_bins = Int[]
			for i in 1:n
				#	rank i goes to bin ceil(i/N * K) = ceil(i/5)
					push!(expected_degree_bins, Int(ceil(i / n * K)))
			end

		#	Expected: J_effective = 1, all ei_bins = 1
			expected_ei_bins = ones(Int, n)
			expected_J_eff   = 1

		#	Validate
			degree_bins_ok = result.degree_bins == expected_degree_bins
			ei_bins_ok     = result.ei_bins == expected_ei_bins
			J_eff_ok       = result.J_effective == expected_J_eff

		#	Bin size check: each bin should have exactly 5 nodes
			bin_sizes = [count(==(b), result.degree_bins) for b in 1:K]
			bin_size_ok = all(bin_sizes .== n ÷ K)

		#	Direction check: highest-centrality node should be in bin K,
		#	lowest in bin 1
			highest_idx = argmax(centrality)   # node 20
			lowest_idx  = argmin(centrality)   # node 1
			direction_high_ok = result.degree_bins[highest_idx] == K
			direction_low_ok  = result.degree_bins[lowest_idx] == 1

		#	Diagnostic output
			println("  N = $n, K = $K, J = $J (degree-only mode)")
			println()
			println("  Per-node bin assignments:")
			println("    Idx  Centrality  DegreeBin  Expected  Match")
			for i in 1:n
				ok_str = result.degree_bins[i] == expected_degree_bins[i] ? "OK" : "FAIL"
				println(@sprintf("    %-4d %-10.1f %-10d %-9d %s",
								  i, centrality[i], result.degree_bins[i],
								  expected_degree_bins[i], ok_str))
			end

			println()
			println("  Bin size distribution:")
			for b in 1:K
				println("    Bin $b: $(bin_sizes[b]) nodes (expected $(n ÷ K))")
			end
			println("    All bins same size: $(bin_size_ok ? "OK" : "FAIL")")

			println()
			println("  Direction checks:")
			println("    Highest-centrality node (idx $highest_idx, value $(centrality[highest_idx])) in bin $(result.degree_bins[highest_idx]) (expected $K): $(direction_high_ok ? "OK" : "FAIL")")
			println("    Lowest-centrality node (idx $lowest_idx, value $(centrality[lowest_idx])) in bin $(result.degree_bins[lowest_idx]) (expected 1): $(direction_low_ok ? "OK" : "FAIL")")

			println()
			println("  E/I bin assignments in :degree_only mode:")
			println("    All ei_bins = 1: $(ei_bins_ok ? "OK" : "FAIL")")
			println("    J_effective = $(result.J_effective) (expected $expected_J_eff): $(J_eff_ok ? "OK" : "FAIL")")

		#	Final result
			all_pass = degree_bins_ok && ei_bins_ok && J_eff_ok &&
					   bin_size_ok && direction_high_ok && direction_low_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-10
	function run_tests_through_10()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_10()

#	Test 11: _bin_observed_nodes uses semantic thresholds at J=3
	function test_bin_ei_semantic_thresholds()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _bin_observed_nodes correctly applies semantic E/I
			thresholds in 2D binning mode with J=3.

			Design specification:
				EI <= -0.33: bin 1 (internal hub)
				-0.33 < EI < 0.33: bin 2 (mixed)
				EI >= 0.33: bin 3 (broker)

			Construction: 12 nodes with hand-set E/I values spanning the
			full [-1, +1] range, including boundary cases at exactly
			-0.33 and +0.33:

				Idx  EI       Expected EI bin
				1    -1.0     1 (well below -0.33)
				2    -0.5     1 (below -0.33)
				3    -0.33    1 (boundary; <= -0.33 fires)
				4    -0.30    2 (just above -0.33)
				5    -0.10    2 (negative-but-mid)
				6     0.0     2 (center)
				7     0.10    2 (positive-but-mid)
				8     0.30    2 (just below +0.33)
				9     0.33    3 (boundary; >= +0.33 fires)
				10    0.50    3 (above +0.33)
				11    0.99    3 (near max)
				12    NaN     2 (isolate; design specifies middle bin)

			Centrality is set to 1..12 to give unambiguous rank order;
			the degree-bin assignment is incidental here.

			Test exercises:
			- Both threshold boundaries (-0.33 hits bin 1, +0.33 hits bin 3)
			- Internal ordering within each bin (multiple nodes per bin)
			- NaN handling: isolate gets the middle bin per design
			- J_effective stays at J=3 (not collapsed) in 2D mode
		"""
		println("─" ^ 70)
		println("Test 11: _bin_observed_nodes -> J=3 semantic E/I thresholds")
		println("─" ^ 70)

		#	Construct hand-set E/I vector spanning the full range with
		#	boundary cases
			n = 12
			K = 4   # degree bins; not the focus of this test
			J = 3
			centrality = Float64.(1:n)
			ei_values  = [-1.0, -0.5, -0.33, -0.30, -0.10, 0.0,
						  0.10, 0.30, 0.33, 0.50, 0.99, NaN]

		#	Call the function in 2D mode
			result = _bin_observed_nodes(centrality, ei_values, K, J, :two_dimensional)

		#	Expected E/I bin assignments per the design's threshold rules
			expected_ei_bins = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2]

		#	Validate E/I bins
			ei_bins_ok = result.ei_bins == expected_ei_bins
			J_eff_ok   = result.J_effective == J

		#	Sanity check: each E/I bin should have its expected node count
			n_lo  = count(==(1), result.ei_bins)
			n_mid = count(==(2), result.ei_bins)
			n_hi  = count(==(3), result.ei_bins)
			lo_expected_count  = 3   # indices 1, 2, 3
			mid_expected_count = 6   # indices 4, 5, 6, 7, 8, 12 (NaN -> mid)
			hi_expected_count  = 3   # indices 9, 10, 11
			counts_ok = (n_lo == lo_expected_count) &&
						(n_mid == mid_expected_count) &&
						(n_hi == hi_expected_count)

		#	Boundary verification: -0.33 should fall in bin 1, +0.33 in bin 3
			boundary_neg_ok = result.ei_bins[3] == 1   # E/I = -0.33 -> bin 1
			boundary_pos_ok = result.ei_bins[9] == 3   # E/I = +0.33 -> bin 3

		#	NaN handling: isolate should be assigned to middle bin (bin 2 for J=3)
			nan_handling_ok = result.ei_bins[12] == 2

		#	Diagnostic output
			println("  N = $n, K = $K, J = $J (two_dimensional mode)")
			println()
			println("  Per-node E/I bin assignments:")
			println("    Idx  EI        EIBin  Expected  Match")
			for i in 1:n
				ei_str = isnan(ei_values[i]) ? "NaN" : @sprintf("%.4f", ei_values[i])
				ok_str = result.ei_bins[i] == expected_ei_bins[i] ? "OK" : "FAIL"
				println(@sprintf("    %-4d %-9s %-6d %-9d %s",
								  i, ei_str, result.ei_bins[i],
								  expected_ei_bins[i], ok_str))
			end

			println()
			println("  Bin population summary:")
			println("    Bin 1 (EI <= -0.33):       $n_lo nodes (expected $lo_expected_count)")
			println("    Bin 2 (-0.33 < EI < 0.33): $n_mid nodes (expected $mid_expected_count)")
			println("    Bin 3 (EI >= +0.33):       $n_hi nodes (expected $hi_expected_count)")
			println("    Counts match expected: $(counts_ok ? "OK" : "FAIL")")

			println()
			println("  Boundary checks:")
			println("    EI = -0.33 (idx 3) -> bin $(result.ei_bins[3]) (expected 1): $(boundary_neg_ok ? "OK" : "FAIL")")
			println("    EI = +0.33 (idx 9) -> bin $(result.ei_bins[9]) (expected 3): $(boundary_pos_ok ? "OK" : "FAIL")")

			println()
			println("  NaN handling:")
			println("    EI = NaN (idx 12) -> bin $(result.ei_bins[12]) (expected 2 = middle): $(nan_handling_ok ? "OK" : "FAIL")")

			println()
			println("  J_effective: $(result.J_effective) (expected $J): $(J_eff_ok ? "OK" : "FAIL")")

		#	Final result
			all_pass = ei_bins_ok && J_eff_ok && counts_ok &&
					   boundary_neg_ok && boundary_pos_ok && nan_handling_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-11
	function run_tests_through_11()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_11()

#	Test 12: _compute_p_matrix with Laplace smoothing on known counts
	function test_compute_p_matrix_laplace()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_p_matrix correctly computes the Laplace-
			smoothed rank-rank connectivity matrix P.

			Construction: a 4-node directed network split into K=2 degree
			bins of 2 nodes each (J=1 for degree-only mode). Hand-set bin
			assignments:
				n1 -> bin 1, n2 -> bin 1, n3 -> bin 2, n4 -> bin 2

			Edges (directed):
				n1 -> n2  (within bin 1, src in bin 1, dst in bin 1)
				n1 -> n3  (cross, bin 1 to bin 2)
				n3 -> n1  (cross, bin 2 to bin 1)
				n3 -> n4  (within bin 2, src in bin 2, dst in bin 2)

			Total 4 edges. Hand-compute the cell-pair edge counts:
				edge_count[1, 1, 1, 1]: edges within bin 1
					n1 -> n2: src=bin 1, dst=bin 1 -> 1 edge
					Total: 1 edge
				edge_count[1, 1, 2, 1]: edges from bin 1 to bin 2
					n1 -> n3: src=bin 1, dst=bin 2 -> 1 edge
					Total: 1 edge
				edge_count[2, 1, 1, 1]: edges from bin 2 to bin 1
					n3 -> n1: src=bin 2, dst=bin 1 -> 1 edge
					Total: 1 edge
				edge_count[2, 1, 2, 1]: edges within bin 2
					n3 -> n4: src=bin 2, dst=bin 2 -> 1 edge
					Total: 1 edge

			Dyad counts (directed, respondent-respondent):
				Within bin 1: 2 nodes -> 2 * (2-1) = 2 ordered dyads
				Within bin 2: 2 nodes -> 2 * (2-1) = 2 ordered dyads
				Cross 1->2: 2 * 2 = 4 ordered dyads
				Cross 2->1: 2 * 2 = 4 ordered dyads

			Laplace-smoothed P:
				P[1, 1, 1, 1] = (1 + 1) / (2 + 2) = 2/4 = 0.5
				P[1, 1, 2, 1] = (1 + 1) / (4 + 2) = 2/6 = 1/3 ≈ 0.3333
				P[2, 1, 1, 1] = (1 + 1) / (4 + 2) = 2/6 = 1/3 ≈ 0.3333
				P[2, 1, 2, 1] = (1 + 1) / (2 + 2) = 2/4 = 0.5

			All four cell pairs have known counts. The function should
			produce these exact values. Empty cell pairs would default
			to (0 + 1)/(n_dyads + 2), but we hit all four pairs in this
			fixture so no empty cells exist.

			The weight matrix w should be:
				w[1, 1, 1, 1] = 1.0 (single edge of weight 1)
				w[1, 1, 2, 1] = 1.0
				w[2, 1, 1, 1] = 1.0
				w[2, 1, 2, 1] = 1.0
			All cell pairs have at least one edge so w is fully defined.
		"""
		println("─" ^ 70)
		println("Test 12: _compute_p_matrix -> Laplace smoothing on known counts")
		println("─" ^ 70)

		#	Construct fixture
			src = ["n1", "n1", "n3", "n3"]
			dst = ["n2", "n3", "n1", "n4"]
			edges = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
			nodes = DataFrame(id = ["n1", "n2", "n3", "n4"], label = ["n1", "n2", "n3", "n4"])

		#	Hand-set bin assignments: bin 1 = {n1, n2}, bin 2 = {n3, n4}
			degree_bins = [1, 1, 2, 2]
			ei_bins     = [1, 1, 1, 1]   # J = 1 (degree-only mode)
			K = 2
			J = 1

		#	Call the function (directed, unweighted)
			result = _compute_p_matrix(edges, nodes, degree_bins, ei_bins,
									   K, J, true, false;
									   partially_observed = Int[])

		#	Hand-computed expected values
			expected_P = zeros(Float64, K, J, K, J)
			expected_P[1, 1, 1, 1] = 2.0 / 4.0   # (1+1)/(2+2)
			expected_P[1, 1, 2, 1] = 2.0 / 6.0   # (1+1)/(4+2)
			expected_P[2, 1, 1, 1] = 2.0 / 6.0   # (1+1)/(4+2)
			expected_P[2, 1, 2, 1] = 2.0 / 4.0   # (1+1)/(2+2)

			expected_w = zeros(Float64, K, J, K, J)
			expected_w[1, 1, 1, 1] = 1.0
			expected_w[1, 1, 2, 1] = 1.0
			expected_w[2, 1, 1, 1] = 1.0
			expected_w[2, 1, 2, 1] = 1.0

		#	Validate
			P_match = isapprox(result.P, expected_P; atol = 1e-12)
			w_match = isapprox(result.w, expected_w; atol = 1e-12)

		#	Per-cell-pair diagnostic output
			println("  Test fixture: 4 nodes, 4 directed edges, K=2, J=1")
			println("  Bin assignment: n1, n2 -> bin 1; n3, n4 -> bin 2")
			println()
			println("  Edges:  n1->n2 (1->1), n1->n3 (1->2), n3->n1 (2->1), n3->n4 (2->2)")
			println()
			println("  Per-cell-pair P matrix:")
			println("    Cell pair (src_deg, src_ei, dst_deg, dst_ei)  Got       Expected  Match")
			for ds in 1:K, dt in 1:K
				got_val = result.P[ds, 1, dt, 1]
				exp_val = expected_P[ds, 1, dt, 1]
				ok = isapprox(got_val, exp_val; atol = 1e-12) ? "OK" : "FAIL"
				println(@sprintf("    (%d, 1, %d, 1)                                 %.4f    %.4f    %s",
								  ds, dt, got_val, exp_val, ok))
			end

			println()
			println("  Per-cell-pair w matrix (conditional mean weight):")
			println("    Cell pair (src_deg, src_ei, dst_deg, dst_ei)  Got       Expected  Match")
			for ds in 1:K, dt in 1:K
				got_val = result.w[ds, 1, dt, 1]
				exp_val = expected_w[ds, 1, dt, 1]
				ok = isapprox(got_val, exp_val; atol = 1e-12) ? "OK" : "FAIL"
				println(@sprintf("    (%d, 1, %d, 1)                                 %.4f    %.4f    %s",
								  ds, dt, got_val, exp_val, ok))
			end

			println()
			println("  Summary:")
			println("    P matrix matches expected:  $(P_match ? "OK" : "FAIL")")
			println("    w matrix matches expected:  $(w_match ? "OK" : "FAIL")")

		#	Final result
			all_pass = P_match && w_match

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-12
	function run_tests_through_12()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_12()

#	Test 13: _compute_p_matrix excludes partially_observed nodes
	function test_compute_p_matrix_excludes_partial()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_p_matrix correctly excludes partially observed
			(nominated non-respondent) nodes from the P calculation.

			Construction: 5-node fixture where n5 is partially observed.
			The remaining 4 nodes have identical bin assignments and
			edges as Test 12. Then n5 is added with edges that would
			perturb P if included:
				n1 -> n5  (would add to bin 1 -> bin 2)
				n5 -> n3  (would add to bin 2 -> bin 2)

			n5 is assigned to bin 2 and marked partially observed.

			Expected behavior:
			- The 4 respondent-respondent edges produce the same P
			  matrix as Test 12
			- The 2 edges involving n5 are excluded
			- n5 is excluded from the dyad-count cell totals
			- Therefore P matches Test 12's expected values exactly

			This verifies:
			- partially_observed filtering removes edges from the count
			- partially_observed filtering removes nodes from the cell
			  totals (denominator) so the smoothing math stays right
			- The function still produces a valid P matrix even when
			  some nodes are excluded

			Note: bin assignments for partially observed nodes must still
			be provided (they're used for Stage 0.5 placement); the
			function just ignores them when computing the respondent-
			respondent P. The degree_bins and ei_bins vectors are sized
			to the full node count (5 nodes), not the respondent count.
		"""
		println("─" ^ 70)
		println("Test 13: _compute_p_matrix excludes partially_observed nodes")
		println("─" ^ 70)

		#	Construct fixture: 5 nodes where n5 is partially observed.
		#	The first 4 nodes match Test 12; n5 is added with extra edges
		#	that should be excluded.
			src = ["n1", "n1", "n3", "n3",          # Test 12 edges (4)
				   "n1", "n5"]                       # n5-involving edges (2)
			dst = ["n2", "n3", "n1", "n4",
				   "n5", "n3"]
			edges = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
			nodes = DataFrame(id = ["n1", "n2", "n3", "n4", "n5"],
							  label = ["n1", "n2", "n3", "n4", "n5"])

		#	Bin assignments: same as Test 12 plus n5 in bin 2.
		#	n5's bin assignment is required by the function signature
		#	but should not affect P since n5 is excluded as partial.
			degree_bins = [1, 1, 2, 2, 2]
			ei_bins     = [1, 1, 1, 1, 1]
			K = 2
			J = 1

		#	Mark n5 (index 5) as partially observed
			partially_observed = [5]

		#	Call the function
			result = _compute_p_matrix(edges, nodes, degree_bins, ei_bins,
									   K, J, true, false;
									   partially_observed = partially_observed)

		#	Expected values: identical to Test 12 because the n5-involving
		#	edges are excluded and n5 doesn't count as a respondent in
		#	the bin-2 cell totals.
			expected_P = zeros(Float64, K, J, K, J)
			expected_P[1, 1, 1, 1] = 2.0 / 4.0
			expected_P[1, 1, 2, 1] = 2.0 / 6.0
			expected_P[2, 1, 1, 1] = 2.0 / 6.0
			expected_P[2, 1, 2, 1] = 2.0 / 4.0

		#	Validate
			P_match = isapprox(result.P, expected_P; atol = 1e-12)

		#	Diagnostic: also explicitly verify that the function's
		#	cell-node count for bin 2 is 2 (n3, n4), NOT 3 (n3, n4, n5).
		#	We can't directly inspect the internal counter, but we can
		#	verify it from the denominators: within-cell P uses
		#	(edge_count + 1) / (n*(n-1) + 2) so the denominator tells
		#	us how many respondents the function counted in that cell.
			#	P[2,1,2,1] = (edge_count + 1) / (2*1 + 2) = 2/4 = 0.5
			#	If n5 were counted, denominator would be 3*2 + 2 = 8 and
			#	P[2,1,2,1] would be 2/8 = 0.25, not 0.5
				cell_count_correct = isapprox(result.P[2, 1, 2, 1], 0.5; atol = 1e-12)

		#	Per-cell-pair diagnostic output
			println("  Test fixture: 5 nodes, 6 directed edges, K=2, J=1")
			println("  Bin assignment: n1, n2 -> bin 1; n3, n4, n5 -> bin 2 (n5 = partial)")
			println()
			println("  Edges (6 total):")
			println("    Respondent-respondent (4): n1->n2, n1->n3, n3->n1, n3->n4")
			println("    Excluded (involves n5):    n1->n5, n5->n3")
			println()
			println("  Per-cell-pair P matrix (should match Test 12 exactly):")
			println("    Cell pair (src_deg, src_ei, dst_deg, dst_ei)  Got       Expected  Match")
			for ds in 1:K, dt in 1:K
				got_val = result.P[ds, 1, dt, 1]
				exp_val = expected_P[ds, 1, dt, 1]
				ok = isapprox(got_val, exp_val; atol = 1e-12) ? "OK" : "FAIL"
				println(@sprintf("    (%d, 1, %d, 1)                                 %.4f    %.4f    %s",
								  ds, dt, got_val, exp_val, ok))
			end

			println()
			println("  Cell-count verification (denominator-based):")
			println("    P[2,1,2,1] = 0.5 implies 2 respondents in bin 2 (not 3)")
			println("    If n5 were counted, this would be 0.25 instead")
			println("    Result: $(cell_count_correct ? "OK (n5 correctly excluded)" : "FAIL")")

			println()
			println("  P matrix matches Test 12 expected: $(P_match ? "OK" : "FAIL")")

		#	Final result
			all_pass = P_match && cell_count_correct

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-13
	function run_tests_through_13()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_13()

#	Test 14: _compute_r_matrix on fully-reciprocated directed graph
	function test_compute_r_matrix_full_reciprocity()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _compute_r_matrix computes Laplace-smoothed reciprocity
			rates correctly on a graph where every directed edge is
			reciprocated.

			Construction: 4 nodes in two bins of 2 each. Every forward edge
			has a corresponding reverse edge (full reciprocity).

			Edges (directed):
				n1 -> n2, n2 -> n1   (mutual within bin 1)
				n1 -> n3, n3 -> n1   (mutual cross 1->2 / 2->1)
				n3 -> n4, n4 -> n3   (mutual within bin 2)

			Hand-computed per-cell-pair counts:
				forward_count[1, 1, 1, 1]: n1->n2 = 1; n2->n1 = 1
					Total forward = 2
				mutual_count[1, 1, 1, 1]: each of n1->n2 and n2->n1
					has its reverse, so both are mutual
					Total mutual = 2
					R[1, 1, 1, 1] = (2 + 1) / (2 + 2) = 3/4 = 0.75

				forward_count[1, 1, 2, 1]: n1->n3 = 1
					Total forward = 1
				mutual_count[1, 1, 2, 1]: n1->n3 has reverse (n3->n1)
					Total mutual = 1
					R[1, 1, 2, 1] = (1 + 1) / (1 + 2) = 2/3 ≈ 0.6667

				forward_count[2, 1, 1, 1]: n3->n1 = 1
					Total forward = 1
				mutual_count[2, 1, 1, 1]: n3->n1 has reverse (n1->n3)
					Total mutual = 1
					R[2, 1, 1, 1] = (1 + 1) / (1 + 2) = 2/3 ≈ 0.6667

				forward_count[2, 1, 2, 1]: n3->n4 = 1, n4->n3 = 1
					Total forward = 2
				mutual_count[2, 1, 2, 1]: both mutual
					Total mutual = 2
					R[2, 1, 2, 1] = (2 + 1) / (2 + 2) = 3/4 = 0.75

			Note: R uses (mutual + 1) / (forward + 2). With full reciprocity,
			mutual = forward, so R = (forward + 1) / (forward + 2), which
			approaches 1 as forward count grows large. Small samples have
			more Laplace pull toward 0.5.
		"""
		println("─" ^ 70)
		println("Test 14: _compute_r_matrix on fully-reciprocated directed graph")
		println("─" ^ 70)

		#	Construct fixture
			src = ["n1", "n2", "n1", "n3", "n3", "n4"]
			dst = ["n2", "n1", "n3", "n1", "n4", "n3"]
			edges = DataFrame(src = src, dst = dst, weight = ones(Int, length(src)))
			nodes = DataFrame(id = ["n1", "n2", "n3", "n4"], label = ["n1", "n2", "n3", "n4"])
			degree_bins = [1, 1, 2, 2]
			ei_bins     = [1, 1, 1, 1]
			K = 2
			J = 1

		#	Call the function
			R = _compute_r_matrix(edges, nodes, degree_bins, ei_bins, K, J;
								  partially_observed = Int[])

		#	Expected R values
			expected_R = zeros(Float64, K, J, K, J)
			expected_R[1, 1, 1, 1] = 3.0 / 4.0   # (2+1)/(2+2)
			expected_R[1, 1, 2, 1] = 2.0 / 3.0   # (1+1)/(1+2)
			expected_R[2, 1, 1, 1] = 2.0 / 3.0   # (1+1)/(1+2)
			expected_R[2, 1, 2, 1] = 3.0 / 4.0   # (2+1)/(2+2)

		#	Validate
			R_match = isapprox(R, expected_R; atol = 1e-12)

		#	Diagnostic: verify the function returns an Array{Float64, 4}
		#	with the right dimensions
			dims_ok  = size(R) == (K, J, K, J)
			type_ok  = eltype(R) == Float64

		#	Diagnostic output
			println("  Test fixture: 4 nodes, 6 directed edges (all reciprocated)")
			println("  Bin assignment: n1, n2 -> bin 1; n3, n4 -> bin 2")
			println()
			println("  Edges (6 total, all reciprocal):")
			println("    n1<->n2 (within bin 1, 2 forward edges)")
			println("    n1<->n3 (across, 1 forward + 1 reverse)")
			println("    n3<->n4 (within bin 2, 2 forward edges)")
			println()
			println("  Per-cell-pair R matrix:")
			println("    Cell pair (src_deg, src_ei, dst_deg, dst_ei)  Got       Expected  Match")
			for ds in 1:K, dt in 1:K
				got_val = R[ds, 1, dt, 1]
				exp_val = expected_R[ds, 1, dt, 1]
				ok = isapprox(got_val, exp_val; atol = 1e-12) ? "OK" : "FAIL"
				println(@sprintf("    (%d, 1, %d, 1)                                 %.4f    %.4f    %s",
								  ds, dt, got_val, exp_val, ok))
			end

			println()
			println("  Summary:")
			println("    R matrix dimensions: $(size(R)) (expected ($K, $J, $K, $J))   $(dims_ok ? "OK" : "FAIL")")
			println("    R matrix element type: $(eltype(R)) (expected Float64)       $(type_ok ? "OK" : "FAIL")")
			println("    R matrix matches expected:                                    $(R_match ? "OK" : "FAIL")")
			println()
			println("  Sanity: in a fully-reciprocated graph, R values reflect the")
			println("  Laplace prior pull-down from the true rate of 1.0:")
			println("    forward=2 cells: R = 3/4 = 0.75 (vs true 1.0)")
			println("    forward=1 cells: R = 2/3 ≈ 0.67 (vs true 1.0)")
			println("  As sample size grows, R approaches the true rate.")

		#	Final result
			all_pass = R_match && dims_ok && type_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-14
	function run_tests_through_14()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()
		t14 = test_compute_r_matrix_full_reciprocity()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_14()

#	Test 15: _determine_n_add arithmetic across edge cases
	function test_determine_n_add_arithmetic()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _determine_n_add against hand-computed values across
			several edge cases.

			Design formula:
				N_add = round(pi_node * (N + N_nom) / (1 - pi_node)) - N_nom

			Test cases:
			(a) pi_node = 0:
				Should return 0 (Stage 1 skipped regardless of N_nom).
				Tests: (0.0, 100, 0) and (0.0, 100, 10) both -> 0

			(b) Standard case, no nominees:
				pi_node = 0.1, N = 100, N_nom = 0
				N_add = round(0.1 * 100 / 0.9) = round(11.111...) = 11

			(c) Standard case with nominees:
				pi_node = 0.1, N = 100, N_nom = 5
				N_add = round(0.1 * 105 / 0.9) - 5 = round(11.667) - 5 = 12 - 5 = 7

			(d) High pi_node:
				pi_node = 0.5, N = 100, N_nom = 0
				N_add = round(0.5 * 100 / 0.5) = 100

			(e) Clamping to zero:
				pi_node = 0.05, N = 100, N_nom = 100
				implied_missing = round(0.05 * 200 / 0.95) = round(10.526) = 11
				N_add = 11 - 100 = -89 -> clamped to 0
				This is the "N_nom exceeds implied total missing" edge case.

			(f) Round-half-to-even:
				pi_node = 0.5, N = 1, N_nom = 0
				N_add = round(0.5 * 1 / 0.5) = round(1.0) = 1
				Trivial because 1.0 has no rounding ambiguity.

				A more interesting rounding case:
				pi_node = 0.5, N = 3, N_nom = 0
				N_add = round(0.5 * 3 / 0.5) = round(3.0) = 3
				Also trivial.

				Round-half: pi_node = 1/3, N = 6, N_nom = 0
				N_add = round((1/3) * 6 / (2/3)) = round(3.0) = 3

			The formula uses Julia's default round (banker's rounding: round-
			half-to-even). For all our test cases, the result is either an
			integer or clearly above/below 0.5, so rounding behavior is
			unambiguous.
		"""
		println("─" ^ 70)
		println("Test 15: _determine_n_add arithmetic")
		println("─" ^ 70)

		#	Test cases as (pi_node, N, N_nom, expected_n_add) tuples
			test_cases = [
				(0.0,  100, 0,   0),     # pi_node = 0
				(0.0,  100, 10,  0),     # pi_node = 0 with nominees
				(0.1,  100, 0,   11),    # standard no nominees
				(0.1,  100, 5,   7),     # standard with nominees
				(0.5,  100, 0,   100),   # high pi_node
				(0.05, 100, 100, 0),     # clamp to zero (N_nom exceeds)
				(0.25, 20,  0,   7),     # round(0.25*20/0.75) = round(6.667) = 7
				(0.4,  50,  10,  30),    # round(0.4*60/0.6) - 10 = 40 - 10 = 30
			]

		#	Run each test case
			all_pass = true
			println("  Per-case results:")
			println("    pi_node  N    N_nom  Got    Expected  Match")
			for (pi_node, N, N_nom, expected) in test_cases
				got = _determine_n_add(pi_node, N, N_nom)
				match = got == expected
				if !match
					all_pass = false
				end
				ok_str = match ? "OK" : "FAIL"
				println(@sprintf("    %-8.3f %-4d %-6d %-6d %-9d %s",
								  pi_node, N, N_nom, got, expected, ok_str))
			end

		#	Diagnostic: edge case verification
			println()
			println("  Edge case verifications:")

			#	pi_node = 0 always returns 0
				zero_test = _determine_n_add(0.0, 1000, 50)
				println("    pi_node=0 (N=1000, N_nom=50) -> $zero_test (expected 0): $(zero_test == 0 ? "OK" : "FAIL")")

			#	Clamping to nonnegative
				clamp_test = _determine_n_add(0.05, 100, 100)
				println("    pi_node=0.05, N=100, N_nom=100 -> $clamp_test (expected 0, clamped from negative): $(clamp_test == 0 ? "OK" : "FAIL")")

		#	Verify guard behaviors (out-of-range inputs should throw)
			println()
			println("  Guard verifications:")
			throws_neg_pi = false
			try
				_determine_n_add(-0.1, 100, 0)
			catch e
				throws_neg_pi = isa(e, ArgumentError)
			end
			println("    pi_node = -0.1 throws ArgumentError: $(throws_neg_pi ? "OK" : "FAIL")")

			throws_pi_one = false
			try
				_determine_n_add(1.0, 100, 0)
			catch e
				throws_pi_one = isa(e, ArgumentError)
			end
			println("    pi_node = 1.0 throws ArgumentError: $(throws_pi_one ? "OK" : "FAIL")")

			throws_zero_n = false
			try
				_determine_n_add(0.1, 0, 0)
			catch e
				throws_zero_n = isa(e, ArgumentError)
			end
			println("    N = 0 throws ArgumentError: $(throws_zero_n ? "OK" : "FAIL")")

			throws_neg_nom = false
			try
				_determine_n_add(0.1, 100, -1)
			catch e
				throws_neg_nom = isa(e, ArgumentError)
			end
			println("    N_nom = -1 throws ArgumentError: $(throws_neg_nom ? "OK" : "FAIL")")

			guards_ok = throws_neg_pi && throws_pi_one && throws_zero_n && throws_neg_nom

			all_pass = all_pass && guards_ok

		#	Final result
			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-15
	function run_tests_through_15()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()
		t14 = test_compute_r_matrix_full_reciprocity()
		t15 = test_determine_n_add_arithmetic()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9,
					   t10, t11, t12, t13, t14, t15]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_15()

#	Test 16: _realized_rho_for_beta analytic properties
	function test_realized_rho_for_beta_properties()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies analytic properties of _realized_rho_for_beta:

			Property 1: beta = 0 returns exactly 0.0
				With beta = 0, q is uniform, so added nodes are distributed
				identically to observed nodes (modulo the rank-equal
				partition). No correlation between bin index and missingness
				indicator. This holds independent of K, N, N_add.

			Property 2: Sign symmetry
				realized_rho(beta) ≈ -realized_rho(-beta)
				Distribution q at +beta and -beta are mirror images across
				the bin index midpoint. Correlation flips sign.

			Property 3: Strict monotonicity
				For beta_1 < beta_2, realized_rho(beta_1) < realized_rho(beta_2)
				Higher beta skews q toward high bins, raising correlation.
				The bisection logic in _solve_bin_distribution depends on
				this monotonicity for the standard left-right bracketing
				to work.

			Test sweep:
				K = 10, N = 100, N_add = 20
				beta values: -5, -2, -1, 0, 1, 2, 5

			Expected behavior:
			- beta = 0 -> realized_rho = 0.0 exactly
			- beta = +x and beta = -x give symmetric rhos
			- realized_rho strictly increases with beta
		"""
		println("─" ^ 70)
		println("Test 16: _realized_rho_for_beta analytic properties")
		println("─" ^ 70)

		K = 10
		N = 100
		N_add = 20

		#	Property 1: beta = 0 returns 0
			rho_at_zero = _realized_rho_for_beta(0.0, K, N, N_add)
			beta_zero_ok = isapprox(rho_at_zero, 0.0; atol = 1e-12)
			println("  Property 1: beta = 0 -> exactly 0.0")
			println("    Got: $rho_at_zero (expected 0.0)  $(beta_zero_ok ? "OK" : "FAIL")")

		#	Property 2: Sign symmetry
			beta_vals_sym = [0.5, 1.0, 2.0, 5.0]
			println()
			println("  Property 2: Sign symmetry: f(-beta) ≈ -f(beta)")
			println("    beta   f(beta)     f(-beta)    Sum (should be ~0)")
			symmetry_ok = true
			for b in beta_vals_sym
				rho_pos = _realized_rho_for_beta(b, K, N, N_add)
				rho_neg = _realized_rho_for_beta(-b, K, N, N_add)
				sum_val = rho_pos + rho_neg
				match = isapprox(sum_val, 0.0; atol = 1e-10)
				if !match
					symmetry_ok = false
				end
				ok_str = match ? "OK" : "FAIL"
				println(@sprintf("    %-6.2f %-11.6f %-11.6f %-10.6e %s",
								  b, rho_pos, rho_neg, sum_val, ok_str))
			end

		#	Property 3: Strict monotonicity
			beta_grid = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
			rho_grid  = [_realized_rho_for_beta(b, K, N, N_add) for b in beta_grid]

			println()
			println("  Property 3: Strict monotonicity")
			println("    Increment-by-increment check (rho should strictly increase):")
			monotonic_ok = true
			println("    beta(i)  beta(i+1)  rho(i)     rho(i+1)   delta      Match")
			for i in 1:(length(beta_grid) - 1)
				b1 = beta_grid[i]
				b2 = beta_grid[i + 1]
				r1 = rho_grid[i]
				r2 = rho_grid[i + 1]
				delta = r2 - r1
				ok = delta > 0
				if !ok
					monotonic_ok = false
				end
				println(@sprintf("    %-7.2f  %-9.2f  %-10.6f %-10.6f %-10.6f %s",
								  b1, b2, r1, r2, delta, ok ? "OK" : "FAIL"))
			end

		#	Diagnostic: print the range of realized rho
			println()
			println("  Realized correlation range (K=$K, N=$N, N_add=$N_add):")
			println(@sprintf("    beta = %-6.2f -> rho = %.4f (low ceiling)", beta_grid[1], rho_grid[1]))
			println(@sprintf("    beta = %-6.2f -> rho = %.4f (high ceiling)", beta_grid[end], rho_grid[end]))

		#	Final result
			all_pass = beta_zero_ok && symmetry_ok && monotonic_ok

			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-16
	function run_tests_through_16()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()
		t14 = test_compute_r_matrix_full_reciprocity()
		t15 = test_determine_n_add_arithmetic()
		t16 = test_realized_rho_for_beta_properties()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9,
					   t10, t11, t12, t13, t14, t15, t16]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_16()

#	Test 17: _solve_bin_distribution bisection convergence
	function test_solve_bin_distribution_convergence()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Verifies _solve_bin_distribution correctly:
			(a) Returns beta=0, q uniform, status=:converged when
				rho_target = 0 (trivial case, no bisection needed)
			(b) Converges to an interior target within tolerance
			(c) Returns :ceiling_hit when target exceeds the achievable
				range (signals via status, returns the closest beta)
			(d) Handles N_add=0 gracefully (no missingness variation)

			Tests use K=10, N=100, N_add=20, which has analytic achievable
			ceiling near +/- 0.52 (per Test 16 diagnostic). Interior targets
			are well below this ceiling (e.g., 0.25), and extreme targets
			(e.g., 0.95) trigger :ceiling_hit.
		"""
		println("─" ^ 70)
		println("Test 17: _solve_bin_distribution bisection convergence")
		println("─" ^ 70)

		K = 10
		N = 100
		N_add = 20

		all_pass = true

		#	Case (a): rho_target = 0 -> beta=0, q uniform, :converged
			println("  Case (a): rho_target = 0")
			result_a = _solve_bin_distribution(0.0, K, N, N_add)
			beta_a_ok    = result_a.beta == 0.0
			q_uniform_ok = all(isapprox.(result_a.q, 1.0/K; atol = 1e-12))
			status_a_ok  = result_a.status == :converged
			iters_a_ok   = result_a.n_iters == 0
			println("    beta = $(result_a.beta) (expected 0.0)             $(beta_a_ok ? "OK" : "FAIL")")
			println("    q uniform (all 0.1):                                 $(q_uniform_ok ? "OK" : "FAIL")")
			println("    status = $(result_a.status) (expected :converged)   $(status_a_ok ? "OK" : "FAIL")")
			println("    n_iters = $(result_a.n_iters) (expected 0)           $(iters_a_ok ? "OK" : "FAIL")")
			all_pass = all_pass && beta_a_ok && q_uniform_ok && status_a_ok && iters_a_ok

		#	Case (b): rho_target = 0.25 (interior, well within ceiling ~0.52)
			println()
			println("  Case (b): rho_target = 0.25 (interior)")
			result_b = _solve_bin_distribution(0.25, K, N, N_add)
			#	Verify status, that beta is positive, and that the realized
			#	rho at the returned beta is within tolerance of target
				status_b_ok = result_b.status == :converged
				beta_b_positive = result_b.beta > 0.0
				realized_b = _realized_rho_for_beta(result_b.beta, K, N, N_add)
				realized_close = isapprox(realized_b, 0.25; atol = 1e-4)
				#	q sums to 1
				q_sums_ok_b = isapprox(sum(result_b.q), 1.0; atol = 1e-12)
				#	q skews toward high bins (q[K] > q[1])
				q_skews_high = result_b.q[K] > result_b.q[1]
			println("    status = $(result_b.status) (expected :converged)   $(status_b_ok ? "OK" : "FAIL")")
			println("    beta = $(round(result_b.beta, digits=4)) (expected > 0)            $(beta_b_positive ? "OK" : "FAIL")")
			println("    realized rho at beta = $(round(realized_b, digits=6)) (target 0.25, tol 1e-4) $(realized_close ? "OK" : "FAIL")")
			println("    q sums to 1.0:                                       $(q_sums_ok_b ? "OK" : "FAIL")")
			println("    q[K]=$(round(result_b.q[K], digits=4)) > q[1]=$(round(result_b.q[1], digits=4)): $(q_skews_high ? "OK" : "FAIL")")
			println("    n_iters = $(result_b.n_iters)")
			all_pass = all_pass && status_b_ok && beta_b_positive && realized_close && q_sums_ok_b && q_skews_high

		#	Case (c): rho_target = -0.25 (interior, negative)
			println()
			println("  Case (c): rho_target = -0.25 (interior, negative)")
			result_c = _solve_bin_distribution(-0.25, K, N, N_add)
			status_c_ok = result_c.status == :converged
			beta_c_negative = result_c.beta < 0.0
			realized_c = _realized_rho_for_beta(result_c.beta, K, N, N_add)
			realized_close_c = isapprox(realized_c, -0.25; atol = 1e-4)
			q_skews_low = result_c.q[1] > result_c.q[K]
			println("    status = $(result_c.status) (expected :converged)   $(status_c_ok ? "OK" : "FAIL")")
			println("    beta = $(round(result_c.beta, digits=4)) (expected < 0)            $(beta_c_negative ? "OK" : "FAIL")")
			println("    realized rho at beta = $(round(realized_c, digits=6)) (target -0.25, tol 1e-4) $(realized_close_c ? "OK" : "FAIL")")
			println("    q[1]=$(round(result_c.q[1], digits=4)) > q[K]=$(round(result_c.q[K], digits=4)) (q skews low): $(q_skews_low ? "OK" : "FAIL")")
			println("    n_iters = $(result_c.n_iters)")
			all_pass = all_pass && status_c_ok && beta_c_negative && realized_close_c && q_skews_low

		#	Case (d): rho_target = 0.95 (above ceiling) -> :ceiling_hit
			println()
			println("  Case (d): rho_target = 0.95 (above achievable ceiling)")
			result_d = _solve_bin_distribution(0.95, K, N, N_add)
			status_d_ok = result_d.status == :ceiling_hit
			beta_d_extreme = result_d.beta > 0.0   # should be at upper bound
			println("    status = $(result_d.status) (expected :ceiling_hit) $(status_d_ok ? "OK" : "FAIL")")
			println("    beta = $(round(result_d.beta, digits=4)) (expected at upper bound) $(beta_d_extreme ? "OK" : "FAIL")")
			println("    n_iters = $(result_d.n_iters)")
			all_pass = all_pass && status_d_ok && beta_d_extreme

		#	Case (e): N_add = 0 (no missingness variation)
			println()
			println("  Case (e): N_add = 0 (no missingness variation)")
			#	With rho_target = 0, N_add = 0: trivially converged
				result_e0 = _solve_bin_distribution(0.0, K, N, 0)
				e0_ok = result_e0.status == :converged && result_e0.beta == 0.0
				println("    rho_target = 0, N_add = 0: status = $(result_e0.status), beta = $(result_e0.beta)   $(e0_ok ? "OK" : "FAIL")")
			#	With rho_target != 0, N_add = 0: failed_other
				result_e1 = _solve_bin_distribution(0.25, K, N, 0)
				e1_ok = result_e1.status == :failed_other
				println("    rho_target = 0.25, N_add = 0: status = $(result_e1.status) (expected :failed_other) $(e1_ok ? "OK" : "FAIL")")
			all_pass = all_pass && e0_ok && e1_ok

		#	Final result
			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-17
	function run_tests_through_17()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()
		t14 = test_compute_r_matrix_full_reciprocity()
		t15 = test_determine_n_add_arithmetic()
		t16 = test_realized_rho_for_beta_properties()
		t17 = test_solve_bin_distribution_convergence()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9,
					   t10, t11, t12, t13, t14, t15, t16, t17]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_17()

#	Test 18: compute_setup end-to-end integration
	function test_compute_setup_end_to_end()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			End-to-end integration test of compute_setup. Uses the
			healthy 2D-supportable fixture from Test 9 (12 nodes, two
			communities, edges engineered for proper E/I distribution
			across all three semantic bins).

			Inputs to compute_setup:
				edges, nodes:           from _build_healthy_2d_fixture()
				community_labels:       [1,1,1,1,1,1, 2,2,2,2,2,2]
				directed:               false (undirected)
				weighted:               false
				pi_node:                0.1
				pi_edge:                0.0
				rho:                    0.25
				partially_observed:     [] (no nominated non-respondents)
				K:                      4
				J:                      3

			Expected SamplerSetup outputs (verified field-by-field):

			Inputs preserved:
				- edges, nodes preserved as-passed
				- directed = false, weighted = false
				- pi_node = 0.1, pi_edge = 0.0, rho = 0.25

			Per-node setup outputs:
				- centrality: length 12, all positive, sum = 2 * n_edges
				- community_labels preserved
				- ei_values: length 12, no NaN (all nodes have edges)
				- binning_mode: :two_dimensional (fixture supports it)
				- degree_bins: length 12, values in 1..K=4
				- ei_bins: length 12, values in 1..J=3
				- K = 4, J = 3

			Matrices:
				- P: Array{Float64,4} of size (4, 3, 4, 3)
				- w: Array{Float64,4} of size (4, 3, 4, 3)
				- R: nothing (undirected)

			Added-node specification:
				- partially_observed: empty
				- N_add: round(0.1 * 12 / 0.9) = round(1.333) = 1
				- beta: converged value (positive, since rho > 0)
				- beta_status: :converged
				- q: length 4, sums to 1, q[K] > q[1]
				- ei_conditional: 4x3 matrix, each row sums to 1

			Diagnostics dict has the expected keys:
				:binning_mode, :fallback_reason, :beta_status, :beta_n_iters

			This test catches integration bugs that the unit tests miss:
			- Component A and B might each work in isolation but fail
			  to compose (e.g., one indexes bins as 1..K, the other as
			  0..K-1, both pass their own tests, but compute_setup
			  produces an out-of-bounds error)
			- The SamplerSetup struct fields might be wired wrong
			  (e.g., R assigned to a field expecting Matrix instead of
			  Array{Float64, 4})
			- The diagnostics dict might be missing keys
		"""
		println("─" ^ 70)
		println("Test 18: compute_setup end-to-end integration")
		println("─" ^ 70)

		#	Build healthy fixture (reusing the helper from Test 9)
			fixture = _build_healthy_2d_fixture()

		#	Call compute_setup
			setup = Network_Credible_Intervals.network_reconstruction.compute_setup(
				fixture.edges, fixture.nodes, fixture.community_labels;
				directed = false,
				weighted = false,
				pi_node  = 0.1,
				pi_edge  = 0.0,
				rho      = 0.25,
				partially_observed_nodes = Int[],
				K        = 4,
				J        = 3
			)

		#	Track per-field pass/fail and aggregate at the end
			results = Dict{Symbol, Bool}()

		#	Inputs preserved
			results[:edges_preserved] = setup.edges === fixture.edges
			results[:nodes_preserved] = setup.nodes === fixture.nodes
			results[:directed_field]  = setup.directed == false
			results[:weighted_field]  = setup.weighted == false
			results[:pi_node_field]   = setup.pi_node == 0.1
			results[:pi_edge_field]   = setup.pi_edge == 0.0
			results[:rho_field]       = setup.rho == 0.25

		#	Per-node centrality
			results[:centrality_length]   = length(setup.centrality) == 12
			results[:centrality_positive] = all(setup.centrality .>= 0.0)
			#	Sum of centrality should be 2 * n_edges for undirected
				results[:centrality_sum] = sum(setup.centrality) == 2.0 * nrow(fixture.edges)

		#	Community labels preserved
			results[:community_labels_preserved] = setup.community_labels == fixture.community_labels

		#	E/I values
			results[:ei_length]   = length(setup.ei_values) == 12
			results[:ei_no_nan]   = !any(isnan, setup.ei_values)   # fixture has no isolates
			results[:ei_range]    = all(-1.0 .<= setup.ei_values .<= 1.0)

		#	Binning mode (fixture is healthy 2D-supportable)
			results[:binning_mode] = setup.binning_mode == :two_dimensional

		#	Bin assignments
			results[:degree_bins_length] = length(setup.degree_bins) == 12
			results[:degree_bins_range]  = all(1 .<= setup.degree_bins .<= 4)
			results[:ei_bins_length]     = length(setup.ei_bins) == 12
			results[:ei_bins_range]      = all(1 .<= setup.ei_bins .<= 3)
			results[:K_value]            = setup.K == 4
			results[:J_value]            = setup.J == 3

		#	Matrices
			results[:P_type]   = isa(setup.P, Array{Float64, 4})
			results[:P_shape]  = size(setup.P) == (4, 3, 4, 3)
			results[:w_type]   = isa(setup.w, Array{Float64, 4})
			results[:w_shape]  = size(setup.w) == (4, 3, 4, 3)
			results[:R_undirected] = setup.R === nothing   # undirected -> R = nothing

		#	Added-node spec
			results[:partial_empty]   = isempty(setup.partially_observed)
			#	N_add: round(0.1 * 12 / 0.9) = round(1.333) = 1
				results[:N_add_value] = setup.N_add == 1
			results[:beta_positive]  = setup.beta > 0.0   # positive rho -> positive beta
			results[:beta_status]    = setup.beta_status == :converged
			results[:q_length]       = length(setup.q) == 4
			results[:q_sums_to_one]  = isapprox(sum(setup.q), 1.0; atol = 1e-12)
			results[:q_skews_high]   = setup.q[end] > setup.q[1]
			results[:ei_cond_shape]  = size(setup.ei_conditional) == (4, 3)
			#	Each row sums to 1 (row-stochastic)
				row_sums = sum(setup.ei_conditional, dims = 2)
				results[:ei_cond_rowsum] = all(isapprox.(row_sums, 1.0; atol = 1e-12))

		#	Diagnostics
			expected_diag_keys = [:binning_mode, :fallback_reason, :beta_status, :beta_n_iters]
			results[:diag_has_keys] = all(haskey(setup.diagnostics, k) for k in expected_diag_keys)
			results[:diag_binning_match] = setup.diagnostics[:binning_mode] == :two_dimensional
			results[:diag_fallback_nothing] = setup.diagnostics[:fallback_reason] === nothing
			results[:diag_beta_status] = setup.diagnostics[:beta_status] == :converged

		#	Diagnostic output: print every check
			println("  Inputs preserved:")
			println("    edges preserved:           $(results[:edges_preserved] ? "OK" : "FAIL")")
			println("    nodes preserved:           $(results[:nodes_preserved] ? "OK" : "FAIL")")
			println("    directed = false:          $(results[:directed_field] ? "OK" : "FAIL")")
			println("    weighted = false:          $(results[:weighted_field] ? "OK" : "FAIL")")
			println("    pi_node = 0.1:             $(results[:pi_node_field] ? "OK" : "FAIL")")
			println("    pi_edge = 0.0:             $(results[:pi_edge_field] ? "OK" : "FAIL")")
			println("    rho = 0.25:                $(results[:rho_field] ? "OK" : "FAIL")")

			println()
			println("  Per-node setup outputs:")
			println("    centrality length 12:      $(results[:centrality_length] ? "OK" : "FAIL")")
			println("    centrality nonneg:         $(results[:centrality_positive] ? "OK" : "FAIL")")
			println("    sum(cent) = 2 * n_edges:   $(results[:centrality_sum] ? "OK" : "FAIL") (got $(sum(setup.centrality)), expected $(2.0 * nrow(fixture.edges)))")
			println("    community_labels preserved: $(results[:community_labels_preserved] ? "OK" : "FAIL")")
			println("    ei_values length 12:       $(results[:ei_length] ? "OK" : "FAIL")")
			println("    ei_values no NaN:          $(results[:ei_no_nan] ? "OK" : "FAIL")")
			println("    ei_values in [-1, 1]:      $(results[:ei_range] ? "OK" : "FAIL")")

			println()
			println("  Binning:")
			println("    binning_mode = :two_dimensional: $(results[:binning_mode] ? "OK" : "FAIL")")
			println("    degree_bins length 12:     $(results[:degree_bins_length] ? "OK" : "FAIL")")
			println("    degree_bins in 1..4:       $(results[:degree_bins_range] ? "OK" : "FAIL")")
			println("    ei_bins length 12:         $(results[:ei_bins_length] ? "OK" : "FAIL")")
			println("    ei_bins in 1..3:           $(results[:ei_bins_range] ? "OK" : "FAIL")")
			println("    K = 4:                     $(results[:K_value] ? "OK" : "FAIL")")
			println("    J = 3:                     $(results[:J_value] ? "OK" : "FAIL")")

			println()
			println("  Matrices:")
			println("    P is Array{Float64, 4}:    $(results[:P_type] ? "OK" : "FAIL")")
			println("    P shape (4, 3, 4, 3):      $(results[:P_shape] ? "OK" : "FAIL")  (got $(size(setup.P)))")
			println("    w is Array{Float64, 4}:    $(results[:w_type] ? "OK" : "FAIL")")
			println("    w shape (4, 3, 4, 3):      $(results[:w_shape] ? "OK" : "FAIL")")
			println("    R = nothing (undirected):  $(results[:R_undirected] ? "OK" : "FAIL")")

			println()
			println("  Added-node specification:")
			println("    partially_observed empty:  $(results[:partial_empty] ? "OK" : "FAIL")")
			println("    N_add = 1:                 $(results[:N_add_value] ? "OK" : "FAIL")  (got $(setup.N_add))")
			println("    beta > 0:                  $(results[:beta_positive] ? "OK" : "FAIL")  (got $(round(setup.beta, digits=4)))")
			println("    beta_status :converged:    $(results[:beta_status] ? "OK" : "FAIL")")
			println("    q length 4:                $(results[:q_length] ? "OK" : "FAIL")")
			println("    sum(q) = 1.0:              $(results[:q_sums_to_one] ? "OK" : "FAIL")")
			println("    q skews to high bins:      $(results[:q_skews_high] ? "OK" : "FAIL")  (q[1]=$(round(setup.q[1], digits=4)), q[K]=$(round(setup.q[end], digits=4)))")
			println("    ei_conditional shape (4, 3): $(results[:ei_cond_shape] ? "OK" : "FAIL")")
			println("    ei_conditional row-stochastic: $(results[:ei_cond_rowsum] ? "OK" : "FAIL")")

			println()
			println("  Diagnostics dict:")
			println("    has expected keys:           $(results[:diag_has_keys] ? "OK" : "FAIL")")
			println("    binning_mode matches:        $(results[:diag_binning_match] ? "OK" : "FAIL")")
			println("    fallback_reason nothing:     $(results[:diag_fallback_nothing] ? "OK" : "FAIL")")
			println("    beta_status :converged:      $(results[:diag_beta_status] ? "OK" : "FAIL")")

		#	Aggregate
			all_pass = all(values(results))
			n_passed = count(identity, values(results))
			n_total = length(results)

			println()
			println("  Per-check tally: $n_passed / $n_total checks passed")
			println()
			println("  Result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end

#	Driver: run Tests 1-18
	function run_tests_through_18()
		println()
		println("=" ^ 70)
		println("Phase 2 Reconstruction Unit Tests — Setup-Phase Helpers")
		println("=" ^ 70)
		println()

		t1  = test_compute_centrality_star()
		t2  = test_compute_centrality_asymmetric_block()
		t3  = test_compute_ei_all_internal()
		t4  = test_compute_ei_all_external()
		t5  = test_compute_ei_mixed()
		t6  = test_detect_single_community()
		t7  = test_detect_too_few_nodes()
		t8  = test_detect_insufficient_ei_variance()
		t9  = test_detect_two_dimensional_active()
		t10 = test_bin_degree_equal_rank()
		t11 = test_bin_ei_semantic_thresholds()
		t12 = test_compute_p_matrix_laplace()
		t13 = test_compute_p_matrix_excludes_partial()
		t14 = test_compute_r_matrix_full_reciprocity()
		t15 = test_determine_n_add_arithmetic()
		t16 = test_realized_rho_for_beta_properties()
		t17 = test_solve_bin_distribution_convergence()
		t18 = test_compute_setup_end_to_end()

		all_results = [t1, t2, t3, t4, t5, t6, t7, t8, t9,
					   t10, t11, t12, t13, t14, t15, t16, t17, t18]

		println("=" ^ 70)
		println("Cumulative result: $(count(identity, all_results)) / $(length(all_results)) tests passed")
		println("=" ^ 70)

		return all(all_results)
	end
	run_tests_through_18()

######################
#   BEHAVIOR TESTS   #
######################

#	Test 19: Moreno algorithmic-contract calibration
	function test_calibration_moreno()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Algorithmic-contract calibration test on
			moreno_highschool_unweighted. Verifies that compute_setup
			honors its three priors:

			Prior 1 (proportion missing, user input):
				N_add satisfies N_add / (N_obs + N_add) ≈ pi_node = realized_rate
				This is verified algebraically: N_add is the output of
				_determine_n_add(pi_node, N_obs, N_nom=0), which solves
				the design's closed-form formula. So Prior 1 is an
				arithmetic identity, not an empirical claim.

			Prior 2 (centrality-missingness correlation, user input):
				The beta solver returns beta such that
				_realized_rho_for_beta(beta, K, N_obs, N_add) ≈ rho_input
				within bisection tolerance. Or, if rho_input is outside
				the achievable range for the given K and N_add, status is
				:ceiling_hit and beta is at the bound.

				This is verified by re-evaluating _realized_rho_for_beta
				at the solved beta and comparing against rho_input.

			Prior 3 (E/I distribution match, inferred assumption):
				ei_conditional[b, j] gives the empirical conditional
				probability P(ei_bin = j | degree_bin = b) in G_obs.
				This is verified by:
				(a) Row-stochasticity: each row of ei_conditional sums to 1
				(b) Match the empirical conditional from the degree_bins
					and ei_bins assignments in G_obs

			Each (rho, rate) cell uses Phase 1 to produce a realistic
			(realized_rho, realized_rate) pair as input for Phase 2.
			The test does not compare Phase 2's q against Phase 1's
			empirical dropped distribution; that conflates the framework's
			contract with Phase 1's specific mechanism.
		"""
		println("─" ^ 70)
		println("Test 19: Algorithmic-contract calibration on moreno_highschool_unweighted")
		println("─" ^ 70)

		#	Load Moreno from disk
			graphml_path = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks/moreno_highschool_unweighted.graphml"
			net = Network_Credible_Intervals.load_graphml(graphml_path)
			println("  Loaded: moreno_highschool_unweighted")
			println("  N = $(nrow(net.nodes)), E = $(nrow(net.edges))")
			println("  Directed: $(net.metadata.directed), Weighted: $(net.metadata.weighted)")
			println()

		#	Run three calibration cells
			K = 4
			J = 3
			R = 20
			cells = [
				(0.0,  0.10),
				(0.5,  0.10),
				(-0.5, 0.10),
			]

			cell_results = NamedTuple[]
			for (rho_input, rate_input) in cells
				print("  Running cell (rho=$rho_input, rate=$rate_input)... ")
				cell = _algorithmic_contract_cell(net, rho_input, rate_input;
													R = R, K = K, J = J)
				push!(cell_results, cell)
				println("done.")
			end

			println()
			println("=" ^ 70)
			println("ALGORITHMIC-CONTRACT RESULTS")
			println("=" ^ 70)
			for cell in cell_results
				_print_contract_cell(cell)
			end

		#	Gates: all three priors must be honored across all replicates
			gate_1 = all(c.prior_1_pass for c in cell_results)
			gate_2 = all(c.prior_2_pass for c in cell_results)
			gate_3 = all(c.prior_3_pass for c in cell_results)

			println("=" ^ 70)
			println("Gates (algorithmic contract):")
			println("  Prior 1 (proportion missing):           $(gate_1 ? "PASS" : "FAIL")")
			println("  Prior 2 (centrality correlation):       $(gate_2 ? "PASS" : "FAIL")")
			println("  Prior 3 (E/I-given-degree consistency): $(gate_3 ? "PASS" : "FAIL")")
			println("=" ^ 70)

			all_pass = gate_1 && gate_2 && gate_3

			println()
			println("  Test 19 result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end
    test_calibration_moreno()

#	Test 20: Scotland algorithmic-contract calibration
	function test_calibration_scotland()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Algorithmic-contract calibration test on
			scotland_interlock_unweighted, an undirected network
			(N = 108, E = 276). Verifies that compute_setup honors its
			three priors on the undirected code path:

			Prior 1 (proportion missing, user input):
				N_add / (N_obs + N_add) ≈ pi_node within tolerance.
				Algebraic identity from _determine_n_add.

			Prior 2 (centrality-missingness correlation, user input):
				The beta solver returns beta such that
				_realized_rho_for_beta(beta, K, N_obs, N_add) ≈ rho_input
				within bisection tolerance. Or status is :ceiling_hit
				if the target was outside the achievable range.

			Prior 3 (E/I distribution match, inferred assumption):
				ei_conditional is row-stochastic and matches the
				empirical conditional distribution from G_obs's
				binning.

			This test exercises two paths different from Moreno (Test 19):
			(a) Undirected branch of compute_setup: R = nothing in
				SamplerSetup
			(b) Slightly larger N: 11 dropped nodes per replicate at
				rate=0.10 vs Moreno's 7

			A FAIL on the undirected path would be diagnostic: the
			directed path already passes (Moreno), so any failure on
			Scotland implicates the undirected branch specifically.
			A FAIL on Prior 2 would suggest a scale-dependent bisection
			issue or a real-network condition not captured by the unit
			tests.
		"""
		println("─" ^ 70)
		println("Test 20: Algorithmic-contract calibration on scotland_interlock_unweighted")
		println("─" ^ 70)

		#	Load Scotland from disk
			graphml_path = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks/scotland_interlock_unweighted.graphml"
			net = Network_Credible_Intervals.load_graphml(graphml_path)
			println("  Loaded: scotland_interlock_unweighted")
			println("  N = $(nrow(net.nodes)), E = $(nrow(net.edges))")
			println("  Directed: $(net.metadata.directed), Weighted: $(net.metadata.weighted)")
			println()

		#	Run three calibration cells
			K = 4
			J = 3
			R = 20
			cells = [
				(0.0,  0.10),
				(0.5,  0.10),
				(-0.5, 0.10),
			]

			cell_results = NamedTuple[]
			for (rho_input, rate_input) in cells
				print("  Running cell (rho=$rho_input, rate=$rate_input)... ")
				cell = _algorithmic_contract_cell(net, rho_input, rate_input;
													R = R, K = K, J = J)
				push!(cell_results, cell)
				println("done.")
			end

			println()
			println("=" ^ 70)
			println("ALGORITHMIC-CONTRACT RESULTS")
			println("=" ^ 70)
			for cell in cell_results
				_print_contract_cell(cell)
			end

		#	Gates: all three priors must be honored across all replicates
			gate_1 = all(c.prior_1_pass for c in cell_results)
			gate_2 = all(c.prior_2_pass for c in cell_results)
			gate_3 = all(c.prior_3_pass for c in cell_results)

			println("=" ^ 70)
			println("Gates (algorithmic contract):")
			println("  Prior 1 (proportion missing):           $(gate_1 ? "PASS" : "FAIL")")
			println("  Prior 2 (centrality correlation):       $(gate_2 ? "PASS" : "FAIL")")
			println("  Prior 3 (E/I-given-degree consistency): $(gate_3 ? "PASS" : "FAIL")")
			println("=" ^ 70)

			all_pass = gate_1 && gate_2 && gate_3

			println()
			println("  Test 20 result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end
	test_calibration_scotland()

#	Test 20b: Scotland algorithmic-contract calibration at K=10
	function test_calibration_scotland_k10()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Same algorithmic-contract test as Test 20 but with K=10 instead
			of K=4. The purpose is to verify the diagnosis that Test 20's
			two :ceiling_hit replicates on the rho=+0.5 cell were due to
			Phase 2's K=4 binning ceiling, not the algorithm.

			Phase 1's realized rho at rho=+0.5 ranged [0.11, 0.48] across
			the 20 replicates. With K=4, Phase 2's achievable ceiling is
			around 0.45 (per Test 16 diagnostic, K=4 ceiling is below
			K=10's ~0.52). At K=10, the achievable ceiling expands to
			around ±0.62 (extrapolating from Test 16's K=10 result).

			Expected: 0 :ceiling_hit replicates across all 60 reps. All
			three priors should still pass.

			If we still see :ceiling_hit on rho=+0.5 at K=10: the issue
			isn't K-discretization but something else (e.g., Phase 1's
			realized rho occasionally reaching values that exceed Phase 2's
			ceiling at any K, suggesting the calibration target itself is
			problematic). Unlikely but worth checking.
		"""
		println("─" ^ 70)
		println("Test 20b: Algorithmic-contract calibration on Scotland at K=10")
		println("─" ^ 70)

		#	Load Scotland from disk
			graphml_path = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks/scotland_interlock_unweighted.graphml"
			net = Network_Credible_Intervals.load_graphml(graphml_path)
			println("  Loaded: scotland_interlock_unweighted")
			println("  N = $(nrow(net.nodes)), E = $(nrow(net.edges))")
			println("  Directed: $(net.metadata.directed), Weighted: $(net.metadata.weighted)")
			println()

		#	Same three cells, K = 10 instead of K = 4
			K = 10
			J = 3
			R = 20
			cells = [
				(0.0,  0.10),
				(0.5,  0.10),
				(-0.5, 0.10),
			]

			cell_results = NamedTuple[]
			for (rho_input, rate_input) in cells
				print("  Running cell (rho=$rho_input, rate=$rate_input)... ")
				cell = _algorithmic_contract_cell(net, rho_input, rate_input;
													R = R, K = K, J = J)
				push!(cell_results, cell)
				println("done.")
			end

			println()
			println("=" ^ 70)
			println("ALGORITHMIC-CONTRACT RESULTS (K=10)")
			println("=" ^ 70)
			for cell in cell_results
				_print_contract_cell(cell)
			end

		#	Gates
			gate_1 = all(c.prior_1_pass for c in cell_results)
			gate_2 = all(c.prior_2_pass for c in cell_results)
			gate_3 = all(c.prior_3_pass for c in cell_results)

			#	Bonus diagnostic: zero ceiling-hit across all cells
				total_ceiling_hit = sum(count(s -> s == :ceiling_hit, c.rep_beta_status)
										for c in cell_results)
				no_ceiling_hit = total_ceiling_hit == 0

			println("=" ^ 70)
			println("Gates (algorithmic contract, K=10):")
			println("  Prior 1 (proportion missing):           $(gate_1 ? "PASS" : "FAIL")")
			println("  Prior 2 (centrality correlation):       $(gate_2 ? "PASS" : "FAIL")")
			println("  Prior 3 (E/I-given-degree consistency): $(gate_3 ? "PASS" : "FAIL")")
			println("  Diagnostic: 0 :ceiling_hit at K=10:     $(no_ceiling_hit ? "PASS" : "FAIL ($total_ceiling_hit)")")
			println("=" ^ 70)

			all_pass = gate_1 && gate_2 && gate_3

			println()
			println("  Test 20b result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end
	test_calibration_scotland_k10()

#	Test 21: Marvel algorithmic-contract calibration (stress test)
	function test_calibration_marvel()
		"""
		Args: none
		Returns:
			Bool: true if test passes
		Notes:
			Algorithmic-contract calibration test on marvel (N=6,486),
			derived from a bipartite hero-comic projection with threshold
			≥ 2 (Alberich-Miró-Juliá-Roselló 2002).

			Marvel exercises three properties that didn't show up on
			Moreno/Scotland:
			(a) Scale: 6,486 nodes vs 70-108. Phase 1 sampling cost rises;
				CHAMP cost rises substantially (~30 sec/call vs <1 sec).
			(b) Asymmetric ceiling: positive ceiling ~+0.41, negative
				ceiling ~-0.03. The negative cell will be heavily
				ceiling-pulled.
			(c) Heavy tail: the centrality distribution has many low-
				centrality nodes and a few hubs. The within-bin
				distribution is dramatically non-uniform.

			K = 10 (the design default) is used. With ~5,800 observed
			nodes at rate=0.10, each bin holds ~580 nodes — empty-bin
			edge cases should not fire.

			Expected behavior:
			- Prior 1: PASS (arithmetic)
			- Prior 2: PASS, possibly with some :ceiling_hit on rho=+0.5
			- Prior 3: PASS

			Wall-clock estimate: ~30 minutes at R=20.
		"""
		println("─" ^ 70)
		println("Test 21: Algorithmic-contract calibration on marvel (STRESS TEST)")
		println("─" ^ 70)
		println("  WARNING: This test runs CHAMP on Marvel many times.")
		println("  Estimated wall clock: ~30 minutes at R=20")
		println()

		#	Load Marvel from disk
			graphml_path = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks/marvel_universe_unweighted.graphml"
			net = Network_Credible_Intervals.load_graphml(graphml_path)
			println("  Loaded: marvel")
			println("  N = $(nrow(net.nodes)), E = $(nrow(net.edges))")
			println("  Directed: $(net.metadata.directed), Weighted: $(net.metadata.weighted)")
			println()

		#	Run three calibration cells at K=10 (design default)
			K = 10
			J = 3
			R = 20
			cells = [
				(0.0,  0.10),
				(0.5,  0.10),
				(-0.5, 0.10),
			]

			cell_results = NamedTuple[]
			for (rho_input, rate_input) in cells
				print("  Running cell (rho=$rho_input, rate=$rate_input)... ")
				flush(stdout)
				cell = _algorithmic_contract_cell(net, rho_input, rate_input;
													R = R, K = K, J = J)
				push!(cell_results, cell)
				println("done.")
			end

			println()
			println("=" ^ 70)
			println("ALGORITHMIC-CONTRACT RESULTS (Marvel, K=10)")
			println("=" ^ 70)
			for cell in cell_results
				_print_contract_cell(cell)
			end

		#	Gates
			gate_1 = all(c.prior_1_pass for c in cell_results)
			gate_2 = all(c.prior_2_pass for c in cell_results)
			gate_3 = all(c.prior_3_pass for c in cell_results)

			#	Informational diagnostic: count of :ceiling_hit
				total_ceiling_hit = sum(count(s -> s == :ceiling_hit, c.rep_beta_status)
										for c in cell_results)

			println("=" ^ 70)
			println("Gates (algorithmic contract, Marvel K=10):")
			println("  Prior 1 (proportion missing):           $(gate_1 ? "PASS" : "FAIL")")
			println("  Prior 2 (centrality correlation):       $(gate_2 ? "PASS" : "FAIL")")
			println("  Prior 3 (E/I-given-degree consistency): $(gate_3 ? "PASS" : "FAIL")")
			println("  Informational: ceiling_hit count:        $total_ceiling_hit / 60 reps")
			println("=" ^ 70)

			all_pass = gate_1 && gate_2 && gate_3

			println()
			println("  Test 21 result: ", all_pass ? "PASS ✓" : "FAIL ✗")
			println()

		return all_pass
	end
	test_calibration_marvel()


