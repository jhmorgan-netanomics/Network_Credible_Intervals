module network_reconstruction

#	Module Packages
	using DataFrames
	using SparseArrays
	using LinearAlgebra
	using Random
	using Statistics
	using StatsBase
	using ProgressMeter
	using ..Network_Credible_Intervals

#	Sibling Submodule Helpers
#	These are deliberately reach-imported from sibling submodules so the
#	reconstruction pipeline can call community detection (Phase 1.5) and
#	the Phase 1 materializers without duplicating their implementations.
#	All three submodules sit inside the same parent package, so this is a
#	within-package import.
	using ..network_community_detection: champ_community_detection
	using ..network_degeneracy: apply_missingness,
	                            apply_missingness_outgoing_only

#	====================================================================
#	network_reconstruction submodule
#
#	Implements Phase 1.5 (community-solution corpus) and Phase 2 (the
#	prior-informed bootstrap framework) of the validation pipeline.
#	Phase 1.5 consumes the Phase 1 degeneration corpus and produces a
#	corpus of CHAMP community solutions, one per Phase 1 row, used as
#	an input to Phase 2's setup phase. Phase 2 implements the framework
#	specified in Network_Confidence_Intervals.tex: a layered bootstrap
#	with Stage 0.5 (nominated non-respondents via the reciprocity
#	matrix R), Stage 1 (synthetic added nodes via the rank-rank matrix
#	P), and Stage 2 (edge-weight redistribution).
#
#	The file is organized into six sections, currently built up to
#	Section 4:
#		1. Phase 1.5 orchestrator (build_community_corpus)
#		2. Phase 2 SamplerSetup struct
#		3. Phase 2 setup-phase developer helpers (one per step)
#		4. Phase 2 setup user-facing function (compute_setup)
#		5. Phase 2 replicate-phase developer helpers (Pass 2)
#		6. Phase 2 replicate user-facing function (generate_replicate)
#		   (Pass 2)
#
#	Design conventions:
#		- Developer helpers are underscore-prefixed and operate on
#		  primitive inputs (edges, nodes, vectors); they have no
#		  knowledge of SamplerSetup.
#		- User-facing phase functions consume and produce SamplerSetup
#		  and orchestrate the helpers in the order specified by the
#		  design document.
#		- The P, w, and R matrices are 4-dimensional arrays indexed
#		  [degree_source, ei_source, degree_target, ei_target]; in
#		  degree-only fallback mode the E/I dimensions have length 1
#		  but the type signature stays the same.
#		- Laplace smoothing constants (+1 numerator, +2 denominator)
#		  are defined as module-level constants for transparency.
#	====================================================================

#	Module-Level Constants
	#	Laplace smoothing applied to P and R matrices to handle empty cells.
	#	Numerator and denominator additions are kept symbolic so that future
	#	tuning (e.g., to a Beta(alpha, alpha) prior) requires only changes
	#	in this one location.
		const LAPLACE_NUMERATOR_ADD = 1
		const LAPLACE_DENOMINATOR_ADD = 2

	#	Default E/I semantic thresholds used when J = 3. These partition
	#	the E/I axis at (-0.33, +0.33), corresponding to "internal hub",
	#	"mixed", and "broker" respectively. Other values of J use equal-
	#	size rank bins instead.
		const EI_SEMANTIC_THRESHOLDS_J3 = (-0.33, 0.33)

	#	Default hyperparameters for the K x J binning grid.
		const DEFAULT_K = 10
		const DEFAULT_J = 3

	#	Beta-solver bounds and tolerance. The numerical bisection on beta
	#	is closed-form per iteration (no Monte Carlo inner samples) so the
	#	tolerance can be tight.
		const BETA_BISECTION_BOUNDS = (-50.0, 50.0)
		const BETA_BISECTION_TOL = 1e-4
		const BETA_BISECTION_MAX_ITERS = 100

#####################################################
#       SECTION 1: PHASE 1.5 ORCHESTRATOR           #
#####################################################

#	Build Community Solutions Corpus: per-row CHAMP labels for Phase 1.5
	function build_community_corpus(phase1_corpus::DataFrame,
									networks::Dict{String, <:NamedTuple};
									master_seed::Int = 42,
									resolution_range::Tuple{Float64,Float64} = (0.1, 1.8),
									n_resolutions::Int = 30,
									n_runs_per_gamma::Int = 5,
									n_iterations_per_run::Int = 10,
									parallel::Bool = true,
									show_progress::Bool = true)
		"""
		Args:
			phase1_corpus::DataFrame: degeneration corpus produced by
				build_degeneration_corpus. Must include columns
				:network_name, :nominal_rho, :nominal_rate,
				:replicate_idx, :mechanism, :dropped_nodes
			networks::Dict{String, <:NamedTuple}: source networks keyed by
				network_name. Each value is a NamedTuple with fields
				:edges (DataFrame), :nodes (DataFrame), :metadata
				(NamedTuple with at least :directed and :weighted)
			master_seed::Int: master seed for the Phase 1.5 namespace,
				independent from Phase 1's master_seed (default = 42)
			resolution_range::Tuple{Float64,Float64}: gamma range passed to
				CHAMP (default = (0.1, 1.8))
			n_resolutions::Int: number of gamma values in the CHAMP sweep
				(default = 30)
			n_runs_per_gamma::Int: Leiden multi-starts per gamma in CHAMP
				(default = 5)
			n_iterations_per_run::Int: max Leiden iterations per run in
				CHAMP (default = 10)
			parallel::Bool: thread the outer loop over corpus rows via
				Threads.@threads :static (default = true). When true,
				CHAMP is called with parallel_runs=false to avoid nested
				threading.
			show_progress::Bool: display a ProgressMeter progress bar
				during the outer loop (default = true)
		Returns:
			DataFrame: one row per row of phase1_corpus, with columns
				:network_name           identifier (from Phase 1)
				:nominal_rho            identifier (from Phase 1)
				:nominal_rate           identifier (from Phase 1)
				:replicate_idx          identifier (from Phase 1)
				:mechanism              identifier (from Phase 1)
				:community_seed         Int seed used for the CHAMP call
				:community_labels       Vector{Int}, CHAMP membership in
				                        materialized-network node order
				:resolution_used        Float64, gamma chosen by CHAMP
				:modularity             Float64, modularity at that gamma
				:n_communities          Int, partition size
				:materialization_n_nodes Int, node count of the materialized
				                        network (verification field for
				                        Phase 2 consumers)
		Notes:
			Phase 1.5 of the validation pipeline. Consumes the Phase 1
			degeneration corpus (which stores dropped-node sets) and
			produces a corpus of community solutions, one per Phase 1
			row. The output is consumed by Phase 2's reconstruction
			framework: each Phase 2 setup-phase call joins to this corpus
			on (network_name, nominal_rho, nominal_rate, replicate_idx,
			mechanism) and reads community_labels rather than running
			CHAMP itself.

			Per-row processing:
			  1. Look up the source network in `networks`.
			  2. Materialize the degenerated network using the appropriate
				 apply_missingness helper based on :mechanism.
			  3. Derive a CHAMP seed via
					community_seed = Int(hash((master_seed, row_idx)) % UInt32)
				 so the corpus is bit-reproducible from master_seed and
				 thread-scheduling order does not affect labels.
			  4. Call champ_community_detection on the materialized network
				 with parallel_runs=false (the outer loop owns parallelism).
				 The :directed and :weighted flags are taken from the source
				 network's metadata; CHAMP handles symmetrization for
				 directed-graph community detection internally per the
				 framework design specification.
			  5. Record the community labels, the gamma CHAMP selected,
				 the modularity at that gamma, the partition size, and
				 the materialized node count for downstream verification.

			Skip handling:
			- Phase 1 rows with status :failed_other have empty
			  dropped_nodes. The materialization for these is the
			  original network; CHAMP runs on the undegenerated network
			  and the labels are recorded. Phase 2's consumer can choose
			  to filter these rows out, but Phase 1.5 records them for
			  completeness.

			Reproducibility:
			- Same phase1_corpus + same networks + same master_seed
			  produces a bit-identical community-solutions corpus
			  regardless of thread count. The per-row seed is derived
			  from (master_seed, row_idx) via hash, so reordering of
			  thread execution does not change which seed gets used
			  for which row.

			Threading:
			- The outer loop is parallelized via Threads.@threads :static.
			  Inside each iteration, CHAMP runs serially (parallel_runs=
			  false) and silently (show_progress=false at the CHAMP call
			  site). This avoids nested thread oversubscription. The
			  outer ProgressMeter progress bar is updated under a lock.

			Output ordering:
			- The result DataFrame preserves row order from phase1_corpus
			  exactly. The k-th row of the output corresponds to the k-th
			  row of the input.
		"""

		#	Guards
			required_cols = [:network_name, :nominal_rho, :nominal_rate,
							:replicate_idx, :mechanism, :dropped_nodes]
			for col in required_cols
				hasproperty(phase1_corpus, col) ||
					throw(ArgumentError("phase1_corpus missing required column :$col"))
			end
			isempty(networks) && throw(ArgumentError("networks dictionary is empty"))
			n_rows = nrow(phase1_corpus)
			n_rows >= 1 || throw(ArgumentError("phase1_corpus has zero rows"))

		#	Pre-Allocate Per-Row Result Vectors
			#	One slot per input row; results are written into pre-allocated
			#	vectors in-place by the threaded loop so that row order is
			#	preserved deterministically.
				community_seeds          = Vector{Int}(undef, n_rows)
				community_labels_vec     = Vector{Vector{Int}}(undef, n_rows)
				resolution_used_vec      = Vector{Float64}(undef, n_rows)
				modularity_vec           = Vector{Float64}(undef, n_rows)
				n_communities_vec        = Vector{Int}(undef, n_rows)
				materialization_n_nodes  = Vector{Int}(undef, n_rows)

		#	Progress Bar Setup
			prog = nothing
			prog_lock = ReentrantLock()
			if show_progress
				desc = parallel ?
					"Community corpus ($n_rows rows, $(Threads.nthreads()) threads)" :
					"Community corpus ($n_rows rows, serial)"
				prog = Progress(n_rows, desc = desc, enabled = true)
			end

		#	Per-Row Processing Closure
			#	Defined as a closure so the threaded and serial branches share
			#	the same body without duplication.
				function process_row(row_idx::Int)
					row = phase1_corpus[row_idx, :]

					#	Look Up Source Network
						haskey(networks, row.network_name) ||
							error("network '$(row.network_name)' not found in networks dict (row $row_idx)")
						net = networks[row.network_name]
						directed = net.metadata.directed
						weighted = net.metadata.weighted

					#	Materialize Degenerated Network
						#	The dropped_nodes column is stored as an Arrow
						#	SubArray on disk; collect() produces a concrete
						#	Vector{Int} for downstream code. Mechanism may
						#	round-trip from Arrow as String rather than Symbol.
							dropped = collect(row.dropped_nodes)
							mech = row.mechanism isa Symbol ? row.mechanism : Symbol(row.mechanism)

							if mech == :full_removal
								materialized = apply_missingness(net.edges, dropped;
																 nodes = net.nodes)
							elseif mech == :outgoing_only
								#	apply_missingness_outgoing_only is a directed-only
								#	mechanism; we trust the Phase 1 corpus to only
								#	have emitted :outgoing_only rows on directed
								#	networks, but the function itself guards against
								#	undirected misuse.
									materialized = apply_missingness_outgoing_only(
										net.edges, dropped;
										nodes    = net.nodes,
										directed = directed
									)
							else
								error("unknown mechanism $mech at row $row_idx")
							end

					#	Derive CHAMP Seed Deterministically From (master_seed, row_idx)
						community_seed = Int(hash((master_seed, row_idx)) % UInt32)

					#	Run CHAMP on Materialized Network
						#	The framework specification requires symmetrization for
						#	community detection on directed networks. CHAMP handles
						#	this internally based on the :directed flag.
							champ_result = champ_community_detection(
								materialized.edges;
								nodes                = materialized.nodes,
								resolution_range     = resolution_range,
								n_resolutions        = n_resolutions,
								n_runs_per_gamma     = n_runs_per_gamma,
								n_iterations_per_run = n_iterations_per_run,
								weighted             = weighted,
								directed             = directed,
								seed                 = community_seed,
								show_progress        = false,
							)

					#	Record Per-Row Results
						community_seeds[row_idx]         = community_seed
						community_labels_vec[row_idx]    = champ_result.membership
						resolution_used_vec[row_idx]     = champ_result.resolution_used
						modularity_vec[row_idx]          = champ_result.modularity
						n_communities_vec[row_idx]       = champ_result.n_communities
						materialization_n_nodes[row_idx] = nrow(materialized.nodes)

					#	Update Progress Bar Under Lock
						if show_progress && prog !== nothing
							lock(prog_lock) do
								next!(prog)
							end
						end

					return nothing
				end

		#	Outer Loop: Threaded or Serial
			if parallel
				Threads.@threads :static for row_idx in 1:n_rows
					process_row(row_idx)
				end
			else
				for row_idx in 1:n_rows
					process_row(row_idx)
				end
			end

		#	Assemble Output DataFrame
			#	Identifier columns are copied from phase1_corpus to preserve
			#	exact join keys. New columns are the CHAMP outputs.
				result = DataFrame(
					network_name             = copy(phase1_corpus.network_name),
					nominal_rho              = copy(phase1_corpus.nominal_rho),
					nominal_rate             = copy(phase1_corpus.nominal_rate),
					replicate_idx            = copy(phase1_corpus.replicate_idx),
					mechanism                = copy(phase1_corpus.mechanism),
					community_seed           = community_seeds,
					community_labels         = community_labels_vec,
					resolution_used          = resolution_used_vec,
					modularity               = modularity_vec,
					n_communities            = n_communities_vec,
					materialization_n_nodes  = materialization_n_nodes,
				)

		#	Return Community Solutions Corpus
			return result
	end
	@doc raw"""
	**Description**
	Build the Phase 1.5 community-solutions corpus: for each row of the Phase 1
	degeneration corpus, materialize the degenerated network and run CHAMP
	to produce a community partition. The result is a corpus of community
	solutions keyed by the same (network_name, nominal_rho, nominal_rate,
	replicate_idx, mechanism) tuple as the Phase 1 corpus, with community
	labels indexed in materialized-network node order. The output is consumed
	by Phase 2's reconstruction framework so that the framework can skip
	community detection entirely and instead join to precomputed labels.

	**Usage**
	`build_community_corpus(phase1_corpus, networks; master_seed=42, resolution_range=(0.1, 1.8), n_resolutions=30, n_runs_per_gamma=5, n_iterations_per_run=10, parallel=true, show_progress=true)`

	**Arguments**
	- `phase1_corpus::DataFrame`: The Phase 1 degeneration corpus produced by `build_degeneration_corpus`. Must contain at minimum `:network_name`, `:nominal_rho`, `:nominal_rate`, `:replicate_idx`, `:mechanism`, and `:dropped_nodes`.
	- `networks::Dict{String, <:NamedTuple}`: The source networks keyed by `network_name`. Each value is a `NamedTuple` with `:edges`, `:nodes`, and `:metadata` (which itself has at least `:directed` and `:weighted` fields).
	- `master_seed::Int`: Master seed for the Phase 1.5 namespace, independent from Phase 1's master_seed (default `42`).
	- `resolution_range::Tuple{Float64,Float64}`: Gamma range passed to CHAMP's resolution sweep (default `(0.1, 1.8)`).
	- `n_resolutions::Int`: Number of gamma values in the CHAMP sweep (default `30`).
	- `n_runs_per_gamma::Int`: Leiden multi-starts per gamma inside CHAMP (default `5`).
	- `n_iterations_per_run::Int`: Maximum Leiden iterations per run inside CHAMP (default `10`).
	- `parallel::Bool`: Parallelize the outer loop over corpus rows via `Threads.@threads :static` (default `true`). When true, the inner CHAMP call uses `parallel_runs=false` to avoid nested threading.
	- `show_progress::Bool`: Display a ProgressMeter progress bar over the outer loop (default `true`).

	**Details**
	Phase 1.5 sits between Phase 1 (degeneration sampling, which produces dropped-node sets) and Phase 2 (reconstruction framework, which produces credible intervals). It runs CHAMP once per Phase 1 corpus row on the materialized degenerated network and stores the resulting community partition. By precomputing community structure as its own corpus, Phase 2 avoids running CHAMP inside its inner replicate loop and gains a clean separation between community detection and framework execution.

	**Reproducibility.** Same `phase1_corpus` + same `networks` + same `master_seed` produces a bit-identical community-solutions corpus regardless of thread count. The seed-per-row scheme uses `hash((master_seed, row_idx))`, so thread-scheduling order does not affect which seed gets used for which row.

	**Threading.** The outer loop is parallelized via `Threads.@threads :static`. Inside each iteration, CHAMP runs serially (`parallel_runs=false`) and silently (`show_progress=false`). This avoids nested thread oversubscription. The outer ProgressMeter progress bar is updated under a `ReentrantLock` to serialize display I/O.

	**Wall-clock cost.** CHAMP at default settings is the dominant per-row cost, scaling roughly with `n_resolutions * n_runs_per_gamma * CHAMP_iter_cost`. For Marvel-scale networks (~6,500 nodes) the per-row cost may exceed 30 seconds; for small networks (Moreno, Scotland, Toledo) it is sub-second.

	**Skip handling.** Phase 1 rows with `bisection_status = :failed_other` have empty `:dropped_nodes`. The materialization for such rows is the original (undegenerated) network. CHAMP still runs and the labels are recorded. Phase 2's consumer can choose to filter these rows out at framework time; Phase 1.5 records them for completeness and reproducibility.

	**Value**
	A `DataFrame` with one row per row of `phase1_corpus`, preserving row order exactly. Columns:
	- Identifier columns (copied from `phase1_corpus`): `:network_name`, `:nominal_rho`, `:nominal_rate`, `:replicate_idx`, `:mechanism`.
	- `:community_seed::Int`: The seed passed to CHAMP for this row.
	- `:community_labels::Vector{Int}`: CHAMP membership vector indexed in materialized-network node order.
	- `:resolution_used::Float64`: The gamma value CHAMP selected at the convex-hull bend.
	- `:modularity::Float64`: Modularity at the selected gamma.
	- `:n_communities::Int`: Number of communities in the selected partition.
	- `:materialization_n_nodes::Int`: Node count of the materialized network.

	**See Also**
	`champ_community_detection`, `build_degeneration_corpus`, `apply_missingness`, `apply_missingness_outgoing_only`, `compute_setup`

	**References**
	Weir WH, Emmons S, Gibson R, Taylor D, Mucha PJ (2017) "Post-processing partitions to identify domains of modularity optimization." *Algorithms* 10(3):93.
	""" build_community_corpus

#####################################################
#     SECTION 2: PHASE 2 SAMPLERSETUP STRUCT        #
#####################################################

#	SamplerSetup: bundle of setup-phase outputs for the Phase 2 framework
	"""
	SamplerSetup

	Bundle of setup-phase outputs produced by compute_setup and consumed by
	generate_replicate. Holds the original edges and nodes DataFrames, the
	derived per-node bin assignments, the rank-rank connectivity matrix P,
	the conditional mean-weight matrix w, the reciprocity matrix R for
	directed networks, the beta-solved bin-assignment distribution q, and
	all diagnostic flags. One SamplerSetup is built per Gobs and reused
	across B replicates; the inner replicate loop in generate_replicate
	does no setup work beyond per-replicate RNG handling.

	Fields:
	  edges::DataFrame
	      The observed-network edge list (post-transformation if weight
	      transformation applies). Columns :src, :dst, optional :weight.
	  nodes::DataFrame
	      The observed-network node roster with columns :id, :label.
	      Includes any partially observed (nominated non-respondent)
	      nodes; these are flagged via partially_observed.
	  directed::Bool
	      Inherited from the network metadata.
	  weighted::Bool
	      Inherited from the network metadata.
	  pi_node::Float64
	      Fraction of nodes believed missing, in [0, 1).
	  pi_edge::Float64
	      Fraction of edge weight believed missing, in [0, 1).
	  rho::Float64
	      Target centrality-missingness correlation in (-1, 1).
	  centrality::Vector{Float64}
	      Per-node centrality scores in nodes-DataFrame order.
	  community_labels::Vector{Int}
	      Per-node community assignment, 1-based, contiguous. Length
	      matches nrow(nodes). In the framework design these are
	      pre-computed by Phase 1.5 (CHAMP) rather than detected here.
	  ei_values::Vector{Float64}
	      Per-node E/I index in (-1, +1). NaN entries are possible for
	      isolates with no edges; the binning step handles these.
	  binning_mode::Symbol
	      Either :two_dimensional (degree x E/I) or :degree_only (1D
	      fallback when community structure is not meaningful).
	  degree_bins::Vector{Int}
	      Per-node degree-bin assignment, 1..K.
	  ei_bins::Vector{Int}
	      Per-node E/I-bin assignment, 1..J. All ones in 1D fallback.
	  K::Int
	      Number of degree bins. Defaults to DEFAULT_K = 10.
	  J::Int
	      Number of E/I bins. Defaults to DEFAULT_J = 3 in 2D mode and
	      collapses to 1 in fallback mode.
	  P::Array{Float64,4}
	      Rank-rank connectivity probability matrix. Indexed as
	      P[deg_src, ei_src, deg_dst, ei_dst]. Laplace-smoothed.
	  w::Array{Float64,4}
	      Conditional mean-weight matrix, same indexing as P. Zero
	      entries indicate no observed edge in that cell pair; the
	      sampler falls back to weight=1 in those cases.
	  R::Union{Array{Float64,4}, Nothing}
	      Reciprocity matrix for directed networks; Nothing for
	      undirected. Same indexing as P. Laplace-smoothed.
	  partially_observed::Vector{Int}
	      Indices into nodes (the canonical node order) of the nominated
	      non-respondent nodes that Stage 0.5 processes. For Phase 1
	      corpus rows with :full_removal mechanism this is empty; for
	      :outgoing_only rows it is the dropped_nodes vector.
	  N_add::Int
	      Number of additional synthetic nodes to add per replicate in
	      Stage 1. Computed via the design's formula from pi_node, N,
	      and the count of partially observed nodes.
	  beta::Float64
	      The logistic-skew parameter for the added-node degree-bin
	      assignment distribution q. Numerically bisected during setup
	      such that the induced correlation between bin index and
	      missingness indicator matches rho.
	  beta_status::Symbol
	      :converged, :ceiling_hit, or :failed_other. Mirrors Phase 1's
	      bisection-status semantics.
	  q::Vector{Float64}
	      The degree-bin assignment distribution, length K, sums to 1.
	      Drawn from once per added node in Step 6 of each replicate.
	  ei_conditional::Matrix{Float64}
	      The empirical conditional distribution P(EI_bin | degree_bin)
	      over observed nodes, size K x J. Used to assign E/I bins to
	      added nodes given their degree bin assignment.
	  diagnostics::Dict{Symbol, Any}
	      Heterogeneous diagnostic fields for inspection: binning_mode,
	      beta_status, beta_n_iters, fallback_reason (if applicable),
	      and any other one-shot computations the setup phase records
	      for downstream verification.
	"""
	struct SamplerSetup
		#	Inputs
			edges::DataFrame
			nodes::DataFrame
			directed::Bool
			weighted::Bool
			pi_node::Float64
			pi_edge::Float64
			rho::Float64

		#	Per-Node Setup Outputs
			centrality::Vector{Float64}
			community_labels::Vector{Int}
			ei_values::Vector{Float64}
			binning_mode::Symbol
			degree_bins::Vector{Int}
			ei_bins::Vector{Int}
			K::Int
			J::Int

		#	Matrices (4D: [deg_src, ei_src, deg_dst, ei_dst])
			P::Array{Float64,4}
			w::Array{Float64,4}
			R::Union{Array{Float64,4}, Nothing}

		#	Added-Node Specification
			partially_observed::Vector{Int}
			N_add::Int
			beta::Float64
			beta_status::Symbol
			q::Vector{Float64}
			ei_conditional::Matrix{Float64}

		#	Diagnostics
			diagnostics::Dict{Symbol, Any}
	end

#####################################################
#     SECTION 3: PHASE 2 SETUP-PHASE HELPERS        #
#####################################################

#	Helper Step 1: Compute observed centrality from edges
	function _compute_observed_centrality(edges::DataFrame,
										   nodes::DataFrame,
										   directed::Bool)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (and optional :weight,
				unused for centrality computation)
			nodes::DataFrame: node roster with at least :id column; defines
				the canonical node order
			directed::Bool: if true, centrality is in-degree + out-degree;
				if false, centrality is undirected degree
		Returns:
			Vector{Float64}: per-node centrality in canonical nodes-DataFrame
				order. Isolates (nodes with no edges) get centrality 0.0.
		Notes:
			Implements Step 1 of the design specification: degree centrality
			used as the centrality driver for binning. Directed centrality
			is the sum of incoming and outgoing degree per the design
			document's "in-degree + out-degree per node" specification.
			Unweighted: each edge contributes 1 to its endpoints' degree
			regardless of weight semantics. This is binary-degree, matching
			Phase 1's centrality driver and ensuring consistency across
			pipeline stages.

			The function tolerates the case where some edge endpoints
			are absent from the nodes DataFrame only insofar as those
			endpoints are silently ignored; the canonical node order is
			determined by the nodes DataFrame's row order, not by the
			edge endpoints.
		"""

		#	Guards
			hasproperty(edges, :src) && hasproperty(edges, :dst) ||
				throw(ArgumentError("edges must have :src and :dst columns"))
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))

		#	Build Node ID -> Index Mapping
			n = nrow(nodes)
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))

		#	Allocate Centrality Vector
			centrality = zeros(Float64, n)

		#	Accumulate Per-Edge Contributions
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)

			@inbounds for k in 1:n_edges
				#	Source endpoint contributes to out-degree (directed) or
				#	degree (undirected); destination contributes to in-degree
				#	(directed) or degree (undirected). For binary degree,
				#	each edge contributes 1 regardless of weight.
					src_i = get(id_to_idx, src_ids[k], 0)
					dst_i = get(id_to_idx, dst_ids[k], 0)
					if src_i > 0
						centrality[src_i] += 1.0
					end
					if dst_i > 0
						centrality[dst_i] += 1.0
					end
			end

		#	Return Per-Node Centrality
			return centrality
	end

#	Helper Step 1.5: Compute E/I index from community labels and edges
	function _compute_ei_values(edges::DataFrame,
								 nodes::DataFrame,
								 community_labels::Vector{Int})
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst
			nodes::DataFrame: node roster with :id; defines canonical order
			community_labels::Vector{Int}: per-node community assignment in
				nodes-DataFrame order; length must equal nrow(nodes)
		Returns:
			Vector{Float64}: per-node E/I index in (-1, +1). Isolates with
				no edges return NaN; callers must handle these in the
				binning step.
		Notes:
			Implements the E/I (External-Internal) index from
			Krackhardt & Stern (1988):
			    EI_i = (E_i - I_i) / (E_i + I_i)
			where E_i is i's number of external edges (to other communities)
			and I_i is i's number of internal edges (within its own
			community). Range:
			    EI =  -1: all edges internal (pure cluster member)
			    EI =   0: equal split
			    EI =  +1: all edges external (pure broker)

			This is binary-degree-style E/I: each edge contributes 1 to
			either the external or internal count regardless of weight,
			matching the framework specification's use of degree-derived
			centrality. The function counts each directed edge once
			(from the source's perspective); for an undirected edge list
			where edges appear in only one direction, the function still
			behaves correctly because both endpoints accumulate from the
			same edge regardless of orientation.
		"""

		#	Guards
			n = nrow(nodes)
			length(community_labels) == n ||
				throw(ArgumentError("community_labels length $(length(community_labels)) does not match nodes count $n"))

		#	Build Node ID -> Index Mapping
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))

		#	Allocate Counters
			external_count = zeros(Int, n)
			internal_count = zeros(Int, n)

		#	Walk Edges and Increment Per-Endpoint Counters
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)

			@inbounds for k in 1:n_edges
				src_i = get(id_to_idx, src_ids[k], 0)
				dst_i = get(id_to_idx, dst_ids[k], 0)
				if src_i > 0 && dst_i > 0
					same_community = community_labels[src_i] == community_labels[dst_i]
					#	Each edge contributes to BOTH endpoints' E/I counters
						if same_community
							internal_count[src_i] += 1
							internal_count[dst_i] += 1
						else
							external_count[src_i] += 1
							external_count[dst_i] += 1
						end
				end
			end

		#	Compute E/I per Node
			ei_values = Vector{Float64}(undef, n)
			@inbounds for i in 1:n
				total = external_count[i] + internal_count[i]
				if total == 0
					ei_values[i] = NaN
				else
					ei_values[i] = (external_count[i] - internal_count[i]) / total
				end
			end

		#	Return E/I Vector
			return ei_values
	end

#	Helper Step 1.5: Detect community structure (consumes precomputed labels)
	function _detect_community_structure(edges::DataFrame,
										  nodes::DataFrame,
										  community_labels::Vector{Int};
										  J::Int = DEFAULT_J,
										  min_nodes_per_ei_bin::Int = 3)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst
			nodes::DataFrame: node roster with :id
			community_labels::Vector{Int}: precomputed labels (e.g., from
				Phase 1.5 build_community_corpus), in nodes-DataFrame order
			J::Int: target number of E/I bins (default = DEFAULT_J = 3)
			min_nodes_per_ei_bin::Int: minimum nodes per E/I bin required
				to keep 2D binning active (default = 3, per design spec)
		Returns:
			NamedTuple: (ei_values, binning_mode, fallback_reason)
				ei_values::Vector{Float64}: per-node E/I (NaN for isolates)
				binning_mode::Symbol: :two_dimensional or :degree_only
				fallback_reason::Union{Symbol, Nothing}: identifies why
					fallback was triggered if mode == :degree_only;
					Nothing otherwise. Values: :single_community,
					:insufficient_ei_variance, :too_few_nodes
		Notes:
			Implements Step 1.5 of the design specification. Phase 2's
			community detection is delegated to Phase 1.5 via the
			community_labels argument; this helper does not call CHAMP.
			Its job is to:
			  1. Compute E/I values from the labels and edges.
			  2. Check whether 2D binning is supportable: more than one
				 community, J E/I bins each with at least
				 min_nodes_per_ei_bin nodes.
			  3. Return the binning mode and a fallback reason if 2D
				 is not supportable.

			The fallback check follows the design specification: 2D
			binning requires at least 2 communities and at least
			J * min_nodes_per_ei_bin nodes with defined (non-NaN) E/I
			values. When the check fails, the framework falls back to
			degree-only binning, which is robust to networks lacking
			meaningful community structure.

			Isolates (nodes with no edges) have NaN E/I values; these
			are excluded from the supportability check.
		"""

		#	Guards
			n = nrow(nodes)
			length(community_labels) == n ||
				throw(ArgumentError("community_labels length $(length(community_labels)) does not match nodes count $n"))
			J >= 1 || throw(ArgumentError("J must be >= 1, got $J"))
			min_nodes_per_ei_bin >= 1 ||
				throw(ArgumentError("min_nodes_per_ei_bin must be >= 1, got $min_nodes_per_ei_bin"))

		#	Compute E/I Values
			ei_values = _compute_ei_values(edges, nodes, community_labels)

		#	Check Fallback Conditions
			n_communities = length(unique(community_labels))
			n_defined_ei = count(!isnan, ei_values)

			if n_communities < 2
				#	Single Community: E/I is identically -1 for all defined
				#	nodes; no meaningful 2D structure.
					return (ei_values = ei_values,
							binning_mode = :degree_only,
							fallback_reason = :single_community)
			end

			if n_defined_ei < J * min_nodes_per_ei_bin
				#	Too few nodes with defined E/I to populate J bins.
					return (ei_values = ei_values,
							binning_mode = :degree_only,
							fallback_reason = :too_few_nodes)
			end

		#	Check E/I Variance Supportability
			#	The semantic-threshold binning at J = 3 uses (-0.33, +0.33)
			#	as cut points. For J = 3 specifically, check that all three
			#	bins receive at least min_nodes_per_ei_bin nodes when those
			#	thresholds are applied. For other J, use equal-quantile bins
			#	and check the same property.
				if J == 3
					lo_thresh, hi_thresh = EI_SEMANTIC_THRESHOLDS_J3
					n_lo  = count(x -> !isnan(x) && x <= lo_thresh, ei_values)
					n_mid = count(x -> !isnan(x) && lo_thresh < x < hi_thresh, ei_values)
					n_hi  = count(x -> !isnan(x) && x >= hi_thresh, ei_values)
					if min(n_lo, n_mid, n_hi) < min_nodes_per_ei_bin
						return (ei_values = ei_values,
								binning_mode = :degree_only,
								fallback_reason = :insufficient_ei_variance)
					end
				else
					#	Equal-quantile bins: rank the defined E/I values and
					#	check that each of the J quantile bins has >= min_nodes.
						defined_ei = filter(!isnan, ei_values)
						quantile_size = length(defined_ei) ÷ J
						if quantile_size < min_nodes_per_ei_bin
							return (ei_values = ei_values,
									binning_mode = :degree_only,
									fallback_reason = :insufficient_ei_variance)
						end
				end

		#	2D Binning Supportable
			return (ei_values = ei_values,
					binning_mode = :two_dimensional,
					fallback_reason = nothing)
	end

#	Helper Step 2: Bin observed nodes into K degree bins and J E/I bins
	function _bin_observed_nodes(centrality::Vector{Float64},
								  ei_values::Vector{Float64},
								  K::Int,
								  J::Int,
								  binning_mode::Symbol)
		"""
		Args:
			centrality::Vector{Float64}: per-node centrality from
				_compute_observed_centrality
			ei_values::Vector{Float64}: per-node E/I from
				_compute_ei_values; NaN entries are tolerated
			K::Int: number of degree bins
			J::Int: number of E/I bins
			binning_mode::Symbol: :two_dimensional or :degree_only
		Returns:
			NamedTuple: (degree_bins, ei_bins, J_effective)
				degree_bins::Vector{Int}: per-node degree bin, 1..K
				ei_bins::Vector{Int}: per-node E/I bin, 1..J or all 1s
					if mode is :degree_only
				J_effective::Int: J if mode is :two_dimensional, 1 if
					:degree_only. The downstream P, w, R matrices use
					J_effective as their E/I dimension size.
		Notes:
			Implements Step 2 of the design specification. Degree bins
			are equal-rank: rank nodes by centrality, partition into K
			contiguous rank groups with ties broken consistently. Bin 1
			is the most peripheral (lowest centrality), bin K is the
			most central.

			E/I bins (when active) use semantic thresholds at J = 3:
			(-inf, -0.33] -> bin 1 (internal hub)
			(-0.33, 0.33) -> bin 2 (mixed)
			[0.33, +inf)  -> bin 3 (broker / external)
			For other J, use equal-quantile bins on the defined
			(non-NaN) E/I values.

			Isolates with NaN E/I are assigned to the middle E/I bin
			(bin (J+1) / 2 rounded up) as a default placement, which
			treats them as "mixed". This is a defensible default but
			the bin assignment for nodes with no edges is effectively
			a no-op because they have no edges to propagate through
			the P, w, R matrices anyway.
		"""

		#	Guards
			n = length(centrality)
			length(ei_values) == n ||
				throw(ArgumentError("centrality and ei_values must have same length"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			J >= 1 || throw(ArgumentError("J must be >= 1, got $J"))
			binning_mode in (:two_dimensional, :degree_only) ||
				throw(ArgumentError("binning_mode must be :two_dimensional or :degree_only, got $binning_mode"))

		#	Degree Binning: Equal-Rank Partition into K Bins
			#	StatsBase.tiedrank gives average rank to ties; for equal-size
			#	bins we then quantize ranks into K groups.
				ranks = StatsBase.tiedrank(centrality)
				degree_bins = Vector{Int}(undef, n)
				@inbounds for i in 1:n
					#	Map rank in [1, n] to bin in [1, K]
						#	Use ceil((rank / n) * K), clamped to [1, K] for safety
						bin = Int(ceil((ranks[i] / n) * K))
						degree_bins[i] = clamp(bin, 1, K)
				end

		#	E/I Binning: Semantic Thresholds at J=3, Otherwise Equal-Quantile
			if binning_mode == :degree_only
				#	1D Fallback: All nodes in E/I bin 1.
					ei_bins = ones(Int, n)
					return (degree_bins = degree_bins,
							ei_bins = ei_bins,
							J_effective = 1)
			end

			#	2D Mode: Bin by E/I
				ei_bins = Vector{Int}(undef, n)
				if J == 3
					lo_thresh, hi_thresh = EI_SEMANTIC_THRESHOLDS_J3
					@inbounds for i in 1:n
						if isnan(ei_values[i])
							ei_bins[i] = 2   # middle bin for isolates
						elseif ei_values[i] <= lo_thresh
							ei_bins[i] = 1
						elseif ei_values[i] >= hi_thresh
							ei_bins[i] = 3
						else
							ei_bins[i] = 2
						end
					end
				else
					#	Equal-quantile bins on defined E/I
						defined_mask = .!isnan.(ei_values)
						defined_ranks = StatsBase.tiedrank(ei_values[defined_mask])
						defined_n = length(defined_ranks)
						defined_bins = Vector{Int}(undef, defined_n)
						@inbounds for i in 1:defined_n
							bin = Int(ceil((defined_ranks[i] / defined_n) * J))
							defined_bins[i] = clamp(bin, 1, J)
						end
						#	Scatter back into full vector; NaN entries get
						#	the middle bin.
							middle_bin = (J + 1) ÷ 2
							defined_idx = 1
							@inbounds for i in 1:n
								if defined_mask[i]
									ei_bins[i] = defined_bins[defined_idx]
									defined_idx += 1
								else
									ei_bins[i] = middle_bin
								end
							end
				end

		#	Return Per-Node Bin Assignments
			return (degree_bins = degree_bins,
					ei_bins = ei_bins,
					J_effective = J)
	end

#	Helper Step 3: Compute the rank-rank connectivity matrix P (and w)
	function _compute_p_matrix(edges::DataFrame,
								nodes::DataFrame,
								degree_bins::Vector{Int},
								ei_bins::Vector{Int},
								K::Int,
								J::Int,
								directed::Bool,
								weighted::Bool;
								partially_observed::Vector{Int} = Int[])
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optional :weight
			nodes::DataFrame: node roster with :id
			degree_bins::Vector{Int}: per-node degree bin, 1..K
			ei_bins::Vector{Int}: per-node E/I bin, 1..J
			K::Int: number of degree bins
			J::Int: number of E/I bins (use 1 in degree-only fallback)
			directed::Bool: if true, P[c, c'] != P[c', c] in general; if
				false, P is symmetric and computed on unordered pairs
			weighted::Bool: if true, also compute the conditional
				mean-weight matrix w; if false, w is filled with zeros
			partially_observed::Vector{Int}: indices of partially observed
				(nominated non-respondent) nodes to exclude from the P
				calculation. The design specifies P is computed over
				respondent-respondent dyads only.
		Returns:
			NamedTuple: (P, w)
				P::Array{Float64,4}: Laplace-smoothed connection
					probabilities, indexed [deg_src, ei_src, deg_dst, ei_dst]
				w::Array{Float64,4}: conditional mean weights, same indexing
		Notes:
			Implements Step 3 of the design specification. The matrix P is
			indexed by cell pairs (c, c') where each cell is (degree_bin,
			ei_bin). Laplace smoothing:
			    P[c, c'] = (n_edges + LAPLACE_NUMERATOR_ADD) /
			               (n_dyads + LAPLACE_DENOMINATOR_ADD)
			where n_edges is the count of observed edges from cell c to
			cell c', and n_dyads is the count of ordered (directed) or
			unordered (undirected) dyads between cells. With LAPLACE_
			NUMERATOR_ADD = 1 and LAPLACE_DENOMINATOR_ADD = 2, the
			smoothing corresponds to a Beta(1, 1) prior, equivalent to
			one prior observation each of "edge present" and "edge
			absent". Empty cells therefore default to P = 0.5.

			The conditional mean-weight matrix is:
			    w[c, c'] = mean weight of edges from cell c to cell c',
			              conditional on edge presence
			              (= 0 if no edges observed in that cell pair)
			Cell pairs with w = 0 fall back to weight = 1 at sampling
			time, per the design specification.

			Partially observed nodes (Stage 0.5 input) are excluded from
			the P calculation per the design: "The matrix is computed
			over all respondent-respondent dyads in Gobs (excluding any
			partially observed nodes, which are placed using their own
			observed edges separately)."
		"""

		#	Guards
			n = nrow(nodes)
			length(degree_bins) == n && length(ei_bins) == n ||
				throw(ArgumentError("bin vectors must match nodes count"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			J >= 1 || throw(ArgumentError("J must be >= 1, got $J"))

		#	Mark Partially Observed Nodes
			partial_mask = falses(n)
			@inbounds for idx in partially_observed
				if 1 <= idx <= n
					partial_mask[idx] = true
				end
			end
			respondent_indices = findall(.!partial_mask)
			n_respondents = length(respondent_indices)
			n_respondents >= 2 || throw(ArgumentError("must have >= 2 respondents to compute P"))

		#	Build Node ID -> Index Mapping
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))

		#	Allocate Per-Cell-Pair Accumulators
			#	edge_count[deg_src, ei_src, deg_dst, ei_dst]
			#	weight_sum[..] is the sum of weights for edges in that cell pair
				edge_count = zeros(Int, K, J, K, J)
				weight_sum = zeros(Float64, K, J, K, J)

		#	Walk Edges and Accumulate Per-Cell-Pair Counts/Weights
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)
			has_weight_col = hasproperty(edges, :weight)

			@inbounds for k in 1:n_edges
				src_i = get(id_to_idx, src_ids[k], 0)
				dst_i = get(id_to_idx, dst_ids[k], 0)
				if src_i == 0 || dst_i == 0
					continue
				end
				if partial_mask[src_i] || partial_mask[dst_i]
					continue   # exclude any edge touching a partial node
				end

				#	Cell coordinates for source and destination
					ds = degree_bins[src_i]; es = ei_bins[src_i]
					dt = degree_bins[dst_i]; et = ei_bins[dst_i]

				#	Weight for this edge (defaults to 1.0 for unweighted)
					w_edge = (weighted && has_weight_col) ? Float64(edges.weight[k]) : 1.0

				#	For directed: increment the (src, dst) cell pair.
				#	For undirected: increment both (src, dst) and (dst, src)
				#	so that P remains symmetric without extra logic
				#	downstream. The dyad count below handles this consistently.
					edge_count[ds, es, dt, et] += 1
					weight_sum[ds, es, dt, et] += w_edge
					if !directed
						edge_count[dt, et, ds, es] += 1
						weight_sum[dt, et, ds, es] += w_edge
					end
			end

		#	Count Dyads per Cell Pair (Respondent-Respondent Only)
			#	cell_node_count[deg, ei] = number of respondents in that cell
				cell_node_count = zeros(Int, K, J)
				@inbounds for i in respondent_indices
					cell_node_count[degree_bins[i], ei_bins[i]] += 1
				end

		#	Compute Laplace-Smoothed P and Conditional w
			P = zeros(Float64, K, J, K, J)
			w = zeros(Float64, K, J, K, J)
			lap_num = Float64(LAPLACE_NUMERATOR_ADD)
			lap_den = Float64(LAPLACE_DENOMINATOR_ADD)

			@inbounds for ds in 1:K, es in 1:J, dt in 1:K, et in 1:J
				n_src = cell_node_count[ds, es]
				n_dst = cell_node_count[dt, et]

				#	Compute ordered-pair count between cells
					if (ds, es) == (dt, et)
						#	Within-cell: ordered pairs are n*(n-1), undirected
						#	uses the same convention because we double-counted
						#	edges above for undirected
							n_dyads = n_src * (n_src - 1)
					else
						n_dyads = n_src * n_dst
					end

				#	Laplace-smoothed probability
					P[ds, es, dt, et] = (edge_count[ds, es, dt, et] + lap_num) /
										(n_dyads + lap_den)

				#	Conditional mean weight (0 if no edges in this cell pair)
					if edge_count[ds, es, dt, et] > 0
						w[ds, es, dt, et] = weight_sum[ds, es, dt, et] /
											edge_count[ds, es, dt, et]
					end
			end

		#	Return P and w
			return (P = P, w = w)
	end

#	Helper Step 3.5: Compute the reciprocity matrix R (directed only)
	function _compute_r_matrix(edges::DataFrame,
								nodes::DataFrame,
								degree_bins::Vector{Int},
								ei_bins::Vector{Int},
								K::Int,
								J::Int;
								partially_observed::Vector{Int} = Int[])
		"""
		Args:
			edges::DataFrame: directed edge list with :src, :dst (and
				optionally :weight, unused here)
			nodes::DataFrame: node roster with :id
			degree_bins::Vector{Int}: per-node degree bin, 1..K
			ei_bins::Vector{Int}: per-node E/I bin, 1..J
			K::Int: number of degree bins
			J::Int: number of E/I bins (use 1 in degree-only fallback)
			partially_observed::Vector{Int}: indices to exclude (computed
				over respondent-respondent dyads only, per design)
		Returns:
			Array{Float64,4}: R matrix indexed [deg_src, ei_src, deg_dst,
				ei_dst]. R[c, c'] is the Laplace-smoothed P(reverse edge
				j -> i | forward edge i -> j, i in c, j in c').
		Notes:
			Implements Step 3.5 of the design specification. The reciprocity
			matrix is asymmetric in general: R[c, c'] != R[c', c]. For each
			ordered cell pair (c, c'):
			    R[c, c'] = (count of mutual edges from c to c' + LAPLACE_NUM)
			               / (count of forward edges from c to c' + LAPLACE_DEN)
			where a "mutual" forward edge i -> j is one where j -> i also
			exists in the observed edge set.

			This generalizes SMM 2022's single-global-reciprocity-rate
			imputation to be conditional on the cell pair the dyad lives in,
			capturing structural variation in reciprocity behavior (e.g.,
			brokers vs clustered hubs).

			Only called for directed networks; undirected networks should
			pass R = Nothing to SamplerSetup.
		"""

		#	Guards
			n = nrow(nodes)
			length(degree_bins) == n && length(ei_bins) == n ||
				throw(ArgumentError("bin vectors must match nodes count"))

		#	Mark Partially Observed
			partial_mask = falses(n)
			@inbounds for idx in partially_observed
				if 1 <= idx <= n
					partial_mask[idx] = true
				end
			end

		#	Build Node ID -> Index Mapping and Edge Lookup Set
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)

			#	Build a Set of (src_idx, dst_idx) pairs for O(1) lookup of
			#	whether a given directed edge exists.
				edge_set = Set{Tuple{Int,Int}}()
				sizehint!(edge_set, n_edges)
				@inbounds for k in 1:n_edges
					si = get(id_to_idx, src_ids[k], 0)
					di = get(id_to_idx, dst_ids[k], 0)
					if si > 0 && di > 0 && !partial_mask[si] && !partial_mask[di]
						push!(edge_set, (si, di))
					end
				end

		#	Accumulate Per-Cell-Pair Forward and Mutual Counts
			forward_count = zeros(Int, K, J, K, J)
			mutual_count  = zeros(Int, K, J, K, J)

			@inbounds for (si, di) in edge_set
				ds = degree_bins[si]; es = ei_bins[si]
				dt = degree_bins[di]; et = ei_bins[di]
				forward_count[ds, es, dt, et] += 1
				if (di, si) in edge_set
					mutual_count[ds, es, dt, et] += 1
				end
			end

		#	Compute Laplace-Smoothed R
			R = zeros(Float64, K, J, K, J)
			lap_num = Float64(LAPLACE_NUMERATOR_ADD)
			lap_den = Float64(LAPLACE_DENOMINATOR_ADD)

			@inbounds for ds in 1:K, es in 1:J, dt in 1:K, et in 1:J
				R[ds, es, dt, et] = (mutual_count[ds, es, dt, et] + lap_num) /
									(forward_count[ds, es, dt, et] + lap_den)
			end

		#	Return R
			return R
	end

#	Helper Step 4: Determine number of synthetic added nodes N_add
	function _determine_n_add(pi_node::Float64, N::Int, N_nom::Int)
		"""
		Args:
			pi_node::Float64: fraction of nodes believed missing, in [0, 1)
			N::Int: number of observed (respondent) nodes
			N_nom::Int: number of nominated non-respondents
		Returns:
			Int: N_add, the number of additional synthetic non-nominated
				nodes to add per replicate. Always >= 0.
		Notes:
			Implements Step 4 of the design specification:
			    N_add = round(pi_node * (N + N_nom) / (1 - pi_node)) - N_nom
			Edge cases:
			- pi_node = 0: returns 0 (Stage 1 is skipped).
			- N_nom = 0: reduces to round(pi_node * N / (1 - pi_node)).
			- Result is clamped to >= 0; if the computed value is negative
			  (e.g., when N_nom exceeds the implied total missing count),
			  it is treated as zero with a diagnostic note. The framework
			  treats N_nom as the full missing count and adds no synthetic
			  nodes in this case.
		"""

		#	Guards
			0.0 <= pi_node < 1.0 ||
				throw(ArgumentError("pi_node must be in [0, 1), got $pi_node"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_nom >= 0 || throw(ArgumentError("N_nom must be >= 0, got $N_nom"))

		#	Compute N_add per Design Formula
			if pi_node == 0.0
				return 0
			end

			implied_missing = Int(round(pi_node * (N + N_nom) / (1.0 - pi_node)))
			n_add = implied_missing - N_nom

		#	Clamp to Nonnegative
			return max(n_add, 0)
	end

#	Helper Step 5 Internal: Compute Realized Correlation for a Given Beta
	function _realized_rho_for_beta(beta::Float64, K::Int, N::Int, N_add::Int)
		"""
		Args:
			beta::Float64: candidate logistic skew parameter
			K::Int: number of degree bins
			N::Int: number of observed respondent nodes
			N_add::Int: number of added synthetic nodes
		Returns:
			Float64: analytic Kendall tau-b between bin index (1..K) and
				missingness indicator (0/1) under the q = softmax(beta * b_tilde)
				distribution for the added nodes. For beta = 0 returns
				exactly 0.0; for large |beta| approaches a K-dependent
				ceiling below the rate-bounded absolute (see Notes).
		Notes:
			This is the function being bisected in _solve_bin_distribution.

			Refactored from Pearson to Kendall's tau-b. The user-facing
			parameter rho still lies in (-1, 1) but is internally
			interpreted as Kendall tau-b. This matches Phase 1's metric
			(which is also Kendall tau-b after the refactor) and the
			user's rank-order intuition about missingness.

			Setup (unchanged from the Pearson version):
			- Observed nodes uniformly distributed across K rank-equal
			  bins: N/K nodes per bin.
			- Added nodes distributed by q[k] = softmax(beta * b_tilde)[k]:
			  N_add * q[k] nodes per bin.
			- x = bin index (1..K, ordinal); y = is_added (0/1).

			Kendall tau-b is closed-form on this discrete bin x binary
			structure. Only cross pairs (one observed, one added)
			contribute to concordance/discordance; within-observed and
			within-added pairs are tied on y and contribute to the
			denominator only.

			Cross-pair counts:
			- C (concordant: observed in lower bin, added in higher) =
			  (N/K) * N_add * sum_k q[k] * (k - 1)
			- D (discordant) = (N/K) * N_add * sum_k q[k] * (K - k)
			- Tx (cross tied on x = same bin) = N * N_add / K
			- C + D + Tx = N * N_add (total cross pairs, sanity check)

			Within-pair counts (tied on y, distinguishable on x):
			- Within-observed cross-bin (uniform observed):
			  C(N, 2) - K * C(N/K, 2) = N^2 * (K-1) / (2K)
			- Within-added cross-bin:
			  C(N_add, 2) - sum_k C(N_add*q[k], 2)
			  = N_add^2 / 2 * (1 - sum_k q[k]^2)
			- Ty = within_obs_cross_bin + within_add_cross_bin

			Final formula:
			tau-b = (C - D) / sqrt((C + D + Tx)(C + D + Ty))

			K-DEPENDENT CEILING. The maximum achievable |tau-b| at fixed
			N, N_add depends on K. Higher K reduces Tx (fewer ties on
			the bin side), bringing the ceiling closer to the rate-bounded
			absolute sqrt(2*p*(1-p)). Lower K increases Tx and lowers
			the ceiling. This is the analytic explanation for why
			Test 20b at K=10 produces fewer :ceiling_hit than K=4 on
			the same network.

			Monotonicity in beta: q skews toward higher bins as beta
			increases. C grows, D shrinks. Tau-b increases monotonically
			in beta. The bisection in _solve_bin_distribution can use
			this without modification.
		"""

		#	Guards
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_add >= 0 || throw(ArgumentError("N_add must be >= 0, got $N_add"))

		#	N_add = 0: No Missingness Variation, Correlation Undefined
			if N_add == 0
				return 0.0
			end

		#	Compute q = softmax(beta * b_tilde)
			#	b_tilde standardized: b_tilde[k] = (k - (K+1)/2) / sd(1..K)
				bins = collect(1.0:Float64(K))
				bin_mean = (K + 1) / 2
				bin_sd = sqrt(sum((bins .- bin_mean).^2) / K)
				b_tilde = (bins .- bin_mean) ./ bin_sd

				#	Stable softmax: subtract max before exponentiating
					raw = beta .* b_tilde
					raw .-= maximum(raw)
					q = exp.(raw)
					q ./= sum(q)

		#	Cross-Pair Counts: C, D, Tx
			n_obs_per_bin = N / K
			C  = 0.0
			D  = 0.0
			@inbounds for k in 1:K
				C += n_obs_per_bin * N_add * q[k] * (k - 1)
				D += n_obs_per_bin * N_add * q[k] * (K - k)
			end
			Tx = N * N_add / K

			#	Sanity: C + D + Tx should equal N * N_add exactly.
			#	Floating-point error makes this approximate; not asserted.

		#	Within-Pair Counts: Ty (tied on y but not x)
			#	Within-observed cross-bin = N^2 * (K-1) / (2K)
				Ty_obs = (N * N * (K - 1)) / (2 * K)

			#	Within-added cross-bin = N_add^2 / 2 * (1 - sum_k q[k]^2)
				sum_q2 = 0.0
				@inbounds for k in 1:K
					sum_q2 += q[k] * q[k]
				end
				Ty_add = (N_add * N_add) / 2 * (1.0 - sum_q2)

			Ty = Ty_obs + Ty_add

		#	Compute tau-b
			numer = C - D
			denom_a = C + D + Tx
			denom_b = C + D + Ty

			if denom_a <= 0.0 || denom_b <= 0.0
				return 0.0
			end

			return numer / sqrt(denom_a * denom_b)
	end

#	Helper Step 5: Numerical Bisection on Beta to Achieve Target Correlation
	function _solve_bin_distribution(rho_target::Real,
									  K::Int,
									  N::Int,
									  N_add::Int;
									  beta_bounds::Tuple{Float64,Float64} = BETA_BISECTION_BOUNDS,
									  tol::Float64 = BETA_BISECTION_TOL,
									  max_iters::Int = BETA_BISECTION_MAX_ITERS)
		"""
		Args:
			rho_target::Real: target centrality-missingness correlation, in
				(-1, 1)
			K::Int: number of degree bins
			N::Int: number of observed nodes
			N_add::Int: number of added nodes
			beta_bounds::Tuple{Float64,Float64}: bisection interval for
				beta (default = BETA_BISECTION_BOUNDS = (-50.0, 50.0))
			tol::Float64: tolerance on realized correlation (default = 1e-4)
			max_iters::Int: max bisection iterations (default = 100)
		Returns:
			NamedTuple: (beta, q, status, n_iters)
				beta::Float64: solved logistic-skew parameter
				q::Vector{Float64}: degree-bin assignment distribution,
					length K, sums to 1
				status::Symbol: :converged, :ceiling_hit, or :failed_other.
					Mirrors Phase 1's bisection-status semantics.
				n_iters::Int: number of bisection iterations performed
		Notes:
			Implements Step 5 of the design specification. The function
			being bisected is _realized_rho_for_beta(beta, K, N, N_add),
			which is closed-form and deterministic in beta. The bisection
			pattern mirrors Phase 1's _bisect_b_for_target_rho but with no
			Monte Carlo inner samples, so the tolerance can be tight.

			Status semantics:
			- :converged when bisection converges to within tol of
			  rho_target before max_iters.
			- :ceiling_hit when beta at one of the interval endpoints
			  produces realized correlation that does not bracket
			  rho_target (i.e., the target is unreachable on this
			  K, N, N_add combination); returns the closest beta and
			  the corresponding q.
			- :failed_other when the bisection fails for an unanticipated
			  reason (e.g., N_add = 0 with nonzero rho_target). Returns
			  beta = 0, uniform q.

			Edge cases:
			- rho_target = 0: returns beta = 0, q uniform, status =
			  :converged in zero iterations.
			- N_add = 0: realized correlation is undefined (no missingness
			  variation). If rho_target = 0, returns beta = 0 q uniform,
			  status = :converged. If rho_target != 0, returns status =
			  :failed_other with beta = 0 q uniform.
			- |rho_target| > ceiling: returns status = :ceiling_hit with
			  beta at the endpoint that produces the largest |realized rho|
			  in the target direction.
		"""

		#	Guards
			-1.0 < rho_target < 1.0 ||
				throw(ArgumentError("rho_target must be in (-1, 1), got $rho_target"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_add >= 0 || throw(ArgumentError("N_add must be >= 0, got $N_add"))

		#	Uniform q for Trivial Cases
			uniform_q = fill(1.0 / K, K)

		#	N_add = 0: No Missingness Variation
			if N_add == 0
				status = isapprox(rho_target, 0.0; atol = tol) ? :converged : :failed_other
				return (beta = 0.0, q = uniform_q, status = status, n_iters = 0)
			end

		#	rho_target = 0: Beta = 0, q Uniform
			if isapprox(rho_target, 0.0; atol = tol)
				return (beta = 0.0, q = uniform_q, status = :converged, n_iters = 0)
			end

		#	Determine Sign of Beta from Sign of rho_target
			#	Positive rho_target => positive beta (q skews high)
			#	Negative rho_target => negative beta (q skews low)
				beta_lo, beta_hi = beta_bounds
				rho_at_lo = _realized_rho_for_beta(beta_lo, K, N, N_add)
				rho_at_hi = _realized_rho_for_beta(beta_hi, K, N, N_add)

		#	Ceiling Check: Does rho_target Fall in [rho_at_lo, rho_at_hi]?
			if rho_target > rho_at_hi || rho_target < rho_at_lo
				#	Target is outside the achievable range. Return the
				#	endpoint closest to the target.
					if abs(rho_target - rho_at_hi) <= abs(rho_target - rho_at_lo)
						beta_ceil = beta_hi
					else
						beta_ceil = beta_lo
					end
					q_ceil = _q_from_beta(beta_ceil, K)
					return (beta = beta_ceil, q = q_ceil,
							status = :ceiling_hit, n_iters = 0)
			end

		#	Bisection on Beta
			beta_mid = 0.0
			rho_mid = 0.0
			iter = 0
			while iter < max_iters
				iter += 1
				beta_mid = (beta_lo + beta_hi) / 2
				rho_mid = _realized_rho_for_beta(beta_mid, K, N, N_add)

				if abs(rho_mid - rho_target) <= tol
					q_mid = _q_from_beta(beta_mid, K)
					return (beta = beta_mid, q = q_mid,
							status = :converged, n_iters = iter)
				end

				#	Tighten the interval based on which side rho_mid is on.
				#	rho is monotone increasing in beta (q skews more central
				#	as beta increases), so the standard bracketing works.
					if rho_mid < rho_target
						beta_lo = beta_mid
					else
						beta_hi = beta_mid
					end
			end

		#	Did Not Converge Within max_iters: Return Best Estimate
			q_mid = _q_from_beta(beta_mid, K)
			return (beta = beta_mid, q = q_mid,
					status = :failed_other, n_iters = iter)
	end

#	Helper: Compute q Distribution from a Given Beta
	function _q_from_beta(beta::Float64, K::Int)
		"""
		Args:
			beta::Float64: logistic skew parameter
			K::Int: number of degree bins
		Returns:
			Vector{Float64}: softmax distribution over bins 1..K,
				length K, sums to 1.
		Notes:
			Helper for _solve_bin_distribution. Uses a numerically stable
			softmax (subtracts max before exp).
		"""
		bins = collect(1.0:Float64(K))
		bin_mean = (K + 1) / 2
		bin_sd = sqrt(sum((bins .- bin_mean).^2) / K)
		b_tilde = (bins .- bin_mean) ./ bin_sd
		raw = beta .* b_tilde
		raw .-= maximum(raw)
		q = exp.(raw)
		q ./= sum(q)
		return q
	end

#	Helper: Compute Empirical Conditional Distribution P(EI_bin | degree_bin)
	function _compute_ei_conditional(degree_bins::Vector{Int},
									  ei_bins::Vector{Int},
									  K::Int,
									  J::Int)
		"""
		Args:
			degree_bins::Vector{Int}: per-node degree bin, 1..K
			ei_bins::Vector{Int}: per-node E/I bin, 1..J
			K::Int: number of degree bins
			J::Int: number of E/I bins
		Returns:
			Matrix{Float64}: size K x J, row-stochastic. Row k is the
				empirical distribution of E/I bin assignment among
				observed nodes in degree bin k.
		Notes:
			Used in Step 5 of the design spec to sample E/I bin
			assignments for added nodes conditional on their drawn
			degree bin. Each row is normalized to sum to 1; rows where
			no observed node falls in that degree bin (shouldn't happen
			with rank-equal binning, but guarded against) default to
			uniform across E/I bins.
		"""

		#	Allocate Counts Matrix
			counts = zeros(Int, K, J)
			@inbounds for i in eachindex(degree_bins)
				counts[degree_bins[i], ei_bins[i]] += 1
			end

		#	Normalize to Row-Stochastic
			cond = zeros(Float64, K, J)
			@inbounds for k in 1:K
				row_total = sum(view(counts, k, :))
				if row_total == 0
					#	Degree bin has no observed nodes; default to uniform
						cond[k, :] .= 1.0 / J
				else
					for j in 1:J
						cond[k, j] = counts[k, j] / row_total
					end
				end
			end

		#	Return Conditional Distribution
			return cond
	end

#####################################################
#       SECTION 4: PHASE 2 SETUP USER-FACING        #
#####################################################

#	feasible_rho_range: Compute Network's Feasible ρ Range at a Given Rate
	function feasible_rho_range(edges::DataFrame,
								  nodes::DataFrame;
								  directed::Bool,
								  weighted::Bool,
								  target_rate::Float64,
								  n_mc_replicates::Int = 20,
								  extreme_beta::Float64 = 5.0,
								  seed::Integer = 1)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optional :weight
			nodes::DataFrame: node roster with :id and :label columns
			directed::Bool: whether the network is directed
			weighted::Bool: whether the network has meaningful edge weights
			target_rate::Float64: missingness rate at which feasibility is
				being assessed, in (0, 1)
			n_mc_replicates::Int: number of Monte Carlo replicates per
				endpoint probe (default 20). Higher gives tighter
				feasibility bounds.
			extreme_beta::Float64: magnitude of beta at the saturation
				probes (default 5.0). Larger values are nearly
				indistinguishable in their realized rho but cost more
				computation.
			seed::Integer: master seed; per-replicate seeds derived via
				hash.
		Returns:
			NamedTuple with fields:
				rho_min::Float64           Minimum realized Kendall tau-b
				                            (at b = -extreme_beta)
				rho_max::Float64           Maximum realized Kendall tau-b
				                            (at b = +extreme_beta)
				rho_mcar::Float64          Realized rho at b = 0 (~ 0)
				rho_min_std::Float64       MC std error of rho_min
				rho_max_std::Float64       MC std error of rho_max
				rho_mcar_std::Float64      MC std error of rho_mcar
				target_rate::Float64       Echoed for caller's record
				is_asymmetric::Bool        True if ratio |rho_min|/|rho_max|
				                            is < 0.5 or > 2.0; rarely fires
				                            under Kendall (see Notes)
				asymmetry_ratio::Float64   |rho_min| / |rho_max|, or 1.0
				                            if both ≈ 0
				diagnostics::Dict{Symbol,Any}  Metadata: extreme_beta_used,
				                                n_mc_replicates_used, seed,
				                                centrality_summary
		Notes:
			Compute the feasible range of centrality-missingness correlation
			(rho, measured as Kendall tau-b) for this network at the
			specified missingness rate. The feasibility range characterizes
			what the network's centrality distribution can structurally
			support; it does NOT depend on any user-specified target rho.

			KENDALL TAU-B AS THE METRIC. The framework's correlation
			parameter rho is interpreted internally as Kendall's tau-b,
			the rank-based correlation between the missingness indicator
			and node centrality. This matches the user's rank-order
			intuition about missingness ("low-ranked nodes are likely
			to be missing") and avoids the asymmetric-ceiling problem
			that Pearson correlation exhibits on heavy-tailed networks.

			The function probes Phase 1's sampling mechanism at three
			beta values:
			- beta = -extreme_beta (saturated toward low-rank nodes):
			  gives rho_min, the most negative achievable Kendall tau-b
			- beta = 0 (uniform sampling, MCAR baseline): gives rho_mcar,
			  which should be ~ 0 with std reflecting noise floor
			- beta = +extreme_beta (saturated toward high-rank nodes):
			  gives rho_max, the most positive achievable Kendall tau-b

			Each beta value is sampled n_mc_replicates times to produce
			a mean and std error estimate of the realized rho at that
			beta. The "feasibility ceiling" is then approximately rho_min
			and rho_max, with the std telling the caller how much MC
			noise is around those estimates.

			TWO CEILING CONSTRAINTS. Under Kendall tau-b on a binary
			indicator (missingness) and a continuous variable (centrality),
			realized rho is structurally bounded:
			- Absolute (rate-bounded) ceiling: |tau-b| <= sqrt(2 * p * (1 - p))
			  where p is the missingness rate. At rate = 0.10, this gives
			  |tau-b|_max ~ 0.42. The bound depends only on the rate, not
			  on network structure.
			- Practical (tie-bounded) ceiling: lower than the absolute on
			  networks with centrality ties (most unweighted networks).
			  Centrality ties contribute to the denominator of tau-b,
			  reducing the maximum achievable correlation. This ceiling
			  is what feasible_rho_range actually measures.

			SYMMETRIC CEILINGS UNDER KENDALL. Under Kendall tau-b with
			rank-based sampling weights, ceilings are typically symmetric
			between negative and positive sides. This is a substantive
			improvement over Pearson, where heavy-tailed networks (Marvel,
			Toledo) exhibited dramatically asymmetric ceilings (e.g.,
			rho_max ~ +0.4 but rho_min ~ -0.03 under Pearson). The
			is_asymmetric flag is retained as a diagnostic for unusual
			cases but rarely fires under Kendall.

			RECOMMENDED USE. Called by reconstruct_network() at startup
			to check whether the user's specified rho is structurally
			supportable. If it falls outside [rho_min, rho_max], emit
			a warning explaining the feasibility constraint in plain
			language.

			COMPUTATIONAL COST. With n_mc_replicates = 20 and three beta
			probes, the function runs 60 Phase 1 samples. On Moreno
			(N = 70) this takes < 1 second. On Marvel (N = 6486) it
			takes roughly 5-10 seconds. The function is intended to be
			called once per network at framework startup, not repeatedly.

			DETERMINISM. Per-replicate seeds are derived via
			hash((beta, rep, seed)) so the same network + parameters
			reproduces identical bounds. The seed argument controls the
			master seed for this derivation.
		"""

		#	Guards
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))
			0.0 < target_rate < 1.0 ||
				throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			n_mc_replicates >= 1 ||
				throw(ArgumentError("n_mc_replicates must be >= 1, got $n_mc_replicates"))
			extreme_beta > 0 ||
				throw(ArgumentError("extreme_beta must be > 0, got $extreme_beta"))

		#	Cache Centrality Once
			centrality = _centrality_for_sampler(edges;
												  nodes = nodes,
												  directed = directed)

		#	Probe at Three Beta Values
			beta_values = [-extreme_beta, 0.0, +extreme_beta]
			beta_labels = [:rho_min, :rho_mcar, :rho_max]
			rho_estimates = Dict{Symbol, Vector{Float64}}()

			for (b, lab) in zip(beta_values, beta_labels)
				rho_per_rep = Float64[]
				for rep in 1:n_mc_replicates
					#	Derive per-replicate seed deterministically
						rep_seed = Int(hash((b, rep, seed)) % UInt32)

					#	Sample dropped-node set at this fixed beta. Uses
					#	the same internal helper Phase 1's bisection
					#	uses; we just hold beta fixed instead of
					#	bisecting.
						record = generate_missingness_mask(
							edges;
							nodes       = nodes,
							directed    = directed,
							target_rate = target_rate,
							fixed_beta  = b,    # bypass bisection at this beta
							seed        = rep_seed,
							centrality  = centrality)

					#	Skip failed_other replicates (rare; usually only
					#	when target_rate is unreachable on the network)
						if record.bisection_status == :failed_other
							continue
						end

					push!(rho_per_rep, record.realized_rho)
				end

				rho_estimates[lab] = rho_per_rep
			end

		#	Aggregate Per-Endpoint Mean and Std
			rho_min_mean  = mean(rho_estimates[:rho_min])
			rho_max_mean  = mean(rho_estimates[:rho_max])
			rho_mcar_mean = mean(rho_estimates[:rho_mcar])

			rho_min_std  = length(rho_estimates[:rho_min]) >= 2 ?
				std(rho_estimates[:rho_min]) : NaN
			rho_max_std  = length(rho_estimates[:rho_max]) >= 2 ?
				std(rho_estimates[:rho_max]) : NaN
			rho_mcar_std = length(rho_estimates[:rho_mcar]) >= 2 ?
				std(rho_estimates[:rho_mcar]) : NaN

		#	Compute Asymmetry Metric
			#	If both rho_min and rho_max are essentially zero (a
			#	degenerate case where centrality is constant), set ratio
			#	to 1.0 and is_asymmetric to false.
				is_degenerate = abs(rho_min_mean) < 0.01 && abs(rho_max_mean) < 0.01

				asymmetry_ratio = if is_degenerate
					1.0
				else
					#	Ratio of |rho_min| to |rho_max|. Always in (0, Inf).
					#	Symmetric ceiling => ratio ~= 1.0.
						denom = abs(rho_max_mean)
						denom > 0 ? abs(rho_min_mean) / denom : Inf
				end

				#	is_asymmetric flags strong asymmetry in either direction
					is_asymmetric = !is_degenerate &&
									(asymmetry_ratio < 0.5 || asymmetry_ratio > 2.0)

		#	Centrality Summary for Diagnostics
			centrality_summary = (
				min  = minimum(centrality),
				max  = maximum(centrality),
				mean = mean(centrality),
				median = median(centrality),
				std  = std(centrality),
				skewness_proxy = (mean(centrality) - median(centrality)) / std(centrality),
			)

		#	Diagnostics Dict
			diagnostics = Dict{Symbol, Any}(
				:extreme_beta_used      => extreme_beta,
				:n_mc_replicates_used   => n_mc_replicates,
				:seed                   => seed,
				:centrality_summary     => centrality_summary,
				:n_completed_rho_min    => length(rho_estimates[:rho_min]),
				:n_completed_rho_mcar   => length(rho_estimates[:rho_mcar]),
				:n_completed_rho_max    => length(rho_estimates[:rho_max]),
			)

		#	Return Feasibility NamedTuple
			return (
				rho_min          = rho_min_mean,
				rho_max          = rho_max_mean,
				rho_mcar         = rho_mcar_mean,
				rho_min_std      = rho_min_std,
				rho_max_std      = rho_max_std,
				rho_mcar_std     = rho_mcar_std,
				target_rate      = target_rate,
				is_asymmetric    = is_asymmetric,
				asymmetry_ratio  = asymmetry_ratio,
				diagnostics      = diagnostics,
			)
	end
	@doc raw"""
	**Description**
	Compute the feasible range of centrality-missingness correlation $\rho$ (Kendall's $\tau_b$) for a network at a given missingness rate. The function probes Phase 1's sampling mechanism at three extreme values of the indicator-probability skew $\beta$ to estimate the negative ceiling, MCAR baseline, and positive ceiling that the network's centrality distribution can structurally support.

	**Usage**
	`feasible_rho_range(edges, nodes; directed, weighted, target_rate, n_mc_replicates=20, extreme_beta=5.0, seed=1)`

	**Arguments**
	- `edges::DataFrame`: Edge list with `:src`, `:dst`, optional `:weight`.
	- `nodes::DataFrame`: Node roster with `:id` and `:label` columns.
	- `directed::Bool`: Whether the network is directed.
	- `weighted::Bool`: Whether the network has meaningful edge weights.
	- `target_rate::Float64`: Missingness rate at which feasibility is assessed, in `(0, 1)`.
	- `n_mc_replicates::Int`: Number of Monte Carlo replicates per endpoint probe (default `20`).
	- `extreme_beta::Float64`: Magnitude of $\beta$ at saturation probes (default `5.0`).
	- `seed::Integer`: Master seed for reproducibility (default `1`).

	**Details**

	The framework treats $\rho$ as Kendall's $\tau_b$, the rank-based correlation between the missingness indicator (binary) and node centrality (continuous). This matches the user's rank-order intuition ("low-ranked nodes are likely to be missing") and avoids the asymmetric-ceiling problem that Pearson correlation exhibits on heavy-tailed networks.

	Phase 1's sampler weights node inclusion probability by a mixture of rank-based centrality and uniform noise: $\text{prob}_i = b \cdot r_i + (1 - b) \cdot u_i$, where $r_i$ is node $i$'s centrality rank scaled to $[0, 1]$ and $u_i \sim \text{Uniform}(0, 1)$. At $b = 0$ this is MCAR; at $|b| \to 1$ the sampling saturates toward high-rank ($b > 0$) or low-rank ($b < 0$) nodes.

	**Two ceiling constraints** apply under Kendall $\tau_b$ on a binary indicator:

	1. **Rate-bounded absolute ceiling**: $|\tau_b|_{\max} = \sqrt{2 \cdot p \cdot (1 - p)}$ where $p$ is the missingness rate. This depends only on $p$, not on network structure. At $p = 0.10$, the absolute ceiling is approximately $0.42$.

	2. **Practical tie-bounded ceiling**: on networks with centrality ties (most unweighted networks), the practical ceiling is lower than the absolute. Centrality ties contribute to the denominator of $\tau_b$, reducing the achievable correlation. This is the ceiling `feasible_rho_range` actually measures.

	Under Kendall with rank-based sampling weights, ceilings are typically symmetric between negative and positive sides. This is a substantive improvement over Pearson, where heavy-tailed networks exhibited dramatic asymmetry. For example, the Phase 1 corpus shows realized $\tau_b$ on Marvel at nominal $\rho = \pm 0.75$ produces $-0.236$ and $+0.230$ respectively — symmetric in magnitude.

	The `is_asymmetric` flag fires when $|\rho_{\min}|/|\rho_{\max}|$ falls outside $[0.5, 2.0]$, which under Kendall is rare. The flag is retained as a diagnostic for unusual cases (e.g., very small networks with extreme degree concentration).

	**Value**

	NamedTuple with fields documented in the in-body docstring. Key fields for caller use:
	- `rho_min`, `rho_max`: Estimated practical (tie-bounded) feasibility ceilings under Kendall $\tau_b$
	- `rho_min_std`, `rho_max_std`: Monte Carlo std errors around the ceiling estimates
	- `is_asymmetric`: Diagnostic flag for unusual ceiling asymmetry; rare under Kendall

	**Intended Use**

	Called by `reconstruct_network()` at framework startup to check whether the user's specified $\rho$ falls within the network's feasible range. If outside, the wrapper emits a plain-language warning explaining the structural constraint and prompts the user to confirm, adjust, or cancel.

	**See Also**
	`reconstruct_network`, `generate_missingness_mask`, `build_degeneration_corpus`

	**References**
	Network_Confidence_Intervals.tex, Phase 1 sampler.
	""" feasible_rho_range

#	find_optimal_K: Find the largest K satisfying all three priors
	function find_optimal_K(edges::DataFrame,
							 nodes::DataFrame,
							 community_labels::Vector{Int};
							 directed::Bool,
							 weighted::Bool,
							 pi_node::Float64,
							 rho::Float64,
							 partially_observed_nodes::Vector{Int} = Int[],
							 K_max::Int = 0,
							 K_min::Int = 4,
							 J::Int = DEFAULT_J,
							 min_nodes_per_ei_bin::Int = 3,
							 min_nodes_per_degree_bin::Int = 5,
							 feasibility_n_mc::Int = 20,
							 feasibility_extreme_beta::Float64 = 5.0,
							 feasibility_seed::Integer = 1,
							 verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: standard compute_setup inputs
			directed::Bool: whether the network is directed
			weighted::Bool: whether the network has meaningful edge weights
			pi_node::Float64: user's proportion-missing prior
			rho::Float64: user's centrality-missingness correlation prior
			partially_observed_nodes::Vector{Int}: indices of nominated
				non-respondents
			K_max::Int: maximum K to consider (default 0 = auto:
				min(20, floor(N_obs / min_nodes_per_degree_bin)))
			K_min::Int: minimum K to consider (default 4)
			J::Int: number of E/I bins (default = DEFAULT_J = 3)
			min_nodes_per_ei_bin::Int: minimum nodes per E/I bin to keep
				2D binning active (default 3, matches design spec)
			min_nodes_per_degree_bin::Int: floor on nodes per degree bin;
				sets the effective ceiling on K_max (default 5)
			feasibility_n_mc::Int: MC replicates for feasibility precondition
				(default 20)
			feasibility_extreme_beta::Float64: extreme-beta probe for
				feasibility (default 5.0)
			feasibility_seed::Integer: seed for feasibility MC (default 1)
			verbose::Bool: print per-K diagnostics during search (default false)
		Returns:
			NamedTuple with fields:
				optimal_K::Int            largest K satisfying all priors,
				                          or -1 if no valid K found
				test_results::Vector      per-K diagnostics (descending K order)
				stopping_reason::Symbol   :found | :rho_infeasible |
				                          :no_valid_K | :K_min_reached
				feasibility::NamedTuple   result of feasible_rho_range probe;
				                          included so caller can surface
				                          feasibility info regardless of
				                          whether K-search ran
		Notes:
			Determines the largest K such that compute_setup on this network
			with the user's specified pi_node and rho satisfies all three
			priors:
			- Prior 1 (proportion missing): trivially satisfied; included
			  for completeness
			- Prior 2 (centrality correlation): beta_status ∈ {:converged,
			  :ceiling_hit}. The :ceiling_hit case is acceptable here
			  because the framework's flag is the contract being honored;
			  :failed_other is a real failure.
			- Prior 3 (E/I given degree): ei_conditional is row-stochastic
			  AND matches the empirical conditional from the binning
			  (with matching empty-bin uniform fallback)

			FEASIBILITY PRECONDITION. Before searching K, the function
			verifies the user's rho is structurally achievable for this
			network at this rate via feasible_rho_range(). If rho falls
			outside [rho_min, rho_max], the function returns immediately
			with stopping_reason = :rho_infeasible. The K question doesn't
			make sense if the prior is wrong: no K can correct a
			structurally infeasible target.

			K_max defaults: when K_max is 0 (auto), the upper bound is
			min(20, floor(N_obs / min_nodes_per_degree_bin)). This
			prevents probing K values that would have fewer than
			min_nodes_per_degree_bin nodes per degree bin.

			SEARCH STRATEGY. Iterates K from K_max down to K_min, returning
			the FIRST K that satisfies all three priors. Since checking is
			deterministic given the inputs, no MC replication is needed
			here (the feasibility check above handled the MC component).

			Computational cost: K_max - K_min + 1 compute_setup calls
			plus 60 Phase 1 samples for feasibility. Each compute_setup
			call is milliseconds for small networks, seconds for Marvel-
			sized.

			EMPTY-BIN HANDLING. Matches _compute_ei_conditional's empty-
			bin fallback (uniform). This was the source of Test 20b's
			initial Prior 3 failure at K=10: the test's empirical
			conditional left empty-bin rows as zeros while the framework
			set them to uniform. The fix is internalized here.
		"""

		#	Guards
			K_min >= 2 || throw(ArgumentError("K_min must be >= 2, got $K_min"))
			K_max >= 0 || throw(ArgumentError("K_max must be >= 0, got $K_max"))
			0.0 <= pi_node < 1.0 ||
				throw(ArgumentError("pi_node must be in [0, 1), got $pi_node"))
			-1.0 < rho < 1.0 ||
				throw(ArgumentError("rho must be in (-1, 1), got $rho"))
			length(community_labels) == nrow(nodes) ||
				throw(ArgumentError("community_labels length must equal nrow(nodes)"))

		#	Step 1: Feasibility Precondition
			feas = Network_Credible_Intervals.network_degeneracy.feasible_rho_range(
				edges, nodes;
				directed = directed,
				weighted = weighted,
				target_rate = pi_node,
				n_mc_replicates = feasibility_n_mc,
				extreme_beta = feasibility_extreme_beta,
				seed = feasibility_seed)

			if rho < feas.rho_min || rho > feas.rho_max
				if verbose
					println("find_optimal_K: rho = $rho is outside feasibility range " *
							"[$(round(feas.rho_min, digits=4)), $(round(feas.rho_max, digits=4))]")
					println("  Skipping K search; no K can correct an infeasible prior.")
				end

				return (
					optimal_K       = -1,
					test_results    = NamedTuple[],
					stopping_reason = :rho_infeasible,
					feasibility     = feas,
				)
			end

		#	Step 2: Compute N_obs and resolve K_max
			N_obs = nrow(nodes) - length(partially_observed_nodes)
			effective_K_max = K_max == 0 ?
				min(20, fld(N_obs, min_nodes_per_degree_bin)) : K_max

			if effective_K_max < K_min
				return (
					optimal_K       = -1,
					test_results    = NamedTuple[],
					stopping_reason = :no_valid_K,
					feasibility     = feas,
				)
			end

			if verbose
				println("find_optimal_K: searching K from $effective_K_max down to $K_min")
				println("  N_obs = $N_obs, min_nodes_per_degree_bin = $min_nodes_per_degree_bin")
				println("  Feasibility range: [$(round(feas.rho_min, digits=4)), $(round(feas.rho_max, digits=4))]")
				println("  User rho = $rho is feasible; proceeding with K search.")
			end

		#	Step 3: Search descending
			test_results = NamedTuple[]
			optimal_K = -1
			stopping_reason = :no_valid_K

			for K_candidate in effective_K_max:-1:K_min
				if verbose
					print("  Testing K = $K_candidate ... ")
				end

				#	Call compute_setup at this K (handle exceptions)
					setup = nothing
					exception_text = ""
					try
						setup = compute_setup(edges, nodes, community_labels;
											   directed = directed,
											   weighted = weighted,
											   pi_node = pi_node,
											   rho = rho,
											   partially_observed_nodes = partially_observed_nodes,
											   K = K_candidate,
											   J = J,
											   min_nodes_per_ei_bin = min_nodes_per_ei_bin)
					catch e
						exception_text = "$e"
					end

					if setup === nothing
						push!(test_results, (
							K               = K_candidate,
							prior_1_ok      = false,
							prior_2_ok      = false,
							prior_3_ok      = false,
							exception       = exception_text,
							all_pass        = false,
						))
						if verbose
							println("FAIL (exception: $exception_text)")
						end
						continue
					end

				#	Verify the three priors

					#	Prior 1: arithmetic identity check
						N_add = setup.N_add
						observed_rate = N_add / (N_obs + N_add)
						prior_1_ok = abs(observed_rate - pi_node) < 0.02

					#	Prior 2: beta_status must be :converged or :ceiling_hit
						prior_2_ok = setup.beta_status in (:converged, :ceiling_hit)

					#	Prior 3: ei_conditional row-stochastic and matches
					#	empirical conditional from setup.degree_bins, setup.ei_bins
					#	(with empty-bin uniform fallback matching the framework)
						ei_cond = setup.ei_conditional
						J_eff = size(ei_cond, 2)
						row_sums = sum(ei_cond, dims = 2)
						ei_row_stochastic = all(isapprox.(row_sums, 1.0; atol = 1e-10))

						empirical_cond = zeros(Float64, K_candidate, J_eff)
						for i in 1:length(setup.degree_bins)
							b = setup.degree_bins[i]
							j = setup.ei_bins[i]
							empirical_cond[b, j] += 1.0
						end
						for b in 1:K_candidate
							rs = sum(empirical_cond[b, :])
							if rs > 0
								empirical_cond[b, :] ./= rs
							else
								empirical_cond[b, :] .= 1.0 / J_eff
							end
						end
						ei_matches = isapprox(ei_cond, empirical_cond; atol = 1e-10)
						prior_3_ok = ei_row_stochastic && ei_matches

					all_pass = prior_1_ok && prior_2_ok && prior_3_ok

					push!(test_results, (
						K               = K_candidate,
						prior_1_ok      = prior_1_ok,
						prior_2_ok      = prior_2_ok,
						prior_3_ok      = prior_3_ok,
						exception       = "",
						all_pass        = all_pass,
					))

					if verbose
						status = all_pass ? "PASS" : "FAIL"
						println("$status (P1=$(prior_1_ok), P2=$(prior_2_ok), P3=$(prior_3_ok))")
					end

					if all_pass
						optimal_K = K_candidate
						stopping_reason = :found
						break
					end
			end

		#	Step 4: Determine final stopping reason
			if optimal_K == -1 && stopping_reason == :no_valid_K
				#	Search exhausted without finding a valid K
				#	Distinguish "reached K_min without success" from "no K worked"
					if !isempty(test_results) && test_results[end].K == K_min
						stopping_reason = :K_min_reached
					end
			end

		return (
			optimal_K       = optimal_K,
			test_results    = test_results,
			stopping_reason = stopping_reason,
			feasibility     = feas,
		)
	end
	@doc raw"""
	**Description**
	Find the largest $K$ (number of degree bins) such that `compute_setup` honors all three priors on the given network, the user's specified $\rho$, and the user's specified `pi_node`. Returns the optimal $K$ along with per-$K$ diagnostics and a feasibility precondition check.

	**Usage**
	`find_optimal_K(edges, nodes, community_labels; directed, weighted, pi_node, rho, partially_observed_nodes=[], K_max=0, K_min=4, J=3, ...)`

	**Arguments**
	- `edges::DataFrame`, `nodes::DataFrame`, `community_labels::Vector{Int}`: Standard `compute_setup` inputs.
	- `directed::Bool`, `weighted::Bool`: Network type.
	- `pi_node::Float64`: User's proportion-missing prior.
	- `rho::Float64`: User's centrality-missingness correlation prior.
	- `partially_observed_nodes::Vector{Int}`: Indices of nominated non-respondents (default `[]`).
	- `K_max::Int`: Maximum $K$ to consider. Default `0` triggers auto: $\min(20, \lfloor N_{\text{obs}} / \text{min\_nodes\_per\_degree\_bin} \rfloor)$.
	- `K_min::Int`: Minimum $K$ to consider (default `4`).
	- `J::Int`: Number of E/I bins (default `3`).
	- `min_nodes_per_ei_bin::Int`: Lower bound for E/I bin (default `3`).
	- `min_nodes_per_degree_bin::Int`: Lower bound for degree bin (default `5`).
	- `feasibility_n_mc`, `feasibility_extreme_beta`, `feasibility_seed`: Configuration for the feasibility precondition check.
	- `verbose::Bool`: Print per-$K$ diagnostics during search (default `false`).

	**Details**

	The function checks a feasibility precondition before searching $K$. If the user's $\rho$ is outside the network's structurally achievable range (computed via `feasible_rho_range`), no $K$ can repair the situation, and the function returns immediately with `stopping_reason = :rho_infeasible`.

	If the prior is feasible, the function iterates $K$ from `effective_K_max` down to `K_min`, calling `compute_setup` at each candidate. The first (largest) $K$ that satisfies all three priors is returned. The search is deterministic; no Monte Carlo replication is needed at this stage since `compute_setup`'s outputs are functions of its inputs only.

	**Value**

	NamedTuple with fields:
	- `optimal_K::Int`: Largest $K$ satisfying all three priors, or `-1` if no valid $K$ was found.
	- `test_results::Vector{NamedTuple}`: Per-$K$ diagnostics (one entry per $K$ tested, in descending order).
	- `stopping_reason::Symbol`: `:found`, `:rho_infeasible`, `:no_valid_K`, or `:K_min_reached`.
	- `feasibility::NamedTuple`: Result of `feasible_rho_range`, included so the caller has access to feasibility info regardless of whether the $K$-search ran.

	**See Also**
	`compute_setup`, `feasible_rho_range`, `reconstruct_network`

	**References**
	Network_Confidence_Intervals.tex, Phase 2 Setup phase.
	""" find_optimal_K

#	compute_setup: orchestrate the Setup phase, return a SamplerSetup
	function compute_setup(edges::DataFrame,
							nodes::DataFrame,
							community_labels::Vector{Int};
							directed::Bool,
							weighted::Bool,
							pi_node::Float64 = 0.0,
							pi_edge::Float64 = 0.0,
							rho::Float64 = 0.0,
							partially_observed_nodes::Vector{Int} = Int[],
							K::Union{Int, Symbol} = :auto,
							J::Int = DEFAULT_J,
							min_nodes_per_ei_bin::Int = 3,
							min_nodes_per_degree_bin::Int = 5,
							verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: standard observed-network inputs
			directed::Bool: whether the network is directed
			weighted::Bool: whether the network has meaningful edge weights
			pi_node::Float64: fraction of nodes believed missing, [0, 1)
				(default = 0.0)
			pi_edge::Float64: fraction of edge weight believed missing,
				[0, 1) (default = 0.0)
			rho::Float64: target centrality-missingness correlation,
				(-1, 1) (default = 0.0)
			partially_observed_nodes::Vector{Int}: indices of nominated
				non-respondents (default = [])
			K::Union{Int, Symbol}: degree bin count, or :auto (default).
				When :auto, find_optimal_K is called.
			J::Int: number of E/I bins (default = DEFAULT_J = 3)
			min_nodes_per_ei_bin::Int: minimum nodes per E/I bin (default 3)
			min_nodes_per_degree_bin::Int: passed to find_optimal_K when
				K = :auto (default 5)
			verbose::Bool: print K-selection diagnostics (default false)
		Returns:
			SamplerSetup struct
		Throws:
			ArgumentError if K = :auto and rho is infeasible.
		"""

		#	Guards
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))
			length(community_labels) == nrow(nodes) ||
				throw(ArgumentError("community_labels length must equal nrow(nodes)"))
			0.0 <= pi_node < 1.0 ||
				throw(ArgumentError("pi_node must be in [0, 1), got $pi_node"))
			0.0 <= pi_edge < 1.0 ||
				throw(ArgumentError("pi_edge must be in [0, 1), got $pi_edge"))
			-1.0 < rho < 1.0 ||
				throw(ArgumentError("rho must be in (-1, 1), got $rho"))
			J >= 1 || throw(ArgumentError("J must be >= 1, got $J"))

		#	Resolve K
			K_resolved = if K == :auto
				if verbose
					println("compute_setup: K = :auto; running find_optimal_K...")
				end
				k_search = find_optimal_K(edges, nodes, community_labels;
										   directed = directed,
										   weighted = weighted,
										   pi_node = pi_node,
										   rho = rho,
										   partially_observed_nodes = partially_observed_nodes,
										   J = J,
										   min_nodes_per_ei_bin = min_nodes_per_ei_bin,
										   min_nodes_per_degree_bin = min_nodes_per_degree_bin,
										   verbose = verbose)

				if k_search.optimal_K == -1
					if k_search.stopping_reason == :rho_infeasible
						throw(ArgumentError(
							"rho = $rho is outside this network's feasibility range " *
							"[$(round(k_search.feasibility.rho_min, digits=4)), " *
							"$(round(k_search.feasibility.rho_max, digits=4))]. " *
							"Use feasible_rho_range() to check what targets this network supports, " *
							"or pass an explicit K value to override."))
					else
						throw(ErrorException(
							"find_optimal_K could not find a valid K " *
							"(stopping_reason = $(k_search.stopping_reason)). " *
							"Pass an explicit K value to override."))
					end
				end

				if verbose
					println("compute_setup: K = $(k_search.optimal_K) selected by find_optimal_K")
				end

				k_search.optimal_K
			else
				K::Int   # user-supplied integer; type assertion confirms
			end

			K_resolved >= 2 ||
				throw(ArgumentError("K must be >= 2, got $K_resolved"))

		#	Initialize Diagnostics
			diag = Dict{Symbol, Any}()
			diag[:K_used] = K_resolved

		#	Step 1: Compute Observed Centrality
			centrality = _compute_observed_centrality(edges, nodes, directed)

		#	Step 1.5: Detect Community Structure (Consume Precomputed Labels)
			ei_result = _detect_community_structure(edges, nodes, community_labels;
													 J = J,
													 min_nodes_per_ei_bin = min_nodes_per_ei_bin)
			ei_values = ei_result.ei_values
			binning_mode = ei_result.binning_mode
			diag[:binning_mode] = binning_mode
			diag[:fallback_reason] = ei_result.fallback_reason

		#	Step 2: Bin Observed Nodes
			binning = _bin_observed_nodes(centrality, ei_values, K_resolved, J, binning_mode)
			degree_bins = binning.degree_bins
			ei_bins = binning.ei_bins
			J_effective = binning.J_effective

		#	Step 3: Compute P (and w) Matrix
			pw = _compute_p_matrix(edges, nodes, degree_bins, ei_bins,
								   K_resolved, J_effective, directed, weighted;
								   partially_observed = partially_observed_nodes)
			P = pw.P
			w = pw.w

		#	Step 3.5: Compute R Matrix (Directed Only)
			if directed
				R = _compute_r_matrix(edges, nodes, degree_bins, ei_bins,
									  K_resolved, J_effective;
									  partially_observed = partially_observed_nodes)
			else
				R = nothing
			end

		#	Step 4: Determine N_add
			N_obs = nrow(nodes) - length(partially_observed_nodes)
			N_nom = length(partially_observed_nodes)
			N_add = _determine_n_add(pi_node, N_obs, N_nom)

		#	Step 5: Solve for Beta and Compute Conditional E/I Distribution
			beta_result = _solve_bin_distribution(rho, K_resolved, N_obs, N_add)
			beta = beta_result.beta
			q = beta_result.q
			beta_status = beta_result.status
			diag[:beta_status] = beta_status
			diag[:beta_n_iters] = beta_result.n_iters

			ei_conditional = _compute_ei_conditional(degree_bins, ei_bins,
													  K_resolved, J_effective)

		#	Assemble SamplerSetup
			return SamplerSetup(
				#	Inputs
					edges,
					nodes,
					directed,
					weighted,
					pi_node,
					pi_edge,
					rho,

				#	Per-Node Setup Outputs
					centrality,
					community_labels,
					ei_values,
					binning_mode,
					degree_bins,
					ei_bins,
					K_resolved,
					J_effective,

				#	Matrices
					P,
					w,
					R,

				#	Added-Node Specification
					partially_observed_nodes,
					N_add,
					beta,
					beta_status,
					q,
					ei_conditional,

				#	Diagnostics
					diag,
			)
	end

#	Exports (public API)
	export SamplerSetup,
		   build_community_corpus,
		   compute_setup,
           feasible_rho_range

end # module network_reconstruction
