__precompile__(true)

@doc raw"""
MIT License

Copyright (c) 2025 Jonathan H. Morgan, Ph.D., Netanomics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
""" 

module Network_Credible_Intervals

#   Module Packages
    using DataFrames
    using Statistics
    using StatsBase

#   Setting Reconstruction Function Constants
    const DEFAULT_J                 = 3
    const EI_SEMANTIC_THRESHOLDS_J3 = (-0.33, 0.33)
    const LAPLACE_NUMERATOR_ADD     = 1.0
    const LAPLACE_DENOMINATOR_ADD   = 2.0

#   Load submodules
    include("graphml_io.jl")
    include("network_community_detection.jl")
    include("network_statistics.jl")
    include("network_reconstruction_functions.jl")
    include("network_degeneracy_functions.jl")

#   Pull their exports into the parent namespace
    using .graphml_io
    using .network_community_detection
    using .network_statistics
    using .network_reconstruction
    using .network_degeneracy

#################
#   UTILITIES   #
#################

#	Credible Interval From a Sample of Measure Values
	function construct_credible_interval(values::AbstractVector{<:Real};
										 prob::Real = 0.89)
		"""
		Args:
			values::AbstractVector{<:Real}: per-sample values of one measure across the
				reconstruction sample (one entry per replicate). Non-finite entries
				(NaN/Inf) are dropped before summarizing.
			prob::Real: credible mass in (0, 1) for the equal-tailed percentile interval
				(default 0.89, after McElreath). The interval runs from the (1 - prob)/2
				quantile to the 1 - (1 - prob)/2 quantile; the point estimate is the
				posterior median.
		Returns:
			NamedTuple (lower, median, upper, mean, std, prob, n_valid):
				lower, upper::Float64 — equal-tailed interval bounds at mass prob.
				median::Float64 — posterior median (the point estimate).
				mean::Float64 — mean of the finite values.
				std::Float64 — sample standard deviation; NaN with fewer than two finite values.
				prob::Float64 — the credible mass used (echoed, so a stored interval is
					self-documenting).
				n_valid::Int — count of finite values summarized.
				lower/median/upper/mean/std are NaN and n_valid is 0 when no finite value
				is present.
		Notes:
			The shared credible-interval primitive, called by the reconstruction-bootstrap
			wrapper once per measure and reused by the bias/calibration diagnostics, so the
			interval definition lives in one place. The interval is the spread of a measure
			across the reconstruction sample — the framework's uncertainty about the true
			value, since no single reconstruction is known to be the truth.

			This is an equal-tailed PERCENTILE interval (Statistics.quantile, linear
			interpolation / type 7), the analogue of rethinking's PI; it is not an HPDI.
			For the skewed posteriors common to network measures the two differ, and an
			HPDI variant can be added later if wanted. Default mass is 0.89; pass
			prob = 0.95 for the nominal-95% coverage the validation's Track B scores
			against.

			Non-finite values are filtered rather than propagated, so one divergent
			replicate does not collapse the interval to NaN; n_valid records the survivors.
		"""

		#	Validation
			0.0 < prob < 1.0 ||
				throw(ArgumentError("prob must lie in (0, 1), got $prob"))

		#	Equal-Tailed Tail Probabilities
			alpha = (1.0 - prob) / 2.0
			q_lo, q_hi = alpha, 1.0 - alpha

		#	Drop Non-Finite Values
			finite = filter(isfinite, values)
			n_valid = length(finite)

		#	Empty Sample: All NaN
			if n_valid == 0
				return (lower = NaN, median = NaN, upper = NaN,
						mean = NaN, std = NaN, prob = Float64(prob), n_valid = 0)
			end

		#	Quantile-Based Interval and Moments
			return (
				lower   = quantile(finite, q_lo),
				median  = quantile(finite, 0.5),
				upper   = quantile(finite, q_hi),
				mean    = mean(finite),
				std     = n_valid >= 2 ? std(finite) : NaN,
				prob    = Float64(prob),
				n_valid = n_valid,
			)
	end
    @doc raw"""
	**Description**
	Summarize a sample of one measure's values — the per-replicate values produced by the
	network-reconstruction bootstrap — into an equal-tailed credible interval plus
	supporting moments. The default credible mass is 89%, following McElreath's choice of
	a non-95% default to discourage reading the interval as a significance test.

	**Usage**
	`construct_credible_interval(values; prob=0.89)`

	**Arguments**
	- `values::AbstractVector{<:Real}`: Per-sample values of one measure, one entry per
	  reconstruction replicate. Non-finite entries (`NaN`/`Inf`) are dropped before
	  summarizing.
	- `prob::Real`: Credible mass in $(0, 1)$ for the interval (default `0.89`). The
	  interval spans the $(1 - \text{prob})/2$ and $1 - (1 - \text{prob})/2$ quantiles;
	  pass `prob = 0.95` for a nominal-95% interval.

	**Details**
	The interval is an equal-tailed **percentile** interval, computed from the sample
	quantiles (`Statistics.quantile`, linear interpolation / type 7) — the analogue of the
	`PI` function in McElreath's `rethinking`, not a highest-posterior-density (HPDI)
	interval. The two coincide for symmetric posteriors but differ for the skewed
	posteriors common to network measures, where an HPDI is narrower; an HPDI variant may
	be added in future without changing this function's contract.

	The point estimate is the posterior **median**, reported alongside the mean so a skew
	between them is visible. Non-finite replicate values are filtered rather than
	propagated, so a single divergent replicate does not collapse the interval to `NaN`;
	`n_valid` records how many finite values remain and is the basis for any
	effective-sample-size check on the result.

	The credible interval reflects the framework's uncertainty about a measure's true
	value: it is the spread of that measure across the reconstruction sample, since no
	single reconstructed network is known to be the truth.

	**Value**
	A `NamedTuple` with fields:
	- `lower::Float64`, `upper::Float64`: The equal-tailed interval bounds at mass `prob`.
	- `median::Float64`: The posterior median (the point estimate).
	- `mean::Float64`: The mean of the finite values.
	- `std::Float64`: The sample standard deviation of the finite values; `NaN` when fewer
	  than two finite values remain.
	- `prob::Float64`: The credible mass used, echoed so a stored interval is
	  self-documenting.
	- `n_valid::Int`: The number of finite values summarized.

	When the sample contains no finite values, `lower`, `median`, `upper`, `mean`, and
	`std` are all `NaN` and `n_valid` is `0`.

	**Examples**
    ```julia
        #	A posterior sample of one scalar measure across reconstruction replicates
            samples = [0.31, 0.34, 0.29, 0.36, 0.33, 0.30, 0.35, 0.32]

        #	Default 89% interval; ci.median is the point estimate
            ci = construct_credible_interval(samples)
            (ci.lower, ci.median, ci.upper, ci.prob)   # prob == 0.89

        #	A 95% interval, e.g. for nominal-95% coverage scoring
            construct_credible_interval(samples; prob = 0.95)

        #	An all-NaN sample yields an empty result rather than an error
            construct_credible_interval([NaN, Inf]).n_valid   # == 0
    ```

	**See Also**
	`reconstruct_network`, `build_reconstruction_corpus`

	**References**
	- McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R
	  and Stan* (2nd ed.). CRC Press. (Percentile vs. highest-posterior-density intervals;
	  the 89% default.)
	""" construct_credible_interval

#	Recover a Degraded (Observed) Network From a Stored Corpus Row
	function recover_degraded_network(true_net::NamedTuple,
									  row;
									  node_loss::Symbol = :emergent,
									  reverse_edges::Bool = false,
									  K::Int             = 4,
									  gc_threshold::Real = 0.30,
									  min_n::Int         = 25,
									  min_edges::Int     = 1,
									  rho_tol::Real      = 0.02,
									  ei_tvd_tol::Real   = 0.25,
									  max_retries::Int   = 10,
									  verify::Bool       = true)
		"""
		Args:
			true_net::NamedTuple: the ground-truth network as returned by load_graphml --
				(edges::DataFrame, nodes::DataFrame, metadata::NamedTuple) with
				metadata.directed and metadata.weighted.
			row: one corpus row (DataFrameRow or NamedTuple) from the degeneration Arrow
				file. Must carry network_name, nominal_pi_node, nominal_pi_edge,
				substituted_rho, seed, and missing_nodes.
			node_loss::Symbol: :emergent or :targeted -- MUST match the value passed to
				build_degeneration_corpus when the corpus was built (default :emergent).
				Used only on the weight-arm path.
			reverse_edges::Bool: MUST match the corpus build (default false). When true, src/dst
				are transposed before recovery, as the orchestrator did.
			K, gc_threshold, min_n, min_edges, rho_tol, ei_tvd_tol, max_retries: the
				generate_missingness_mask tuning knobs -- MUST match the corpus build for the
				weight-arm path to reproduce the stored row (defaults match
				build_degeneration_corpus's defaults).
			verify::Bool: when true, check that the recovered missing set equals the row's
				stored missing_nodes and error if not -- catching a knob/node_loss mismatch
				(default true).
		Returns:
			NamedTuple (sample_edges, sample_nodes, true_edges, true_nodes, record):
				sample_edges, sample_nodes::DataFrame -- the recovered degraded network G_obs.
				true_edges, true_nodes::DataFrame -- the ground-truth network (orientation-matched).
				record -- the recovery provenance: the materialization result on the node-only
					path, or the full regenerated mask record (with observed/degraded edges) on
					the weight-arm path.
		Notes:
			Produces the (sample, true) pair that feeds compute_bias_score. The Arrow corpus
			stores only the per-replicate record (deltas and summaries), not the edge lists, so a
			row is recovered by re-deriving G_obs from the true network.

			Two paths. When the row's pi_edge is zero, G_obs is fully determined by the stored
			missing_nodes -- _materialize_missing_nodes reproduces it with no seed, no gate, and
			no knob matching. When pi_edge > 0, the per-edge weight removal is not stored, only
			regenerable from the seed, so generate_missingness_mask is re-run with the row's seed,
			targets (substituted_rho is the conditioned target), and node_loss; the cached
			artifacts the orchestrator used are recomputed (true_community via _true_communities,
			both deterministic), and return_removed=true yields the materialized observed network.

			The weight-arm path reproduces the stored row only if node_loss and the build knobs
			match the build_degeneration_corpus call; verify guards this by comparing the
			recovered missing set against the row.
		"""

		#	Guards
			node_loss in (:emergent, :targeted) ||
				throw(ArgumentError("node_loss must be :emergent or :targeted, got $node_loss"))

		#	Ground-Truth Network, Oriented to Match the Corpus Build
			directed   = true_net.metadata.directed
			weighted   = true_net.metadata.weighted
			true_nodes = true_net.nodes
			true_edges = true_net.edges
			if reverse_edges
				true_edges = copy(true_edges)
				true_edges.src, true_edges.dst = true_edges.dst, true_edges.src
			end

		#	Recover the Degraded (Observed) Network
			pi_edge = Float64(row.nominal_pi_edge)
			if pi_edge == 0.0
				#	Node-only: the stored missing set fully determines G_obs.
					record = _materialize_missing_nodes(true_edges, collect(row.missing_nodes);
														nodes = true_nodes, directed = directed)
					sample_edges, sample_nodes = record.edges, record.nodes
			else
				#	Weight arm: per-edge removal is regenerable only from the seed.
					adjb = _graph_to_sparse_matrix(true_edges; nodes = true_nodes, weighted = false)[1]
					comm = _true_communities(adjb)
					record = generate_missingness_mask(true_edges;
								nodes          = true_nodes,
								directed       = directed,
								weighted       = weighted,
								node_loss      = node_loss,
								target_pi_node = row.nominal_pi_node,
								target_pi_edge = row.nominal_pi_edge,
								target_rho     = row.substituted_rho,
								seed           = row.seed,
								true_community = comm,
								adj_binary     = adjb,
								K = K, gc_threshold = gc_threshold, min_n = min_n,
								min_edges = min_edges, rho_tol = rho_tol,
								ei_tvd_tol = ei_tvd_tol, max_retries = max_retries,
								return_removed = true)
					sample_edges, sample_nodes = record.observed_edges, record.observed_nodes
			end

		#	Verify Faithful Reproduction Against the Stored Row
			if verify
				recovered = pi_edge == 0.0 ? collect(row.missing_nodes) : record.missing_nodes
				Set(recovered) == Set(collect(row.missing_nodes)) ||
					throw(ErrorException("recovered missing set does not match stored row for " *
						"$(row.network_name); check that node_loss, reverse_edges, and the build " *
						"knobs (K, rho_tol, ei_tvd_tol, max_retries, ...) match the corpus build"))
			end

		#	Return the (sample, true) Pair for compute_bias_score
			return (sample_edges = sample_edges, sample_nodes = sample_nodes,
					true_edges = true_edges, true_nodes = true_nodes,
					record = record)
	end
    @doc raw"""
	**Description**
	Recover the degraded (observed) network $G_{\text{obs}}$ for one row of a degeneration
	corpus, paired with its ground-truth network and ready to feed `compute_bias_score`. The
	Arrow corpus stores only the per-replicate record — deltas and summaries, not edge lists —
	so a row is recovered by re-deriving $G_{\text{obs}}$ from the true network, exactly as the
	corpus generator produced it.

	**Usage**
	`recover_degraded_network(true_net, row; node_loss=:emergent, reverse_edges=false, K=4, gc_threshold=0.30, min_n=25, min_edges=1, rho_tol=0.02, ei_tvd_tol=0.25, max_retries=10, verify=true)`

	**Arguments**
	- `true_net::NamedTuple`: The ground-truth network as returned by `load_graphml` —
	  `(edges, nodes, metadata)` with `metadata.directed` and `metadata.weighted`.
	- `row`: One corpus row (`DataFrameRow` or `NamedTuple`) from the degeneration Arrow file.
	  Must carry `network_name`, `nominal_pi_node`, `nominal_pi_edge`, `substituted_rho`,
	  `seed`, and `missing_nodes`.
	- `node_loss::Symbol`: `:emergent` or `:targeted`. **Must match** the value passed to
	  `build_degeneration_corpus` when the corpus was built (default `:emergent`); used only on
	  the weight-arm path.
	- `reverse_edges::Bool`: **Must match** the corpus build (default `false`). When `true`,
	  `src`/`dst` are transposed before recovery, as the orchestrator did.
	- `K`, `gc_threshold`, `min_n`, `min_edges`, `rho_tol`, `ei_tvd_tol`, `max_retries`: the
	  `generate_missingness_mask` tuning knobs (see that function for meanings). Their defaults
	  match `build_degeneration_corpus`'s defaults; **they must match the values used to build
	  the corpus** for the weight-arm path to reproduce the stored row.
	- `verify::Bool`: When `true`, check that the recovered missing set equals the row's stored
	  `missing_nodes` and error otherwise — catching a `node_loss` or knob mismatch (default
	  `true`).

	**Details**
	Recovery takes one of two paths. When the row's `nominal_pi_edge` is zero (the node arm),
	$G_{\text{obs}}$ is fully determined by the stored `missing_nodes`: it is reproduced with
	`_materialize_missing_nodes` using no seed, no gate, and no knob matching — so node-only
	rows are robust regardless of the tuning knobs. When `nominal_pi_edge > 0` (the weight arm),
	the per-edge weight removal is not stored, only regenerable from the seed, so
	`generate_missingness_mask` is re-run with the row's `seed`, its targets (`substituted_rho`
	is the conditioned $\rho$ target the orchestrator actually used), and `node_loss`; the
	cached artifacts the orchestrator relied on are recomputed — `true_community` via the
	deterministic weakly-connected-component labeller — and `return_removed=true` yields the
	materialized observed network.

	Both paths are deterministic. The weight-arm path reproduces the stored row bit-for-bit
	only if `node_loss`, `reverse_edges`, and the tuning knobs match the
	`build_degeneration_corpus` call; `verify` exists to make a mismatch fail loudly rather than
	silently return a different degraded network.

	**Value**
	A `NamedTuple` with fields:
	- `sample_edges::DataFrame`, `sample_nodes::DataFrame`: The recovered degraded network
	  $G_{\text{obs}}$.
	- `true_edges::DataFrame`, `true_nodes::DataFrame`: The ground-truth network, orientation-
	  matched to the recovery (transposed when `reverse_edges`).
	- `record`: Recovery provenance — the materialization result on the node-only path, or the
	  full regenerated mask record (carrying the observed and degraded edge lists) on the
	  weight-arm path.

	**Examples**
    ```julia
        using Arrow, DataFrames

        #	The degeneration corpus (Arrow) and the ground-truth network (graphml)
            corpus = DataFrame(Arrow.Table("Data/Degenerate_Networks/smoke_degeneration_corpus.arrow"))
            tn     = load_graphml("Data/Networks/moreno_highschool.graphml")

        #	Recover G_obs for one node-arm row; the result pairs the degraded and true networks
            idx = findfirst(==("moreno_highschool_unweighted"), corpus.network_name)
            rec = recover_degraded_network(tn, corpus[idx, :])

            rec.sample_edges   # the degraded observed network
            rec.true_edges     # the ground truth, orientation-matched

        #	Hand the pair to the bias score (which runs the reconstruction internally)
            compute_bias_score(rec.sample_edges, rec.true_edges;
                            sample_nodes = rec.sample_nodes, true_nodes = rec.true_nodes,
                            metrics = metrics,
                            directed = tn.metadata.directed, weighted = tn.metadata.weighted,
                            pi_node = 0.10, pi_edge = 0.0)

        #	A weight-arm row must be recovered with the same node_loss / knobs used at build time
            recover_degraded_network(tn, corpus[weighted_idx, :]; node_loss = :emergent)
    ```

	**See Also**
	`generate_missingness_mask`, `build_degeneration_corpus`, `compute_bias_score`,
	`load_graphml`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling coverage III:
	  Imputation of missing network data under different network and missing data conditions.
	  *Social Networks*, 68, 148–178. (The induce-missingness-then-recover paradigm this
	  validation follows.)
	""" recover_degraded_network

#	Helper Function for network_credible_intervals: community labels aligned to the roster
	function _community_labels_for_nodes(edges::DataFrame,
										  nodes::DataFrame;
										  directed::Bool,
										  weighted::Bool,
										  method::Symbol,
										  seed::Integer,
										  verbose::Bool)
		"""
		Args:
			edges::DataFrame: the (possibly weight-transformed) network edges.
			nodes::DataFrame: node roster with :id; the full universe, so isolates are labelled.
			directed, weighted::Bool: passed to the detector.
			method::Symbol: :champ (resolution sweep) or :leiden (fixed resolution 1.0).
			seed::Integer: detector RNG seed, for reproducibility.
			verbose::Bool: detector progress printing.
		Returns:
			Vector{Int}: one community label per node, in nodes-row order.
		Notes:
			The detector returns membership in its own canonical node order alongside the matching
			node ids (node_names); this maps id -> label and re-orders to the roster, so the labels
			line up with the rows compute_setup expects. nodes is passed through so the detector's
			universe matches the roster (isolates included).
		"""

		#	Run the Chosen Detector on the Full Node Universe
			result = method === :leiden ?
				leiden_community_detection(edges; nodes = nodes, directed = directed,
										   weighted = weighted, seed = Int(seed),
										   show_progress = verbose) :
				champ_community_detection(edges; nodes = nodes, directed = directed,
										  weighted = weighted, seed = Int(seed),
										  show_progress = verbose)

		#	Map Node Id -> Label, Then Align to Roster Row Order
			label_of = Dict(result.node_names[i] => result.membership[i]
							for i in eachindex(result.membership))
			return Int[label_of[id] for id in nodes.id]
	end

#	Helper Function for compute_bias_score: respondent node ids
	function _respondent_ids(setup::SamplerSetup)
		"""
		Args:
			setup::SamplerSetup: the bootstrap setup; setup.nodes is the observed roster and
				setup.partially_observed indexes the nominated non-respondents within it.
		Returns:
			Vector of node ids for the observed, responding nodes.
		Notes:
			Respondents exclude nominated non-respondents (no outgoing data) and synthetic
			added nodes (absent from setup.nodes by construction), matching the
			respondents-only convention of the node-level bias.
		"""

		#	Indices of Nominated Non-Respondents
			nonresp = Set(setup.partially_observed)

		#	Observed Responding Node Ids
			ids = setup.nodes.id
			return [ids[i] for i in eachindex(ids) if !(i in nonresp)]
	end

#	Helper Function for compute_bias_score: resolve a node roster
	function _resolve_nodes(edges::DataFrame, nodes::Union{Nothing, DataFrame})
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst.
			nodes::Union{Nothing,DataFrame}: explicit roster, or nothing.
		Returns:
			DataFrame: the explicit roster when supplied, else a minimal roster of the unique
				edge endpoints under an :id column.
		Notes:
			An inferred roster omits isolates and any node attributes, so an explicit roster is
			preferred wherever the node universe matters.
		"""

		#	Explicit Roster Wins
			nodes === nothing || return nodes

		#	Infer Roster From Edge Endpoints
			ids = unique(vcat(edges.src, edges.dst))
			return DataFrame(id = ids)
	end

#	Helper Function for compute_bias_score: node-level 1 - correlation over respondents
	function _score_node_metric(node_metric::Function,
								setup::SamplerSetup,
								corpus::DataFrame,
								true_edges::DataFrame,
								true_nodes::DataFrame,
								respondent_ids::AbstractVector,
								cor_method::Symbol)
		"""
		Args:
			node_metric::Function: node-level measure, metric(edges, nodes) -> vector aligned to
				nodes.id order.
			setup::SamplerSetup: bootstrap setup, for re-materializing each replicate.
			corpus::DataFrame: the stored reconstruction sample (one delta row per replicate).
			true_edges, true_nodes::DataFrame: the (scale-matched) ground-truth network.
			respondent_ids::AbstractVector: ids to score over (from _respondent_ids).
			cor_method::Symbol: :pearson or :spearman.
		Returns:
			NamedTuple (correlation, bias, n_respondents, n_replicates), with bias = 1 - correlation
			(SmithMorganMoody2022 eq. 2). correlation is NaN with fewer than two finite respondents.
		Notes:
			Each replicate is rebuilt with materialize_reconstruction and re-measured; the
			per-respondent point estimate is the posterior median across replicates. Respondents
			with a non-finite value on either side are dropped, and n_respondents records the
			survivors.
		"""

		#	True Values Keyed by Node Id
			true_vals = node_metric(true_edges, true_nodes)
			true_map  = Dict(true_nodes.id[i] => Float64(true_vals[i]) for i in eachindex(true_vals))

		#	Accumulate Per-Respondent Values Across Replicates
			B   = nrow(corpus)
			acc = Dict(id => Float64[] for id in respondent_ids)
			for r in 1:B
				rep  = materialize_reconstruction(setup, corpus[r, :])
				vals = node_metric(rep.edges, rep.nodes)
				@inbounds for i in eachindex(vals)
					id = rep.nodes.id[i]
					haskey(acc, id) && push!(acc[id], Float64(vals[i]))
				end
			end

		#	Posterior Median per Respondent, Aligned to True Values
			est = Float64[]
			tru = Float64[]
			for id in respondent_ids
				s = filter(isfinite, acc[id])
				push!(est, isempty(s) ? NaN : median(s))
				push!(tru, get(true_map, id, NaN))
			end

		#	Correlation Over Respondents With Finite Values on Both Sides
			keep = isfinite.(est) .& isfinite.(tru)
			n_respondents = count(keep)
			correlation = if n_respondents >= 2
				cor_method === :spearman ? corspearman(tru[keep], est[keep]) : cor(tru[keep], est[keep])
			else
				NaN
			end

		#	1 - Correlation Is the Bias
			return (correlation = correlation, bias = 1.0 - correlation,
					n_respondents = n_respondents, n_replicates = B)
	end

#######################
#   PACKAGE WRAPPER   #
#######################

#	Credible-Interval Bootstrap From an Observed Network (user entry point)
	function network_credible_intervals(edges::DataFrame,
										nodes::DataFrame;
										metrics::Dict{Symbol, <:Function},
										directed::Bool,
										weighted::Bool,
										pi_node::Float64,
										pi_edge::Float64,
										rho::Float64 = 0.0,
										community_labels::Union{Nothing, Vector{Int}} = nothing,
										community_method::Symbol = :champ,
										weight_semantics::Symbol = :count,
										weight_method::Symbol = :scaled_reciprocal,
										tau::Union{Nothing, Float64} = nothing,
										n_replicates::Int = 1000,
										prob::Real = 0.89,
										K::Union{Int, Symbol} = :auto,
										partially_observed_nodes::Vector{Int} = Int[],
										allocation::Symbol = :observed,
										seed::Integer = 1,
										store_raw::Bool = false,
										verbose::Bool = false)
		"""
		Args:
			edges::DataFrame: observed-network edges (:src, :dst, optional :weight). Treated as the
				observed graph the bootstrap reconstructs from.
			nodes::DataFrame: observed-network roster with :id and any node attributes.
			metrics::Dict{Symbol,Function}: scalar measures, each metric(edges, nodes) -> Real,
				evaluated on every replicate and summarized into one interval per measure. Required,
				non-empty.
			directed::Bool: true for directed, false for undirected. Must be supplied.
			weighted::Bool: whether edge weights carry tie intensity. Forced true after a distance
				transform.
			pi_node::Float64: assumed fraction of nodes missing, in [0, 1).
			pi_edge::Float64: assumed fraction of edge weight missing, in [0, 1). Floored internally.
			rho::Float64: assumed Kendall tau-b between missingness and centrality (default 0.0, MCAR);
				clamped to the achievable envelope.
			community_labels::Union{Nothing,Vector{Int}}: community solution in nodes order; detected
				on the (transformed) network when nothing.
			community_method::Symbol: detector used when community_labels is nothing -- :champ (default,
				resolution sweep) or :leiden (fixed resolution). Ignored when labels are supplied.
			weight_semantics::Symbol: :count (default) or :distance (lower = stronger; transformed
				to the count scale before anything else).
			weight_method::Symbol: distance-transform method when weight_semantics == :distance --
				:scaled_reciprocal (default), :max_minus, or :exp_decay. Ignored otherwise.
			tau::Union{Nothing,Float64}: decay length for the :exp_decay transform; required (> 0) for
				that method, ignored otherwise.
			n_replicates::Int: number of accepted replicates to draw (default 1000).
			prob::Real: credible mass for every interval, in (0, 1) (default 0.89).
			K::Union{Int,Symbol}: number of centrality-rank bins, or :auto (default).
			partially_observed_nodes::Vector{Int}: indices (in nodes order) of nominated
				non-respondents. Empty when there are none.
			allocation::Symbol: Stage-2 weight-allocation mode, :observed (default, Bellutta
				proportional-to-current) or :deficit (estimate-based inverse, re-anchoring recovery
				to estimated-true weight). Forwarded to build_reconstruction_corpus / compute_setup;
				identical to :observed at rho = 0. See compute_setup.
			seed::Integer: master seed; all replicate seeds and the detector derive from it (default 1).
			store_raw::Bool: also return setup and corpus so the sample can feed compute_bias_score or
				be re-measured (default false).
			verbose::Bool: progress and rho-adjustment printing (default false).
		Returns:
			NamedTuple:
				intervals::Dict{Symbol,NamedTuple}: per measure, the construct_credible_interval result
					(lower, median, upper, mean, std, prob, n_valid).
				weight_accounting::NamedTuple: floored-pi_edge decomposition.
				rho_requested, rho_conditioned::Float64, rho_adjusted::Bool: assumed rho, conditioned
					value after clamping, and whether clamping occurred.
				n_gate_failures::Int: replicates accepted by fallback (gate not passed).
				weight_transform::Union{Nothing,NamedTuple}: (method, c, n_dropped) when a distance
					transform was applied, else nothing.
				community_source::Symbol: :supplied or :detected.
				n_replicates::Int, prob::Float64: echoed.
				setup::Union{Nothing,SamplerSetup}, corpus::Union{Nothing,DataFrame}: when store_raw,
					else nothing.
		Notes:
			User entry point: observed network + prior in, intervals out. Steps: (1) if
			weight_semantics == :distance, transform weights to the count-like scale -- all later
			steps, and the intervals, then refer to the transformed network; (2) if no community
			solution was supplied, detect one on the transformed network and align it to the roster;
			(3) draw the gated B-replicate sample with build_reconstruction_corpus; (4) summarize each
			metric column with construct_credible_interval at mass prob.

			metrics are SCALAR; node-level centrality vectors are scored separately by
			compute_bias_score, not turned into intervals here. Measures must be configured for the
			resulting network's weighted-ness (strength-based when weighted, binary when not). rho is
			clamped to the achievable envelope inside the bootstrap; rho_requested / rho_conditioned /
			rho_adjusted report what was actually enforced.
		"""

		#	Guards
			!isempty(metrics) ||
				throw(ArgumentError("metrics must be non-empty"))
			weight_semantics in (:count, :distance) ||
				throw(ArgumentError("weight_semantics must be :count or :distance, got $weight_semantics"))
			community_method in (:champ, :leiden) ||
				throw(ArgumentError("community_method must be :champ or :leiden, got $community_method"))
			allocation in (:observed, :deficit) ||
				throw(ArgumentError("allocation must be :observed or :deficit, got $allocation"))
			0.0 < prob < 1.0 ||
				throw(ArgumentError("prob must lie in (0, 1), got $prob"))
			n_replicates >= 1 ||
				throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))

		#	Weight Treatment (distance -> count-like, before anything else)
			if weight_semantics === :distance
				wt = transform_distance_weights(edges; method = weight_method, tau = tau)
				work_edges       = wt.edges
				work_weighted    = true
				weight_transform = (method = wt.method, c = wt.c, n_dropped = wt.n_dropped)
			else
				work_edges       = edges
				work_weighted    = weighted
				weight_transform = nothing
			end

		#	Community Solution (supplied or detected on the transformed network)
			if community_labels === nothing
				labels = _community_labels_for_nodes(work_edges, nodes;
													 directed = directed, weighted = work_weighted,
													 method = community_method, seed = seed,
													 verbose = verbose)
				community_source = :detected
			else
				length(community_labels) == nrow(nodes) ||
					throw(ArgumentError("community_labels length $(length(community_labels)) " *
										"does not match nodes count $(nrow(nodes))"))
				labels = community_labels
				community_source = :supplied
			end

		#	Reconstruction Bootstrap (gated B-replicate sample, measured per metric)
			built = build_reconstruction_corpus(work_edges, nodes, labels;
												directed = directed, weighted = work_weighted,
												pi_node = pi_node, pi_edge = pi_edge, rho = rho,
												metrics = metrics, n_replicates = n_replicates, K = K,
												partially_observed_nodes = partially_observed_nodes,
												allocation = allocation,
												seed = seed, verbose = verbose)

		#	Credible Interval per Measure
			intervals = Dict{Symbol, NamedTuple}()
			for name in keys(metrics)
				intervals[name] = construct_credible_interval(built.corpus[!, name]; prob = prob)
			end

		#	Return
			return (
				intervals         = intervals,
				weight_accounting = built.weight_accounting,
				rho_requested     = built.rho_requested,
				rho_conditioned   = built.rho_conditioned,
				rho_adjusted      = built.rho_adjusted,
				n_gate_failures   = built.n_gate_failures,
				weight_transform  = weight_transform,
				community_source  = community_source,
				n_replicates      = n_replicates,
				prob              = Float64(prob),
				setup             = store_raw ? built.setup : nothing,
				corpus            = store_raw ? built.corpus : nothing,
			)
	end
	@doc raw"""
	**Description**
	The package's main entry point. Given an observed network and a prior about what is
	missing, return credible intervals on user-specified network measures by bootstrapping
	reconstructions of the missing nodes and weight. The interval for a measure is the spread
	of its value across the reconstruction sample — the framework's uncertainty about the
	measure's true value, conditional on the prior actually enforced, since no single
	reconstruction is known to be the truth. Optionally transforms distance-semantic weights
	and detects community structure when none is supplied.

	**Usage**
	`network_credible_intervals(edges, nodes; metrics, directed, weighted, pi_node, pi_edge, rho=0.0, community_labels=nothing, community_method=:champ, weight_semantics=:count, weight_method=:scaled_reciprocal, tau=nothing, n_replicates=1000, prob=0.89, K=:auto, partially_observed_nodes=Int[], allocation=:observed, seed=1, store_raw=false, verbose=false)`

	**Arguments**
	- `edges::DataFrame`: Observed-network edges (`:src`, `:dst`, optional `:weight`), treated
	  as the graph to reconstruct from.
	- `nodes::DataFrame`: Observed-network roster with `:id` and any node attributes.
	- `metrics::Dict{Symbol,Function}`: Scalar measures, each `metric(edges, nodes) -> Real`,
	  evaluated on every replicate and summarized into one interval per measure. Required and
	  non-empty.
	- `directed::Bool`: Directed or undirected; cannot be inferred from the edge list.
	- `weighted::Bool`: Whether edge weights carry tie intensity. Forced to `true` after a
	  distance transform.
	- `pi_node::Float64`: Assumed fraction of nodes missing, in $[0, 1)$.
	- `pi_edge::Float64`: Assumed fraction of edge weight missing, in $[0, 1)$. Floored
	  internally to the weight the added nodes already imply.
	- `rho::Float64`: Assumed Kendall $\tau_b$ between missingness and centrality (default
	  `0.0`, MCAR); clamped to the achievable envelope.
	- `community_labels::Union{Nothing,Vector{Int}}`: Community solution in `nodes` order;
	  detected when `nothing`.
	- `community_method::Symbol`: Detector used when `community_labels` is `nothing` — `:champ`
	  (default, resolution sweep) or `:leiden` (fixed resolution). Ignored when labels are given.
	- `weight_semantics::Symbol`: `:count` (default; higher = stronger) or `:distance`
	  (lower = stronger; transformed before anything else).
	- `weight_method::Symbol`: Distance-transform method when `weight_semantics = :distance` —
	  `:scaled_reciprocal` (default), `:max_minus`, or `:exp_decay`. Ignored otherwise.
	- `tau::Union{Nothing,Float64}`: Decay length for `:exp_decay`; required ($> 0$) for that
	  method, ignored otherwise.
	- `n_replicates::Int`: Number of accepted replicates (default `1000`).
	- `prob::Real`: Credible mass for every interval, in $(0, 1)$ (default `0.89`, after McElreath).
	- `K::Union{Int,Symbol}`: Number of centrality-rank bins, or `:auto` (default).
	- `partially_observed_nodes::Vector{Int}`: Indices (in `nodes` order) of nominated
	  non-respondents — incoming ties observed, outgoing missing. Empty when none.
	- `allocation::Symbol`: Stage-2 weight-allocation mode, `:observed` (default) or `:deficit`.
	  `:observed` is the Bellutta proportional-to-current inverse; `:deficit` re-anchors recovery
	  to an estimated *true* weight, restoring heavily-depleted high-weight structure that
	  `:observed` under-serves at $\rho \neq 0$. The two are identical at $\rho = 0$. See
	  `compute_setup` for the modes.
	- `seed::Integer`: Master seed; all replicate seeds and the detector derive from it (default `1`).
	- `store_raw::Bool`: Also return `setup` and `corpus`, so the sample can feed
	  `compute_bias_score` or be re-measured (default `false`).
	- `verbose::Bool`: Progress and $\rho$-adjustment printing (default `false`).

	**Details**
	The call proceeds in four steps. (1) If `weight_semantics = :distance`, the weights are
	transformed onto the count-like scale the framework assumes; all later steps, and the
	intervals, then refer to the transformed network. (2) If no community solution was
	supplied, one is detected on the (transformed) network and re-aligned to the roster by id.
	(3) `build_reconstruction_corpus` draws the gated `n_replicates`-replicate sample and
	evaluates every metric on each accepted replicate. (4) Each metric's per-replicate values
	are summarized into a credible interval by `construct_credible_interval` at mass `prob`.

	`metrics` are scalar: each returns one number per replicate. Node-level centrality, which
	yields a vector per replicate, is not turned into intervals here — it is scored against a
	known ground truth by `compute_bias_score`. Measures must be configured for the resulting
	network's weighted-ness (strength-based when weighted, binary when not); a distance
	transform makes the network weighted regardless of the input `weighted` flag.

	`rho` is clamped to the structurally achievable envelope inside the bootstrap, since a
	credible interval is only meaningful relative to the prior actually enforced; the returned
	`rho_requested`, `rho_conditioned`, and `rho_adjusted` report what was requested, what was
	conditioned on, and whether clamping occurred. The run is reproducible in `seed`.

	`allocation` controls only how the missing-weight budget is distributed across edges during
	reconstruction; it does not affect node addition, gating, or the realized priors, and it
	leaves the $\rho = 0$ result unchanged. `:deficit` differs from `:observed` only when
	$\rho \neq 0$, where it allocates by estimated per-edge deficit rather than by current
	weight. See `compute_setup`.

	**Value**
	A `NamedTuple` with fields:
	- `intervals::Dict{Symbol,NamedTuple}`: Per measure, the `construct_credible_interval`
	  result `(lower, median, upper, mean, std, prob, n_valid)`.
	- `weight_accounting::NamedTuple`: The floored-`pi_edge` decomposition
	  (`W_observed`, `implied_min_weight`, `additional_weight`, `W_true`, ...).
	- `rho_requested::Float64`, `rho_conditioned::Float64`, `rho_adjusted::Bool`: The assumed
	  $\rho$, the value conditioned on after clamping, and whether clamping occurred.
	- `n_gate_failures::Int`: Replicates accepted by fallback (the 3-prior gate did not pass).
	- `weight_transform::Union{Nothing,NamedTuple}`: `(method, c, n_dropped)` when a distance
	  transform was applied, else `nothing`.
	- `community_source::Symbol`: `:supplied` or `:detected`.
	- `n_replicates::Int`, `prob::Float64`: Echoed.
	- `setup::Union{Nothing,SamplerSetup}`, `corpus::Union{Nothing,DataFrame}`: The shared setup
	  and the per-sample corpus when `store_raw`, else `nothing`.

	**Examples**
    ```julia
            using DataFrames

            #	A small directed, weighted observed network
                edges = DataFrame(src = [1, 1, 2, 3, 4, 4], dst = [2, 3, 3, 4, 1, 2],
                                weight = [3.0, 1, 2, 1, 4, 1])
                nodes = DataFrame(id = 1:4)

            #	Scalar measures as (edges, nodes) closures around the package's measures
                metrics = Dict(
                    :indeg_central => (e, n) -> centralization(in_degree(e; nodes = n, weighted = true)),
                    :total_central => (e, n) -> centralization(total_degree(e; nodes = n, weighted = true)),
                )

            #	89% intervals under an assumed MCAR prior with 10% of nodes missing
                res = network_credible_intervals(edges, nodes;
                                                metrics = metrics,
                                                directed = true, weighted = true,
                                                pi_node = 0.10, pi_edge = 0.05, rho = 0.0)

                res.intervals[:indeg_central]   # (lower, median, upper, mean, std, prob, n_valid)
                res.rho_conditioned             # the rho actually enforced after clamping

            #	Centrality-tilted prior with estimate-based weight allocation (re-anchors
            #	recovery to estimated-true weight; differs from :observed only when rho != 0)
                network_credible_intervals(edges, nodes; metrics = metrics,
                                        directed = true, weighted = true,
                                        pi_node = 0.10, pi_edge = 0.20, rho = -0.35,
                                        allocation = :deficit)

            #	Distance-semantic weights (smaller = stronger), transformed first; keep the raw
            #	sample so it can later feed compute_bias_score
                network_credible_intervals(edges, nodes; metrics = metrics,
                                        directed = true, weighted = true,
                                        pi_node = 0.10, pi_edge = 0.05,
                                        weight_semantics = :distance, weight_method = :scaled_reciprocal,
                                        store_raw = true)
    ```

	**See Also**
	`construct_credible_interval`, `compute_bias_score`, `build_reconstruction_corpus`,
	`transform_distance_weights`, `feasible_rho_range`, `reconstruct_network`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling coverage III:
	  Imputation of missing network data under different network and missing data conditions.
	  *Social Networks*, 68, 148–178. (The missing-data setting the framework addresses.)
	- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. (The 89% default
	  credible mass.)
	""" network_credible_intervals

#	Bias Score Against a Known Ground-Truth Network
	function compute_bias_score(sample_edges::DataFrame,
								true_edges::DataFrame;
								sample_nodes::Union{Nothing, DataFrame} = nothing,
								true_nodes::Union{Nothing, DataFrame} = nothing,
								metrics::Dict{Symbol, <:Function} = Dict{Symbol, Function}(),
								node_metrics::Dict{Symbol, <:Function} = Dict{Symbol, Function}(),
								directed::Bool,
								weighted::Bool,
								pi_node::Float64,
								pi_edge::Float64,
								rho::Float64 = 0.0,
								community_labels::Union{Nothing, Vector{Int}} = nothing,
								weight_semantics::Symbol = :count,
								weight_method::Symbol = :scaled_reciprocal,
								tau::Union{Nothing, Float64} = nothing,
								n_replicates::Int = 1000,
								prob::Real = 0.89,
								K::Union{Int, Symbol} = :auto,
								partially_observed_nodes::Vector{Int} = Int[],
								allocation::Symbol = :observed,
								seed::Integer = 1,
								cor_method::Symbol = :pearson,
								verbose::Bool = false)
		"""
		Args:
			sample_edges::DataFrame: the degraded/observed network's edges (:src, :dst, optional
				:weight); the graph the bootstrap reconstructs from.
			true_edges::DataFrame: the ground-truth network's edges (full, pre-degradation).
			sample_nodes::Union{Nothing,DataFrame}: degraded roster (:id, attributes); inferred from
				sample_edges when nothing, though an explicit roster is preferred.
			true_nodes::Union{Nothing,DataFrame}: ground-truth roster; inferred from true_edges when
				nothing. Must contain every respondent id.
			metrics::Dict{Symbol,Function}: scalar (network-level) measures, metric(edges, nodes) ->
				Real; scored by standardized absolute bias and also driving the returned intervals.
			node_metrics::Dict{Symbol,Function}: node-level measures, metric(edges, nodes) -> vector
				aligned to nodes.id order; scored by 1 - correlation over respondents. At least one
				of metrics / node_metrics must be non-empty.
			directed, weighted, pi_node, pi_edge, rho, community_labels, weight_semantics,
				weight_method, tau, n_replicates, prob, K, partially_observed_nodes, allocation, seed,
				verbose: reconstruction controls, forwarded unchanged to network_credible_intervals
				(see its Args). allocation selects the Stage-2 weight-allocation mode (:observed or
				:deficit) and so can shift the bias scores when rho != 0.
			cor_method::Symbol: :pearson (default, matching SmithMorganMoody2022) or :spearman.
		Returns:
			NamedTuple (network, node, reconstruction):
				network::Dict{Symbol,NamedTuple}: per scalar measure, (true_value, point_estimate,
					signed_bias, bias). point_estimate is the posterior median; signed_bias =
					(point - true)/true (positive = over-estimate); bias = abs(signed_bias); NaN when
					true_value == 0 or the point estimate is non-finite.
				node::Dict{Symbol,NamedTuple}: per node-level measure, (correlation, bias,
					n_respondents, n_replicates); bias = 1 - correlation.
				reconstruction::NamedTuple: the full network_credible_intervals result for the sample
					network, so the intervals built in the same pass are not recomputed.
		Notes:
			Track A of the roadmap, with both forms from SmithMorganMoody2022: standardized absolute
			bias |(Observed - True)/True| for graph-level measures (eq. 3) and 1 - cor(True, Observed)
			over respondents for node-level centrality (eq. 2), where "Observed" is the posterior
			median across the B replicates.

			The utility runs network_credible_intervals on the sample network (store_raw forced) to
			draw the gated sample once, then scores against the true network. Scalar bias reuses the
			posterior medians the bootstrap already produced; node-level bias re-materializes each
			replicate and recomputes the per-node vector (the expensive path).

			When weight_semantics == :distance, the bootstrap transforms the sample's weights
			internally and the true network is transformed here by the same method, so both sides are
			scored on the count-like scale the intervals live on. Respondents exclude nominated
			non-respondents and synthetic added nodes; the node-level correlation is over respondents
			only. Point-estimate diagnostic only (Track A); coverage/width (Track B) and prior
			sensitivity (Track C) are separate.
		"""

		#	Guards
			(!isempty(metrics) || !isempty(node_metrics)) ||
				throw(ArgumentError("at least one of metrics / node_metrics must be non-empty"))
			cor_method in (:pearson, :spearman) ||
				throw(ArgumentError("cor_method must be :pearson or :spearman, got $cor_method"))

		#	Resolve Rosters
			s_nodes = _resolve_nodes(sample_edges, sample_nodes)
			t_nodes = _resolve_nodes(true_edges, true_nodes)

		#	Match the True Network to the Sample's Measurement Scale
			t_edges = weight_semantics === :distance ?
				transform_distance_weights(true_edges; method = weight_method, tau = tau).edges :
				true_edges

		#	Draw the Reconstruction Sample From the Degraded Network
			recon = network_credible_intervals(sample_edges, s_nodes;
											   metrics = metrics, directed = directed, weighted = weighted,
											   pi_node = pi_node, pi_edge = pi_edge, rho = rho,
											   community_labels = community_labels,
											   weight_semantics = weight_semantics,
											   weight_method = weight_method, tau = tau,
											   n_replicates = n_replicates, prob = prob, K = K,
											   partially_observed_nodes = partially_observed_nodes,
											   allocation = allocation,
											   seed = seed, store_raw = true, verbose = verbose)
			setup  = recon.setup
			corpus = recon.corpus

		#	Respondents (observed responding nodes)
			respondents = _respondent_ids(setup)

		#	Network-Level Bias: standardized absolute bias on the posterior median
			network = Dict{Symbol, NamedTuple}()
			for name in keys(metrics)
				true_value = Float64(metrics[name](t_edges, t_nodes))
				point      = recon.intervals[name].median
				signed     = (true_value == 0.0 || !isfinite(point)) ?
					NaN : (point - true_value) / true_value
				network[name] = (true_value = true_value, point_estimate = point,
								 signed_bias = signed, bias = abs(signed))
			end

		#	Node-Level Bias: 1 - correlation over respondents
			node = Dict{Symbol, NamedTuple}()
			for name in keys(node_metrics)
				node[name] = _score_node_metric(node_metrics[name], setup, corpus,
												t_edges, t_nodes, respondents, cor_method)
			end

		#	Return
			return (network = network, node = node, reconstruction = recon)
	end
	@doc raw"""
	**Description**
	Score the framework's reconstruction of a degraded network against a known
	ground-truth network, measure by measure. Two bias forms are reported, both taken
	from Smith, Morgan & Moody (2022): a correlation-based score for node-level centrality
	and a standardized absolute bias for graph-level measures. This is the
	known-ground-truth (validation / calibration) diagnostic; in production, where no
	truth is available, use `network_credible_intervals` alone.

	**Usage**
	`compute_bias_score(sample_edges, true_edges; sample_nodes=nothing, true_nodes=nothing, metrics=Dict(), node_metrics=Dict(), directed, weighted, pi_node, pi_edge, rho=0.0, community_labels=nothing, weight_semantics=:count, weight_method=:scaled_reciprocal, tau=nothing, n_replicates=1000, prob=0.89, K=:auto, partially_observed_nodes=Int[], allocation=:observed, seed=1, cor_method=:pearson, verbose=false)`

	**Arguments**
	- `sample_edges::DataFrame`: Degraded/observed network edges (`:src`, `:dst`, optional
	  `:weight`) — the graph the bootstrap reconstructs from.
	- `true_edges::DataFrame`: Ground-truth network edges (full, pre-degradation).
	- `sample_nodes::Union{Nothing,DataFrame}`: Degraded roster (`:id`, attributes); inferred
	  from `sample_edges` when `nothing`, though an explicit roster is preferred.
	- `true_nodes::Union{Nothing,DataFrame}`: Ground-truth roster; inferred from `true_edges`
	  when `nothing`. Must contain every respondent id.
	- `metrics::Dict{Symbol,Function}`: Scalar (network-level) measures, each
	  `metric(edges, nodes) -> Real`. Scored by standardized absolute bias; these also drive
	  the returned intervals.
	- `node_metrics::Dict{Symbol,Function}`: Node-level measures, each
	  `metric(edges, nodes) -> AbstractVector{<:Real}` aligned to `nodes.id` order. Scored by
	  $1 - \text{correlation}$ over respondents. At least one of `metrics` / `node_metrics`
	  must be non-empty.
	- `directed::Bool`, `weighted::Bool`: Network type; forwarded to the bootstrap.
	- `pi_node::Float64`, `pi_edge::Float64`: Assumed missing fractions of nodes and of edge
	  weight (the analyst's priors). `pi_edge` is floored internally.
	- `rho::Float64`: Assumed Kendall $\tau_b$ between missingness and centrality (default
	  `0.0`, MCAR); clamped to the achievable envelope.
	- `community_labels::Union{Nothing,Vector{Int}}`: Community solution in `sample_nodes`
	  order; detected on the (transformed) sample network when `nothing`.
	- `weight_semantics::Symbol`: `:count` (default) or `:distance` (lower = stronger).
	- `weight_method::Symbol`, `tau::Union{Nothing,Float64}`: Distance-transform method and,
	  for `:exp_decay`, its decay length. Used only when `weight_semantics = :distance`.
	- `n_replicates::Int`: Replicates drawn from the sample network (default `1000`).
	- `prob::Real`: Credible mass for the returned intervals (default `0.89`); does not affect
	  the bias scores, which use the posterior median.
	- `K::Union{Int,Symbol}`: Number of centrality-rank bins, or `:auto` (default).
	- `partially_observed_nodes::Vector{Int}`: Indices (in `sample_nodes` order) of nominated
	  non-respondents. These are excluded from the node-level correlation.
	- `allocation::Symbol`: Stage-2 weight-allocation mode, `:observed` (default) or `:deficit`,
	  forwarded to `network_credible_intervals`. `:deficit` re-anchors weight recovery to an
	  estimated *true* weight and differs from `:observed` only when $\rho \neq 0$, so it can
	  shift the graph- and node-level bias scores in the centrality-tilted regime. See
	  `compute_setup`.
	- `seed::Integer`: Master seed for reproducibility (default `1`).
	- `cor_method::Symbol`: `:pearson` (default, matching the cited convention) or `:spearman`,
	  for the node-level correlation.
	- `verbose::Bool`: Progress printing (default `false`).

	**Details**
	The utility runs `network_credible_intervals` once on the sample network (with the raw
	sample retained), then scores the resulting posterior against the true network. The
	point estimate throughout is the **posterior median** across the $B$ replicates.

	For graph-level measures the score is the standardized absolute bias
	$\left| (m_{\text{point}} - m(T)) / m(T) \right|$ (eq. 3 of the reference), with the
	signed version retained so the direction of error is visible (positive = over-estimate).
	For node-level centrality the score is $1 - \mathrm{cor}(\text{true}, \text{estimate})$
	(eq. 2), taken over **respondents only** — nominated non-respondents and synthetic added
	nodes are excluded, matching the cited convention — with each respondent's estimate being
	its posterior median across replicates. Node-level measures are recomputed on each
	replicate by re-materializing it, which is the expensive part of the call.

	When `weight_semantics = :distance`, the sample network is transformed to the count-like
	scale inside the bootstrap and the true network is transformed by the same method here,
	so both sides are scored on the scale the intervals live on. This is a point-estimate
	diagnostic only (Track A); interval coverage and width (Track B) and prior sensitivity
	(Track C) are computed elsewhere.

	**Value**
	A `NamedTuple` with fields:
	- `network::Dict{Symbol,NamedTuple}`: One entry per `metrics` key, each
	  `(true_value, point_estimate, signed_bias, bias)`. `signed_bias = (point - true)/true`;
	  `bias = abs(signed_bias)`; all `NaN` when `true_value == 0` or the point estimate is
	  non-finite.
	- `node::Dict{Symbol,NamedTuple}`: One entry per `node_metrics` key, each
	  `(correlation, bias, n_respondents, n_replicates)`, with `bias = 1 - correlation` and
	  `correlation` `NaN` when fewer than two respondents have finite values on both sides.
	- `reconstruction::NamedTuple`: The full `network_credible_intervals` result for the sample
	  network, so the intervals built in the same pass are available without recomputing.

	**Examples**
    ```julia
            using DataFrames

            #	Ground truth, and a degraded version missing one node's out-ties
                true_edges   = DataFrame(src = [1, 1, 2, 3, 4], dst = [2, 3, 3, 4, 1])
                sample_edges = DataFrame(src = [1, 1, 2, 3],     dst = [2, 3, 3, 4])
                nodes        = DataFrame(id = 1:4)

            #	A scalar measure and a node-level measure, each as an (edges, nodes) closure
                metrics      = Dict(:deg_central =>
                                    (e, n) -> centralization(total_degree(e; nodes = n, weighted = false)))
                node_metrics = Dict(:total_degree =>
                                    (e, n) -> total_degree(e; nodes = n, weighted = false))

            #	Score the reconstruction against the truth under an MCAR prior
                bias = compute_bias_score(sample_edges, true_edges;
                                        sample_nodes = nodes, true_nodes = nodes,
                                        metrics = metrics, node_metrics = node_metrics,
                                        directed = true, weighted = false,
                                        pi_node = 0.25, pi_edge = 0.0, n_replicates = 200)

                bias.network[:deg_central].bias       # standardized absolute bias
                bias.node[:total_degree].correlation  # true vs. reconstructed, over respondents
    ```

	**See Also**
	`network_credible_intervals`, `construct_credible_interval`, `materialize_reconstruction`,
	`transform_distance_weights`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling coverage III:
	  Imputation of missing network data under different network and missing data conditions.
	  *Social Networks*, 68, 148–178. (Bias definitions: node-level eq. 2, graph-level eq. 3.)
	""" compute_bias_score

#   Re-export to users of Network_Credible_Intervals
    export write_graphml,
           load_graphml,
           calculate_modularity,
           delta_modularity_undirected_best!,
           delta_modularity_directed_best!,
           delta_modularity_best!,
           _leiden_single_run_preprocessed,
           leiden_community_detection,
           champ_community_detection,
           gini_coefficient,
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
           structural_equivalence_blockmodel,
           generate_missingness_mask,
           apply_weight_removal,
           build_degeneration_corpus,
           SamplerSetup,
		   Replicate,
		   compute_setup,
		   feasible_rho_range,
		   generate_replicate,
		   reconstruct_network,
		   materialize_reconstruction,
		   build_reconstruction_corpus,
           recover_degraded_network,
           construct_credible_interval,
           network_credible_intervals,
           compute_bias_score

end # module Network_Credible_Intervals