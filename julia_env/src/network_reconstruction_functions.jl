module network_reconstruction

#	Module Packages
	using DataFrames
    using Distributions
	using SparseArrays
	using LinearAlgebra
	using Random
	using Statistics
	using StatsBase
	using ..Network_Credible_Intervals
	using ..Network_Credible_Intervals: DEFAULT_J,
								 EI_SEMANTIC_THRESHOLDS_J3,
								 LAPLACE_NUMERATOR_ADD,
								 LAPLACE_DENOMINATOR_ADD

#	Sibling Submodule Helpers
#	These are deliberately reach-imported from sibling submodules so the
#	reconstruction pipeline can call community detection (Phase 1.5) and
#	the Phase 1 materializers without duplicating their implementations.
#	All three submodules sit inside the same parent package, so this is a
#	within-package import.
	using ..network_community_detection: _graph_to_sparse_matrix

#	SamplerSetup: bundle of setup-phase outputs for the Phase 2 reconstruction framework
	"""
	SamplerSetup

	Bundle of setup-phase outputs produced by compute_setup and consumed by
	generate_replicate. Built once per observed network and reused across B
	replicates; the inner replicate loop does no setup work beyond per-replicate
	RNG handling.

	UNIFIED-SPEC NOTE. Reconstruction is the inverse of degeneration: it ADDS
	nodes and weight rather than removing them, but a single rho-governed
	propensity field tilts every stage, exactly as one field drives both stages
	on the degeneration side. The fields below are grouped to make that explicit:
	the shared rho-field (q, d) governs both which bins added nodes occupy and
	which existing edges absorb additional weight; the per-bin tendencies carry
	the directed-aware structure the node-injection and weight-floor stages need;
	and the weight-accounting block records the floored-pi_edge decomposition
	W_observed + implied_min_weight + additional_weight = W_true.

	Fields:

	  -- Inputs / configuration --
	  edges::DataFrame
	      Observed-network edge list (post weight-transformation if applied).
	      Columns :src, :dst, optional :weight.
	  nodes::DataFrame
	      Observed-network roster with :id, :label. Includes any nominated
	      non-respondents (flagged via partially_observed).
	  directed::Bool, weighted::Bool
	      Inherited from network metadata.
	  pi_node::Float64
	      Prior fraction of nodes believed missing, in [0, 1). The FREE node dial.
	  rho::Float64
	      Target Kendall tau-b between missingness and centrality, in (-1, 1).
	  pi_edge::Float64
	      Total fraction of edge weight believed missing, in [0, 1). This is the
	      FLOORED prior: it is bounded below by implied_min_weight / W_true (the
	      weight the added nodes already commit). compute_setup raises pi_edge to
	      that floor and records the adjustment in diagnostics when the user's
	      prior comes in under it. pi_edge is the DEPENDENT weight dial — node
	      additions plus rho fix its lower bound.
	  allocation::Symbol
          weight-allocation mode: :observed (Bellutta proportional-to-current) or :deficit (estimate-based inverse)

	  -- Per-node observed structure --
	  centrality::Vector{Float64}
	      Per-node centrality in nodes order: binarized in-degree (directed) or
	      binarized degree (undirected). This is the rho basis and the rank-bin
	      basis; it is weight-stripped by construction.
	  community_labels::Vector{Int}
	      Per-node community label (1-based, contiguous); precomputed by Phase 1.5.
	  ei_values::Vector{Float64}
	      Per-node E/I index in [-1, 1]; NaN possible for isolates (binning handles).
	  binning_mode::Symbol
	      :two_dimensional (degree x E/I) or :degree_only (1D fallback).
	  degree_bins::Vector{Int}
	      Per-node rank bin 1..K, rank-equal by centrality (bin 1 peripheral, bin K
	      central) to match the degeneration field's bin convention.
	  ei_bins::Vector{Int}
	      Per-node E/I bin 1..J. All ones in 1D fallback.
	  K::Int, J::Int
	      Degree- and E/I-bin counts (K defaults to DEFAULT_K; J to DEFAULT_J in
	      2D mode, 1 in fallback).

	  -- Shared rho-field (the unification) --
	  beta::Float64
	      Logistic-skew solved (via _solve_bin_distribution) so the bin-index /
	      missingness correlation matches rho.
	  beta_status::Symbol
	      :converged, :ceiling_hit, or :failed_other.
	  q::Vector{Float64}
	      Per-bin added-node distribution, length K, sums to 1. Drawn from per
	      added node in Stage 1.
	  d::Vector{Float64}
	      Per-node RELATIVE propensity (node-mean 1), d_i = K * q[degree_bins[i]].
	      The single rho-tilt the Stage-2 weight allocation draws from, so node
	      placement and weight distribution share one field (mirrors the
	      degeneration side's _solve_propensity_field.d).

	  -- Per-bin tendencies (directed-aware injection + implied-weight floor) --
	  bin_exp_degree::Vector{Float64}
	      E[binarized degree | bin], length K. Sets the tie COUNT a synthetic node
	      placed in a bin receives.
	  bin_exp_strength::Vector{Float64}
	      E[total incident strength | bin], length K. Each full synthetic node's
	      contribution to the implied-weight floor (incident strength = summed
	      incident edge weight).
	  bin_exp_out_strength::Union{Nothing,Vector{Float64}}
	      E[out-strength | bin], length K, directed only (Nothing undirected). Each
	      nominated non-respondent's floor contribution (its incoming ties are
	      already observed; only its outgoing strength is missing).
	  bin_out_fraction::Union{Nothing,Vector{Float64}}
	      Per-bin outgoing-tie fraction = out-degree / (in-degree + out-degree),
	      length K, directed only (Nothing undirected). Splits a synthetic node's
	      tie counts into outgoing vs incoming. Sparse bins are pre-filled at setup
	      with the global out-fraction, so the vector is always usable.

	  -- Attachment matrices (4D: [deg_src, ei_src, deg_dst, ei_dst]) --
	  P::Array{Float64,4}
	      Rank-rank connectivity probability matrix, Laplace-smoothed.
	  w::Array{Float64,4}
	      Conditional mean-weight matrix, same indexing; zero entries fall back to
	      weight 1 at draw time.
	  R::Union{Array{Float64,4}, Nothing}
	      Reciprocity matrix (directed); Nothing undirected.

	  -- Added-node / nomination specification --
	  partially_observed::Vector{Int}
	      Indices (canonical node order) of nominated non-respondents: in the
	      roster with incoming ties observed, outgoing ties missing. Stage 0.5
	      imputes their out-ties. The inverse of degeneration's outgoing-only
	      materialization. Empty when no nominations.
	  N_add::Int
	      Count of fully-synthetic added nodes (both directions missing) injected
	      in Stage 1. From _determine_n_add(pi_node, N, length(partially_observed)).
	  ei_conditional::Matrix{Float64}
	      Empirical P(EI_bin | degree_bin) over observed nodes, K x J. Assigns E/I
	      bins to added nodes given their drawn degree bin.

	  -- Weight accounting (floored pi_edge; W_obs + A + B = W_true) --
	  W_observed::Float64
	      Total observed edge weight (sum of :weight, or edge count when unweighted).
	  implied_min_weight::Float64
	      A: the imputed-EDGE COUNT — one weight unit per binary tie, since missing
	      ties are presumed weak. A = (expected full-add tie count) + (nomination
	      out-tie count) = N_add * sum_k q[k]*bin_exp_degree[k]
	      + sum over nominations of bin_exp_degree[bin]*bin_out_fraction[bin]. On
	      undirected input the nomination term is zero. Equals exactly the weight
	      Stages 0.5/1 lay down (one unit per imputed tie). The floor reported to
	      the user.
	  additional_weight::Float64
	      B: weight to allocate BEYOND the floor, = pi_edge * W_true - A, clamped to
	      >= 0. Spread across existing and imputed edges by the rho-field in Stage 2
	      (the inverse of _sample_weight_removal). Zero when the user's pi_edge sits
	      at the floor.

	  -- Diagnostics --
	  diagnostics::Dict{Symbol, Any}
	      One-shot setup records: binning_mode, beta_status, beta_n_iters,
	      fallback_reason, pi_edge_floor and whether pi_edge was raised to it,
	      W_true estimate, and any other downstream-verification values.
	"""
	struct SamplerSetup
		#	Inputs / configuration
			edges::DataFrame
			nodes::DataFrame
			directed::Bool
			weighted::Bool
			pi_node::Float64
			rho::Float64
			pi_edge::Float64
			allocation::Symbol          # weight-allocation mode: :observed (Bellutta proportional-to-current) or :deficit (estimate-based inverse)

		#	Per-node observed structure
			centrality::Vector{Float64}
			community_labels::Vector{Int}
			ei_values::Vector{Float64}
			binning_mode::Symbol
			degree_bins::Vector{Int}
			ei_bins::Vector{Int}
			K::Int
			J::Int

		#	Shared rho-field (one tilt drives node placement AND weight allocation)
			beta::Float64
			beta_status::Symbol
			q::Vector{Float64}
			d::Vector{Float64}

		#	Per-bin tendencies (directed-aware injection + implied-weight floor)
			bin_exp_degree::Vector{Float64}
			bin_exp_strength::Vector{Float64}
			bin_exp_out_strength::Union{Nothing, Vector{Float64}}
			bin_out_fraction::Union{Nothing, Vector{Float64}}

		#	Attachment matrices (4D: [deg_src, ei_src, deg_dst, ei_dst])
			P::Array{Float64,4}
			w::Array{Float64,4}
			R::Union{Array{Float64,4}, Nothing}

		#	Added-node / nomination specification
			partially_observed::Vector{Int}
			N_add::Int
			ei_conditional::Matrix{Float64}

		#	Weight accounting (floored pi_edge; W_obs + A + B = W_true)
			W_observed::Float64
			implied_min_weight::Float64
			additional_weight::Float64

		#	Diagnostics
			diagnostics::Dict{Symbol, Any}
	end

########################
#   SET-UP FUNCTIONS   #
########################

#	_centrality_for_sampler: shared binarized centrality driver (rho basis)
	function _centrality_for_sampler(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
									directed::Bool)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (any :weight column is
				ignored — the measure is binarized).
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
				(includes isolates); when nothing, inferred from edges.
			directed::Bool: true => binarized in-degree (directed); false =>
				binarized degree (undirected).
		Returns:
			Vector{Float64}: per-node centrality in the canonical node order
				returned by _graph_to_sparse_matrix.
		Notes:
			The shared centrality driver — the c_i the rho-governed propensity
			field is solved against. It is used by BOTH the degeneration sampler
			(via _solve_propensity_field) and the reconstruction setup (rank
			binning, P/R, feasible_rho_range), which is why it lives in this base
			layer and is reach-imported by network_degeneracy.

			BINARIZED: edge weights are stripped (weighted=false in
			_graph_to_sparse_matrix) so the measure reflects tie PRESENCE, not
			intensity. Node loss is the loss of all incident ties, so the rho
			basis must track presence. Tie INTENSITY (incident strength) is a
			separate measure the setup helpers compute for the implied-weight
			floor and the Stage-2 allocation; it is deliberately NOT returned here.

			Directed: in-degree = column sums (edges with dst == node j).
			Undirected: full degree = column sums + row sums, because
			_graph_to_sparse_matrix stores each undirected edge as a single
			directed entry rather than symmetrizing, so the column sum alone
			captures only half the stubs. Orientation — including any reverse/flip
			of the link semantic — is canonicalized by the caller at the network
			boundary, so no flip argument is needed here.

			Returned as Float64 because the values feed floating-point rank-binning
			and the propensity solve. Per-network and target-invariant: the
			orchestrator / setup caches it once per network and reuses it across
			the entire (rho, pi_node, pi_edge) grid.
		"""

		#	Build Binarized Adjacency
			#	weighted=false strips edge weights; isolates are included when
			#	the nodes argument is supplied.
				adj, _, _ = isnothing(nodes) ?
					_graph_to_sparse_matrix(edges; weighted=false) :
					_graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)
				n = size(adj, 1)

		#	Compute Centrality
			#	Directed: in-degree = column sums of A (incoming edges per node).
			#	Undirected: degree = column sums (incoming) + row sums (outgoing
			#	tie-stubs), since each undirected edge is stored as one directed
			#	entry. For directed networks only the column sum is taken.
				centrality = Vector{Float64}(undef, n)
				nzv  = nonzeros(adj)
				rows = rowvals(adj)
				@inbounds for j in 1:n
					s = 0.0
					#	Column j: edges with dst == node j (in-degree contribution)
						for k in nzrange(adj, j)
							s += nzv[k]
						end
					centrality[j] = s
				end
				if !directed
					#	Add row sums (out-degree contribution) so the undirected
					#	centrality is the full degree, not just the half captured
					#	by the column sum.
						@inbounds for j in 1:n
							for k in nzrange(adj, j)
								i = rows[k]
								centrality[i] += nzv[k]
							end
						end
				end

		#	Return
			return centrality
	end
	@doc raw"""
	**Description**
	The shared binarized centrality driver. For directed networks it is
	in-degree; for undirected networks, degree. Edge weights are ignored — the
	measure reflects tie presence, which is the basis the rho-governed propensity
	field is solved against on both the degeneration and reconstruction sides.

	**Usage**
	`_centrality_for_sampler(edges; nodes=nothing, directed=true)`

	**Arguments**
	- `edges::DataFrame`: edge list with `:src`, `:dst`; any `:weight` is ignored.
	- `nodes::Union{Nothing,DataFrame,Vector}`: optional explicit node universe
	  including isolates. When `nothing`, only nodes appearing in `edges` are
	  included.
	- `directed::Bool`: `true` for binarized in-degree, `false` for binarized
	  degree.

	**Value**
	`Vector{Float64}` of per-node centrality in the canonical node order returned
	by `_graph_to_sparse_matrix`.

	**Notes**
	Per-network and invariant across the $(\rho, \pi_{\text{node}}, \pi_{\text{edge}})$
	grid; callers cache it once per network. This is tie PRESENCE only — tie
	intensity (strength) is computed separately by the setup phase for the weight
	floor and allocation.

	**See Also**
	`_solve_propensity_field`, `_solve_bin_distribution`, `feasible_rho_range`,
	`_graph_to_sparse_matrix`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling coverage
	  III: Imputation of missing network data under different network and missing
	  data conditions. *Social Networks*, 68, 148–178.
	""" _centrality_for_sampler

#	_solve_bin_distribution: bisect beta so the bin-tilt's analytic tau-b matches a target rho
	function _solve_bin_distribution(target_rho::Real,
									  K::Int,
									  N::Int,
									  N_add::Int;
									  rho_tol::Float64 = 1e-4,
									  max_iters::Int = 100,
									  beta_bound::Float64 = 50.0)
		"""
		Args:
			target_rho::Real: target Kendall tau-b between bin index and the
				missing/added indicator, in (-1, 1).
			K::Int: number of centrality-rank bins (>= 2).
			N::Int: retained/observed count (degeneration: N - k; reconstruction:
				N_obs). >= 1.
			N_add::Int: missing/added count (degeneration: k; reconstruction:
				N_nom + N_add). >= 0.
			rho_tol::Float64: convergence tolerance on |realized - target| (default 1e-4).
			max_iters::Int: bisection iteration cap (default 100).
			beta_bound::Float64: saturation bound for the beta bracket; |beta| is
				searched in [0, beta_bound] on the sign of target_rho (default 50.0).
		Returns:
			NamedTuple (q, beta, status, n_iters):
				q::Vector{Float64}: per-bin distribution = _q_from_beta(beta, K),
					length K, sums to 1.
				beta::Float64: solved logistic skew.
				status::Symbol: :converged (hit target within rho_tol),
					:ceiling_hit (target exceeds the K-dependent achievable ceiling;
					beta returned at the saturated bound), or :failed_other (bracket
					valid but tolerance not met within max_iters — pathological).
				n_iters::Int: bisection iterations used (0 on the early-exit paths).
		Notes:
			The shared bin-tilt solver. _realized_rho_for_beta is monotone increasing
			in beta and exactly 0 at beta = 0, so a sign-aware bisection finds the
			beta whose analytic tau-b equals target_rho. Both consumers — the
			reconstruction setup (compute_setup Step 5) and the degeneration field
			(_solve_propensity_field) — pass their (retained, missing) split as
			(N, N_add); the same analytic tau-b serves both.

			Early exits: N_add == 0 (no missing set) or |target_rho| <= rho_tol return
			beta = 0 with the uniform q (status :converged when the target is ~0,
			else :ceiling_hit since no tilt can manufacture correlation with an empty
			missing set). When |target_rho| exceeds the realized tau-b at the
			saturated bound, the target is structurally unreachable: beta is pinned at
			the bound and status is :ceiling_hit — feasible_rho_range/find_optimal_K
			honor this rather than chase an impossible rho.

			The achievable ceiling is K-dependent (larger K, higher ceiling), which is
			why feasibility must be probed at the same K used here.
		"""

		#	Guards
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_add >= 0 || throw(ArgumentError("N_add must be >= 0, got $N_add"))
			(-1.0 < target_rho < 1.0) ||
				throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))

			target = Float64(target_rho)

		#	Early Exit: No Missing Set => No Achievable Correlation
			if N_add == 0
				status = abs(target) <= rho_tol ? :converged : :ceiling_hit
				return (q = _q_from_beta(0.0, K), beta = 0.0, status = status, n_iters = 0)
			end

		#	Early Exit: Target ~ 0 => Uniform Tilt Exactly
			if abs(target) <= rho_tol
				return (q = _q_from_beta(0.0, K), beta = 0.0, status = :converged, n_iters = 0)
			end

		#	Bracket on the Sign of the Target (rho monotone increasing in beta)
			lo, hi = target > 0 ? (0.0, beta_bound) : (-beta_bound, 0.0)

		#	Ceiling Check at the Saturated Bound
			beta_sat = target > 0 ? hi : lo
			rho_bound = _realized_rho_for_beta(beta_sat, K, N, N_add)
			if abs(target) > abs(rho_bound)
				return (q = _q_from_beta(beta_sat, K), beta = beta_sat,
						status = :ceiling_hit, n_iters = 0)
			end

		#	Bisection
			beta_mid = 0.0
			iters = 0
			converged = false
			for i in 1:max_iters
				iters = i
				beta_mid = (lo + hi) / 2
				rho_mid = _realized_rho_for_beta(beta_mid, K, N, N_add)
				if abs(rho_mid - target) <= rho_tol
					converged = true
					break
				end
				if rho_mid < target
					lo = beta_mid
				else
					hi = beta_mid
				end
			end

		#	Return
			status = converged ? :converged : :failed_other
			return (q = _q_from_beta(beta_mid, K), beta = beta_mid,
					status = status, n_iters = iters)
	end

#	_q_from_beta: numerically-stable softmax tilt over K centrality-rank bins
	function _q_from_beta(beta::Float64, K::Int)
		"""
		Args:
			beta::Float64: logistic skew parameter. 0 => uniform; > 0 tilts mass
				toward higher (more central) bins; < 0 toward lower (peripheral) bins.
			K::Int: number of centrality-rank bins (>= 2; callers guard this).
		Returns:
			Vector{Float64}: per-bin distribution over bins 1..K, length K, sums to 1.
		Notes:
			The tilt shape shared across the unified pipeline. _solve_bin_distribution
			bisects beta against _realized_rho_for_beta to hit a target rho, and the
			degeneration field (_solve_propensity_field) reads the resulting q as its
			per-bin propensity. Bin indices 1..K are standardized to mean 0 and unit
			sd (b_tilde), so q_k is proportional to exp(beta * b_tilde_k); the
			standardization keeps the tilt scale comparable across different K. A
			numerically stable softmax (subtract max before exp) avoids overflow at
			the saturation bounds (|beta| up to BETA_BISECTION_BOUNDS). Assumes
			K >= 2 — at K = 1 the bin sd is 0 and the standardization is undefined;
			every caller enforces K >= 2.
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
	@doc raw"""
	**Description**
	The softmax tilt over $K$ centrality-rank bins used throughout the unified
	pipeline: $q_k \propto \exp(\beta \, \tilde b_k)$, where $\tilde b_k$
	standardizes the bin index to mean 0 and unit standard deviation. $\beta = 0$
	is uniform; $\beta > 0$ concentrates mass on higher (more central) bins,
	$\beta < 0$ on lower (peripheral) bins.

	**Usage**
	`_q_from_beta(beta, K)`

	**Arguments**
	- `beta::Float64`: logistic skew parameter.
	- `K::Int`: number of centrality-rank bins ($\ge 2$).

	**Value**
	`Vector{Float64}` of length $K$, summing to 1.

	**Notes**
	Helper for `_solve_bin_distribution`; standardizing the bin index keeps
	$\beta$ comparable across $K$, and the subtract-max softmax stays stable at
	the saturation bounds.

	**See Also**
	`_solve_bin_distribution`, `_realized_rho_for_beta`, `_solve_propensity_field`
	""" _q_from_beta

#	_realized_rho_for_beta: analytic Kendall tau-b for a bin-tilt at a given beta
	function _realized_rho_for_beta(beta::Float64, K::Int, N::Int, N_add::Int)
		"""
		Args:
			beta::Float64: candidate logistic skew parameter.
			K::Int: number of centrality-rank bins (>= 2).
			N::Int: count of the RETAINED/observed side (degeneration: N - k
				retained; reconstruction: N observed).
			N_add::Int: count of the MISSING/added side (degeneration: k missing;
				reconstruction: N_add synthetic adds).
		Returns:
			Float64: analytic Kendall tau-b between bin index (1..K) and the
				missing/added indicator (0/1) under q = softmax(beta * b_tilde).
				Exactly 0.0 at beta = 0; approaches a K-dependent ceiling (below the
				rate-bounded sqrt(2 p (1-p))) as |beta| grows.
		Notes:
			The closed-form tau-b at the heart of the shared solver. It is the
			function _solve_bin_distribution bisects to hit a target rho, and the
			value feasible_rho_range evaluates at the beta-bounds to get the
			reachable envelope. (N, N_add) is the (retained, missing) split on the
			degeneration side and the (observed, added) split on the reconstruction
			side — the same analytic tau-b serves both.

			Construction: observed nodes spread uniformly across K rank-equal bins
			(N/K per bin); the missing/added nodes follow q[k] (N_add*q[k] per bin);
			x = bin index (ordinal 1..K), y = indicator (0/1). Only cross pairs (one
			observed, one added) can be concordant/discordant; within-observed and
			within-added pairs are tied on y and enter the denominator only.

			Cross-pair counts:
			  C  (concordant) = (N/K) * N_add * sum_k q[k] * (k - 1)
			  D  (discordant) = (N/K) * N_add * sum_k q[k] * (K - k)
			  Tx (cross, tied on x) = N * N_add / K        [C + D + Tx = N * N_add]
			Within-pair counts (tied on y, distinguishable on x):
			  Ty_obs = N^2 * (K - 1) / (2K)
			  Ty_add = N_add^2 / 2 * (1 - sum_k q[k]^2)
			  Ty = Ty_obs + Ty_add
			tau-b = (C - D) / sqrt((C + D + Tx)(C + D + Ty)).

			K-DEPENDENT CEILING. The maximum achievable |tau-b| at fixed (N, N_add)
			rises with K: larger K shrinks the bin-side ties Tx, pushing the ceiling
			toward the rate-bounded absolute sqrt(2 p (1-p)); smaller K raises Tx and
			lowers it. This is why the envelope and the ceiling-hit rate depend on K,
			and why feasible_rho_range must be evaluated at the SAME K the field
			solve uses.

			MONOTONICITY. q skews toward higher bins as beta rises, so C grows and D
			shrinks: tau-b is monotone increasing in beta, which the bisection in
			_solve_bin_distribution relies on.
		"""

		#	Guards
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_add >= 0 || throw(ArgumentError("N_add must be >= 0, got $N_add"))

		#	N_add = 0: no indicator variation, correlation undefined -> 0
			if N_add == 0
				return 0.0
			end

		#	Per-bin tilt (shared softmax shape)
			q = _q_from_beta(beta, K)

		#	Cross-pair counts: C, D, Tx
			n_obs_per_bin = N / K
			C = 0.0
			D = 0.0
			@inbounds for k in 1:K
				C += n_obs_per_bin * N_add * q[k] * (k - 1)
				D += n_obs_per_bin * N_add * q[k] * (K - k)
			end
			Tx = N * N_add / K

		#	Within-pair counts: Ty (tied on y, distinguishable on x)
			Ty_obs = (N * N * (K - 1)) / (2 * K)
			sum_q2 = 0.0
			@inbounds for k in 1:K
				sum_q2 += q[k] * q[k]
			end
			Ty_add = (N_add * N_add) / 2 * (1.0 - sum_q2)
			Ty = Ty_obs + Ty_add

		#	tau-b
			numer   = C - D
			denom_a = C + D + Tx
			denom_b = C + D + Ty
			(denom_a <= 0.0 || denom_b <= 0.0) && return 0.0
			return numer / sqrt(denom_a * denom_b)
	end
	@doc raw"""
	**Description**
	Closed-form Kendall $\tau_b$ between centrality-rank bin index and the
	missing/added indicator under the tilt $q = \mathrm{softmax}(\beta\,\tilde b)$.
	Exactly 0 at $\beta = 0$; rises monotonically in $\beta$ to a $K$-dependent
	ceiling below the rate bound $\sqrt{2p(1-p)}$.

	**Usage**
	`_realized_rho_for_beta(beta, K, N, N_add)`

	**Arguments**
	- `beta::Float64`: logistic skew.
	- `K::Int`: rank-bin count ($\ge 2$).
	- `N::Int`: retained/observed count.
	- `N_add::Int`: missing/added count.

	**Value**
	`Float64` in $(-1, 1)$.

	**Notes**
	The shared analytic $\tau_b$: bisected by `_solve_bin_distribution` to hit a
	target $\rho$, and evaluated at the $\beta$-bounds by `feasible_rho_range` for
	the reachable envelope. Monotone in $\beta$, so the bisection is well-posed.

	**See Also**
	`_solve_bin_distribution`, `_q_from_beta`, `feasible_rho_range`,
	`_solve_propensity_field`
	""" _realized_rho_for_beta

#	_compute_observed_centrality: per-node rho-basis centrality for the setup phase (Step 1)
	function _compute_observed_centrality(edges::DataFrame,
										   nodes::DataFrame,
										   directed::Bool)
		"""
		Args:
			edges::DataFrame: observed edge list with :src, :dst (any :weight is
				ignored — the rho basis is binarized).
			nodes::DataFrame: node roster with :id; its row order is the canonical
				node order for the whole setup phase.
			directed::Bool: true => binarized in-degree; false => binarized degree.
		Returns:
			Vector{Float64}: per-node centrality in nodes-DataFrame row order;
				isolates get 0.0.
		Notes:
			Step 1 of the setup phase. Produces the SAME rho basis the degeneration
			field uses — binarized in-degree (directed) or binarized degree
			(undirected) — so the reconstruction binning and the rho-governed
			placement agree on what "central" means.

			ALIGNMENT FIX: the prior version summed in-degree AND out-degree (total
			degree) for directed networks. That disagreed with the one-rho
			definition (rho is measured against in-degree, flipped only when the
			link semantic is flipped); binning observed nodes on total degree while
			the field tilts on in-degree would desynchronize placement from rho.
			Directed now counts the destination endpoint only (in-degree);
			undirected counts both endpoints (degree).

			Computed in nodes-DataFrame row order via an id -> index map rather than
			by delegating to _centrality_for_sampler, because the rest of the setup
			phase (binning, P/R matrices) indexes by this same nodes-row order; the
			definition matches _centrality_for_sampler but the ordering convention
			is the setup's, not _graph_to_sparse_matrix's.

			Tie INTENSITY (incident strength) is a separate measure the setup
			computes downstream for the implied-weight floor and the Stage-2
			allocation; it is not part of this rho basis.
		"""

		#	Guards
			hasproperty(edges, :src) && hasproperty(edges, :dst) ||
				throw(ArgumentError("edges must have :src and :dst columns"))
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))

		#	Build Node ID -> Index Mapping (nodes-DataFrame row order is canonical)
			n = nrow(nodes)
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))

		#	Allocate Centrality Vector
			centrality = zeros(Float64, n)

		#	Accumulate Per-Edge Contributions (in-degree basis for directed)
			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)
			@inbounds for k in 1:n_edges
				src_i = get(id_to_idx, src_ids[k], 0)
				dst_i = get(id_to_idx, dst_ids[k], 0)
				#	Destination always contributes: in-degree (directed) or one
				#	endpoint of an undirected edge.
					if dst_i > 0
						centrality[dst_i] += 1.0
					end
				#	Source contributes ONLY when undirected (the other endpoint's
				#	degree). For directed networks the rho basis is in-degree, so
				#	the source's outgoing stub is excluded.
					if !directed && src_i > 0
						centrality[src_i] += 1.0
					end
			end

		#	Return Per-Node Centrality (rho basis)
			return centrality
	end

#	_compute_ei_values: per-node Krackhardt-Stern E/I index (community embeddedness)
	function _compute_ei_values(edges::DataFrame,
								 nodes::DataFrame,
								 community_labels::Vector{Int})
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (any :weight ignored; E/I
				is binary-edge).
			nodes::DataFrame: node roster with :id; its row order is canonical.
			community_labels::Vector{Int}: per-node community assignment in
				nodes-DataFrame row order; length must equal nrow(nodes).
		Returns:
			Vector{Float64}: per-node E/I index in [-1, +1]. Isolates (no incident
				edges) return NaN; the binning step must handle these.
		Notes:
			Step 1.5 of the setup phase. Computes the External-Internal index of
			Krackhardt & Stern (1988):
			    EI_i = (E_i - I_i) / (E_i + I_i)
			with E_i the count of i's edges to OTHER communities and I_i the count
			within i's own community. EI = -1 fully internal (cluster member),
			0 balanced, +1 fully external (broker).

			Binary-edge E/I: each edge contributes 1, weight ignored. Each edge
			contributes to BOTH of its endpoints' counters, so orientation does not
			matter — an A->B tie across communities raises external for both A and
			B. This is the all-incident-edge form (directed and undirected handled
			identically, assuming undirected edges are stored once). EI is
			deliberately NOT restricted to in-edges: it measures embeddedness, not
			centrality.

			EI is a community-EMBEDDEDNESS attribute, ORTHOGONAL to the rho
			centrality basis: rho/in-degree decides which centrality bin a node
			occupies; EI describes how that node's ties split across communities.
			Downstream, _compute_ei_conditional turns these per-node values into
			per-bin EI distributions so synthetic nodes can be assigned realistic
			community roles when their edges are placed.
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

#	_detect_community_structure: decide 2D (degree x EI) vs degree-only binning
	function _detect_community_structure(edges::DataFrame,
										  nodes::DataFrame,
										  community_labels::Vector{Int};
										  J::Int = DEFAULT_J,
										  min_nodes_per_ei_bin::Int = 3)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst.
			nodes::DataFrame: node roster with :id; row order is canonical.
			community_labels::Vector{Int}: precomputed labels (from Phase 1.5
				build_community_corpus), in nodes-DataFrame row order.
			J::Int: target number of E/I bins (default DEFAULT_J = 3).
			min_nodes_per_ei_bin::Int: minimum nodes per E/I bin to keep 2D
				binning active (default 3).
		Returns:
			NamedTuple (ei_values, binning_mode, fallback_reason):
				ei_values::Vector{Float64}: per-node E/I (NaN for isolates).
				binning_mode::Symbol: :two_dimensional or :degree_only.
				fallback_reason::Union{Symbol,Nothing}: why 2D was abandoned when
					mode == :degree_only (:single_community,
					:insufficient_ei_variance, :too_few_nodes); nothing otherwise.
		Notes:
			Step 1.5 of the setup phase. Community detection itself is delegated to
			Phase 1.5 (CHAMP) via community_labels; this helper does NOT call CHAMP.
			It computes E/I from the labels and decides whether the second binning
			axis is usable.

			The two binning axes are the in-degree rho basis (the K degree bins from
			_compute_observed_centrality, which the rho-field governs) and, when
			supportable, an orthogonal J-way E/I (community-embeddedness) axis. This
			helper gates only the E/I axis; the degree/rho axis is always present.

			2D binning is supportable iff there are >= 2 communities, at least
			J * min_nodes_per_ei_bin nodes with defined (non-NaN) E/I, and each E/I
			bin receives at least min_nodes_per_ei_bin nodes (semantic cut points
			EI_SEMANTIC_THRESHOLDS_J3 at J = 3, equal-quantile bins otherwise).
			Failing any check, the framework falls back to degree-only binning,
			which is robust to networks without meaningful community structure.

			Isolates (NaN E/I) are excluded from every supportability count.
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

#	_bin_observed_nodes: assign each node a K degree-bin and a J E/I-bin (Step 2)
	function _bin_observed_nodes(centrality::Vector{Float64},
								  ei_values::Vector{Float64},
								  K::Int,
								  J::Int,
								  binning_mode::Symbol)
		"""
		Args:
			centrality::Vector{Float64}: per-node in-degree rho basis from
				_compute_observed_centrality (nodes-DataFrame row order).
			ei_values::Vector{Float64}: per-node E/I from _compute_ei_values; NaN
				entries (isolates) are tolerated.
			K::Int: number of degree bins (>= 2).
			J::Int: number of E/I bins (>= 1).
			binning_mode::Symbol: :two_dimensional or :degree_only.
		Returns:
			NamedTuple (degree_bins, ei_bins, J_effective):
				degree_bins::Vector{Int}: per-node degree bin, 1..K.
				ei_bins::Vector{Int}: per-node E/I bin, 1..J (all 1s if degree_only).
				J_effective::Int: J if :two_dimensional, else 1. Downstream P, w, R
					matrices use J_effective as their E/I dimension.
		Notes:
			Step 2 of the setup phase. Degree bins are equal-rank: rank nodes by the
			in-degree rho basis (StatsBase.tiedrank), then quantize ranks into K
			contiguous groups. Bin 1 is the most peripheral (lowest centrality),
			bin K the most central. This orientation is deliberate — it matches the
			q-distribution convention from _solve_bin_distribution / _q_from_beta,
			where positive beta (positive rho) skews mass toward HIGH bin indices,
			so a positive-rho placement correctly lands added nodes in the central
			(high-index) bins.

			E/I bins (2D mode) use semantic thresholds at J = 3:
			(-inf, -0.33] -> bin 1 (internal hub); (-0.33, 0.33) -> bin 2 (mixed);
			[0.33, +inf) -> bin 3 (broker / external). For other J, equal-quantile
			bins on the defined (non-NaN) E/I values. Isolates (NaN E/I) go to the
			middle bin ((J+1) / 2); since they have no edges they do not propagate
			through P, w, R, so the placement is effectively a no-op.

			This function returns bin ASSIGNMENTS only. The per-bin weight and
			degree tendencies the implied-weight floor and Stage-2 allocation need
			(expected strength, expected out-strength, out-fraction, expected
			degree) are computed by a separate helper from the weighted edges, since
			this function intentionally carries no edge or weight information.
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

#	_compute_p_matrix: cell-cell attachment probabilities P and conditional weights w (Step 3)
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
			edges::DataFrame: edge list with :src, :dst, optional :weight.
			nodes::DataFrame: node roster with :id; row order is canonical.
			degree_bins::Vector{Int}: per-node degree bin, 1..K.
			ei_bins::Vector{Int}: per-node E/I bin, 1..J.
			K::Int: number of degree bins.
			J::Int: number of E/I bins (1 in degree-only fallback).
			directed::Bool: if true P[c,c'] != P[c',c] in general; if false P is
				symmetric (edges double-counted both ways).
			weighted::Bool: if true compute the conditional mean-weight matrix w;
				if false every edge contributes weight 1.0.
			partially_observed::Vector{Int}: indices of partially observed
				(nominated non-respondent) nodes to exclude. P is computed over
				respondent-respondent dyads only.
		Returns:
			NamedTuple (P, w):
				P::Array{Float64,4}: Laplace-smoothed connection probabilities,
					indexed [deg_src, ei_src, deg_dst, ei_dst].
				w::Array{Float64,4}: conditional mean weights, same indexing.
		Notes:
			Step 3 of the setup phase. P is the rank-rank attachment structure: for
			a source cell c = (degree_bin, ei_bin) and target cell c', P[c,c'] is the
			probability an edge runs from a c-node to a c'-node. Stage 1 uses P to
			impute a synthetic node's edges (presence) and w to weight them.

			This is ORTHOGONAL to rho: rho decides which degree bin an added node
			occupies; P decides which cells it then attaches to. The implied-weight
			floor is derivable from P and w directly — expected out-strength of a
			c-node is sum over c' of P[c,c'] * n_resp(c') * w[c,c'] (transpose for
			in-strength) — so deriving the floor from these matrices keeps it
			consistent with what Stage 1 actually places.

			Laplace smoothing (Beta(1,1), one prior edge-present + one edge-absent):
			    P[c,c'] = (edges_cc' + LAPLACE_NUMERATOR_ADD) /
			              (dyads_cc' + LAPLACE_DENOMINATOR_ADD)
			so empty cell pairs default to P = 0.5. Dyad counts are ordered
			n_src*n_dst across cells and n*(n-1) within a cell; undirected edges are
			double-counted above so the same ordered convention yields a symmetric P.

			w[c,c'] is the mean weight of observed c->c' edges, or 0 if none; a
			0 entry falls back to weight 1 at sampling time. CAVEAT (weighted only):
			for a weighted network this fallback under-weights imputed edges in
			unobserved cell pairs relative to the network's mean weight, which
			slightly under-states the reported floor A (Stage 2's top-up partly
			compensates). Worth checking in weighted calibration.

			Partially observed (Stage 0.5 nominated) nodes are excluded entirely;
			they are placed later from their own observed edges, not from P.
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

#	_compute_r_matrix: per-cell-pair reciprocity matrix R (directed only, Step 3.5)
	function _compute_r_matrix(edges::DataFrame,
								nodes::DataFrame,
								degree_bins::Vector{Int},
								ei_bins::Vector{Int},
								K::Int,
								J::Int;
								partially_observed::Vector{Int} = Int[])
		"""
		Args:
			edges::DataFrame: directed edge list with :src, :dst (any :weight is
				unused — reciprocity is a presence property).
			nodes::DataFrame: node roster with :id; row order is canonical.
			degree_bins::Vector{Int}: per-node degree bin, 1..K.
			ei_bins::Vector{Int}: per-node E/I bin, 1..J.
			K::Int: number of degree bins.
			J::Int: number of E/I bins (1 in degree-only fallback).
			partially_observed::Vector{Int}: indices to exclude; R is computed over
				respondent-respondent dyads only.
		Returns:
			Array{Float64,4}: R indexed [deg_src, ei_src, deg_dst, ei_dst]. R[c,c']
				is the Laplace-smoothed P(reverse edge j -> i | forward edge i -> j),
				for i in cell c and j in cell c'.
		Notes:
			Step 3.5 of the setup phase. The reciprocity matrix is asymmetric in
			general (R[c,c'] != R[c',c]). For each ordered cell pair:
			    R[c,c'] = (mutual forward edges c->c' + LAPLACE_NUMERATOR_ADD)
			              / (forward edges c->c' + LAPLACE_DENOMINATOR_ADD)
			where a forward edge i -> j is "mutual" when j -> i also exists in the
			observed edge set. Beta(1,1) smoothing makes an empty cell pair default
			to R = 0.5.

			Role: in Stage 1, when a directed synthetic node's forward edge i -> j
			(cells c -> c') is imputed via P, R[c,c'] is the probability the reverse
			j -> i is imputed too, producing a mutual tie. This generalizes SMM
			2022's single global reciprocity rate to be conditional on the cell pair,
			capturing structural variation (brokers vs clustered hubs).

			Orthogonal to rho: rho sets which degree bin a node occupies and the
			in/out split sets edge directionality; R only decides which of those
			directed edges become mutual.

			Directed networks only. For undirected networks R is not computed and
			SamplerSetup carries R = nothing (every undirected edge is mutual by
			definition, so a reciprocity matrix is meaningless).
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

#	_determine_n_add: synthetic full-add count N_add from pi_node (Step 4)
	function _determine_n_add(pi_node::Float64, N::Int, N_nom::Int)
		"""
		Args:
			pi_node::Float64: target fraction of TRUE nodes that are missing, in
				[0, 1). Counts both nominated non-respondents and fully-missing
				synthetic nodes as "missing".
			N::Int: number of observed respondents (fully observed nodes).
			N_nom::Int: number of nominated non-respondents (in the observed roster
				with incoming edges observed, outgoing edges missing).
		Returns:
			Int: N_add, the number of fully-synthetic (non-nominated) nodes to add
				per replicate. Always >= 0.
		Notes:
			Step 4. Inverts degeneration's node loss. In degeneration a fraction
			pi_node of the N_true nodes become missing; the survivors (respondents)
			are N = N_true * (1 - pi_node), so N_true = N / (1 - pi_node) and the
			total missing count is
			    M = pi_node * N_true = pi_node * N / (1 - pi_node).
			That total splits into the nominated nodes already visible in the
			observed roster (N_nom) and the fully-missing synthetic nodes to recover:
			    N_add = M - N_nom = round(pi_node * N / (1 - pi_node)) - N_nom.

			BASE IS N, NOT N + N_nom. Using N (respondents = the (1 - pi_node)
			fraction) makes the realized missing fraction equal pi_node exactly:
			N_true = N + N_nom + N_add = N / (1 - pi_node), so M / N_true = pi_node.
			The earlier (N + N_nom) numerator overshoots the target whenever
			N_nom > 0.

			Edge cases:
			- pi_node = 0: returns 0 (Stage 1 adds nothing).
			- N_nom = 0: reduces to round(pi_node * N / (1 - pi_node)).
			- N_add < 0 is clamped to 0. This happens when N_nom exceeds the implied
			  total missing M — i.e., the observed nomination count is already larger
			  than pi_node implies. So the observed N_nom carries an IMPLIED-pi_node
			  FLOOR of N_nom / (N + N_nom) (the value at which N_add = 0); a target
			  below it is infeasible and the realized missing fraction exceeds the
			  target. This is the node-count analogue of the implied-weight floor A
			  on pi_edge, and likewise worth reporting to the user.
		"""

		#	Guards
			0.0 <= pi_node < 1.0 ||
				throw(ArgumentError("pi_node must be in [0, 1), got $pi_node"))
			N >= 1 || throw(ArgumentError("N must be >= 1, got $N"))
			N_nom >= 0 || throw(ArgumentError("N_nom must be >= 0, got $N_nom"))

		#	pi_node = 0: No Missingness
			if pi_node == 0.0
				return 0
			end

		#	Total Missing M = pi_node * N / (1 - pi_node); Split Off the Nominated
			implied_missing = Int(round(pi_node * N / (1.0 - pi_node)))
			n_add = implied_missing - N_nom

		#	Clamp to Nonnegative (N_nom below the implied-pi_node floor)
			return max(n_add, 0)
	end

#	_compute_ei_conditional: P(E/I bin | degree bin) from observed nodes (Step 5 helper)
	function _compute_ei_conditional(degree_bins::Vector{Int},
									  ei_bins::Vector{Int},
									  K::Int,
									  J::Int)
		"""
		Args:
			degree_bins::Vector{Int}: per-node degree bin, 1..K.
			ei_bins::Vector{Int}: per-node E/I bin, 1..J.
			K::Int: number of degree bins.
			J::Int: number of E/I bins.
		Returns:
			Matrix{Float64}: K x J, row-stochastic. Row k is the empirical
				distribution of E/I bin among observed nodes in degree bin k.
		Notes:
			Setup-phase helper feeding Stage 1. A synthetic node's degree bin is
			drawn first from the rho-governed q distribution (the centrality axis);
			this conditional then assigns its E/I bin, so the added node inherits a
			community role (internal hub / mixed / broker) representative of real
			nodes at its degree level. Sampling the two axes in this order — degree
			from q, E/I from P(E/I | degree) — keeps rho governing centrality alone
			while preserving the observed degree-embeddedness association.

			Each row is normalized to sum to 1. A degree bin with no observed nodes
			defaults to uniform across E/I bins; with rank-equal degree binning this
			should not occur, but it is guarded.
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

#	Helper Propensity Field: logistic-beta tilt over centrality-rank bins (shared with reconstruction)
	function _solve_propensity_field(centrality::AbstractVector{<:Real},
										target_rho::Real,
										pi_node::Real,
										K::Int)
		"""
		Args:
			centrality::AbstractVector{<:Real}: per-node centrality from
				_centrality_for_sampler, in canonical node order.
			target_rho::Real: target Kendall tau-b between the binary missingness
				indicator and centrality; in (-1, 1).
			pi_node::Real: target missing-node fraction; in (0, 1). Sets the
				reference split (retained, missing) = (N - k, k), k = round(pi_node*N),
				that the tilt is solved against.
			K::Int: number of centrality-rank bins; 2 <= K <= N.
		Returns:
			NamedTuple: (d, q, beta, bin_index, status)
				- d::Vector{Float64}: per-node RELATIVE propensity (the rho-tilted
					shape), normalized to node-mean 1. Each stage applies its own
					level (pi_edge for weight removal, pi_node for node accounting).
				- q::Vector{Float64}: per-bin softmax distribution, length K.
				- beta::Float64: solved logistic-skew parameter.
				- bin_index::Vector{Int}: per-node centrality-rank bin (1..K).
				- status::Symbol: :converged, :ceiling_hit, or :failed_other.
		Notes:
			The single rho-governed field that both the weight-removal stage and
			the node-accounting stage draw from, replacing the old b-mixing prob
			vector. Reuses the reconstruction module's logistic-beta machinery
			directly via _solve_bin_distribution, which bisects
			_realized_rho_for_beta to find q = softmax(beta * b_tilde) over K bins.
			Degradation's (retained, missing) counts map onto reconstruction's
			(N, N_add), so the identical analytic tau-b applies and both modules
			share one tilt.

			At target_rho = 0 the solve returns beta = 0, q uniform, d constant 1,
			so weight removal is proportional-to-current-weight (Bellutta MCAR)
			and node selection is uniform.

			The retained-uniform assumption inside _realized_rho_for_beta is only
			approximate for degradation (carving out missing nodes makes the
			retained per-bin counts non-uniform under a strong tilt), so the
			realized tau-b is verified by the end-of-pipeline 3-prior gate, not
			trusted from this solve. :ceiling_hit is propagated so the caller can
			honor the up-front feasibility decision rather than chase an
			unreachable rho.

			Bins are rank-equal (~N/K per bin); bin 1 is most peripheral, bin K
			most central, matching reconstruction's higher-bin = higher-centrality
			convention.
		"""

		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes, got $n"))
			(-1.0 < target_rho < 1.0) ||
				throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))
			(0.0 < pi_node < 1.0) ||
				throw(ArgumentError("pi_node must be in (0, 1), got $pi_node"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))
			K <= n || throw(ArgumentError("K must be <= N = $n, got $K"))

		#	Assign Centrality-Rank Bins
			#	Sort node indices by centrality ascending; the rank_pos-th
			#	lowest-centrality node lands in bin ceil(rank_pos * K / n),
			#	giving ~N/K nodes per bin to match the analytic tau-b's
			#	uniform-observed assumption.
				order = sortperm(collect(centrality))
				bin_index = Vector{Int}(undef, n)
				@inbounds for rank_pos in 1:n
					node = order[rank_pos]
					bin_index[node] = Int(cld(rank_pos * K, n))
				end

		#	Solve the Logistic Tilt (reused from network_reconstruction)
			#	(retained, missing) = (N - k, k) map onto reconstruction's
			#	(N, N_add); q is the target per-bin distribution of the missing
			#	set; status carries the ceiling decision to the caller.
				k = clamp(Int(round(pi_node * n)), 1, n - 1)
				solve  = _solve_bin_distribution(target_rho, K, n - k, k)
				q      = solve.q
				beta   = solve.beta
				status = solve.status

		#	Map q to a Per-Node Relative Propensity (node-mean 1)
			#	d_i = K * q[bin_i]: with ~N/K nodes per bin the node-mean is
			#	sum_b (N/K)/N * (K*q_b) = sum_b q_b = 1. At beta = 0, q_b = 1/K
			#	so d_i = 1 for all i (constant -> MCAR).
				d = Vector{Float64}(undef, n)
				@inbounds for i in 1:n
					d[i] = K * q[bin_index[i]]
				end

		#	Return
			return (d = d, q = q, beta = beta, bin_index = bin_index, status = status)
	end

#	Helper Node Top-Up: conditional rho-correlated selection given an organic-loss prefix
	function _topup_missing_nodes(centrality::AbstractVector{<:Real},
									organic_losses::AbstractVector{<:Integer},
									pi_node::Real,
									target_rho::Real,
									sample_seed::Integer;
									K::Int = 4)
		"""
		Args:
			centrality::AbstractVector{<:Real}: per-node centrality, canonical order.
			organic_losses::AbstractVector{<:Integer}: node indices already lost to
				the edge stage (the fixed prefix). May be empty (pi_edge = 0 case).
			pi_node::Real: target TOTAL missing-node fraction; in (0, 1).
			target_rho::Real: target Kendall tau-b of the combined missing set vs
				centrality; in (-1, 1).
			sample_seed::Integer: seed for the supplementary draw.
			K::Int: number of rank bins for the propensity field (default 4).
		Returns:
			NamedTuple: (missing_nodes, n_organic, n_topup)
				- missing_nodes::Vector{Int}: combined missing set (organic + top-up),
					sorted ascending, of size round(pi_node*N).
				- n_organic::Int: number from the edge stage.
				- n_topup::Int: number added here.
		Notes:
			The node-accounting stage. The organic losses are FIXED; this selects
			the additional nodes so the combined set hits round(pi_node*N) with the
			rho-tilt, drawing from the same logistic-beta propensity field
			(_solve_propensity_field) the edge stage used. Because the organic set
			is peripheral, at rho > 0 the top-up must add central nodes
			aggressively to flip the combined correlation positive — exactly where
			the up-front feasibility envelope earns its keep; this function does
			not itself enforce feasibility, it just draws toward the target and
			lets the end gate verify.

			When organic_losses is empty (pi_edge = 0), this is the full node
			selection and reduces to a logistic-beta analogue of the old
			_sample_missingness (one field, not the b-mixing prob vector).

			The supplementary nodes are drawn weighted-without-replacement from the
			NON-organic nodes by per-node propensity d. The sign of target_rho is
			carried INSIDE d: _solve_propensity_field solves on the signed target, so
			beta (hence q, hence d = K*q[bin]) already puts the larger d on the
			preferred end — high-centrality nodes for rho > 0, low-centrality nodes
			for rho < 0. The draw therefore weights by d directly for either sign;
			high-d nodes are preferred regardless of sign. (Applying max(d) - d on
			the negative branch re-inverts the sign already in d — the double-flip
			that this version removes.)

			Determinism in sample_seed.
		"""

		#	Guards
			n = length(centrality)
			(0.0 < pi_node < 1.0) || throw(ArgumentError("pi_node must be in (0, 1), got $pi_node"))
			k_total = clamp(Int(round(pi_node * n)), 1, n - 1)

		#	Organic prefix as a mask
			is_missing = falses(n)
			@inbounds for v in organic_losses
				(1 <= v <= n) || throw(ArgumentError("organic loss index $v out of range"))
				is_missing[v] = true
			end
			n_organic = count(is_missing)

		#	Already at/over target: trim deterministically is out of scope; accept
		#	the organic set as-is when it meets or exceeds the node budget. The end
		#	gate reports the realized proportion.
			if n_organic >= k_total
				missing_nodes = sort!(findall(is_missing))
				return (missing_nodes = missing_nodes, n_organic = n_organic, n_topup = 0)
			end

		#	Propensity field for the supplementary draw
			field = _solve_propensity_field(centrality, target_rho, pi_node, K)
			d = field.d

		#	Candidate (non-organic) nodes and their weights
			cand = [v for v in 1:n if !is_missing[v]]
			k_add = k_total - n_organic
			k_add = min(k_add, length(cand))
			#	Weight by the propensity d directly for BOTH signs of target_rho.
			#	_solve_propensity_field solves on the SIGNED target, so beta — and
			#	therefore q and d = K*q[bin] — already carries the sign: for rho > 0
			#	the high-centrality nodes get the larger d, for rho < 0 the
			#	low-centrality nodes do. d is thus always larger on the preferred
			#	end, so weighting by d directly realizes the target correlation.
			#	(Applying max(d) - d on the negative branch would re-invert the sign
			#	already in d — the double-flip bug.) d > 0 everywhere (softmax q > 0),
			#	so every candidate carries strictly positive mass.
				wts = Vector{Float64}(undef, length(cand))
				@inbounds for (t, v) in enumerate(cand)
					wts[t] = d[v]
				end

		#	Weighted-without-replacement supplementary draw
			rng = Xoshiro(sample_seed)
			n_pos = count(>(0.0), wts)
			if n_pos >= k_add
				added = StatsBase.sample(rng, cand, StatsBase.Weights(wts), k_add; replace=false)
			else
				#	Saturation fallback: take all positive-weight candidates, fill
				#	the rest uniformly from the zero-weight remainder. With d > 0
				#	everywhere this path is currently unreachable, but it is retained
				#	defensively in case a future field admits zero-propensity nodes.
					pos_idx = [cand[t] for t in eachindex(cand) if wts[t] > 0.0]
					zero_idx = [cand[t] for t in eachindex(cand) if wts[t] <= 0.0]
					k_rem = k_add - length(pos_idx)
					filler = StatsBase.sample(rng, zero_idx, max(k_rem, 0); replace=false)
					added = vcat(pos_idx, filler)
			end
			@inbounds for v in added
				is_missing[v] = true
			end

		#	Return combined set
			missing_nodes = sort!(findall(is_missing))
			return (missing_nodes = missing_nodes,
					n_organic = n_organic,
					n_topup = length(added))
	end

#	feasible_rho_range: raw-space tau-b envelope via the shared selection core
	function feasible_rho_range(edges::DataFrame,
								nodes::DataFrame;
								directed::Bool,
								weighted::Bool,
								target_rate::Float64,
								K::Int = 4,
								n_mc_replicates::Int = 20,
								seed::Integer = 1)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, optional :weight.
			nodes::DataFrame: node roster with :id (and :label) columns.
			directed::Bool: selects the centrality driver (in-degree vs degree).
			weighted::Bool: VESTIGIAL — the rho basis is binarized centrality, which
				is weight-independent. Retained for caller-signature symmetry.
			target_rate::Float64: missingness rate at which feasibility is assessed,
				in (0, 1). This is the pi_node the missing set is drawn to.
			K::Int: number of FIELD rank bins (default 4, matching the degeneration
				mask). NOT the reconstruction setup degree-bin K; feasibility is a
				property of the field mechanism and is independent of the setup K.
			n_mc_replicates::Int: MC draws per endpoint (default 20).
			seed::Integer: master seed; per-(endpoint, replicate) seeds via hash.
		Returns:
			NamedTuple:
				rho_min, rho_max, rho_mcar::Float64: mean realized Kendall tau-b at
					the negative-saturated, positive-saturated, and MCAR draws.
				rho_min_std, rho_max_std, rho_mcar_std::Float64: MC std errors.
				target_rate::Float64: echoed.
				is_asymmetric::Bool, asymmetry_ratio::Float64: ceiling-asymmetry
					diagnostic (|rho_min| / |rho_max|).
				diagnostics::Dict{Symbol,Any}: K_used, n_mc, seed, rho_probe,
					centrality_summary.
		Notes:
			Computes the feasible range of centrality-missingness correlation (rho,
			Kendall tau-b) for this network at the given rate. RAW-SPACE: it draws an
			actual missing-node set via the shared selection core and measures
			corkendall(indicator, raw_centrality) — the same correlation the end gate
			scores — so the reported ceilings are in the metric the user observes,
			not the solver's bin-index proxy.

			MECHANISM. For each endpoint it calls _topup_missing_nodes with
			organic_losses = Int[] (pure node selection) at an extreme target_rho:
			    +RHO_PROBE -> positive-saturated draw -> rho_max
			     0.0       -> MCAR draw               -> rho_mcar (~ 0)
			    -RHO_PROBE -> negative-saturated draw -> rho_min
			RHO_PROBE sits just inside 1.0 so the field's beta saturates at the
			bisection bound (the solve returns :ceiling_hit at the bound beta),
			giving the TRUE structural ceiling rather than a milder fixed-beta probe.
			Each endpoint is averaged over n_mc_replicates draws. No degraded network
			is materialized, no end gate runs, and there is no call into degeneracy.

			KENDALL TAU-B AS THE METRIC. rho is Kendall's tau-b, the rank-based
			correlation between the binary missingness indicator and node centrality.
			This matches the user's rank-order intuition and avoids the asymmetric-
			ceiling pathology Pearson exhibits on heavy-tailed networks.

			TWO CEILING CONSTRAINTS under tau-b on a binary indicator vs continuous:
			- Rate-bounded absolute ceiling |tau-b| <= sqrt(2 * p * (1 - p)),
			  depending only on the rate p (~ 0.42 at p = 0.10).
			- Practical tie-bounded ceiling, lower on networks with centrality ties:
			  ties enter tau-b's denominator and cap the achievable correlation.
			  This is what the probe actually measures.
			Under Kendall the ceilings are typically symmetric in sign; is_asymmetric
			(|rho_min|/|rho_max| outside [0.5, 2.0]) rarely fires and is retained as a
			diagnostic for unusual small/concentrated networks.

			DEGENERATE CENTRALITY. If centrality is constant (no variance), tau-b is
			undefined; returns rho_min = rho_max = rho_mcar = 0, is_asymmetric = false.

			K is the FIELD bin count; feasibility does not depend on the setup K, so
			find_optimal_K checks it once before its K-search.
		"""

		#	Guards
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))
			0.0 < target_rate < 1.0 ||
				throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			n_mc_replicates >= 1 ||
				throw(ArgumentError("n_mc_replicates must be >= 1, got $n_mc_replicates"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))

		#	Cache Centrality Once (rho basis: in-degree directed / degree undirected)
			centrality = _centrality_for_sampler(edges; nodes = nodes, directed = directed)
			n = length(centrality)
			K_eff = clamp(K, 2, max(2, n))

		#	Degenerate Centrality: tau-b Undefined
			if all(==(centrality[1]), centrality)
				return (
					rho_min = 0.0, rho_max = 0.0, rho_mcar = 0.0,
					rho_min_std = NaN, rho_max_std = NaN, rho_mcar_std = NaN,
					target_rate = target_rate,
					is_asymmetric = false, asymmetry_ratio = 1.0,
					diagnostics = Dict{Symbol,Any}(
						:K_used => K_eff, :n_mc => n_mc_replicates, :seed => seed,
						:degenerate_centrality => true),
				)
			end

		#	Probe Three Endpoints via the Selection Core
			RHO_PROBE = 0.9999
			endpoints = ((-RHO_PROBE, :rho_min), (0.0, :rho_mcar), (RHO_PROBE, :rho_max))
			rho_estimates = Dict{Symbol, Vector{Float64}}(
				:rho_min => Float64[], :rho_mcar => Float64[], :rho_max => Float64[])

			for (rho_target, lab) in endpoints
				for rep in 1:n_mc_replicates
					rep_seed = Int(hash((lab, rep, seed)) % UInt32)
					sel = _topup_missing_nodes(centrality, Int[], target_rate,
												rho_target, rep_seed; K = K_eff)
					indicator = zeros(Float64, n)
					@inbounds for v in sel.missing_nodes
						indicator[v] = 1.0
					end
					push!(rho_estimates[lab], StatsBase.corkendall(indicator, centrality))
				end
			end

		#	Aggregate Per-Endpoint Mean and Std
			rho_min_mean  = mean(rho_estimates[:rho_min])
			rho_max_mean  = mean(rho_estimates[:rho_max])
			rho_mcar_mean = mean(rho_estimates[:rho_mcar])

			rho_min_std  = n_mc_replicates >= 2 ? std(rho_estimates[:rho_min])  : NaN
			rho_max_std  = n_mc_replicates >= 2 ? std(rho_estimates[:rho_max])  : NaN
			rho_mcar_std = n_mc_replicates >= 2 ? std(rho_estimates[:rho_mcar]) : NaN

		#	Asymmetry Diagnostic
			is_degenerate = abs(rho_min_mean) < 0.01 && abs(rho_max_mean) < 0.01
			asymmetry_ratio = if is_degenerate
				1.0
			else
				denom = abs(rho_max_mean)
				denom > 0 ? abs(rho_min_mean) / denom : Inf
			end
			is_asymmetric = !is_degenerate &&
							(asymmetry_ratio < 0.5 || asymmetry_ratio > 2.0)

		#	Centrality Summary for Diagnostics
			centrality_summary = (
				min  = minimum(centrality),
				max  = maximum(centrality),
				mean = mean(centrality),
				median = median(centrality),
				std  = std(centrality),
			)

		#	Diagnostics Dict
			diagnostics = Dict{Symbol, Any}(
				:K_used             => K_eff,
				:n_mc               => n_mc_replicates,
				:seed               => seed,
				:rho_probe          => RHO_PROBE,
				:centrality_summary => centrality_summary,
			)

		#	Return Feasibility NamedTuple
			return (
				rho_min = rho_min_mean, rho_max = rho_max_mean, rho_mcar = rho_mcar_mean,
				rho_min_std = rho_min_std, rho_max_std = rho_max_std, rho_mcar_std = rho_mcar_std,
				target_rate = target_rate,
				is_asymmetric = is_asymmetric, asymmetry_ratio = asymmetry_ratio,
				diagnostics = diagnostics,
			)
	end
	@doc raw"""
	**Description**
	Estimate the feasible range of centrality--missingness correlation $\rho$
	(Kendall's $\tau_b$) for a network at a given missingness rate, in
	raw-centrality space. Draws a missing-node set via the shared selection core at
	saturated $\pm\rho$ and at MCAR, and measures the realized $\tau_b$ on raw
	centrality.

	**Usage**
	`feasible_rho_range(edges, nodes; directed, weighted, target_rate, K=4, n_mc_replicates=20, seed=1)`

	**Arguments**
	- `edges::DataFrame`, `nodes::DataFrame`: network (`nodes` needs `:id`).
	- `directed::Bool`: selects the centrality driver.
	- `weighted::Bool`: vestigial; the $\rho$ basis is binarized centrality.
	- `target_rate::Float64`: missingness rate assessed, in $(0,1)$.
	- `K::Int`: field bin count (default 4); independent of the setup degree-bin $K$.
	- `n_mc_replicates::Int`, `seed::Integer`: MC draws per endpoint and master seed.

	**Value**
	NamedTuple with `rho_min`, `rho_max`, `rho_mcar` (means), their MC std errors,
	`target_rate`, `is_asymmetric`, `asymmetry_ratio`, and a `diagnostics` Dict.

	**Details**
	Self-contained on reconstruction's selection core (`_topup_missing_nodes` with
	an empty organic prefix): no degraded network is materialized, no gate runs, and
	there is no call into degeneracy. The endpoints push `target_rho` just inside
	$\pm 1$ so the field's $\beta$ saturates at the bisection bound, giving the true
	structural ceiling. Reported in raw $\tau_b$ — the metric the end gate scores —
	rather than the solver's bin-index proxy.

	**See Also**
	`_topup_missing_nodes`, `_solve_propensity_field`, `find_optimal_K`,
	`generate_missingness_mask`
	""" feasible_rho_range

#	find_optimal_K: select the largest valid degree-bin count
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
							 feasibility_seed::Integer = 1,
							 verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: standard compute_setup inputs.
			directed::Bool: whether the network is directed.
			weighted::Bool: whether the network has meaningful edge weights.
			pi_node::Float64: user's requested proportion-missing prior.
			rho::Float64: user's centrality-missingness correlation prior.
			partially_observed_nodes::Vector{Int}: nominated non-respondents.
			K_max::Int: maximum K to consider (default 0 = auto:
				min(20, floor(N_obs / min_nodes_per_degree_bin))).
			K_min::Int: minimum K to consider (default 4).
			J::Int: number of E/I bins (default = DEFAULT_J).
			min_nodes_per_ei_bin::Int: minimum nodes per E/I bin.
			min_nodes_per_degree_bin::Int: minimum nodes per degree bin.
			feasibility_n_mc::Int: MC draws per endpoint for feasibility.
			feasibility_seed::Integer: master seed for feasibility.
			verbose::Bool: print diagnostics.
		Returns:
			NamedTuple with fields:
				optimal_K::Int
				test_results::Vector
				stopping_reason::Symbol
				feasibility::NamedTuple
		Notes:
			Uses the realized node-missing rate implied by pi_node and nominations.
			If nominations impose a node-missingness floor, the feasibility probe and
			Prior 1 use the realized rate rather than the raw requested pi_node.
			When the realized rate is 0, rho must also be approximately 0; otherwise
			the requested correlation is infeasible because there is no missing set.
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
			(directed || isempty(partially_observed_nodes)) ||
				throw(ArgumentError("undirected networks do not support nominated non-respondents"))

		#	Compute Realized Node-Missingness Rate
			N_obs = nrow(nodes) - length(partially_observed_nodes)
			N_nom = length(partially_observed_nodes)
			N_add_target = _determine_n_add(pi_node, N_obs, N_nom)
			realized_total = N_obs + N_nom + N_add_target
			realized_pi_node = realized_total > 0 ?
				(N_nom + N_add_target) / realized_total : 0.0

		#	Step 1: Feasibility Precondition
			feas = if realized_pi_node == 0.0
				(rho_min = 0.0, rho_max = 0.0, target_rate = 0.0)
			else
				feasible_rho_range(
					edges, nodes;
					directed = directed,
					weighted = weighted,
					target_rate = realized_pi_node,
					n_mc_replicates = feasibility_n_mc,
					seed = feasibility_seed)
			end

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

		#	Step 2: Resolve K_max
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
				println("  N_obs = $N_obs, N_nom = $N_nom, N_add = $N_add_target")
				println("  requested pi_node = $pi_node, realized pi_node = $realized_pi_node")
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

				#	Call compute_setup at this K
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

				#	Prior 1: realized missing fraction matches realized target
					realized_rate = setup.diagnostics[:realized_pi_node]
					prior_1_ok = abs(realized_rate - realized_pi_node) < 0.02

				#	Prior 2: beta_status must be acceptable
					prior_2_ok = setup.beta_status in (:converged, :ceiling_hit)

				#	Prior 3: ei_conditional row-stochastic and empirically matched
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

				#	Record Result
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
				if !isempty(test_results) && test_results[end].K == K_min
					stopping_reason = :K_min_reached
				end
			end

		#	Return Search Result
			return (
				optimal_K       = optimal_K,
				test_results    = test_results,
				stopping_reason = stopping_reason,
				feasibility     = feas,
			)
	end
	@doc raw"""
	**Description**
	Find the largest $K$ (number of degree bins) such that `compute_setup`
	successfully satisfies the reconstruction requirements for a given network,
	requested `pi_node`, and requested $\rho$. Returns the selected $K$, per-$K$
	diagnostics, and the feasibility precondition result.

	**Usage**
	`find_optimal_K(edges, nodes, community_labels; directed, weighted, pi_node, rho, partially_observed_nodes=[], K_max=0, K_min=4, J=DEFAULT_J, ...)`

	**Arguments**

	* `edges::DataFrame`, `nodes::DataFrame`, `community_labels::Vector{Int}`:
	standard `compute_setup` inputs.
	* `directed::Bool`, `weighted::Bool`: network type.
	* `pi_node::Float64`: requested missing-node fraction. The realized value may
	be higher when nominated non-respondents impose a node-missingness floor.
	* `rho::Float64`: requested Kendall $\tau_b$ correlation between missingness
	and centrality.
	* `partially_observed_nodes::Vector{Int}`: nominated non-respondents
	(default `[]`); directed networks only.
	* `K_max::Int`: maximum $K$. `0` triggers the automatic rule
	$\min(20, \lfloor N_{\text{obs}} / \text{min_nodes_per_degree_bin} \rfloor)$.
	* `K_min::Int`, `J::Int`, `min_nodes_per_ei_bin::Int`,
	`min_nodes_per_degree_bin::Int`: binning bounds.
	* `feasibility_n_mc::Int`, `feasibility_seed::Integer`: configuration for the
	feasibility precondition probe.
	* `verbose::Bool`: print per-$K$ diagnostics.

	**Details**
	The function first computes the realized node-missing rate implied by the
	requested `pi_node`, the nominated non-respondent count, and the resulting
	`N_add`. This realized rate is used for the feasibility precondition and for
	the missingness-rate validation check. If the realized rate is zero, the
	feasibility range is degenerate at zero; therefore nonzero $\rho$ is infeasible
	because there is no missing set with which to correlate centrality.

	If the requested $\rho$ is outside the achievable range returned by
	`feasible_rho_range`, the function returns immediately with
	`stopping_reason = :rho_infeasible`. Otherwise it searches from the largest
	allowed $K$ down to `K_min`, calling `compute_setup` at each candidate. The
	first $K$ whose setup passes all validation checks is returned.

	The per-$K$ validation checks are:

	1. The realized missing fraction matches the realized target rate.
	2. `beta_status` is either `:converged` or `:ceiling_hit`.
	3. `ei_conditional` is row-stochastic and matches the empirical conditional
	distribution implied by the setup bins.

	The third check is an internal consistency check on the E/I conditional
	construction rather than a user-specified prior.

	**Value**
	A `NamedTuple` with fields:

	* `optimal_K::Int`: selected degree-bin count, or `-1` if no valid value is found.
	* `test_results::Vector`: per-candidate diagnostics in descending $K$ order.
	* `stopping_reason::Symbol`: `:found`, `:rho_infeasible`, `:no_valid_K`, or
	`:K_min_reached`.
	* `feasibility::NamedTuple`: feasibility range used before the $K$ search.

	**See Also**
	`compute_setup`, `feasible_rho_range`
	""" find_optimal_K

#	_compute_bin_tendencies: per-degree-bin observed degree/strength/out-split tendencies
	function _compute_bin_tendencies(edges::DataFrame,
									  nodes::DataFrame,
									  degree_bins::Vector{Int},
									  K::Int,
									  directed::Bool,
									  weighted::Bool;
									  partially_observed::Vector{Int} = Int[])
		"""
		Args:
			edges::DataFrame: observed edge list with :src, :dst, optional :weight.
			nodes::DataFrame: node roster with :id; row order is canonical and
				matches degree_bins.
			degree_bins::Vector{Int}: per-node degree bin (1..K) from
				_bin_observed_nodes, nodes-DataFrame row order.
			K::Int: number of degree bins.
			directed::Bool: directed => out tendencies populated; undirected =>
				out tendencies are nothing.
			weighted::Bool: weighted => strength sums edge :weight; unweighted =>
				every edge contributes strength 1 (so strength == degree).
			partially_observed::Vector{Int}: nominated non-respondent indices to
				EXCLUDE; their outgoing ties are unobserved, so they cannot inform
				out tendencies (mirrors _compute_p_matrix's respondent-only rule).
		Returns:
			NamedTuple (bin_exp_degree, bin_exp_strength, bin_exp_out_strength,
						bin_out_fraction):
				bin_exp_degree::Vector{Float64}: length K, mean total binarized
					degree (in + out) of respondents per bin — the expected tie
					COUNT for a synthetic node placed in that bin.
				bin_exp_strength::Vector{Float64}: length K, mean total incident
					strength per bin (== bin_exp_degree when unweighted) — the
					expected total strength LOAD.
				bin_exp_out_strength::Union{Nothing,Vector{Float64}}: length K, mean
					out-strength per bin (directed); nothing (undirected).
				bin_out_fraction::Union{Nothing,Vector{Float64}}: length K, bin
					aggregate out-degree fraction = sum(out_deg)/sum(total_deg)
					(directed); nothing (undirected). Splits a synthetic add's tie
					count into outgoing vs incoming.
		Notes:
			Setup-phase tendency builder feeding the implied-weight floor and Stage 1.
			Computed over RESPONDENTS only. Each value is a per-bin MEAN over observed
			nodes in that bin: a synthetic node placed in bin k inherits the bin's
			mean degree (count), out-fraction (split), and mean strength (load).

			Total (in + out) degree drives the tie COUNT and bin_out_fraction does
			the directional split. This is distinct from the in-degree rho basis,
			which governs only which bin a node occupies, not how many ties it has.
			In + out also yields the correct degree/strength for undirected input
			stored once per edge (each endpoint accrues one incident edge).

			These split into the two floor terms downstream: nominations contribute
			E[out-strength | bin] (in roster, incoming observed, outgoing imputed),
			full synthetic adds contribute E[total strength | bin].

			EMPTY/SPARSE BIN FALLBACK. A bin with no respondents takes the global
			(all-respondent) means and the global aggregate out-fraction, so a
			synthetic node placed in an unobserved bin still gets a sensible nonzero
			tendency instead of 0.
		"""

		#	Guards
			n = nrow(nodes)
			length(degree_bins) == n ||
				throw(ArgumentError("degree_bins length $(length(degree_bins)) must equal nrow(nodes) $n"))
			K >= 2 || throw(ArgumentError("K must be >= 2, got $K"))

		#	Respondent mask (exclude nominated / partially observed)
			is_partial = falses(n)
			@inbounds for idx in partially_observed
				(1 <= idx <= n) ||
					throw(ArgumentError("partially_observed index $idx out of range [1, $n]"))
				is_partial[idx] = true
			end

		#	Node ID -> index map
			id_to_idx = Dict(String(id) => i for (i, id) in enumerate(nodes.id))

		#	Per-Node Degree Counts and Strength Sums
			in_deg  = zeros(Int, n)
			out_deg = zeros(Int, n)
			in_str  = zeros(Float64, n)
			out_str = zeros(Float64, n)

			src_ids = String.(edges.src)
			dst_ids = String.(edges.dst)
			n_edges = length(src_ids)
			has_w = weighted && hasproperty(edges, :weight)

			@inbounds for e in 1:n_edges
				si = get(id_to_idx, src_ids[e], 0)
				di = get(id_to_idx, dst_ids[e], 0)
				(si == 0 || di == 0) && continue
				w = has_w ? Float64(edges.weight[e]) : 1.0
				out_deg[si] += 1
				out_str[si] += w
				in_deg[di]  += 1
				in_str[di]  += w
			end

		#	Aggregate per Bin over Respondents (+ global accumulators for fallback)
			bin_count      = zeros(Int, K)
			bin_deg_sum    = zeros(Float64, K)
			bin_str_sum    = zeros(Float64, K)
			bin_outdeg_sum = zeros(Float64, K)
			bin_outstr_sum = zeros(Float64, K)

			g_count = 0
			g_deg = 0.0; g_str = 0.0; g_outdeg = 0.0; g_outstr = 0.0

			@inbounds for i in 1:n
				is_partial[i] && continue
				b = degree_bins[i]
				tdeg = Float64(in_deg[i] + out_deg[i])
				tstr = in_str[i] + out_str[i]
				bin_count[b]      += 1
				bin_deg_sum[b]    += tdeg
				bin_str_sum[b]    += tstr
				bin_outdeg_sum[b] += Float64(out_deg[i])
				bin_outstr_sum[b] += out_str[i]
				g_count  += 1
				g_deg    += tdeg;            g_str    += tstr
				g_outdeg += Float64(out_deg[i]); g_outstr += out_str[i]
			end

		#	Global Means / Aggregate Out-Fraction (empty-bin fallback)
			g_exp_deg    = g_count > 0 ? g_deg    / g_count : 0.0
			g_exp_str    = g_count > 0 ? g_str    / g_count : 0.0
			g_exp_outstr = g_count > 0 ? g_outstr / g_count : 0.0
			g_out_frac   = g_deg   > 0 ? g_outdeg / g_deg   : 0.0

		#	Per-Bin Tendencies with Empty-Bin Fallback
			bin_exp_degree         = Vector{Float64}(undef, K)
			bin_exp_strength       = Vector{Float64}(undef, K)
			bin_exp_out_strength_v = Vector{Float64}(undef, K)
			bin_out_fraction_v     = Vector{Float64}(undef, K)

			@inbounds for k in 1:K
				if bin_count[k] > 0
					bin_exp_degree[k]         = bin_deg_sum[k] / bin_count[k]
					bin_exp_strength[k]       = bin_str_sum[k] / bin_count[k]
					bin_exp_out_strength_v[k] = bin_outstr_sum[k] / bin_count[k]
					bin_out_fraction_v[k]     = bin_deg_sum[k] > 0 ?
						bin_outdeg_sum[k] / bin_deg_sum[k] : g_out_frac
				else
					bin_exp_degree[k]         = g_exp_deg
					bin_exp_strength[k]       = g_exp_str
					bin_exp_out_strength_v[k] = g_exp_outstr
					bin_out_fraction_v[k]     = g_out_frac
				end
			end

		#	Directed-Only Out Tendencies
			bin_exp_out_strength = directed ? bin_exp_out_strength_v : nothing
			bin_out_fraction     = directed ? bin_out_fraction_v     : nothing

		#	Return
			return (bin_exp_degree       = bin_exp_degree,
					bin_exp_strength     = bin_exp_strength,
					bin_exp_out_strength = bin_exp_out_strength,
					bin_out_fraction     = bin_out_fraction)
	end

#	_compute_weight_floor: implied-minimum added weight A (binary-tie count), additional weight B, floored pi_edge
	function _compute_weight_floor(edges::DataFrame,
									weighted::Bool,
									directed::Bool,
									q::Vector{Float64},
									N_add::Int,
									degree_bins::Vector{Int},
									partially_observed::Vector{Int},
									bin_exp_degree::Vector{Float64},
									bin_out_fraction::Union{Nothing,Vector{Float64}},
									pi_edge::Float64)
		"""
		Args:
			edges::DataFrame: observed edge list with :src, :dst, optional :weight.
			weighted::Bool: weighted => W_observed sums :weight; unweighted =>
				W_observed = edge count.
			directed::Bool: directed => nominations contribute their out-tie count;
				undirected => no nomination term.
			q::Vector{Float64}: length-K setup distribution over degree bins. Expected
				full-add count in bin k is N_add * q[k], matching _draw_bin_assignments.
			N_add::Int: number of full synthetic adds.
			degree_bins::Vector{Int}: per-node degree bin (1..K), nodes-row order;
				used to read each nominated node's bin.
			partially_observed::Vector{Int}: nominated non-respondent indices.
			bin_exp_degree::Vector{Float64}: length-K mean total binarized degree per
				bin — the expected tie COUNT a synthetic node placed there receives.
			bin_out_fraction::Union{Nothing,Vector{Float64}}: length-K out-degree
				fraction per bin (directed); nothing (undirected). Splits a node's tie
				count into outgoing vs incoming.
			pi_edge::Float64: user's target missing-weight fraction of the TRUE
				network, in [0, 1).
		Returns:
			NamedTuple:
				W_observed::Float64: total observed weight.
				implied_min_weight::Float64: A — the weight needed to bring the
					imputed ties into existence, one unit per binary tie, so A is the
					expected imputed-EDGE COUNT.
				additional_weight::Float64: B — diffuse weight Stage 2 redistributes
					on top of A; 0 when pi_edge is at/below the floor.
				pi_edge_floor::Float64: implied-pi_edge floor A / (W_observed + A).
				realized_pi_edge::Float64: pi_edge clamped up to pi_edge_floor when
					the user's value is below it.
				W_true::Float64: W_observed + A + B = W_observed / (1 - realized_pi_edge).
				below_floor::Bool: true when pi_edge was below the floor and clamped.
		Notes:
			The weight-accounting closure W_observed + A + B = W_true. A is the
			conservative floor: every imputed tie enters at weight 1 (missing ties are
			presumed weak, else well-conducted sampling would not have missed them),
			so the minimum weight to materialize the structure is just the imputed-
			edge count. q places the N_add full adds (expected N_add*q[k] in bin k),
			each contributing bin_exp_degree[k] ties; each nomination contributes its
			out-ties, bin_exp_degree[bin]*bin_out_fraction[bin]. Because the floor is
			one-unit-per-edge, A equals exactly the weight Stage 0.5/1 lay down — no
			estimator gap between reported floor and realized placement.

			pi_edge is a FLOORED prior. If the implied missing weight
			M_w = W_observed * pi_edge / (1 - pi_edge) is below A, the target is
			infeasible; pi_edge is clamped to pi_edge_floor (B = 0, below_floor = true).
			Otherwise B = M_w - A, and Stage 2 redistributes B across existing and
			imputed edges by the rho-field (the inverse of _sample_weight_removal).

			Undirected: no nomination term; A = full-add tie count only.
		"""

		#	W_observed: Total Observed Weight
			has_w = weighted && hasproperty(edges, :weight)
			W_observed = has_w ? sum(Float64.(edges.weight)) : Float64(nrow(edges))

		#	A_full: Expected Full-Add Tie Count = N_add * sum_k q[k] * bin_exp_degree[k]
			K = length(q)
			length(bin_exp_degree) == K ||
				throw(ArgumentError("bin_exp_degree length must equal length(q) = $K"))
			A_full = 0.0
			@inbounds for k in 1:K
				A_full += q[k] * bin_exp_degree[k]
			end
			A_full *= N_add

		#	A_nom: Nominations Contribute Their Out-Tie Count (directed)
			A_nom = 0.0
			if directed && !isempty(partially_observed)
				bin_out_fraction === nothing &&
					throw(ArgumentError("directed network with nominations requires bin_out_fraction"))
				@inbounds for j in partially_observed
					b = degree_bins[j]
					A_nom += bin_exp_degree[b] * bin_out_fraction[b]
				end
			end

		#	A: Implied Minimum Added Weight (one unit per imputed binary tie)
			A = A_full + A_nom

		#	Implied-pi_edge Floor and Additional Weight B
			pi_edge_floor = (W_observed + A) > 0 ? A / (W_observed + A) : 0.0

			below_floor      = pi_edge < pi_edge_floor
			realized_pi_edge = below_floor ? pi_edge_floor : pi_edge

			M_w = realized_pi_edge < 1.0 ?
				W_observed * realized_pi_edge / (1.0 - realized_pi_edge) : Inf
			B = max(M_w - A, 0.0)

		#	W_true Closes the Accounting
			W_true = W_observed + A + B

		#	Return
			return (W_observed         = W_observed,
					implied_min_weight  = A,
					additional_weight   = B,
					pi_edge_floor       = pi_edge_floor,
					realized_pi_edge    = realized_pi_edge,
					W_true              = W_true,
					below_floor         = below_floor)
	end

#	Helper Function for compute_setup: aggregate duplicate dyads
	function _aggregate_duplicate_dyads(edges::DataFrame, directed::Bool, weighted::Bool)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst, and optional :weight.
			directed::Bool: whether dyad orientation is meaningful.
			weighted::Bool: whether to preserve and sum edge weights.
		Returns:
			DataFrame: edge list with one row per canonical dyad. The :src/:dst
				endpoint type is PRESERVED from the input (not coerced to String).
		Notes:
			Endpoint id type is preserved rather than stringified, so the aggregated
			edge list stays homogeneous with the node roster's id type. This keeps
			setup.edges, the augmented roster, the synthetic-node ids, and the
			Stage-1 edge endpoints all ONE type — the homogeneity the replicate
			(_stage_*), _extract_reconstruction_delta, and materialize_reconstruction
			stages depend on (they key parent vs imputed edges by raw endpoint
			tuples, not stringified ones). Assumes edges.src/.dst and nodes.id share
			one id type, which holds when both originate from load_graphml (String)
			per the package convention.

			Directed dyads keep orientation. Undirected dyads are canonicalized so
			A--B and B--A collapse to one row, putting the smaller id first under the
			id type's natural order — matching the (s <= d ? (s,d) : (d,s)) canonical
			key used by _extract_reconstruction_delta and materialize_reconstruction.
		"""

		#	Guards
			hasproperty(edges, :src) && hasproperty(edges, :dst) ||
				throw(ArgumentError("edges must have :src and :dst columns"))

		#	Build working edge table (preserve endpoint id type; weights -> Float64)
			work = DataFrame(
				src = copy(edges.src),
				dst = copy(edges.dst),
				weight = (weighted && hasproperty(edges, :weight)) ?
					Float64.(edges.weight) : ones(Float64, nrow(edges))
			)

		#	Canonicalize undirected endpoints (smaller id first, id type's order)
			if !directed
				@inbounds for i in 1:nrow(work)
					if work.src[i] > work.dst[i]
						work.src[i], work.dst[i] = work.dst[i], work.src[i]
					end
				end
			end

		#	Aggregate duplicate dyads (sum weights within each canonical dyad)
			agg = combine(groupby(work, [:src, :dst]), :weight => sum => :weight)

		#	Return aggregated edges
			return agg
	end

#	compute_setup: assemble the full SamplerSetup for an observed network (Phase 2 setup)
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
							allocation::Symbol = :observed,
							J::Int = DEFAULT_J,
							min_nodes_per_ei_bin::Int = 3,
							min_nodes_per_degree_bin::Int = 5,
							verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: observed-network inputs (labels
				precomputed by Phase 1.5, nodes-row order).
			directed::Bool, weighted::Bool: network type.
			pi_node::Float64: fraction of TRUE nodes believed missing, [0, 1); the
				free node dial (default 0.0). At 0.0 no synthetic nodes are added.
			pi_edge::Float64: fraction of TRUE weight believed missing, [0, 1); the
				FLOORED dial — raised to the implied-weight floor when the user's
				value sits below what the node additions already commit (default 0.0).
			rho::Float64: target Kendall tau-b between the missing set and centrality,
				(-1, 1), measured over the TRUE network: present = respondents,
				missing = nominated + added (default 0.0).
			partially_observed_nodes::Vector{Int}: nominated non-respondent indices.
				DIRECTED ONLY — a non-empty vector on an undirected network is
				rejected, since undirected missingness is full removal (Stage 1), not
				nomination.
			K::Union{Int,Symbol}: degree-bin count, or :auto (calls find_optimal_K,
				except when pi_node = 0; see Notes).
			allocation::Symbol: Stage-2 weight-allocation mode, :observed or :deficit
				(default :observed). :observed distributes the additional-weight budget
				B proportional to (current edge weight * rho-tilt) — the Bellutta
				proportional-to-current inverse. :deficit distributes B proportional to
				the estimated per-edge DEFICIT implied by the rho-field and pi_edge, so
				expected reconstructed weight targets estimated-true rather than
				observed weight. The two coincide at rho = 0. Stored on the setup and
				read by _stage_2!; validated here, not used elsewhere in setup.
			J::Int, min_nodes_per_ei_bin::Int, min_nodes_per_degree_bin::Int: binning.
			verbose::Bool: print diagnostics (default false).
		Returns:
			SamplerSetup: the full setup bundle.
		Notes:
			Assembly point for the setup phase.

			Edge endpoints (:src, :dst) and the node roster (:id) must share one id
			type, which is validated up front. Per the package convention (ids flow
			through load_graphml as String) this is String, but any single id type
			works: it is preserved through canonicalization, so setup.edges, the
			augmented roster, the synthetic-node ids, and the imputed edge endpoints
			all stay one type — the homogeneity the replicate and delta stages rely on.

			Duplicate dyads are canonicalized before setup construction. Directed
			dyads keep orientation; undirected dyads are endpoint-sorted and
			aggregated. All downstream quantities are computed from this one-row-per-
			canonical-dyad edge list.
		"""

		#	Guards
			hasproperty(edges, :src) && hasproperty(edges, :dst) ||
				throw(ArgumentError("edges must have :src and :dst columns"))
			hasproperty(nodes, :id) ||
				throw(ArgumentError("nodes must have :id column"))

			#	Id-Type Homogeneity (edge endpoints must match the roster id type)
				eltype(edges.src) == eltype(edges.dst) ||
					throw(ArgumentError(
						"edges :src and :dst must share an id type, got " *
						"$(eltype(edges.src)) and $(eltype(edges.dst))"))
				(eltype(edges.src) <: AbstractString && eltype(nodes.id) <: AbstractString) ||
				(eltype(edges.src) <: Integer && eltype(nodes.id) <: Integer) ||
				eltype(edges.src) == eltype(nodes.id) ||
					throw(ArgumentError(
						"edge endpoint id type $(eltype(edges.src)) must match node roster " *
						"id type $(eltype(nodes.id)); both should be String per the package " *
						"convention (ids arrive via load_graphml)"))

			length(community_labels) == nrow(nodes) ||
				throw(ArgumentError("community_labels length must equal nrow(nodes)"))
			0.0 <= pi_node < 1.0 ||
				throw(ArgumentError("pi_node must be in [0, 1), got $pi_node"))
			0.0 <= pi_edge < 1.0 ||
				throw(ArgumentError("pi_edge must be in [0, 1), got $pi_edge"))
			-1.0 < rho < 1.0 ||
				throw(ArgumentError("rho must be in (-1, 1), got $rho"))
			J >= 1 || throw(ArgumentError("J must be >= 1, got $J"))
			(directed || isempty(partially_observed_nodes)) ||
				throw(ArgumentError(
					"undirected networks do not support nominated non-respondents " *
					"(partially_observed_nodes): Stage 0.5 imputes missing outgoing ties, " *
					"which has no undirected analogue. Pass partially_observed_nodes only " *
					"when directed = true."))

		#	Checking Weight Allocation Input
			allocation in (:observed, :deficit) ||
				throw(ArgumentError("allocation must be :observed or :deficit, got $allocation"))

		#	Canonicalize Duplicate Dyads
			edges = _aggregate_duplicate_dyads(edges, directed, weighted)

		#	Resolve K
			K_resolved = if K == :auto
				if pi_node == 0.0
					#	No synthetic adds
						N_obs_k = nrow(nodes) - length(partially_observed_nodes)
						k_auto = max(2, min(20, fld(N_obs_k, min_nodes_per_degree_bin)))
						if verbose
							println("compute_setup: pi_node = 0 (no node additions); " *
									"skipping feasibility search, K = $k_auto.")
						end
						k_auto
				else
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
				end
			else
				K::Int
			end

			K_resolved >= 2 ||
				throw(ArgumentError("K must be >= 2, got $K_resolved"))

		#	Initialize Diagnostics
			diag = Dict{Symbol, Any}()
			diag[:K_used] = K_resolved

		#	Step 1: Observed Centrality
			centrality = _compute_observed_centrality(edges, nodes, directed)

		#	Step 1.5: Detect Community Structure
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

		#	Step 2.5: Per-Bin Tendencies
			tend = _compute_bin_tendencies(edges, nodes, degree_bins, K_resolved,
											directed, weighted;
											partially_observed = partially_observed_nodes)

		#	Step 3: P and w Matrix
			pw = _compute_p_matrix(edges, nodes, degree_bins, ei_bins,
								   K_resolved, J_effective, directed, weighted;
								   partially_observed = partially_observed_nodes)
			P = pw.P
			w = pw.w

		#	Step 3.5: R Matrix
			R = directed ?
				_compute_r_matrix(edges, nodes, degree_bins, ei_bins,
								  K_resolved, J_effective;
								  partially_observed = partially_observed_nodes) :
				nothing

		#	Step 4: Determine N_add
			N_obs = nrow(nodes) - length(partially_observed_nodes)
			N_nom = length(partially_observed_nodes)
			N_add = _determine_n_add(pi_node, N_obs, N_nom)

		#	Node-Missingness Floor
			node_total = N_obs + N_nom
			diag[:pi_node_floor] = node_total > 0 ? N_nom / node_total : 0.0
			diag[:pi_node_raised] = pi_node < diag[:pi_node_floor]

			realized_total = N_obs + N_nom + N_add
			diag[:realized_pi_node] = realized_total > 0 ?
				(N_nom + N_add) / realized_total : 0.0

		#	Step 5: Solve Beta / q
			beta_result = _solve_bin_distribution(rho, K_resolved, N_obs, N_nom + N_add)
			beta = beta_result.beta
			q = beta_result.q
			beta_status = beta_result.status
			diag[:beta_status] = beta_status
			diag[:beta_n_iters] = beta_result.n_iters

		#	Step 5.5: Per-Node Propensity d and Conditional E/I Distribution
			d = K_resolved .* q[degree_bins]
			ei_conditional = _compute_ei_conditional(degree_bins, ei_bins,
													  K_resolved, J_effective)

		#	Step 6: Weight Floor
			floor_result = _compute_weight_floor(edges, weighted, directed, q, N_add,
												  degree_bins, partially_observed_nodes,
												  tend.bin_exp_degree,
												  tend.bin_out_fraction, pi_edge)
			diag[:W_observed] = floor_result.W_observed
			diag[:implied_min_weight] = floor_result.implied_min_weight
			diag[:additional_weight] = floor_result.additional_weight
			diag[:pi_edge_floor] = floor_result.pi_edge_floor
			diag[:pi_edge_raised] = floor_result.below_floor
			diag[:W_true] = floor_result.W_true

			if verbose && floor_result.below_floor
				println("compute_setup: pi_edge = $pi_edge is below the implied floor " *
						"$(round(floor_result.pi_edge_floor, digits=4)); raised to the floor " *
						"(additional_weight = 0). A = $(round(floor_result.implied_min_weight, digits=3)).")
			end

		#	Assemble SamplerSetup
			return SamplerSetup(
				edges,
				nodes,
				directed,
				weighted,
				diag[:realized_pi_node],
				rho,
				floor_result.realized_pi_edge,
				allocation,
				centrality,
				community_labels,
				ei_values,
				binning_mode,
				degree_bins,
				ei_bins,
				K_resolved,
				J_effective,
				beta,
				beta_status,
				q,
				d,
				tend.bin_exp_degree,
				tend.bin_exp_strength,
				tend.bin_exp_out_strength,
				tend.bin_out_fraction,
				P,
				w,
				R,
				partially_observed_nodes,
				N_add,
				ei_conditional,
				floor_result.W_observed,
				floor_result.implied_min_weight,
				floor_result.additional_weight,
				diag,
			)
	end
	@doc raw"""
	**Description**
	Build the full `SamplerSetup` for an observed network: per-node centrality and
	binning, community/EI structure, the shared $\rho$-field ($q$, $d$), the per-bin
	degree/strength tendencies, the $P$/$w$/$R$ attachment matrices, the added-node
	and nomination specification, the chosen weight-allocation mode, and the weight
	accounting $W_{\text{obs}} + A + B = W_{\text{true}}$. Computed once per network
	and reused across replicates.

	**Usage**
	`compute_setup(edges, nodes, community_labels; directed, weighted, pi_node=0, pi_edge=0, rho=0, K=:auto, allocation=:observed, ...)`

	**Arguments**

	* `edges`, `nodes`, `community_labels`: observed network and precomputed labels.
	  Edge endpoints (`:src`, `:dst`) and node ids (`:id`) must share one id type
	  (see Details).
	* `directed`, `weighted`: network type.
	* `pi_node`: target fraction of true nodes believed missing. This is the free
	  node dial supplied by the user; the realized value may be higher when the
	  observed nomination count already implies a larger missing fraction.
	* `rho`: shared Kendall $\tau_b$ correlation between missingness and centrality.
	  Measured over the true network, where present = respondents and missing =
	  nominated non-respondents + fully synthetic additions.
	* `pi_edge`: target fraction of true edge weight believed missing. This is a
	  floored dial and may be raised when the node additions already imply more
	  missing weight than the requested value allows.
	* `partially_observed_nodes`: nominated non-respondents; directed networks only.
	* `K`: degree-bin count or `:auto` (selected by `find_optimal_K`, except when
	  `pi_node = 0`; see Details).
	* `allocation`: Stage-2 weight-allocation mode, `:observed` (default) or
	  `:deficit`. Controls how the additional-weight budget is distributed across
	  edges; see Weight allocation under Details. Validated here and carried on the
	  returned setup for `_stage_2!`.

	**Details**
	Edge endpoints and node ids must share a single id type, validated before
	canonicalization. By the package convention this is `String` (ids arrive via
	`load_graphml`), but any one id type is accepted; it is preserved through
	canonicalization, so the edge list, augmented roster, synthetic-node ids, and
	imputed edges remain homogeneous — the property the replicate and delta stages
	depend on.

	Duplicate dyads are canonicalized before any setup quantities are computed.
	For directed networks, duplicate `(src,dst)` rows are aggregated. For undirected
	networks, `(A,B)` and `(B,A)` are first collapsed to the same canonical dyad.

	When `pi_node = 0` no fully synthetic nodes are added, so the `K = :auto` path
	skips the `find_optimal_K` feasibility search (which is undefined at
	`target_rate = 0`) and applies the same automatic bin-selection rule directly.
	Nominations, if present, still impute missing outgoing ties through Stage 0.5.
	With `pi_node = 0` and no nominations there is no missing set, so a nonzero
	`rho` cannot be realized: it is recorded as `beta_status = :ceiling_hit` rather
	than raised, because the feasibility check that would otherwise reject it is
	skipped on this path.

	Nominated non-respondents imply a minimum achievable missing-node fraction of

	`N_nom / (N_obs + N_nom)`.

	If the requested `pi_node` falls below this floor, no additional synthetic nodes
	are added (`N_add = 0`) and the realized missing-node fraction exceeds the
	target. The realized value and the nomination-implied floor are recorded in
	`diagnostics[:realized_pi_node]` and `diagnostics[:pi_node_floor]`.

	Similarly, `pi_edge` is constrained by the implied-weight floor generated by the
	node additions and nomination imputation. If the requested `pi_edge` falls below
	that floor, it is raised automatically and the adjustment is recorded in
	`diagnostics[:pi_edge_raised]`.

	*Weight allocation.* `allocation` selects how Stage 2 distributes the additional
	weight budget $B$. `:observed` (the Bellutta inverse) distributes $B$ in
	proportion to current edge weight times the $\rho$-tilt, anchoring recovery to
	where weight is presently observed. `:deficit` estimates each edge's removed
	fraction from the $\rho$-field and `pi_edge` and distributes $B$ in proportion to
	the implied per-edge deficit, so the expected reconstructed weight targets an
	estimated *true* weight rather than the observed weight; this restores
	heavily-depleted high-weight structure that `:observed` under-serves at
	$\rho \neq 0$. The two modes coincide exactly at $\rho = 0$, where the flat field
	makes the per-edge gross-up a global constant. The choice is stored on the setup
	and applied by `_stage_2!`; it does not affect any other setup quantity.

	Supplying `partially_observed_nodes` on an undirected network is rejected,
	because undirected missingness is modeled as full node removal rather than
	missing outgoing nominations.

	**Value**
	A `SamplerSetup`. The stored `pi_node` and `pi_edge` values are the realized
	values used by the reconstruction process, and `allocation` is the mode applied
	by `_stage_2!`. The `diagnostics` field records the node and weight floors,
	realized missing fractions, `beta_status`, binning mode, fallback behavior, and
	the estimated true-network weight.

	**See Also**
	`SamplerSetup`, `find_optimal_K`, `feasible_rho_range`, `generate_replicate`
	""" compute_setup

#############################################
#    NOMINATION & NODE ADDITION FUNCTIONS   #
#############################################

#	_stage_0_5_directed: impute a nominated non-respondent's missing outgoing ties (directed)
	function _stage_0_5_directed(setup::SamplerSetup,
								  augmented_nodes::DataFrame,
								  rng::AbstractRNG)
		"""
		Args:
			setup::SamplerSetup: must have non-empty setup.partially_observed,
				directed == true, and a non-nothing R matrix.
			augmented_nodes::DataFrame: from _build_augmented_nodes; carries
				:degree_bin, :ei_bin, :node_type, :id for every row.
			rng::AbstractRNG: random source.
		Returns:
			DataFrame of new edges added by Stage 0.5 (columns :src, :dst, :weight).
		Notes:
			Stage 0.5 of a replicate — the inverse of degeneration's
			apply_missingness_outgoing_only. A nominated non-respondent is in the
			roster with its INCOMING ties observed and its OUTGOING ties missing;
			this restores the missing outgoing ties from the observed attachment
			structure. (Full synthetic adds, both directions missing, are Stage 1.)

			Three sub-stages:
			(a) Reverse edges: for each observed A -> E with E nominated, draw
			    Bernoulli(R[c_A, c_E]) for the reverse E -> A.
			(b) Outgoing to non-nominators: for each respondent A with no observed
			    tie to E in either direction, draw Bernoulli(P[c_E, c_A]) for E -> A.
			(c) Pairs of nominated (E, F): draw E -> F independently per ordered pair.

			Weight is the conservative floor: every imputed tie enters at weight 1
			(a binary tie). Missing ties are presumed weak — well-conducted sampling
			would not have missed a strong one — so Stage 0.5 commits only the single
			unit that makes each tie present, and the implied-weight floor A is just
			the imputed-edge count. The diffuse additional weight B is layered on
			later in Stage 2, NOT here.

			Edge buffers are pre-allocated at the combined upper bound (forward-to-nom
			+ n_nom*n_resp + n_nom*(n_nom-1)); the counter advances only on successful
			draws and resize! truncates at the end. Edge id type is inferred from
			eltype(augmented_nodes.id).
		"""

		#	Guards
			setup.directed ||
				throw(ArgumentError("_stage_0_5_directed called on undirected setup"))
			!isempty(setup.partially_observed) ||
				throw(ArgumentError("_stage_0_5_directed requires non-empty setup.partially_observed"))
			!isnothing(setup.R) ||
				throw(ArgumentError("_stage_0_5_directed requires non-nothing R matrix"))

		#	Identify Nominated Non-Respondents and Their Cells
			nominated_idx = findall(==(:nominated), augmented_nodes.node_type)
			respondent_idx = findall(==(:observed), augmented_nodes.node_type)
			nominated_ids = Set(augmented_nodes.id[nominated_idx])
			id_to_idx = Dict(augmented_nodes.id[i] => i for i in 1:nrow(augmented_nodes))

		#	Build Per-Nominee Connected-Set and Count Forward Edges to Nominees
			edges_with_nom = Dict{eltype(augmented_nodes.id), Set{eltype(augmented_nodes.id)}}(
				E => Set{eltype(augmented_nodes.id)}() for E in nominated_ids)
			n_forward_to_nom = 0
			for row in eachrow(setup.edges)
				if row.src in nominated_ids
					push!(edges_with_nom[row.src], row.dst)
				end
				if row.dst in nominated_ids
					push!(edges_with_nom[row.dst], row.src)
					n_forward_to_nom += 1
				end
			end

		#	Pre-Allocate Edge Buffers at Upper Bound
			id_type = eltype(augmented_nodes.id)
			n_nom = length(nominated_idx)
			n_resp = length(respondent_idx)
			upper_bound = n_forward_to_nom + n_nom * n_resp + n_nom * (n_nom - 1)
			src_buf = Vector{id_type}(undef, upper_bound)
			dst_buf = Vector{id_type}(undef, upper_bound)
			weight_buf = setup.weighted ? Vector{Float64}(undef, upper_bound) : Vector{Int}(undef, upper_bound)
			n_filled = 0

		#	Stage 0.5a: Reverse Edges via R Matrix
			for row in eachrow(setup.edges)
				if !(row.dst in nominated_ids)
					continue
				end
				A_idx = id_to_idx[row.src]
				E_idx = id_to_idx[row.dst]
				dA = augmented_nodes.degree_bin[A_idx]
				eA = augmented_nodes.ei_bin[A_idx]
				dE = augmented_nodes.degree_bin[E_idx]
				eE = augmented_nodes.ei_bin[E_idx]
				p = setup.R[dA, eA, dE, eE]
				if rand(rng) < p
					n_filled += 1
					src_buf[n_filled] = row.dst
					dst_buf[n_filled] = row.src
					weight_buf[n_filled] = 1
				end
			end

		#	Stage 0.5b: Outgoing Edges to Non-Nominator Respondents via P
			for E_idx in nominated_idx
				E_id = augmented_nodes.id[E_idx]
				dE = augmented_nodes.degree_bin[E_idx]
				eE = augmented_nodes.ei_bin[E_idx]
				connected = edges_with_nom[E_id]
				for A_idx in respondent_idx
					A_id = augmented_nodes.id[A_idx]
					if A_id in connected
						continue
					end
					dA = augmented_nodes.degree_bin[A_idx]
					eA = augmented_nodes.ei_bin[A_idx]
					p = setup.P[dE, eE, dA, eA]
					if rand(rng) < p
						n_filled += 1
						src_buf[n_filled] = E_id
						dst_buf[n_filled] = A_id
						weight_buf[n_filled] = 1
					end
				end
			end

		#	Stage 0.5c: Edges Between Pairs of Nominated Non-Respondents
			for i in eachindex(nominated_idx)
				E_idx = nominated_idx[i]
				E_id = augmented_nodes.id[E_idx]
				dE = augmented_nodes.degree_bin[E_idx]
				eE = augmented_nodes.ei_bin[E_idx]
				for j in eachindex(nominated_idx)
					if i == j
						continue
					end
					F_idx = nominated_idx[j]
					F_id = augmented_nodes.id[F_idx]
					dF = augmented_nodes.degree_bin[F_idx]
					eF = augmented_nodes.ei_bin[F_idx]
					p = setup.P[dE, eE, dF, eF]
					if rand(rng) < p
						n_filled += 1
						src_buf[n_filled] = E_id
						dst_buf[n_filled] = F_id
						weight_buf[n_filled] = 1
					end
				end
			end

		#	Truncate Buffers to Filled Length and Return
			resize!(src_buf, n_filled)
			resize!(dst_buf, n_filled)
			resize!(weight_buf, n_filled)
			return DataFrame(src = src_buf, dst = dst_buf, weight = weight_buf)
	end

#	_stage_0_5_undirected: undirected has no nominations — Stage 0.5 is directed-only (guard)
	function _stage_0_5_undirected(setup::SamplerSetup,
									augmented_nodes::DataFrame,
									rng::AbstractRNG)
		"""
		Args:
			setup::SamplerSetup: must have directed == false.
			augmented_nodes::DataFrame: unused; present for dispatch symmetry with
				_stage_0_5_directed.
			rng::AbstractRNG: unused; present for signature symmetry.
		Returns:
			DataFrame (src, dst, weight) — only on the no-nomination path, an empty
				frame; otherwise throws.
		Notes:
			Stage 0.5 imputes a nominated non-respondent's missing OUTGOING ties, which
			is an inherently directed notion (incoming observed, outgoing missing).
			Undirected networks have no in/out asymmetry, so under the unified spec an
			undirected missing node is always a FULL REMOVAL (recovered by Stage 1),
			never a nomination — setup.partially_observed should be empty. This branch
			is therefore reachable only on incoherent input (undirected + nominations),
			and it fails loudly rather than fabricating undirected-nomination behavior.

			If partially_observed is empty it returns an empty edge frame so a stray
			dispatch is harmless; a non-empty partially_observed on an undirected setup
			throws. (Better still, reject undirected + nominations up front in
			compute_setup so this is never entered.)
		"""

		#	Guards
			!setup.directed ||
				throw(ArgumentError("_stage_0_5_undirected called on directed setup"))

		#	Undirected Has No Nominations
			if !isempty(setup.partially_observed)
				throw(ArgumentError(
					"undirected networks do not support nominated non-respondents: " *
					"Stage 0.5 imputes missing OUTGOING ties, which has no undirected " *
					"analogue. Undirected missing nodes are full removals (Stage 1). " *
					"Pass partially_observed_nodes only for directed networks."))
			end

		#	No-Nomination Path: Empty Edge Frame (matches the directed return schema)
			id_type = eltype(augmented_nodes.id)
			wbuf = setup.weighted ? Float64[] : Int[]
			return DataFrame(src = id_type[], dst = id_type[], weight = wbuf)
	end

#	_stage_1_directed: inject a full synthetic node's ties (directed) — counts from tendencies, partners from P
	function _stage_1_directed(setup::SamplerSetup,
								augmented_nodes::DataFrame,
								rng::AbstractRNG)
		"""
		Args:
			setup::SamplerSetup: directed == true. Uses setup.P, setup.bin_exp_degree,
				setup.bin_out_fraction.
			augmented_nodes::DataFrame: from _build_augmented_nodes; carries
				:degree_bin, :ei_bin, :node_type, :id for every row.
			rng::AbstractRNG: random source.
		Returns:
			DataFrame of new edges added by Stage 1 (columns :src, :dst, :weight).
		Notes:
			Stage 1 of a replicate — the inverse of degeneration's full node removal.
			For each FULL synthetic add (node_type :added, both directions missing) it
			materializes the ties needed to bring the node into the network.

			Counts come from the per-bin tendencies, NOT from P (the binary-era
			behavior, where an added node's degree was an accident of summed Bernoulli
			mass):
			- out-tie count m_out ~ Poisson(bin_exp_degree[dv] * bin_out_fraction[dv])
			- in-tie  count m_in  ~ Poisson(bin_exp_degree[dv] * (1 - bin_out_fraction[dv]))
			so a node placed in bin dv receives that bin's characteristic out/in
			degree. P then decides WHICH partners: m_out targets sampled without
			replacement weighted by P[dv,ev, .] over all other nodes; m_in sources
			sampled weighted by P[., dv,ev] over NON-ADDED nodes only.

			Double counting. In-ties are drawn from non-added sources only; an
			added -> added edge is materialized solely through the source node's
			out-selection, so each ordered added-added pair is considered once.

			Weight is the conservative floor: every imputed tie enters at weight 1 (a
			binary tie). Missing ties are presumed weak, so Stage 1 commits only the
			unit that makes each tie present; the diffuse additional weight B is
			redistributed in Stage 2, NOT here. A Poisson count may exceed the
			candidate pool; it is clamped to the number of available partners.

			Buffers pre-allocated at the per-node maximum and resize!d at the end;
			edge id type inferred from eltype(augmented_nodes.id).
		"""

		#	Guards
			setup.directed ||
				throw(ArgumentError("_stage_1_directed called on undirected setup"))
			setup.bin_out_fraction === nothing &&
				throw(ArgumentError("_stage_1_directed requires non-nothing bin_out_fraction"))

		#	Identify Added Nodes and Candidate Pools
			added_idx = findall(==(:added), augmented_nodes.node_type)
			non_added_idx = findall(!=(:added), augmented_nodes.node_type)
			n_total = nrow(augmented_nodes)
			n_added = length(added_idx)
			n_non_added = length(non_added_idx)

			deg = augmented_nodes.degree_bin
			eib = augmented_nodes.ei_bin
			ids = augmented_nodes.id
			id_type = eltype(ids)

		#	Fast-Path: No Added Nodes
			if n_added == 0
				wbuf = setup.weighted ? Float64[] : Int[]
				return DataFrame(src = id_type[], dst = id_type[], weight = wbuf)
			end

		#	Pre-Allocate Edge Buffers at Upper Bound
			upper_bound = n_added * ((n_total - 1) + n_non_added)
			src_buf = Vector{id_type}(undef, upper_bound)
			dst_buf = Vector{id_type}(undef, upper_bound)
			weight_buf = setup.weighted ? Vector{Float64}(undef, upper_bound) : Vector{Int}(undef, upper_bound)
			n_filled = 0

		#	Inject Each Added Node's Ties
			for v_idx in added_idx
				dv = deg[v_idx]
				ev = eib[v_idx]
				v_id = ids[v_idx]

				#	Counts from tendencies, split by out-fraction
					exp_deg  = setup.bin_exp_degree[dv]
					out_frac = setup.bin_out_fraction[dv]
					m_out = rand(rng, Poisson(exp_deg * out_frac))
					m_in  = rand(rng, Poisson(exp_deg * (1.0 - out_frac)))

				#	Out-Ties: m_out targets from all other nodes, weighted by P[v, .]
					if m_out > 0
						cand = [j for j in 1:n_total if j != v_idx]
						if !isempty(cand)
							wts = [setup.P[dv, ev, deg[j], eib[j]] for j in cand]
							k = min(m_out, length(cand))
							targets = StatsBase.sample(rng, cand, StatsBase.Weights(wts), k; replace = false)
							@inbounds for j in targets
								n_filled += 1
								src_buf[n_filled] = v_id
								dst_buf[n_filled] = ids[j]
								weight_buf[n_filled] = 1
							end
						end
					end

				#	In-Ties: m_in sources from NON-ADDED nodes, weighted by P[., v]
					if m_in > 0 && n_non_added > 0
						wts = [setup.P[deg[i], eib[i], dv, ev] for i in non_added_idx]
						k = min(m_in, n_non_added)
						sources = StatsBase.sample(rng, non_added_idx, StatsBase.Weights(wts), k; replace = false)
						@inbounds for i in sources
							n_filled += 1
							src_buf[n_filled] = ids[i]
							dst_buf[n_filled] = v_id
							weight_buf[n_filled] = 1
						end
					end
			end

		#	Truncate Buffers to Filled Length and Return
			resize!(src_buf, n_filled)
			resize!(dst_buf, n_filled)
			resize!(weight_buf, n_filled)
			return DataFrame(src = src_buf, dst = dst_buf, weight = weight_buf)
	end

#	_stage_1_undirected: inject a full synthetic node's ties (undirected) — count from tendencies, partners from P
	function _stage_1_undirected(setup::SamplerSetup,
								  augmented_nodes::DataFrame,
								  rng::AbstractRNG)
		"""
		Args:
			setup::SamplerSetup: directed == false. Uses setup.P, setup.bin_exp_degree.
			augmented_nodes::DataFrame: from _build_augmented_nodes; carries
				:degree_bin, :ei_bin, :node_type, :id for every row.
			rng::AbstractRNG: random source.
		Returns:
			DataFrame of new undirected edges (columns :src, :dst, :weight), stored
				with src < dst.
		Notes:
			Stage 1 of a replicate, undirected case — the inverse of degeneration's
			full node removal, and the split-free counterpart of _stage_1_directed.
			Undirected nodes have a single degree, so there is no in/out split and
			bin_out_fraction is not consulted.

			Count from the tendency, NOT from P (the binary-era behavior): tie count
			m ~ Poisson(bin_exp_degree[dv]), so a node placed in bin dv receives that
			bin's characteristic degree. P decides WHICH partners: m partners sampled
			without replacement weighted by P[dv,ev, .] over all other nodes.

			Double counting. An undirected added-added edge {v, u} can be proposed by
			both endpoints' draws; a seen-set of unordered pairs keeps it once (so an
			added node's realized degree may fall just short of m when a partner was
			already linked by the other node's draw). Added-to-non-added pairs never
			collide, since non-added nodes are not iterated as sources.

			Weight is the conservative floor: every imputed tie enters at weight 1 (a
			binary tie); the diffuse additional weight B is redistributed in Stage 2,
			NOT here. A Poisson count may exceed the candidate pool; it is clamped to
			the number of available partners.

			Buffers pre-allocated at the per-node maximum and resize!d; edges stored
			with src < dst; edge id type inferred from eltype(augmented_nodes.id).
		"""

		#	Guards
			!setup.directed ||
				throw(ArgumentError("_stage_1_undirected called on directed setup"))

		#	Identify Added Nodes
			added_idx = findall(==(:added), augmented_nodes.node_type)
			n_total = nrow(augmented_nodes)
			n_added = length(added_idx)

			deg = augmented_nodes.degree_bin
			eib = augmented_nodes.ei_bin
			ids = augmented_nodes.id
			id_type = eltype(ids)

		#	Fast-Path: No Added Nodes
			if n_added == 0
				wbuf = setup.weighted ? Float64[] : Int[]
				return DataFrame(src = id_type[], dst = id_type[], weight = wbuf)
			end

		#	Pre-Allocate Edge Buffers at Upper Bound
			upper_bound = n_added * (n_total - 1)
			src_buf = Vector{id_type}(undef, upper_bound)
			dst_buf = Vector{id_type}(undef, upper_bound)
			weight_buf = setup.weighted ? Vector{Float64}(undef, upper_bound) : Vector{Int}(undef, upper_bound)
			n_filled = 0

		#	Seen-Set of Unordered Pairs (dedup added-added)
			seen = Set{Tuple{id_type, id_type}}()

		#	Inject Each Added Node's Ties
			for v_idx in added_idx
				dv = deg[v_idx]
				ev = eib[v_idx]
				v_id = ids[v_idx]

				#	Count from tendency (undirected: single degree, no split)
					m = rand(rng, Poisson(setup.bin_exp_degree[dv]))
					m == 0 && continue

				#	Partners: m from all other nodes, weighted by P[v, .]
					cand = [j for j in 1:n_total if j != v_idx]
					isempty(cand) && continue
					wts = [setup.P[dv, ev, deg[j], eib[j]] for j in cand]
					k = min(m, length(cand))
					partners = StatsBase.sample(rng, cand, StatsBase.Weights(wts), k; replace = false)

				#	Place Each Edge Once (canonical src < dst, dedup via seen-set)
					@inbounds for j in partners
						u_id = ids[j]
						a, b = v_id < u_id ? (v_id, u_id) : (u_id, v_id)
						key = (a, b)
						key in seen && continue
						push!(seen, key)
						n_filled += 1
						src_buf[n_filled] = a
						dst_buf[n_filled] = b
						weight_buf[n_filled] = 1
					end
			end

		#	Truncate Buffers to Filled Length and Return
			resize!(src_buf, n_filled)
			resize!(dst_buf, n_filled)
			resize!(weight_buf, n_filled)
			return DataFrame(src = src_buf, dst = dst_buf, weight = weight_buf)
	end

#	_draw_bin_assignments: sample degree + E/I bins for the N_add synthetic added nodes
	function _draw_bin_assignments(q::Vector{Float64},
									ei_conditional::Matrix{Float64},
									N_add::Int,
									binning_mode::Symbol,
									rng::AbstractRNG)
		"""
		Args:
			q::Vector{Float64}: rho-governed per-bin distribution over the K degree
				bins (length K, sums to 1).
			ei_conditional::Matrix{Float64}: K x J, row-stochastic; row k is
				P(E/I bin | degree bin k).
			N_add::Int: number of synthetic added nodes to assign (>= 0).
			binning_mode::Symbol: :two_dimensional or :degree_only. In :degree_only
				every E/I bin is 1.
			rng::AbstractRNG: random source.
		Returns:
			NamedTuple (degree_bins::Vector{Int}, ei_bins::Vector{Int}), each length
				N_add.
		Notes:
			The two-axis draw feeding Stage 1: degree bin first from q (the centrality
			axis the rho-field governs), then E/I bin from ei_conditional[degree_bin, :]
			(community role conditional on degree), so a synthetic node inherits a
			realistic embeddedness for its centrality level. Sampling in that order
			keeps rho governing centrality alone while preserving the observed
			degree-embeddedness association.

			Weighted draws use StatsBase.sample over Weights (no strict sum-to-1
			requirement), matching the rest of the module. N_add == 0 returns empty
			vectors.
		"""

		#	Guards
			K = length(q)
			K >= 2 || throw(ArgumentError("q must have length >= 2, got $K"))
			size(ei_conditional, 1) == K ||
				throw(ArgumentError("ei_conditional must have K = $K rows, got $(size(ei_conditional, 1))"))
			N_add >= 0 || throw(ArgumentError("N_add must be >= 0, got $N_add"))

		#	Draw Each Added Node's Bins
			J = size(ei_conditional, 2)
			degree_bins = Vector{Int}(undef, N_add)
			ei_bins     = Vector{Int}(undef, N_add)
			qw = StatsBase.Weights(q)

			@inbounds for i in 1:N_add
				d = StatsBase.sample(rng, 1:K, qw)
				degree_bins[i] = d
				ei_bins[i] = binning_mode == :degree_only ?
					1 :
					StatsBase.sample(rng, 1:J, StatsBase.Weights(ei_conditional[d, :]))
			end

		#	Return
			return (degree_bins = degree_bins, ei_bins = ei_bins)
	end

#	_build_augmented_nodes: roster of observed + nominated + synthetic added nodes, with bin/type tags
	function _build_augmented_nodes(setup::SamplerSetup,
									 degree_bins_added::Vector{Int},
									 ei_bins_added::Vector{Int})
		"""
		Args:
			setup::SamplerSetup: supplies nodes, degree_bins, ei_bins,
				partially_observed, N_add.
			degree_bins_added::Vector{Int}: drawn degree bins for the N_add synthetic
				nodes (from _draw_bin_assignments); length must equal setup.N_add.
			ei_bins_added::Vector{Int}: drawn E/I bins for the synthetic nodes; same
				length.
		Returns:
			DataFrame: all original setup.nodes columns, plus :degree_bin::Int,
				:ei_bin::Int, :node_type::Symbol. node_type is :observed (respondent),
				:nominated (index in setup.partially_observed), or :added (synthetic).
		Notes:
			The augmented roster. Existing rows keep their setup degree/EI bins and are
			tagged :observed or :nominated; N_add synthetic rows are appended tagged
			:added with their drawn bins and original columns set to missing apart from
			:id.

			SYNTHETIC IDS. Added-node ids are generated to MATCH eltype(setup.nodes.id)
			so the :id column — and every augmented edge endpoint, and the
			reconstruction-corpus id_t — stays homogeneous. Collision-free by
			construction: integer rosters take (max existing id) + 1 .. (max existing
			id) + N_add, which exceeds every existing id regardless of contiguity;
			string rosters take a reserved "syn_<k>" prefix, checked against the
			existing id set (and the ids already issued) so a pre-existing "syn_<k>"
			cannot collide. Edge endpoints (setup.edges.src/dst) must share the
			node-id type for the downstream id matches to work.
		"""

		#	Guards
			n_obs = nrow(setup.nodes)
			N_add = setup.N_add
			length(degree_bins_added) == N_add ||
				throw(ArgumentError("degree_bins_added length $(length(degree_bins_added)) != N_add $N_add"))
			length(ei_bins_added) == N_add ||
				throw(ArgumentError("ei_bins_added length $(length(ei_bins_added)) != N_add $N_add"))
			hasproperty(setup.nodes, :id) ||
				throw(ArgumentError("setup.nodes must have an :id column"))

		#	Synthetic Added Ids (collision-free, matching the existing id type)
			id_t = eltype(setup.nodes.id)
			added_ids = Vector{id_t}(undef, N_add)
			if id_t <: Integer
				#	Offset off the max existing id: base + i exceeds every existing
				#	id, so collisions are impossible regardless of contiguity.
					base = isempty(setup.nodes.id) ? zero(id_t) : maximum(setup.nodes.id)
					@inbounds for i in 1:N_add
						added_ids[i] = base + id_t(i)
					end
			elseif id_t <: AbstractString
				#	Reserved "syn_" prefix, checked against existing ids and the ones
				#	already issued so a pre-existing "syn_<k>" cannot collide.
					used = Set{String}(string.(setup.nodes.id))
					counter = 0
					@inbounds for i in 1:N_add
						local cand
						while true
							counter += 1
							cand = "syn_" * string(counter)
							cand in used || break
						end
						push!(used, cand)
						added_ids[i] = id_t(cand)
					end
			else
				throw(ArgumentError("node id type $id_t unsupported; expected Integer or AbstractString"))
			end

		#	Start From the Observed Roster; Append Added Rows (id set, others missing)
			aug = copy(setup.nodes)
			cols = Symbol.(names(aug))
			@inbounds for aid in added_ids
				push!(aug, Any[c === :id ? aid : missing for c in cols]; promote = true)
			end

		#	Tag node_type and Attach the Bin Columns
			is_nom = falses(n_obs)
			@inbounds for idx in setup.partially_observed
				(1 <= idx <= n_obs) && (is_nom[idx] = true)
			end

			aug.degree_bin = vcat(setup.degree_bins, degree_bins_added)
			aug.ei_bin     = vcat(setup.ei_bins, ei_bins_added)
			aug.node_type  = vcat([is_nom[i] ? :nominated : :observed for i in 1:n_obs],
								   fill(:added, N_add))

		#	Return
			return aug
	end

#############################
#    EDGE REDISTRIBUTION    #
#############################

#	_stage_2!: redistribute the additional weight B across edges by the rho-field (mutating)
	function _stage_2!(setup::SamplerSetup,
						augmented_edges::DataFrame,
						augmented_nodes::DataFrame,
						rng::AbstractRNG)
		"""
		Args:
			setup::SamplerSetup: weighted == true; supplies additional_weight (B),
				the per-bin q, K for the rho-field, and the allocation mode.
			augmented_edges::DataFrame: edges after Stages 0.5 and 1 (:src, :dst,
				:weight); mutated in place.
			augmented_nodes::DataFrame: supplies :degree_bin and :id for every node,
				including added nodes, so the per-node rho-tilt covers imputed edges.
			rng::AbstractRNG: random source.
		Returns:
			Int: number of additional weight units distributed (= round(B)).
		Notes:
			Stage 2 -- the inverse of degeneration's _sample_weight_removal. Removal
			took W_removed units OFF edges by an ITERATIVE draw with per-edge weight
			(current-available weight * endpoint tilt), clipping at each edge's
			remaining weight and redistributing overflow -- a proportional-hazard
			depletion. The budget B = setup.additional_weight (the weight floor) is
			distributed so W_observed + A + B = W_true closes; round(B) <= 0 is a no-op.

			Per-edge tilt. d_v = K*q[degree_bin_v] (node-mean 1); s_v = d_v/(2*max d)
			in [0,0.5]; an edge's tilt is 1 - (1-s_i)(1-s_j). At rho = 0 the field is
			constant and tilt is uniform.

			Allocation modes (setup.allocation):
			  :observed (default) -- Bellutta: p_e proportional to (current weight *
				tilt). Mean-unbiased at rho = 0; at rho != 0 it allocates by where
				weight currently SITS, starving spared-but-drained high-weight edges.
			  :deficit -- estimate-based: invert the proportional-hazard removal.
				Survival exp(-lambda*tilt_e) implies deficit_e = w_e*(exp(lambda*
				tilt_e) - 1) and est_true_e = w_e*exp(lambda*tilt_e); the single
				lambda is solved so the implied total removed equals B
				(sum_e w_e*(exp(lambda*tilt_e)-1) = B, monotone, bisection). p_e is
				proportional to deficit_e, so E[reconstructed_e] = est_true_e --
				recovery re-anchors to estimated-true, not observed. At rho = 0 the
				tilt is constant and this reduces exactly to :observed. A fully zeroed
				edge (w_e = 0) gets zero deficit (no resurrection -- that is the
				node/imputation path's job).

			A single Multinomial(B, probs) suffices (no per-edge ceiling on addition).
			Determinism flows from the passed rng.
		"""

		#	Guards
			setup.weighted ||
				throw(ArgumentError("_stage_2! requires weighted == true"))
			:weight in propertynames(augmented_edges) ||
				throw(ArgumentError("augmented_edges must have :weight column"))

		#	Budget B (the additional weight beyond the floor)
			W_add = round(Int, setup.additional_weight)
			(W_add <= 0 || nrow(augmented_edges) == 0) && return 0

		#	Per-Node rho-Tilt d for Every Augmented Node (from its degree bin)
			K = setup.K
			q = setup.q
			n_aug = nrow(augmented_nodes)
			d_aug = Vector{Float64}(undef, n_aug)
			@inbounds for v in 1:n_aug
				d_aug[v] = K * q[augmented_nodes.degree_bin[v]]
			end
			dmax = maximum(d_aug)
			dmax <= 0 && (dmax = 1.0)
			s = d_aug ./ (2.0 * dmax)

			id_to_idx = Dict(augmented_nodes.id[v] => v for v in 1:n_aug)

		#	Per-Edge Tilt and Current Weight (shared by both allocation modes)
			n_edges = nrow(augmented_edges)
			tilt = Vector{Float64}(undef, n_edges)
			wcur = Vector{Float64}(undef, n_edges)
			@inbounds for e in 1:n_edges
				i_idx = get(id_to_idx, augmented_edges.src[e], 0)
				j_idx = get(id_to_idx, augmented_edges.dst[e], 0)
				si = i_idx > 0 ? s[i_idx] : 0.0
				sj = j_idx > 0 ? s[j_idx] : 0.0
				tilt[e] = 1.0 - (1.0 - si) * (1.0 - sj)
				wcur[e] = Float64(augmented_edges.weight[e])
			end

		#	Sampling Weights per Allocation Mode
			probs = Vector{Float64}(undef, n_edges)
			if setup.allocation === :deficit
				#	Solve lambda so the implied total removed matches the budget:
				#	g(lambda) = sum_e w_e*expm1(lambda*tilt_e) = W_add  (monotone in lambda)
					gλ = function (λ)
						acc = 0.0
						@inbounds for e in 1:n_edges
							acc += wcur[e] * expm1(λ * tilt[e])
						end
						return acc
					end
					target = Float64(W_add)
					λ_lo = 0.0; λ_hi = 1.0
					while gλ(λ_hi) < target && λ_hi < 64.0
						λ_hi *= 2.0
					end
					λ = λ_hi
					if gλ(λ_hi) >= target
						@inbounds for _ in 1:60
							λm = 0.5 * (λ_lo + λ_hi)
							if gλ(λm) < target
								λ_lo = λm
							else
								λ_hi = λm
							end
						end
						λ = 0.5 * (λ_lo + λ_hi)
					end
					@inbounds for e in 1:n_edges
						probs[e] = wcur[e] * expm1(λ * tilt[e])
					end
			else  # :observed -- Bellutta proportional-to-current (default)
				@inbounds for e in 1:n_edges
					probs[e] = wcur[e] * tilt[e]
				end
			end

		#	Normalize and Allocate B Units (In-Place Increment)
			total = 0.0
			@inbounds for e in 1:n_edges
				total += probs[e]
			end
			total <= 0 && return 0
			probs ./= total
			counts = rand(rng, Multinomial(W_add, probs))
			augmented_edges.weight .+= counts

		#	Return Units Added
			return W_add
	end

######################################
#   REPLICATE GENERATION FUNCTIONS   #
######################################

#	Replicate: one augmented network and its per-replicate diagnostics
	struct Replicate
		augmented_edges::DataFrame
		augmented_nodes::DataFrame
		diag::Dict{Symbol, Any}
	end
	@doc raw"""
	**Description**
	Container for one augmented-network replicate produced by `generate_replicate`.
	Holds the augmented edge table, the augmented node roster (with bin labels and
	node-type tags), and a per-replicate diagnostics dictionary.

	**Fields**
	- `augmented_edges::DataFrame`: Columns `:src`, `:dst`, `:weight`. The original
	observed edges, Stage 0.5 imputed edges (when nominated non-respondents
	exist), Stage 1 edges incident to synthetic added nodes, and the in-place
	Stage 2 weight increments (when weighted and there is additional weight to
	place).

	- `augmented_nodes::DataFrame`: All columns of the original `setup.nodes`, plus
	`:degree_bin::Int`, `:ei_bin::Int`, and `:node_type::Symbol`
	(`:observed`, `:nominated`, or `:added`). Synthetic added nodes carry
	collision-safe ids matching the roster's id type. Integer rosters use ids
	above the maximum observed id; string rosters use generated `syn_*`
	identifiers that are checked against the existing roster before use.

	- `diag::Dict{Symbol, Any}`: Per-replicate diagnostics. Standard keys:
	`:seed`, `:realized_rho` (Kendall $\tau_b$ between the missing indicator —
	nominated + added — and the raw augmented centrality from
	`_compute_observed_centrality`: binarized in-degree for directed, degree for
	undirected, over the full augmented roster. This is the realized rho on the
	post-reconstruction field, identical to the metric `_passes_three_prior_gate`
	scores — NOT the degree bin and not the solver's bin-index target; NaN when the
	missing set is empty/complete or that centrality is constant),
	`:added_degree_bins`, `:added_ei_bins`, `:n_stage_0_5_edges`,
	`:n_stage_1_edges`, `:stage_2_weight_added`.

	**See Also**
	`generate_replicate`, `compute_setup`, `SamplerSetup`
	""" Replicate

#	generate_replicate: compose Stages 0.5, 1, and 2 into one augmented network
	function generate_replicate(setup::SamplerSetup, seed::Int)
		"""
		Args:
			setup::SamplerSetup: produced by compute_setup.
			seed::Int: deterministic seed for this replicate.
		Returns:
			Replicate struct with augmented_edges, augmented_nodes, diag.
		Notes:
			Steps in order: draw bin assignments; build augmented_nodes; copy edges;
			run Stage 0.5 (if N_nom > 0); run Stage 1; run Stage 2 (if weighted and
			additional_weight > 0); record realized rho and per-stage diagnostics.

			Realized rho is the AUTHORITATIVE metric, identical to the one
			_passes_three_prior_gate scores: binarized in-degree (directed) or degree
			(undirected) recomputed on the augmented edges via
			_compute_observed_centrality, correlated (Kendall tau-b) with the missing
			indicator (present = respondents; missing = nominated + added) over the
			full augmented roster. It is measured on the post-reconstruction field —
			the field the act of reconstruction has shifted — NOT the solver's
			bin-index target. Because the basis is binarized, Stage 2's weight
			redistribution does not affect it. NaN when the missing set is
			empty/complete or the augmented centrality is constant. Computing it the
			same way the gate does keeps rep.diag[:realized_rho] consistent with the
			gate's verdict for this replicate.
		"""

		#	Initialize RNG
			rng = Xoshiro(seed)

		#	Step 6: Draw Bin Assignments for Added Nodes
			bin_assignments = _draw_bin_assignments(setup.q,
													 setup.ei_conditional,
													 setup.N_add,
													 setup.binning_mode,
													 rng)

		#	Build Augmented Nodes Roster
			augmented_nodes = _build_augmented_nodes(setup,
													  bin_assignments.degree_bins,
													  bin_assignments.ei_bins)

		#	Initialize Augmented Edges
			augmented_edges = copy(setup.edges)

		#	Step 6.5: Stage 0.5 (if N_nom > 0)
			N_nom = length(setup.partially_observed)
			n_stage_0_5_edges = 0
			if N_nom > 0
				stage_0_5_edges = setup.directed ?
					_stage_0_5_directed(setup, augmented_nodes, rng) :
					_stage_0_5_undirected(setup, augmented_nodes, rng)
				n_stage_0_5_edges = nrow(stage_0_5_edges)
				append!(augmented_edges, stage_0_5_edges)
			end

		#	Step 7: Stage 1
			stage_1_edges = setup.directed ?
				_stage_1_directed(setup, augmented_nodes, rng) :
				_stage_1_undirected(setup, augmented_nodes, rng)
			n_stage_1_edges = nrow(stage_1_edges)
			append!(augmented_edges, stage_1_edges)

		#	Step 8: Stage 2 (if weighted and there is additional weight B to place)
			stage_2_weight_added = 0
			if setup.weighted && setup.additional_weight > 0
				stage_2_weight_added = _stage_2!(setup, augmented_edges, augmented_nodes, rng)
			end

		#	Step 9: Realized Kendall tau_b — missing (nominated + added) vs RAW augmented centrality
			#	Authoritative metric, matching _passes_three_prior_gate: binarized
			#	in-degree (directed) or degree (undirected) recomputed on the
			#	augmented edges, correlated with the missing indicator over the full
			#	augmented roster. Binarized, so Stage 2 weights do not affect it.
				node_type = augmented_nodes.node_type
				is_missing = (node_type .== :nominated) .| (node_type .== :added)
				n_missing = count(is_missing)
				n_total = length(is_missing)
				realized_rho = NaN
				if 0 < n_missing < n_total
					aug_centrality = _compute_observed_centrality(augmented_edges,
																   augmented_nodes,
																   setup.directed)
					if !all(==(aug_centrality[1]), aug_centrality)
						realized_rho = corkendall(Float64.(is_missing), aug_centrality)
					end
				end

		#	Assemble Diagnostics
			diag = Dict{Symbol, Any}(
				:seed => seed,
				:realized_rho => realized_rho,
				:added_degree_bins => bin_assignments.degree_bins,
				:added_ei_bins => bin_assignments.ei_bins,
				:n_stage_0_5_edges => n_stage_0_5_edges,
				:n_stage_1_edges => n_stage_1_edges,
				:stage_2_weight_added => stage_2_weight_added,
			)

		#	Return Replicate
			return Replicate(augmented_edges, augmented_nodes, diag)
	end

#	reconstruct_network: credible intervals on metrics over a reconstructed-network sample
	function reconstruct_network(edges::DataFrame,
								  nodes::DataFrame,
								  community_labels::Vector{Int};
								  directed::Bool,
								  weighted::Bool,
								  pi_node::Float64,
								  pi_edge::Float64,
								  rho::Float64,
								  metrics::Dict{Symbol, <:Function},
								  n_replicates::Int = 1000,
								  K::Union{Int, Symbol} = :auto,
								  partially_observed_nodes::Vector{Int} = Int[],
								  seed::Integer = 1,
								  quantiles::NTuple{3, Float64} = (0.025, 0.5, 0.975),
								  store_raw::Bool = false,
								  verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: observed-network inputs.
			directed, weighted, pi_node, pi_edge, rho: priors (pi_edge floored).
			metrics::Dict{Symbol,Function}: each metric(augmented_edges,
				augmented_nodes) -> Real. Required and non-empty.
			n_replicates, K, partially_observed_nodes, seed: sampling controls.
			quantiles::NTuple{3,Float64}: (lower, point, upper) credible-interval
				quantiles (default (0.025, 0.5, 0.975)).
			store_raw::Bool: return the full reconstruction corpus as raw (default
				false).
			verbose::Bool: progress printing.
		Returns:
			NamedTuple: intervals::Dict{Symbol,NamedTuple} (lower, median, upper, mean,
				std, n_valid per metric), weight_accounting, setup, n_replicates,
				quantiles, raw (the corpus DataFrame when store_raw, else nothing).
		Notes:
			Thin wrapper over build_reconstruction_corpus: that draws the sample and
			evaluates the metrics (one column per metric), and this aggregates each
			metric column into a credible interval. The credible interval is the
			spread of a measure across the reconstructed sample — our uncertainty
			about the true value, since no single reconstruction is known to be the
			true network.

			When store_raw, raw is the corpus DataFrame (per-sample metric values,
			deltas, and seeds), not a bare vector — so the same sample can be
			re-measured or replicated via materialize_reconstruction.
		"""

		#	Guards
			n_replicates >= 1 ||
				throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))
			!isempty(metrics) ||
				throw(ArgumentError("metrics must be non-empty"))
			all(0.0 .<= collect(quantiles) .<= 1.0) ||
				throw(ArgumentError("quantiles must be in [0, 1], got $quantiles"))

		#	Draw the Sample and Evaluate Metrics (one column per metric)
			built = build_reconstruction_corpus(edges, nodes, community_labels;
												 directed = directed, weighted = weighted,
												 pi_node = pi_node, pi_edge = pi_edge, rho = rho,
												 metrics = metrics,
												 n_replicates = n_replicates, K = K,
												 partially_observed_nodes = partially_observed_nodes,
												 seed = seed, verbose = verbose)
			corpus = built.corpus

		#	Aggregate Each Metric Column Into a Credible Interval
			q_lo, q_mid, q_hi = quantiles
			intervals = Dict{Symbol, NamedTuple}()
			for name in keys(metrics)
				finite = filter(isfinite, corpus[!, name])
				if isempty(finite)
					intervals[name] = (lower = NaN, median = NaN, upper = NaN,
										mean = NaN, std = NaN, n_valid = 0)
				else
					intervals[name] = (
						lower   = quantile(finite, q_lo),
						median  = quantile(finite, q_mid),
						upper   = quantile(finite, q_hi),
						mean    = mean(finite),
						std     = length(finite) >= 2 ? std(finite) : NaN,
						n_valid = length(finite),
					)
				end
			end

		#	Return
			return (
				intervals         = intervals,
				weight_accounting = built.weight_accounting,
				setup             = built.setup,
				n_replicates      = n_replicates,
				quantiles         = quantiles,
				raw               = store_raw ? corpus : nothing,
			)
	end
    @doc raw"""
	**Description**
	Produce credible intervals on user-specified network metrics for one observed
	network and prior. The credible interval is the spread of a measure across a
	sample of reconstructed networks — our uncertainty about the metric's true value,
	since no single reconstruction is known to be the true network. Delegates drawing
	and measuring the sample to `build_reconstruction_corpus`, then aggregates each
	metric's per-sample values into quantile-based intervals.

	**Usage**
	`reconstruct_network(edges, nodes, community_labels; directed, weighted, pi_node, pi_edge, rho, metrics, n_replicates=1000, K=:auto, quantiles=(0.025,0.5,0.975), store_raw=false, ...)`

	**Arguments**
	- `edges`, `nodes`, `community_labels`: observed network and precomputed labels.
	- `directed`, `weighted`: network type.
	- `pi_node`, `pi_edge`, `rho`: the priors; `pi_edge` is floored by `compute_setup`.
	- `metrics::Dict{Symbol,Function}`: required and non-empty; each
	  `metric(augmented_edges, augmented_nodes) -> Real`.
	- `n_replicates`, `K`, `partially_observed_nodes`, `seed`, `quantiles`, `store_raw`, `verbose`.

	**Value**
	NamedTuple:
	- `intervals::Dict{Symbol,NamedTuple}`: per metric, `(lower, median, upper, mean, std, n_valid)` over the finite sample values.
	- `weight_accounting::NamedTuple`: the floored-`pi_edge` decomposition (`W_observed`, `implied_min_weight`, `additional_weight`, `W_true`, `pi_edge_floor`, `realized_pi_edge`, `pi_edge_raised`).
	- `setup::SamplerSetup`: the setup reused across the sample.
	- `n_replicates::Int`, `quantiles::NTuple{3,Float64}`: echoed.
	- `raw`: the full reconstruction corpus `DataFrame` when `store_raw`, else `nothing` — per-sample metric values alongside the storage deltas and seeds, so the same sample can be re-measured or rebuilt with `materialize_reconstruction`.

	**Details**
	`build_reconstruction_corpus` runs `compute_setup` once and draws `n_replicates`
	reconstructions, sample `r` seeded `hash((seed, r))`, evaluating every metric on
	each sample as a column; this function reads those columns back and summarizes
	them. The run is reproducible in `seed`, and any single sample regenerates in
	isolation. A metric returning a non-finite value on a sample is dropped from that
	metric's interval (`n_valid` records the survivors).

	Because the sampling and the aggregation are separated, the held sample (in the
	returned corpus when `store_raw`, or by re-running `build_reconstruction_corpus`)
	can be re-measured with different metrics without redrawing.

	**See Also**
	`build_reconstruction_corpus`, `materialize_reconstruction`, `compute_setup`, `generate_replicate`, `feasible_rho_range`
	""" reconstruct_network

#	_extract_reconstruction_delta: split one replicate into structural + weight deltas vs the parent
	function _extract_reconstruction_delta(setup::SamplerSetup, rep::Replicate)
		"""
		Args:
			setup::SamplerSetup: holds the parent observed edges (setup.edges) and
				directedness.
			rep::Replicate: one augmented network from generate_replicate.
		Returns:
			NamedTuple:
				added_node_ids: synthetic added-node ids (node_type :added);
					nominated/observed nodes are already in the parent roster.
				struct_src, struct_dst: imputed (ego) edges absent from the parent,
					each entering at the weight-1 binary floor.
				wd_src, wd_dst, wd_add::Vector{Float64}: Stage 2 per-edge weight
					increments (final - floor) over every touched edge, parent and
					imputed alike.
		Notes:
			The storage delta for one reconstructed sample, mirroring how degeneracy
			stores missing_nodes rather than the whole degraded network. A sample is
			reconstructed as: parent UNION the structural ego edges (at weight 1), then
			+= the weight increments. The split is recovered by diffing the augmented
			network against the parent: an augmented edge whose canonical endpoints
			match a parent edge is a parent edge with floor = its observed weight;
			otherwise it is an imputed edge with floor 1. The increment is final - floor
			wherever positive.

			Canonical endpoint key honors directedness: (src, dst) directed,
			(min, max) undirected, so undirected edges stored src<dst match regardless
			of orientation. Imputation should not reuse an existing dyad or create
			duplicate augmented dyads; both invariants are checked here.
		"""

		#	Canonical Endpoint Key
			directed = setup.directed
			id_pt = eltype(setup.edges.src)
			canon = directed ?
				((s, d) -> (s, d)) :
				((s, d) -> (s <= d ? (s, d) : (d, s)))

		#	Parent Edge Lookup
			parent_w = Dict{Tuple{id_pt, id_pt}, Float64}()
			has_pw = hasproperty(setup.edges, :weight)
			@inbounds for r in 1:nrow(setup.edges)
				k = canon(setup.edges.src[r], setup.edges.dst[r])
				parent_w[k] = has_pw ? Float64(setup.edges.weight[r]) : 1.0
			end

		#	Guard Against Duplicate Parent Dyads
			length(parent_w) == nrow(setup.edges) ||
				throw(ArgumentError("setup.edges contains duplicate canonical dyads"))

		#	Added Node Ids
			added_mask = rep.augmented_nodes.node_type .== :added
			added_node_ids = rep.augmented_nodes.id[added_mask]

		#	Diff Augmented Edges Into Structural and Weight Deltas
			id_t = eltype(rep.augmented_edges.src)
			struct_src = id_t[]
			struct_dst = id_t[]
			wd_src = id_t[]
			wd_dst = id_t[]
			wd_add = Float64[]
			aug = rep.augmented_edges
			has_aw = hasproperty(aug, :weight)
			seen_aug = Set{Tuple{id_t, id_t}}()

			@inbounds for r in 1:nrow(aug)
				s = aug.src[r]
				d = aug.dst[r]
				w = has_aw ? Float64(aug.weight[r]) : 1.0
				k = canon(s, d)

				#	Guard Against Duplicate Augmented Dyads
					if k in seen_aug
						throw(ArgumentError("rep.augmented_edges contains duplicate canonical dyad $(k)"))
					end
					push!(seen_aug, k)

				#	Classify Edge and Compute Weight Delta
					if haskey(parent_w, k)
						floor_w = parent_w[k]
					else
						push!(struct_src, s)
						push!(struct_dst, d)
						floor_w = 1.0
					end

					delta = w - floor_w
					if delta > 0
						push!(wd_src, s)
						push!(wd_dst, d)
						push!(wd_add, delta)
					end
			end

		#	Return
			return (
				added_node_ids = added_node_ids,
				struct_src = struct_src,
				struct_dst = struct_dst,
				wd_src = wd_src,
				wd_dst = wd_dst,
				wd_add = wd_add
			)
	end

#	materialize_reconstruction: rebuild one reconstructed network exactly from its stored delta
	function materialize_reconstruction(setup::SamplerSetup, corpus_row)
		"""
		Args:
			setup::SamplerSetup: the shared setup from build_reconstruction_corpus;
				supplies the parent edges (setup.edges), the parent roster
				(setup.nodes), and directedness.
			corpus_row: one row of the corpus (DataFrameRow or NamedTuple) carrying
				added_node_ids, struct_src, struct_dst, wd_src, wd_dst, wd_add.
		Returns:
			NamedTuple (edges::DataFrame, nodes::DataFrame): the exact reconstructed
				network for that sample.
		Notes:
			The inverse of _extract_reconstruction_delta and the replication entry
			point: with setup + a stored row, anyone rebuilds the identical network
			without re-running the sampler. Construction is
				edges = parent edges (observed weights)
				        UNION structural ego edges at weight 1
				        then += the (wd_src, wd_dst, wd_add) increments;
				nodes = parent roster + the synthetic added-node ids.
			Increments are matched to edges by canonical endpoint key (orientation-
			aware, as in the extractor), so parent and ego edges both pick up their
			Stage 2 weight.

			The function assumes the parent setup and stored structural deltas contain
			one row per canonical dyad. It checks this invariant before applying
			weight increments, so duplicate dyads fail early rather than silently
			overwriting a row index.
		"""

		#	Canonical Endpoint Key
			directed = setup.directed
			id_t = eltype(setup.edges.src)
			canon = directed ?
				((s, d) -> (s, d)) :
				((s, d) -> (s <= d ? (s, d) : (d, s)))

		#	Edges: Parent UNION Structural Ego Edges
			has_pw = hasproperty(setup.edges, :weight)
			n_struct = length(corpus_row.struct_src)
			src = vcat(setup.edges.src, corpus_row.struct_src)
			dst = vcat(setup.edges.dst, corpus_row.struct_dst)
			parent_w = has_pw ? Float64.(setup.edges.weight) : ones(Float64, nrow(setup.edges))
			weight = vcat(parent_w, ones(Float64, n_struct))

		#	Row Index by Canonical Key
			row_of = Dict{Tuple{id_t, id_t}, Int}()
			@inbounds for r in eachindex(src)
				row_of[canon(src[r], dst[r])] = r
			end

		#	Guard Against Duplicate Reconstructed Dyads
			length(row_of) == length(src) ||
				throw(ArgumentError("reconstructed edge union contains duplicate canonical dyads"))

		#	Apply Stage 2 Weight Increments
			@inbounds for e in eachindex(corpus_row.wd_src)
				k = canon(corpus_row.wd_src[e], corpus_row.wd_dst[e])
				r = get(row_of, k, 0)
				r == 0 &&
					throw(ArgumentError("weight-delta edge $(k) has no matching edge in the union"))
				weight[r] += corpus_row.wd_add[e]
			end

			recon_edges = DataFrame(src = src, dst = dst, weight = weight)

		#	Nodes: Parent Roster plus Synthetic Added-Node Ids
			recon_nodes = copy(setup.nodes)
			hasproperty(recon_nodes, :id) ||
				throw(ArgumentError("setup.nodes must have an :id column"))
			ncols = Symbol.(names(recon_nodes))
			@inbounds for aid in corpus_row.added_node_ids
				push!(recon_nodes, Any[c === :id ? aid : missing for c in ncols]; promote = true)
			end

		#	Return
			return (edges = recon_edges, nodes = recon_nodes)
	end
	@doc raw"""
	**Description**
	Rebuild one reconstructed network exactly from its stored corpus delta. The
	replication entry point for `build_reconstruction_corpus`: given the shared
	`setup` and a corpus row, reproduces the identical network a sample represented,
	with no re-running of the sampler.

	**Usage**
	`materialize_reconstruction(setup, corpus_row)`

	**Value**
	NamedTuple `(edges, nodes)`. `edges` is `parent ∪ structural ego edges (weight 1)`
	with the stored Stage 2 increments added; `nodes` is the parent roster plus the
	synthetic added-node ids.

	**Details**
	Increments match edges by orientation-aware canonical key, so parent and ego
	edges both receive their weight. Parent and structural edge inputs are checked
	for one-row-per-canonical-dyad consistency before increments are applied; duplicate
	dyads fail early rather than silently overwriting row indices.

	Per-node generation internals (bins, node_type) are not reproduced — the delta
	fixes topology and weights; measure during corpus construction if per-node fields
	are needed.

	**See Also**
	`build_reconstruction_corpus`, `_extract_reconstruction_delta`, `reconstruct_network`
	""" materialize_reconstruction

#	Helper Function for build_reconstruction_corpus: 3-prior acceptance gate
	function _passes_three_prior_gate(rep::Replicate,
									   setup::SamplerSetup;
									   rho_target::Float64,
									   rho_tol::Float64 = 0.05,
									   pi_edge_tol::Float64 = 0.02,
									   pi_node_tol::Float64 = 1e-9)
		"""
		Args:
			rep::Replicate: a materialized replicate from generate_replicate, after
				all operations (Stage 0.5, Stage 1, Stage 2). augmented_edges /
				augmented_nodes are the full network.
			setup::SamplerSetup: supplies the conditioning targets (setup.pi_node,
				setup.pi_edge — both realized/floored), W_observed, and directedness.
			rho_target::Float64: the rho the sample is conditioned on (requested rho
				after adjustment to the achievable envelope). The gate only checks
				against it; build_reconstruction_corpus owns the adjustment.
			rho_tol, pi_edge_tol, pi_node_tol::Float64: acceptance half-widths — the
				realization-granularity bands that define "consistent with the prior"
				on a finite network, NOT free tuning knobs.
		Returns:
			NamedTuple (pass, pass_pi_node, pass_pi_edge, pass_rho,
						realized_pi_node, realized_pi_edge, realized_rho). realized_rho
				is raw-centrality tau-b, NaN when the missing set is empty/complete or
				centrality is constant.
		Notes:
			The end-of-pipeline acceptance gate, scored on the materialized network.
			PURE (no RNG, no regeneration) so it is unit-testable in isolation and
			reusable by the corpus builder's reject-and-regenerate loop.

			rho is the authoritative metric: binarized in-degree (directed) or degree
			(undirected) recomputed on the augmented edges via
			_compute_observed_centrality, then corkendall against the missing
			indicator (present = :observed; missing = :nominated | :added) over the
			full augmented roster. This is the realized rho on the post-reconstruction
			field — the field the act of reconstruction shifted — not the solver's
			bin-index target.

			pi_node is deterministic (N_add fixed at setup) so it equals setup.pi_node
			every replicate; a failure signals a setup-level inconsistency, not
			something regeneration can fix. pi_edge varies per replicate with the
			realized imputed-tie counts, hence a band rather than equality.
		"""

		#	Full Augmented Network
			aug_edges = rep.augmented_edges
			aug_nodes = rep.augmented_nodes
			n = nrow(aug_nodes)

		#	Prior 1: Realized Node-Missingness Fraction
			node_type = aug_nodes.node_type
			is_missing = (node_type .== :nominated) .| (node_type .== :added)
			n_missing = count(is_missing)
			realized_pi_node = n > 0 ? n_missing / n : 0.0
			pass_pi_node = abs(realized_pi_node - setup.pi_node) <= pi_node_tol

		#	Prior 2: Realized Weight-Missingness Fraction
			total_weight = hasproperty(aug_edges, :weight) ?
				sum(Float64.(aug_edges.weight)) : Float64(nrow(aug_edges))
			realized_pi_edge = total_weight > 0 ?
				(total_weight - setup.W_observed) / total_weight : 0.0
			pass_pi_edge = abs(realized_pi_edge - setup.pi_edge) <= pi_edge_tol

		#	Prior 3: Realized Centrality-Missingness Correlation (raw, augmented)
			realized_rho = NaN
			if 0 < n_missing < n
				centrality = _compute_observed_centrality(aug_edges, aug_nodes, setup.directed)
				if !all(==(centrality[1]), centrality)
					realized_rho = corkendall(Float64.(is_missing), centrality)
				end
			end
			pass_rho = isnan(realized_rho) ?
				(abs(rho_target) <= rho_tol) :
				(abs(realized_rho - rho_target) <= rho_tol)

		#	Overall Verdict
			pass = pass_pi_node && pass_pi_edge && pass_rho

		#	Return
			return (pass = pass,
					pass_pi_node = pass_pi_node,
					pass_pi_edge = pass_pi_edge,
					pass_rho = pass_rho,
					realized_pi_node = realized_pi_node,
					realized_pi_edge = realized_pi_edge,
					realized_rho = realized_rho)
	end

#	build_reconstruction_corpus: draw a gated, replicable sample of reconstructed networks
	function build_reconstruction_corpus(edges::DataFrame,
										  nodes::DataFrame,
										  community_labels::Vector{Int};
										  directed::Bool,
										  weighted::Bool,
										  pi_node::Float64,
										  pi_edge::Float64,
										  rho::Float64,
										  n_replicates::Int = 1000,
										  K::Union{Int, Symbol} = :auto,
										  allocation::Symbol = :observed,
										  partially_observed_nodes::Vector{Int} = Int[],
										  seed::Integer = 1,
										  metrics::Dict{Symbol, <:Function} = Dict{Symbol, Function}(),
										  rho_tol::Float64 = 0.05,
										  pi_edge_tol::Float64 = 0.02,
										  max_attempts::Int = 50,
										  feasibility_n_mc::Int = 20,
										  verbose::Bool = false)
		"""
		Args:
			edges, nodes, community_labels: observed-network inputs.
			directed, weighted, pi_node, pi_edge, rho: priors. rho is adjusted to the
				achievable envelope; the adjusted value is the conditioning prior.
			n_replicates::Int: number of accepted samples to draw.
			K, partially_observed_nodes, seed: as in reconstruct_network.
			allocation::Symbol: Stage-2 weight-allocation mode, :observed (default) or
				:deficit; forwarded unchanged to compute_setup, which validates it and
				carries it on the setup for _stage_2!. See compute_setup for the modes.
			metrics::Dict{Symbol,Function}: OPTIONAL measures on each accepted sample.
			rho_tol, pi_edge_tol::Float64: 3-prior gate acceptance half-widths.
			max_attempts::Int: regeneration cap per replicate before fallback.
			feasibility_n_mc::Int: MC draws for the rho feasibility screen.
			verbose::Bool: progress printing.
		Returns:
			NamedTuple: corpus::DataFrame (one accepted sample per row), setup,
				weight_accounting, rho_requested, rho_conditioned, rho_adjusted::Bool,
				n_gate_failures::Int, n_replicates, seed.
		Notes:
			Draws a sample of reconstructed networks, each accepted only if it passes
			the end-of-pipeline 3-prior gate (_passes_three_prior_gate) on the
			materialized network. Because the credible interval is conditional on the
			priors actually enforced, an over-extreme rho is first clamped into the
			feasible envelope (feasible_rho_range) and the conditioned value recorded;
			the clamp also keeps compute_setup from throwing :rho_infeasible.

			Per replicate, generate_replicate is retried with attempt-indexed seeds
			until the gate passes or max_attempts is hit; on exhaustion the closest
			draw (by rho deviation) is accepted with gate_passed = false. Each accepted
			sample is stored as a DELTA against the shared parent (as before), so any
			row reconstructs exactly via materialize_reconstruction.

			Corpus columns: sample_id, seed, added_node_ids, struct_src, struct_dst,
			wd_src, wd_dst, wd_add, realized_rho (raw, augmented), realized_pi_node,
			realized_pi_edge, gate_passed, n_attempts, n_stage_0_5_edges,
			n_stage_1_edges, stage_2_weight_added, n_added, plus one column per metric.
		"""

		#	Guards
			n_replicates >= 1 ||
				throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))
			max_attempts >= 1 ||
				throw(ArgumentError("max_attempts must be >= 1, got $max_attempts"))

		#	Canonicalize Edges Once (shared by the feasibility screen and setup)
			agg_edges = _aggregate_duplicate_dyads(edges, directed, weighted)

		#	Realized Node-Missingness Rate (the rate the feasibility screen uses)
			N_obs = nrow(nodes) - length(partially_observed_nodes)
			N_nom = length(partially_observed_nodes)
			N_add = _determine_n_add(pi_node, N_obs, N_nom)
			realized_total = N_obs + N_nom + N_add
			realized_pi_node = realized_total > 0 ? (N_nom + N_add) / realized_total : 0.0

		#	Adjust rho to the Achievable Envelope (the conditioning prior)
			#	The interval is conditional on the prior actually enforced, so an
			#	over-extreme rho is clamped into the feasible range and recorded.
			#	(Envelope is the observed-only screen; residual gap vs the augmented
			#	metric is absorbed by the gate fallback. Swap this block for a
			#	pilot-batch ceiling for exact augmented-space conditioning.)
				rho_conditioned = rho
				if realized_pi_node > 0.0
					feas = feasible_rho_range(agg_edges, nodes;
											  directed = directed, weighted = weighted,
											  target_rate = realized_pi_node,
											  n_mc_replicates = feasibility_n_mc,
											  seed = seed)
					rho_conditioned = clamp(rho, feas.rho_min, feas.rho_max)
				end
				rho_adjusted = rho_conditioned != rho
				if verbose && rho_adjusted
					println("build_reconstruction_corpus: requested rho = $rho is outside " *
							"the achievable envelope; conditioning on rho = " *
							"$(round(rho_conditioned, digits=4)).")
				end

		#	Setup Phase: Once (rho pre-clamped, so compute_setup won't throw)
			setup = compute_setup(agg_edges, nodes, community_labels;
								   directed = directed, weighted = weighted,
								   pi_node = pi_node, rho = rho_conditioned, pi_edge = pi_edge,
								   partially_observed_nodes = partially_observed_nodes,
								   K = K, allocation = allocation, verbose = verbose)

		#	Pre-Allocate Per-Sample Columns
			id_t = eltype(agg_edges.src)
			metric_names = collect(keys(metrics))
			sample_id        = collect(1:n_replicates)
			seed_col         = Vector{Int}(undef, n_replicates)
			added_ids_col    = Vector{Vector{id_t}}(undef, n_replicates)
			struct_src_col   = Vector{Vector{id_t}}(undef, n_replicates)
			struct_dst_col   = Vector{Vector{id_t}}(undef, n_replicates)
			wd_src_col       = Vector{Vector{id_t}}(undef, n_replicates)
			wd_dst_col       = Vector{Vector{id_t}}(undef, n_replicates)
			wd_add_col       = Vector{Vector{Float64}}(undef, n_replicates)
			realized_rho_col = Vector{Float64}(undef, n_replicates)
			realized_pin_col = Vector{Float64}(undef, n_replicates)
			realized_pie_col = Vector{Float64}(undef, n_replicates)
			gate_pass_col    = Vector{Bool}(undef, n_replicates)
			n_attempts_col   = Vector{Int}(undef, n_replicates)
			n05_col          = Vector{Int}(undef, n_replicates)
			n1_col           = Vector{Int}(undef, n_replicates)
			s2w_col          = Vector{Int}(undef, n_replicates)
			nadd_col         = Vector{Int}(undef, n_replicates)
			metric_cols      = Dict{Symbol, Vector{Float64}}(
				name => Vector{Float64}(undef, n_replicates) for name in metric_names)

		#	Draw, Gate, and Store Each Sample
			for r in 1:n_replicates
				#	Reject-and-Regenerate Against the 3-Prior Gate
					accepted_rep = nothing
					accepted_seed = 0
					accepted_verdict = nothing
					best_rep = nothing
					best_seed = 0
					best_verdict = nothing
					best_dev = Inf
					attempt = 0
					while attempt < max_attempts
						attempt += 1
						rep_seed = Int(hash((seed, r, attempt)) % UInt32)
						rep = generate_replicate(setup, rep_seed)
						verdict = _passes_three_prior_gate(rep, setup;
														   rho_target = rho_conditioned,
														   rho_tol = rho_tol,
														   pi_edge_tol = pi_edge_tol)
						#	Track the closest draw for the fallback (by rho deviation)
							dev = isnan(verdict.realized_rho) ?
								abs(rho_conditioned) : abs(verdict.realized_rho - rho_conditioned)
							if dev < best_dev
								best_dev = dev
								best_rep = rep
								best_seed = rep_seed
								best_verdict = verdict
							end
						if verdict.pass
							accepted_rep = rep
							accepted_seed = rep_seed
							accepted_verdict = verdict
							break
						end
					end

				#	Accept the Passing Draw, or Fall Back to the Closest
					if accepted_rep === nothing
						rep = best_rep; rep_seed = best_seed
						verdict = best_verdict; gate_passed = false
					else
						rep = accepted_rep; rep_seed = accepted_seed
						verdict = accepted_verdict; gate_passed = true
					end

				#	Store the Delta, Realized Priors, and Acceptance Diagnostics
					delta = _extract_reconstruction_delta(setup, rep)
					seed_col[r]         = rep_seed
					added_ids_col[r]    = delta.added_node_ids
					struct_src_col[r]   = delta.struct_src
					struct_dst_col[r]   = delta.struct_dst
					wd_src_col[r]       = delta.wd_src
					wd_dst_col[r]       = delta.wd_dst
					wd_add_col[r]       = delta.wd_add
					realized_rho_col[r] = verdict.realized_rho
					realized_pin_col[r] = verdict.realized_pi_node
					realized_pie_col[r] = verdict.realized_pi_edge
					gate_pass_col[r]    = gate_passed
					n_attempts_col[r]   = attempt
					n05_col[r]          = rep.diag[:n_stage_0_5_edges]
					n1_col[r]           = rep.diag[:n_stage_1_edges]
					s2w_col[r]          = rep.diag[:stage_2_weight_added]
					nadd_col[r]         = length(delta.added_node_ids)

				#	Evaluate Metrics on the Accepted Network
					for name in metric_names
						metric_cols[name][r] = Float64(metrics[name](rep.augmented_edges, rep.augmented_nodes))
					end

				if verbose && (r % max(1, n_replicates ÷ 10) == 0)
					println("  ... $r / $n_replicates samples")
				end
			end

		#	Assemble Corpus DataFrame
			corpus = DataFrame(
				sample_id            = sample_id,
				seed                 = seed_col,
				added_node_ids       = added_ids_col,
				struct_src           = struct_src_col,
				struct_dst           = struct_dst_col,
				wd_src               = wd_src_col,
				wd_dst               = wd_dst_col,
				wd_add               = wd_add_col,
				realized_rho         = realized_rho_col,
				realized_pi_node     = realized_pin_col,
				realized_pi_edge     = realized_pie_col,
				gate_passed          = gate_pass_col,
				n_attempts           = n_attempts_col,
				n_stage_0_5_edges    = n05_col,
				n_stage_1_edges      = n1_col,
				stage_2_weight_added = s2w_col,
				n_added              = nadd_col,
			)
			for name in metric_names
				corpus[!, name] = metric_cols[name]
			end

		#	Weight Accounting
			weight_accounting = (
				W_observed         = setup.W_observed,
				implied_min_weight = setup.implied_min_weight,
				additional_weight  = setup.additional_weight,
				W_true             = get(setup.diagnostics, :W_true, NaN),
				pi_edge_floor      = get(setup.diagnostics, :pi_edge_floor, NaN),
				realized_pi_edge   = setup.pi_edge,
				pi_edge_raised     = get(setup.diagnostics, :pi_edge_raised, false),
			)

		#	Return
			return (corpus            = corpus,
					setup             = setup,
					weight_accounting = weight_accounting,
					rho_requested     = rho,
					rho_conditioned   = rho_conditioned,
					rho_adjusted      = rho_adjusted,
					n_gate_failures   = count(!, gate_pass_col),
					n_replicates      = n_replicates,
					seed              = seed)
	end
	@doc raw"""
		**Description**
		Draw a replicable, gated sample of reconstructed networks for one observed
		network and prior. Each sample is materialized by `generate_replicate` and
		accepted only if it passes the end-of-pipeline 3-prior gate
		(`_passes_three_prior_gate`) on the full augmented network — realized
		$\pi_{\text{node}}$, $\pi_{\text{edge}}$, and $\rho$ within tolerance of the
		conditioning targets. Accepted samples are stored as per-sample DELTAS against
		the shared parent, so any row reconstructs exactly via
		`materialize_reconstruction`. Measures may be evaluated on each accepted sample
		in the same pass.

		**Usage**
		`build_reconstruction_corpus(edges, nodes, community_labels; directed, weighted, pi_node, pi_edge, rho, n_replicates=1000, K=:auto, allocation=:observed, metrics=Dict(), rho_tol=0.05, pi_edge_tol=0.02, max_attempts=50, ...)`

		**Arguments**
		- `edges`, `nodes`, `community_labels`: observed network and precomputed labels.
		- `directed`, `weighted`: network type.
		- `pi_node`, `pi_edge`, `rho`: the priors. `pi_edge` is floored by `compute_setup`;
		`rho` is clamped to the achievable envelope before setup (see Details), and the
		clamped value is the prior the corpus is conditioned on.
		- `n_replicates::Int`: number of ACCEPTED samples to draw.
		- `K`, `partially_observed_nodes`, `seed`: as in `reconstruct_network`.
		- `allocation::Symbol`: Stage-2 weight-allocation mode, `:observed` (default) or
		`:deficit`. Forwarded to `compute_setup` (which validates it) and applied by
		`_stage_2!`; does not affect gating or any other corpus quantity. See
		`compute_setup` for the two modes.
		- `metrics::Dict{Symbol,Function}`: optional measures, each
		`metric(augmented_edges, augmented_nodes) -> Real`, evaluated on each accepted
		sample and stored as a column. Empty stores deltas only.
		- `rho_tol`, `pi_edge_tol::Float64`: 3-prior gate acceptance half-widths — the
		realization-granularity bands that define "consistent with the prior" on a
		finite network. `pi_node` is deterministic and checked at machine tolerance.
		- `max_attempts::Int`: per-replicate regeneration cap before the fallback fires.
		- `feasibility_n_mc::Int`: MC draws for the up-front $\rho$ feasibility screen.
		- `verbose::Bool`: progress and adjustment printing.

		**Details**
		Because a credible interval is conditional on the prior actually enforced, an
		over-extreme `rho` is first clamped into the feasible envelope returned by
		`feasible_rho_range` at the realized node-missingness rate, and the conditioned
		value is recorded. The clamp also keeps `compute_setup` from throwing
		`:rho_infeasible`. `compute_setup` then runs once on the conditioned `rho`.

		Sample `r` is drawn by retrying `generate_replicate` with attempt-indexed seeds
		`hash((seed, r, attempt))` until the gate passes or `max_attempts` is reached.
		On exhaustion the closest draw (by $\rho$ deviation) is accepted with
		`gate_passed = false`. The gate scores $\rho$ as raw-centrality $\tau_b$
		(binarized in-degree directed / degree undirected, recomputed on the augmented
		edges) against the missing indicator over the full augmented roster; $\pi_{\text{edge}}$
		as $(\sum w_{\text{aug}} - W_{\text{observed}}) / \sum w_{\text{aug}}$ against
		`setup.pi_edge`; and $\pi_{\text{node}}$ as the missing fraction against
		`setup.pi_node`. The gate is allocation-agnostic: it scores realized $\rho$,
		$\pi_{\text{node}}$, and $\pi_{\text{edge}}$, none of which depend on how the
		weight budget was distributed across edges.

		Deltas are always stored even when metrics are supplied, so the held sample can
		be re-measured without re-drawing. The run is reproducible in `seed`, and any
		single sample regenerates in isolation via `materialize_reconstruction`.

		**Value**
		NamedTuple:
		- `corpus::DataFrame`: one accepted sample per row. Columns: `sample_id`, `seed`
		(accepted attempt's seed), `added_node_ids`, `struct_src`, `struct_dst`,
		`wd_src`, `wd_dst`, `wd_add`, `realized_rho` (raw-centrality $\tau_b$ on the
		augmented network), `realized_pi_node`, `realized_pi_edge`, `gate_passed`,
		`n_attempts`, `n_stage_0_5_edges`, `n_stage_1_edges`, `stage_2_weight_added`,
		`n_added`, plus one column per metric.
		- `setup::SamplerSetup`: the shared setup (carries the parent for materialization).
		- `weight_accounting::NamedTuple`: the floored-$\pi_{\text{edge}}$ decomposition.
		- `rho_requested::Float64`, `rho_conditioned::Float64`, `rho_adjusted::Bool`: the
		requested $\rho$, the value conditioned on after clamping, and whether clamping
		occurred.
		- `n_gate_failures::Int`: accepted-by-fallback count (`gate_passed == false`).
		- `n_replicates::Int`, `seed::Integer`: echoed.

		**See Also**
		`_passes_three_prior_gate`, `generate_replicate`, `reconstruct_network`,
		`compute_setup`, `feasible_rho_range`, `materialize_reconstruction`
		""" build_reconstruction_corpus

#	Exports (public API)
	export SamplerSetup,
		   Replicate,
		   compute_setup,
		   feasible_rho_range,
		   generate_replicate,
		   reconstruct_network,
		   materialize_reconstruction,
		   build_reconstruction_corpus
 
end # module network_reconstruction
