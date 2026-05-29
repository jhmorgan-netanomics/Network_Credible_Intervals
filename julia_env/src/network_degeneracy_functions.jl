module network_degeneracy

#   Module Packages
    using DataFrames
    using SparseArrays
    using LinearAlgebra
    using Statistics
    using Random
    using Distributions
    using StatsBase
    using ProgressMeter

#   Sibling-Module Helpers
    #   Sibling-relative import — submodules cannot `using` their parent
    #   (the parent is still loading when this line runs). Pull in only the
    #   specific helpers this module needs from network_community_detection.
        using ..network_community_detection: _graph_to_sparse_matrix

#	Helper Centrality Driver: Binarized in-degree (directed) / binarized degree (undirected)
	function _centrality_for_sampler(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
									directed::Bool)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst (weights ignored — binarized)
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe (includes isolates)
			directed::Bool: true for directed networks (use binarized in-degree),
				false for undirected (use binarized degree)
		Returns:
			Vector{Float64}: per-node centrality score in the same order as the
				node universe (the order returned by _graph_to_sparse_matrix)
		Notes:
			This is the centrality driver for the Phase 1 sampler — the c_i in
			prob_i = b*c_i + (1-b)*u_i. It is binarized regardless of any
			:weight column on the edges DataFrame, following SMM 2022 for the
			directed case and extending naturally to the undirected case. Edge
			weights are stripped (weighted=false in _graph_to_sparse_matrix) so
			the centrality reflects tie presence, not tie intensity — the
			degradation grid varies tie presence (dropped nodes lose all their
			edges) and the centrality driver must align with that.

			For directed networks the centrality is the in-degree of the
			binarized graph: sum of incoming edges per node. For undirected
			networks the centrality is the degree of the binarized graph: sum
			of edges per node (each undirected edge counted once per endpoint).
			On a symmetric adjacency the row sums and column sums are equal, so
			either axis would work; we use the row sums for consistency with
			the directed case (where row sums = out-degree, column sums = in-
			degree, and we want in-degree).

			Returned as Float64 rather than Int because downstream the values
			are used as weights in StatsBase.sample(..., weights=...) and in
			the floating-point convex combination prob_i = b*c_i + (1-b)*u_i.
			The Float64 conversion is done here rather than at every call site.

			Output is in the canonical node order from _graph_to_sparse_matrix
			— the same order used elsewhere in the package. Callers indexing
			dropped-node sets back to network names must use the same node
			ordering convention.
		"""

		#	Build Binarized Adjacency
			#	weighted=false strips edge weights; isolates are included when
			#	the nodes argument is supplied.
				adj, _, _ = isnothing(nodes) ?
					_graph_to_sparse_matrix(edges; weighted=false) :
					_graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)
				n = size(adj, 1)

		#	Compute Centrality
			#	Directed: in-degree = column sums of A (number of incoming
			#	edges to each node). Undirected: degree = sum of incoming AND
			#	outgoing tie-stubs at each node, because _graph_to_sparse_matrix
			#	stores each undirected edge as a single directed entry (one
			#	direction, determined by the edges DataFrame's src/dst layout)
			#	rather than symmetrizing. We therefore add column sums (incoming)
			#	to row sums (outgoing) for the undirected case to recover the
			#	true degree. For directed networks the in-degree is what we
			#	want and only the column sum is added.
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
	Computes the binarized centrality driver used by the Phase 1 missingness
	sampler. For directed networks the centrality is in-degree; for undirected
	networks the centrality is degree. Edge weights are ignored — the
	centrality reflects tie presence, consistent with the SMM 2022 convention.

	This is the $c_i$ in the sampler's selection-probability formula:
	$$\text{prob}_i = b \cdot c_i + (1 - b) \cdot u_i,\quad u_i \sim \text{Uniform}(0,1).$$

	**Usage**
	`_centrality_for_sampler(edges; nodes=nothing, directed=true)`

	**Arguments**
	- `edges::DataFrame`: edge list with `:src`, `:dst` columns. Any `:weight`
	  column is ignored.
	- `nodes::Union{Nothing,DataFrame,Vector}`: optional explicit node universe
	  including isolates. When `nothing`, only nodes appearing in `edges` are
	  in the node set.
	- `directed::Bool`: `true` to use binarized in-degree, `false` to use
	  binarized degree. Caller typically passes the network's directedness
	  metadata.

	**Value**
	`Vector{Float64}` of per-node centrality scores in the canonical node
	order returned by `_graph_to_sparse_matrix`. Length equals the number of
	nodes (including any isolates supplied via `nodes`).

	**Notes**
	Per-network value — does not change across $(\rho, \text{rate})$ targets
	in the validation grid. The caller (typically `generate_missingness_mask`
	or the orchestrator `build_degeneration_corpus`) caches the result for
	each network and reuses it across all 30 cells of the $(\rho, \text{rate})$
	grid for that network.

	**See Also**
	`generate_missingness_mask`, `_centrality_correlation_for_b`,
	`_graph_to_sparse_matrix`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling
	  coverage III: Imputation of missing network data under different
	  network and missing data conditions. *Social Networks*, 68, 148–178.
	""" _centrality_for_sampler

#	Helper Centrality Correlation: Monte Carlo estimate of E[cor(is_dropped, c)] at candidate b
	function _centrality_correlation_for_b(centrality::AbstractVector{<:Real},
											b::Real,
											sgn::Integer,
											target_rate::Real,
											bisection_seed::Integer,
											iter_idx::Integer;
											M::Int = 50)
		"""
		Args:
			centrality::AbstractVector{<:Real}: centrality driver from
				_centrality_for_sampler
			b::Real: candidate mixing scalar, in [0, 1]
			sgn::Integer: +1 to target positive realized rho (drop central
				nodes preferentially), -1 to target negative; must be +1 or -1
			target_rate::Real: target fraction of nodes to drop; in (0, 1).
				Needed because the inner samples must draw the same k that
				the actual sampler will draw at production time.
			bisection_seed::Integer: master seed for this bisection; combines
				with iter_idx and inner sample index m to produce per-sample
				seeds via hash
			iter_idx::Integer: bisection iteration index (1, 2, 3, ...);
				ensures that different outer iterations use different inner
				draws
			M::Int: number of inner Monte Carlo samples per call (default 50)
		Returns:
			Float64: empirical mean of cor(is_dropped, centrality) across M
				inner weighted-without-replacement draws. Returns NaN when
				the centrality vector is constant (max == min); the bisection
				short-circuits on that case via the :failed_other status.
		Notes:
			This function is the substantive change from the prob-vector
			bisection. Instead of returning cor(prob, c) — the deterministic
			sampler-recipe quantity — it returns an estimate of cor(is_dropped, c),
			the realized indicator quantity that the analysis cares about.
			The estimate is the average of M independent weighted-without-
			replacement draws at this candidate b.

			The probability vector is constructed once per call:
				prob_i = b * c_normalized_i + (1 - b) * u_i  (with u_i drawn
				once from a single RNG seeded deterministically; see below)
			For negative-rho targets (sgn == -1), the prob vector is inverted
			via prob_i <- 1 - prob_i before sampling. The inner draws share
			one prob vector and differ only in which k nodes get sampled
			from it.

			Determinism scheme: each inner sample m in 1:M uses a seed
			derived as Int(hash((bisection_seed, iter_idx, m)) % UInt32).
			This makes the function reproducible in bisection_seed regardless
			of thread scheduling (when threaded), and ensures that successive
			bisection iterations consume different inner draws (no false
			convergence from re-using the same M samples at every b).

			Cost: M weighted-without-replacement draws plus M correlation
			computations. On N <= 1000 networks each iteration is sub-
			millisecond. At the default M=50 with ~30 bisection iterations
			per cell, the bisection adds ~1500 sampling operations per
			generate_missingness_mask call.

			Edge cases:
			- Constant centrality returns NaN: cor(is_dropped, constant) is
			  undefined regardless of the draw. The bisection caller checks
			  for NaN and returns status = :failed_other.
			- Inner draws use a separate u-draw per call; this is necessary
			  because changing b changes the prob vector. The 'inner' part
			  is purely the sampling-given-prob; the u_i noise is regenerated
			  fresh at every bisection iteration.
			- Saturation: when fewer than k entries of prob are strictly
			  positive (e.g., extreme-skew centrality at b=1 on a star
			  fixture), weighted-without-replacement sampling cannot draw k
			  distinct nodes. The function falls back to selecting all
			  positive-weight nodes deterministically and uniform-randomly
			  filling the remaining k - n_positive slots from zero-weight
			  nodes. This is the regime where the centrality signal is
			  exhausted; the bisection should detect the resulting realized
			  cor as below target and return :ceiling_hit.
		"""
		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes"))
			(0.0 <= b <= 1.0) || throw(ArgumentError("b must be in [0, 1], got $b"))
			(sgn == 1 || sgn == -1) || throw(ArgumentError("sgn must be +1 or -1, got $sgn"))
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			M >= 1 || throw(ArgumentError("M must be >= 1, got $M"))

		#	Min-Max Normalize Centrality, Short-Circuit on Constant
			c_min = minimum(centrality)
			c_max = maximum(centrality)
			if c_max == c_min
				return NaN
			end
			c_norm = (centrality .- c_min) ./ (c_max - c_min)

		#	Probability Vector
			#	The u-noise is regenerated each call; this is deterministic
			#	in (bisection_seed, iter_idx) via a dedicated seed below.
				u_seed = Int(hash((bisection_seed, iter_idx, 0)) % UInt32)
				u_rng  = Xoshiro(u_seed)
				u      = rand(u_rng, n)
				prob   = b .* c_norm .+ (1 - b) .* u
				if sgn == -1
					prob = 1.0 .- prob
				end
				#	Defensive clamp; floating-point can push prob slightly negative
					@inbounds for i in 1:n
						prob[i] < 0.0 && (prob[i] = 0.0)
					end

		#	M Inner Weighted-Without-Replacement Draws
			k = Int(round(target_rate * n))
			k >= 1     || throw(ArgumentError("target_rate too small: k = $k < 1"))
			k <= n - 1 || throw(ArgumentError("target_rate too large: k = $k > N - 1"))

			#	Saturation check: count strictly positive prob entries.
			#	When fewer than k entries are positive (extreme-skew centrality
			#	at b=1 produces a (1, 0, ..., 0)-like prob vector), StatsBase.sample
			#	cannot draw k distinct nodes from the positive support. We handle
			#	this deterministic edge case by selecting all positive-weight
			#	nodes and filling the remaining slots uniformly at random from
			#	zero-weight nodes. This is the saturation regime where the
			#	centrality signal is exhausted; the achievable realized cor
			#	is whatever this construction produces, and the bisection will
			#	correctly classify it as :ceiling_hit since the saturation cor
			#	is below any target_rho close to 1.
				n_positive = count(p -> p > 0.0, prob)

			cor_sum = 0.0
			centrality_vec = Vector{Float64}(centrality)   #	avoid repeated conversion
			is_dropped_buf = zeros(Float64, n)              #	reused across inner draws

			if n_positive >= k
				#	Standard path: enough positive-weight nodes for weighted-
				#	without-replacement sampling
					weights = StatsBase.Weights(prob)
					@inbounds for m in 1:M
						sample_seed = Int(hash((bisection_seed, iter_idx, m)) % UInt32)
						rng = Xoshiro(sample_seed)
						dropped = StatsBase.sample(rng, 1:n, weights, k; replace=false)
						#	Reset and fill the indicator buffer for this draw
							fill!(is_dropped_buf, 0.0)
							for idx in dropped
								is_dropped_buf[idx] = 1.0
							end
						cor_sum += cor(is_dropped_buf, centrality_vec)
					end
			else
				#	Saturation path: take all positive-weight nodes deterministically,
				#	then uniform-randomly fill the remaining k - n_positive slots
				#	from the zero-weight nodes. The positive nodes are always
				#	dropped; the zero nodes are chosen at random.
					positive_idxs = findall(p -> p > 0.0, prob)
					zero_idxs     = findall(p -> p == 0.0, prob)
					k_remaining   = k - n_positive
					@inbounds for m in 1:M
						sample_seed = Int(hash((bisection_seed, iter_idx, m)) % UInt32)
						rng = Xoshiro(sample_seed)
						#	Reset and fill: positives always dropped, plus
						#	k_remaining uniform draws from zeros
							fill!(is_dropped_buf, 0.0)
							for idx in positive_idxs
								is_dropped_buf[idx] = 1.0
							end
							zero_dropped = StatsBase.sample(rng, zero_idxs, k_remaining; replace=false)
							for idx in zero_dropped
								is_dropped_buf[idx] = 1.0
							end
						cor_sum += cor(is_dropped_buf, centrality_vec)
					end
			end

		#	Return Mean Realized Indicator Correlation
			return cor_sum / M
	end

#	Helper Bisection: solve for b such that the realized cor(is_dropped, c) hits target_rho
	function _bisect_b_for_target_rho(centrality::AbstractVector{<:Real},
										target_rho::Real,
										target_rate::Real,
										bisection_seed::Integer;
										tol::Real = 0.02,
										max_iters::Int = 50,
										M::Int = 50)
		"""
		Args:
			centrality::AbstractVector{<:Real}: centrality driver
			target_rho::Real: target realized correlation, in (-1, 1)
			target_rate::Real: target fraction dropped; passed through to
				_centrality_correlation_for_b
			bisection_seed::Integer: master seed for this bisection
			tol::Real: convergence tolerance on |estimate - |target_rho||;
				default 0.02 (loosened from the prob-vector version's 1e-3
				to account for inner Monte Carlo noise at M=50)
			max_iters::Int: maximum bisection iterations; default 50
			M::Int: inner Monte Carlo samples per iteration; default 50
		Returns:
			NamedTuple: (b, sign, realized_rho_pos, status)
				- b::Float64: the bisected mixing scalar
				- sign::Int: +1 for positive target, -1 for negative
				- realized_rho_pos::Float64: the Monte Carlo estimate of the
					realized cor at the bisected b, before sign-flipping;
					NOT the actual single-sample realized rho the production
					draw will produce
				- status::Symbol: :converged, :ceiling_hit, or :failed_other
		Notes:
			This bisection targets the realized indicator correlation, not
			the prob-vector correlation. Each iteration calls
			_centrality_correlation_for_b with the current iteration index,
			which produces an M-sample average estimate of E[cor(is_dropped, c)]
			at the candidate b. We bisect that estimate against |target_rho|.

			Tolerance rationale: with M=50 inner samples, the standard error
			of the inner-mean estimate is roughly 0.015 on Scotland-scale
			centrality vectors. A tolerance of 0.02 is one-and-a-third SEs;
			tighter tolerances would chase Monte Carlo noise rather than
			real bisection signal.

			Status semantics (unchanged from prob-vector version):
			- :converged: |estimate - |target_rho|| < tol within max_iters
			- :ceiling_hit: the estimate at b=1 is below |target_rho|;
				 the centrality distribution is not extreme enough to reach
				 the target. We return b=1 and the actual ceiling estimate
				 so the caller knows what was achievable.
			- :failed_other: constant centrality (NaN estimate at b=0), or
				 max_iters exceeded without convergence. Caller treats record
				 as unusable.

			The MCAR fast-path (target_rho == 0): we still short-circuit to
			b=0 (pure uniform sampling). At b=0 the prob vector is just u,
			the realized cor is sampling noise centered on zero.

			Reproducibility: the function is deterministic in bisection_seed.
			The same seed produces the same sequence of iter_idx values
			(1, 2, 3, ...), each consuming the same inner draws. Test 4
			verifies this by calling with identical inputs twice.

			Convergence behavior: bisection on a noisy signal can occasionally
			oscillate when the truth is near the tolerance boundary. The
			max_iters cap is the safety net; in practice convergence happens
			in 8-15 iterations because the bisection's halving rate is much
			faster than the M=50 noise floor.
		"""
		#	Guards
			(-1.0 < target_rho < 1.0) || throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			tol > 0     || throw(ArgumentError("tol must be positive, got $tol"))
			max_iters >= 1 || throw(ArgumentError("max_iters must be >= 1, got $max_iters"))
			M >= 1      || throw(ArgumentError("M must be >= 1, got $M"))

		#	MCAR Fast Path
			if target_rho == 0.0
				return (b = 0.0, sign = 1, realized_rho_pos = 0.0, status = :converged)
			end

		#	Resolve Sign; Bisect Against |target_rho|
			sgn = target_rho > 0 ? 1 : -1
			abs_target = abs(target_rho)

		#	Probe at b = 1 to Detect Ceiling
			ceiling_est = _centrality_correlation_for_b(centrality, 1.0, sgn, target_rate,
														 bisection_seed, 1; M = M)
			if isnan(ceiling_est)
				return (b = 0.0, sign = sgn, realized_rho_pos = NaN, status = :failed_other)
			end
			if ceiling_est < abs_target - tol
				#	Ceiling fires: the maximum achievable estimate is below target
					return (b = 1.0, sign = sgn, realized_rho_pos = ceiling_est, status = :ceiling_hit)
			end

		#	Standard Bisection on [0, 1]
			#	Iter index 1 was used by the ceiling probe above; the inner
			#	bisection starts from iter_idx = 2 and increments each step.
			#	This guarantees that no two calls to _centrality_correlation_for_b
			#	in this bisection share an iter_idx, so the inner draws differ
			#	across bisection iterations.
				lo, hi = 0.0, 1.0
				b_mid = 0.5
				est_mid = NaN
				iter_idx = 2
				converged = false
				for iter in 1:max_iters
					b_mid = (lo + hi) / 2
					est_mid = _centrality_correlation_for_b(centrality, b_mid, sgn, target_rate,
															  bisection_seed, iter_idx; M = M)
					iter_idx += 1
					if isnan(est_mid)
						return (b = 0.0, sign = sgn, realized_rho_pos = NaN, status = :failed_other)
					end
					if abs(est_mid - abs_target) < tol
						converged = true
						break
					elseif est_mid < abs_target
						lo = b_mid
					else
						hi = b_mid
					end
				end

			if converged
				return (b = b_mid, sign = sgn, realized_rho_pos = est_mid, status = :converged)
			else
				return (b = b_mid, sign = sgn, realized_rho_pos = est_mid, status = :failed_other)
			end
	end

#	Helper Sampler: weighted-without-replacement draw with sign-flip support
	function _sample_missingness(centrality::AbstractVector{<:Real},
									target_rate::Real,
									b::Real,
									sgn::Integer,
									sample_seed::Integer)
		"""
		Args:
			centrality::AbstractVector{<:Real}: centrality driver from
				_centrality_for_sampler
			target_rate::Real: target fraction of nodes to drop; in (0, 1)
			b::Real: mixing scalar from _bisect_b_for_target_rho; in [0, 1]
			sgn::Integer: +1 to drop high-centrality nodes preferentially
				(positive target rho), -1 to drop low-centrality preferentially
				(negative target rho). MUST be +1 or -1.
			sample_seed::Integer: RNG seed for the u_i Uniform(0,1) draws and
				for the StatsBase.sample(...) call
		Returns:
			NamedTuple: (dropped_nodes, realized_rate, realized_rho)
				- dropped_nodes::Vector{Int}: indices of the dropped nodes in
					the canonical node order, sorted ascending
				- realized_rate::Float64: |dropped_nodes| / N
				- realized_rho::Float64: cor(is_dropped, centrality) across
					the node set, signed against the original centrality
					direction (NOT inverted by sgn — this is the realized
					correlation the analysis cares about)
		Notes:
			Implements the weighted-without-replacement sampler from SMM 2022.
			The probability vector is constructed as
			    prob_i = b * c_normalized_i + (1 - b) * u_i,
			where c_normalized_i is the [0,1] min-max-normalized centrality
			and u_i ~ Uniform(0,1). When sgn = -1 (negative target rho), the
			prob vector is inverted via prob_i ← 1 - prob_i before sampling,
			producing draws where low-centrality nodes are preferentially
			selected.

			Rate-targeting is exact: k = round(target_rate * N) nodes are
			drawn via StatsBase.sample(..., weights=prob, replace=false),
			which produces a draw of exactly k nodes from the node set with
			selection probabilities proportional to prob. The realized rate
			is therefore round(target_rate * N) / N for every replicate
			(modulo Julia's round-half-to-even convention).

			realized_rho is computed from the SAMPLED dropped-node set
			(is_dropped indicator vector) against the original centrality
			vector — not from the prob vector. This is what the analysis
			cares about: the realized correlation between dropped status
			and centrality, which is what coverage and calibration are
			conditioned on. The mean of realized_rho across many seeds
			converges to the target_rho the bisection was solving for (per
			Test 1 in the harness); single-replicate realized_rho is noisy.

			The realized_rho is signed against the ORIGINAL centrality
			direction, so for negative-target-rho draws it will be negative
			as expected. This matches what the caller expects to record in
			the per-replicate record alongside the nominal target_rho.

			sample_seed should be distinct from any seed used in the
			bisection (Function 3), otherwise the deterministic bisection
			result becomes entangled with the sampling stochasticity in
			confusing ways. The public wrapper generate_missingness_mask
			handles this seed-splitting; direct callers must do so manually.

			Edge cases:
			- If centrality is constant (max == min), c_normalized is the zero
			  vector and prob_i = (1-b) * u_i, giving uniform-noise sampling
			  regardless of b. This is the correct behavior for a regular
			  graph where no value of b can produce target rho; the caller
			  (Function 3) should have returned status = :failed_other on this
			  network, but the sampler still returns a valid uniform draw if
			  called directly.
			- If b = 1.0 exactly and centrality is non-constant, prob_i =
			  c_normalized_i with no noise component. When the centrality
			  distribution is extreme-skew (e.g., a star fixture where one
			  node has all the mass), the prob vector saturates with fewer
			  than k positive entries. The function detects this and falls
			  back to a deterministic-positive-plus-uniform-zero-fill draw
			  rather than failing. This path is exercised when the bisection
			  returns :ceiling_hit and the caller proceeds with the saturated
			  b; the realized indicator correlation matches what the
			  bisection's saturation probe estimated.
		"""

		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes"))
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			(0.0 <= b <= 1.0) || throw(ArgumentError("b must be in [0, 1], got $b"))
			(sgn == 1 || sgn == -1) || throw(ArgumentError("sgn must be +1 or -1, got $sgn"))

		#	Min-Max Normalize Centrality to [0, 1]
			c_min  = minimum(centrality)
			c_max  = maximum(centrality)
			c_norm = c_max == c_min ? zeros(Float64, n) : (centrality .- c_min) ./ (c_max - c_min)

		#	Draw u_i and Form the Probability Vector
			rng  = Xoshiro(sample_seed)
			u    = rand(rng, n)
			prob = b .* c_norm .+ (1 - b) .* u

		#	Sign Flip for Negative Target rho
			#	Inverting via 1 - prob keeps the prob vector in [0, 1] and
			#	flips which nodes are preferred for dropping.
				if sgn == -1
					prob = 1.0 .- prob
				end

		#	Numerical Safety: ensure no negative weights
			#	prob should already be in [0, 1] but floating-point error can
			#	produce tiny negatives near the boundaries. Clamp.
				@inbounds for i in 1:n
					if prob[i] < 0.0
						prob[i] = 0.0
					end
				end

		#	Weighted-Without-Replacement Sample of k = round(target_rate * N)
			k = Int(round(target_rate * n))
			k >= 1                || throw(ArgumentError("target_rate too small: k = $k < 1"))
			k <= n - 1            || throw(ArgumentError("target_rate too large: k = $k > N - 1 = $(n - 1)"))

			#	Saturation check: count strictly positive prob entries.
			#	Mirror of the saturation handling in _centrality_correlation_for_b:
			#	when the bisection returns :ceiling_hit on extreme-skew centrality,
			#	the call here arrives with a saturated prob vector (e.g., b=1
			#	on a star fixture gives prob = (1, 0, ..., 0)). StatsBase.sample
			#	cannot draw k distinct nodes from fewer than k positive weights.
			#	The fallback selects all positive-weight nodes deterministically
			#	and uniform-randomly fills the remaining slots from zero-weight
			#	nodes — the realized indicator correlation matches what the
			#	bisection's saturation probe predicted.
				n_positive = count(p -> p > 0.0, prob)
				if n_positive >= k
					#	Standard path
						dropped = StatsBase.sample(rng, 1:n, StatsBase.Weights(prob), k; replace=false)
				else
					#	Saturation path: positives always dropped, plus uniform
					#	draws from zeros to fill remaining slots
						positive_idxs = findall(p -> p > 0.0, prob)
						zero_idxs     = findall(p -> p == 0.0, prob)
						k_remaining   = k - n_positive
						zero_dropped  = StatsBase.sample(rng, zero_idxs, k_remaining; replace=false)
						dropped       = vcat(positive_idxs, zero_dropped)
				end
				sort!(dropped)

		#	Compute Realized Quantities from the Actual Draw
			is_dropped = falses(n)
			@inbounds for idx in dropped
				is_dropped[idx] = true
			end
			realized_rate = k / n
			realized_rho  = cor(Float64.(is_dropped), centrality)

		#	Return
			return (dropped_nodes  = dropped,
					realized_rate  = realized_rate,
					realized_rho   = realized_rho)
	end

#	Helper Topological Degeneracy: gc fraction, node count, edge count + threshold flags
	function _topological_degeneracy(adj::SparseMatrixCSC,
										dropped_nodes::AbstractVector{<:Integer};
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1)
		"""
		Args:
			adj::SparseMatrixCSC: original binarized adjacency matrix (the
				same matrix _centrality_for_sampler used; weights ignored)
			dropped_nodes::AbstractVector{<:Integer}: indices of dropped nodes
				in the canonical node order; need not be sorted
			gc_threshold::Real: giant-component fraction threshold (default
				0.30); the gc_collapse flag fires when gc_fraction_of_remaining
				falls below this
			min_n::Int: floor on remaining node count (default 25); the
				too_small flag fires when n_observed < min_n
			min_edges::Int: floor on remaining edge count (default 1); the
				no_edges flag fires when n_edges_observed < min_edges
		Returns:
			NamedTuple: (gc_fraction_of_remaining, n_observed, n_edges_observed,
						 gc_collapse, too_small, no_edges, any_topo_degenerate)
				All three continuous quantities are reported alongside the
				threshold-derived Booleans, per the design's continuous-record
				principle (post-hoc re-thresholding without re-running).
		Notes:
			The continuous quantities are the primary record; the Booleans are
			a convenience-summary view at the locked thresholds. Downstream
			analysis can recompute the Booleans against alternative thresholds
			using only the continuous fields.

			Giant-component fraction is computed against REMAINING nodes
			(n_largest_component / n_observed), not original nodes. This is
			the design choice that asks 'is what survives structurally
			coherent?' rather than 'how much of the original survives?'

			For directed graphs the giant component is the WEAKLY connected
			component — matching network_statistics.largest_component_size
			and the SMM 2022 convention. The BFS considers an edge (i, j) as
			traversable in either direction.

			Implementation: BFS over the original adjacency with the dropped
			set as a node mask, rather than materializing the degraded sparse
			matrix. The dropped-set membership is checked via a Bool vector
			of length N for O(1) lookups inside the BFS inner loop.

			Edges incident to at least one dropped node are excluded from
			n_edges_observed. Counted by walking the CSR once and testing
			both endpoints against the dropped mask.

			Edge cases:
			- All nodes dropped (n_observed = 0): gc_fraction is set to 0.0
			  (vacuously), all three flags fire. Should not occur in practice
			  since target_rate < 1 is enforced in _sample_missingness.
			- One node remaining: gc_fraction = 1.0 trivially, but too_small
			  fires (1 < min_n). The any_topo_degenerate disjunction catches it.
			- No edges remaining: gc_fraction is computed from connectivity
			  via the (empty) edge set, so every remaining node is its own
			  component of size 1; gc_fraction = 1/n_observed.
		"""

		#	Guards
			n = size(adj, 1)
			size(adj, 2) == n || throw(ArgumentError("adj must be square"))

		#	Build Dropped Mask
			#	is_dropped[i] = true iff node i is in the dropped set
				is_dropped = falses(n)
				@inbounds for idx in dropped_nodes
					(1 <= idx <= n) || throw(ArgumentError("dropped node index $idx out of range [1, $n]"))
					is_dropped[idx] = true
				end

		#	n_observed: remaining node count
			n_observed = n - count(is_dropped)

		#	Edge count among remaining nodes
			#	Walk CSR: for each column j, iterate row indices via nzrange;
			#	count edges where both endpoints survive. On directed input
			#	this counts arcs (i,j) and (j,i) separately, which is the
			#	convention used elsewhere in the package for edge counts.
				rows = rowvals(adj)
				n_edges_observed = 0
				@inbounds for j in 1:n
					is_dropped[j] && continue
					for k in nzrange(adj, j)
						i = rows[k]
						if !is_dropped[i]
							n_edges_observed += 1
						end
					end
				end

		#	Largest weakly-connected component size via BFS
			#	Standard iterative BFS with a queue (Vector{Int} used as a
			#	circular-free queue via an explicit head pointer); explores
			#	both column-direction (out-neighbors) and via row-scan
			#	(in-neighbors) to give weak connectivity on directed input.
			#	For undirected input the matrix is symmetric so this is
			#	identical to ordinary BFS, just with each edge enumerated
			#	from both endpoints — harmless duplicate visits caught by
			#	the visited mask.
				if n_observed == 0
					return _emit_degeneracy(0.0, 0, n_edges_observed, gc_threshold, min_n, min_edges)
				end
				visited      = falses(n)
				largest_size = 0
				#	Pre-allocate queue; reused across BFS passes
					queue = Vector{Int}(undef, n_observed)
				for start in 1:n
					(is_dropped[start] || visited[start]) && continue
					#	BFS from start
						queue[1]   = start
						head, tail = 1, 1
						visited[start] = true
						comp_size      = 1
						while head <= tail
							v   = queue[head]
							head += 1
							#	Out-neighbors of v (column v of CSR gives nodes
							#	with edges INTO v in the package's convention;
							#	for weak connectivity we want both directions).
							#	Iterate column v for in-neighbors of v:
								for k in nzrange(adj, v)
									w = rows[k]
									if !is_dropped[w] && !visited[w]
										visited[w] = true
										tail += 1
										queue[tail] = w
										comp_size  += 1
									end
								end
							#	Iterate row v for out-neighbors of v. On a CSC
							#	matrix this is the expensive direction (no
							#	direct row access); we walk all columns and
							#	check whether row v is nonzero in that column.
							#	Acceptable for the network sizes in the corpus;
							#	if it becomes a bottleneck the alternative is
							#	to symmetrize once outside the BFS.
								for j in 1:n
									is_dropped[j] && continue
									visited[j]    && continue
									#	Is (v, j) an edge? Search column j for row v.
										for k in nzrange(adj, j)
											if rows[k] == v
												visited[j] = true
												tail += 1
												queue[tail] = j
												comp_size  += 1
												break
											end
										end
								end
						end
						if comp_size > largest_size
							largest_size = comp_size
						end
				end

		#	gc fraction of remaining (guard against n_observed == 0 already handled above)
			gc_fraction = largest_size / n_observed

		#	Return the result via the shared emitter
			return _emit_degeneracy(gc_fraction, n_observed, n_edges_observed,
									gc_threshold, min_n, min_edges)
	end

#	Helper: assemble the degeneracy NamedTuple from raw quantities + thresholds
	function _emit_degeneracy(gc_fraction::Real,
								n_observed::Integer,
								n_edges_observed::Integer,
								gc_threshold::Real,
								min_n::Integer,
								min_edges::Integer)
		"""
		Args:
			gc_fraction::Real: largest-component fraction of remaining nodes
			n_observed::Integer: remaining node count
			n_edges_observed::Integer: remaining edge count
			gc_threshold::Real: threshold for the gc_collapse flag
			min_n::Integer: floor for the too_small flag
			min_edges::Integer: floor for the no_edges flag
		Returns:
			NamedTuple matching the degeneracy field layout in the per-replicate
			record (Section 4.3 of the validation roadmap).
		Notes:
			Pure assembly; no computation. Factored out so _topological_degeneracy
			has a single return statement and so the field layout is in one
			place if it ever needs to change.
		"""
		gc_collapse = gc_fraction < gc_threshold
		too_small   = n_observed < min_n
		no_edges    = n_edges_observed < min_edges
		any_topo    = gc_collapse || too_small || no_edges
		return (gc_fraction_of_remaining = Float64(gc_fraction),
				n_observed               = Int(n_observed),
				n_edges_observed         = Int(n_edges_observed),
				gc_collapse              = gc_collapse,
				too_small                = too_small,
				no_edges                 = no_edges,
				any_topo_degenerate      = any_topo)
	end

#	Generate Missingness Mask: full per-replicate record for one (network, target) draw
	function generate_missingness_mask(edges::DataFrame;
										nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
										directed::Bool,
										target_rate::Real,
										target_rho::Real,
										seed::Integer,
										centrality::Union{Nothing,AbstractVector{<:Real}} = nothing,
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1,
										bisection_tol::Real      = 0.02,
										bisection_max_iters::Int = 50,
										bisection_M::Int         = 50)
		"""
		Args:
			edges::DataFrame: original edge list with :src, :dst (weights ignored)
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
				(includes isolates)
			directed::Bool: true for directed networks, false for undirected;
				selects the centrality driver (binarized in-degree vs degree)
			target_rate::Real: target fraction of nodes to drop; in (0, 1)
			target_rho::Real: target induced correlation between prob_i and
				centrality; in (-1, 1)
			seed::Integer: master seed for this replicate; deterministically
				split into bisection and sampling sub-seeds
			centrality::Union{Nothing,Vector{<:Real}}: optional precomputed
				centrality vector. When nothing, computed via
				_centrality_for_sampler internally. Pass the cached value when
				calling repeatedly for one network (the orchestrator
				build_degeneration_corpus does this).
			gc_threshold::Real: passed through to _topological_degeneracy
			min_n::Int: passed through to _topological_degeneracy
			min_edges::Int: passed through to _topological_degeneracy
			bisection_tol::Real: convergence tolerance, passed to bisection
			bisection_max_iters::Int: iteration cap, passed to bisection
		Returns:
			NamedTuple per Section 4.3 of the validation roadmap:
				(dropped_nodes, nominal, realized_rate, realized_rho,
				 bisection_status, sampler_degeneracy, seed)
		Notes:
			This is the public per-replicate sampler. One call produces one
			degraded network's full record. The orchestrator
			build_degeneration_corpus invokes this in a loop over the
			(network, rho, rate, replicate) grid.

			Seed splitting. The master seed produces two sub-seeds via Xoshiro
			advancement — one consumed by _bisect_b_for_target_rho (for the
			deterministic prob-vector construction at each candidate b), the
			other consumed by _sample_missingness (for the prob vector and
			weighted-without-replacement draw that produces dropped_nodes).
			The splitting is itself deterministic, so the same master seed
			always produces the same record (Test 4).

			Centrality caching. When the caller knows it will call this
			function many times for the same network (the grid orchestrator
			does, 30 times), it should compute centrality once via
			_centrality_for_sampler and pass it in. When called one-off (tests,
			diagnostics), pass nothing and accept the per-call recompute.

			Record fields. The returned NamedTuple matches the per-replicate
			record specified in the validation roadmap Section 4.3:
				- dropped_nodes::Vector{Int}    sorted ascending
				- nominal::NamedTuple           (rate, rho) target
				- realized_rate::Float64        actual fraction dropped
				- realized_rho::Float64         realized cor(is_dropped, c)
				- bisection_status::Symbol      :converged, :ceiling_hit, :failed_other
				- sampler_degeneracy::NT        from _topological_degeneracy
				- seed::Int                     the master seed for this record
			The measure_degeneracy and framework_degeneracy fields specified
			in Section 4.3 are NOT populated here — they're populated by the
			measure code and framework code respectively when they run.

			Failure modes. When the bisection returns :ceiling_hit, sampling
			still proceeds at b = 1 (the closest the formula can get to
			target rho), and the realized_rho will reflect the achievable
			ceiling. When the bisection returns :failed_other (constant
			centrality, max_iters exceeded, etc.), sampling is NOT performed
			and the returned record has empty dropped_nodes, NaN realized
			values, and degeneracy fields filled with placeholders. The
			caller should treat :failed_other records as unusable and either
			retry with a different seed or exclude from analysis.
		"""

		#	Guards
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			(-1.0 < target_rho < 1.0) || throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))

		#	Centrality: compute or accept cached
			c = isnothing(centrality) ?
				_centrality_for_sampler(edges; nodes=nodes, directed=directed) :
				centrality
			n = length(c)

		#	Seed Splitting: master seed → (bisection_seed, sample_seed)
			#	Advance a master RNG twice to produce two independent UInt
			#	sub-seeds. This is deterministic in `seed` and gives the two
			#	internal functions non-overlapping random streams.
				master_rng     = Xoshiro(seed)
				bisection_seed = Int(rand(master_rng, UInt32))
				sample_seed    = Int(rand(master_rng, UInt32))

		#	Bisection: find b such that realized cor(is_dropped, c) ~ |target_rho|
			bisection = _bisect_b_for_target_rho(c, target_rho, target_rate, bisection_seed;
												  tol = bisection_tol,
												  max_iters = bisection_max_iters,
												  M = bisection_M)

		#	Failure-mode short-circuit: :failed_other yields a placeholder record
			if bisection.status == :failed_other
				#	Need adj for the degeneracy-fields placeholder; cheaper to
				#	skip and emit zeros than to build adj on a failed record.
				#	Caller treats :failed_other as unusable.
					return (dropped_nodes      = Int[],
							nominal            = (rate = target_rate, rho = target_rho),
							realized_rate      = NaN,
							realized_rho       = NaN,
							bisection_status   = :failed_other,
							sampler_degeneracy = (gc_fraction_of_remaining = NaN,
													n_observed             = 0,
													n_edges_observed       = 0,
													gc_collapse            = true,
													too_small              = true,
													no_edges               = true,
													any_topo_degenerate    = true),
							seed               = Int(seed))
			end

		#	Sampling: draw the dropped-node set at the bisected b
			draw = _sample_missingness(c, target_rate, bisection.b, bisection.sign, sample_seed)

		#	Build adj for the topological degeneracy check
			#	Use the same binarized adjacency the centrality computation
			#	used; pass weighted=false explicitly.
				adj, _, _ = isnothing(nodes) ?
					_graph_to_sparse_matrix(edges; weighted=false) :
					_graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)

		#	Topological degeneracy on the degraded network
			degen = _topological_degeneracy(adj, draw.dropped_nodes;
											gc_threshold = gc_threshold,
											min_n        = min_n,
											min_edges    = min_edges)

		#	Assemble per-replicate record
			return (dropped_nodes      = draw.dropped_nodes,
					nominal            = (rate = target_rate, rho = target_rho),
					realized_rate      = draw.realized_rate,
					realized_rho       = draw.realized_rho,
					bisection_status   = bisection.status,
					sampler_degeneracy = degen,
					seed               = Int(seed))
	end
	@doc raw"""
	**Description**
	Generates one per-replicate missingness record for the Phase~1 degeneration
	grid. Given a network and a target $(\text{rate}, \rho)$ cell, this function
	composes the full sampler pipeline --- centrality computation, bisection
	for the mixing scalar~$b$, weighted-without-replacement sampling of the
	dropped-node set, and topological-degeneracy detection --- into a single
	record matching the schema in Section~4.3 of the validation roadmap.

	**Usage**
	`generate_missingness_mask(edges; nodes=nothing, directed=true, target_rate=0.10, target_rho=0.25, seed=1, ...)`

	**Arguments**
	- `edges::DataFrame`: original edge list with `:src`, `:dst`. Weights ignored.
	- `nodes::Union{Nothing,DataFrame,Vector}`: optional node universe.
	- `directed::Bool`: directedness of the network (selects centrality driver).
	- `target_rate::Real`: nominal missingness rate, in $(0, 1)$.
	- `target_rho::Real`: nominal induced correlation, in $(-1, 1)$.
	- `seed::Integer`: master seed; deterministically split into bisection
	  and sampling sub-seeds.
	- `centrality::Union{Nothing,Vector}`: precomputed centrality vector;
	  when `nothing`, computed internally. Cache and reuse across all 30
	  $(\rho, \text{rate})$ cells of a single network.
	- `gc_threshold`, `min_n`, `min_edges`: passed to topological degeneracy.
	- `bisection_tol`, `bisection_max_iters`: passed to bisection.

	**Value**
	`NamedTuple` per Section~4.3:
	- `dropped_nodes::Vector{Int}` --- sorted ascending
	- `nominal::NamedTuple` --- `(rate, rho)` target
	- `realized_rate::Float64`
	- `realized_rho::Float64`
	- `bisection_status::Symbol` --- `:converged`, `:ceiling_hit`, or `:failed_other`
	- `sampler_degeneracy::NamedTuple` --- from `_topological_degeneracy`
	- `seed::Int` --- the master seed for this record

	The companion fields `measure_degeneracy` and `framework_degeneracy`
	(Section~4.3) are populated by the measure code and framework code when
	they run, not by this function.

	**Notes**
	Deterministic in the master seed: identical $(edges, nodes, directed,
	\text{rate}, \rho, \text{seed})$ inputs produce identical records (Test~4).

	When `bisection_status == :ceiling_hit`, sampling still proceeds at $b = 1$
	and `realized_rho` reflects the achievable ceiling. When
	`bisection_status == :failed_other`, sampling is skipped and the record
	is a placeholder marked as fully degenerate; the caller should treat such
	records as unusable.

	**See Also**
	`apply_missingness`, `build_degeneration_corpus`,
	`_bisect_b_for_target_rho`, `_sample_missingness`,
	`_topological_degeneracy`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling
	  coverage III. *Social Networks*, 68, 148--178.
	""" generate_missingness_mask

#	Apply Missingness: materialize degraded (edges, nodes) from original + dropped set
	function apply_missingness(edges::DataFrame,
								dropped_nodes::AbstractVector{<:Integer};
								nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing)
		"""
		Args:
			edges::DataFrame: original edge list with :src, :dst, optionally
				:weight and other columns; all columns preserved in the output
			dropped_nodes::AbstractVector{<:Integer}: node indices to drop, in
				the canonical node order (matching _centrality_for_sampler /
				_graph_to_sparse_matrix). Need not be sorted.
			nodes::Union{Nothing,DataFrame,Vector}: original node universe.
				When nothing, the node universe is inferred from the edges
				(no isolates), matching _graph_to_sparse_matrix's convention.
		Returns:
			NamedTuple: (edges, nodes)
				- edges::DataFrame: filtered edge list. Same columns as input;
					rows where either endpoint was dropped are removed.
				- nodes::DataFrame: filtered node list. Same columns as input;
					rows corresponding to dropped indices removed. When `nodes`
					argument was a Vector, the returned `nodes` is a DataFrame
					with a single :name column (matching internal conventions).
		Notes:
			Surviving node IDs are NOT re-indexed. The analysis compares
			degraded-network per-node measures against full-network per-node
			measures, which requires the surviving nodes to keep their
			original identifiers. Re-indexing would break this comparison
			and is not what the validation pipeline wants.

			Edge filtering: an edge is dropped iff either endpoint is dropped.
			This is listwise deletion at the edge level, the standard
			convention from SMM 2022 and Galaskiewicz (1991).

			The node universe convention matches _graph_to_sparse_matrix: if
			the caller passes `nodes`, that order defines the canonical
			indexing; otherwise the index is derived from edges. Either way
			`dropped_nodes` must use the same indexing the centrality vector
			used.

			Edge-case: when dropped_nodes is empty, the returned edges and
			nodes are identical to the input (modulo DataFrame copy
			semantics). When dropped_nodes contains every index, the
			returned edges DataFrame is empty (zero rows, same columns) and
			the nodes DataFrame is empty.

			This is the full-node-removal mechanism: both the node and ALL
			of its edges (incoming and outgoing) are removed. For the SMM-
			style outgoing-edge-only mechanism, where the node remains in
			the roster with its incoming edges intact, use
			apply_missingness_outgoing_only.
		"""

		#	Resolve canonical node order
			#	Mirror _graph_to_sparse_matrix's node-ordering logic so that
			#	dropped_nodes indices match the centrality vector indices.
				if isnothing(nodes)
					node_names = sort!(unique(vcat(string.(edges.src), string.(edges.dst))))
				elseif nodes isa DataFrame
					#	Convention: the first :name column (or :id, :node, etc.)
					#	is the canonical order. Match what _graph_to_sparse_matrix
					#	does — fall back to the first String-typed column if no
					#	well-known name is present.
						node_names = string.(nodes[!, 1])
				else
					node_names = string.(collect(nodes))
				end
				n = length(node_names)

		#	Guards
			@inbounds for idx in dropped_nodes
				(1 <= idx <= n) || throw(ArgumentError("dropped node index $idx out of range [1, $n]"))
			end

		#	Build dropped-name set for O(1) edge filtering
			is_dropped_idx = falses(n)
			@inbounds for idx in dropped_nodes
				is_dropped_idx[idx] = true
			end
			dropped_name_set = Set{String}(node_names[i] for i in 1:n if is_dropped_idx[i])

		#	Filter edges: keep rows where neither endpoint is dropped
			keep_edge = BitVector(undef, nrow(edges))
			@inbounds for r in 1:nrow(edges)
				s = string(edges.src[r])
				d = string(edges.dst[r])
				keep_edge[r] = !(s in dropped_name_set || d in dropped_name_set)
			end
			degraded_edges = edges[keep_edge, :]

		#	Filter nodes: keep rows for surviving indices
			keep_node = .!is_dropped_idx
			if isnothing(nodes)
				#	No nodes DataFrame supplied; emit a minimal one
					degraded_nodes = DataFrame(name = node_names[keep_node])
			elseif nodes isa DataFrame
				degraded_nodes = nodes[keep_node, :]
			else
				degraded_nodes = DataFrame(name = node_names[keep_node])
			end

		#	Return
			return (edges = degraded_edges, nodes = degraded_nodes)
	end

	@doc raw"""
	**Description**
	Materializes the degraded $(\text{edges}, \text{nodes})$ pair from an
	original network and a dropped-node set. Pure data manipulation: no
	random draws, no degeneracy checks. Called just before the measure
	battery runs, implementing the lazy-materialization design --- degraded
	networks are constructed on demand from the original plus the dropped-node
	set, not stored to disk.

	This is the full-node-removal mechanism: the dropped nodes are removed
	from the node roster along with all of their edges, incoming and
	outgoing. For the SMM-style mechanism where non-respondents remain in
	the roster with their incoming edges preserved and only outgoing edges
	removed, use `apply_missingness_outgoing_only`.

	**Usage**
	`apply_missingness(edges, dropped_nodes; nodes=nothing)`

	**Arguments**
	- `edges::DataFrame`: original edge list. All columns are preserved.
	- `dropped_nodes::AbstractVector{<:Integer}`: indices in the canonical
	  node order (matching `_centrality_for_sampler`).
	- `nodes::Union{Nothing,DataFrame,Vector}`: original node universe.

	**Value**
	`NamedTuple(edges, nodes)` where both are filtered to surviving nodes
	and surviving edges. Surviving node IDs are not re-indexed --- they
	retain their original identifiers, which the comparison against the
	full-network measures requires.

	**See Also**
	`apply_missingness_outgoing_only`, `generate_missingness_mask`,
	`build_degeneration_corpus`

	**References**
	- Galaskiewicz, J. (1991). Estimating point centrality using different
	  network sampling techniques. *Social Networks*, 13(4), 347--386.
	""" apply_missingness

#	Apply Missingness (Outgoing-Only Variant): SMM-style materialization
	function apply_missingness_outgoing_only(edges::DataFrame,
												dropped_nodes::AbstractVector{<:Integer};
												nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
												directed::Bool = true)
		"""
		Args:
			edges::DataFrame: original edge list with :src, :dst, optionally
				:weight and other columns
			dropped_nodes::AbstractVector{<:Integer}: indices in canonical
				node order; the same vector produced by _sample_missingness
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe;
				canonical-order convention identical to apply_missingness
			directed::Bool: must be true for this mechanism; the function
				throws on undirected input. Default true. See Notes.
		Returns:
			NamedTuple: (edges, nodes)
				- edges::DataFrame: filtered edge list. Edges where SRC is
					a dropped node are removed; edges where DST is a dropped
					node are PRESERVED (these are the observed actors'
					nominations TO the non-respondent, which the survey
					instrument captured even though the non-respondent
					didn't reply themselves).
				- nodes::DataFrame: UNCHANGED from input. The non-respondent
					nodes remain in the network with their in-degree intact
					but their out-degree zeroed.
		Notes:
			This is the SMM 2022 missing-data mechanism: non-respondents are
			treated as actors who failed to provide nomination data, but
			whose existence is known because others nominated them. The
			degraded network is the same node set as the original (the
			roster is intact) with only the outgoing-edge information from
			non-respondents removed.

			Contrast with apply_missingness, which removes the entire node
			(both outgoing and incoming edges plus the node itself from the
			node list). The two mechanisms correspond to different missing-
			data scenarios in the literature: apply_missingness corresponds
			to fully unobserved nodes (the Stage 1 case in the design
			document), apply_missingness_outgoing_only corresponds to
			nominated non-respondents (the Stage 0.5 case in the design
			document).

			This function is for directed networks only. Undirected
			networks have no 'outgoing' direction — every edge is mutual by
			definition — so the SMM mechanism is undefined. Calling this
			function with directed=false throws an ArgumentError; the
			caller should use apply_missingness on undirected networks
			instead, with documentation noting the mechanism mismatch.

			Selection invariance: this function consumes the SAME dropped_nodes
			vector that _sample_missingness produces and that apply_missingness
			would use. The selection of which nodes are non-respondents is
			driven by the same centrality-correlated sampler; only the
			materialization of the degraded network differs between the two
			mechanisms. This means a single validation grid can produce both
			mechanism variants for each (network, rho, rate, replicate)
			cell by running both materializers on the same sampler output.

			Edge cases:
			- dropped_nodes is empty: returned edges and nodes are identical
			  to input (modulo DataFrame copy semantics).
			- dropped_nodes contains every index: returned edges is empty
			  (every edge has at least one dropped src) and nodes is unchanged.
			- An edge is dropped iff src is in dropped_nodes; dst membership
			  is irrelevant. An edge from a non-respondent A to another
			  non-respondent B is removed (because A is dropped); an edge
			  from a respondent A to non-respondent B is kept (because A
			  is not dropped).
		"""

		directed || throw(ArgumentError("apply_missingness_outgoing_only is defined for directed networks only; use apply_missingness for undirected networks"))

		#	Resolve canonical node order (same convention as apply_missingness)
			if isnothing(nodes)
				node_names = sort!(unique(vcat(string.(edges.src), string.(edges.dst))))
			elseif nodes isa DataFrame
				node_names = string.(nodes[!, 1])
			else
				node_names = string.(collect(nodes))
			end
			n = length(node_names)

		#	Guard dropped_nodes indices
			@inbounds for idx in dropped_nodes
				(1 <= idx <= n) || throw(ArgumentError("dropped node index $idx out of range [1, $n]"))
			end

		#	Build dropped-name set keyed by SRC only
			is_dropped_idx = falses(n)
			@inbounds for idx in dropped_nodes
				is_dropped_idx[idx] = true
			end
			dropped_src_set = Set{String}(node_names[i] for i in 1:n if is_dropped_idx[i])

		#	Filter edges: keep rows where SRC is NOT a dropped node
			#	dst is checked nowhere; observed actors' nominations to
			#	non-respondents survive as in-degree to the non-respondent.
				keep_edge = BitVector(undef, nrow(edges))
				@inbounds for r in 1:nrow(edges)
					keep_edge[r] = !(string(edges.src[r]) in dropped_src_set)
				end
				degraded_edges = edges[keep_edge, :]

		#	Nodes: unchanged (the roster includes everyone, respondents and not)
			if isnothing(nodes)
				degraded_nodes = DataFrame(name = node_names)
			elseif nodes isa DataFrame
				degraded_nodes = nodes
			else
				degraded_nodes = DataFrame(name = node_names)
			end

		#	Return
			return (edges = degraded_edges, nodes = degraded_nodes)
	end
	@doc raw"""
	**Description**
	Materializes the SMM-style degraded $(\text{edges}, \text{nodes})$ pair
	from an original directed network and a dropped-node set. Non-respondents
	remain in the node roster but their outgoing edges are removed; their
	incoming edges (from observed actors who nominated them) are preserved.
	Pure data manipulation; no randomness.

	**Usage**
	`apply_missingness_outgoing_only(edges, dropped_nodes; nodes=nothing, directed=true)`

	**Arguments**
	- `edges::DataFrame`: original edge list. All columns preserved.
	- `dropped_nodes::AbstractVector{<:Integer}`: indices of non-respondents
	  in the canonical node order, identical in semantics to the dropped_nodes
	  consumed by `apply_missingness`.
	- `nodes::Union{Nothing,DataFrame,Vector}`: optional node universe.
	- `directed::Bool`: must be `true`; defined for directed networks only.

	**Value**
	`NamedTuple(edges, nodes)` where `edges` has rows removed for any edge
	whose source is a non-respondent. The `nodes` DataFrame is returned
	unchanged --- non-respondents remain in the roster as actors with zeroed
	out-degree.

	**Notes**
	This is the missing-data mechanism described in Smith, Morgan, and
	Moody (2022) for survey actor non-response: the non-respondent's roster
	entry remains, their incoming nominations from others are still recorded
	on the survey, but their own nomination data is unrecorded. The selection
	of non-respondents is unchanged from `apply_missingness` --- both functions
	consume the same dropped-node set produced by the centrality-correlated
	sampler.

	**See Also**
	`apply_missingness`, `generate_missingness_mask`, `build_degeneration_corpus`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). *Social Networks*, 68.
	""" apply_missingness_outgoing_only

#	Build Degeneration Corpus: orchestrate the full grid across networks, rhos, rates,
#	replicates, AND mechanisms (full_removal and/or outgoing_only)
	function build_degeneration_corpus(networks::Dict;
										target_rhos::AbstractVector{<:Real} = [-0.75, -0.25, 0.0, 0.25, 0.75],
										target_rates::AbstractVector{<:Real} = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
										n_replicates::Int = 100,
										replicates_per_network::Dict{String,Int} = Dict{String,Int}(),
										mechanisms::Vector{Symbol} = [:full_removal],
										master_seed::Integer = 42,
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1,
										bisection_tol::Real      = 0.02,
										bisection_max_iters::Int = 50,
										bisection_M::Int         = 50,
										parallel::Bool     = true,
										show_progress::Bool = true)
		"""
		Args:
			networks::Dict: corpus keyed by network name → NamedTuple with
				:edges (DataFrame), :nodes (DataFrame), :metadata (NamedTuple
				containing at least :directed::Bool)
			target_rhos::AbstractVector{<:Real}: nominal rho grid
			target_rates::AbstractVector{<:Real}: nominal rate grid
			n_replicates::Int: default replicate count per cell (default 100)
			replicates_per_network::Dict{String,Int}: per-network override
			mechanisms::Vector{Symbol}: which materialization mechanisms to
				run. Valid values: :full_removal (apply_missingness) and
				:outgoing_only (apply_missingness_outgoing_only). Default
				[:full_removal] preserves single-mechanism behavior. Pass
				[:full_removal, :outgoing_only] for the doubled-mechanism
				grid. :outgoing_only is silently skipped on undirected
				networks (a startup diagnostic lists which networks will
				be skipped).
			master_seed::Integer: master seed; per-replicate seeds derived
				deterministically from (name, rho, rate, rep, master_seed)
				— mechanism NOT included in the seed because both mechanisms
				on a cell consume the same sampler output
			gc_threshold, min_n, min_edges: passed to topological degeneracy
			bisection_tol, bisection_max_iters, bisection_M: passed to bisection
			parallel::Bool: parallelize the flat-grid loop (default true)
			show_progress::Bool: display a progress bar (default true)
		Returns:
			DataFrame with one row per (network, target_rho, target_rate,
			replicate_idx, mechanism) tuple. Columns:
				- network_name::String
				- nominal_rho::Float64
				- nominal_rate::Float64
				- replicate_idx::Int
				- mechanism::Symbol             (:full_removal or :outgoing_only)
				- seed::Int
				- dropped_nodes::Vector{Int}
				- realized_rate::Float64
				- realized_rho::Float64
				- bisection_status::Symbol
				- gc_fraction_of_remaining::Float64
				- n_observed::Int
				- n_edges_observed::Int
				- gc_collapse::Bool
				- too_small::Bool
				- no_edges::Bool
				- any_topo_degenerate::Bool
		Notes:
			The doubled-mechanism design exploits selection invariance: both
			:full_removal and :outgoing_only consume the same dropped_nodes
			vector from the same sampler call. The orchestrator runs
			generate_missingness_mask once per (network, rho, rate, rep)
			cell and emits one row per mechanism that applies to that
			network. This means the doubled grid's marginal cost is just
			the materializer calls (cheap) plus the doubled output rows,
			NOT a doubling of sampler calls.

			Degeneracy values (gc_fraction_of_remaining, n_observed, etc.)
			are sampler-level quantities computed against the original
			adjacency and the dropped-node mask. They are mechanism-agnostic:
			the two rows for the same cell will share these values but
			differ on mechanism. This reflects the design's separation of
			concerns — sampler degeneracy describes the dropped set;
			measure degeneracy (computed when measures run on the materialized
			network) is a separate downstream concern.

			Directedness handling: :outgoing_only is undefined for undirected
			networks. When the corpus contains undirected networks AND
			:outgoing_only is requested, those network-mechanism pairs are
			silently skipped. A startup diagnostic lists which networks
			will be skipped (so the user can verify the configuration
			matches their intent).

			Row ordering: name → rho → rate → replicate → mechanism, with
			mechanism varying fastest. Consecutive output rows for the
			same cell are the two mechanism variants — convenient for
			downstream diffing.

			Seed scheme: identical to the prior single-mechanism design.
			rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32).
			Mechanism is NOT in the seed because both mechanisms on a cell
			use the same sampler output.
		"""

		#	Guards
			isempty(networks) && throw(ArgumentError("networks dict is empty"))
			n_replicates >= 1 || throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))
			isempty(mechanisms) && throw(ArgumentError("mechanisms must be non-empty"))
			for m in mechanisms
				m in (:full_removal, :outgoing_only) ||
					throw(ArgumentError("invalid mechanism $m; valid: :full_removal, :outgoing_only"))
			end

		#	Compute centrality once per network
			network_names = sort!(collect(keys(networks)))
			centrality_cache = Dict{String, Vector{Float64}}()
			for name in network_names
				net = networks[name]
				centrality_cache[name] = _centrality_for_sampler(net.edges;
																  nodes    = net.nodes,
																  directed = net.metadata.directed)
			end

		#	Determine per-network applicable mechanisms
			#	:outgoing_only requires directed; skip silently on undirected
				mechanisms_for_network = Dict{String, Vector{Symbol}}()
				skipped_undirected = String[]
				for name in network_names
					net = networks[name]
					applicable = Symbol[]
					for m in mechanisms
						if m == :outgoing_only && !net.metadata.directed
							push!(skipped_undirected, name)
						else
							push!(applicable, m)
						end
					end
					mechanisms_for_network[name] = applicable
				end
				if show_progress && !isempty(skipped_undirected)
					unique_skipped = unique(skipped_undirected)
					println("Note: :outgoing_only skipped on undirected networks: ", join(unique_skipped, ", "))
				end

		#	Enumerate flat grid
			#	Pre-count grid_size accounting for per-network mechanism applicability
				grid_size = 0
				for name in network_names
					n_reps = get(replicates_per_network, name, n_replicates)
					applicable = mechanisms_for_network[name]
					grid_size += length(target_rhos) * length(target_rates) * n_reps * length(applicable)
				end
				flat_tuples = Vector{Tuple{String,Float64,Float64,Int,Symbol}}(undef, grid_size)
				let idx = 1
					for name in network_names
						n_reps = get(replicates_per_network, name, n_replicates)
						applicable = mechanisms_for_network[name]
						for rho in target_rhos
							for rate in target_rates
								for rep in 1:n_reps
									for mech in applicable
										flat_tuples[idx] = (name, Float64(rho), Float64(rate), rep, mech)
										idx += 1
									end
								end
							end
						end
					end
				end

		#	Pre-allocate result storage
			results = Vector{NamedTuple}(undef, grid_size)

		#	Progress bar
			use_threads = parallel && Threads.nthreads() > 1 && grid_size > 1
			desc        = "Degeneration grid (" * string(grid_size) * " rows, " *
						  (use_threads ? "$(Threads.nthreads()) threads" : "serial") * ")"
			prog        = show_progress ? Progress(grid_size, desc = desc, enabled = true) : nothing
			prog_lock   = ReentrantLock()

		#	Sampler-call cache per (name, rho, rate, rep) cell
			#	Both mechanisms on the same cell consume the same sampler output.
			#	Rather than re-running generate_missingness_mask for each mechanism,
			#	we cache the record keyed by (name, rho, rate, rep) and apply
			#	the materializers downstream when the per-row degeneracy is computed.
			#	BUT: the current grid design computes degeneracy inside
			#	generate_missingness_mask, so the cache key is just the sampler call;
			#	the per-row output is constructed from the cached record. Cache
			#	is mutable Dict; thread-safe via per-key locking via lock_dict.
				sampler_cache = Dict{Tuple{String,Float64,Float64,Int}, NamedTuple}()
				cache_lock = ReentrantLock()

		#	Run flat-grid loop
			if use_threads
				Threads.@threads :static for k in 1:grid_size
					name, rho, rate, rep, mech = flat_tuples[k]
					cache_key = (name, rho, rate, rep)
					#	Get or compute the sampler record
						local record
						lock(cache_lock) do
							record = get(sampler_cache, cache_key, nothing)
						end
						if record === nothing
							net = networks[name]
							rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
							record = generate_missingness_mask(net.edges;
																  nodes       = net.nodes,
																  directed    = net.metadata.directed,
																  target_rate = rate,
																  target_rho  = rho,
																  seed        = rep_seed,
																  centrality  = centrality_cache[name],
																  gc_threshold        = gc_threshold,
																  min_n               = min_n,
																  min_edges           = min_edges,
																  bisection_tol       = bisection_tol,
																  bisection_max_iters = bisection_max_iters,
																  bisection_M         = bisection_M)
							lock(cache_lock) do
								sampler_cache[cache_key] = record
							end
						end
					#	Per-mechanism row
						rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
						results[k] = (name = name, rho = rho, rate = rate, rep = rep,
									   mechanism = mech, seed = rep_seed, record = record)
					if show_progress
						lock(prog_lock) do
							next!(prog)
						end
					end
				end
			else
				for k in 1:grid_size
					name, rho, rate, rep, mech = flat_tuples[k]
					cache_key = (name, rho, rate, rep)
					record = get(sampler_cache, cache_key, nothing)
					if record === nothing
						net = networks[name]
						rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
						record = generate_missingness_mask(net.edges;
															  nodes       = net.nodes,
															  directed    = net.metadata.directed,
															  target_rate = rate,
															  target_rho  = rho,
															  seed        = rep_seed,
															  centrality  = centrality_cache[name],
															  gc_threshold        = gc_threshold,
															  min_n               = min_n,
															  min_edges           = min_edges,
															  bisection_tol       = bisection_tol,
															  bisection_max_iters = bisection_max_iters,
															  bisection_M         = bisection_M)
						sampler_cache[cache_key] = record
					end
					rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
					results[k] = (name = name, rho = rho, rate = rate, rep = rep,
								   mechanism = mech, seed = rep_seed, record = record)
					if show_progress
						next!(prog)
					end
				end
			end

		#	Flatten per-replicate records into rectangular DataFrame columns
			out = DataFrame(
				network_name             = String[],
				nominal_rho              = Float64[],
				nominal_rate             = Float64[],
				replicate_idx            = Int[],
				mechanism                = Symbol[],
				seed                     = Int[],
				dropped_nodes            = Vector{Int}[],
				realized_rate            = Float64[],
				realized_rho             = Float64[],
				bisection_status         = Symbol[],
				gc_fraction_of_remaining = Float64[],
				n_observed               = Int[],
				n_edges_observed         = Int[],
				gc_collapse              = Bool[],
				too_small                = Bool[],
				no_edges                 = Bool[],
				any_topo_degenerate      = Bool[],
			)
			for r in results
				rec  = r.record
				sdeg = rec.sampler_degeneracy
				push!(out, (network_name             = r.name,
							nominal_rho              = r.rho,
							nominal_rate             = r.rate,
							replicate_idx            = r.rep,
							mechanism                = r.mechanism,
							seed                     = r.seed,
							dropped_nodes            = rec.dropped_nodes,
							realized_rate            = rec.realized_rate,
							realized_rho             = rec.realized_rho,
							bisection_status         = rec.bisection_status,
							gc_fraction_of_remaining = sdeg.gc_fraction_of_remaining,
							n_observed               = sdeg.n_observed,
							n_edges_observed         = sdeg.n_edges_observed,
							gc_collapse              = sdeg.gc_collapse,
							too_small                = sdeg.too_small,
							no_edges                 = sdeg.no_edges,
							any_topo_degenerate      = sdeg.any_topo_degenerate))
			end

		return out
	end
	@doc raw"""
	**Description**
	Orchestrates the full degeneration grid. Given a corpus of networks and
	a design grid (target $\rho$ values, target rates, replicate count,
	missingness mechanisms), this function produces a per-replicate record
	for every cell and materializer combination, and returns the results as
	a single rectangular DataFrame.

	The framework validates two missing-data mechanisms drawn from the
	literature: full node removal (the dropped nodes and all of their edges
	are removed from the network) and SMM-style outgoing-edge-only removal
	(non-respondents remain in the roster with their incoming edges preserved
	and only their outgoing edges removed). Both mechanisms operate on the
	same centrality-correlated dropped-node set produced by the sampler;
	only the materialization differs.

	**Usage**
	```julia
		corpus = build_degeneration_corpus(networks;
			target_rhos    = [-0.75, -0.25, 0.0, 0.25, 0.75],
			target_rates   = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
			n_replicates   = 100,
			mechanisms     = [:full_removal, :outgoing_only],
			master_seed    = 42,
		)
	```

	**Arguments**
	- `networks::Dict`: corpus keyed by network name $\to$ NamedTuple with
	  `:edges`, `:nodes`, `:metadata.directed`.
	- `target_rhos::AbstractVector{<:Real}`: the $\rho$ grid. Default is the
	  five levels in the validation roadmap, $\{-0.75, -0.25, 0.0, 0.25, 0.75\}$.
	- `target_rates::AbstractVector{<:Real}`: the rate grid. Default is the
	  six rates in the validation roadmap, $\{0.05, 0.10, 0.15, 0.25, 0.40, 0.50\}$.
	- `n_replicates::Int`: default replicate count per cell (default 100).
	- `replicates_per_network::Dict{String,Int}`: per-network override of
	  replicate count (e.g., to reduce Marvel's count for budget reasons).
	  Networks absent from this dict use `n_replicates`.
	- `mechanisms::Vector{Symbol}`: which materialization mechanisms to run
	  on each cell. Valid values are `:full_removal` and `:outgoing_only`.
	  Default `[:full_removal]` preserves single-mechanism behavior; pass
	  `[:full_removal, :outgoing_only]` for the doubled-mechanism grid.
	  `:outgoing_only` is silently skipped on undirected networks; a
	  startup diagnostic lists which networks will be skipped so the user
	  can verify the configuration matches their intent.
	- `master_seed::Integer`: master seed; per-replicate seeds derived
	  deterministically from $(\text{name}, \rho, \text{rate}, \text{rep}, \text{master})$.
	- `gc_threshold`, `min_n`, `min_edges`: passed to topological degeneracy
	  detection inside `generate_missingness_mask`.
	- `bisection_tol`, `bisection_max_iters`, `bisection_M`: passed to the
	  Monte Carlo bisection inside `generate_missingness_mask`.
	- `parallel::Bool`: parallelize the flat-grid loop (default true).
	- `show_progress::Bool`: display a progress bar and startup diagnostics
	  (default true).

	**Value**
	`DataFrame` with one row per
	$(\text{network}, \rho, \text{rate}, \text{rep}, \text{mechanism})$ tuple.
	Seventeen columns: identifiers and provenance (`network_name`,
	`nominal_rho`, `nominal_rate`, `replicate_idx`, `mechanism`, `seed`),
	the sampler output (`dropped_nodes`, `realized_rate`, `realized_rho`,
	`bisection_status`), and the topological degeneracy fields flattened
	from `sampler_degeneracy` (`gc_fraction_of_remaining`, `n_observed`,
	`n_edges_observed`, `gc_collapse`, `too_small`, `no_edges`,
	`any_topo_degenerate`).

	Row ordering: $\text{name} \to \rho \to \text{rate} \to \text{rep} \to \text{mechanism}$,
	with mechanism varying fastest. Consecutive rows for the same cell are
	the mechanism variants of that cell --- convenient for downstream
	diffing and per-cell summaries.

	**Notes**
	The DataFrame contains every replicate, including those flagged
	degenerate or with `bisection_status == :failed_other`. Filtering is the
	consumer's job: a coverage analysis might restrict to
	`any_topo_degenerate == false`, while a degeneracy-rate analysis wants
	all rows. Per-cell degeneracy rates (used for grid trimming) are
	computed via grouping on
	(`network_name`, `nominal_rho`, `nominal_rate`, `mechanism`) and
	summarizing `any_topo_degenerate`.

	The doubled-mechanism grid exploits selection invariance: both
	`:full_removal` and `:outgoing_only` consume the same dropped-node set
	from the same sampler call. The orchestrator runs the sampler once per
	$(\text{network}, \rho, \text{rate}, \text{rep})$ cell and emits one
	row per applicable mechanism. The marginal cost of adding `:outgoing_only`
	to the grid is the materializer overhead and the doubled output rows,
	NOT a doubling of sampler calls.

	The degeneracy columns describe the sampler-level structural state of
	the network when the dropped-node mask is applied to the original
	adjacency. They are mechanism-agnostic by design: both mechanism rows
	for the same cell share these values. This reflects the separation of
	concerns in the validation framework --- sampler degeneracy describes
	the dropped set, while measure degeneracy (computed downstream when
	the framework's measure code runs on the materialized network) is a
	separate concern tracked in its own output columns.

	Threading is enabled by default and uses `Threads.@threads :static`
	with pre-allocated per-tuple result storage. The sampler-record cache
	uses a `ReentrantLock` to coordinate read-then-compute-then-write
	across threads. Output is deterministic in `master_seed` regardless
	of thread count: the seed scheme is keyed by
	$(\text{name}, \rho, \text{rate}, \text{rep}, \text{master})$ with
	mechanism deliberately excluded, ensuring that the two mechanism rows
	for the same cell share the same seed and the same sampler record.

	**See Also**
	`generate_missingness_mask`, `apply_missingness`,
	`apply_missingness_outgoing_only`, `_centrality_for_sampler`,
	`_bisect_b_for_target_rho`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). Network sampling
	  coverage III: Imputation of missing network data under different
	  network and missing data conditions. *Social Networks*, 68, 148--178.
	""" build_degeneration_corpus

#   Exports (public API)
    export generate_missingness_mask,
           apply_missingness,
           build_degeneration_corpus

end # module network_degeneracy