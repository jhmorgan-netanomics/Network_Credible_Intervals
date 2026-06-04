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
    #   Sibling-relative imports — submodules cannot `using` their parent
    #   (the parent is still loading when this line runs). Pull in only the
    #   specific helpers this module needs.
    #
    #   network_reconstruction is included BEFORE network_degeneracy in the
    #   parent umbrella, so calling into it here is safe. The degeneracy module
    #   is the testing special case; reconstruction is the primary, user-facing
    #   module and owns the shared logistic-beta tilt machinery and the
    #   feasibility envelope.
        using ..network_community_detection: _graph_to_sparse_matrix
        using ..network_reconstruction: _solve_propensity_field, _topup_missing_nodes, _solve_bin_distribution, _centrality_for_sampler, 
                                        feasible_rho_range

#	Helper Centrality Correlation: Monte Carlo estimate of E[corkendall(is_dropped, c)] at candidate b
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
			sgn::Integer: +1 to target positive realized Kendall tau (drop
				central nodes preferentially), -1 to target negative; must
				be +1 or -1
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
			Float64: empirical mean of corkendall(is_dropped, centrality)
				across M inner weighted-without-replacement draws. Returns
				NaN when the centrality vector is constant (max == min);
				the bisection short-circuits on that case via the
				:failed_other status.
		Notes:
			This function targets the realized RANK correlation (Kendall's
			tau-b) between is_dropped and centrality, not the value-based
			Pearson correlation. The rank-based formulation matches the
			user's intuition: "low-ranked nodes are likely missing" is a
			rank-order statement, not a linear-correlation statement.

			On heavy-tailed networks (e.g., Marvel) the Pearson formulation
			produces an asymmetric ceiling — the negative-rho side can
			only reach ~-0.03 because the many low-centrality nodes look
			alike under min-max normalization, making the rank-style
			"peripheral nodes missing" theory unrepresentable. Kendall
			tau-b treats both ends of the centrality distribution
			symmetrically via ranks, eliminating the asymmetric ceiling.

			Two coupled changes from the Pearson version:
			(1) Probability vector uses RANK-based normalization:
			    prob_i = b * r_normalized_i + (1 - b) * u_i
			    where r_normalized_i = (tiedrank(centrality)[i] - 1) / (N - 1)
			    is the rank position scaled to [0, 1]. Min-max value-based
			    normalization is replaced because, on heavy-tailed networks,
			    it puts almost all weight mass on a few hubs and leaves
			    the low end indistinguishable.
			(2) Realized correlation is Kendall's tau-b (StatsBase.corkendall)
			    rather than Pearson cor. Kendall tau-b handles the ties on
			    the binary is_dropped vector correctly.

			A consequence of using Kendall tau-b on a binary indicator:
			|tau| is structurally bounded by 2*p*(1-p) where p is the
			missingness rate. At rate=0.10, max |tau| ~= 0.18 regardless
			of network. This rate-bounded ceiling is reported as
			:ceiling_hit by the bisection caller.

			For negative-rho targets (sgn == -1), the prob vector is
			inverted via prob_i <- 1 - prob_i before sampling. The inner
			draws share one prob vector and differ only in which k nodes
			get sampled from it.

			Determinism scheme: each inner sample m in 1:M uses a seed
			derived as Int(hash((bisection_seed, iter_idx, m)) % UInt32).
			This makes the function reproducible in bisection_seed
			regardless of thread scheduling, and ensures that successive
			bisection iterations consume different inner draws.

			Cost: M weighted-without-replacement draws plus M Kendall
			tau-b computations. corkendall is O(N log N) via the Knight's
			algorithm; on N <= 10000 networks each inner sample is sub-
			10ms. At M=50 with ~30 bisection iterations per cell, the
			bisection adds ~1500 sampling+corkendall operations per
			generate_missingness_mask call. On Marvel (N=6486), expect
			~15 seconds per generate_missingness_mask call.

			Edge cases:
			- Constant centrality returns NaN: corkendall(is_dropped,
			  constant) is undefined. The bisection caller checks for NaN
			  and returns status = :failed_other.
			- Saturation: when fewer than k entries of prob are strictly
			  positive, weighted-without-replacement sampling cannot draw
			  k distinct nodes. The function falls back to selecting all
			  positive-weight nodes deterministically and uniform-randomly
			  filling the remaining k - n_positive slots from zero-weight
			  nodes. This regime exhausts the centrality signal; the
			  bisection should detect the resulting realized tau as below
			  target and return :ceiling_hit.
		"""
		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes"))
			(0.0 <= b <= 1.0) || throw(ArgumentError("b must be in [0, 1], got $b"))
			(sgn == 1 || sgn == -1) || throw(ArgumentError("sgn must be +1 or -1, got $sgn"))
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			M >= 1 || throw(ArgumentError("M must be >= 1, got $M"))

		#	Constant-Centrality Short-Circuit
			#	Kendall tau-b on a constant vector is undefined (no concordant
			#	or discordant pairs can be formed). Mirror the previous
			#	min-max short-circuit so the bisection caller's :failed_other
			#	branch fires correctly.
				c_min = minimum(centrality)
				c_max = maximum(centrality)
				if c_max == c_min
					return NaN
				end

		#	Rank-Based Normalization of Centrality
			#	tiedrank assigns Float64 ranks with ties handled by averaging.
			#	Subtracting 1 and dividing by n - 1 gives a [0, 1] range
			#	where rank-1 nodes (lowest centrality) map to 0 and rank-N
			#	nodes (highest centrality) map to 1. Unlike min-max value-
			#	based normalization, rank-based normalization is symmetric
			#	across the centrality distribution and doesn't concentrate
			#	weight on a few hubs on heavy-tailed networks.
				ranks = StatsBase.tiedrank(centrality)
				c_norm = (ranks .- 1.0) ./ (n - 1)

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
			#	When fewer than k entries are positive, StatsBase.sample
			#	cannot draw k distinct nodes from the positive support.
			#	The saturation path mirrors the standard-path math:
			#	positive nodes are deterministically dropped, remaining
			#	slots are filled uniformly from zero-weight nodes.
				n_positive = count(p -> p > 0.0, prob)

			tau_sum = 0.0
			centrality_vec = Vector{Float64}(centrality)
			is_dropped_buf = zeros(Float64, n)

			if n_positive >= k
				#	Standard path: weighted-without-replacement sampling
					weights = StatsBase.Weights(prob)
					@inbounds for m in 1:M
						sample_seed = Int(hash((bisection_seed, iter_idx, m)) % UInt32)
						rng = Xoshiro(sample_seed)
						dropped = StatsBase.sample(rng, 1:n, weights, k; replace=false)
						fill!(is_dropped_buf, 0.0)
						for idx in dropped
							is_dropped_buf[idx] = 1.0
						end
						tau_sum += StatsBase.corkendall(is_dropped_buf, centrality_vec)
					end
			else
				#	Saturation path: positives always dropped, plus uniform
				#	draws from zeros to fill remaining slots
					positive_idxs = findall(p -> p > 0.0, prob)
					zero_idxs     = findall(p -> p == 0.0, prob)
					k_remaining   = k - n_positive
					@inbounds for m in 1:M
						sample_seed = Int(hash((bisection_seed, iter_idx, m)) % UInt32)
						rng = Xoshiro(sample_seed)
						fill!(is_dropped_buf, 0.0)
						for idx in positive_idxs
							is_dropped_buf[idx] = 1.0
						end
						zero_dropped = StatsBase.sample(rng, zero_idxs, k_remaining; replace=false)
						for idx in zero_dropped
							is_dropped_buf[idx] = 1.0
						end
						tau_sum += StatsBase.corkendall(is_dropped_buf, centrality_vec)
					end
			end

		#	Return Mean Realized Kendall tau-b
			return tau_sum / M
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
				(positive target Kendall tau), -1 to drop low-centrality
				preferentially (negative target). MUST be +1 or -1.
			sample_seed::Integer: RNG seed for the u_i Uniform(0,1) draws and
				for the StatsBase.sample(...) call
		Returns:
			NamedTuple: (dropped_nodes, realized_rate, realized_rho)
				- dropped_nodes::Vector{Int}: indices of the dropped nodes in
					the canonical node order, sorted ascending
				- realized_rate::Float64: |dropped_nodes| / N
				- realized_rho::Float64: corkendall(is_dropped, centrality)
					across the node set, signed against the original
					centrality direction (NOT inverted by sgn). The field
					name is preserved for backward compatibility with
					downstream code; the metric is now Kendall's tau-b.
		Notes:
			Implements the weighted-without-replacement sampler with rank-
			based weighting and Kendall tau-b as the realized correlation
			metric. This is the production sampler called by
			generate_missingness_mask.

			Two coupled choices that distinguish this from a Pearson sampler:
			(1) Rank-based normalization. The probability vector is
			    constructed from ranks rather than raw centrality values:
			        prob_i = b * r_normalized_i + (1 - b) * u_i
			    where r_normalized_i = (tiedrank(centrality)[i] - 1) / (N - 1)
			    is the rank position scaled to [0, 1]. This gives all parts
			    of the centrality distribution equal weight-density, in
			    contrast to value-based min-max normalization which puts
			    almost all weight on a few hubs in heavy-tailed networks.
			(2) Realized correlation is Kendall's tau-b. This matches the
			    rank-based weighting and the user's rank-order intuition
			    about missingness.

			When sgn = -1 (negative target tau), the prob vector is
			inverted via prob_i <- 1 - prob_i before sampling, producing
			draws where low-rank nodes are preferentially selected.

			Rate-targeting is exact: k = round(target_rate * N) nodes are
			drawn via StatsBase.sample(..., weights=prob, replace=false).
			The realized rate is therefore round(target_rate * N) / N for
			every replicate.

			realized_rho is computed from the SAMPLED dropped-node set
			(is_dropped indicator vector) against the original centrality
			vector using corkendall — not from the prob vector and not
			from ranks. The mean of realized_rho across many seeds
			converges to the target_rho the bisection was solving for.
			Single-replicate realized_rho is noisy.

			The realized_rho is signed against the original centrality
			direction, so for negative-target draws it will be negative
			as expected. Kendall tau-b is invariant under monotone
			transformations of centrality, so it doesn't matter whether
			we pass raw centrality or ranks to corkendall here.

			RATE-BOUNDED CEILING. Kendall tau-b on a binary indicator is
			bounded by |tau| <= 2*p*(1-p) where p = mean(is_dropped). At
			target_rate = 0.10, max |tau| ~= 0.18 regardless of network
			structure. This is a structural property of Kendall tau-b on
			binary indicators, not a network artifact. Users specifying
			|target_rho| above this rate-bounded ceiling will have the
			bisection return :ceiling_hit; the framework will honor
			whatever realized correlation the saturated sampler produces.

			sample_seed should be distinct from any seed used in the
			bisection, otherwise the deterministic bisection result
			becomes entangled with the sampling stochasticity. The public
			wrapper generate_missingness_mask handles this seed-splitting.

			Edge cases:
			- If centrality is constant (max == min), the rank vector is
			  also constant (all ties tiedrank to (n+1)/2), and corkendall
			  with a constant variable is undefined. The caller should
			  have returned :failed_other on this network via the
			  bisection's constant-centrality short-circuit.
			- If b = 1.0 exactly and centrality is non-constant, prob_i
			  = r_normalized_i with no noise component. Saturation can
			  still occur if many nodes share the lowest rank (extreme
			  tie patterns), in which case the fallback handles the
			  draw deterministically.
		"""

		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes"))
			(0.0 < target_rate < 1.0) || throw(ArgumentError("target_rate must be in (0, 1), got $target_rate"))
			(0.0 <= b <= 1.0) || throw(ArgumentError("b must be in [0, 1], got $b"))
			(sgn == 1 || sgn == -1) || throw(ArgumentError("sgn must be +1 or -1, got $sgn"))

		#	Rank-Based Normalization of Centrality to [0, 1]
			#	Constant centrality produces an all-equal rank vector
			#	(all entries tiedrank to (n+1)/2). Resulting r_normalized
			#	is the zero vector, giving prob = (1-b)*u. The caller's
			#	bisection should have caught this and returned
			#	:failed_other before reaching here.
				c_min = minimum(centrality)
				c_max = maximum(centrality)
				if c_max == c_min
					c_norm = zeros(Float64, n)
				else
					ranks = StatsBase.tiedrank(centrality)
					c_norm = (ranks .- 1.0) ./ (n - 1)
				end

		#	Draw u_i and Form the Probability Vector
			rng  = Xoshiro(sample_seed)
			u    = rand(rng, n)
			prob = b .* c_norm .+ (1 - b) .* u

		#	Sign Flip for Negative Target tau
			#	Inverting via 1 - prob keeps the prob vector in [0, 1] and
			#	flips which nodes are preferred for dropping.
				if sgn == -1
					prob = 1.0 .- prob
				end

		#	Numerical Safety
			@inbounds for i in 1:n
				if prob[i] < 0.0
					prob[i] = 0.0
				end
			end

		#	Weighted-Without-Replacement Sample of k = round(target_rate * N)
			k = Int(round(target_rate * n))
			k >= 1                || throw(ArgumentError("target_rate too small: k = $k < 1"))
			k <= n - 1            || throw(ArgumentError("target_rate too large: k = $k > N - 1 = $(n - 1)"))

			n_positive = count(p -> p > 0.0, prob)
			if n_positive >= k
				#	Standard path
					dropped = StatsBase.sample(rng, 1:n, StatsBase.Weights(prob), k; replace=false)
			else
				#	Saturation path
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
			realized_rho  = StatsBase.corkendall(Float64.(is_dropped), Vector{Float64}(centrality))

		#	Return
			return (dropped_nodes  = dropped,
					realized_rate  = realized_rate,
					realized_rho   = realized_rho)
	end

#	Helper Function for _sample_weight_removal: multinomial counts over edges
	function _multinomial_counts(rng::AbstractRNG, probs::AbstractVector{<:Real}, n_draws::Integer)
		"""
		Args:
			rng::AbstractRNG: random stream.
			probs::AbstractVector{<:Real}: probability vector over edges, sums to ~1.
			n_draws::Integer: number of units to allocate.
		Returns:
			Vector{Int}: per-edge counts summing to n_draws.
		Notes:
			Thin wrapper over Distributions.Multinomial; isolated so the removal
			loop reads cleanly and so the draw is swappable. Falls back to a
			sequential categorical draw when n_draws is small relative to the
			edge count (cheaper than allocating the full Multinomial sampler for
			single-unit redistribution passes).
		"""

		#	Degenerate cases
			m = length(probs)
			counts = zeros(Int, m)
			(n_draws <= 0 || m == 0) && return counts

		#	Small-draw sequential path
			if n_draws <= 8
				w = StatsBase.Weights(collect(float.(probs)))
				@inbounds for _ in 1:n_draws
					counts[StatsBase.sample(rng, 1:m, w)] += 1
				end
				return counts
			end

		#	Bulk multinomial path
			d = Distributions.Multinomial(Int(n_draws), collect(float.(probs)))
			return rand(rng, d)
	end

#	Helper Function for _sample_weight_removal: rebuild a sparse matrix on a fixed pattern
	function _rebuild_sparse(template::SparseMatrixCSC,
								values::AbstractVector{<:Real},
								edge_i::AbstractVector{<:Integer},
								edge_j::AbstractVector{<:Integer})
		"""
		Args:
			template::SparseMatrixCSC: matrix whose size defines the output.
			values::AbstractVector{<:Real}: per-edge values in the (edge_i, edge_j)
				enumeration order produced by walking the template's CSC.
			edge_i, edge_j::AbstractVector{<:Integer}: row/col of each edge, same
				order as values.
		Returns:
			SparseMatrixCSC{Int,Int}: matrix with values placed at (edge_i, edge_j).
		Notes:
			Helper for _sample_weight_removal. Reconstructs the removal/weight
			matrix on the SAME sparsity pattern as the template so edges are
			preserved even at zero value (zeros retained). Built via sparse()
			with explicit I, J, V triples.
		"""
		n = size(template, 1)
		I = collect(Int, edge_i)
		J = collect(Int, edge_j)
		V = collect(Int, round.(Int, values))
		return sparse(I, J, V, n, n)
	end

#	Helper Weighted Adjacency: integer-count weighted matrix for the edge-degradation stage
	function _weighted_adjacency(edges::DataFrame;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing)
		"""
		Args:
			edges::DataFrame: edge list with :src, :dst and a :weight column.
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe
				(includes isolates), same convention as _graph_to_sparse_matrix.
		Returns:
			Tuple{SparseMatrixCSC{Float64,Int}, Vector{String}}: (adj, node_names)
				adj[i, j] = weight of edge from node i to node j (or the symmetric
				entry for undirected input, exactly as _graph_to_sparse_matrix lays
				it out). node_names is the canonical node order.
		Notes:
			The centrality driver deliberately binarizes (weighted=false). The
			edge-degradation stage needs the actual integer weights, so this
			builds a weighted adjacency via _graph_to_sparse_matrix with
			weighted=true. Edge weights are treated as integer counts (the
			Bellutta Chapter-3 weight-unit model); non-integer weights are
			accepted as-is but the weight-removal stage rounds the per-edge
			budget, so fractional weights degrade gracefully to their rounded
			counterparts.
		"""

		#	Build Weighted Adjacency
			adj, node_names, _ = isnothing(nodes) ?
				_graph_to_sparse_matrix(edges; weighted=true) :
				_graph_to_sparse_matrix(edges; nodes=nodes, weighted=true)

		#	Return
			return (adj, node_names)
	end

#	Helper Weight Removal: survival-tilted, budget-hitting integer weight-unit removal
	function _sample_weight_removal(adj::SparseMatrixCSC,
										d::AbstractVector{<:Real},
										pi_edge::Real,
										sample_seed::Integer)
		"""
		Args:
			adj::SparseMatrixCSC: weighted integer-count adjacency (from
				_weighted_adjacency), canonical node order.
			d::AbstractVector{<:Real}: per-node relative propensity from
				_solve_propensity_field (node-mean 1). Higher d_i => node i's
				incident weight is preferentially removed.
			pi_edge::Real: fraction of TOTAL true weight to remove; in [0, 1).
			sample_seed::Integer: seed for the multinomial removal draws.
		Returns:
			NamedTuple: (removed, W_true, W_removed, organic_losses, n_edges_zeroed)
				- removed::SparseMatrixCSC{Float64,Int}: same sparsity pattern and
					node order as adj; removed[i,j] = units removed from edge (i,j).
					Edge entries are PRESERVED even when fully removed (zeros
					retained); subtract from adj to get the degraded weights.
				- W_true::Int: total weight before removal.
				- W_removed::Int: total weight actually removed (= round(pi_edge*W_true)).
				- organic_losses::Vector{Int}: node indices whose incident weight is
					fully removed (zero remaining), sorted ascending.
				- n_edges_zeroed::Int: count of edges driven to zero weight.
		Notes:
			Implements the unified edge-degradation stage. Each weight unit on
			edge (i,j) is removed with probability proportional to its edge's
			survival-tilted hazard: edge weight w_ij carries a per-unit removal
			tendency tied to its endpoints' propensities via INDEPENDENT ENDPOINT
			SURVIVAL — the unit survives iff it survives at both endpoints, so
			the removal hazard scales with 1 - (1-s*d_i)(1-s*d_j) for a small base
			rate s. We implement this with the batched, current-weight-proportional
			draw from the Bellutta Chapter-3 prototype (induce_edge_weight_missingness):
			sample units proportional to (current weight * endpoint tilt), clip
			per-edge removals to available weight, decrement, and redistribute the
			clipped quota until the W_removed budget is met.

			At target_rho = 0 the propensity field is constant (d_i = 1 for all i),
			so the endpoint tilt is uniform and removal is exactly proportional to
			current weight — the Bellutta MCAR corner, matching the prototype.

			Edges are PRESERVED at zero weight (the dyad was observed-as-sampled);
			row count is invariant. A node whose every incident edge reaches zero
			is an ORGANIC LOSS (no observed incident weight) — these are the
			tie-degradation-induced missing nodes handed to the node-accounting
			stage. Because low-weight (weak-tie, peripheral) nodes have the least
			weight to lose, organic losses concentrate on the periphery by
			construction, even at rho = 0.

			Determinism: all draws derive from sample_seed via a single Xoshiro
			stream, so the removal is reproducible in sample_seed.

			Edge cases:
			- pi_edge = 0: returns an all-zero removal matrix, W_removed = 0, no
			  organic losses. Caller skips the edge stage in this case.
			- W_removed >= W_true: clamped to W_true (cannot remove more than
			  exists); every edge zeroed, every node an organic loss.
		"""

		#	Guards
			n = size(adj, 1)
			size(adj, 2) == n || throw(ArgumentError("adj must be square"))
			length(d) == n || throw(ArgumentError("d length $(length(d)) != N $n"))
			(0.0 <= pi_edge < 1.0) || throw(ArgumentError("pi_edge must be in [0, 1), got $pi_edge"))

		#	Materialize Edge List from Sparsity Pattern
			#	Walk the CSC once to collect (i, j, w) triples plus an endpoint
			#	tilt per edge. The tilt uses independent endpoint survival on the
			#	normalized propensities scaled into [0,1] via d/(2*max(d)) so that
			#	the constant-d (rho=0) case yields a uniform tilt and removal
			#	reduces to current-weight-proportional.
				rows = rowvals(adj)
				nzv  = nonzeros(adj)
				n_edges = nnz(adj)
				edge_i = Vector{Int}(undef, n_edges)
				edge_j = Vector{Int}(undef, n_edges)
				weight = Vector{Float64}(undef, n_edges)
				W_true = 0.0
				let e = 0
					@inbounds for j in 1:n
						for k in nzrange(adj, j)
							e += 1
							i = rows[k]
							edge_i[e] = i
							edge_j[e] = j
							weight[e] = nzv[k]
							W_true   += nzv[k]
						end
					end
				end
				W_true_int = Int(round(W_true))

		#	Budget
			W_removed = min(Int(round(pi_edge * W_true)), W_true_int)
			removed = zeros(Int, n_edges)

		#	Early exit on no-op
			if W_removed <= 0 || n_edges == 0
				#	Build empty removal matrix (same pattern) and return.
					rem_mat = _rebuild_sparse(adj, removed, edge_i, edge_j)
					return (removed = rem_mat, W_true = W_true_int, W_removed = 0,
							organic_losses = Int[], n_edges_zeroed = 0)
			end

		#	Endpoint Tilt per Edge (independent endpoint survival)
			#	s_i = d_i / (2*max(d)) in [0, 0.5]; survival at an endpoint is
			#	(1 - s); the edge's removal hazard is 1 - (1-s_i)(1-s_j). At
			#	constant d (rho=0) every s is equal so hazard is constant across
			#	edges => removal proportional to current weight only.
				dmax = maximum(d)
				dmax <= 0 && (dmax = 1.0)
				s = (d ./ (2.0 * dmax))
				tilt = Vector{Float64}(undef, n_edges)
				@inbounds for e in 1:n_edges
					si = s[edge_i[e]]
					sj = s[edge_j[e]]
					tilt[e] = 1.0 - (1.0 - si) * (1.0 - sj)
				end

		#	Batched Current-Weight-Proportional Removal (clip + redistribute)
			rng = Xoshiro(sample_seed)
			remaining = W_removed
			#	avail[e] = weight[e] - removed[e] tracked implicitly via removed
				safety_iters = 0
				while remaining > 0 && safety_iters < 10_000
					safety_iters += 1
					#	Build current sampling weights: (current weight) * tilt,
					#	zero where the edge is exhausted.
						probs = Vector{Float64}(undef, n_edges)
						total = 0.0
						@inbounds for e in 1:n_edges
							avail = weight[e] - removed[e]
							pe = avail > 0 ? avail * tilt[e] : 0.0
							probs[e] = pe
							total += pe
						end
						total <= 0 && break
					#	Draw `remaining` units multinomially over edges, then clip
					#	each edge's draw to its available weight; decrement, and
					#	loop with the unmet quota.
						draws = _multinomial_counts(rng, probs ./ total, remaining)
						drawn_total = 0
						@inbounds for e in 1:n_edges
							avail = Int(round(weight[e])) - removed[e]
							take  = min(draws[e], avail)
							removed[e] += take
							drawn_total += take
						end
						remaining = W_removed - sum(removed)
						#	Defensive: if a full pass removed nothing (all clipped),
						#	force a single deterministic unit onto the max-available
						#	edge to guarantee progress.
							if drawn_total == 0
								best_e = 0; best_avail = 0
								@inbounds for e in 1:n_edges
									avail = Int(round(weight[e])) - removed[e]
									if avail > best_avail
										best_avail = avail; best_e = e
									end
								end
								best_e == 0 && break
								removed[best_e] += 1
								remaining = W_removed - sum(removed)
							end
				end

		#	Organic Losses and Zeroed Edges
			#	Per-node remaining incident weight; a node with zero remaining is
			#	an organic loss. Edges with full removal are counted.
				node_remaining = zeros(Float64, n)
				n_edges_zeroed = 0
				@inbounds for e in 1:n_edges
					rem_w = weight[e] - removed[e]
					if rem_w <= 1e-9
						n_edges_zeroed += 1
					end
					node_remaining[edge_i[e]] += rem_w
					node_remaining[edge_j[e]] += rem_w
				end
				organic_losses = Int[]
				@inbounds for v in 1:n
					#	A node is an organic loss only if it had incident weight and
					#	now has effectively none. Float tolerance guards non-integer
					#	weights; isolates are filtered out below via had_incident.
						node_remaining[v] <= 1e-9 || continue
						push!(organic_losses, v)
				end
				#	Filter to nodes that originally had incident weight: a true
				#	isolate has node_remaining == 0 but was never "lost". Recompute
				#	original incidence and intersect.
					had_incident = falses(n)
					@inbounds for e in 1:n_edges
						had_incident[edge_i[e]] = true
						had_incident[edge_j[e]] = true
					end
					organic_losses = sort!([v for v in organic_losses if had_incident[v]])

		#	Build Removal Matrix (same pattern as adj)
			rem_mat = _rebuild_sparse(adj, removed, edge_i, edge_j)

		#	Return
			return (removed = rem_mat,
					W_true = W_true_int,
					W_removed = sum(removed),
					organic_losses = organic_losses,
					n_edges_zeroed = n_edges_zeroed)
	end

#	Helper Function for _three_prior_gate: E/I-by-degree-rank profile of a node subgraph
	function _ei_rank_profile(adj_binary::SparseMatrixCSC,
								community::AbstractVector{<:Integer},
								keep::AbstractVector{Bool},
								n_rank_bins::Int,
								n_ei_bins::Int)
		"""
		Args:
			adj_binary::SparseMatrixCSC: binarized adjacency, canonical node order.
			community::AbstractVector{<:Integer}: per-node TRUE community labels.
			keep::AbstractVector{Bool}: node mask defining the subgraph to profile
				(all-true for the true network; survivors for the degraded one).
			n_rank_bins::Int: number of degree-rank bins.
			n_ei_bins::Int: number of E/I bins on the fixed [-1, 1] range.
		Returns:
			Matrix{Float64}: n_rank_bins x n_ei_bins, row-stochastic per occupied
				rank bin — row r is P(E/I bin | degree-rank bin r) among the kept
				nodes. Empty rank bins are left as a zero row (occupancy reported
				separately) so the caller can weight/skip them.
			Also returns per-rank-bin occupancy counts as the second tuple element.
		Notes:
			The E/I-by-degree-rank profile at the heart of prior 3. Both E/I and
			degree rank are computed ON THE SUBGRAPH induced by `keep` — an edge
			counts only if BOTH endpoints are kept — so removal genuinely re-ranks
			survivors and re-scores their brokerage (a node whose same-community
			neighbors were dropped reads as more external on the thinned graph).
			Community LABELS are ground truth (passed in); only incidence thins.

			Degree rank is rank-equal over the KEPT nodes (bin = ceil(rank*K/M),
			M = #kept), so the true-network profile and the survivor profile are
			compared by RELATIVE rank position — scale-free and robust to the
			change in node count after removal. E/I is binned on the fixed [-1, 1]
			range so both profiles share an identical E/I axis.

			Returns the conditional P(E/I | rank) per bin (row-normalized), not the
			joint, so that the degree-tilt prior 2 already controls does not enter
			the comparison; prior 3 is purely the brokerage-given-rank relationship.
		"""

		#	Subgraph degree + E/I on kept nodes only (both endpoints kept)
			n = size(adj_binary, 1)
			rows = rowvals(adj_binary)
			deg = zeros(Int, n)
			internal = zeros(Int, n)
			external = zeros(Int, n)
			@inbounds for j in 1:n
				keep[j] || continue
				for k in nzrange(adj_binary, j)
					i = rows[k]
					keep[i] || continue
					deg[i] += 1; deg[j] += 1
					if community[i] == community[j]
						internal[i] += 1; internal[j] += 1
					else
						external[i] += 1; external[j] += 1
					end
				end
			end

		#	Kept node indices and their within-subgraph degree rank
			kept = findall(keep)
			M = length(kept)
			profile = zeros(Float64, n_rank_bins, n_ei_bins)
			occ = zeros(Int, n_rank_bins)
			M == 0 && return (profile, occ)

			#	Rank kept nodes by subgraph degree (ascending); rank_pos -> rank bin
				order = sort(kept; by = v -> deg[v])
				rank_bin = Dict{Int,Int}()
				@inbounds for rank_pos in 1:M
					rank_bin[order[rank_pos]] = Int(cld(rank_pos * n_rank_bins, M))
				end

		#	Accumulate E/I counts per rank bin
			@inbounds for v in kept
				rb = clamp(rank_bin[v], 1, n_rank_bins)
				tot = internal[v] + external[v]
				eiv = tot == 0 ? 0.0 : (external[v] - internal[v]) / tot
				eb = clamp(Int(cld((eiv + 1.0) / 2.0 * n_ei_bins, 1)), 1, n_ei_bins)
				profile[rb, eb] += 1.0
				occ[rb] += 1
			end

		#	Row-normalize occupied rank bins to P(E/I | rank bin)
			@inbounds for rb in 1:n_rank_bins
				occ[rb] > 0 && (profile[rb, :] ./= occ[rb])
			end

		#	Return profile + occupancy
			return (profile, occ)
	end

#	Helper Function for _three_prior_gate: worst per-rank-bin profile distortion (survivors vs truth)
	function _profile_distortion(true_profile::Matrix{Float64},
									true_occ::AbstractVector{<:Integer},
									surv_profile::Matrix{Float64},
									surv_occ::AbstractVector{<:Integer};
									min_count::Int = 5)
		"""
		Args:
			true_profile::Matrix{Float64}: n_rank_bins x n_ei_bins, P(E/I | rank)
				on the TRUE network (computed once at setup).
			true_occ::AbstractVector{<:Integer}: per-rank-bin counts on the true
				network.
			surv_profile::Matrix{Float64}: same shape, on the SURVIVING subgraph.
			surv_occ::AbstractVector{<:Integer}: per-rank-bin counts among survivors.
			min_count::Int: minimum survivor count for a rank bin to contribute
				(default 5; mirrors the framework's min-nodes-per-bin floors).
		Returns:
			Float64: the MAXIMUM per-rank-bin total-variation distance between the
				survivor and true E/I-given-rank distributions, taken over rank
				bins that clear min_count (the "worst-bin" distortion), in [0, 1].
				NaN if no rank bin clears min_count (caller treats NaN as "prior 3
				not assessable" and skips it).
		Notes:
			Compares the survivors' E/I-by-degree-rank curve to the true network's
			RELATIVE-rank curve, bin for bin, and reports the WORST qualifying bin.

			Why the worst bin rather than an average: broker-stripping — the
			canonical malignant removal — concentrates its distortion in the HIGH-
			degree rank bins, because brokers are high-degree by construction.
			Averaging the per-bin TVD across all rank bins dilutes that signal by
			roughly a factor of n_rank_bins (only the top bin moves), which can hide
			a real bend below the gate tolerance. Taking the max over qualifying
			bins keeps prior 3 sensitive to a distortion that lives in a single bin,
			which is exactly the broker-stripping signature.

			The min_count guard still excludes thinly populated bins from
			contributing, so a one- or two-node bin cannot trip the gate on sampling
			noise; among the bins that DO clear min_count, the worst one governs.

			A benign removal leaves survivor P(E/I | rank) ~ true P(E/I | rank) in
			every bin, so the worst-bin TVD stays small; a removal that strips
			brokers bends the top bin and the worst-bin TVD jumps.

			TOLERANCE NOTE: because this returns a max rather than a mean, the gate's
			ei_tvd_tol is on a different (strictly larger) scale than under the prior
			occupancy-weighted mean — recalibrate ei_tvd_tol against a real network
			before trusting it in the corpus.
		"""

		#	Guards
			size(true_profile) == size(surv_profile) ||
				throw(ArgumentError("profile shapes must match"))
			n_rank_bins = size(true_profile, 1)

		#	Worst per-rank-bin TVD among bins clearing min_count
			worst = -1.0
			any_qual = false
			@inbounds for rb in 1:n_rank_bins
				surv_occ[rb] >= min_count || continue
				true_occ[rb] > 0 || continue
				any_qual = true
				tvd = 0.5 * sum(abs.(surv_profile[rb, :] .- true_profile[rb, :]))
				tvd > worst && (worst = tvd)
			end

		#	Return worst qualifying bin (or NaN if none qualified)
			return any_qual ? worst : NaN
	end

#	Helper Three-Prior Gate: end-of-pipeline contract check on the combined missing set
	function _three_prior_gate(missing_nodes::AbstractVector{<:Integer},
								centrality::AbstractVector{<:Real},
								true_community::AbstractVector{<:Integer},
								adj_binary::SparseMatrixCSC,
								true_profile::Matrix{Float64},
								true_occ::AbstractVector{<:Integer},
								pi_node::Real,
								target_rho::Real;
								node_loss::Symbol = :emergent,
								rho_tol::Real = 0.02,
								ei_tvd_tol::Real = 0.25,
								n_rank_bins::Int = 4,
								n_ei_bins::Int = 3,
								min_count::Int = 5)
		"""
		Args:
			missing_nodes::AbstractVector{<:Integer}: combined missing set.
			centrality::AbstractVector{<:Real}: per-node centrality, canonical order.
			true_community::AbstractVector{<:Integer}: per-node community label on the
				TRUE network (precomputed once at setup), canonical order.
			adj_binary::SparseMatrixCSC: binarized adjacency of the TRUE network.
			true_profile::Matrix{Float64}: the TRUE network's E/I-by-degree-rank
				profile (n_rank_bins x n_ei_bins, P(E/I | rank)), precomputed once
				at setup via _ei_rank_profile on all nodes.
			true_occ::AbstractVector{<:Integer}: per-rank-bin occupancy of the true
				profile.
			pi_node::Real: nominal missing-node fraction.
			target_rho::Real: nominal correlation.
			node_loss::Symbol: :targeted gates priors 1 (proportion) and 2
				(correlation) against pi_node and target_rho; :emergent records both
				without gating (node loss is emergent, so neither is a target).
				Default :emergent.
			rho_tol::Real: tolerance on |realized_rho - target_rho| (default 0.02).
			ei_tvd_tol::Real: tolerance on the survivor-vs-true profile distortion
				(default 0.25).
			n_rank_bins, n_ei_bins::Int: profile binning; must match the binning
				used to build true_profile.
			min_count::Int: minimum survivor count for a rank bin to contribute to
				the distortion (default 5).
		Returns:
			NamedTuple: (passed, realized_pi_node, realized_rho, ei_tvd,
						 prior1_ok, prior2_ok, prior3_ok)
		Notes:
			The single end-of-pipeline gate (Spec v3 Section 4.4). Validates the
			final missing set against up to three priors. Priors 1 and 2 gate only
			in :targeted mode; in :emergent mode they are recorded but not gated
			(node loss is emergent, so proportion and correlation are outcomes, not
			targets), and only prior 3 can fail the gate. The priors:
			  1. proportion: realized |missing|/N ~= pi_node (near-exact).
			  2. correlation: Kendall tau-b(missing-indicator, centrality) ~= target_rho.
			  3. E/I-by-degree-rank distortion: does removing these nodes BEND the
			     observed E/I-versus-degree-rank relationship away from the truth?
			     The SURVIVING subgraph's E/I-by-degree-rank profile is compared to
			     the TRUE network's profile (occupancy-weighted per-rank-bin TVD via
			     _profile_distortion); a benign removal leaves the curve intact, a
			     malignant one (stripping brokers so the new high-rank nodes are
			     disproportionately internal/external) inflates the distortion.
			Both E/I and degree rank for the survivor profile are recomputed ON THE
			THINNED subgraph (community labels are ground truth; incidence thins),
			so re-ranking and re-scored brokerage are captured. Uses the precomputed
			true profile, so no per-replicate community detection runs.

			Prior 3 is skipped (prior3_ok = true) when there is no usable community
			structure (single community) or no survivor rank bin clears min_count
			(distortion NaN); the binary-undirected floor still gets priors 1 and 2.
			In :emergent prior 3 is record-only: ei_tvd is still computed and returned
			as realized_ei_score, but it never gates (gating would admit only
			distortion-preserving draws, rigging the stimulus). :targeted gates above.
		"""

		#	Guards
			n = length(centrality)
			length(true_community) == n ||
				throw(ArgumentError("true_community length != N"))
			(node_loss in (:emergent, :targeted)) ||
				throw(ArgumentError("node_loss must be :emergent or :targeted, got $node_loss"))

		#	Prior 1: proportion
			is_missing = falses(n)
			@inbounds for v in missing_nodes
				(1 <= v <= n) || throw(ArgumentError("missing node $v out of range"))
				is_missing[v] = true
			end
			realized_pi_node = count(is_missing) / n
			prior1_ok = node_loss === :targeted ?
				isapprox(realized_pi_node, pi_node; atol = 1.0 / n + 1e-9) : true

		#	Prior 2: Kendall tau-b correlation
			realized_rho = StatsBase.corkendall(Float64.(is_missing), Vector{Float64}(centrality))
			isnan(realized_rho) && (realized_rho = 0.0)
			prior2_ok = node_loss === :targeted ?
				(abs(realized_rho - target_rho) <= rho_tol) : true

		#	Prior 3: E/I-by-degree-rank distortion of the SURVIVING network vs truth
			#	Build the survivor profile on the thinned subgraph (E/I and degree
			#	rank both recomputed on survivors; true community labels), then
			#	score its occupancy-weighted per-rank-bin distortion from the
			#	precomputed true profile. NaN distortion (no rank bin clears
			#	min_count, or single community) => prior 3 not assessable => pass.
			#	In :emergent the distortion is recorded but never gates (record-only);
			#	only :targeted enforces it (gating here would rig the stimulus).
				ei_tvd = NaN
				prior3_ok = true
				ncomm = length(unique(true_community))
				if ncomm >= 2
					keep = .!is_missing
					surv_profile, surv_occ = _ei_rank_profile(adj_binary, true_community,
															   keep, n_rank_bins, n_ei_bins)
					ei_tvd = _profile_distortion(true_profile, true_occ,
												  surv_profile, surv_occ;
												  min_count = min_count)
					prior3_ok = node_loss === :targeted ? (isnan(ei_tvd) ? true : (ei_tvd <= ei_tvd_tol)) : true
				end

		#	Return
			passed = prior1_ok && prior2_ok && prior3_ok
			return (passed = passed,
					realized_pi_node = realized_pi_node,
					realized_rho = realized_rho,
					ei_tvd = ei_tvd,
					prior1_ok = prior1_ok,
					prior2_ok = prior2_ok,
					prior3_ok = prior3_ok)
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

#	Helper Function for build_degeneration_corpus: TRUE-network community labels (precomputed once)
	function _true_communities(adj_binary::SparseMatrixCSC)
		"""
		Args:
			adj_binary::SparseMatrixCSC: binarized adjacency of the TRUE network.
		Returns:
			Vector{Int}: per-node community label, canonical node order.
		Notes:
			Computed ONCE per true network at setup so the per-replicate three-prior
			E/I check is O(|missing|) with no per-replicate detection (Spec v3
			Section 4.4). This is the ground-truth reference for the gate; the
			noisier observed-network detection stays in Phase 1.5.

			SEAM: this default labels weakly-connected components, which is
			dependency-free and always available. When the community-detection
			sibling's public entrypoint is wired in, replace the body with the
			CHAMP/Leiden call (e.g. network_community_detection.detect(...)) and
			keep this signature. Single-community outputs cause the gate to skip
			prior 3 (E/I undefined), so the connected-components fallback degrades
			gracefully on networks with one weak component.
		"""

		#	Weakly-connected component labeling via union-find over the pattern
			n = size(adj_binary, 1)
			parent = collect(1:n)
			find(x) = begin
				root = x
				while parent[root] != root; root = parent[root]; end
				while parent[x] != root; parent[x], x = root, parent[x]; end
				root
			end
			union!(a, b) = begin
				ra, rb = find(a), find(b)
				ra != rb && (parent[ra] = rb)
			end
			rows = rowvals(adj_binary)
			@inbounds for j in 1:n
				for k in nzrange(adj_binary, j)
					union!(rows[k], j)
				end
			end

		#	Compact root ids to 1..C
			label = Vector{Int}(undef, n)
			remap = Dict{Int,Int}()
			next_id = 0
			@inbounds for v in 1:n
				r = find(v)
				if !haskey(remap, r)
					next_id += 1
					remap[r] = next_id
				end
				label[v] = remap[r]
			end
			return label
	end

#	Materialize Missing Nodes: compose full-removal and nominations automatically per node
	function _materialize_missing_nodes(edges::DataFrame,
											missing_nodes::AbstractVector{<:Integer};
											nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
											directed::Bool)
		"""
		Args:
			edges::DataFrame: edge list (already edge-degraded if weighted), with
				:src, :dst and any other columns preserved.
			missing_nodes::AbstractVector{<:Integer}: combined missing set in
				canonical node order.
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe.
			directed::Bool: directedness; controls nomination vs full-removal.
		Returns:
			NamedTuple: (edges, nodes, n_full_removal, n_nominations)
				The materialized degraded network plus the per-node materialization
				counts.
		Notes:
			The unified materializer (Spec v3 Section 4.3). Each missing
			(non-responding) node loses its OUTGOING edges; whether it is a
			NOMINATION or a FULL REMOVAL is decided automatically:
			  - directed network AND the node still has a surviving incoming edge
			    (some retained node points to it) => NOMINATION: it stays in the
			    roster with its incoming edges, outgoing zeroed.
			  - otherwise (no surviving incoming, or undirected network) =>
			    FULL REMOVAL: the node and all incident edges are dropped, no trace.
			Undirected networks have no in/out asymmetry, so every missing node is
			a full removal. This composes both mechanisms in one degraded network
			rather than emitting separate per-mechanism corpus rows.

			Implementation: first strip every missing node's outgoing edges (src in
			missing set). Then, on the surviving edges, a missing node is a
			nomination iff it appears as a dst; the rest are full removals and are
			deleted from the roster along with any remaining incident edges.

			Mirrors the per-node filtering logic of the standalone full-removal
			and outgoing-only materializers (now in network_reconstruction),
			reimplemented inline here because this composes BOTH per node by
			partitioning the missing set — neither standalone materializer does
			that split.
		"""

		#	Resolve canonical node order (same convention as the materializers)
			if isnothing(nodes)
				node_names = sort!(unique(vcat(string.(edges.src), string.(edges.dst))))
			elseif nodes isa DataFrame
				node_names = string.(nodes[!, 1])
			else
				node_names = string.(collect(nodes))
			end
			n = length(node_names)

		#	Missing mask + name set
			is_missing = falses(n)
			@inbounds for v in missing_nodes
				(1 <= v <= n) || throw(ArgumentError("missing node index $v out of range [1, $n]"))
				is_missing[v] = true
			end
			missing_name_set = Set{String}(node_names[i] for i in 1:n if is_missing[i])

		#	Step 1: strip outgoing edges of missing nodes (src in missing set).
		#	Undirected networks have both directions present in the edge list (or
		#	are symmetric), so a missing node's incident edges are removed as src
		#	on one orientation; the dst-side survival check below then determines
		#	nomination vs removal. For undirected we force full removal regardless.
			keep_edge = BitVector(undef, nrow(edges))
			@inbounds for r in 1:nrow(edges)
				keep_edge[r] = !(string(edges.src[r]) in missing_name_set)
			end
			after_out = edges[keep_edge, :]

		#	Step 2: classify each missing node as nomination (directed + surviving
		#	incoming) or full removal.
			surviving_dst = Set{String}()
			if directed
				@inbounds for r in 1:nrow(after_out)
					d = string(after_out.dst[r])
					(d in missing_name_set) && push!(surviving_dst, d)
				end
			end
			#	Undirected => surviving_dst stays empty => all missing are full removals.

			n_nominations = length(surviving_dst)
			full_removal_names = Set{String}(nm for nm in missing_name_set if !(nm in surviving_dst))
			n_full_removal = length(full_removal_names)

		#	Step 3: drop edges incident (either endpoint) to a full-removal node,
		#	and drop those nodes from the roster. Nomination nodes stay with their
		#	incoming edges (already retained in after_out).
			keep_edge2 = BitVector(undef, nrow(after_out))
			@inbounds for r in 1:nrow(after_out)
				s = string(after_out.src[r]); d = string(after_out.dst[r])
				keep_edge2[r] = !(s in full_removal_names || d in full_removal_names)
			end
			degraded_edges = after_out[keep_edge2, :]

		#	Roster: drop full-removal nodes; keep nominations and respondents.
			keep_node = BitVector(undef, n)
			@inbounds for i in 1:n
				keep_node[i] = !(node_names[i] in full_removal_names)
			end
			if isnothing(nodes) || !(nodes isa DataFrame)
				degraded_nodes = DataFrame(name = node_names[keep_node])
			else
				degraded_nodes = nodes[keep_node, :]
			end

		#	Return
			return (edges = degraded_edges,
					nodes = degraded_nodes,
					n_full_removal = n_full_removal,
					n_nominations = n_nominations)
	end

#	Apply Weight Removal: materialize diffuse weight depletion onto the edge list (zeros retained)
	function apply_weight_removal(edges::DataFrame,
									removed::SparseMatrixCSC;
									nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing)
		"""
		Args:
			edges::DataFrame: original edge list with :src, :dst, :weight and any
				other columns preserved.
			removed::SparseMatrixCSC: per-edge units removed, same node order and
				sparsity pattern as the weighted adjacency (_weighted_adjacency).
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe.
		Returns:
			DataFrame: a copy of edges with :weight reduced by the removed amount
				per edge. Edges whose weight reaches zero are RETAINED (zeros kept,
				per the edge-preservation decision); row count is invariant.
		Notes:
			Materializes the diffuse edge-degradation stage. Maps each edge row to
			its (i, j) indices in canonical node order, subtracts removed[i, j] from
			:weight, and floors at zero. The edge skeleton is preserved so that
			downstream a fully-depleted edge remains an observed-as-zero dyad,
			which is what makes node loss the LIMIT of weight loss rather than a
			separate primitive.
		"""

		#	Resolve canonical node order and a name->index map
			if isnothing(nodes)
				node_names = sort!(unique(vcat(string.(edges.src), string.(edges.dst))))
			elseif nodes isa DataFrame
				node_names = string.(nodes[!, 1])
			else
				node_names = string.(collect(nodes))
			end
			idx = Dict{String,Int}(nm => i for (i, nm) in enumerate(node_names))

		#	Guard: a weight column is required
			("weight" in names(edges)) ||
				throw(ArgumentError("apply_weight_removal requires a :weight column"))

		#	Copy and decrement per edge
			out = copy(edges)
			w = Float64.(out.weight)
			@inbounds for r in 1:nrow(out)
				i = get(idx, string(out.src[r]), 0)
				j = get(idx, string(out.dst[r]), 0)
				(i == 0 || j == 0) && continue
				rem_units = removed[i, j]
				neww = w[r] - rem_units
				w[r] = neww < 0 ? 0.0 : neww
			end
			out.weight = w

		#	Return (zeros retained; row count invariant)
			return out
	end

#	Generate Missingness Mask: full per-replicate record for one (network, target) draw
	function generate_missingness_mask(edges::DataFrame;
										nodes::Union{Nothing,DataFrame,AbstractVector{<:AbstractString}} = nothing,
										directed::Bool,
										weighted::Bool,
										node_loss::Symbol = :emergent,
										target_pi_node::Real,
										target_pi_edge::Real,
										target_rho::Real,
										seed::Integer,
										centrality::Union{Nothing,AbstractVector{<:Real}} = nothing,
										true_community::Union{Nothing,AbstractVector{<:Integer}} = nothing,
										adj_binary::Union{Nothing,SparseMatrixCSC} = nothing,
										adj_weighted::Union{Nothing,SparseMatrixCSC} = nothing,
										K::Int = 4,
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1,
										rho_tol::Real      = 0.02,
										ei_tvd_tol::Real   = 0.25,
										n_rank_bins::Int   = 4,
										n_ei_bins::Int     = 3,
										ei_min_count::Int  = 5,
										max_retries::Int   = 10)
		"""
		Args:
			edges::DataFrame: original edge list with :src, :dst (+ :weight if weighted).
			nodes::Union{Nothing,DataFrame,Vector}: optional node universe.
			directed::Bool: directedness (selects centrality driver and nominations).
			weighted::Bool: selects the edge-removal mechanism (weight removal
				vs whole-tie removal); the edge stage runs for both types.
			target_pi_node::Real: in (0, 1). In :targeted mode the node-missingness
				target; in :emergent mode a propensity-tilt calibration only.
			node_loss::Symbol: :emergent -> missing set = edge-induced organic losses
				(edge-primary, this paper); :targeted -> top up to target_pi_node
				(node non-response, Phase 1.5). Default :emergent.
			target_pi_edge::Real: diffuse edge-degradation budget (fraction of
				total edge weight); in [0, 1). Applies to both types (weight
				removal when weighted, whole-tie removal when binary).
			target_rho::Real: target Kendall tau-b of the missing set vs centrality.
			seed::Integer: master seed; deterministically split, retries advance
				a deterministic sub-seed sequence.
			centrality, true_community, adj_binary, adj_weighted: optional
				precomputed per-network artifacts (the orchestrator caches them).
			K::Int: rank-bin count for the propensity field.
			gc_threshold, min_n, min_edges: passed to topological degeneracy.
			rho_tol, ei_tvd_tol: passed to the three-prior gate.
			max_retries::Int: bounded whole-replicate re-run budget (default 10).
		Returns:
			NamedTuple: the per-replicate record (see Notes for fields).
		Notes:
			The public per-replicate sampler, rewired for the unified pipeline
			(Spec v3): edge degradation (weighted -> weight removal; binary ->
			tie removal, the unit-weight limit) -> record organic losses +
			edge-stage diagnostic -> node accounting (node_loss: :emergent = organic only, :targeted = top-up to pi_node) ->
			materialize (auto nomination/full-removal) -> single three-prior end
			gate -> retry-on-stochastic-miss / accept-and-flag on :ceiling_hit.

			There is no `mechanism` argument: full-removal vs nomination is decided
			per node inside _materialize_missing_nodes.

			Seed splitting + deterministic retries. The master seed seeds an
			attempt-seed stream; attempt a in 1..max_retries derives
			(edge_seed_a, topup_seed_a) deterministically, so the same master seed
			yields the same accepted draw AND the same retry count.

			Feasibility is NOT re-checked here — the orchestrator clamps target_rho
			to the up-front envelope and flags :ceiling_hit before calling, so a
			ceiling draw is accepted-and-flagged (no retry) and only stochastic
			misses inside the feasible region trigger re-runs.

			Record fields:
				- missing_nodes::Vector{Int}, n_full_removal::Int, n_nominations::Int
				- nominal::NamedTuple (pi_node, pi_edge, rho)
				- realized_pi_node, realized_rho, realized_ei_score::Float64
				- realized_pi_edge::Float64, edge diagnostic fields
				  (W_true, W_removed, n_edges_zeroed, n_organic_losses, beta)
				- gate_status::Symbol, contract_passed::Bool, retry_count::Int
				- field_status::Symbol (propensity-field solve status)
				- sampler_degeneracy::NamedTuple
				- seed::Int
		"""

		#	Guards
			(0.0 < target_pi_node < 1.0) ||
				throw(ArgumentError("target_pi_node must be in (0, 1), got $target_pi_node"))
			(0.0 <= target_pi_edge < 1.0) ||
				throw(ArgumentError("target_pi_edge must be in [0, 1), got $target_pi_edge"))
			(-1.0 < target_rho < 1.0) ||
				throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))
			(node_loss in (:emergent, :targeted)) ||
				throw(ArgumentError("node_loss must be :emergent or :targeted, got $node_loss"))

		#	Edge stage runs for both types. Binary is the unit-weight limit of
		#	weighted removal: with avail==1 per edge, _sample_weight_removal can
		#	only zero whole ties (zero-only tie removal), never underweight.
			pi_edge = Float64(target_pi_edge)

		#	Per-network artifacts: compute or accept cached
			c = isnothing(centrality) ?
				_centrality_for_sampler(edges; nodes=nodes, directed=directed) : centrality
			n = length(c)
			K_eff = clamp(K, 2, max(2, n))

			adjb = isnothing(adj_binary) ?
				(isnothing(nodes) ? _graph_to_sparse_matrix(edges; weighted=false)[1] :
									 _graph_to_sparse_matrix(edges; nodes=nodes, weighted=false)[1]) :
				adj_binary

			comm = isnothing(true_community) ? fill(1, n) : true_community

			#	True-network E/I-by-degree-rank profile for prior 3, computed once
			#	(all nodes kept). Skipped to an empty profile when there is no
			#	community structure; the gate then treats prior 3 as not assessable.
				true_profile, true_occ = length(unique(comm)) >= 2 ?
					_ei_rank_profile(adjb, comm, trues(n), n_rank_bins, n_ei_bins) :
					(zeros(Float64, n_rank_bins, n_ei_bins), zeros(Int, n_rank_bins))

			adjw = nothing
			if pi_edge > 0.0
				adjw = !isnothing(adj_weighted) ? adj_weighted :
					   (weighted ? _weighted_adjacency(edges; nodes=nodes)[1] : adjb)
			end

		#	Propensity field (shared tilt); status recorded
			field = _solve_propensity_field(c, target_rho, target_pi_node, K_eff)

		#	Attempt loop: deterministic re-runs on stochastic gate miss
			master_rng = Xoshiro(seed)
			best = nothing
			retry_count = 0
			gate_status = :failed_other
			for attempt in 1:max_retries
				edge_seed  = Int(rand(master_rng, UInt32))
				topup_seed = Int(rand(master_rng, UInt32))	#	used by the :targeted top-up; in :emergent mode unused, but the draw is kept so the edge-seed stream matches across modes

				#	Stage 1: edge degradation (both types) -> organic losses + diagnostic
					organic = Int[]
					W_true = 0; W_removed = 0; n_edges_zeroed = 0
					degraded_edges = edges
					if pi_edge > 0.0
						wr = _sample_weight_removal(adjw, field.d, pi_edge, edge_seed)
						organic        = wr.organic_losses
						W_true         = wr.W_true
						W_removed      = wr.W_removed
						n_edges_zeroed = wr.n_edges_zeroed
						degraded_edges = apply_weight_removal(edges, wr.removed; nodes=nodes)
					end

				#	Stage 2: node accounting -- mode switch
				#	:emergent -> missing set is the edge-induced organic losses (this paper)
				#	:targeted -> top the missing set up to target_pi_node (node non-response, Phase 1.5)
					if node_loss === :targeted
						topup = _topup_missing_nodes(c, organic, target_pi_node, target_rho,
													  topup_seed; K = K_eff)
						missing_nodes = topup.missing_nodes
					else
						missing_nodes = copy(organic)
					end

				#	Stage 3: materialize (auto nomination/full-removal)
					mat = _materialize_missing_nodes(degraded_edges, missing_nodes;
													  nodes=nodes, directed=directed)

				#	Stage 4: three-prior end gate
					gate = _three_prior_gate(missing_nodes, c, comm, adjb,
											  true_profile, true_occ,
											  target_pi_node, target_rho;
											  node_loss = node_loss,
											  rho_tol = rho_tol, ei_tvd_tol = ei_tvd_tol,
											  n_rank_bins = n_rank_bins, n_ei_bins = n_ei_bins,
											  min_count = ei_min_count)

				#	Record the candidate
					candidate = (missing_nodes = missing_nodes,
								  n_full_removal = mat.n_full_removal,
								  n_nominations  = mat.n_nominations,
								  realized_pi_node = gate.realized_pi_node,
								  realized_rho     = gate.realized_rho,
								  realized_ei_score = gate.ei_tvd,
								  realized_pi_edge = W_true > 0 ? W_removed / W_true : 0.0,
								  W_true = W_true, W_removed = W_removed,
								  n_edges_zeroed = n_edges_zeroed,
								  n_organic_losses = length(organic),
								  beta = field.beta,
								  gate = gate)

				#	Accept on pass; otherwise keep the best-so-far for the
				#	exhaustion path and re-run.
					if gate.passed
						best = candidate; gate_status = :converged; retry_count = attempt - 1
						break
					elseif field.status == :ceiling_hit
						#	Structurally unreachable rho: accept-and-flag, no retry.
							best = candidate; gate_status = :ceiling_hit; retry_count = attempt - 1
						break
					else
						best = candidate; retry_count = attempt - 1
					end
			end
			if gate_status == :failed_other && best !== nothing && best.gate.passed
				gate_status = :converged
			elseif !(gate_status in (:converged, :ceiling_hit))
				gate_status = :failed_other
			end

		#	Degeneracy on the final missing set (binary adjacency + missing mask)
			degen = _topological_degeneracy(adjb, best.missing_nodes;
											gc_threshold = gc_threshold,
											min_n        = min_n,
											min_edges    = min_edges)

		#	Assemble per-replicate record
			return (missing_nodes      = best.missing_nodes,
					n_full_removal      = best.n_full_removal,
					n_nominations       = best.n_nominations,
					nominal             = (pi_node = Float64(target_pi_node),
											pi_edge = Float64(pi_edge),
											rho     = Float64(target_rho)),
					realized_pi_node    = best.realized_pi_node,
					realized_rho        = best.realized_rho,
					realized_ei_score   = best.realized_ei_score,
					realized_pi_edge    = best.realized_pi_edge,
					W_true              = best.W_true,
					W_removed           = best.W_removed,
					n_edges_zeroed      = best.n_edges_zeroed,
					n_organic_losses    = best.n_organic_losses,
					beta                = best.beta,
					field_status        = field.status,
					gate_status         = gate_status,
					contract_passed     = best.gate.passed,
					retry_count         = retry_count,
					sampler_degeneracy  = degen,
					seed                = Int(seed))
	end
	@doc raw"""
	**Description**
	Generates one per-replicate missingness record for the unified Phase~1
	degeneration pipeline (Spec~v3). Runs edge degradation (weight removal on
	weighted networks, whole-tie removal on binary networks -- the unit-weight
	limit), records the organic node losses, and sets the missing set per
	`node_loss`: the organic losses alone when `:emergent` (this paper), or
	topped up to the target node fraction via the shared logistic-$\beta$
	propensity field when `:targeted`. It then materializes the degraded network
	(composing nominations and full removal per node) and verifies a single
	three-prior end gate with bounded deterministic re-runs.

	**Usage**
	`generate_missingness_mask(edges; nodes, directed, weighted, target_pi_node, target_pi_edge, target_rho, seed, ...)`

	**Arguments**
	- `edges::DataFrame`: edge list with `:src`, `:dst` (and `:weight` if `weighted`).
	- `directed::Bool`, `weighted::Bool`: network type. `weighted` selects the
	  edge-removal mechanism (weight removal vs whole-tie removal), not whether
	  the edge stage runs.
	- `node_loss::Symbol`: `:emergent` -> missing set is the edge-induced organic
	  losses (this paper); `:targeted` -> top up to `target_pi_node` (node
	  non-response, Phase 1.5). Default `:emergent`.
	- `target_pi_node`, `target_pi_edge`, `target_rho`: node target (only in
	  `:targeted`; a propensity-tilt calibration in `:emergent`), edge-degradation
	  budget, and the shared correlation. `target_pi_edge` applies to both types
	  (weight removal when weighted, whole-tie removal when binary).
	- `seed::Integer`: master seed; retries advance a deterministic sub-seed stream.
	- `centrality`, `true_community`, `adj_binary`, `adj_weighted`: optional cached
	  per-network artifacts (the orchestrator supplies these).
	- `K`, `gc_threshold`, `min_n`, `min_edges`, `rho_tol`, `ei_tvd_tol`,
	  `max_retries`: tuning knobs.

	**Value**
	`NamedTuple` carrying the combined missing set, materialization counts, final
	realized priors, the non-gating edge-stage diagnostic, the gate/field status,
	the retry count, sampler degeneracy, and the master seed.

	**Notes**
	Deterministic in the master seed: identical inputs produce identical records
	and identical retry counts. `:ceiling_hit` (from the propensity-field solve /
	the orchestrator's up-front feasibility clamp) is accepted-and-flagged without
	retry; only stochastic gate misses inside the feasible region re-run.

	**See Also**
	`_solve_propensity_field`, `_sample_weight_removal`, `_topup_missing_nodes`,
	`_materialize_missing_nodes`, `_three_prior_gate`, `build_degeneration_corpus`
	""" generate_missingness_mask

#	Build Degeneration Corpus: orchestrate the unified grid across networks, rhos, and the two dials
	function build_degeneration_corpus(networks::Dict;
										target_rhos::AbstractVector{<:Real} = [-0.75, -0.25, 0.0, 0.25, 0.75],
										target_pi_nodes::AbstractVector{<:Real} = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
										target_pi_edges::AbstractVector{<:Real} = [0.0, 0.10, 0.25],
										node_loss::Symbol = :emergent,
										n_replicates::Int = 100,
										replicates_per_network::Dict{String,Int} = Dict{String,Int}(),
										reverse_edges::Bool = false,
										master_seed::Integer = 42,
										K::Int             = 4,
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1,
										rho_tol::Real      = 0.02,
										ei_tvd_tol::Real   = 0.25,
										max_retries::Int   = 10,
										parallel::Bool     = true,
										show_progress::Bool = true)
		"""
		Args:
			networks::Dict: corpus keyed by network name → NamedTuple with :edges
				(DataFrame), :nodes (DataFrame), :metadata (NamedTuple with at least
				:directed::Bool and :weighted::Bool).
			target_rhos::AbstractVector{<:Real}: nominal rho grid.
			target_pi_nodes::AbstractVector{<:Real}: nominal node-missingness grid.
			target_pi_edges::AbstractVector{<:Real}: nominal edge-degradation grid
				(fraction of total edge weight). 0.0 means node-only. Applies to
				both types: weighted nets remove weight, binary nets remove whole
				ties (the unit-weight limit); grid is deduplicated.
			node_loss::Symbol: :emergent -> missing set is the edge-induced organic
				losses (this paper); :targeted -> top up to target_pi_node (node
				non-response, Phase 1.5); threaded to generate_missingness_mask.
				Default :emergent.
			n_replicates::Int: default replicate count per cell.
			replicates_per_network::Dict{String,Int}: per-network override.
			reverse_edges::Bool: explicit orientation flag (default false). When
				true, src/dst are transposed at the boundary so all internal logic
				stays direction-fixed; affects only the asymmetric nomination path.
				Cannot be trusted to metadata — supplied per run.
			master_seed::Integer: master seed; per-replicate seeds derived from
				(name, rho, pi_node, pi_edge, rep, master_seed).
			K, gc_threshold, min_n, min_edges, rho_tol, ei_tvd_tol, max_retries:
				passed through to generate_missingness_mask.
			parallel::Bool, show_progress::Bool: execution controls.
		Returns:
			DataFrame with one row per (network, rho, pi_node, pi_edge, replicate)
			tuple. There is NO mechanism column — full-removal vs nomination is
			composed per node and reported via n_full_removal / n_nominations.
		Notes:
			The unified orchestrator (Spec v3). For each network it follows the
			shared front-matter: detect type → resolve the edge-removal weight
			matrix by type (true weights when weighted, unit-weight binary
			adjacency otherwise) → up-front feasibility via
			feasible_rho_range (warn + substitute the clamped rho, recording which
			value was used) → run generate_missingness_mask per cell.

			Per-network artifacts (centrality, binarized adjacency, weighted
			adjacency, and TRUE-network community labels for the E/I gate) are
			computed once and cached, so per-replicate cost excludes detection.

			reverse_edges canonicalization happens once per network at the
			boundary; outputs reference original node identifiers (the transpose
			only swaps the src/dst columns feeding the directed nomination logic).

			Determinism: rep_seed = Int(hash((name, rho, pi_node, pi_edge, rep,
			master_seed)) % UInt32); retries inside the mask advance a deterministic
			sub-seed stream, so the corpus is bit-reproducible.

			feasibility: out-of-envelope rho is clamped and flagged; the mask then
			accepts the ceiling draw without burning retries.
		"""

		#	Guards
			isempty(networks) && throw(ArgumentError("networks dict is empty"))
			n_replicates >= 1 || throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))
			(node_loss in (:emergent, :targeted)) || throw(ArgumentError("node_loss must be :emergent or :targeted, got $node_loss"))

		#	Per-network setup: detect type, canonicalize, cache artifacts, feasibility
			network_names = sort!(collect(keys(networks)))
			centrality_cache = Dict{String, Vector{Float64}}()
			adjb_cache       = Dict{String, SparseMatrixCSC}()
			adjw_cache       = Dict{String, Union{Nothing,SparseMatrixCSC}}()
			comm_cache       = Dict{String, Vector{Int}}()
			edges_cache      = Dict{String, DataFrame}()
			pi_edges_for_net = Dict{String, Vector{Float64}}()
			feasible_cache   = Dict{Tuple{String,Float64,Float64}, NTuple{2,Float64}}()

			for name in network_names
				net = networks[name]
				directed = net.metadata.directed
				weighted = net.metadata.weighted

				#	Canonicalize orientation at the boundary
					ed = net.edges
					if reverse_edges
						ed = copy(ed)
						ed.src, ed.dst = ed.dst, ed.src
					end
					edges_cache[name] = ed

				#	Cache centrality + binarized adjacency
					centrality_cache[name] = _centrality_for_sampler(ed; nodes=net.nodes, directed=directed)
					adjb_cache[name] = isnothing(net.nodes) ?
						_graph_to_sparse_matrix(ed; weighted=false)[1] :
						_graph_to_sparse_matrix(ed; nodes=net.nodes, weighted=false)[1]

				#	Edge-removal weight matrix: true weights when weighted, the unit-
				#	weight (binary) adjacency otherwise. The binary case is the unit-
				#	weight limit of weight removal -> tie removal (zero-only).
					adjw_cache[name] = weighted ? _weighted_adjacency(ed; nodes=net.nodes)[1] : adjb_cache[name]

				#	TRUE-network community labels for the E/I gate (precomputed once)
					comm_cache[name] = _true_communities(adjb_cache[name])

				#	pi_edge dial applies to both types: weighted -> weight removal,
				#	binary -> tie removal (the unit-weight limit of the same mechanism)
					pe = sort!(unique(Float64.(target_pi_edges)))
					pi_edges_for_net[name] = pe
			end

		#	Enumerate flat grid (no mechanism axis)
			grid_size = 0
			for name in network_names
				n_reps = get(replicates_per_network, name, n_replicates)
				grid_size += length(target_rhos) * length(target_pi_nodes) *
							 length(pi_edges_for_net[name]) * n_reps
			end
			flat = Vector{Tuple{String,Float64,Float64,Float64,Int}}(undef, grid_size)
			let idx = 1
				for name in network_names
					n_reps = get(replicates_per_network, name, n_replicates)
					for rho in target_rhos
						for pin in target_pi_nodes
							for pie in pi_edges_for_net[name]
								for rep in 1:n_reps
									flat[idx] = (name, Float64(rho), Float64(pin), Float64(pie), rep)
									idx += 1
								end
							end
						end
					end
				end
			end

		#	Up-front feasibility per (network, pi_node, pi_edge); clamp + warn
			substituted = Dict{Tuple{String,Float64,Float64,Float64}, Float64}()
			for name in network_names
				net = networks[name]
				for pin in target_pi_nodes
					for pie in pi_edges_for_net[name]
						key = (name, Float64(pin), Float64(pie))
						haskey(feasible_cache, key) && continue
						#	SEAM: feasible_rho_range currently takes a single
						#	target_rate (positional nodes); v3 calls for generalizing
						#	it to (pi_node, pi_edge) and the organic-loss tightening.
						#	Until that reconstruction-side change lands, we probe the
						#	envelope at the node rate, which is the binding axis for
						#	the node-missingness correlation the gate checks.
							fr = feasible_rho_range(edges_cache[name], net.nodes;
													 directed    = net.metadata.directed,
													 weighted    = net.metadata.weighted,
													 target_rate = Float64(pin))
							rng_lo, rng_hi = fr.rho_min, fr.rho_max
						feasible_cache[key] = (rng_lo, rng_hi)
						for rho in target_rhos
							clamped = clamp(Float64(rho), rng_lo, rng_hi)
							if !isapprox(clamped, rho; atol = 1e-9)
								substituted[(name, Float64(rho), Float64(pin), Float64(pie))] = clamped
								if show_progress
									println("Note: rho=$rho infeasible for $name ",
											 "(pi_node=$pin, pi_edge=$pie); using $clamped")
								end
							end
						end
					end
				end
			end

		#	Pre-allocate result storage
			results = Vector{NamedTuple}(undef, grid_size)
			use_threads = parallel && Threads.nthreads() > 1 && grid_size > 1
			desc = "Degeneration grid (" * string(grid_size) * " rows, " *
				   (use_threads ? "$(Threads.nthreads()) threads" : "serial") * ")"
			prog = show_progress ? Progress(grid_size, desc = desc, enabled = true) : nothing
			prog_lock = ReentrantLock()

		#	Worker closure body factored for serial / threaded reuse
			run_cell = function (k::Int)
				name, rho, pin, pie, rep = flat[k]
				net = networks[name]
				#	Apply the feasibility substitution if any
					rho_used = get(substituted, (name, rho, pin, pie), rho)
				rep_seed = Int(hash((name, rho, pin, pie, rep, master_seed)) % UInt32)
				rec = generate_missingness_mask(edges_cache[name];
						nodes          = net.nodes,
						directed       = net.metadata.directed,
						weighted       = net.metadata.weighted,
						node_loss      = node_loss,
						target_pi_node = pin,
						target_pi_edge = pie,
						target_rho     = rho_used,
						seed           = rep_seed,
						centrality     = centrality_cache[name],
						true_community = comm_cache[name],
						adj_binary     = adjb_cache[name],
						adj_weighted   = adjw_cache[name],
						K = K, gc_threshold = gc_threshold, min_n = min_n,
						min_edges = min_edges, rho_tol = rho_tol,
						ei_tvd_tol = ei_tvd_tol, max_retries = max_retries)
				results[k] = (name = name, rho = rho, pin = pin, pie = pie, rep = rep,
							   rho_used = rho_used, seed = rep_seed, record = rec)
			end

		#	Run flat-grid loop
			if use_threads
				Threads.@threads :static for k in 1:grid_size
					run_cell(k)
					if show_progress
						lock(prog_lock) do; next!(prog); end
					end
				end
			else
				for k in 1:grid_size
					run_cell(k)
					show_progress && next!(prog)
				end
			end

		#	Flatten per-replicate records into a rectangular DataFrame
			out = DataFrame(
				network_name        = String[],
				nominal_rho         = Float64[],
				nominal_pi_node     = Float64[],
				nominal_pi_edge     = Float64[],
				replicate_idx       = Int[],
				seed                = Int[],
				substituted_rho     = Float64[],
				rho_was_substituted = Bool[],
				missing_nodes       = Vector{Int}[],
				n_full_removal      = Int[],
				n_nominations       = Int[],
				realized_pi_node    = Float64[],
				realized_rho        = Float64[],
				realized_ei_score   = Float64[],
				realized_pi_edge    = Float64[],
				W_true              = Int[],
				W_removed           = Int[],
				n_edges_zeroed      = Int[],
				n_organic_losses    = Int[],
				beta                = Float64[],
				field_status        = Symbol[],
				gate_status         = Symbol[],
				contract_passed     = Bool[],
				retry_count         = Int[],
				gc_fraction_of_remaining = Float64[],
				n_observed          = Int[],
				n_edges_observed    = Int[],
				gc_collapse         = Bool[],
				too_small           = Bool[],
				no_edges            = Bool[],
				any_topo_degenerate = Bool[],
			)
			for r in results
				rec  = r.record
				sdeg = rec.sampler_degeneracy
				push!(out, (network_name        = r.name,
							nominal_rho         = r.rho,
							nominal_pi_node     = r.pin,
							nominal_pi_edge     = r.pie,
							replicate_idx       = r.rep,
							seed                = r.seed,
							substituted_rho     = r.rho_used,
							rho_was_substituted = !isapprox(r.rho_used, r.rho; atol = 1e-9),
							missing_nodes       = rec.missing_nodes,
							n_full_removal      = rec.n_full_removal,
							n_nominations       = rec.n_nominations,
							realized_pi_node    = rec.realized_pi_node,
							realized_rho        = rec.realized_rho,
							realized_ei_score   = rec.realized_ei_score,
							realized_pi_edge    = rec.realized_pi_edge,
							W_true              = rec.W_true,
							W_removed           = rec.W_removed,
							n_edges_zeroed      = rec.n_edges_zeroed,
							n_organic_losses    = rec.n_organic_losses,
							beta                = rec.beta,
							field_status        = rec.field_status,
							gate_status         = rec.gate_status,
							contract_passed     = rec.contract_passed,
							retry_count         = rec.retry_count,
							gc_fraction_of_remaining = sdeg.gc_fraction_of_remaining,
							n_observed          = sdeg.n_observed,
							n_edges_observed    = sdeg.n_edges_observed,
							gc_collapse         = sdeg.gc_collapse,
							too_small           = sdeg.too_small,
							no_edges            = sdeg.no_edges,
							any_topo_degenerate = sdeg.any_topo_degenerate))
			end

		return out
	end
	@doc raw"""
	**Description**
	Orchestrates the unified degeneration grid (Spec~v3). For each network it
	detects type, gates the two dials ($\pi_{\text{node}}$, $\pi_{\text{edge}}$)
	by type, checks the up-front feasibility envelope (warning and substituting an
	infeasible~$\rho$), and produces one per-replicate record per
	$(\text{network}, \rho, \pi_{\text{node}}, \pi_{\text{edge}}, \text{rep})$
	cell. Full-removal versus nomination is composed per node, so there is no
	mechanism axis and no doubled rows.

	**Usage**
	```julia
				corpus = build_degeneration_corpus(networks;
					target_rhos     = [-0.75, -0.25, 0.0, 0.25, 0.75],
					target_pi_nodes = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
					target_pi_edges = [0.0, 0.10, 0.25],
					n_replicates    = 100,
					reverse_edges   = false,
					master_seed     = 42,
				)
	```

	**Arguments**
	- `networks::Dict`: keyed by name $\to$ `(:edges, :nodes, :metadata)` with
	  `:metadata.directed` and `:metadata.weighted`.
	- `target_rhos`, `target_pi_nodes`, `target_pi_edges`: the grid; `pi_edge`
	  entries apply to both types (weight removal when weighted, whole-tie
	  removal when binary).
	- `node_loss::Symbol`: `:emergent` (edge-induced organic losses, this paper)
	  or `:targeted` (top up to `target_pi_node`, Phase 1.5); threaded to the
	  mask. Default `:emergent`.
	- `reverse_edges::Bool`: explicit per-run orientation flag.
	- `master_seed`, `K`, `gc_threshold`, `min_n`, `min_edges`, `rho_tol`,
	  `ei_tvd_tol`, `max_retries`, `parallel`, `show_progress`: controls.

	**Value**
	A `DataFrame` with the unified schema: keys, the realized priors and gate
	status, the non-gating edge-stage diagnostic, materialization counts, and
	sampler degeneracy. Diffuse weight removal is regenerable from the seed and
	is not stored per edge.

	**Notes**
	Regenerating this corpus invalidates the Phase~1.5 community corpus, whose
	join keys must drop `mechanism`. Bit-reproducible from the master seed.

	**See Also**
	`generate_missingness_mask`, `feasible_rho_range`, `_materialize_missing_nodes`
	""" build_degeneration_corpus

#   Exports (public API)
    export generate_missingness_mask,
           apply_weight_removal,
           build_degeneration_corpus

end # module network_degeneracy