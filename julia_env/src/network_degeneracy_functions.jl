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
			#	edges to each node). Undirected: degree = row sums of the
			#	symmetric adjacency. On a symmetric matrix row sums equal
			#	column sums, so we use column sums uniformly — this also means
			#	the directed/undirected branches share the implementation and
			#	we only differ semantically.
				centrality = Vector{Float64}(undef, n)
				@inbounds for j in 1:n
					s = 0.0
					for k in nzrange(adj, j)
						s += nonzeros(adj)[k]
					end
					centrality[j] = s
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

#	Helper Realized Correlation: cor(prob_i, c_i) at a given b
	function _centrality_correlation_for_b(centrality::AbstractVector{<:Real},
											b::Real,
											seed::Integer)
		"""
		Args:
			centrality::AbstractVector{<:Real}: the centrality driver (from
				_centrality_for_sampler); typically Vector{Float64}
			b::Real: the mixing scalar; must be in [0, 1]
			seed::Integer: RNG seed for the u_i ~ Uniform(0,1) draws
		Returns:
			Float64: the Pearson correlation between prob_i and centrality_i
				across the node set, where prob_i = b*c_normalized_i + (1-b)*u_i
				Returns NaN when the correlation is undefined (e.g., on a
				perfectly regular network where centrality is constant).
		Notes:
			This is the deterministic function the bisection routine inverts to
			find the b that produces a target induced rho. The function does
			NOT perform sampling — no nodes are dropped here. It only evaluates
			what the configured rho would be for a given b, by constructing the
			probability vector and computing cor(prob, c).

			Centrality is min-max-normalized to [0,1] before forming the
			convex combination. Without normalization the semantic meaning of
			b would depend on the absolute scale of centrality, which varies
			by orders of magnitude across networks (Marvel degrees ~hundreds,
			Moreno ~single digits). Normalization makes b = 0.5 mean the same
			thing on every network.

			When centrality is constant (max(c) == min(c)), normalization is
			degenerate (0/0). The function returns NaN in this case, signaling
			to the bisection that no value of b can produce a non-zero
			correlation on this network — the achievable-rho ceiling is 0.
			This applies to perfectly regular networks (the 4-regular ring
			test fixture) and to networks where all nodes have identical
			centrality after binarization (rare in practice but possible on
			tiny inputs).

			Seed determines the u_i draws via a local RNG (Xoshiro), so calling
			this function does NOT perturb the global RNG state. Two calls
			with the same (centrality, b, seed) produce identical correlations
			— required for bisection convergence and for seed reproducibility
			(Test 4 in the harness).

			At b = 0 the prob vector is pure uniform noise; cor(u, c) is
			approximately 0 but not exactly 0 due to finite-N sampling — the
			expected value over many seeds is 0. At b = 1 the prob vector
			equals normalized centrality; cor(c_normalized, c) = 1 exactly
			(up to floating-point). Between, the function is monotonically
			increasing in b for any fixed seed.
		"""

		#	Guards
			n = length(centrality)
			n >= 2 || throw(ArgumentError("centrality must have at least 2 nodes"))
			(0.0 <= b <= 1.0) || throw(ArgumentError("b must be in [0, 1], got $b"))

		#	Min-Max Normalize Centrality to [0, 1]
			#	If max == min the network has no centrality variation; return
			#	NaN to signal the bisection that this network has an
			#	achievable-rho ceiling of 0.
				c_min = minimum(centrality)
				c_max = maximum(centrality)
				if c_max == c_min
					return NaN
				end
				c_norm = (centrality .- c_min) ./ (c_max - c_min)

		#	Draw u_i with a Local RNG (Does Not Perturb Global State)
			rng = Xoshiro(seed)
			u   = rand(rng, n)

		#	Form the Probability Vector and Correlate
			#	prob_i = b * c_norm_i + (1 - b) * u_i
				prob = b .* c_norm .+ (1 - b) .* u

		#	Return cor(prob, centrality) — note: use ORIGINAL centrality, not
		#	c_norm, since the realized rho is defined against the centrality
		#	scale the caller cares about. (cor is scale-invariant, so this is
		#	mathematically identical to cor(prob, c_norm); using the original
		#	for clarity.)
			return cor(prob, centrality)
	end

#	Helper Bisection: solve b so cor(prob, c) ≈ |target_rho|; record sign separately
	function _bisect_b_for_target_rho(centrality::AbstractVector{<:Real},
										target_rho::Real,
										seed::Integer;
										tol::Real           = 1e-3,
										max_iters::Int      = 50,
										b_ceiling_eps::Real = 1e-6)
		"""
		Args:
			centrality::AbstractVector{<:Real}: centrality driver from
				_centrality_for_sampler
			target_rho::Real: target induced correlation between prob_i and c_i;
				must be in (-1, 1)
			seed::Integer: RNG seed used by _centrality_correlation_for_b for
				the u_i draws
			tol::Real: convergence tolerance on |realized_rho - |target_rho||;
				default 1e-3
			max_iters::Int: maximum bisection iterations; default 50 (more than
				enough — 50 iterations bisects to 2^-50 of the [0,1] range)
			b_ceiling_eps::Real: how close to b = 1 counts as 'hit the ceiling';
				default 1e-6
		Returns:
			NamedTuple: (b, sign, realized_rho_pos, status)
				- b::Float64: the bisected mixing scalar in [0, 1]
				- sign::Int: +1 if target_rho > 0, -1 if target_rho < 0,
					+1 if target_rho == 0 (the sign Function 4 applies when
					constructing the prob vector for sampling)
				- realized_rho_pos::Float64: cor(prob, c) at the chosen b
					BEFORE sign flip; sign * realized_rho_pos is the realized
					correlation against centrality, matching target_rho's sign
				- status::Symbol: :converged, :ceiling_hit, or :failed_other
		Notes:
			The selection formula prob_i = b*c_norm_i + (1-b)*u_i can sweep
			cor(prob, c) from ~0 (at b=0) to ~1 (at b=1) but cannot reach
			negative correlations. To target a negative rho we bisect for
			|target_rho| against the positive formulation and tell the
			sampler (Function 4) to invert the prob vector before drawing:
			prob_i ← 1 - prob_i flips the sign of the correlation without
			changing its magnitude. This bisection therefore solves only the
			positive sub-problem; the sign-flip is the caller's responsibility,
			communicated via the returned `sign` field.

			Status semantics:
			- :converged means |cor(prob, c) at b| - |target_rho|| < tol.
			- :ceiling_hit means the bisection reached b = 1 (within
			  b_ceiling_eps) and the realized correlation is still below
			  |target_rho|. This is the achievable-rho ceiling on skewed
			  networks: the degree distribution caps how high cor(prob, c)
			  can go even when b is fully committed to the centrality term.
			- :failed_other means something unexpected — NaN in
			  _centrality_correlation_for_b (signaling constant centrality),
			  target_rho out of range, or max_iters exceeded without
			  convergence and without hitting the ceiling. Should be rare in
			  practice but caught explicitly so corrupted cells don't slip
			  silently into the corpus.

			The bisection assumes cor(prob, c) is monotonically increasing in b
			for fixed (centrality, seed). This is true in expectation and
			approximately true per-seed for any non-pathological centrality
			vector. Numerical non-monotonicity (a few permil) can occur from
			finite-N sampling noise in u_i; the tolerance absorbs this.

			realized_rho_pos is the realized correlation at the bisected b
			against the POSITIVE prob formulation — i.e., before the sign
			flip Function 4 applies. The caller computing the full per-replicate
			record reports `sign * realized_rho_pos` as the realized_rho field.
			This separation keeps the bisection oblivious to sign and the
			sign-handling explicit at the call sites that need it.
		"""

		#	Guards
			(-1 < target_rho < 1) || throw(ArgumentError("target_rho must be in (-1, 1), got $target_rho"))

		#	Trivial Case: target_rho == 0 (MCAR Baseline)
			#	b = 0 exactly produces prob = u; cor(u, c) ≈ 0. No bisection
			#	needed. Compute the realized correlation at b = 0 for the
			#	record but skip the search.
				if target_rho == 0
					rho_at_0 = _centrality_correlation_for_b(centrality, 0.0, seed)
					if isnan(rho_at_0)
						return (b = 0.0, sign = 1, realized_rho_pos = NaN, status = :failed_other)
					end
					return (b = 0.0, sign = 1, realized_rho_pos = rho_at_0, status = :converged)
				end

		#	Sign Extraction and Absolute Target
			sgn = target_rho > 0 ? 1 : -1
			abs_target = abs(target_rho)

		#	Bisection Setup
			b_lo, b_hi = 0.0, 1.0
			rho_at_lo = _centrality_correlation_for_b(centrality, b_lo, seed)
			rho_at_hi = _centrality_correlation_for_b(centrality, b_hi, seed)

			#	Constant-Centrality Network: return :failed_other
				if isnan(rho_at_lo) || isnan(rho_at_hi)
					return (b = NaN, sign = sgn, realized_rho_pos = NaN, status = :failed_other)
				end

			#	Achievable-Ceiling Check: if cor at b=1 is below target, no
			#	value of b can reach the target — declare :ceiling_hit at b=1.
				if rho_at_hi < abs_target - tol
					return (b = 1.0, sign = sgn, realized_rho_pos = rho_at_hi, status = :ceiling_hit)
				end

		#	Bisection Loop
			b_mid          = 0.5
			rho_at_mid     = 0.0
			converged_iter = 0
			for iter in 1:max_iters
				b_mid      = 0.5 * (b_lo + b_hi)
				rho_at_mid = _centrality_correlation_for_b(centrality, b_mid, seed)

				if isnan(rho_at_mid)
					return (b = NaN, sign = sgn, realized_rho_pos = NaN, status = :failed_other)
				end

				#	Converged: realized matches target within tolerance
					if abs(rho_at_mid - abs_target) < tol
						return (b = b_mid, sign = sgn, realized_rho_pos = rho_at_mid, status = :converged)
					end

				#	Hit the b = 1 boundary: ceiling
					if 1.0 - b_mid < b_ceiling_eps && rho_at_mid < abs_target - tol
						return (b = 1.0, sign = sgn, realized_rho_pos = rho_at_hi, status = :ceiling_hit)
					end

				#	Update bracket (monotone increasing assumption)
					if rho_at_mid < abs_target
						b_lo = b_mid
					else
						b_hi = b_mid
					end
				converged_iter = iter
			end

		#	Max Iterations Exceeded Without Convergence — unusual; should not
		#	happen for max_iters = 50 since [0, 1] bisected 50 times reaches
		#	2^-50 ≈ 1e-15 resolution, well below any tolerance > 0.
			return (b = b_mid, sign = sgn, realized_rho_pos = rho_at_mid, status = :failed_other)
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
			  c_normalized_i with no noise component. The draw still works
			  but ties in centrality are broken arbitrarily by StatsBase.
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

			dropped = StatsBase.sample(rng, 1:n, StatsBase.Weights(prob), k; replace=false)
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
										bisection_tol::Real = 1e-3,
										bisection_max_iters::Int = 50)
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

		#	Bisection: find b for |target_rho|
			bisection = _bisect_b_for_target_rho(c, target_rho, bisection_seed;
												  tol = bisection_tol,
												  max_iters = bisection_max_iters)

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
			convention from SMM 2022 and \citet{Galaskiewicz1991}.

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
	`generate_missingness_mask`, `build_degeneration_corpus`

	**References**
	- Galaskiewicz, J. (1991). Estimating point centrality using different
	  network sampling techniques. *Social Networks*, 13(4), 347--386.
	""" apply_missingness

#	Build Degeneration Corpus: orchestrate the full (network, rho, rate, replicate) grid
	function build_degeneration_corpus(networks::Dict;
										target_rhos::AbstractVector{<:Real} = [-0.75, -0.25, 0.0, 0.25, 0.75],
										target_rates::AbstractVector{<:Real} = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
										n_replicates::Int = 100,
										replicates_per_network::Dict{String,Int} = Dict{String,Int}(),
										master_seed::Integer = 42,
										gc_threshold::Real = 0.30,
										min_n::Int         = 25,
										min_edges::Int     = 1,
										bisection_tol::Real      = 1e-3,
										bisection_max_iters::Int = 50,
										parallel::Bool     = true,
										show_progress::Bool = true)
		"""
		Args:
			networks::Dict: corpus keyed by network name → NamedTuple with
				:edges (DataFrame), :nodes (DataFrame), :metadata (NamedTuple
				containing at least :directed::Bool)
			target_rhos::AbstractVector{<:Real}: nominal rho grid; default
				[-0.75, -0.25, 0.0, 0.25, 0.75] per the validation roadmap
			target_rates::AbstractVector{<:Real}: nominal rate grid; default
				[0.05, 0.10, 0.15, 0.25, 0.40, 0.50]
			n_replicates::Int: default replicate count per cell; default 100
			replicates_per_network::Dict{String,Int}: per-network override of
				replicate count. Networks absent from this dict use n_replicates.
				Use for Marvel and other expensive networks.
			master_seed::Integer: master RNG seed; per-replicate seeds derived
				deterministically from master_seed and the (name, rho, rate,
				replicate_idx) tuple via hash
			gc_threshold, min_n, min_edges: passed to topological degeneracy
			bisection_tol, bisection_max_iters: passed to bisection
			parallel::Bool: when true, parallelize the flat grid loop via
				Threads.@threads :static (deterministic given pre-allocated
				per-tuple result storage). When false, run serially.
			show_progress::Bool: display a progress bar over the flat grid
		Returns:
			DataFrame with one row per (network, target_rho, target_rate,
			replicate_idx) cell. Columns:
				- network_name::String
				- nominal_rho::Float64
				- nominal_rate::Float64
				- replicate_idx::Int
				- seed::Int                 (the per-replicate seed actually used)
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
			Output structure. The flat DataFrame is the natural shape for
			downstream coverage analysis, degeneracy summaries, and grouping
			by (network, rho, rate). The dropped_nodes column is
			Vector{Vector{Int}}; Julia handles this without issue, but
			downstream code retrieves vectors by row index rather than via
			cross-table joins.

			Centrality and adjacency caching. The orchestrator computes
			centrality once per network (a property of the network, invariant
			across the 30 (rho, rate) cells), passing the cached vector into
			generate_missingness_mask. Adjacency is rebuilt once per call
			inside generate_missingness_mask — see the note on
			generate_missingness_mask about this minor waste; acceptable at
			Marvel scale.

			Seed scheme. The per-replicate seed is hash((network_name,
			target_rho, target_rate, replicate_idx, master_seed)) truncated
			to UInt32 then converted to Int. This is deterministic, gives
			independent streams per cell, and means any single (network, rho,
			rate, replicate) tuple can be regenerated from master_seed alone
			without rerunning the grid — required by the punch list's seed
			discipline.

			Replication counts. n_replicates is the default. To use a
			different count for Marvel (or any expensive network), pass
			replicates_per_network = Dict("marvel_universe_unweighted" => 25,
			                               "marvel_universe_weighted" => 25).
			Per-network counts override the default for those entries.

			Threading. The flat grid (network × rho × rate × replicate) is
			enumerated upfront into a Vector, and the loop over that vector
			is parallelized via Threads.@threads :static with pre-allocated
			per-tuple result storage. Deterministic regardless of thread
			count. Each tuple has its own seed (per the seed scheme above)
			and writes to its own pre-allocated slot, so there's no shared
			mutable state and no risk of accumulator races.

			Failure handling. generate_missingness_mask returns
			:failed_other records as placeholders (empty dropped_nodes, NaN
			realized values, fully degenerate flags). The orchestrator
			collects them like normal records; downstream analysis decides
			whether to filter or summarize. No try/catch in the orchestrator
			itself — the per-replicate failure surface is small and well-
			defined inside the kernel.
		"""

		#	Guards
			isempty(networks) && throw(ArgumentError("networks dict is empty"))
			n_replicates >= 1 || throw(ArgumentError("n_replicates must be >= 1, got $n_replicates"))

		#	Compute centrality once per network (cached for the (rho, rate, rep) inner loops)
			network_names = sort!(collect(keys(networks)))
			centrality_cache = Dict{String, Vector{Float64}}()
			for name in network_names
				net = networks[name]
				centrality_cache[name] = _centrality_for_sampler(net.edges;
																  nodes    = net.nodes,
																  directed = net.metadata.directed)
			end

		#	Enumerate the flat grid: every (network, rho, rate, replicate) tuple
			#	Pre-counted so result storage can be pre-allocated.
				grid_size = 0
				for name in network_names
					n_reps = get(replicates_per_network, name, n_replicates)
					grid_size += length(target_rhos) * length(target_rates) * n_reps
				end
				flat_tuples = Vector{Tuple{String,Float64,Float64,Int}}(undef, grid_size)
				let idx = 1
					for name in network_names
						n_reps = get(replicates_per_network, name, n_replicates)
						for rho in target_rhos
							for rate in target_rates
								for rep in 1:n_reps
									flat_tuples[idx] = (name, Float64(rho), Float64(rate), rep)
									idx += 1
								end
							end
						end
					end
				end

		#	Pre-allocate result storage by flat-grid index (race-free under threading)
			#	Each thread writes only to its own slot. Results are folded into
			#	the final DataFrame in grid-tuple order after the loop.
				results = Vector{NamedTuple}(undef, grid_size)

		#	Progress bar setup (same pattern as _triad_census_layered)
			use_threads = parallel && Threads.nthreads() > 1 && grid_size > 1
			desc        = "Degeneration grid (" * string(grid_size) * " cells, " *
						  (use_threads ? "$(Threads.nthreads()) threads" : "serial") * ")"
			prog        = show_progress ? Progress(grid_size, desc = desc, enabled = true) : nothing
			prog_lock   = ReentrantLock()

		#	Run the Flat Grid Loop (Threaded or Serial)
			if use_threads
				Threads.@threads :static for k in 1:grid_size
					name, rho, rate, rep = flat_tuples[k]
					net      = networks[name]
					rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
					record   = generate_missingness_mask(net.edges;
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
														  bisection_max_iters = bisection_max_iters)
					results[k] = (name = name, rho = rho, rate = rate, rep = rep, seed = rep_seed, record = record)
					if show_progress
						lock(prog_lock) do
							next!(prog)
						end
					end
				end
			else
				for k in 1:grid_size
					name, rho, rate, rep = flat_tuples[k]
					net      = networks[name]
					rep_seed = Int(hash((name, rho, rate, rep, master_seed)) % UInt32)
					record   = generate_missingness_mask(net.edges;
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
														  bisection_max_iters = bisection_max_iters)
					results[k] = (name = name, rho = rho, rate = rate, rep = rep, seed = rep_seed, record = record)
					if show_progress
						next!(prog)
					end
				end
			end

		#	Flatten per-replicate records into rectangular DataFrame columns
			#	Each results[k].record is a NamedTuple per Function 6's contract.
			#	We unpack the nested NamedTuples (.nominal, .sampler_degeneracy)
			#	into top-level columns.
				out = DataFrame(
					network_name             = String[],
					nominal_rho              = Float64[],
					nominal_rate             = Float64[],
					replicate_idx            = Int[],
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

		#	Return
			return out
	end
	@doc raw"""
	**Description**
	Orchestrates the full Phase~1 degeneration grid. Given a corpus of
	networks and the design grid (target $\rho$ values, target rates,
	replicate count), this function produces a per-replicate record for
	every cell and returns the results as a single rectangular DataFrame.

	**Usage**
    ```julia
        corpus = build_degeneration_corpus(networks;
            target_rhos    = [-0.75, -0.25, 0.0, 0.25, 0.75],
            target_rates   = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50],
            n_replicates   = 100,
            master_seed    = 42,
        )
    ```

	**Arguments**
	- `networks::Dict`: corpus keyed by network name $\to$ NamedTuple with
	  `:edges`, `:nodes`, `:metadata.directed`.
	- `target_rhos::AbstractVector{<:Real}`: the $\rho$ grid. Default is the
	  five levels in the validation roadmap.
	- `target_rates::AbstractVector{<:Real}`: the rate grid. Default is the
	  six rates in the validation roadmap.
	- `n_replicates::Int`: default replicate count per cell (default 100).
	- `replicates_per_network::Dict{String,Int}`: per-network override
	  (e.g., to reduce Marvel's count).
	- `master_seed::Integer`: master seed; per-replicate seeds derived
	  deterministically from $(\text{name}, \rho, \text{rate}, \text{rep}, \text{master})$.
	- `gc_threshold`, `min_n`, `min_edges`: passed to topological degeneracy.
	- `bisection_tol`, `bisection_max_iters`: passed to bisection.
	- `parallel::Bool`: parallelize the flat grid loop (default true).
	- `show_progress::Bool`: display a progress bar (default true).

	**Value**
	`DataFrame` with one row per $(network, \rho, \text{rate}, \text{rep})$
	cell. Sixteen columns: identifiers (`network_name`, `nominal_rho`,
	`nominal_rate`, `replicate_idx`, `seed`), the sampler output
	(`dropped_nodes`, `realized_rate`, `realized_rho`, `bisection_status`),
	and the topological degeneracy fields flattened from
	`sampler_degeneracy`.

	**Notes**
	The DataFrame contains every replicate including those flagged degenerate
	or with `bisection_status == :failed_other`. Filtering is the consumer's
	job: a coverage analysis might restrict to `any_topo_degenerate == false`,
	while a degeneracy-rate analysis wants all rows. The per-cell degeneracy
	rate (used for grid trimming) is computed via grouping on (`network_name`,
	`nominal_rho`, `nominal_rate`) and summarizing `any_topo_degenerate`.

	Threading is enabled by default and uses `Threads.@threads :static` with
	pre-allocated per-tuple result storage. Output is deterministic in
	`master_seed` regardless of thread count.

	**See Also**
	`generate_missingness_mask`, `apply_missingness`,
	`_centrality_for_sampler`

	**References**
	- Smith, J. A., Morgan, J. H., & Moody, J. (2022). *Social Networks*, 68.
	""" build_degeneration_corpus

#   Exports (public API)
    export generate_missingness_mask,
           apply_missingness,
           build_degeneration_corpus

end # module network_degeneracy