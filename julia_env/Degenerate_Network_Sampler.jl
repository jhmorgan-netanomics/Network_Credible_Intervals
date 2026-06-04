#Generate Degraded Networks
#Jonathan H. Morgan
#29 May 2026

```
This stimulus generates the Phase-1 validation corpus for the reconstruction method — the controlled degradation against which its credible intervals are scored.
It follows the evaluation philosophy of Smith et al. (2022): take a known true network, induce missing data systematically, recompute the measures of interest,
and characterize the bias (and here, interval coverage) as a function of the amount and type of missingness. Smith et al. induce missingness as node non-response —
a roster of identifiable actors, some of whom report no out-ties — and impute edges to and from those known non-respondents. The node-non-response case is covered
by the framework in Phase 1.5; it is not a factor varied in this Phase-1 stimulus. Phase 1 instead samples fixed-roster true network and induces missingness by
zeroing edge weight, and the nodes that fully zero out are recovered along with the edges. The two phases exercise complementary mechanisms
— node non-response in 1.5, edge missingness here.

Missingness is applied through a single operation: zeroing edge weight. A missing node is not a separate object that is deleted; it is the limiting case of a
node whose entire edge set has gone to zero. Node loss is therefore emergent rather than dialed — there is no node-missingness target. The single varied quantity
is edge missingness, the fraction of total edge weight removed, swept over {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}; the 0.0 level is the undegraded baseline, at which
bias is zero and every interval contains the truth by construction.

The mechanism that realizes edge missingness depends on network type, but the axis is shared. On weighted networks, weight is removed, producing both
underweighted ties and fully zeroed ones. On binary networks, where no partial weight exists, the same operation reduces to tie removal — the unit-weight limit,
exemplified by the binary Toledo Crime network. Both feed the same edge-missingness axis and both produce emergent node loss, so weighted and binary networks remain
comparable along a single x-axis.

Centrality-biased missingness is controlled by rho, the rank correlation (Kendall tau-b) between a node's centrality and the missingness it experiences. The
EXPERIMENTAL CONDITION IS THE REALIZED rho — the correlation measured on the materialized degraded network. Realized rho is binned by SIGN into 
three conditions: negative (centrality-anti-correlated loss, periphery orphans first), zero (no centrality bias), and positive (centrality-correlated loss, 
hubs shed first). It is bounded to a baseline range of |realized rho| <= 0.25: a draw whose realized rho lands outside [-0.25, +0.25] is NOT kept. The bound is a 
chosen baseline so every network is compared over one common rho range. We note explicitly that at high edge-missingness the mechanism naturally produces realized 
rho well past +-0.25 (sparse networks can exceed +-0.6); the bounded corpus is therefore the low-rho slice of what actually occurs there. This buys 
cross-network comparability at the stated cost of not representing the extreme-rho regime. rho interacts with the level: negative rho concentrates removal on 
low-degree peripheral nodes, which orphan quickly and yield more node loss at a given level, while positive rho lands removal on high-degree nodes with ties to 
spare and yields less. Induced node loss is thus a joint function of network, level, and rho — not of the level alone.

Degradation runs from edges to nodes: zero edges, and observe which nodes drop. Reconstruction runs the other way: assume some number of nodes are missing, then
distribute weight to recover them and their ties. The bridge is the recorded emergent node-loss count (n_organic_losses, equivalently the recorded missing-node set),
which is precisely the input the reconstruction consumes. Because node loss can in principle be zero — a low level on a heavy or dense network may leave every node
with a surviving tie, retaining the full roster, though this is uncommon among the empirical networks — the count is recorded per cell rather than derived from the
level, and a count of zero is a valid record, handled by reconstructing edge weight alone.

For this validation the analyst is granted ground-truth knowledge of two quantities: the edge-loss fraction and the induced node-loss count. rho is withheld; it is
the unknown whose misspecification the reconstruction probes. Downstream of this stimulus, each recorded sample is reconstructed under each rho value, so that the
gap between the matched line (assumed rho equal to the true generating rho) and the two mismatched lines quantifies the cost of getting the centrality mechanism wrong. The stimulus itself varies only the true conditions — network, edge-missingness level, and true (realized) rho sign — and records the realized quantities the reconstruction will need.

One extension is noted but not implemented here. In practice, analysts typically estimate the proportion of nodes lost more reliably than the proportion of ties.
A natural variant would invert which quantity is treated as known: grant an accurate node proportion while misspecifying or omitting the tie percentage, and measure
the resulting degradation. The current design fixes both as known and probes only rho.
````

#=  ──────────────────────────────────────────────────────────────────────────
    STIMULUS DESIGN (what this script produces and how)
    ──────────────────────────────────────────────────────────────────────────
    Varied true factors (the only axes that exist in the corpus):
      - network          : all 18 (9 topologies x {weighted, binary})
      - edge missingness : pi_edge in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}  (figure x-axis)
      - true rho SIGN    : negative / zero / positive  (figure rows), defined on
                           REALIZED rho, bounded to |realized| <= RHO_BOUND (0.25):
                             negative : realized in [-0.25, -CORE_BAND)
                             zero     : |realized| <= CORE_BAND          (CORE_BAND = 0.03)
                             positive : realized in (+CORE_BAND, +0.25]

    Generation strategy (generate-then-keep, not a fixed grid):
      1. Run the validated build_degeneration_corpus over a SWEEP of nominal rho
         dials (NOMINAL_SWEEP). The dial is invisible plumbing; sweeping it spreads
         the realized rho so that, across networks and levels, realized lands
         throughout [-0.25, +0.25]. (Weak dials catch sparse/high-missingness cells
         where a strong dial overshoots; strong dials catch dense cells that barely
         move.) node_loss = :emergent, so the missing set is the organic losses.
      2. KEEP only draws with |realized_rho| <= RHO_BOUND. Overshoots are discarded
         (a draw aimed negative that realizes -0.40 is thrown out).
      3. Bin the survivors by realized-rho sign and take up to PER_BIN (500) per
         (network, pi_edge, sign bin) so the three rows have comparable precision.
      4. pi_edge = 0.0 is the undegraded baseline (no removal, realized rho = 0);
         it is the x=0 anchor / truth, so we keep ONE baseline record per network
         rather than 500 identical copies.

    Bookkeeping (which nodes and edges were removed):
      - Removed NODES are stored directly as `missing_nodes` (the organic-loss set).
      - Removed EDGES are documented by DETERMINISTIC REGENERATION: each row stores
        `seed` + the nominal params actually used (`nominal_rho_used`, nominal_pi_node,
        nominal_pi_edge). generate_missingness_mask is deterministic in the seed, so
        the exact degraded network — and hence the removed-edge set — is recoverable
        on demand. Full per-sample edge lists are NOT stored (prohibitive: marvel at
        pi_edge=0.5 removes ~38k edges x 162k rows). The VERIFICATION section below
        regenerates a sample and recovers its removed nodes (and edges, once the mask
        exposes them) to prove the seed-based documentation round-trips.
        NOTE: recovering the removed EDGES requires generate_missingness_mask to
        return the degraded edge list (a `return_removed=true` path). Until that
        addition lands, node recovery works and the edge block is marked PENDING.

    Out of scope here (all reconstruction-side, downstream of this corpus):
      - assumed rho (the three bands per panel: -0.25 / 0 / +0.25)
      - the four centrality measures (panel columns)
      - posterior means, 89% credible bands, the bias table
    ────────────────────────────────────────────────────────────────────────── =#

#	Generates the degeneration corpus across all eighteen networks (nine
#	topologies x {weighted, binary}), six edge-missingness levels, and three
#	realized-rho sign conditions bounded to |realized| <= 0.25, in emergent
#	node-loss mode. No node-rate axis and no materialization-mechanism axis:
#	node loss is emergent, full-removal vs nomination is composed per node.

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
    using SparseArrays
    using Statistics
    using StatsBase
    using Network_Credible_Intervals

    import Network_Credible_Intervals.network_degeneracy: build_degeneration_corpus

#	Internals used only by the reproducibility / bookkeeping verification.
    const NDG = Network_Credible_Intervals.network_degeneracy

###################
#   IMPORT DATA   #
###################

#   Load Corpus
#	Mirrors the loader in Network_Degradation_Tests.jl: read every
#	GraphML in the test-networks directory into the canonical
#	(edges, nodes, metadata) NamedTuple format expected by the
#	orchestrator. metadata carries both :directed and :weighted.
	data_dir      = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks"
	graphml_files = sort(filter(f -> endswith(f, ".graphml"), readdir(data_dir)))
	networks      = Dict{String, NamedTuple}()
	for filename in graphml_files
		filepath     = joinpath(data_dir, filename)
		network_name = replace(filename, ".graphml" => "")
		networks[network_name] = load_graphml(filepath)
	end

#   Examine Data
    println("Corpus: ", length(networks), " networks")
    for (name, net) in sort(collect(networks), by = first)
        println("  $(rpad(name, 42)) N=$(rpad(nrow(net.nodes), 8)) E=$(rpad(nrow(net.edges), 10)) directed=$(rpad(net.metadata.directed, 6)) weighted=$(net.metadata.weighted)")
    end
    println()

###################################
#   GENERATE DEGENERATE SAMPLES   #
###################################

#   Configuration
    master_seed   = 42
    output_dir    = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degenerate_Networks"
    output_file   = joinpath(output_dir, "degeneration_corpus.arrow")

#	The realized-rho baseline and the sign-bin partition (see DESIGN block).
    RHO_BOUND     = 0.25      # keep only |realized_rho| <= RHO_BOUND
    CORE_BAND     = 0.03      # |realized_rho| <= CORE_BAND is the zero condition
    PER_BIN       = 500       # kept samples per (network, pi_edge, sign bin)

#	Edge-missingness levels (figure x-axis). 0.0 is the undegraded baseline.
    edge_levels   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

#	NOMINAL dial sweep — invisible plumbing. Dense near zero so sparse/high-
#	missingness cells (which overshoot at strong dials) still land in-bound, with
#	a few strong dials so dense cells reach the wings. Realized, not these, is the
#	condition; these only spread it across [-0.25, +0.25].
    nominal_sweep = [-0.30, -0.20, -0.12, -0.07, -0.03, 0.0, 0.03, 0.07, 0.12, 0.20, 0.30]

#	Single propensity-tilt calibration (emergent mode: pi_node is not a node target).
    tilt_pi_node  = [0.10]

#	Replicates per (network, dial, level) in the over-generated POOL. The pool is
#	filtered to the bound and truncated to PER_BIN, so this must be generous enough
#	that each sign bin still fills after rejection. Start at the pilot value, raise
#	for production once the per-bin fill report looks healthy.
    pool_reps     = 60        # PILOT. Production: raise until bins fill (see fill report).

#   Pre-Flight
    println("─" ^ 70)
    println("Phase 1 Production Corpus Generation")
    println("─" ^ 70)
    println("Started:           ", now())
    println("Julia version:     ", VERSION)
    println("Threads:           ", Threads.nthreads())
    println("Master seed:       ", master_seed)
    println("Networks:          ", length(networks))
    println("Edge levels:       ", edge_levels)
    println("Nominal sweep:     ", nominal_sweep)
    println("rho bound:         |realized| <= ", RHO_BOUND, "   core band: ", CORE_BAND)
    println("Per-bin target:    ", PER_BIN, " kept per (network, level, sign)")
    println("Pool reps:         ", pool_reps, " per (network, dial, level)")
    println("Output file:       ", output_file)
    println()

    mkpath(output_dir)
    Threads.nthreads() >= 4 || @warn "Only $(Threads.nthreads()) threads available; orchestrator was tuned for >= 4"

#   Step 1 — Generate the over-generated POOL via the validated orchestrator.
#	Sweeping nominal_sweep spreads realized rho; we filter/bin afterward.
    println("Step 1/3: generating pool via build_degeneration_corpus (nominal sweep) ...")
    println()

    @time pool = build_degeneration_corpus(networks;
                                           target_rhos     = nominal_sweep,
                                           target_pi_nodes = tilt_pi_node,
                                           target_pi_edges = edge_levels,
                                           node_loss       = :emergent,
                                           n_replicates    = pool_reps,
                                           master_seed     = master_seed,
                                           parallel        = true,
                                           show_progress   = true)
    println("Pool rows: ", nrow(pool))
    println()

#   Step 2 — Bound + sign-bin + per-bin truncation.
    println("Step 2/3: applying |realized| <= $RHO_BOUND bound, sign-binning, truncating to $PER_BIN/bin ...")

#	The dial actually used by the orchestrator is substituted_rho (feasibility may
#	clamp the swept nominal); we keep it as the regeneration key, not nominal_rho.
    pool.nominal_rho_used = pool.substituted_rho

#	Sign-bin on realized rho. Baseline (pi_edge == 0) is forced to :core — it is the
#	undegraded truth, realized rho is 0 there by construction.
    function sign_bin(realized::Real, pie::Real)
        pie == 0.0            && return :core
        realized < -CORE_BAND && return :negative
        realized >  CORE_BAND && return :positive
        return :core
    end
    pool.rho_bin = sign_bin.(pool.realized_rho, pool.nominal_pi_edge)

#	Keep only in-baseline draws. (At pi_edge == 0 realized is 0, trivially in-bound.)
    in_bound = pool[abs.(pool.realized_rho) .<= RHO_BOUND, :]

#	Truncate to PER_BIN per (network, level, sign). At pi_edge == 0 keep ONE
#	baseline row per network (the x=0 anchor), not PER_BIN identical copies.
    kept = DataFrame()
    for g in groupby(in_bound, [:network_name, :nominal_pi_edge, :rho_bin])
        cap  = g.nominal_pi_edge[1] == 0.0 ? 1 : PER_BIN
        take = min(nrow(g), cap)
        append!(kept, first(g, take))      # deterministic: seeds make rows reproducible
    end
    corpus_df = kept
    println("Kept rows: ", nrow(corpus_df), "  (from pool ", nrow(pool), ")")
    println()

#   Step 3 — Per-bin fill report (watch for under-filled cells, esp. wings at
#	low pi_edge / on dense nets, where realized rho structurally can't leave ~0).
    println("Step 3/3: per-bin fill (network x level x sign) — flag bins under $PER_BIN")
    println("─" ^ 70)
    fill = combine(groupby(corpus_df, [:network_name, :nominal_pi_edge, :rho_bin]), nrow => :n)
    for grp in groupby(fill, :network_name)
        under = grp[(grp.n .< PER_BIN) .& (grp.nominal_pi_edge .!= 0.0), :]
        flag  = isempty(under) ? "ok" : "UNDER: " * join(["$(r.rho_bin)@$(r.nominal_pi_edge)=$(r.n)" for r in eachrow(under)], ", ")
        println("  $(rpad(grp.network_name[1], 42)) rows=$(rpad(sum(grp.n), 6)) $flag")
    end
    println()

#	Diagnostic: max emergent node loss per network (compare to the loss maps).
    println("Max emergent node loss per network (should track the loss maps)")
    println("─" ^ 70)
    for grp in groupby(corpus_df, :network_name)
        println("  $(rpad(grp.network_name[1], 42)) maxloss=$(rpad(maximum(grp.n_organic_losses), 6)) realized_rho in [$(round(minimum(grp.realized_rho); digits=3)), $(round(maximum(grp.realized_rho); digits=3))]")
    end
    println()

#   Write-Out Arrow File
#	Stored columns ARE the bookkeeping: missing_nodes is the removed-node set;
#	seed + nominal_rho_used + nominal_pi_node + nominal_pi_edge regenerate the exact
#	degraded network (and thus the removed-edge set) on demand.
    println("Writing Arrow file ...")
    Arrow.write(output_file, corpus_df; compress = :zstd)
    println("Wrote: ", output_file, " (", round(filesize(output_file) / 1024^2, digits = 1), " MB)")
    println()
    println("Done: ", now())

####################
#   VERIFICATION   #
####################

#   Read Back
    read_df = DataFrame(Arrow.Table(output_file))
    println("Rows:            ", nrow(read_df))
    println("Columns:         ", names(read_df))
    println("Networks:        ", length(unique(read_df.network_name)))
    println("Sign conditions: ", sort(unique(String.(read_df.rho_bin))))
    println("Edge levels:     ", sort(unique(read_df.nominal_pi_edge)))
    println("realized_rho in: [", round(minimum(read_df.realized_rho); digits=3), ", ", round(maximum(read_df.realized_rho); digits=3), "]  (bound = +-", RHO_BOUND, ")")
    println()

#	Bound sanity: nothing past the baseline should have survived.
    n_oob = count(abs.(read_df.realized_rho) .> RHO_BOUND + 1e-9)
    println("Out-of-bound rows (expect 0): ", n_oob)
    println()

#	Pick representative rows across networks / levels / signs.
    sample_indices = round.(Int, range(1, nrow(read_df); length = 5))
    for idx in sample_indices
        row  = read_df[idx, :]
        miss = collect(row.missing_nodes)
        println("Row $idx: $(row.network_name)  level=$(row.nominal_pi_edge)  sign=$(row.rho_bin)")
        println("  realized_rho=$(round(row.realized_rho; digits=3))  realized_pi_edge=$(round(row.realized_pi_edge; digits=3))  n_organic_losses=$(row.n_organic_losses)  |missing|=$(length(miss))")
        println("  identity (|missing| == n_organic_losses): $(length(miss) == row.n_organic_losses)")
    end
    println()

#	Bookkeeping round-trip: regenerate one row from its stored seed + params,
#	confirm the removed NODES match, and recover the removed EDGES.
    repro_idx = clamp(nrow(read_df) ÷ 3, 1, nrow(read_df))
    row = read_df[repro_idx, :]
    net = networks[row.network_name]

    c    = NDG._centrality_for_sampler(net.edges; nodes = net.nodes, directed = net.metadata.directed)
    adjb = NDG._graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = false)[1]
    comm = NDG._true_communities(adjb)

    rec = NDG.generate_missingness_mask(net.edges;
                nodes          = net.nodes,
                directed       = net.metadata.directed,
                weighted       = net.metadata.weighted,
                node_loss      = :emergent,
                target_pi_node = row.nominal_pi_node,
                target_pi_edge = row.nominal_pi_edge,
                target_rho     = row.nominal_rho_used,
                seed           = row.seed,
                centrality     = c,
                true_community = comm,
                return_removed = true)

    println("Bookkeeping round-trip on row $repro_idx ($(row.network_name)):")
    println("  removed-node set matches stored: ", collect(row.missing_nodes) == rec.missing_nodes)
    println("  n_organic_losses matches:        ", row.n_organic_losses == rec.n_organic_losses)
    println("  realized_pi_edge matches:        ", isapprox(row.realized_pi_edge, rec.realized_pi_edge))

#	Removed-EDGE recovery from the same regenerated record. degraded_edges is
#	row-aligned with net.edges (apply_weight_removal copies the rows and reduces
#	weight in place, zeros retained), so a removed tie is a row whose weight
#	reached zero; on binary nets that is the whole-tie removal count and must
#	equal n_edges_zeroed. observed_edges/observed_nodes is the materialized
#	network the reconstruction consumes (full-removal nodes + edges dropped).
    w_orig        = Float64.(net.edges.weight)
    w_deg         = Float64.(rec.degraded_edges.weight)
    zeroed_mask   = (w_orig .> 0.0) .& (w_deg .== 0.0)
    reduced_mask  = (w_deg .< w_orig) .& (w_deg .> 0.0)
    removed_edges = net.edges[zeroed_mask, :]    # the documented removed-tie set
    println("  removed-edge count (weight->0): ", count(zeroed_mask), "   (record n_edges_zeroed = ", row.n_edges_zeroed, ")")
    println("  zeroed matches record:          ", count(zeroed_mask) == row.n_edges_zeroed)
    println("  underweighted ties (weighted):  ", count(reduced_mask))
    println("  observed network:               nodes $(nrow(rec.observed_nodes))/$(nrow(net.nodes))  edges $(nrow(rec.observed_edges))/$(nrow(net.edges))")
    println()

#	SBM-directed is the deliberate zero-loss baseline; confirm it stays ~0.
    sbm_dir = filter(r -> occursin("sbm_directed", r.network_name), read_df)
    println("SBM-directed max loss (expect 0): ", isempty(sbm_dir) ? "n/a" : maximum(sbm_dir.n_organic_losses))
    println()