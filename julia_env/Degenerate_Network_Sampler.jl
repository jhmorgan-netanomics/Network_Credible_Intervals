#Generate Degraded Networks for Testing
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

Edge missingness is applied through a single operation: zeroing edge weight. A missing node is not a separate object that is deleted; it is the limiting case of a
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

Node Missingness...

````

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
    using Random
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

#############################################
#   GENERATE DEGENERATE EDGE LOSS SAMPLES   #
#############################################

#   Configuration
    master_seed   = 42
    output_dir    = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degenerate_Networks"
    output_file   = joinpath(output_dir, "degeneration_corpus.arrow")

#	The realized-rho baseline and the sign-bin partition (see DESIGN block).
    RHO_BOUND     = 0.75      # keep only |realized_rho| <= RHO_BOUND
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
    pool_reps     = 200        # PILOT. Production: raise until bins fill (see fill report).

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

#	Sign-bin on realized rho, conditioned on emergent node loss.
#	  pi_edge == 0 -> :baseline (undegraded anchor; realized rho = 0 by construction)
#	  losses == 0  -> :no_loss  (edges removed, no node orphaned; a valid edge-only
#	                             record, but NOT a rho condition — rho is undefined
#	                             with an empty missing set)
#	  otherwise bin by realized-rho sign; :core is the genuine random-loss (rho ~ 0) row.
    function sign_bin(realized::Real, pie::Real, losses::Integer)
        pie == 0.0            && return :baseline
        losses == 0           && return :no_loss
        realized < -CORE_BAND && return :negative
        realized >  CORE_BAND && return :positive
        return :core
    end
    pool.rho_bin = sign_bin.(pool.realized_rho, pool.nominal_pi_edge, pool.n_organic_losses)

#	Keep only in-baseline draws. (At pi_edge == 0 realized is 0, trivially in-bound.)
    in_bound = pool[abs.(pool.realized_rho) .<= RHO_BOUND, :]

#	Sample (not truncate) to PER_BIN per (network, level, sign), seeded for
#	reproducibility, so each bin spans the realized-rho range rather than clustering
#	on the early (weak) dials. At pi_edge == 0 keep ONE baseline row per network.
    rng  = MersenneTwister(master_seed)
    kept = DataFrame()
    for g in groupby(in_bound, [:network_name, :nominal_pi_edge, :rho_bin])
        cap  = g.nominal_pi_edge[1] == 0.0 ? 1 : PER_BIN
        n    = nrow(g)
        take = min(n, cap)
        idx  = take == n ? collect(1:n) : sort(sample(rng, 1:n, take; replace = false))
        append!(kept, g[idx, :])
    end
    corpus_df = kept

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

#   Idenitifying Rho Bounds
#   Write-Out Realized-Rho Envelope (from the FULL pool, not the sampled corpus)
#	Per-network reach that sets the category bands and the reconstruction guesses.
#	corpus_df is sampled to PER_BIN, so its extremes understate the reach; these come
#	from every pool draw. One row per network, so cheap to persist.
    envelope = combine(groupby(pool, :network_name),
                       :realized_rho     => minimum => :min_neg,
                       :realized_rho     => maximum => :max_pos,
                       :n_organic_losses => maximum => :max_losses,
                       nrow              => :n_pool_draws)
    envelope.max_loss_rate = envelope.max_losses ./ [nrow(networks[n].nodes) for n in envelope.network_name]
    envelope_file = joinpath(output_dir, "rho_envelope.arrow")
    Arrow.write(envelope_file, envelope; compress = :zstd)
    println("Wrote envelope: ", envelope_file)
    show(sort(envelope, :max_pos), allrows = true, allcols = true)
    println()

##############################################
#    GENERATE DEGENERATE NODE LOSS SAMPLES   #
##############################################

#   Configuration
#	NODE_LEVELS is the shared missingness axis (matched to the edge arm's
#	headline %, here as % of NODES). rho_sweep is the nominal dial — invisible
#	plumbing; realized rho, not these, is the condition. Symmetric and wider
#	than the edge sweep because the node arm reaches both wings; dense near the
#	design values (+-0.15, +-0.35) and a few strong dials to push to the ceiling.
    NODE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
    rho_sweep   = [-0.70, -0.55, -0.45, -0.35, -0.25, -0.15, -0.07, 0.0,
                    0.07,  0.15,  0.25,  0.35,  0.45,  0.55,  0.70]

#	PURE node arm: no edge degradation, so the missing set is the targeted
#	top-up only (organic losses == 0 throughout).
    edge_levels = [0.0]

#	Replicates per (network, dial, level) in the pool. Extremes drift outward
#	with reps, so match the edge production run (200). Lower for a pilot.
    pool_reps   = 200

    master_seed = 42

#	Outputs.
    output_dir      = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degraded_Networks"
    isdir(output_dir) || mkpath(output_dir)
    pool_file       = joinpath(output_dir, "node_pool.arrow")
    envelope_file   = joinpath(output_dir, "node_rho_envelope.arrow")
    net_env_file    = joinpath(output_dir, "node_rho_envelope_by_network.arrow")

#   Pre-Flight
    println("─" ^ 70)
    println("NODE-MISSINGNESS ARM — envelope run")
    println("Node levels:       ", NODE_LEVELS, "  (% of nodes; missingness axis)")
    println("Nominal rho sweep: ", rho_sweep)
    println("Edge degradation:  ", edge_levels, "  (pure node arm)")
    println("Pool reps:         ", pool_reps, " per (network, dial, level)")
    println("─" ^ 70)
    println()

#   Step 1 — Generate the over-generated POOL via the validated orchestrator.
#	node_loss = :targeted tops up to target_pi_node via the centrality tilt;
#	the bisection solves b per cell to chase each nominal target_rho.
    println("Step 1/3: generating node pool via build_degeneration_corpus (targeted) ...")
    println()

    @time pool = build_degeneration_corpus(networks;
                                           target_rhos     = rho_sweep,
                                           target_pi_nodes = NODE_LEVELS,
                                           target_pi_edges = edge_levels,
                                           node_loss       = :targeted,
                                           n_replicates    = pool_reps,
                                           master_seed     = master_seed,
                                           parallel        = true,
                                           show_progress   = true)
    println("Pool rows: ", nrow(pool))
    println()

#	The dial actually used by the orchestrator is substituted_rho (feasibility
#	may clamp the swept nominal); keep it as the regeneration key.
    pool.nominal_rho_used = pool.substituted_rho

#   Step 2 — Reach envelope. NODE arm keys on nominal_pi_node (the level), not
#	pi_edge. Per (network, level): realized-rho reach, the rate ceiling at that
#	level, and the achieved FRACTION of the ceiling on each wing.
    println("Step 2/3: building per-(network, pi_node) reach envelope ...")

    level_env = combine(groupby(pool, [:network_name, :nominal_pi_node]),
                        :realized_rho     => minimum => :min_neg,
                        :realized_rho     => maximum => :max_pos,
                        :realized_pi_node => maximum => :realized_pi_node_max,
                        :gc_collapse      => (x -> count(x) / length(x)) => :frac_gc_collapse,
                        nrow              => :n_draws)

#	Rate ceiling at the EXACT node rate (targeted mode hits pi_node exactly),
#	and the achieved fraction of it on each wing.
    level_env.ceiling  = sqrt.(2 .* level_env.nominal_pi_node .* (1 .- level_env.nominal_pi_node))
    level_env.frac_neg = abs.(level_env.min_neg) ./ level_env.ceiling
    level_env.frac_pos = level_env.max_pos ./ level_env.ceiling
    sort!(level_env, [:network_name, :nominal_pi_node])

#	Per-network collapse (reach over all levels) — the headline, analogue of the
#	edge arm's per-network envelope.
    net_env = combine(groupby(pool, :network_name),
                      :realized_rho => minimum => :min_neg,
                      :realized_rho => maximum => :max_pos,
                      nrow          => :n_draws)
    sort!(net_env, :max_pos)

    println()
    println("Per-network reach (sorted by max_pos):")
    show(net_env, allrows = true, allcols = true); println()
    println()

#   Step 3 — Write-Out. Pool (raw, for the later select-on-realized corpus step)
#	plus the per-level and per-network reach envelopes (the deliverables).
    println("Step 3/3: writing pool + envelopes ...")
    Arrow.write(pool_file,     pool;      compress = :zstd)
    Arrow.write(envelope_file, level_env; compress = :zstd)
    Arrow.write(net_env_file,  net_env;   compress = :zstd)
    println("Wrote: ", pool_file,     " (", round(filesize(pool_file)     / 1024^2, digits = 1), " MB)")
    println("Wrote: ", envelope_file, " (", round(filesize(envelope_file) / 1024^2, digits = 1), " MB)")
    println("Wrote: ", net_env_file,  " (", round(filesize(net_env_file)  / 1024^2, digits = 1), " MB)")
    println()
    println("Done: ", now())

####################
#   VERIFICATION   #
####################

#   Read Back
    read_pool = DataFrame(Arrow.Table(pool_file))
    read_env  = DataFrame(Arrow.Table(envelope_file))
    println("Pool rows:       ", nrow(read_pool))
    println("Columns:         ", names(read_pool))
    println("Networks:        ", length(unique(read_pool.network_name)))
    println("Node levels:     ", sort(unique(read_pool.nominal_pi_node)))
    println("realized_rho in: [", round(minimum(read_pool.realized_rho); digits=3), ", ", round(maximum(read_pool.realized_rho); digits=3), "]")
    println()

#	Pure node arm: NO weight removal anywhere, so organic losses must be 0.
    println("Organic losses (expect all 0 in pure node arm): max = ", maximum(read_pool.n_organic_losses))
    println()

#	Rate exactness: targeted mode drops round(pi_node * N) nodes, so realized
#	pi_node should equal nominal_pi_node up to one node (1/N).
    rate_ok = all(abs.(read_pool.realized_pi_node .- read_pool.nominal_pi_node) .<= (1.0 ./ map(r -> nrow(networks[r].nodes), read_pool.network_name)) .+ 1e-9)
    println("Rate exactness (|realized - nominal| <= 1/N): ", rate_ok)
    println()

#	Ceiling sanity: |realized_rho| should sit at or below sqrt(2p(1-p)); any
#	overshoot is centrality-tie inflation (the toledo signature on the edge side)
#	or small-net discreteness — flag, don't fail.
    read_pool.ceiling = sqrt.(2 .* read_pool.nominal_pi_node .* (1 .- read_pool.nominal_pi_node))
    n_over = count(abs.(read_pool.realized_rho) .> read_pool.ceiling .+ 1e-6)
    println("Rows above the tie-free ceiling (tie/discreteness, informational): ", n_over)
    println()

#	Bookkeeping round-trip: regenerate one row from its stored seed + params,
#	confirm the removed NODE set matches.
    repro_idx = clamp(nrow(read_pool) ÷ 3, 1, nrow(read_pool))
    row = read_pool[repro_idx, :]
    net = networks[row.network_name]

    c    = NDG._centrality_for_sampler(net.edges; nodes = net.nodes, directed = net.metadata.directed)
    adjb = NDG._graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = false)[1]
    comm = NDG._true_communities(adjb)

    rec = NDG.generate_missingness_mask(net.edges;
                nodes          = net.nodes,
                directed       = net.metadata.directed,
                weighted       = net.metadata.weighted,
                node_loss      = :targeted,
                target_pi_node = row.nominal_pi_node,
                target_pi_edge = 0.0,
                target_rho     = row.nominal_rho_used,
                seed           = row.seed,
                centrality     = c,
                true_community = comm,
                return_removed = true)

    println("Bookkeeping round-trip on row $repro_idx ($(row.network_name), pi_node=$(row.nominal_pi_node), rho_used=$(round(row.nominal_rho_used; digits=3))):")
    println("  removed-node set matches stored: ", collect(row.missing_nodes) == rec.missing_nodes)
    println("  realized_rho matches:            ", isapprox(row.realized_rho, rec.realized_rho; atol = 1e-9))
    println("  realized_pi_node matches:        ", isapprox(row.realized_pi_node, rec.realized_pi_node; atol = 1e-9))
    println("  organic losses (expect 0):       ", rec.n_organic_losses)
    println("  observed network:                nodes $(nrow(rec.observed_nodes))/$(nrow(net.nodes))  edges $(nrow(rec.observed_edges))/$(nrow(net.edges))")
    println()

#	Per-(network, level) reach + achieved-fraction table (the deliverable).
#	frac_neg / frac_pos = how much of the rate ceiling each wing actually reaches;
#	this is the column the design table was waiting on.
    println("Per-(network, pi_node) reach and achieved fraction of ceiling:")
    println("─" ^ 70)
    show(select(read_env, :network_name, :nominal_pi_node, :min_neg, :max_pos,
                :ceiling, :frac_neg, :frac_pos, :frac_gc_collapse, :n_draws),
         allrows = true, allcols = true)
    println()
