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
comparable along a single x-axis. Centrality-biased missingness is controlled by rho, the rank correlation (Kendall tau) between a node's centrality and the 
missingness it experiences. rho tilts which edges zero, and therefore which nodes are most likely to be orphaned. It is varied as the true generating condition 
across a small achievable band — approximately {−0.15, 0, +0.15} — rather than the ±0.75 of Smith et al., because rho here is measured on the materialized network 
and is structurally bounded; the wider range was attainable in that paper only because it imposed a selection-probability correlation without sampling a realized 
network. rho interacts with the level: negative rho concentrates removal on low-degree peripheral nodes, which orphan quickly and yield more node loss at a given 
level, while positive rho lands removal on high-degree nodes with ties to spare and yields less. Induced node loss is thus a joint function of network, level, 
and rho — not of the level alone.

Degradation runs from edges to nodes: zero edges, and observe which nodes drop. Reconstruction runs the other way: assume some number of nodes are missing, then 
distribute weight to recover them and their ties. The bridge is the recorded emergent node-loss count (n_organic_losses, equivalently the recorded missing-node set), 
which is precisely the input the reconstruction consumes. Because node loss can in principle be zero — a low level on a heavy or dense network may leave every node 
with a surviving tie, retaining the full roster, though this is uncommon among the empirical networks — the count is recorded per cell rather than derived from the 
level, and a count of zero is a valid record, handled by reconstructing edge weight alone.

For this validation the analyst is granted ground-truth knowledge of two quantities: the edge-loss fraction and the induced node-loss count. rho is withheld; it is 
the unknown whose misspecification the reconstruction probes. Downstream of this stimulus, each recorded sample is reconstructed under each rho value, so that the 
gap between the matched line (assumed rho equal to the true generating rho) and the two mismatched lines quantifies the cost of getting the centrality mechanism wrong. The stimulus itself varies only the true conditions — network, edge-missingness level, and true rho — and records the realized quantities the reconstruction will need.

One extension is noted but not implemented here. In practice, analysts typically estimate the proportion of nodes lost more reliably than the proportion of ties. 
A natural variant would invert which quantity is treated as known: grant an accurate node proportion while misspecifying or omitting the tie percentage, and measure 
the resulting degradation. The current design fixes both as known and probes only rho.
```

#	Generates the degeneration corpus across all sixteen networks,
#	five centrality–missingness correlation levels, six missingness
#	rates, 100 replicates per cell, and both materialization mechanisms.

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

###################
#   IMPORT DATA   #
###################

#   Load Corpus
#	Mirrors the loader in Network_Degradation_Tests.jl: read every
#	GraphML in the test-networks directory into the canonical
#	(edges, nodes, metadata) NamedTuple format expected by the
#	orchestrator.
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
    for (name, net) in sort(collect(networks), by=first)
        println("  $(rpad(name, 40)) N=$(rpad(nrow(net.nodes), 8)) E=$(rpad(nrow(net.edges), 10)) directed=$(net.metadata.directed)")
    end
    println()

###################################
#   GENERATE DEGENERATE SAMPLES   #
###################################

#   Configuration
    master_seed    = 42
    output_dir     = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degenerate_Networks"
    output_file    = joinpath(output_dir, "degeneration_corpus.arrow")

    target_rhos    = [-0.75, -0.25, 0.0, 0.25, 0.75]
    target_rates   = [0.05, 0.10, 0.15, 0.25, 0.40, 0.50]
    n_replicates   = 100
    mechanisms_cfg = [:full_removal, :outgoing_only]

#   Pre-Flight
    println("─" ^ 70)
    println("Phase 1 Production Corpus Generation")
    println("─" ^ 70)
    println("Started:           ", now())
    println("Julia version:     ", VERSION)
    println("Threads:           ", Threads.nthreads())
    println("Master seed:       ", master_seed)
    println("Output file:       ", output_file)
    println()

    mkpath(output_dir)
    Threads.nthreads() >= 4 || @warn "Only $(Threads.nthreads()) threads available; orchestrator was tuned for >= 4"

#   Run Production Grid
    println("Launching build_degeneration_corpus (production configuration) ...")
    println()

    @time corpus_df = build_degeneration_corpus(networks;
                                                target_rhos    = target_rhos,
                                                target_rates   = target_rates,
                                                n_replicates   = n_replicates,
                                                mechanisms     = mechanisms_cfg,
                                                master_seed    = master_seed,
                                                parallel       = true,
                                                show_progress  = true)

#	Diagnostic Readout
    println()
    println("─" ^ 70)
    println("Production corpus complete")
    println("─" ^ 70)
    println("Rows:              ", nrow(corpus_df))
    println("Networks:          ", length(unique(corpus_df.network_name)))
    println()

    println("Per-network breakdown")
    println("─" ^ 70)
    for grp in groupby(corpus_df, :network_name)
        n_rows  = nrow(grp)
        n_full  = count(==(:full_removal), grp.mechanism)
        n_out   = count(==(:outgoing_only), grp.mechanism)
        n_conv  = count(==(:converged), grp.bisection_status)
        n_ceil  = count(==(:ceiling_hit), grp.bisection_status)
        n_fail  = count(==(:failed_other), grp.bisection_status)
        n_deg   = count(grp.any_topo_degenerate)
        println("  $(rpad(grp.network_name[1], 40)) rows=$(rpad(n_rows, 6)) full=$(rpad(n_full, 5)) out=$(rpad(n_out, 5)) conv=$(rpad(n_conv, 5)) ceil=$(rpad(n_ceil, 3)) fail=$(rpad(n_fail, 3)) deg=$n_deg")
    end
    println()

#   Write-Out Arrow Files
    println("Writing Arrow file ...")
    Arrow.write(output_file, corpus_df; compress = :zstd)
    println("Wrote: ", output_file, " (", round(filesize(output_file) / 1024^2, digits=1), " MB)")
    println()

    println("Done: ", now())

#   Checking File
    production_file = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degenerate_Networks/degeneration_corpus.arrow"

    read_df = DataFrame(Arrow.Table(production_file))
    println("Rows:              ", nrow(read_df))
    println("Columns:           ", names(read_df))
    println("Networks:          ", length(unique(read_df.network_name)))
    println("Mechanisms:        ", unique(read_df.mechanism))
    println("ρ values:          ", sort(unique(read_df.nominal_rho)))
    println("Rate values:       ", sort(unique(read_df.nominal_rate)))
    println("dropped_nodes type:", typeof(read_df.dropped_nodes[1]))
    println("Example row:")
    println(read_df[1, :])

#	Pick 5 representative rows: small network, large network, both
#	mechanisms, both ρ signs.
    sample_indices = [1, 1000, 30000, 50000, 78000]   # arbitrary spread
    for idx in sample_indices
        row = read_df[idx, :]
        net = networks[row.network_name]
        dropped = collect(row.dropped_nodes)    # convert Arrow vector to Vector{Int}
        
        if row.mechanism == :full_removal
            result = Network_Credible_Intervals.network_degeneracy.apply_missingness(
                        net.edges, dropped; nodes=net.nodes)
        else
            result = Network_Credible_Intervals.network_degeneracy.apply_missingness_outgoing_only(
                        net.edges, dropped; nodes=net.nodes, directed=true)
        end
        
        #	Validate
            n_orig         = nrow(net.nodes)
            n_dropped      = length(dropped)
            expected_nodes = row.mechanism == :full_removal ? n_orig - n_dropped : n_orig
            actual_nodes   = nrow(result.nodes)
            nodes_match    = actual_nodes == expected_nodes
            rate_match     = abs(row.realized_rate - n_dropped / n_orig) < 1e-10
        
        println("Row $idx: $(row.network_name) mech=$(row.mechanism) ρ_nom=$(row.nominal_rho) rate_nom=$(row.nominal_rate)")
        println("  k = $n_dropped, realized_rate = $(row.realized_rate), rate_match = $rate_match")
        println("  nodes: $actual_nodes / $expected_nodes ($(nodes_match ? "OK" : "FAIL"))")
        println("  edges: $(nrow(result.edges)) (n_edges_observed in record: $(row.n_edges_observed))")
        println("  bisection_status: $(row.bisection_status), realized_ρ: $(row.realized_rho)")
    end

#	Pick one row, regenerate it from scratch using the recorded seed,
#	confirm bit-identical dropped_nodes.
    row = read_df[5000, :]    # arbitrary directed network for variety
    net = networks[row.network_name]

    c = Network_Credible_Intervals.network_degeneracy._centrality_for_sampler(
            net.edges; nodes=net.nodes, directed=net.metadata.directed)

    rec = Network_Credible_Intervals.network_degeneracy.generate_missingness_mask(
                net.edges; nodes=net.nodes, directed=net.metadata.directed,
                target_rate=row.nominal_rate, target_rho=row.nominal_rho,
                seed=row.seed, centrality=c)

    println("Reproducibility check on row 5000:")
    println("  network:           $(row.network_name)")
    println("  nominal ρ, rate:   $(row.nominal_rho), $(row.nominal_rate)")
    println("  seed:              $(row.seed)")
    println("  dropped match:     $(collect(row.dropped_nodes) == rec.dropped_nodes)")
    println("  realized_ρ match:  $(isapprox(row.realized_rho, rec.realized_rho))")
    println("  status match:      $(row.bisection_status == rec.bisection_status)")

#	Group by network, ρ sign, rate, mechanism — report mean realized ρ
#	at :ceiling_hit cells. We expect: positive ρ ceilings around 0.3-0.5,
#	negative ρ ceilings often near zero on heavy-tailed networks.
    ceiling_rows = filter(:bisection_status => ==(:ceiling_hit), read_df)
    println("Total ceiling-hit rows: ", nrow(ceiling_rows))
    println()

    for grp in groupby(ceiling_rows, [:network_name, :nominal_rho])
        mean_realized = round(mean(grp.realized_rho), digits=3)
        median_realized = round(median(grp.realized_rho), digits=3)
        n_rows = nrow(grp)
        println("  $(rpad(grp.network_name[1], 40)) ρ_nom=$(grp.nominal_rho[1])  n=$n_rows  mean=$mean_realized  median=$median_realized")
    end