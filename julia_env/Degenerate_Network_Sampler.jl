#Generate Degraded Networks
#Jonathan H. Morgan
#29 May 2026

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