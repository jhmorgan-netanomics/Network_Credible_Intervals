#Generate Degraded Networks
#Jonathan H. Morgan
#29 May 2026

#	Generates the degeneration corpus across all sixteen networks,
#	five centrality–missingness correlation levels, six missingness
#	rates, 100 replicates per cell, and both materialization mechanisms.
#
#	Expected runtime: ~4 hours on a single workstation with
#	JULIA_NUM_THREADS=8.

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

#	Diagnostic readout ───────────────────────────────────────────
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