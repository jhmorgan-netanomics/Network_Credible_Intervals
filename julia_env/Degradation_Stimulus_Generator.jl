#Generate Degraded Networks Stimulus
#Jonathan H. Morgan
#5 June 2026

#=  ══════════════════════════════════════════════════════════════════════════
    UNIFIED DESIGN — EDGE vs NODE MISSINGNESS
 
    Two missingness mechanisms, the same 18 networks, the same missingness
    levels. rho — the Kendall tau-b of the missing-node indicator against the
    sampler centrality — is the realized truth of each degraded sample, and it
    is WITHHELD: recorded only to score, never handed to reconstruction (an
    analyst cannot measure it in practice — it lives in the data they did not
    collect). What reconstruction receives is an ASSUMED rho, and the experiment
    is built around setting that assumption right or wrong. The comparison is a
    CORRECT assumption against the cost of getting it wrong in each direction:
    assuming missingness is random when it is actually signed, assuming it is
    positive when it is not, and assuming it is negative when it is not. Assumed
    and true rho are the same binary-tau-b-vs-centrality quantity, so right and
    wrong line up cleanly — a +0.35 assumption is scored against a +0.35 truth.
 
    Conditions are POINT values at fixed rho — selected post-hoc on realized rho
    within +-TOL (the nearest achievable tau-b on small/discrete nets), not
    ranges — so each condition's single reconstruction guess faithfully stands
    in for every sample in it.
 
    ── MECHANISMS ─────────────────────────────────────────────────────────────
    EDGE  weight-proportional removal (Belluta 2026); node loss EMERGENT (a node is
          lost only when all its incident weight reaches zero). Conservative:
          the cost to orphan a node = its incident weight, so hubs survive and
          the missing set skews to the periphery. Reaches NEGATIVE rho only in most cases.
          Encodes the prior "the frame is right; weights are under-reported."
            node_loss = :emergent ; sweep target_pi_edges ; target_pi_nodes=[0.10]
            (calibration tilt only) ; negative-dense target_rhos.
 
    NODE  targeted non-response WITH nomination recovery; frame RETAINED — a
          non-respondent's own ties go unobserved, alters' nominations of it
          survive, and a node leaves the frame only if no one names it. This is
          node-CLUSTERED edge missingness, the structural complement to EDGE's
          diffuse erosion. Targeting bypasses the cost structure, so it reaches
          BOTH signs on every topology. Encodes "actors may be missing for
          reasons the design could not control."
            node_loss = :targeted ; target_pi_edges=[0.0] ; sweep target_pi_nodes
            ; symmetric target_rhos.
          (NB: non-response, NOT coverage/latent-node deletion — reconstruction
           recovers a known non-respondent's ties, it does not infer unseen
           actors. Nomination recovery also softens the consequence gap.)
 
    ── SHARED AXES ────────────────────────────────────────────────────────────
    Missingness %   1, 10, 20, 30, 40, 50     EDGE = % of total weight
                                              NODE = % of nodes (exact)
    Tolerance       realized rho within +-0.03 of target ; snap to nearest
                    achievable tau-b on small/discrete nets
    Selection       keep-all pool -> select within +-TOL of each target rho,
                    seeded, N per (net, %, rho) cell ; N set by the fill report
 
    rho grid  EDGE  -0.35  -0.15   0                            (negative-only)
    rho grid  NODE  -0.35  -0.15   0   +0.15  +0.35             (symmetric)
 
    ── NETWORK 2x2  (weight heterogeneity x degree heterogeneity) ─────────────
                         Binary (unweighted)          Weighted
    Hub-Periphery  HP    marvel  balikatan  toledo    marvel  balikatan  toledo
                         pa_undir  pa_dir              pa_undir  pa_dir
    Homogeneous    HO    scotland  sbm_undir           scotland  sbm_undir
                         sbm_dir  moreno              sbm_dir  moreno
    full names: <stem>_{unweighted|weighted}; pa = synthetic_2_pa_{un,}directed,
    sbm = synthetic_1_sbm_{un,}directed, moreno = moreno_highschool,
    scotland = scotland_interlock, balikatan = balikatan_2022, toledo =
    toledo_crime, marvel = marvel_universe.
    scotland: reclassified HP -> HO. The NODE arm shows symmetric reach
    (frac_neg ~ frac_pos ~ 0.97-1.00 at 10%), not the hub-periphery positive-
    lean PA and toledo show. Confirm with degree_cv.
 
    ── EDGE ARM reachable rho  (realized, 200-rep production; reach over ALL %) ─
      -0.35,-0.15,0   balikatan toledo pa(x4) scotland sbm_undir moreno  (14)
      -0.15,0 only    marvel(x2)  sbm_directed_unweighted                ( 3)
      0 only          sbm_directed_weighted  (sheds ~1% of nodes)        ( 1)
      positive        ~0 on every net (the cost structure cannot orphan hubs)
    NOTE: firm reaches, but COLLAPSED over pi_edge. The edge arm's rho is coupled
    to pi_edge through the rate ceiling, so a net's reach to -0.79 does NOT mean
    it reaches -0.35 at 10% — strong rho lives at the higher %. The (net x
    pi_edge) breakdown is a groupby on the existing edge pool (the node arm's
    per-level combine, keyed on nominal_pi_edge), giving the same per-level table
    the NODE arm has; it also surfaces in the fill report when conditions are
    selected.
 
    ── NODE ARM reachable rho  (per pi_node; CONFIRMED from node envelope) ─────
      pi_node >= 0.20   full {-0.35,-0.15,0,+0.15,+0.35} on ALL 18 nets
      pi_node  = 0.10   +0.35 on all ; -0.35 on all EXCEPT pa(x4) & toledo(x2),
                        which reach only ~-0.30/-0.32 -> snap to -0.30 or drop
                        the -0.35 negative cell at 10%.
      The 10% gap is the MIRROR of the EDGE asymmetry: heavy-tailed nets reach
      POSITIVE first (rank-distinct hubs) and negative last (tied periphery) —
      the opposite of EDGE, which reaches only negative on the same graphs.
      headroom: NODE reaches +-0.7+ at high % (achieved fraction ~1.0, often >1
      via centrality-tie inflation) -> an OPTIONAL node-only EXTREME condition
      (+-0.6 / +-0.7) can stress the adversarial corner EDGE cannot reach.
 
    ── GIANT-COMPONENT COLLAPSE  (frac of NODE-arm draws, by pi_node) ──────────
      HO dense  sbm_undir  sbm_dir  moreno          0.00 at every level
      HP sparse (collapsed frac @ .1 / .2 / .3 / .4 / .5):
        toledo                 .04  .15  .24  .32  .40
        pa(x4) / scotland      .00  .02  .14  .20  .28
        balikatan / marvel     .00  .00  .13  .16  .23
    -> at high node-% a real fraction of sparse-HP draws shatter the giant
       component. Decide: exclude collapsed draws from the conditions, or record
       collapse as an outcome (it is itself a finding — node non-response
       fragments sparse graphs that dense ones absorb).
    ══════════════════════════════════════════════════════════════════════════ =#

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

#########################
#   CONFIGURATION       #
#########################

#   Output
	output_dir     = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Degraded_Networks"
	isdir(output_dir) || mkpath(output_dir)
	edge_pool_file = joinpath(output_dir, "edge_pool.arrow")
	node_pool_file = joinpath(output_dir, "node_pool.arrow")
	stimulus_file  = joinpath(output_dir, "degradation_stimulus.arrow")
	report_file    = joinpath(output_dir, "degradation_fill_report.arrow")

#   rho condition grids — the WITHHELD truth values the analyst's assumption is scored against
	EDGE_RHOS = [-0.35, -0.15, 0.0]                       # negative-only
	NODE_RHOS = [-0.35, -0.15, 0.0, 0.15, 0.35]           # symmetric

#   Point-condition selection
	TOL               = 0.03          # keep draws with |realized - target| <= TOL
	N_PER_CELL        = 500           # target per (net, level, rho); fill-limited
	EXCLUDE_COLLAPSED = true          # drop draws whose giant component collapsed
	SELECT_SEED       = 20260605

#   Pool generation (reproduces the envelope runs: same sweeps, reps, seed)
	POOL_REPS    = 200
	MASTER_SEED  = 42
	EDGE_LEVELS  = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]        # pi_edge (% of weight); 0.01 = minimal degradation
	NODE_LEVELS  = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]        # pi_node (% of nodes); 1% = round(0.01N), needs N>=50
	EDGE_NOMINAL = [-0.30, -0.20, -0.12, -0.07, -0.03, 0.0, 0.03, 0.07, 0.12, 0.20, 0.30]
	NODE_NOMINAL = [-0.70, -0.55, -0.45, -0.35, -0.25, -0.15, -0.07, 0.0,
	                 0.07,  0.15,  0.25,  0.35,  0.45,  0.55,  0.70]

#########################
#   FUNCTIONS           #
#########################

#   select_point_conditions
#	Land the POINT conditions from a keep-all pool, per (network, level, target).
#	SIGNED targets: take up to n_per_cell draws whose realized rho is within +-tol
#	of target (achieved_rho records where the quantized tau-b landed). RANDOM
#	anchor (target == 0) is selected by the GENERATING process (nominal_rho == 0 =
#	uniform dropping), NOT realized rho: on the edge arm random dropping realizes a
#	weakly-negative emergent node-rho, so a realized window would discard most of
#	it; realized_rho is recorded as the outcome. The anchor TAKES ALL its nominal-0
#	draws (its ceiling is the rep depth, not n_per_cell), so it is never flagged
#	UNDER unless anchor_n_per_cell is set explicitly. Optionally drops collapsed-
#	giant-component draws. Returns (stimulus, report). Calls are module-qualified
#	so a stale DataFrame in Main cannot shadow them.
	function select_point_conditions(pool::DataFrames.DataFrame;
	                                 arm::AbstractString,
	                                 level_col::Symbol,
	                                 rho_targets::AbstractVector{<:Real},
	                                 tol::Real                       = 0.03,
	                                 n_per_cell::Int                 = 500,
	                                 anchor_n_per_cell::Union{Nothing,Int} = nothing,
	                                 exclude_collapsed::Bool         = true,
	                                 master_seed::Integer            = 20260605)
		rng      = Random.MersenneTwister(master_seed)
		pool_use = exclude_collapsed ? pool[.!pool.gc_collapse, :] : pool
		selected = DataFrames.DataFrame[]
		rows     = Base.NamedTuple[]
		for gnet in DataFrames.groupby(pool_use, :network_name)
			netname = gnet.network_name[1]
			for lvl in Base.sort(Base.unique(gnet[!, level_col]))
				cell = gnet[gnet[!, level_col] .== lvl, :]
				for target in rho_targets
					#	Random/ignorable anchor (target == 0) is defined by the
					#	GENERATING process (nominal_rho == 0 = uniform dropping),
					#	NOT by realized rho: on the edge arm random dropping
					#	realizes a weakly-negative EMERGENT node-rho, so a
					#	realized-window filter would silently discard most of it.
					#	realized_rho is recorded as the outcome it is. Signed
					#	targets stay realized-filtered (chased by value).
					is_random = Base.abs(target) < 1e-9
					cand = is_random ?
					       cell[Base.abs.(cell.nominal_rho) .< 1e-9, :] :
					       cell[Base.abs.(cell.realized_rho .- target) .<= tol, :]
					n_avail = DataFrames.nrow(cand)

					#	Per-target cap. The anchor is generating-process-defined,
					#	so it TAKES ALL its nominal-0 draws (ceiling = rep depth,
					#	not n_per_cell) and is never spuriously flagged UNDER.
					#	Signed targets are value-chased and capped at n_per_cell.
					#	anchor_n_per_cell === nothing => take-all; pass an Int to
					#	cap the anchor explicitly (a short anchor then reads UNDER).
					cap  = is_random ?
					       (anchor_n_per_cell === nothing ? n_avail : anchor_n_per_cell) :
					       n_per_cell
					take = Base.min(n_avail, cap)
					ach  = NaN
					if take > 0
						idx = take == n_avail ? Base.collect(1:n_avail) :
						      Base.sort(StatsBase.sample(rng, 1:n_avail, take; replace = false))
						sel = cand[idx, :]
						sel[!, :arm]        = Base.fill(Base.String(arm),     take)
						sel[!, :level]      = Base.fill(Base.Float64(lvl),    take)
						sel[!, :target_rho] = Base.fill(Base.Float64(target), take)
						Base.push!(selected, sel)
						ach = Statistics.mean(sel.realized_rho)
					end

					#	Fill status. EMPTY: nothing landed in the window. UNDER:
					#	wanted the cap but came up short (anchor only when an
					#	explicit anchor_n_per_cell was set). ok: target met
					#	(signed), or all anchor draws taken.
					fill_status = take == 0 ? :EMPTY :
					              is_random ?
					                  (anchor_n_per_cell === nothing || take >= anchor_n_per_cell ? :ok : :UNDER) :
					                  (take < n_per_cell ? :UNDER : :ok)
					Base.push!(rows, (arm = Base.String(arm), network_name = netname,
					             level = Base.Float64(lvl), target_rho = Base.Float64(target),
					             n_available = n_avail, n_selected = take, achieved_rho = ach,
					             status = fill_status))
				end
			end
		end
		stimulus = Base.isempty(selected) ? DataFrames.DataFrame() : Base.vcat(selected...; cols = :union)
		return stimulus, DataFrames.DataFrame(rows)
	end

#########################
#   POOLS               #
#########################
#	Generate-then-select: build the keep-all pool per arm (or load if already
#	written), then select point conditions from it. To skip a rebuild, write the
#	in-session envelope pool to *_pool.arrow once; this block will load it.

#   EDGE pool  (node_loss = :emergent; node loss EMERGES from weight removal)
#	Self-healing against EDGE_LEVELS (same pattern as the node pool): append any
#	configured level the pool lacks, drop any it carries that is no longer
#	configured. On a re-run this swaps the old 0% baseline for 1% with no manual
#	rebuild — the prune drops the stale 0.0 block, the append adds 0.01, and the
#	0.1-0.5 draws are retained untouched (per-cell seeds key on pi_edge).
	if !isfile(edge_pool_file)
		println("Building edge pool ...")
		@time edge_pool = build_degeneration_corpus(networks;
		                        target_rhos     = EDGE_NOMINAL,
		                        target_pi_nodes = [0.10],            # calibration tilt only
		                        target_pi_edges = EDGE_LEVELS,
		                        node_loss       = :emergent,
		                        n_replicates    = POOL_REPS,
		                        master_seed     = MASTER_SEED,
		                        parallel        = true,
		                        show_progress   = true)
		edge_pool.nominal_rho_used = edge_pool.substituted_rho
		Arrow.write(edge_pool_file, edge_pool; compress = :zstd)
	else
		println("Loading edge pool: ", edge_pool_file)
		edge_pool = DataFrame(Arrow.Table(edge_pool_file))
		want    = round.(Float64.(EDGE_LEVELS); digits = 6)
		have    = Set(round.(edge_pool.nominal_pi_edge; digits = 6))
		add_lvl = [l for l in want if !(l in have)]
		dirty   = false
		if !isempty(add_lvl)
			println("  edge pool missing levels ", add_lvl, " — building and appending ...")
			@time edge_add = build_degeneration_corpus(networks;
			                        target_rhos     = EDGE_NOMINAL,
			                        target_pi_nodes = [0.10],            # calibration tilt only
			                        target_pi_edges = add_lvl,
			                        node_loss       = :emergent,
			                        n_replicates    = POOL_REPS,
			                        master_seed     = MASTER_SEED,
			                        parallel        = true,
			                        show_progress   = true)
			edge_add.nominal_rho_used = edge_add.substituted_rho
			edge_pool = vcat(edge_pool, edge_add; cols = :union)
			dirty = true
		end
		keep = [round(x; digits = 6) in Set(want) for x in edge_pool.nominal_pi_edge]
		if !all(keep)
			println("  edge pool dropping ", count(.!keep), " rows at unconfigured levels ...")
			edge_pool = edge_pool[keep, :]
			dirty = true
		end
		dirty && Arrow.write(edge_pool_file, edge_pool; compress = :zstd)
	end

#   NODE pool  (node_loss = :targeted, pi_edge = 0; pure node non-response)
#	Self-healing against NODE_LEVELS: build the full pool if absent; otherwise
#	load it, append any configured levels it lacks, and drop any it carries that
#	are no longer configured — so a pool on disk can never silently mask a
#	NODE_LEVELS change. Per-cell seeds key on pi_node, so appended levels are
#	independently seeded and retained levels are untouched.
	if !isfile(node_pool_file)
		println("Building node pool ...")
		@time node_pool = build_degeneration_corpus(networks;
		                        target_rhos     = NODE_NOMINAL,
		                        target_pi_nodes = NODE_LEVELS,
		                        target_pi_edges = [0.0],
		                        node_loss       = :targeted,
		                        n_replicates    = POOL_REPS,
		                        master_seed     = MASTER_SEED,
		                        parallel        = true,
		                        show_progress   = true)
		node_pool.nominal_rho_used = node_pool.substituted_rho
		Arrow.write(node_pool_file, node_pool; compress = :zstd)
	else
		println("Loading node pool: ", node_pool_file)
		node_pool = DataFrame(Arrow.Table(node_pool_file))
		want    = round.(Float64.(NODE_LEVELS); digits = 6)
		have    = Set(round.(node_pool.nominal_pi_node; digits = 6))
		add_lvl = [l for l in want if !(l in have)]
		dirty   = false
		if !isempty(add_lvl)
			println("  node pool missing levels ", add_lvl, " — building and appending ...")
			@time node_add = build_degeneration_corpus(networks;
			                        target_rhos     = NODE_NOMINAL,
			                        target_pi_nodes = add_lvl,
			                        target_pi_edges = [0.0],
			                        node_loss       = :targeted,
			                        n_replicates    = POOL_REPS,
			                        master_seed     = MASTER_SEED,
			                        parallel        = true,
			                        show_progress   = true)
			node_add.nominal_rho_used = node_add.substituted_rho
			node_pool = vcat(node_pool, node_add; cols = :union)
			dirty = true
		end
		keep = [round(x; digits = 6) in Set(want) for x in node_pool.nominal_pi_node]
		if !all(keep)
			println("  node pool dropping ", count(.!keep), " rows at unconfigured levels ...")
			node_pool = node_pool[keep, :]
			dirty = true
		end
		dirty && Arrow.write(node_pool_file, node_pool; compress = :zstd)
	end

#	When the file is run in chunks (VS Code / REPL) or after a restart, the
#	POOLS block above may not have populated Main in this session. Reload from
#	disk if needed. Prerequisite: the CONFIGURATION block has run (it defines
#	edge_pool_file / node_pool_file).
	if !@isdefined(edge_pool)
		isfile(edge_pool_file) ||
			error("edge_pool is undefined and $edge_pool_file is missing — run the POOLS block first")
		println("edge_pool not in session; reloading: ", edge_pool_file)
		edge_pool = DataFrame(Arrow.Table(edge_pool_file))
		("nominal_rho_used" in names(edge_pool)) ||
			(edge_pool.nominal_rho_used = edge_pool.substituted_rho)
	end
	if !@isdefined(node_pool)
		isfile(node_pool_file) ||
			error("node_pool is undefined and $node_pool_file is missing — run the POOLS block first")
		println("node_pool not in session; reloading: ", node_pool_file)
		node_pool = DataFrame(Arrow.Table(node_pool_file))
		("nominal_rho_used" in names(node_pool)) ||
			(node_pool.nominal_rho_used = node_pool.substituted_rho)
	end

#	Heal edge pool: build 0.01, drop the stale 0.0 baseline, rewrite
	want    = round.(Float64.(EDGE_LEVELS); digits = 6)
	have    = Set(round.(edge_pool.nominal_pi_edge; digits = 6))
	add_lvl = [l for l in want if !(l in have)]
	if !isempty(add_lvl)
		println("edge pool building missing levels ", add_lvl, " ...")
		edge_add = build_degeneration_corpus(networks;
		                target_rhos     = EDGE_NOMINAL,
		                target_pi_nodes = [0.10],
		                target_pi_edges = add_lvl,
		                node_loss       = :emergent,
		                n_replicates    = POOL_REPS,
		                master_seed     = MASTER_SEED,
		                parallel        = true,
		                show_progress   = true)
		edge_add.nominal_rho_used = edge_add.substituted_rho
		edge_pool = vcat(edge_pool, edge_add; cols = :union)
	end
	keep = [round(x; digits = 6) in Set(want) for x in edge_pool.nominal_pi_edge]
	edge_pool = edge_pool[keep, :]
	Arrow.write(edge_pool_file, edge_pool; compress = :zstd)
	sort(unique(edge_pool.nominal_pi_edge))

#	Regeneration key (present after a build; add it if loading an older pool).
	("nominal_rho_used" in names(edge_pool)) || (edge_pool.nominal_rho_used = edge_pool.substituted_rho)
	("nominal_rho_used" in names(node_pool)) || (node_pool.nominal_rho_used = node_pool.substituted_rho)

#########################
#   SELECT STIMULUS     #
#########################

#   Examining Stimulus
	println("Selecting edge-arm point conditions ...")
	edge_stim, edge_report = select_point_conditions(edge_pool;
	                            arm = "edge", level_col = :nominal_pi_edge,
	                            rho_targets = EDGE_RHOS, tol = TOL, n_per_cell = N_PER_CELL,
	                            exclude_collapsed = EXCLUDE_COLLAPSED, master_seed = SELECT_SEED)

	println("Selecting node-arm point conditions ...")
	node_stim, node_report = select_point_conditions(node_pool;
	                            arm = "node", level_col = :nominal_pi_node,
	                            rho_targets = NODE_RHOS, tol = TOL, n_per_cell = N_PER_CELL,
	                            exclude_collapsed = EXCLUDE_COLLAPSED, master_seed = SELECT_SEED)

	stimulus    = vcat(edge_stim, node_stim; cols = :union)
	fill_report = vcat(edge_report, node_report)

	Arrow.write(stimulus_file, stimulus;    compress = :zstd)
	Arrow.write(report_file,   fill_report; compress = :zstd)
	println("Wrote stimulus: ", stimulus_file, "  rows=", nrow(stimulus))
	println("Wrote report:   ", report_file)
	println("Done: ", now())

#   Fill summary
	println("\nFill summary (cells per arm by status):")
	show(combine(groupby(fill_report, [:arm, :status]), nrow => :cells), allrows = true); println()
	flagged = fill_report[fill_report.status .!= :ok, :]
	println("\nUnder-filled / empty cells (expected: 1% level signed cells EMPTY ",
	        "[rate ceiling |tau| <= 2p(1-p)]; edge low-% strong rho UNDER/EMPTY; ",
	        "node 0.10 PA/toledo -0.35 UNDER/EMPTY):")
	show(sort(flagged, [:arm, :network_name, :level, :target_rho]), allrows = true, allcols = true)

#	Edge random-anchor cells (target_rho = 0) across levels and nets
	sort(fill_report[(fill_report.arm .== "edge") .& (abs.(fill_report.target_rho) .< 1e-9),
	                 [:network_name, :level, :n_available, :n_selected, :achieved_rho, :status]],
	     [:network_name, :level])

#	Edge stimulus row counts by level x target_rho
	sort(combine(groupby(stimulus[stimulus.arm .== "edge", :], [:level, :target_rho]), nrow => :n),
	     [:level, :target_rho])

####################
#   VERIFICATION   #
####################
#	Test-style integrity checks: confirm the selection landed on-target, the arm
#	wiring held, nothing collapsed slipped through, and a stored row regenerates
#	bit-for-bit. Nothing here mutates the stimulus; it only reads it back.
	stim = DataFrame(Arrow.Table(stimulus_file))
	println("\n=== VERIFICATION ===")
	println("Rows: ", nrow(stim), "   arms: ", sort(unique(stim.arm)))

#   [1] Coverage — arms, levels, rho conditions, networks present
	println("\n[1] Coverage")
	for a in sort(unique(stim.arm))
		s = stim[stim.arm .== a, :]
		println("  $(rpad(a,5)) levels=", sort(unique(s.level)),
		        "  rho=", sort(unique(s.target_rho)),
		        "  nets=", length(unique(s.network_name)))
	end

#   [2] Within-tolerance — every SIGNED-target draw within TOL (random anchor exempt)
	n_oot = count(@. (abs(stim.target_rho) >= 1e-9) & (abs(stim.realized_rho - stim.target_rho) > TOL + 1e-9))
	println("\n[2] Within-tolerance violations on signed targets (expect 0): ", n_oot)

#   [3] Collapsed draws excluded
	println("[3] Collapsed draws in stimulus (expect 0 when EXCLUDE_COLLAPSED): ", count(stim.gc_collapse))

#   [4] Arm invariants
	e = stim[stim.arm .== "edge", :]
	n = stim[stim.arm .== "node", :]
	println("\n[4] Arm invariants")
	if !isempty(e)
		ed = e[e.level .> 0.0, :]
		println("  edge: any positive target (expect false):      ", any(e.target_rho .> 0))
		println("  edge: realized_pi_edge on degraded rows in:    [",
		        round(minimum(ed.realized_pi_edge); digits=3), ", ", round(maximum(ed.realized_pi_edge); digits=3), "]")
	end
	if !isempty(n)
		println("  node: max n_organic_losses (expect 0):         ", maximum(n.n_organic_losses))
		println("  node: max n_edges_zeroed   (expect 0):         ", maximum(n.n_edges_zeroed))
		println("  node: max |realized_pi_node - nominal|:        ",
		        round(maximum(abs.(n.realized_pi_node .- n.nominal_pi_node)); digits=4))
	end

#   [5] Achieved vs target — tolerance only on SIGNED targets; the random anchor
#	    (target 0) is nominal-selected, so its realized node-rho is an outcome,
#	    reported below as a level-by-level slide, not a tolerance miss.
	ff     = fill_report[fill_report.n_selected .> 0, :]
	signed = ff[abs.(ff.target_rho) .>= 1e-9, :]
	println("\n[5] |achieved - target| over signed filled cells — mean ",
	        round(Statistics.mean(abs.(signed.achieved_rho .- signed.target_rho)); digits=4),
	        "  max ", round(maximum(abs.(signed.achieved_rho .- signed.target_rho)); digits=4), "  (<= TOL)")
	rand_ff = ff[abs.(ff.target_rho) .< 1e-9, :]
	if !isempty(rand_ff)
		println("    random anchor realized node-rho by level: ",
		        sort(combine(groupby(rand_ff, :level), :achieved_rho => Statistics.mean => :mean_realized), :level))
	end

#   [6] Bookkeeping round-trip — regenerate one row per arm from its stored seed+params
	println("\n[6] Round-trip regeneration (regenerate from seed + nominal params)")
	for a in sort(unique(stim.arm))
		sub  = stim[stim.arm .== a, :]
		row  = sub[nrow(sub) ÷ 2 + 1, :]
		net  = networks[row.network_name]
		c    = NDG._centrality_for_sampler(net.edges; nodes = net.nodes, directed = net.metadata.directed)
		comm = NDG._true_communities(NDG._graph_to_sparse_matrix(net.edges; nodes = net.nodes, weighted = false)[1])
		rec  = NDG.generate_missingness_mask(net.edges;
		            nodes          = net.nodes,
		            directed       = net.metadata.directed,
		            weighted       = net.metadata.weighted,
		            node_loss      = a == "node" ? :targeted : :emergent,
		            target_pi_node = row.nominal_pi_node,
		            target_pi_edge = row.nominal_pi_edge,
		            target_rho     = row.nominal_rho_used,
		            seed           = row.seed,
		            centrality     = c,
		            true_community = comm,
		            return_removed = true)
		println("  [$(rpad(a,4))] $(rpad(row.network_name,34)) level=$(row.level) target=$(row.target_rho):",
		        "  nodes match=", collect(row.missing_nodes) == rec.missing_nodes,
		        "  rho match=",  isapprox(row.realized_rho, rec.realized_rho; atol = 1e-9))
	end
	println("\n=== END VERIFICATION ===")

    rand_ff = ff[abs.(ff.target_rho) .< 1e-9, :]
	if !isempty(rand_ff)
		println("    random anchor realized node-rho by arm x level:")
		show(sort(combine(groupby(rand_ff, [:arm, :level]),
		                  :achieved_rho => Statistics.mean => :mean_realized),
		          [:arm, :level]), allrows = true)
		println()
	end

    node_anchor = stimulus[(stimulus.arm .== "node") .& (abs.(stimulus.target_rho) .< 1e-9), :]
	combine(groupby(node_anchor, :level),
	        :realized_rho => Statistics.mean => :mean,
	        :realized_rho => Statistics.std  => :std,
	        :realized_rho => minimum         => :min,
	        :realized_rho => maximum         => :max)