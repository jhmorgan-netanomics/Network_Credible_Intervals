#Dataset Conversion/Generation File
#Jonathan H. Morgan
#26 May 2026

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

using CSV
using DataFrames
using Distributions
using Random
using StatsBase
using SparseArrays
using EzXML
using Network_Credible_Intervals

#################
#   FUNCTIONS   #
#################

#	Helper Function for parse_pajek_2mode: Strip Pajek Whitespace and Quotes
	function _strip_pajek_token(s::AbstractString)
		"""
		Args:
			s::AbstractString: a raw Pajek field
		Returns:
			String: stripped of leading/trailing whitespace and surrounding double quotes
		Notes:
			Pajek vertex labels are typically wrapped in double quotes and may include
			internal whitespace. This helper preserves internal whitespace and strips
			only the wrapping quotes and outer whitespace.
		"""

		#	Strip Whitespace
			t = strip(s)

		#	Strip Wrapping Quotes
			if length(t) ≥ 2 && first(t) == '"' && last(t) == '"'
				t = t[2:end-1]
			end

		#	Return
			return String(t)
	end

#	Helper Function for parse_pajek_2mode: Parse One Vertex Line
	function _parse_pajek_vertex_line(line::AbstractString)
		"""
		Args:
			line::AbstractString: a single vertex line from a Pajek file
		Returns:
			Tuple{Int, String}: (vertex_id, label)
		Notes:
			Format: leading-whitespace integer-id "quoted label with possible spaces" coords...
			Coordinates after the label are ignored.

			Extracts the integer id and the quoted label. If no quotes are present,
			falls back to taking the first whitespace-delimited token after the id
			as the label.
		"""

		#	Find First Token (Integer ID)
			s = strip(line)
			m_id = match(r"^(\d+)\s+(.*)$", s)
			if m_id === nothing
				throw(ArgumentError("Cannot parse vertex line: $line"))
			end
			vertex_id = parse(Int, m_id.captures[1])
			rest = m_id.captures[2]

		#	Extract Quoted Label If Present
			m_label = match(r"^\"([^\"]*)\"", rest)
			if m_label !== nothing
				label = String(m_label.captures[1])
			else
				#	Fallback: First Whitespace-Delimited Token
					m_fallback = match(r"^(\S+)", rest)
					label = m_fallback === nothing ? "" : String(m_fallback.captures[1])
			end

		#	Return
			return (vertex_id, label)
	end

#	Helper Function for parse_pajek_2mode: Parse One Edge Line
	function _parse_pajek_edge_line(line::AbstractString)
		"""
		Args:
			line::AbstractString: a single edge line from a Pajek file
		Returns:
			Tuple{Int, Int, Float64}: (source, target, weight)
		Notes:
			Format: leading-whitespace source-id target-id [weight]
			If weight is missing, assumes 1.0.
		"""

		#	Tokenize
			tokens = split(strip(line))
			if length(tokens) < 2
				throw(ArgumentError("Cannot parse edge line (too few tokens): $line"))
			end

		#	Source and Target
			src = parse(Int, tokens[1])
			dst = parse(Int, tokens[2])

		#	Optional Weight
			weight = length(tokens) ≥ 3 ? parse(Float64, tokens[3]) : 1.0

		#	Return
			return (src, dst, weight)
	end

#	Helper Function for parse_pajek_2mode: Find Section Start Line Index
	function _find_pajek_section(lines::Vector{String}, header::AbstractString)
		"""
		Args:
			lines::Vector{String}: all lines from the Pajek file
			header::AbstractString: section header prefix (e.g., "*Vertices", "*Edges")
		Returns:
			Union{Int, Nothing}: line index (1-based) of the header, or nothing if absent
		Notes:
			Case-insensitive match on the header prefix. Returns the index of the
			line containing the header, not the first content line.
		"""

		#	Lower-Case Header for Match
			h = lowercase(strip(header))

		#	Scan Lines
			for (i, ln) in pairs(lines)
				if startswith(lowercase(strip(ln)), h)
					return i
				end
			end

		#	Not Found
			return nothing
	end

#	Parse a 2-Mode Pajek File with Vertices, Edges, Affiliation Partition, Industry Partition, Capital Vector
	function parse_pajek_2mode(filepath::String;
	                           n_firms::Int = 108,
	                           n_total::Int = 244)
		"""
		Args:
			filepath::String: path to the Pajek (.paj or .net) file
			n_firms::Int: number of firms (first mode); for Scotland = 108 (default)
			n_total::Int: total vertex count; for Scotland = 244 (default)
		Returns:
			NamedTuple: (firms, directors, edges, industry, capital)
				firms::DataFrame   — columns :id, :label, :mode (always "firm")
				directors::DataFrame — columns :id, :label, :mode (always "director")
				edges::DataFrame   — columns :firm_id, :director_id, :weight
				industry::Vector{Int} — per-firm industry code (length n_firms)
				capital::Vector{Float64} — per-firm capital value (length n_firms)
		Notes:
			Reads a Pajek .paj or .net file containing a 2-mode (bipartite)
			network where the first n_firms vertices are firms and the remaining
			(n_total - n_firms) vertices are directors.

			Pajek files are commonly encoded in Latin-1 (ISO-8859-1) rather than
			UTF-8 because European name characters (ü, é, etc.) predate widespread
			UTF-8 adoption. The reader decodes the file byte-by-byte as Latin-1,
			which never fails since every byte 0x00..0xFF maps to a valid Unicode
			code point. The resulting strings are valid UTF-8 internally.

			Extracts five pieces of data:
			- Firms (vertices 1..n_firms) with their labels
			- Directors (vertices n_firms+1..n_total) with their labels
			- Bipartite edges from the *Edges section
			- Industry partition from the *Partition Industrial_categories.clu section
			- Capital values from the *Vector Capital.vec section

			The *Partition Affiliation section is read for consistency-checking but
			not returned (its values just encode firm vs director, which is
			already determined by the n_firms split).

			Assumes the standard Scotland.paj layout. Other 2-mode Pajek files may
			require parameter adjustments or modifications to this function.
		"""

		#	Validation
			if !isfile(filepath)
				throw(ArgumentError("File not found: $filepath"))
			end
			if n_firms < 1 || n_firms ≥ n_total
				throw(ArgumentError("n_firms ($n_firms) must satisfy 1 ≤ n_firms < n_total ($n_total)"))
			end

		#	Read All Lines (Decode as Latin-1 / ISO-8859-1)
			#	Pajek files often use Latin-1 encoding for European name characters
			#	(ü, ö, é, etc.) which would fail UTF-8 decoding. Reading as raw bytes
			#	and decoding each byte as its Latin-1 code point sidesteps the issue.
				raw_bytes = read(filepath)
				decoded   = String(Char.(raw_bytes))
				raw       = String.(split(decoded, r"\r?\n"))

		#	Locate Section Headers
			ix_vertices = _find_pajek_section(raw, "*Vertices")
			ix_edges    = _find_pajek_section(raw, "*Edges")
			if ix_vertices === nothing
				throw(ArgumentError("Pajek file missing *Vertices section"))
			end
			if ix_edges === nothing
				throw(ArgumentError("Pajek file missing *Edges section"))
			end

		#	Locate Optional Sections
			ix_industry = nothing
			ix_capital  = nothing
			for (i, ln) in pairs(raw)
				lc = lowercase(strip(ln))
				if startswith(lc, "*partition") && occursin("industrial_categories", lc)
					ix_industry = i
				elseif startswith(lc, "*vector") && occursin("capital", lc)
					ix_capital = i
				end
			end

		#	Parse Vertices
			firm_labels     = Vector{String}(undef, n_firms)
			director_labels = Vector{String}(undef, n_total - n_firms)
			for line_ix in (ix_vertices + 1):(ix_edges - 1)
				ln = strip(raw[line_ix])
				isempty(ln) && continue
				startswith(ln, "*") && break    # next section header
				vid, label = _parse_pajek_vertex_line(ln)
				if 1 ≤ vid ≤ n_firms
					firm_labels[vid] = label
				elseif n_firms < vid ≤ n_total
					director_labels[vid - n_firms] = label
				end
			end

		#	Parse Edges (Stop at Next Section)
			edge_srcs     = Int[]
			edge_dsts     = Int[]
			edge_weights  = Float64[]
			for line_ix in (ix_edges + 1):length(raw)
				ln = strip(raw[line_ix])
				isempty(ln) && continue
				if startswith(ln, "*")
					break                       # reached next section
				end
				#	Some Pajek files have an "*Arcs" header above "*Edges" (empty arcs).
				#	If the first edge line is malformed, skip it gracefully.
					try
						src, dst, w = _parse_pajek_edge_line(ln)
						push!(edge_srcs, src)
						push!(edge_dsts, dst)
						push!(edge_weights, w)
					catch
						continue
					end
			end

		#	Parse Industry Partition (Optional)
			industry = Int[]
			if ix_industry !== nothing
				#	Skip the *Partition header AND the *Vertices N line beneath it
					skip_count_line = true
					for line_ix in (ix_industry + 1):length(raw)
						ln = strip(raw[line_ix])
						if skip_count_line && startswith(lowercase(ln), "*vertices")
							skip_count_line = false
							continue
						end
						isempty(ln) && continue
						startswith(ln, "*") && break
						val = tryparse(Int, ln)
						if val !== nothing
							push!(industry, val)
						end
						length(industry) ≥ n_firms && break
					end
			end

		#	Parse Capital Vector (Optional)
			capital = Float64[]
			if ix_capital !== nothing
				skip_count_line = true
				for line_ix in (ix_capital + 1):length(raw)
					ln = strip(raw[line_ix])
					if skip_count_line && startswith(lowercase(ln), "*vertices")
						skip_count_line = false
						continue
					end
					isempty(ln) && continue
					startswith(ln, "*") && break
					val = tryparse(Float64, ln)
					if val !== nothing
						push!(capital, val)
					end
					length(capital) ≥ n_firms && break
				end
			end

		#	Build Firms DataFrame
			firms_df = DataFrame(
				id    = 1:n_firms,
				label = firm_labels,
				mode  = fill("firm", n_firms)
			)

		#	Build Directors DataFrame
			directors_df = DataFrame(
				id    = (n_firms + 1):n_total,
				label = director_labels,
				mode  = fill("director", n_total - n_firms)
			)

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src = edge_srcs,
				dst = edge_dsts,
				weight = edge_weights
			)

		#	Return All Components
			return (
				firms     = firms_df,
				directors = directors_df,
				edges     = edges_df,
				industry  = industry,
				capital   = capital
			)
	end

#	Project Firm-Firm Network from Bipartite Firm-Director Edges
	function project_firm_firm(parser_result::NamedTuple)
		"""
		Args:
			parser_result::NamedTuple: output of parse_pajek_2mode with fields
				firms, directors, edges, industry, capital
		Returns:
			NamedTuple: (edges, nodes) where
				edges::DataFrame   — columns :src, :dst, :weight (firm-firm weighted, undirected)
				nodes::DataFrame   — columns :id, :label, :industry, :capital (firm attributes)
		Notes:
			Computes the firm-firm projection of the bipartite firm-director network.
			Two firms i and j are connected with weight equal to the number of
			directors they share. Mathematically, if B is the bipartite incidence
			matrix (firms × directors), the projection is B × B', with diagonal
			set to zero (we exclude self-loops representing firm board size).

			Returns an undirected weighted network. Edge ordering is canonical:
			for each pair {i, j} with i < j, only one edge (src=i, dst=j) is
			emitted, not both (i, j) and (j, i). This matches the convention used
			by the rest of the community detection pipeline for undirected graphs.

			Node attributes carried through from the parser:
			- :id     — firm Pajek vertex id (1..108)
			- :label  — firm name
			- :industry — industry partition code (1..8)
			- :capital  — financial capital value
		"""

		#	Validation
			n_firms     = nrow(parser_result.firms)
			n_directors = nrow(parser_result.directors)
			n_bipartite = nrow(parser_result.edges)
			if n_bipartite == 0
				throw(ArgumentError("project_firm_firm: parser_result has no edges"))
			end

		#	Build Bipartite Incidence Matrix B (firms × directors)
			#	Director column index is (director_id - n_firms), 1-based
				bipartite_src = parser_result.edges.src             # firm ids 1..n_firms
				bipartite_dst = parser_result.edges.dst             # director ids n_firms+1..n_total
				bipartite_w   = parser_result.edges.weight

			#	Map director ids to columns 1..n_directors
				director_cols = bipartite_dst .- n_firms
				B = sparse(bipartite_src, director_cols, bipartite_w, n_firms, n_directors)

		#	Compute Projection B × B'
			#	Result is n_firms × n_firms, symmetric; entry (i,j) = shared director count
				P = B * B'

		#	Extract Upper-Triangular Off-Diagonal Entries
			#	For an undirected graph, we keep one edge per pair {i,j} with i < j
				rows, cols, vals = findnz(P)
				edge_srcs    = Int[]
				edge_dsts    = Int[]
				edge_weights = Float64[]
				for k in eachindex(vals)
					i, j, w = rows[k], cols[k], vals[k]
					if i < j && w > 0
						push!(edge_srcs, i)
						push!(edge_dsts, j)
						push!(edge_weights, w)
					end
				end

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

		#	Build Nodes DataFrame with Firm Attributes
			nodes_df = DataFrame(
				id       = 1:n_firms,
				label    = parser_result.firms.label,
				industry = isempty(parser_result.industry) ? fill(missing, n_firms) : parser_result.industry,
				capital  = isempty(parser_result.capital)  ? fill(missing, n_firms) : parser_result.capital
			)

		#	Return Projection
			return (edges = edges_df, nodes = nodes_df)
	end

#	Parse a 1-Section 2-Mode Pajek File with a Single Bipartite Vertex Block (Marvel-style)
	function parse_pajek_marvel(filepath::String)
		"""
		Args:
			filepath::String: path to the Marvel_Universe.paj (or compatible) file
		Returns:
			NamedTuple: (heroes, comics, edges, vertex_type)
				heroes::DataFrame   — columns :id, :label, :mode (always "hero")
				comics::DataFrame   — columns :id, :label, :mode (always "comic")
				edges::DataFrame    — columns :src, :dst, :weight (unweighted bipartite, weight = 1.0)
				vertex_type::Vector{Int} — per-vertex partition codes (1 = hero, 2 = comic), length n_total
		Notes:
			The Marvel Universe Pajek file declares its layout in the *Vertices header:

				*Vertices 19428 6486

			Here 19428 is the total vertex count and 6486 is the size of mode 1 (heroes).
			Heroes occupy vertex IDs 1..6486; comics occupy vertex IDs 6487..19428. Edges
			in the *Edges section are bipartite hero–comic appearances, one row per
			appearance, with no explicit weight (weight = 1.0 throughout).

			The file also carries a *Partition Vertex_Type.clu block that labels every
			vertex as 1 (hero) or 2 (comic). The parser reads it and returns it as
			vertex_type for consistency-checking against the header-derived mode
			assignment.

			This parser is intentionally distinct from parse_pajek_2mode, which assumes
			the caller supplies the bipartite split (n_firms) externally and looks for
			separate *Partition and *Vector attribute blocks tied to mode 1. The Marvel
			file declares the split in the header itself and has no per-vertex
			attributes beyond the type partition, so the two parsers have different
			argument shapes and different expected section structures.

			Pajek files are commonly Latin-1 (ISO-8859-1) encoded. The reader decodes
			byte-by-byte as Latin-1, which never fails and produces valid UTF-8 strings
			internally. The four Pajek helpers (_strip_pajek_token, _parse_pajek_vertex_line,
			_parse_pajek_edge_line, _find_pajek_section) are shared with parse_pajek_2mode.

			Reference:
				Alberich, R., Miro-Julia, J., and Rossello, F. (2002).
				"Marvel Universe looks almost like a real social network."
				arXiv preprint cond-mat/0202174.
		"""

		#	Validation
			if !isfile(filepath)
				throw(ArgumentError("File not found: $filepath"))
			end

		#	Read All Lines (Decode as Latin-1 / ISO-8859-1)
			#	Pajek files often use Latin-1 encoding for non-ASCII characters; decoding
			#	byte-by-byte sidesteps any UTF-8 validation failures on legacy bytes.
				raw_bytes = read(filepath)
				decoded   = String(Char.(raw_bytes))
				raw       = String.(split(decoded, r"\r?\n"))

		#	Locate Required Section Headers
			ix_vertices  = _find_pajek_section(raw, "*Vertices")
			ix_edges     = _find_pajek_section(raw, "*Edges")
			ix_partition = _find_pajek_section(raw, "*Partition")
			if ix_vertices === nothing
				throw(ArgumentError("Pajek file missing *Vertices section"))
			end
			if ix_edges === nothing
				throw(ArgumentError("Pajek file missing *Edges section"))
			end

		#	Parse Bipartite Header (Total Count + Mode-1 Count)
			#	Marvel header form:  *Vertices 19428 6486
			#	→ n_total = 19428, n_heroes = 6486 (mode 1), n_comics = n_total - n_heroes
				header_line   = strip(raw[ix_vertices])
				header_tokens = split(header_line)
				if length(header_tokens) < 3
					throw(ArgumentError("Expected bipartite *Vertices header with total and mode-1 counts; got: $header_line"))
				end
				n_total  = parse(Int, header_tokens[2])
				n_heroes = parse(Int, header_tokens[3])
				if !(1 ≤ n_heroes < n_total)
					throw(ArgumentError("Bipartite split invalid: n_heroes=$n_heroes, n_total=$n_total"))
				end
				n_comics = n_total - n_heroes

		#	Parse Vertices
			hero_labels  = Vector{String}(undef, n_heroes)
			comic_labels = Vector{String}(undef, n_comics)
			for line_ix in (ix_vertices + 1):(ix_edges - 1)
				ln = strip(raw[line_ix])
				isempty(ln) && continue
				startswith(ln, "*") && break    # next section header
				startswith(ln, "%") && continue # comment line
				vid, label = _parse_pajek_vertex_line(ln)
				if 1 ≤ vid ≤ n_heroes
					hero_labels[vid] = label
				elseif n_heroes < vid ≤ n_total
					comic_labels[vid - n_heroes] = label
				end
			end

		#	Parse Edges (Stop at Next Section or EOF)
			edge_srcs    = Int[]
			edge_dsts    = Int[]
			edge_weights = Float64[]
			for line_ix in (ix_edges + 1):length(raw)
				ln = strip(raw[line_ix])
				isempty(ln) && continue
				if startswith(ln, "*")
					break                       # reached next section
				end
				startswith(ln, "%") && continue # comment line
				#	The Marvel edge format is unweighted: "src_id dst_id". If a future
				#	variant adds weights, _parse_pajek_edge_line handles that too.
					try
						src, dst, w = _parse_pajek_edge_line(ln)
						push!(edge_srcs, src)
						push!(edge_dsts, dst)
						push!(edge_weights, w)
					catch
						continue
					end
			end

		#	Parse Vertex Type Partition (Optional Sanity Check)
			vertex_type = Int[]
			if ix_partition !== nothing
				#	Skip the *Partition header AND the *Vertices N line beneath it
					skip_count_line = true
					for line_ix in (ix_partition + 1):length(raw)
						ln = strip(raw[line_ix])
						if skip_count_line && startswith(lowercase(ln), "*vertices")
							skip_count_line = false
							continue
						end
						isempty(ln) && continue
						startswith(ln, "*") && break
						val = tryparse(Int, ln)
						if val !== nothing
							push!(vertex_type, val)
						end
						length(vertex_type) ≥ n_total && break
					end
			end

		#	Build Heroes DataFrame
			heroes_df = DataFrame(
				id    = 1:n_heroes,
				label = hero_labels,
				mode  = fill("hero", n_heroes)
			)

		#	Build Comics DataFrame
			comics_df = DataFrame(
				id    = (n_heroes + 1):n_total,
				label = comic_labels,
				mode  = fill("comic", n_comics)
			)

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

		#	Return All Components
			return (
				heroes      = heroes_df,
				comics      = comics_df,
				edges       = edges_df,
				vertex_type = vertex_type
			)
	end

#	Project Hero-Hero Network from Bipartite Hero-Comic Edges, Thresholded at Weight ≥ 2
	function project_marvel_characters(parser_result::NamedTuple;
	                                   min_weight::Int = 2)
		"""
		Args:
			parser_result::NamedTuple: output of parse_pajek_marvel with fields
				heroes, comics, edges, vertex_type
			min_weight::Int: minimum co-appearance count for an edge to be kept
				(default = 2, dropping single co-appearances per the punchlist)
		Returns:
			NamedTuple: (edges, nodes) where
				edges::DataFrame   — columns :src, :dst, :weight (hero-hero weighted, undirected)
				nodes::DataFrame   — columns :id, :label (hero attributes)
		Notes:
			Computes the hero-hero co-appearance projection of the bipartite hero-comic
			network. Two heroes i and j are connected with weight equal to the number
			of comics in which they both appear. Mathematically, if B is the bipartite
			incidence matrix (heroes × comics), the projection is C = B × B', with the
			diagonal zeroed (no self-loops) and entries below min_weight dropped.

			The threshold at min_weight = 2 removes pairs of heroes who appear together
			in only a single comic. This is the punchlist-specified policy: single
			co-appearances are dropped, retaining only pairs with documented recurring
			collaboration. Pre-threshold the projection contains a very large number of
			weak (weight = 1) edges driven by ensemble issues; thresholding produces a
			network of meaningful collaboration structure.

			Returns an undirected weighted network. Edge ordering is canonical: for
			each pair {i, j} with i < j, only one edge (src=i, dst=j) is emitted, not
			both (i, j) and (j, i). This matches the convention used by the rest of
			the community detection pipeline for undirected graphs.

			Node attributes carried through from the parser:
			- :id    — hero Pajek vertex id (1..n_heroes)
			- :label — hero name (e.g., "CAPTAIN AMERICA", "SPIDER-MAN/PETER PARKER")
		"""

		#	Validation
			n_heroes    = nrow(parser_result.heroes)
			n_comics    = nrow(parser_result.comics)
			n_bipartite = nrow(parser_result.edges)
			if n_bipartite == 0
				throw(ArgumentError("project_marvel_characters: parser_result has no edges"))
			end
			if min_weight < 1
				throw(ArgumentError("min_weight must be ≥ 1; got $min_weight"))
			end

		#	Build Bipartite Incidence Matrix B (heroes × comics)
			#	Comic column index is (comic_id - n_heroes), 1-based
				bipartite_src = parser_result.edges.src             # hero ids 1..n_heroes
				bipartite_dst = parser_result.edges.dst             # comic ids n_heroes+1..n_total
				bipartite_w   = parser_result.edges.weight          # all 1.0 for Marvel

			#	Map comic ids to columns 1..n_comics
				comic_cols = bipartite_dst .- n_heroes
				B = sparse(bipartite_src, comic_cols, bipartite_w, n_heroes, n_comics)

		#	Compute Projection C = B × B'
			#	Result is n_heroes × n_heroes, symmetric; entry (i,j) = shared comic count
			#	(i.e., number of comics in which heroes i and j co-appear)
				C = B * B'

		#	Extract Upper-Triangular Off-Diagonal Entries Meeting Threshold
			#	For an undirected graph, we keep one edge per pair {i,j} with i < j.
			#	Single co-appearances (weight = 1) are dropped per min_weight policy.
				rows, cols, vals = findnz(C)
				edge_srcs    = Int[]
				edge_dsts    = Int[]
				edge_weights = Float64[]
				n_pre_threshold = 0
				for k in eachindex(vals)
					i, j, w = rows[k], cols[k], vals[k]
					if i < j && w > 0
						n_pre_threshold += 1
						if w ≥ min_weight
							push!(edge_srcs, i)
							push!(edge_dsts, j)
							push!(edge_weights, w)
						end
					end
				end

		#	Report Threshold Impact
			println("    Projection pre-threshold:  $n_pre_threshold unordered hero-hero pairs")
			println("    Projection post-threshold: $(length(edge_srcs)) edges (weight ≥ $min_weight)")
			println("    Dropped:                   $(n_pre_threshold - length(edge_srcs)) single-coappearance pairs")

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

		#	Build Nodes DataFrame with Hero Attributes
			nodes_df = DataFrame(
				id    = 1:n_heroes,
				label = parser_result.heroes.label
			)

		#	Return Projection
			return (edges = edges_df, nodes = nodes_df)
	end

#	Parse a Konect-Format Edge List File
	function parse_konect(filepath::String)
		"""
		Args:
			filepath::String: path to a Konect-format edge list file
		Returns:
			NamedTuple: (edges, nodes, properties)
				edges::DataFrame   — columns :src, :dst, :weight
				nodes::DataFrame   — columns :id (1-based integer IDs)
				properties::NamedTuple — graph properties extracted from header
		Notes:
			Reads the Konect edge list format. The format is:

				% [graph_type] [weight_type]      (line 1, optional)
				% n_edges n_nodes_src n_nodes_dst (line 2, optional)
				src1  dst1  [weight1]
				src2  dst2  [weight2]
				...

			Header lines begin with '%'. Common graph_type values:
				asym         — directed (asymmetric)
				sym          — undirected (symmetric)
				bip          — bipartite
			Common weight_type values:
				unweighted   — all edges have weight 1
				posweighted  — weighted with positive integers
				signed       — weights can be negative
				multiple     — multiedges represented by repeated entries

			Returns 1-based integer node IDs in :id, with the same value duplicated
			as :label so downstream code that expects a label column has one.

			The properties NamedTuple contains :directed, :weighted, :n_edges_declared,
			:n_nodes_declared, and :raw_header_lines (the original header text).
		"""

		#	Validation
			if !isfile(filepath)
				throw(ArgumentError("File not found: $filepath"))
			end

		#	Read All Lines
			raw = readlines(filepath)

		#	Separate Header Lines from Data Lines
			header_lines = String[]
			data_lines   = String[]
			for ln in raw
				if startswith(strip(ln), "%")
					push!(header_lines, ln)
				elseif !isempty(strip(ln))
					push!(data_lines, ln)
				end
			end

		#	Parse Header Properties
			directed     = false
			weighted     = false
			n_edges_decl = nothing
			n_nodes_decl = nothing

			for hdr in header_lines
				#	Strip leading '%' and Whitespace
					content = strip(replace(hdr, r"^\s*%\s*" => ""))
					isempty(content) && continue

				#	Tokenize
					tokens = split(content)

				#	First Line Often Contains Type Keywords
					for tok in tokens
						lc = lowercase(tok)
						if lc == "asym"
							directed = true
						elseif lc == "sym"
							directed = false
						elseif lc in ("posweighted", "signed", "multiple", "weighted")
							weighted = true
						elseif lc == "unweighted"
							weighted = false
						end
					end

				#	Second Line Often Contains "n_edges n_nodes_src n_nodes_dst" or "n_edges n_nodes"
					#	Take the first numeric triple/pair as the count
						if n_edges_decl === nothing
							ints = [tryparse(Int, t) for t in tokens]
							int_tokens = filter(!isnothing, ints)
							if length(int_tokens) ≥ 2
								n_edges_decl = int_tokens[1]
								n_nodes_decl = int_tokens[2]
							end
						end
			end

		#	Parse Edge Lines
			edge_srcs    = Int[]
			edge_dsts    = Int[]
			edge_weights = Float64[]
			for ln in data_lines
				tokens = split(strip(ln))
				length(tokens) < 2 && continue

				src = tryparse(Int, tokens[1])
				dst = tryparse(Int, tokens[2])
				(src === nothing || dst === nothing) && continue

				weight = 1.0
				if length(tokens) ≥ 3
					w = tryparse(Float64, tokens[3])
					if w !== nothing
						weight = w
					end
				end

				push!(edge_srcs, src)
				push!(edge_dsts, dst)
				push!(edge_weights, weight)
			end

		#	Build Edges DataFrame
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

		#	Build Nodes DataFrame
			#	Determine node universe: max ID seen, or declared count
				max_id = isempty(edge_srcs) ? 0 : max(maximum(edge_srcs), maximum(edge_dsts))
				n_nodes = n_nodes_decl === nothing ? max_id : max(max_id, n_nodes_decl)
				nodes_df = DataFrame(
					id    = 1:n_nodes,
					label = string.(1:n_nodes)
				)

		#	Build Properties NamedTuple
			properties = (
				directed         = directed,
				weighted         = weighted,
				n_edges_declared = n_edges_decl,
				n_nodes_declared = n_nodes_decl,
				raw_header_lines = header_lines
			)

		#	Return All Components
			return (edges = edges_df, nodes = nodes_df, properties = properties)
	end

#	Parse Balikatan ORA-Exported CSV Files (Node + Edge Pair)
	function parse_balikatan_ora(nodes_path::String,
	                              edges_path::String)
		"""
		Args:
			nodes_path::String: path to ORA-exported nodeset CSV with Node ID and attribute columns
			edges_path::String: path to ORA-exported edge CSV with Source Node ID, Target Node ID, Link Value
		Returns:
			NamedTuple: (edges, nodes, properties)
				edges::DataFrame   — columns :src, :dst, :weight
				nodes::DataFrame   — column :id plus all original attribute columns (renamed
					to snake_case for consistency)
				properties::NamedTuple — graph properties
					(:directed, :weighted, :n_edges, :n_nodes, :format)
		Notes:
			ORA exports node IDs as Twitter user IDs (Int64). The loader preserves
			these as integers without conversion to Float64.

			The function builds an explicit positional index of columns from the
			CSV header, then constructs pre-allocated new-name vectors filled by
			position. This avoids dictionary-keyed lookup semantics that have
			changed across DataFrames.jl versions and avoids any push! growth
			that could fail under unusual column counts.

			Attribute columns in the nodeset file are renamed to snake_case:
			- "Node ID" → :id
			- "Node Label" → :label
			- Other columns → lowercase with spaces, commas, and hyphens
			  replaced by underscores

			The edges file is expected to have exactly the columns:
			"Source Node ID", "Target Node ID", "Link Value"
			renamed to :src, :dst, :weight.

			CSV.read is called without an explicit `types` argument because
			CSV.jl's column-type inference handles ORA exports correctly.
		"""

		#	Validation
			if !isfile(nodes_path)
				throw(ArgumentError("Nodes file not found: $nodes_path"))
			end
			if !isfile(edges_path)
				throw(ArgumentError("Edges file not found: $edges_path"))
			end

		#	Helper: Convert a Column Name to snake_case
			function _to_snake_case(name::AbstractString)
				canonical = lowercase(name)
				canonical = replace(canonical, r"[\s,\-]+" => "_")
				canonical = replace(canonical, r"_+" => "_")
				canonical = strip(canonical, '_')
				return Symbol(canonical)
			end

		#	Load Edges CSV (Auto-Detected Types)
			edges_raw = CSV.read(edges_path, DataFrame)

			#	Build Edges Column Index and Verify Required Columns
				edge_cols = names(edges_raw)
				n_edge_cols = length(edge_cols)
				required_edge_cols = ["Source Node ID", "Target Node ID", "Link Value"]
				for col in required_edge_cols
					if !(col in edge_cols)
						throw(ArgumentError("Edges file missing required column '$col'. Found: $edge_cols"))
					end
				end

			#	Build New Column Names by Position (Pre-Allocated)
				new_edge_names = Vector{Symbol}(undef, n_edge_cols)
				@inbounds for i in 1:n_edge_cols
					col = edge_cols[i]
					if col == "Source Node ID"
						new_edge_names[i] = :src
					elseif col == "Target Node ID"
						new_edge_names[i] = :dst
					elseif col == "Link Value"
						new_edge_names[i] = :weight
					else
						new_edge_names[i] = _to_snake_case(col)
					end
				end

			#	Apply Renaming by Position
				rename!(edges_raw, new_edge_names)

			#	Ensure Weight Column is Float64
				edges_raw.weight = Float64.(edges_raw.weight)

		#	Load Nodes CSV (Auto-Detected Types)
			nodes_raw = CSV.read(nodes_path, DataFrame)

			#	Build Nodes Column Index
				node_cols = names(nodes_raw)
				n_node_cols = length(node_cols)

			#	Build New Column Names by Position (Pre-Allocated)
				new_node_names = Vector{Symbol}(undef, n_node_cols)
				@inbounds for i in 1:n_node_cols
					col = node_cols[i]
					if col == "Node ID"
						new_node_names[i] = :id
					elseif col == "Node Label"
						new_node_names[i] = :label
					else
						new_node_names[i] = _to_snake_case(col)
					end
				end

			#	Apply Renaming by Position
				rename!(nodes_raw, new_node_names)

		#	Verify :id Column Exists After Renaming
			if !hasproperty(nodes_raw, :id)
				throw(ErrorException("Failed to find 'Node ID' column after renaming. Found columns: $(names(nodes_raw))"))
			end

		#	Reorder Columns: id, label, rest (Pre-Allocated)
			current_cols = propertynames(nodes_raw)
			col_order = Vector{Symbol}(undef, length(current_cols))
			col_order[1] = :id
			next_ix = 2

			#	Place label Second If Present
				if hasproperty(nodes_raw, :label)
					col_order[next_ix] = :label
					next_ix += 1
				end

			#	Append Remaining Columns in Original Order
				@inbounds for c in current_cols
					if c != :id && c != :label
						col_order[next_ix] = c
						next_ix += 1
					end
				end

			#	Apply Reordering
				select!(nodes_raw, col_order)

		#	Build Properties NamedTuple
			properties = (
				directed = true,
				weighted = true,
				n_edges  = nrow(edges_raw),
				n_nodes  = nrow(nodes_raw),
				format   = "ora_csv"
			)

		#	Return All Components
			return (
				edges      = edges_raw,
				nodes      = nodes_raw,
				properties = properties
			)
	end

#	Helper Function for parse_toledo_crime: Read One Crime Layer CSV
	function _read_toledo_layer(filepath::String)
		"""
		Args:
			filepath::String: path to a single Toledo crime layer CSV (semicolon-separated)
		Returns:
			Vector{Tuple{Int, Int}}: unordered edge pairs (min_node, max_node)
		Notes:
			Reads a single Toledo layer file. The format is:
				source;target
				1;2
				3;4
				...

			Files may have UTF-8 BOM and CRLF line endings. The reader handles both.
			Each edge is normalized to (min, max) so that the same underlying
			undirected edge produces the same tuple regardless of how it was
			written in the source file.

			Skipping headers, blank lines, and lines that fail integer parsing.
		"""

		#	Read Raw Bytes and Decode
			raw_bytes = read(filepath)
			content   = String(Char.(raw_bytes))

		#	Strip UTF-8 BOM If Present
			if startswith(content, "\ufeff")
				content = content[nextind(content, 1):end]
			end

		#	Split Lines (Handle Both CRLF and LF)
			lines = String.(split(content, r"\r?\n"))

		#	Parse Edges (Skip Header and Blank Lines)
			edges = Tuple{Int, Int}[]
			for (i, ln) in pairs(lines)
				stripped = strip(ln)
				isempty(stripped) && continue
				i == 1 && continue                  # header line

				#	Split on Semicolon
					tokens = split(stripped, ';')
					length(tokens) < 2 && continue

				#	Parse Integers
					a = tryparse(Int, strip(tokens[1]))
					b = tryparse(Int, strip(tokens[2]))
					(a === nothing || b === nothing) && continue

				#	Normalize to Unordered Pair
					if a < b
						push!(edges, (a, b))
					elseif b < a
						push!(edges, (b, a))
					end
					#	Self-loops (a == b) are silently dropped
			end

		#	Return
			return edges
	end

#	Parse Toledo Crime Networks: Collapse 7 Layers Into One Network
	function parse_toledo_crime(layer_dir::String;
	                            crime_layers::Vector{String} = [
	                                "edges_homicide.csv",
	                                "edges_kidnapping.csv",
	                                "edges_theft.csv",
	                                "edges_drug_trafficking.csv",
	                                "edges_arms_trafficking.csv",
	                                "edges_weapon_carrying.csv",
	                                "edges_ideological_falsehood.csv",
	                            ])
		"""
		Args:
			layer_dir::String: path to directory containing the Toledo CSV files
			crime_layers::Vector{String}: filenames of the layers to include in the collapse.
				Defaults to the 7 offender-cooperation layers. Excludes the target_network
				layer (offender-victim) and the crime_network.csv summary.
		Returns:
			NamedTuple: (edges, nodes, layer_counts, properties)
				edges::DataFrame   — columns :src, :dst, :weight
					weight = number of layers in which this edge appears (1..7)
				nodes::DataFrame   — column :id (1-based integer IDs)
				layer_counts::Dict{String, Int} — number of edges in each layer file
				properties::NamedTuple — graph properties:
					:directed=false, :weighted=true, :n_layers, :n_edges, :n_nodes, :format
		Notes:
			Toledo crime networks (Toledo et al. 2023) are distributed as 8 layer
			CSV files, one per crime category, plus a summary file. Each layer file
			has semicolon-separated source;target pairs with no weights or
			directionality.

			This loader collapses 7 offender-cooperation layers (homicide,
			kidnapping, theft, drug_trafficking, arms_trafficking, weapon_carrying,
			ideological_falsehood) into a single weighted network where edge
			weight equals the number of layers in which the two offenders
			co-participated. The target_network layer (offender-victim
			relationships) is excluded as it represents a different relationship
			semantics.

			Layer files may have UTF-8 BOM and CRLF line endings; both are handled.

			Edges are treated as undirected: each pair is normalized to
			(min_node, max_node) before aggregation, so a→b and b→a from
			different layers count as the same edge.
		"""

		#	Validation
			if !isdir(layer_dir)
				throw(ArgumentError("Directory not found: $layer_dir"))
			end

		#	Aggregate Edges Across Layers
			#	Dict key is normalized pair (i, j) with i < j; value is the layer count
				edge_counts = Dict{Tuple{Int, Int}, Int}()
				layer_counts = Dict{String, Int}()

			for layer_file in crime_layers
				path = joinpath(layer_dir, layer_file)
				if !isfile(path)
					@warn "Skipping missing layer file: $path"
					layer_counts[layer_file] = 0
					continue
				end

				#	Read This Layer's Edges
					layer_edges = _read_toledo_layer(path)
					layer_counts[layer_file] = length(layer_edges)

				#	Add to Aggregate Count
					#	Use Set to avoid double-counting within a single layer
					#	(if a layer has multi-edges, we count them once for this layer)
						unique_layer_edges = Set(layer_edges)
						for edge in unique_layer_edges
							edge_counts[edge] = get(edge_counts, edge, 0) + 1
						end
			end

		#	Build Edges DataFrame
			n_edges_total = length(edge_counts)
			edge_srcs     = Vector{Int}(undef, n_edges_total)
			edge_dsts     = Vector{Int}(undef, n_edges_total)
			edge_weights  = Vector{Float64}(undef, n_edges_total)

			ix = 1
			for ((s, d), w) in edge_counts
				edge_srcs[ix]    = s
				edge_dsts[ix]    = d
				edge_weights[ix] = Float64(w)
				ix += 1
			end

			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

		#	Determine Node Universe
			all_nodes = Set{Int}()
			for ((s, d), _) in edge_counts
				push!(all_nodes, s)
				push!(all_nodes, d)
			end
			node_ids = sort(collect(all_nodes))

		#	Build Nodes DataFrame
			nodes_df = DataFrame(
				id    = node_ids,
				label = string.(node_ids)
			)

		#	Build Properties NamedTuple
			properties = (
				directed = false,
				weighted = true,
				n_layers = length(crime_layers),
				n_edges  = nrow(edges_df),
				n_nodes  = nrow(nodes_df),
				format   = "toledo_layered_csv"
			)

		#	Return All Components
			return (
				edges        = edges_df,
				nodes        = nodes_df,
				layer_counts = layer_counts,
				properties   = properties
			)
	end

#	Binarize an Edge DataFrame
	function binarize_edges_df(edges_df::DataFrame)
		"""
		Args:
			edges_df::DataFrame: edges with :src, :dst, :weight columns
		Returns:
			DataFrame: edges with weight column replaced by 1.0 everywhere
		Notes:
			Used to produce the unweighted variant of a weighted network. Edge
			existence is preserved; only the weight column is replaced. Does not
			modify the input DataFrame; returns a copy.
		"""

		#	Copy and Replace Weight Column
			out = copy(edges_df)
			out.weight = ones(Float64, nrow(out))

		#	Return
			return out
	end

#	Helper Function for write_graphml: Map Julia Type to GraphML Attribute Type
	function _graphml_attr_type(col::AbstractVector)
		"""
		Args:
			col::AbstractVector: a DataFrame column
		Returns:
			String: GraphML attribute type ("int", "double", "boolean", or "string")
		Notes:
			Inspects the non-missing element type of the column. Falls back to
			"string" for any type not recognized as int/float/bool.
		"""

		#	Determine Non-Missing Element Type
			T = eltype(col)
			T = T isa Union ? Base.nonmissingtype(T) : T

		#	Map to GraphML Type Code
			if T <: Integer && !(T <: Bool)
				return "int"
			elseif T <: AbstractFloat
				return "double"
			elseif T <: Bool
				return "boolean"
			else
				return "string"
			end
	end

#	Helper Function for write_graphml: Escape XML Special Characters
	function _xml_escape(s::AbstractString)
		"""
		Args:
			s::AbstractString: raw text to be embedded in XML
		Returns:
			String: text with &, <, >, ', " replaced by their entity references
		Notes:
			Required for safe inclusion in both attribute values and element text.
		"""

		#	Replace Special Characters
			s2 = replace(String(s),
			             "&" => "&amp;",
			             "<" => "&lt;",
			             ">" => "&gt;",
			             "\"" => "&quot;",
			             "'" => "&apos;")

		#	Return
			return s2
	end

#	Helper Function for write_graphml: Format Value as GraphML String
	function _format_graphml_value(v, attr_type::AbstractString)
		"""
		Args:
			v: a Julia value (Int, Float, Bool, String, Missing)
			attr_type::AbstractString: target GraphML type
		Returns:
			Union{String, Nothing}: serialized value, or nothing if missing
		Notes:
			Booleans are written as "true" / "false" per GraphML convention.
			Floats use default Julia string conversion (full precision).
			Missing values return nothing so the caller can skip emission entirely.
			An empty string ("") is NOT missing — it is a legitimate value and is
			returned as "", which the writer then emits as a present-but-empty
			<data> element. The reader (_parse_graphml_value) distinguishes
			these on the way back: empty text for a string attribute returns "",
			while empty text for a non-string attribute returns missing.
		"""

		#	Handle Missing (Not Empty String)
			(v === missing || v === nothing) && return nothing

		#	Format by Type
			if attr_type == "boolean"
				return v ? "true" : "false"
			elseif attr_type == "int"
				return string(Int(v))
			elseif attr_type == "double"
				return string(Float64(v))
			else
				return string(v)
			end
	end

#	Write a Network as a GraphML File
	function write_graphml(edges_df::DataFrame,
	                       nodes_df::DataFrame,
	                       metadata::Union{NamedTuple, AbstractDict},
	                       output_path::String)
		"""
		Args:
			edges_df::DataFrame: must contain :src, :dst, :weight; other columns
				become edge attributes
			nodes_df::DataFrame: must contain :id; other columns become node attributes
			metadata::NamedTuple or Dict: graph-level attributes; should include at
				minimum :network_name, :source_format, :directed, :weighted; additional
				fields are written as graph-level <data> elements
			output_path::String: file path for the output GraphML file
		Returns:
			Nothing
		Notes:
			Writes a valid GraphML 1.0 file following the standard schema. Auto-
			detects attribute types from DataFrame column element types (int,
			double, boolean, or string). Missing values are written as absent
			<data> elements rather than empty strings.

			Node IDs in the output are prefixed with "n" (e.g., id=42 becomes "n42")
			to satisfy GraphML's requirement that IDs be valid XML names.

			Edge direction is set via the graph's edgedefault attribute, drawn from
			metadata.directed. Individual edge directionality overrides are not
			emitted.

			Following conventions are reserved (used by the writer, not turned into
			attributes):
			- nodes_df: :id
			- edges_df: :src, :dst
			Other columns in either DataFrame are emitted as attributes.
		"""

		#	Validation
			if !hasproperty(edges_df, :src) || !hasproperty(edges_df, :dst)
				throw(ArgumentError("edges_df must have :src and :dst columns"))
			end
			if !hasproperty(nodes_df, :id)
				throw(ArgumentError("nodes_df must have :id column"))
			end

		#	Normalize Metadata to Dict
			meta = metadata isa AbstractDict ? Dict(metadata) :
			       Dict(string(k) => v for (k, v) in pairs(metadata))

			#	Required Metadata Fields
				directed = get(meta, "directed", false) === true
				edgedefault = directed ? "directed" : "undirected"

		#	Identify Node Attribute Columns (Exclude Reserved)
			node_attr_cols = [c for c in propertynames(nodes_df) if c != :id]
			node_attr_types = Dict(c => _graphml_attr_type(nodes_df[!, c]) for c in node_attr_cols)

		#	Identify Edge Attribute Columns (Exclude Reserved)
			edge_attr_cols = [c for c in propertynames(edges_df) if c != :src && c != :dst]
			edge_attr_types = Dict(c => _graphml_attr_type(edges_df[!, c]) for c in edge_attr_cols)

		#	Identify Graph Metadata Keys and Types
			meta_keys = collect(keys(meta))
			meta_types = Dict{String, String}()
			for k in meta_keys
				v = meta[k]
				t = if v isa Bool
					"boolean"
				elseif v isa Integer
					"int"
				elseif v isa AbstractFloat
					"double"
				else
					"string"
				end
				meta_types[k] = t
			end

		#	Build XML Document
			doc = XMLDocument()
			root = ElementNode("graphml")
			setroot!(doc, root)

			#	GraphML Namespace Declarations
				root["xmlns"] = "http://graphml.graphdrawing.org/xmlns"
				root["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
				root["xsi:schemaLocation"] = "http://graphml.graphdrawing.org/xmlns " *
				                              "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"

			#	Declare Graph-Level Metadata Keys
				for k in meta_keys
					key_node = ElementNode("key")
					key_node["id"] = "g_$k"
					key_node["for"] = "graph"
					key_node["attr.name"] = k
					key_node["attr.type"] = meta_types[k]
					link!(root, key_node)
				end

			#	Declare Node Attribute Keys
				for c in node_attr_cols
					key_node = ElementNode("key")
					key_node["id"] = "n_$c"
					key_node["for"] = "node"
					key_node["attr.name"] = string(c)
					key_node["attr.type"] = node_attr_types[c]
					link!(root, key_node)
				end

			#	Declare Edge Attribute Keys
				for c in edge_attr_cols
					key_node = ElementNode("key")
					key_node["id"] = "e_$c"
					key_node["for"] = "edge"
					key_node["attr.name"] = string(c)
					key_node["attr.type"] = edge_attr_types[c]
					link!(root, key_node)
				end

			#	Graph Element
				graph_node = ElementNode("graph")
				graph_node["edgedefault"] = edgedefault
				link!(root, graph_node)

			#	Emit Graph-Level Metadata Values
				for k in meta_keys
					val_str = _format_graphml_value(meta[k], meta_types[k])
					val_str === nothing && continue
					data_node = ElementNode("data")
					data_node["key"] = "g_$k"
					link!(data_node, TextNode(val_str))
					link!(graph_node, data_node)
				end

			#	Emit Nodes with Attributes
				for row in eachrow(nodes_df)
					node_el = ElementNode("node")
					node_el["id"] = "n$(row.id)"
					link!(graph_node, node_el)

					for c in node_attr_cols
						v = row[c]
						val_str = _format_graphml_value(v, node_attr_types[c])
						val_str === nothing && continue
						data_node = ElementNode("data")
						data_node["key"] = "n_$c"
						link!(data_node, TextNode(val_str))
						link!(node_el, data_node)
					end
				end

			#	Emit Edges with Attributes
				for row in eachrow(edges_df)
					edge_el = ElementNode("edge")
					edge_el["source"] = "n$(row.src)"
					edge_el["target"] = "n$(row.dst)"
					link!(graph_node, edge_el)

					for c in edge_attr_cols
						v = row[c]
						val_str = _format_graphml_value(v, edge_attr_types[c])
						val_str === nothing && continue
						data_node = ElementNode("data")
						data_node["key"] = "e_$c"
						link!(data_node, TextNode(val_str))
						link!(edge_el, data_node)
					end
				end

		#	Write to Disk
			write(output_path, doc)

		#	Return
			return nothing
	end

#	Helper Function for load_graphml: Parse a Single GraphML Value
	function _parse_graphml_value(text::AbstractString, attr_type::AbstractString)
		"""
		Args:
			text::AbstractString: raw text content from a <data> element
			attr_type::AbstractString: GraphML type ("int", "double", "boolean", "string")
		Returns:
			Union{Int, Float64, Bool, String, Missing}: parsed value
		Notes:
			For string-typed attributes, the text content is returned verbatim
			(after XML entity unescape) — including any leading or trailing
			whitespace, which is treated as part of the value. This matters for
			datasets where labels carry trailing spaces or other whitespace as
			meaningful data (e.g., Marvel character names from the
			Alberich/Miro-Julia/Rossello 2002 dataset, where some labels end
			with a trailing space as a marker for missing surname components).

			An empty (or whitespace-only) text content for a string-typed attr
			is returned as the empty string "". This preserves writer/reader
			symmetry: write_graphml emits "" as a present-but-empty <data>
			element (an absent attribute is emitted as no <data> element at
			all), and the reader must recover "" rather than coercing to missing.

			For non-string types (int, double, boolean), surrounding whitespace
			is stripped before parsing, and empty (or whitespace-only) text
			returns missing because it cannot be parsed.

			Numeric and boolean parses use the declared attr_type. String parses
			run XML entity unescaping (&amp;, &lt;, &gt;, &quot;, &apos;).
		"""

		#	String-Typed Attribute: Preserve Text Verbatim (After Unescape)
		#	Empty or whitespace-only content collapses to "" (a legitimate value).
			if attr_type == "string"
				return isempty(strip(text)) ? "" : _xml_unescape(String(text))
			end

		#	Non-String Types: Strip Whitespace Before Parsing
			t = strip(text)
			if isempty(t)
				return missing
			end

		#	Parse by Declared Type
			if attr_type == "int"
				return parse(Int, t)
			elseif attr_type == "double"
				return parse(Float64, t)
			elseif attr_type == "boolean"
				return lowercase(t) in ("true", "1")
			else
				#	Unknown type: fall back to verbatim string
					return _xml_unescape(String(text))
			end
	end

#	Helper Function for load_graphml: Strip Node ID Prefix
	function _strip_node_prefix(id_str::AbstractString)
		"""
		Args:
			id_str::AbstractString: a GraphML node ID, expected to be "n{integer}" or similar
		Returns:
			String: the stripped ID (e.g., "n42" → "42")
		Notes:
			The writer prefixes node IDs with "n" (e.g., id=42 becomes "n42") to
			satisfy GraphML's requirement that element IDs be valid XML names.
			This helper reverses the prefix and returns the result as a String,
			preserving the type convention used by the rest of the
			Network_Credible_Intervals package — where node IDs flow through
			internal helpers (e.g., _graph_to_sparse_matrix) as strings.

			If no "n" prefix is found, returns the input unchanged (still as a
			String). This supports GraphML files from other sources that may
			use different ID conventions (Gephi, igraph, NetworkX).

			Previously this helper parsed the result to Int. That choice was
			inconsistent with how IDs are represented elsewhere in the package
			and required adapter code in every downstream consumer. Returning
			String here puts the type convention in one place.
		"""

		#	Strip "n" Prefix If Present, Otherwise Return Unchanged
			s = String(id_str)
			if startswith(s, "n")
				return s[2:end]
			else
				return s
			end
	end

#	Helper Function for load_graphml: Unescape XML Entities
	function _xml_unescape(s::AbstractString)
		"""
		Args:
			s::AbstractString: XML text content with possibly-escaped entities
		Returns:
			String: text with &amp;, &lt;, &gt;, &quot;, &apos; replaced by their characters
		Notes:
			EzXML's nodecontent() returns text without unescaping entity references,
			so we unescape manually. Handles the five standard XML predefined entities.
			Numeric character references (&#NNN; or &#xHHHH;) are not handled — add
			support if needed for future datasets.

			Note: order matters. &amp; must be replaced last because the replacements
			for the other entities (&lt; → <, etc.) don't introduce ampersands, but
			replacing &amp; first would risk re-interpreting any literal &amp; that
			follows a properly-unescaped entity.
		"""

		#	Replace Standard XML Entities (Order Matters)
			s2 = replace(String(s),
			             "&lt;"   => "<",
			             "&gt;"   => ">",
			             "&quot;" => "\"",
			             "&apos;" => "'",
			             "&amp;"  => "&")

		#	Return
			return s2
	end

#	Load a Network from a GraphML File
	function load_graphml(filepath::String)
		"""
		Args:
			filepath::String: path to a GraphML file (typically produced by write_graphml)
		Returns:
			NamedTuple: (edges, nodes, metadata)
				edges::DataFrame   — :src, :dst, :weight, plus any other edge attributes
				nodes::DataFrame   — :id, plus any other node attributes
				metadata::NamedTuple — graph-level attributes from the file
		Notes:
			Reverses write_graphml. Type-aware: numeric attributes come back as
			Int or Float64, booleans as Bool, everything else as String.

			Node and edge IDs are returned as String, not Int. This is a
			deliberate convention choice that aligns with the rest of the
			Network_Credible_Intervals package, where internal helpers
			(particularly _graph_to_sparse_matrix in network_community_detection)
			expect string-valued IDs. The "n{integer}" prefix that write_graphml
			emits is stripped on read; the integer portion is returned as a
			String rather than parsed to Int.

			Reads only well-formed GraphML 1.0 documents matching the schema
			emitted by write_graphml. Edges are expected to have :src/:dst (the
			source/target attributes on <edge> elements). Node IDs are expected
			to have a leading "n" prefix that this function strips.

			GraphML files from other sources (Gephi, igraph, NetworkX) may not
			use the "n{integer}" ID convention. This loader handles other prefix
			schemes by returning the ID unchanged (still as a String) if "n" is
			absent.

			Missing values are emitted as absent <data> elements by write_graphml,
			and load_graphml reads them back as `missing` in the corresponding
			DataFrame column.
		"""

		#	Validation
			if !isfile(filepath)
				throw(ArgumentError("File not found: $filepath"))
			end

		#	Parse XML
			doc = readxml(filepath)
			root = EzXML.root(doc)
			if EzXML.nodename(root) != "graphml"
				throw(ArgumentError("Not a GraphML file: root element is $(EzXML.nodename(root))"))
			end

		#	Build Key Schema: id → (for_type, attr_name, attr_type)
			key_schema = Dict{String, NamedTuple{(:for_type, :attr_name, :attr_type),
			                                     Tuple{String, String, String}}}()
			for k in eachelement(root)
				if EzXML.nodename(k) == "key"
					id = haskey(k, "id") ? String(k["id"]) : continue
					for_t = haskey(k, "for") ? String(k["for"]) : "all"
					attr_name = haskey(k, "attr.name") ? String(k["attr.name"]) : id
					attr_type = haskey(k, "attr.type") ? String(k["attr.type"]) : "string"
					key_schema[id] = (for_type = for_t, attr_name = attr_name, attr_type = attr_type)
				end
			end

		#	Find Graph Element
			graph_el = nothing
			for child in eachelement(root)
				if EzXML.nodename(child) == "graph"
					graph_el = child
					break
				end
			end
			if graph_el === nothing
				throw(ArgumentError("GraphML file has no <graph> element"))
			end

		#	Determine Edge Default (Directed vs Undirected)
			edgedefault = haskey(graph_el, "edgedefault") ? String(graph_el["edgedefault"]) : "undirected"
			directed = edgedefault == "directed"

		#	Extract Graph-Level Metadata
			metadata_dict = Dict{Symbol, Any}()
			metadata_dict[:directed] = directed
			for child in eachelement(graph_el)
				if EzXML.nodename(child) == "data"
					key_id = haskey(child, "key") ? String(child["key"]) : continue
					if haskey(key_schema, key_id) && key_schema[key_id].for_type == "graph"
						attr_name = key_schema[key_id].attr_name
						attr_type = key_schema[key_id].attr_type
						val = _parse_graphml_value(EzXML.nodecontent(child), attr_type)
						metadata_dict[Symbol(attr_name)] = val
					end
				end
			end

		#	Identify Node and Edge Attribute Keys
			node_attr_keys = [k for (k, v) in key_schema if v.for_type == "node"]
			edge_attr_keys = [k for (k, v) in key_schema if v.for_type == "edge"]

		#	Initialize Node Attribute Storage
			node_ids = String[]
			node_attrs = Dict{Symbol, Vector{Any}}()
			for k in node_attr_keys
				node_attrs[Symbol(key_schema[k].attr_name)] = Any[]
			end

		#	Initialize Edge Attribute Storage
			edge_srcs = String[]
			edge_dsts = String[]
			edge_attrs = Dict{Symbol, Vector{Any}}()
			for k in edge_attr_keys
				edge_attrs[Symbol(key_schema[k].attr_name)] = Any[]
			end

		#	Walk Children of <graph>: Nodes and Edges
			for child in eachelement(graph_el)
				tag = EzXML.nodename(child)

				if tag == "node"
					#	Parse Node ID
						id_str = haskey(child, "id") ? String(child["id"]) : continue
						push!(node_ids, _strip_node_prefix(id_str))

					#	Read Node Attributes
						this_node_data = Dict{String, Any}()
						for sub in eachelement(child)
							if EzXML.nodename(sub) == "data"
								key_id = haskey(sub, "key") ? String(sub["key"]) : continue
								if haskey(key_schema, key_id) && key_schema[key_id].for_type == "node"
									attr_type = key_schema[key_id].attr_type
									attr_name = key_schema[key_id].attr_name
									val = _parse_graphml_value(EzXML.nodecontent(sub), attr_type)
									this_node_data[attr_name] = val
								end
							end
						end

					#	Push Each Attribute (Missing if Absent)
						for k in node_attr_keys
							attr_name = key_schema[k].attr_name
							push!(node_attrs[Symbol(attr_name)], get(this_node_data, attr_name, missing))
						end

				elseif tag == "edge"
					#	Parse Endpoints
						src_str = haskey(child, "source") ? String(child["source"]) : continue
						dst_str = haskey(child, "target") ? String(child["target"]) : continue
						push!(edge_srcs, _strip_node_prefix(src_str))
						push!(edge_dsts, _strip_node_prefix(dst_str))

					#	Read Edge Attributes
						this_edge_data = Dict{String, Any}()
						for sub in eachelement(child)
							if EzXML.nodename(sub) == "data"
								key_id = haskey(sub, "key") ? String(sub["key"]) : continue
								if haskey(key_schema, key_id) && key_schema[key_id].for_type == "edge"
									attr_type = key_schema[key_id].attr_type
									attr_name = key_schema[key_id].attr_name
									val = _parse_graphml_value(EzXML.nodecontent(sub), attr_type)
									this_edge_data[attr_name] = val
								end
							end
						end

					#	Push Each Attribute (Missing if Absent)
						for k in edge_attr_keys
							attr_name = key_schema[k].attr_name
							push!(edge_attrs[Symbol(attr_name)], get(this_edge_data, attr_name, missing))
						end
				end
			end

		#	Build Nodes DataFrame
			nodes_df = DataFrame(id = node_ids)
			for (attr_sym, vals) in node_attrs
				nodes_df[!, attr_sym] = vals
			end

		#	Build Edges DataFrame
			edges_df = DataFrame(src = edge_srcs, dst = edge_dsts)
			for (attr_sym, vals) in edge_attrs
				edges_df[!, attr_sym] = vals
			end

		#	Build Metadata NamedTuple
			metadata = NamedTuple(metadata_dict)

		#	Return All Components
			return (edges = edges_df, nodes = nodes_df, metadata = metadata)
	end

##########################
#   EMPIRICAL DATASETS   #
##########################

#   Setting Import Directory to Data Directory
    cd("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data")

#   SCOTTISH CORPORATE INTERLOCKS (1904-1905)

#   Creating 
    scottish_firms_net = parse_pajek_2mode("Scotland.paj")

#   Sanity checks
    println("Firms:      $(nrow(scottish_firms_net.firms))")           # should print 108
    println("Directors:  $(nrow(scottish_firms_net.directors))")       # should print 136
    println("Edges:      $(nrow(scottish_firms_net.edges))")           # should print ~358
    println("Industries: $(length(scottish_firms_net.industry))")     # should print 108
    println("Capital:    $(length(scottish_firms_net.capital))")      # should print 108

#   Spot-check a few firms
    println("\nFirst 3 firms:")
    println(first(scottish_firms_net.firms, 3))

#   Spot-check first edge
    println("\nFirst edge: firm $(scottish_firms_net.edges.src[1]) ↔ director $( scottish_firms_net.edges.dst[1])")

#   Spot-check industry distribution
    println("\nIndustry distribution:")
    println(countmap(scottish_firms_net.industry))

#   Spot-check capital range
    println("\nCapital range: $(extrema(scottish_firms_net.capital))")

#   Creating a Pojection of the Network: Firms x Firms - Shared Directors
    firm_network = project_firm_firm(scottish_firms_net)

    println("Firm-firm edges:  $(nrow(firm_network.edges))")
    println("Nodes:            $(nrow(firm_network.nodes))")
    println("Max weight:       $(maximum(firm_network.edges.weight))")
    println("Mean weight:      $(round(sum(firm_network.edges.weight) / nrow(firm_network.edges), digits=2))")
    println("Total weight:     $(sum(firm_network.edges.weight))")

#   How many firm pairs share more than 1 director?
    multi_share = sum(firm_network.edges.weight .> 1)
    println("Pairs with >1 shared director: $multi_share")

#   Spot-check first 5 edges with firm names
    println("\nFirst 5 firm-firm edges:")
    for i in 1:5
        src_name = firm_network.nodes.label[firm_network.edges.src[i]]
        dst_name = firm_network.nodes.label[firm_network.edges.dst[i]]
        w = firm_network.edges.weight[i]
        println("  $src_name ↔ $dst_name (weight=$w)")
    end

#   Build metadata
    metadata_weighted = (
        network_name   = "Scotland Interlock 1904",
        source_format  = "pajek_2mode_projection",
        source_file    = "Scotland.paj",
        directed       = false,
        weighted       = true,
        is_binary      = false,
        n_firms        = 108,
        n_directors    = 136,
        n_bipartite_edges = 358,
        projection_method = "B times B-transpose"
    )

#   Write weighted GraphML
    write_graphml(firm_network.edges, firm_network.nodes, metadata_weighted,
                "scotland_interlock_weighted.graphml")

#   Write unweighted variant
    edges_binary = binarize_edges_df(firm_network.edges)
    metadata_binary = merge(metadata_weighted, (weighted = false, is_binary = true))
    write_graphml(edges_binary, firm_network.nodes, metadata_binary,
                "scotland_interlock_unweighted.graphml")

#	Test: Verify GraphML Round-Trip Integrity for Scotland
	function test_scotland_graphml_integrity(weighted_path::String,
	                                          unweighted_path::String,
	                                          pajek_path::String)
		"""
		Args:
			weighted_path::String:   path to weighted GraphML file
			unweighted_path::String: path to unweighted GraphML file
			pajek_path::String:      path to original Pajek file
		Returns:
			NamedTuple: (all_passed::Bool, per_test::Dict)
		Notes:
			Loads both written GraphML files, reproduces the projection from the
			Pajek file, and checks that the loaded contents match the in-memory
			versions. Reports per-check pass/fail status. On label mismatch, prints
			a diagnostic listing of the first few differing labels for inspection.
		"""

		println("=" ^ 70)
		println("Scotland GraphML Round-Trip Integrity Test")
		println("=" ^ 70)

		results = Dict{Symbol, Bool}()

		#	Regenerate Reference Network
			println("\n  Step 1: Regenerate reference network from Pajek...")
			parsed       = parse_pajek_2mode(pajek_path)
			reference    = project_firm_firm(parsed)
			ref_edges    = reference.edges
			ref_nodes    = reference.nodes
			println("    Reference: $(nrow(ref_edges)) edges, $(nrow(ref_nodes)) nodes")

		#	Load Weighted GraphML
			println("\n  Step 2: Load weighted GraphML...")
			loaded_w = load_graphml(weighted_path)
			println("    Loaded:   $(nrow(loaded_w.edges)) edges, $(nrow(loaded_w.nodes)) nodes")

		#	Check Counts Match (Weighted)
			counts_ok_w = (nrow(loaded_w.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_w.nodes) == nrow(ref_nodes))
			results[:weighted_counts] = counts_ok_w
			println("    Counts match: $(counts_ok_w ? "PASS" : "FAIL")")

		#	Check Total Weight Preserved
			ref_total_w = sum(ref_edges.weight)
			loaded_total_w = sum(loaded_w.edges.weight)
			weight_ok_w = isapprox(ref_total_w, loaded_total_w; atol = 1e-9)
			results[:weighted_total_weight] = weight_ok_w
			println("    Total weight: ref = $ref_total_w, loaded = $loaded_total_w  $(weight_ok_w ? "PASS" : "FAIL")")

		#	Check Max Weight Preserved
			max_ok_w = isapprox(maximum(ref_edges.weight), maximum(loaded_w.edges.weight); atol = 1e-9)
			results[:weighted_max_weight] = max_ok_w
			println("    Max weight:   ref = $(maximum(ref_edges.weight)), loaded = $(maximum(loaded_w.edges.weight))  $(max_ok_w ? "PASS" : "FAIL")")

		#	Check Industry Attribute Round-Tripped
			if hasproperty(loaded_w.nodes, :industry)
				industry_ok = sort(collect(skipmissing(loaded_w.nodes.industry))) == sort(collect(skipmissing(ref_nodes.industry)))
				results[:weighted_industry] = industry_ok
				println("    Industry attribute: $(industry_ok ? "PASS" : "FAIL")")
			else
				results[:weighted_industry] = false
				println("    Industry attribute: FAIL (column not found in loaded file)")
			end

		#	Check Capital Attribute Round-Tripped
			if hasproperty(loaded_w.nodes, :capital)
				cap_ok = isapprox(sum(collect(skipmissing(loaded_w.nodes.capital))),
				                  sum(collect(skipmissing(ref_nodes.capital))); atol = 1e-6)
				results[:weighted_capital] = cap_ok
				println("    Capital attribute (sum check): $(cap_ok ? "PASS" : "FAIL")")
			else
				results[:weighted_capital] = false
				println("    Capital attribute: FAIL (column not found)")
			end

		#	Check Label Attribute Round-Tripped (with Diagnostic Output)
			if hasproperty(loaded_w.nodes, :label)
				ref_labels    = sort(string.(ref_nodes.label))
				loaded_labels = sort(string.(loaded_w.nodes.label))
				labels_ok     = ref_labels == loaded_labels
				results[:weighted_labels] = labels_ok
				println("    Label attribute (set equality): $(labels_ok ? "PASS" : "FAIL")")

				#	On Failure, Show First Few Mismatches
					if !labels_ok
						n_diff = sum(ref_labels .!= loaded_labels)
						println("       $n_diff label(s) differ out of $(length(ref_labels))")
						println("       First 5 mismatches (ref → loaded):")
						shown = 0
						for i in eachindex(ref_labels)
							if ref_labels[i] != loaded_labels[i] && shown < 5
								#	repr() makes whitespace and special characters visible
									println("         [$i]  ref:    $(repr(ref_labels[i]))")
									println("              loaded: $(repr(loaded_labels[i]))")
								#	Show byte-level comparison
									ref_bytes    = collect(codeunits(ref_labels[i]))
									loaded_bytes = collect(codeunits(loaded_labels[i]))
									if ref_bytes != loaded_bytes
										println("              ref bytes:    $(ref_bytes[1:min(end, 30)])")
										println("              loaded bytes: $(loaded_bytes[1:min(end, 30)])")
									end
								shown += 1
							end
						end
					end
			else
				results[:weighted_labels] = false
				println("    Label attribute: FAIL (column not found)")
			end

		#	Check Metadata Preserved
			meta = loaded_w.metadata
			meta_ok = haskey(meta, :network_name) && haskey(meta, :weighted) &&
			          meta[:weighted] === true && meta[:directed] === false
			results[:weighted_metadata] = meta_ok
			println("    Metadata (network_name, weighted=true, directed=false): $(meta_ok ? "PASS" : "FAIL")")
			println("    Loaded metadata: $(meta)")

		#	Load Unweighted GraphML
			println("\n  Step 3: Load unweighted GraphML...")
			loaded_u = load_graphml(unweighted_path)
			println("    Loaded:   $(nrow(loaded_u.edges)) edges, $(nrow(loaded_u.nodes)) nodes")

		#	Check Counts Match (Unweighted)
			counts_ok_u = (nrow(loaded_u.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_u.nodes) == nrow(ref_nodes))
			results[:unweighted_counts] = counts_ok_u
			println("    Counts match: $(counts_ok_u ? "PASS" : "FAIL")")

		#	Check Binarization
			all_ones = all(loaded_u.edges.weight .== 1.0)
			results[:unweighted_binarized] = all_ones
			println("    All weights = 1.0: $(all_ones ? "PASS" : "FAIL")")

		#	Check Metadata Reflects Unweighted
			meta_u = loaded_u.metadata
			meta_u_ok = haskey(meta_u, :is_binary) && meta_u[:is_binary] === true &&
			            meta_u[:weighted] === false
			results[:unweighted_metadata] = meta_u_ok
			println("    Metadata (is_binary=true, weighted=false): $(meta_u_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = all(values(results))
			println("\n" * "=" ^ 70)
			println("SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			for (k, v) in results
				println("  $k: $(v ? "PASS" : "FAIL")")
			end
			println("=" ^ 70)

		return (all_passed = all_passed, per_test = results)
	end

	results = test_scotland_graphml_integrity(
		"scotland_interlock_weighted.graphml",
		"scotland_interlock_unweighted.graphml",
		"Scotland.paj"
	)

#	MARVEL UNIVERSE (Alberich, Miró-Julià, & Roselló 2002)

#   Creating
    marvel_bipartite = parse_pajek_marvel("Marvel_Universe.paj")

#   Sanity checks
    println("Heroes:     $(nrow(marvel_bipartite.heroes))")           # should print 6486
    println("Comics:     $(nrow(marvel_bipartite.comics))")           # should print 12942
    println("Edges:      $(nrow(marvel_bipartite.edges))")            # should print ~96663
    println("Vertex types: $(length(marvel_bipartite.vertex_type))")  # should print 19428

#   Spot-check a few heroes
    println("\nFirst 3 heroes:")
    println(first(marvel_bipartite.heroes, 3))

#   Spot-check a few comics
    println("\nFirst 3 comics:")
    println(first(marvel_bipartite.comics, 3))

#   Spot-check first edge
    println("\nFirst edge: hero $(marvel_bipartite.edges.src[1]) ↔ comic $(marvel_bipartite.edges.dst[1])")

#   Verify Vertex Type Partition Matches Header-Derived Mode Assignment
    if !isempty(marvel_bipartite.vertex_type)
        n_heroes = nrow(marvel_bipartite.heroes)
        partition_heroes_ok = all(marvel_bipartite.vertex_type[1:n_heroes] .== 1)
        partition_comics_ok = all(marvel_bipartite.vertex_type[(n_heroes + 1):end] .== 2)
        println("\nVertex_Type.clu consistency check:")
        println("  Heroes labeled 1 in partition: $(partition_heroes_ok ? "PASS" : "FAIL")")
        println("  Comics labeled 2 in partition: $(partition_comics_ok ? "PASS" : "FAIL")")
    end

#   Creating a Projection of the Network: Heroes x Heroes - Shared Comics (Threshold ≥ 2)
    marvel_network = project_marvel_characters(marvel_bipartite; min_weight = 2)

    println("\nHero-hero edges:  $(nrow(marvel_network.edges))")
    println("Nodes:            $(nrow(marvel_network.nodes))")
    println("Max weight:       $(maximum(marvel_network.edges.weight))")
    println("Mean weight:      $(round(sum(marvel_network.edges.weight) / nrow(marvel_network.edges), digits=2))")
    println("Total weight:     $(sum(marvel_network.edges.weight))")

#   Network density (undirected, post-threshold)
    n_heroes_proj = nrow(marvel_network.nodes)
    n_possible    = n_heroes_proj * (n_heroes_proj - 1) ÷ 2
    density       = nrow(marvel_network.edges) / n_possible
    println("Density:          $(round(density, digits=5))  ($(nrow(marvel_network.edges)) / $n_possible)")

#   Isolates after thresholding (heroes with no surviving co-appearance edge)
    nodes_with_edges = union(marvel_network.edges.src, marvel_network.edges.dst)
    n_isolates       = n_heroes_proj - length(nodes_with_edges)
    println("Isolates after threshold: $n_isolates / $n_heroes_proj heroes ($(round(100 * n_isolates / n_heroes_proj, digits=1))%)")

#   How many hero pairs share more than 5 comics? More than 20?
    hi_share_5  = sum(marvel_network.edges.weight .> 5)
    hi_share_20 = sum(marvel_network.edges.weight .> 20)
    println("Pairs co-appearing in >5 comics:  $hi_share_5")
    println("Pairs co-appearing in >20 comics: $hi_share_20")

#   Spot-check first 5 edges with hero names
    println("\nFirst 5 hero-hero edges:")
    for i in 1:5
        src_name = marvel_network.nodes.label[marvel_network.edges.src[i]]
        dst_name = marvel_network.nodes.label[marvel_network.edges.dst[i]]
        w = marvel_network.edges.weight[i]
        println("  $src_name ↔ $dst_name (weight=$w)")
    end

#   Top 5 highest-weight edges (most collaborative pairs)
    println("\nTop 5 highest-weight hero-hero edges:")
    top_idx = sortperm(marvel_network.edges.weight; rev = true)[1:5]
    for i in top_idx
        src_name = marvel_network.nodes.label[marvel_network.edges.src[i]]
        dst_name = marvel_network.nodes.label[marvel_network.edges.dst[i]]
        w = marvel_network.edges.weight[i]
        println("  $src_name ↔ $dst_name (weight=$w)")
    end

#   Build metadata
    metadata_weighted = (
        network_name      = "Marvel Universe Collaboration 2002",
        source_format     = "pajek_2mode_projection",
        source_file       = "Marvel_Universe.paj",
        directed          = false,
        weighted          = true,
        is_binary         = false,
        n_heroes          = nrow(marvel_bipartite.heroes),
        n_comics          = nrow(marvel_bipartite.comics),
        n_bipartite_edges = nrow(marvel_bipartite.edges),
        projection_method = "B times B-transpose, threshold weight >= 2",
        min_coappearance  = 2,
        reference         = "Alberich, Miro-Julia, & Rossello 2002 (arXiv:cond-mat/0202174)"
    )

#   Write weighted GraphML
    write_graphml(marvel_network.edges, marvel_network.nodes, metadata_weighted,
                "GraphML_Test_Networks/marvel_universe_weighted.graphml")

#   Write unweighted variant (post-threshold edge set, weights flattened to 1.0)
    edges_binary    = binarize_edges_df(marvel_network.edges)
    metadata_binary = merge(metadata_weighted, (weighted = false, is_binary = true))
    write_graphml(edges_binary, marvel_network.nodes, metadata_binary,
                "GraphML_Test_Networks/marvel_universe_unweighted.graphml")

#	Test: Verify GraphML Round-Trip Integrity for Marvel
	function test_marvel_graphml_integrity(weighted_path::String,
	                                        unweighted_path::String,
	                                        pajek_path::String;
	                                        min_weight::Int = 2)
		"""
		Args:
			weighted_path::String:   path to weighted GraphML file
			unweighted_path::String: path to unweighted GraphML file
			pajek_path::String:      path to original Pajek file
			min_weight::Int:         co-appearance threshold used at projection time
				(default = 2; must match what was passed to project_marvel_characters)
		Returns:
			NamedTuple: (all_passed::Bool, per_test::Dict)
		Notes:
			Loads both written GraphML files, reproduces the projection from the
			Pajek file, and checks that the loaded contents match the in-memory
			versions. Reports per-check pass/fail status. On label mismatch, prints
			a diagnostic listing of the first few differing labels for inspection.
		"""

		println("=" ^ 70)
		println("Marvel GraphML Round-Trip Integrity Test")
		println("=" ^ 70)

		results = Dict{Symbol, Bool}()

		#	Regenerate Reference Network
			println("\n  Step 1: Regenerate reference network from Pajek...")
			parsed    = parse_pajek_marvel(pajek_path)
			reference = project_marvel_characters(parsed; min_weight = min_weight)
			ref_edges = reference.edges
			ref_nodes = reference.nodes
			println("    Reference: $(nrow(ref_edges)) edges, $(nrow(ref_nodes)) nodes")

		#	Load Weighted GraphML
			println("\n  Step 2: Load weighted GraphML...")
			loaded_w = load_graphml(weighted_path)
			println("    Loaded:   $(nrow(loaded_w.edges)) edges, $(nrow(loaded_w.nodes)) nodes")

		#	Check Counts Match (Weighted)
			counts_ok_w = (nrow(loaded_w.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_w.nodes) == nrow(ref_nodes))
			results[:weighted_counts] = counts_ok_w
			println("    Counts match: $(counts_ok_w ? "PASS" : "FAIL")")

		#	Check Total Weight Preserved
			ref_total_w    = sum(ref_edges.weight)
			loaded_total_w = sum(loaded_w.edges.weight)
			weight_ok_w    = isapprox(ref_total_w, loaded_total_w; atol = 1e-9)
			results[:weighted_total_weight] = weight_ok_w
			println("    Total weight: ref = $ref_total_w, loaded = $loaded_total_w  $(weight_ok_w ? "PASS" : "FAIL")")

		#	Check Max Weight Preserved
			max_ok_w = isapprox(maximum(ref_edges.weight), maximum(loaded_w.edges.weight); atol = 1e-9)
			results[:weighted_max_weight] = max_ok_w
			println("    Max weight:   ref = $(maximum(ref_edges.weight)), loaded = $(maximum(loaded_w.edges.weight))  $(max_ok_w ? "PASS" : "FAIL")")

		#	Check Threshold Was Respected (No Weights Below min_weight in Loaded File)
			threshold_ok = minimum(loaded_w.edges.weight) ≥ min_weight
			results[:weighted_threshold] = threshold_ok
			println("    Min weight ≥ $min_weight (threshold respected): $(threshold_ok ? "PASS" : "FAIL")  (observed min = $(minimum(loaded_w.edges.weight)))")

		#	Check Label Attribute Round-Tripped (with Diagnostic Output)
			if hasproperty(loaded_w.nodes, :label)
				ref_labels    = sort(string.(ref_nodes.label))
				loaded_labels = sort(string.(loaded_w.nodes.label))
				labels_ok     = ref_labels == loaded_labels
				results[:weighted_labels] = labels_ok
				println("    Label attribute (set equality): $(labels_ok ? "PASS" : "FAIL")")

				#	On Failure, Show First Few Mismatches
					if !labels_ok
						n_diff = sum(ref_labels .!= loaded_labels)
						println("       $n_diff label(s) differ out of $(length(ref_labels))")
						println("       First 5 mismatches (ref → loaded):")
						shown = 0
						for i in eachindex(ref_labels)
							if ref_labels[i] != loaded_labels[i] && shown < 5
								#	repr() makes whitespace and special characters visible
									println("         [$i]  ref:    $(repr(ref_labels[i]))")
									println("              loaded: $(repr(loaded_labels[i]))")
								#	Show byte-level comparison
									ref_bytes    = collect(codeunits(ref_labels[i]))
									loaded_bytes = collect(codeunits(loaded_labels[i]))
									if ref_bytes != loaded_bytes
										println("              ref bytes:    $(ref_bytes[1:min(end, 30)])")
										println("              loaded bytes: $(loaded_bytes[1:min(end, 30)])")
									end
								shown += 1
							end
						end
					end
			else
				results[:weighted_labels] = false
				println("    Label attribute: FAIL (column not found)")
			end

		#	Check Metadata Preserved
			meta = loaded_w.metadata
			meta_ok = haskey(meta, :network_name) && haskey(meta, :weighted) &&
			          meta[:weighted] === true && meta[:directed] === false &&
			          haskey(meta, :min_coappearance) && meta[:min_coappearance] == min_weight
			results[:weighted_metadata] = meta_ok
			println("    Metadata (network_name, weighted=true, directed=false, min_coappearance=$min_weight): $(meta_ok ? "PASS" : "FAIL")")
			println("    Loaded metadata: $(meta)")

		#	Load Unweighted GraphML
			println("\n  Step 3: Load unweighted GraphML...")
			loaded_u = load_graphml(unweighted_path)
			println("    Loaded:   $(nrow(loaded_u.edges)) edges, $(nrow(loaded_u.nodes)) nodes")

		#	Check Counts Match (Unweighted)
			counts_ok_u = (nrow(loaded_u.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_u.nodes) == nrow(ref_nodes))
			results[:unweighted_counts] = counts_ok_u
			println("    Counts match (same edge set as weighted variant): $(counts_ok_u ? "PASS" : "FAIL")")

		#	Check Binarization
			all_ones = all(loaded_u.edges.weight .== 1.0)
			results[:unweighted_binarized] = all_ones
			println("    All weights = 1.0: $(all_ones ? "PASS" : "FAIL")")

		#	Check Metadata Reflects Unweighted
			meta_u    = loaded_u.metadata
			meta_u_ok = haskey(meta_u, :is_binary) && meta_u[:is_binary] === true &&
			            meta_u[:weighted] === false
			results[:unweighted_metadata] = meta_u_ok
			println("    Metadata (is_binary=true, weighted=false): $(meta_u_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = all(values(results))
			println("\n" * "=" ^ 70)
			println("SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			for (k, v) in results
				println("  $k: $(v ? "PASS" : "FAIL")")
			end
			println("=" ^ 70)

		return (all_passed = all_passed, per_test = results)
	end

	marvel_results = test_marvel_graphml_integrity(
		"GraphML_Test_Networks/marvel_universe_weighted.graphml",
		"GraphML_Test_Networks/marvel_universe_unweighted.graphml",
		"Marvel_Universe.paj"
	)

#   MORENO DIRECTED HIGH SCHOOL FRIENDSHIP NETWORK (1957-1958)

#   Parsing File
    moreno = parse_konect("moreno_highschool/out_edgelist.moreno_highschool_highschool")

    println("Edges declared in header: $(moreno.properties.n_edges_declared)")
    println("Edges parsed:             $(nrow(moreno.edges))")
    println("Nodes:                    $(nrow(moreno.nodes))")
    println("Directed:                 $(moreno.properties.directed)")
    println("Weighted:                 $(moreno.properties.weighted)")
    println("Weight values:            $(sort(unique(moreno.edges.weight)))")

#   Spot check
    println("\nFirst 5 edges:")
    println(first(moreno.edges, 5))

#	Build Weighted Metadata
	metadata_weighted = (
		network_name   = "Moreno High School 1934",
		source_format  = "konect_edge_list",
		source_file    = "out_edgelist.moreno_highschool_highschool",
		directed       = moreno.properties.directed,
		weighted       = moreno.properties.weighted,
		is_binary      = false,
		n_nodes        = nrow(moreno.nodes),
		n_edges        = nrow(moreno.edges),
		weight_meaning = "friendship choice rank (1 = first choice, 2 = second choice)"
	)

#	Write Weighted GraphML
	write_graphml(moreno.edges, moreno.nodes, metadata_weighted,
	              "moreno_highschool_weighted.graphml")

#	Write Unweighted Variant
	edges_binary = binarize_edges_df(moreno.edges)
	metadata_binary = merge(metadata_weighted, (weighted = false, is_binary = true))
	write_graphml(edges_binary, moreno.nodes, metadata_binary,
	              "moreno_highschool_unweighted.graphml")

#	Test: Verify GraphML Round-Trip Integrity for Moreno High School
	function test_moreno_graphml_integrity(weighted_path::String,
	                                        unweighted_path::String,
	                                        konect_path::String)
		"""
		Args:
			weighted_path::String:   path to weighted GraphML file
			unweighted_path::String: path to unweighted GraphML file
			konect_path::String:     path to original Konect edge list
		Returns:
			NamedTuple: (all_passed::Bool, per_test::Dict)
		Notes:
			Loads both written GraphML files, reproduces the parsing from the
			Konect file, and checks that the loaded contents match the in-memory
			versions. Same verification pattern as test_scotland_graphml_integrity
			but for the simpler Moreno schema (no node attributes beyond label).
		"""

		println("=" ^ 70)
		println("Moreno High School GraphML Round-Trip Integrity Test")
		println("=" ^ 70)

		results = Dict{Symbol, Bool}()

		#	Regenerate Reference Network
			println("\n  Step 1: Regenerate reference network from Konect file...")
			parsed       = parse_konect(konect_path)
			ref_edges    = parsed.edges
			ref_nodes    = parsed.nodes
			println("    Reference: $(nrow(ref_edges)) edges, $(nrow(ref_nodes)) nodes, directed=$(parsed.properties.directed)")

		#	Load Weighted GraphML
			println("\n  Step 2: Load weighted GraphML...")
			loaded_w = load_graphml(weighted_path)
			println("    Loaded:   $(nrow(loaded_w.edges)) edges, $(nrow(loaded_w.nodes)) nodes")

		#	Check Counts Match (Weighted)
			counts_ok_w = (nrow(loaded_w.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_w.nodes) == nrow(ref_nodes))
			results[:weighted_counts] = counts_ok_w
			println("    Counts match: $(counts_ok_w ? "PASS" : "FAIL")")

		#	Check Total Weight Preserved
			ref_total_w    = sum(ref_edges.weight)
			loaded_total_w = sum(loaded_w.edges.weight)
			weight_ok_w    = isapprox(ref_total_w, loaded_total_w; atol = 1e-9)
			results[:weighted_total_weight] = weight_ok_w
			println("    Total weight: ref = $ref_total_w, loaded = $loaded_total_w  $(weight_ok_w ? "PASS" : "FAIL")")

		#	Check Weight Distribution (1.0 vs 2.0)
			ref_weights    = sort(unique(ref_edges.weight))
			loaded_weights = sort(unique(loaded_w.edges.weight))
			weights_ok     = ref_weights == loaded_weights
			results[:weighted_weight_values] = weights_ok
			println("    Weight values:  ref = $ref_weights, loaded = $loaded_weights  $(weights_ok ? "PASS" : "FAIL")")

		#	Check Edge Set Identity
			ref_set    = Set((row.src, row.dst, row.weight) for row in eachrow(ref_edges))
			loaded_set = Set((row.src, row.dst, row.weight) for row in eachrow(loaded_w.edges))
			edges_ok   = ref_set == loaded_set
			results[:weighted_edge_identity] = edges_ok
			println("    Edge set identity (src, dst, weight): $(edges_ok ? "PASS" : "FAIL")")

		#	Check Metadata Preserved
			meta = loaded_w.metadata
			meta_ok = haskey(meta, :network_name) &&
			          haskey(meta, :weighted) && meta[:weighted] === true &&
			          haskey(meta, :directed) && meta[:directed] === true
			results[:weighted_metadata] = meta_ok
			println("    Metadata (network_name, weighted=true, directed=true): $(meta_ok ? "PASS" : "FAIL")")
			println("    Loaded metadata: $(meta)")

		#	Load Unweighted GraphML
			println("\n  Step 3: Load unweighted GraphML...")
			loaded_u = load_graphml(unweighted_path)
			println("    Loaded:   $(nrow(loaded_u.edges)) edges, $(nrow(loaded_u.nodes)) nodes")

		#	Check Counts Match (Unweighted)
			counts_ok_u = (nrow(loaded_u.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_u.nodes) == nrow(ref_nodes))
			results[:unweighted_counts] = counts_ok_u
			println("    Counts match: $(counts_ok_u ? "PASS" : "FAIL")")

		#	Check Binarization
			all_ones = all(loaded_u.edges.weight .== 1.0)
			results[:unweighted_binarized] = all_ones
			println("    All weights = 1.0: $(all_ones ? "PASS" : "FAIL")")

		#	Check Edge Set Preserved (Ignoring Weights)
			ref_pairs    = Set((row.src, row.dst) for row in eachrow(ref_edges))
			loaded_pairs = Set((row.src, row.dst) for row in eachrow(loaded_u.edges))
			pairs_ok     = ref_pairs == loaded_pairs
			results[:unweighted_edge_pairs] = pairs_ok
			println("    Edge pair set identity (src, dst): $(pairs_ok ? "PASS" : "FAIL")")

		#	Check Metadata Reflects Unweighted
			meta_u = loaded_u.metadata
			meta_u_ok = haskey(meta_u, :is_binary) && meta_u[:is_binary] === true &&
			            haskey(meta_u, :weighted) && meta_u[:weighted] === false &&
			            haskey(meta_u, :directed) && meta_u[:directed] === true
			results[:unweighted_metadata] = meta_u_ok
			println("    Metadata (is_binary=true, weighted=false, directed=true): $(meta_u_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = all(values(results))
			println("\n" * "=" ^ 70)
			println("SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			for (k, v) in results
				println("  $k: $(v ? "PASS" : "FAIL")")
			end
			println("=" ^ 70)

		return (all_passed = all_passed, per_test = results)
	end

	results = test_moreno_graphml_integrity(
		"moreno_highschool_weighted.graphml",
		"moreno_highschool_unweighted.graphml",
		"moreno_highschool/out_edgelist.moreno_highschool_highschool"
	)

#   BALIKATAN 2022

#   Parse Files
    balikatan = parse_balikatan_ora("Balikatan_2022_Processed-nodeset-Agent.csv", 
                                    "Balikatan_2022_Processed-network-Agent x Agent - All Communication.csv")

    println("Edges:    $(nrow(balikatan.edges))")
    println("Nodes:    $(nrow(balikatan.nodes))")
    println("Directed: $(balikatan.properties.directed)")
    println("Weighted: $(balikatan.properties.weighted)")
    println("Weight range: $(extrema(balikatan.edges.weight))")
    println("Total weight: $(sum(balikatan.edges.weight))")

    println("\nFirst 3 edges:")
    println(first(balikatan.edges, 3))

    println("\nFirst 3 nodes (id and label):")
    println(first(balikatan.nodes[:, [:id, :label]], 3))

    println("\nAll node columns:")
    println(names(balikatan.nodes))

#   Spot check: bot proportion
    println("\nBot count: $(sum(skipmissing(balikatan.nodes.is_bot)))")

#   Language distribution
    println("\nTop 5 languages:")
    lang_counts = countmap(string.(skipmissing(balikatan.nodes.language)))
    for (lang, n) in sort(collect(lang_counts), by = x -> -x[2])[1:min(5, end)]
        println("  $lang: $n")
    end

#	Build Weighted Metadata
	metadata_weighted = (
		network_name      = "Balikatan 2022 Communication Network",
		source_format     = "ora_csv",
		source_file       = "Balikatan_2022_Processed-network-Agent_x_Agent_-_All_Communication.csv",
		directed          = true,
		weighted          = true,
		is_binary         = false,
		n_nodes           = nrow(balikatan.nodes),
		n_edges           = nrow(balikatan.edges),
		weight_meaning    = "number of communications from source to target",
		platform          = "Twitter/X",
		event             = "Balikatan 2022 US-Philippines joint military exercise"
	)

#	Write Weighted GraphML
	write_graphml(balikatan.edges, balikatan.nodes, metadata_weighted,
	              "balikatan_2022_weighted.graphml")

#	Write Unweighted Variant
	edges_binary = binarize_edges_df(balikatan.edges)
	metadata_binary = merge(metadata_weighted, (weighted = false, is_binary = true))
	write_graphml(edges_binary, balikatan.nodes, metadata_binary,
	              "balikatan_2022_unweighted.graphml")

#	Test: Verify GraphML Round-Trip Integrity for Balikatan
	function test_balikatan_graphml_integrity(weighted_path::String,
	                                            unweighted_path::String,
	                                            nodes_csv_path::String,
	                                            edges_csv_path::String)
		"""
		Args:
			weighted_path::String:    path to weighted GraphML file
			unweighted_path::String:  path to unweighted GraphML file
			nodes_csv_path::String:   path to original ORA nodeset CSV
			edges_csv_path::String:   path to original ORA edges CSV
		Returns:
			NamedTuple: (all_passed::Bool, per_test::Dict)
		Notes:
			Loads both GraphML files, reproduces parsing from the ORA CSVs,
			and checks that the loaded contents match the in-memory versions.
			Includes attribute-preservation checks for the rich node schema
			(role, is_bot, language, etc.).
		"""

		println("=" ^ 70)
		println("Balikatan 2022 GraphML Round-Trip Integrity Test")
		println("=" ^ 70)

		results = Dict{Symbol, Bool}()

		#	Regenerate Reference Network
			println("\n  Step 1: Regenerate reference network from ORA CSVs...")
			reference = parse_balikatan_ora(nodes_csv_path, edges_csv_path)
			ref_edges = reference.edges
			ref_nodes = reference.nodes
			println("    Reference: $(nrow(ref_edges)) edges, $(nrow(ref_nodes)) nodes")

		#	Load Weighted GraphML
			println("\n  Step 2: Load weighted GraphML...")
			loaded_w = load_graphml(weighted_path)
			println("    Loaded:   $(nrow(loaded_w.edges)) edges, $(nrow(loaded_w.nodes)) nodes")

		#	Check Counts Match (Weighted)
			counts_ok_w = (nrow(loaded_w.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_w.nodes) == nrow(ref_nodes))
			results[:weighted_counts] = counts_ok_w
			println("    Counts match: $(counts_ok_w ? "PASS" : "FAIL")")

		#	Check Total Weight Preserved
			ref_total_w    = sum(ref_edges.weight)
			loaded_total_w = sum(loaded_w.edges.weight)
			weight_ok_w    = isapprox(ref_total_w, loaded_total_w; atol = 1e-9)
			results[:weighted_total_weight] = weight_ok_w
			println("    Total weight: ref = $ref_total_w, loaded = $loaded_total_w  $(weight_ok_w ? "PASS" : "FAIL")")

		#	Check Max Weight Preserved
			max_ok_w = isapprox(maximum(ref_edges.weight), maximum(loaded_w.edges.weight); atol = 1e-9)
			results[:weighted_max_weight] = max_ok_w
			println("    Max weight:   ref = $(maximum(ref_edges.weight)), loaded = $(maximum(loaded_w.edges.weight))  $(max_ok_w ? "PASS" : "FAIL")")

		#	Check Bot Count Preserved
			if hasproperty(loaded_w.nodes, :is_bot)
				ref_bot_count    = sum(skipmissing(ref_nodes.is_bot))
				loaded_bot_count = sum(skipmissing(loaded_w.nodes.is_bot))
				bot_ok = ref_bot_count == loaded_bot_count
				results[:weighted_bot_count] = bot_ok
				println("    Bot count:    ref = $ref_bot_count, loaded = $loaded_bot_count  $(bot_ok ? "PASS" : "FAIL")")
			else
				results[:weighted_bot_count] = false
				println("    Bot count: FAIL (is_bot column not found)")
			end

		#	Check Language Distribution Preserved
			if hasproperty(loaded_w.nodes, :language)
				ref_langs    = sort(collect(skipmissing(ref_nodes.language)))
				loaded_langs = sort(collect(skipmissing(loaded_w.nodes.language)))
				lang_ok = ref_langs == loaded_langs
				results[:weighted_language] = lang_ok
				println("    Language list: $(lang_ok ? "PASS" : "FAIL")")
			else
				results[:weighted_language] = false
				println("    Language: FAIL (column not found)")
			end

		#	Check Follower Counts Sum Preserved
			if hasproperty(loaded_w.nodes, :number_followers)
				ref_followers    = sum(skipmissing(ref_nodes.number_followers))
				loaded_followers = sum(skipmissing(loaded_w.nodes.number_followers))
				followers_ok = ref_followers == loaded_followers
				results[:weighted_followers] = followers_ok
				println("    Total followers: ref = $ref_followers, loaded = $loaded_followers  $(followers_ok ? "PASS" : "FAIL")")
			else
				results[:weighted_followers] = false
				println("    Followers: FAIL (column not found)")
			end

		#	Check Label Set Preserved (Sample of Long Twitter Names)
			if hasproperty(loaded_w.nodes, :label)
				ref_labels    = sort(string.(ref_nodes.label))
				loaded_labels = sort(string.(loaded_w.nodes.label))
				labels_ok     = ref_labels == loaded_labels
				results[:weighted_labels] = labels_ok
				println("    Label set: $(labels_ok ? "PASS" : "FAIL")")
				if !labels_ok
					n_diff = sum(ref_labels .!= loaded_labels)
					println("       $n_diff label(s) differ out of $(length(ref_labels))")
				end
			else
				results[:weighted_labels] = false
				println("    Label set: FAIL (column not found)")
			end

		#	Check Metadata Preserved
			meta = loaded_w.metadata
			meta_ok = haskey(meta, :network_name) &&
			          haskey(meta, :directed) && meta[:directed] === true &&
			          haskey(meta, :weighted) && meta[:weighted] === true
			results[:weighted_metadata] = meta_ok
			println("    Metadata (network_name, directed=true, weighted=true): $(meta_ok ? "PASS" : "FAIL")")

		#	Load Unweighted GraphML
			println("\n  Step 3: Load unweighted GraphML...")
			loaded_u = load_graphml(unweighted_path)
			println("    Loaded:   $(nrow(loaded_u.edges)) edges, $(nrow(loaded_u.nodes)) nodes")

		#	Check Counts Match (Unweighted)
			counts_ok_u = (nrow(loaded_u.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_u.nodes) == nrow(ref_nodes))
			results[:unweighted_counts] = counts_ok_u
			println("    Counts match: $(counts_ok_u ? "PASS" : "FAIL")")

		#	Check Binarization
			all_ones = all(loaded_u.edges.weight .== 1.0)
			results[:unweighted_binarized] = all_ones
			println("    All weights = 1.0: $(all_ones ? "PASS" : "FAIL")")

		#	Check Unweighted Metadata
			meta_u = loaded_u.metadata
			meta_u_ok = haskey(meta_u, :is_binary) && meta_u[:is_binary] === true &&
			            haskey(meta_u, :weighted) && meta_u[:weighted] === false &&
			            haskey(meta_u, :directed) && meta_u[:directed] === true
			results[:unweighted_metadata] = meta_u_ok
			println("    Metadata (is_binary=true, weighted=false, directed=true): $(meta_u_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = all(values(results))
			println("\n" * "=" ^ 70)
			println("SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			for (k, v) in results
				println("  $k: $(v ? "PASS" : "FAIL")")
			end
			println("=" ^ 70)

		return (all_passed = all_passed, per_test = results)
	end

	results = test_balikatan_graphml_integrity(
		"balikatan_2022_weighted.graphml",
		"balikatan_2022_unweighted.graphml",
		joinpath("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data",
		         "Balikatan_2022_Processed-nodeset-Agent.csv"),
		joinpath("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data",
		         "Balikatan_2022_Processed-network-Agent x Agent - All Communication.csv")
	)

#   TOLEDO CRIME NETWORKS

#   Parse Files
    toledo = parse_toledo_crime("Crime_Networks (Toledo et al., 2023)")

    println("Edges in collapsed network: $(nrow(toledo.edges))")
    println("Nodes in collapsed network: $(nrow(toledo.nodes))")
    println("Layer counts:")
    for (layer, n) in sort(collect(toledo.layer_counts), by = x -> -x[2])
        println("  $layer: $n edges")
    end

    println("\nWeight distribution (number of layers each edge appears in):")
    using StatsBase
    weight_counts = countmap(Int.(toledo.edges.weight))
    for w in sort(collect(keys(weight_counts)))
        println("  $w layers: $(weight_counts[w]) edges")
    end

    println("\nMax weight: $(maximum(toledo.edges.weight))")
    println("Mean weight: $(round(sum(toledo.edges.weight) / nrow(toledo.edges), digits=2))")

#	Cross-check against the crime_network.csv summary
	println("\n=== Cross-check against edges_crime_network.csv ===")
	summary_path = "/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/Crime_Networks (Toledo et al., 2023)/edges_crime_network.csv"
	summary_edges = Set(_read_toledo_layer(summary_path))
	collapsed_edges = Set(
		(Int(toledo.edges.src[i]), Int(toledo.edges.dst[i]))
		for i in 1:nrow(toledo.edges)
	)
	println("Summary file edges:    $(length(summary_edges))")
	println("Our collapsed edges:   $(length(collapsed_edges))")
	println("Equal sets:            $(summary_edges == collapsed_edges)")
	in_summary_not_ours = setdiff(summary_edges, collapsed_edges)
	in_ours_not_summary = setdiff(collapsed_edges, summary_edges)
	println("In summary but not ours: $(length(in_summary_not_ours))")
	println("In ours but not summary: $(length(in_ours_not_summary))")
	if !isempty(in_summary_not_ours)
		println("  Sample missing edges: $(collect(in_summary_not_ours)[1:min(5, end)])")
	end
	if !isempty(in_ours_not_summary)
		println("  Sample extra edges: $(collect(in_ours_not_summary)[1:min(5, end)])")
	end

#	Build Weighted Metadata
	metadata_weighted = (
		network_name      = "Toledo Criminal Cooperation Network 2023",
		source_format     = "toledo_layered_csv",
		source_file       = "edges_crime_network.csv (verified against 7-layer collapse)",
		directed          = false,
		weighted          = true,
		is_binary         = false,
		n_nodes           = nrow(toledo.nodes),
		n_edges           = nrow(toledo.edges),
		weight_meaning    = "number of crime categories in which the two offenders co-participated (1..7)",
		n_layers          = 7,
		layers_included   = "homicide, kidnapping, theft, drug_trafficking, arms_trafficking, weapon_carrying, ideological_falsehood",
		excluded_layers   = "target_network (offender-victim relationships, not cooperation)"
	)

#	Write Weighted GraphML
	write_graphml(toledo.edges, toledo.nodes, metadata_weighted,
	              "toledo_crime_weighted.graphml")

#	Write Unweighted Variant
	edges_binary = binarize_edges_df(toledo.edges)
	metadata_binary = merge(metadata_weighted, (weighted = false, is_binary = true))
	write_graphml(edges_binary, toledo.nodes, metadata_binary,
	              "toledo_crime_unweighted.graphml")

#	Test: Verify GraphML Round-Trip Integrity for Toledo Crime Network
	function test_toledo_graphml_integrity(weighted_path::String,
	                                        unweighted_path::String,
	                                        layer_dir::String)
		"""
		Args:
			weighted_path::String:   path to weighted GraphML file
			unweighted_path::String: path to unweighted GraphML file
			layer_dir::String:       path to Toledo layer CSV directory
		Returns:
			NamedTuple: (all_passed::Bool, per_test::Dict)
		Notes:
			Loads both written GraphML files, reproduces the 7-layer collapse
			from the original CSVs, and checks that the loaded contents match
			the in-memory versions.

			Includes a structural check against edges_crime_network.csv (the
			summary file distributed with the dataset) to verify the binarized
			edge set matches what Toledo et al. published as the union network.
		"""

		println("=" ^ 70)
		println("Toledo Crime Network GraphML Round-Trip Integrity Test")
		println("=" ^ 70)

		results = Dict{Symbol, Bool}()

		#	Regenerate Reference Network
			println("\n  Step 1: Regenerate reference network from 7 layer CSVs...")
			reference = parse_toledo_crime(layer_dir)
			ref_edges = reference.edges
			ref_nodes = reference.nodes
			println("    Reference: $(nrow(ref_edges)) edges, $(nrow(ref_nodes)) nodes")

		#	Load Weighted GraphML
			println("\n  Step 2: Load weighted GraphML...")
			loaded_w = load_graphml(weighted_path)
			println("    Loaded:   $(nrow(loaded_w.edges)) edges, $(nrow(loaded_w.nodes)) nodes")

		#	Check Counts Match (Weighted)
			counts_ok_w = (nrow(loaded_w.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_w.nodes) == nrow(ref_nodes))
			results[:weighted_counts] = counts_ok_w
			println("    Counts match: $(counts_ok_w ? "PASS" : "FAIL")")

		#	Check Total Weight Preserved
			ref_total_w    = sum(ref_edges.weight)
			loaded_total_w = sum(loaded_w.edges.weight)
			weight_ok_w    = isapprox(ref_total_w, loaded_total_w; atol = 1e-9)
			results[:weighted_total_weight] = weight_ok_w
			println("    Total weight: ref = $ref_total_w, loaded = $loaded_total_w  $(weight_ok_w ? "PASS" : "FAIL")")

		#	Check Weight Distribution Preserved
			ref_weight_counts    = sort(unique(ref_edges.weight))
			loaded_weight_counts = sort(unique(loaded_w.edges.weight))
			weights_ok           = ref_weight_counts == loaded_weight_counts
			results[:weighted_weight_values] = weights_ok
			println("    Weight values: ref = $ref_weight_counts, loaded = $loaded_weight_counts  $(weights_ok ? "PASS" : "FAIL")")

		#	Check Max Weight Preserved
			max_ok_w = isapprox(maximum(ref_edges.weight), maximum(loaded_w.edges.weight); atol = 1e-9)
			results[:weighted_max_weight] = max_ok_w
			println("    Max weight:   ref = $(maximum(ref_edges.weight)), loaded = $(maximum(loaded_w.edges.weight))  $(max_ok_w ? "PASS" : "FAIL")")

		#	Check Edge Set Identity (Weighted)
			ref_set    = Set((row.src, row.dst, row.weight) for row in eachrow(ref_edges))
			loaded_set = Set((row.src, row.dst, row.weight) for row in eachrow(loaded_w.edges))
			edges_ok   = ref_set == loaded_set
			results[:weighted_edge_identity] = edges_ok
			println("    Edge set identity (src, dst, weight): $(edges_ok ? "PASS" : "FAIL")")

		#	Check Metadata Preserved
			meta = loaded_w.metadata
			meta_ok = haskey(meta, :network_name) &&
			          haskey(meta, :directed) && meta[:directed] === false &&
			          haskey(meta, :weighted) && meta[:weighted] === true &&
			          haskey(meta, :n_layers) && meta[:n_layers] == 7
			results[:weighted_metadata] = meta_ok
			println("    Metadata (directed=false, weighted=true, n_layers=7): $(meta_ok ? "PASS" : "FAIL")")

		#	Cross-Check Against Distributed Summary File
			println("\n  Step 3: Cross-check binarized edges against edges_crime_network.csv...")
			summary_path = joinpath(layer_dir, "edges_crime_network.csv")
			if isfile(summary_path)
				summary_edges_raw = _read_toledo_layer(summary_path)
				summary_edge_set  = Set(summary_edges_raw)
				ref_edge_set      = Set((row.src, row.dst) for row in eachrow(ref_edges))
				summary_match     = summary_edge_set == ref_edge_set
				results[:summary_file_match] = summary_match
				println("    Summary file edges:        $(length(summary_edge_set))")
				println("    Our 7-layer collapse:      $(length(ref_edge_set))")
				println("    Set equality: $(summary_match ? "PASS" : "FAIL")")
			else
				results[:summary_file_match] = false
				println("    Summary file not found at $summary_path; skipping cross-check")
			end

		#	Load Unweighted GraphML
			println("\n  Step 4: Load unweighted GraphML...")
			loaded_u = load_graphml(unweighted_path)
			println("    Loaded:   $(nrow(loaded_u.edges)) edges, $(nrow(loaded_u.nodes)) nodes")

		#	Check Counts Match (Unweighted)
			counts_ok_u = (nrow(loaded_u.edges) == nrow(ref_edges)) &&
			              (nrow(loaded_u.nodes) == nrow(ref_nodes))
			results[:unweighted_counts] = counts_ok_u
			println("    Counts match: $(counts_ok_u ? "PASS" : "FAIL")")

		#	Check Binarization
			all_ones = all(loaded_u.edges.weight .== 1.0)
			results[:unweighted_binarized] = all_ones
			println("    All weights = 1.0: $(all_ones ? "PASS" : "FAIL")")

		#	Check Edge Set Preserved
			ref_pairs    = Set((row.src, row.dst) for row in eachrow(ref_edges))
			loaded_pairs = Set((row.src, row.dst) for row in eachrow(loaded_u.edges))
			pairs_ok     = ref_pairs == loaded_pairs
			results[:unweighted_edge_pairs] = pairs_ok
			println("    Edge pair set identity: $(pairs_ok ? "PASS" : "FAIL")")

		#	Check Unweighted Metadata
			meta_u = loaded_u.metadata
			meta_u_ok = haskey(meta_u, :is_binary) && meta_u[:is_binary] === true &&
			            haskey(meta_u, :weighted) && meta_u[:weighted] === false &&
			            haskey(meta_u, :directed) && meta_u[:directed] === false
			results[:unweighted_metadata] = meta_u_ok
			println("    Metadata (is_binary=true, weighted=false, directed=false): $(meta_u_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = all(values(results))
			println("\n" * "=" ^ 70)
			println("SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			for (k, v) in results
				println("  $k: $(v ? "PASS" : "FAIL")")
			end
			println("=" ^ 70)

		return (all_passed = all_passed, per_test = results)
	end

	results = test_toledo_graphml_integrity(
		"toledo_crime_weighted.graphml",
		"toledo_crime_unweighted.graphml",
		"Crime_Networks (Toledo et al., 2023)"
	)

##########################
#   SYNTHETIC DATASETS   #
##########################

#	Helper Function for generate_sbm_network: Build Block Membership Vector
	function _build_block_membership(n_nodes::Int, n_blocks::Int)
		"""
		Args:
			n_nodes::Int: total number of nodes
			n_blocks::Int: number of blocks (communities)
		Returns:
			Vector{Int}: block label (1..n_blocks) for each node
		Notes:
			Distributes nodes across blocks as evenly as possible. If n_nodes is
			not divisible by n_blocks, the first (n_nodes mod n_blocks) blocks
			receive one extra node.
		"""

		#	Compute Block Sizes
			base_size  = div(n_nodes, n_blocks)
			remainder  = rem(n_nodes, n_blocks)
			block_sizes = [base_size + (b ≤ remainder ? 1 : 0) for b in 1:n_blocks]

		#	Build Membership Vector
			membership = Vector{Int}(undef, n_nodes)
			ix = 1
			for b in 1:n_blocks
				for _ in 1:block_sizes[b]
					membership[ix] = b
					ix += 1
				end
			end

		#	Return
			return membership
	end

#	Generate a Stochastic Block Model Network
	function generate_sbm_network(; n_nodes::Int = 600,
	                                n_blocks::Int = 6,
	                                p_in::Float64 = 0.10,
	                                p_out::Float64 = 0.008,
	                                directed::Bool = false,
	                                weighted::Bool = false,
	                                weight_lambda::Float64 = 2.0,
	                                seed::Int = 42)
		"""
		Args:
			n_nodes::Int: total number of nodes (default 600)
			n_blocks::Int: number of equal-sized blocks (default 6)
			p_in::Float64: within-block edge probability (default 0.10)
			p_out::Float64: between-block edge probability (default 0.008)
			directed::Bool: produce directed network (default false)
			weighted::Bool: produce weighted edges (default false)
			weight_lambda::Float64: Poisson rate for weight - 1 (default 2.0;
				generates weights from 1 + Poisson(λ), so weights ≥ 1)
			seed::Int: RNG seed for reproducibility (default 42)
		Returns:
			NamedTuple: (edges, nodes, metadata)
				edges::DataFrame   — :src, :dst, :weight
				nodes::DataFrame   — :id, :label, :block (block label 1..n_blocks)
				metadata::NamedTuple — graph properties
		Notes:
			Stochastic Block Model with n_blocks communities of roughly equal size.
			Within a block, each pair of distinct nodes is independently connected
			with probability p_in. Between blocks, the probability is p_out.

			For directed = true: each ordered pair (i, j) with i ≠ j is sampled
			independently. The result has potential asymmetry: A→B may exist
			without B→A.

			For undirected: only upper-triangular pairs (i < j) are sampled.

			For weighted: each edge receives weight 1 + X where X ~ Poisson(λ).
			This ensures all weights are at least 1, so binarization produces
			exactly the same edge set.
		"""

		#	Build Block Membership
			rng = Xoshiro(seed)
			block = _build_block_membership(n_nodes, n_blocks)

		#	Aggregate Edges
			#	Use temporary vectors with sizehint to avoid push! overhead
				expected_n_edges = if directed
					Int(round(p_in  * n_nodes * (n_nodes / n_blocks - 1)) +
					    round(p_out * n_nodes * n_nodes * (1 - 1/n_blocks)))
				else
					Int(round((p_in  * n_nodes * (n_nodes / n_blocks - 1) +
					          p_out * n_nodes * n_nodes * (1 - 1/n_blocks)) / 2))
				end
				edge_srcs    = Int[]
				edge_dsts    = Int[]
				edge_weights = Float64[]
				sizehint!(edge_srcs,    Int(round(expected_n_edges * 1.2)))
				sizehint!(edge_dsts,    Int(round(expected_n_edges * 1.2)))
				sizehint!(edge_weights, Int(round(expected_n_edges * 1.2)))

		#	Sample Edges
			weight_dist = Poisson(weight_lambda)
			if directed
				#	Sample All Ordered Pairs (i, j) with i != j
					for i in 1:n_nodes
						for j in 1:n_nodes
							i == j && continue
							p = block[i] == block[j] ? p_in : p_out
							if rand(rng) < p
								push!(edge_srcs, i)
								push!(edge_dsts, j)
								w = weighted ? 1.0 + Float64(rand(rng, weight_dist)) : 1.0
								push!(edge_weights, w)
							end
						end
					end
			else
				#	Sample Upper-Triangular Pairs (i, j) with i < j
					for i in 1:(n_nodes - 1)
						for j in (i + 1):n_nodes
							p = block[i] == block[j] ? p_in : p_out
							if rand(rng) < p
								push!(edge_srcs, i)
								push!(edge_dsts, j)
								w = weighted ? 1.0 + Float64(rand(rng, weight_dist)) : 1.0
								push!(edge_weights, w)
							end
						end
					end
			end

		#	Build DataFrames
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

			nodes_df = DataFrame(
				id    = 1:n_nodes,
				label = string.(1:n_nodes),
				block = block
			)

		#	Build Metadata
			metadata = (
				network_name        = "Synthetic 1 - SBM (low centralization)",
				source_format       = "synthetic_sbm",
				directed            = directed,
				weighted            = weighted,
				is_binary           = !weighted,
				n_nodes             = n_nodes,
				n_edges             = nrow(edges_df),
				n_blocks            = n_blocks,
				p_in                = p_in,
				p_out               = p_out,
				weight_distribution = weighted ? "1 + Poisson(λ=$weight_lambda)" : "binary",
				generator_seed      = seed
			)

		#	Return Triple
			return (edges = edges_df, nodes = nodes_df, metadata = metadata)
	end

#	Helper Function for generate_pa_network: Sample One Existing Node Weighted by Degree
	function _pa_sample_existing!(degrees::Vector{Int}, exclude::Set{Int}, n_current::Int, rng::AbstractRNG)
		"""
		Args:
			degrees::Vector{Int}: current degree of each node (1..n_current valid)
			exclude::Set{Int}: nodes to skip (already chosen this round)
			n_current::Int: how many nodes currently exist
			rng::AbstractRNG: RNG for sampling
		Returns:
			Int: selected node ID in 1..n_current, with probability proportional to degree
		Notes:
			Linear-time sampling. Computes cumulative degree of eligible nodes,
			samples a target threshold, then walks until cumulative >= threshold.
			Fast enough for N up to a few thousand.
		"""

		#	Compute Total Eligible Degree
			total_eligible_degree = 0
			for k in 1:n_current
				if !(k in exclude)
					total_eligible_degree += degrees[k]
				end
			end

		#	Sample Threshold and Walk
			if total_eligible_degree == 0
				#	Degenerate case: pick any non-excluded node uniformly
					eligible = [k for k in 1:n_current if !(k in exclude)]
					return eligible[rand(rng, 1:length(eligible))]
			end
			threshold = rand(rng) * total_eligible_degree
			cumulative = 0.0
			for k in 1:n_current
				if !(k in exclude)
					cumulative += degrees[k]
					if cumulative >= threshold
						return k
					end
				end
			end

		#	Fallback (Shouldn't Reach Here)
			for k in 1:n_current
				if !(k in exclude)
					return k
				end
			end

		#	Should Not Reach
			error("_pa_sample_existing!: no eligible node found")
	end

#	Generate a Preferential Attachment Network
	function generate_pa_network(; n_nodes::Int = 250,
	                               m::Int = 2,
	                               directed::Bool = false,
	                               weighted::Bool = false,
	                               weight_lambda::Float64 = 1.0,
	                               seed::Int = 42)
		"""
		Args:
			n_nodes::Int: total number of nodes (default 250)
			m::Int: number of edges each new node attaches with (default 2)
			directed::Bool: produce directed network (default false)
			weighted::Bool: produce weighted edges (default false)
			weight_lambda::Float64: Poisson rate for weight - 1 (default 1.0)
			seed::Int: RNG seed (default 42)
		Returns:
			NamedTuple: (edges, nodes, metadata)
				edges::DataFrame   — :src, :dst, :weight
				nodes::DataFrame   — :id, :label
				metadata::NamedTuple — graph properties
		Notes:
			Barabási-Albert preferential attachment model. Starts with a clique of
			(m+1) initial nodes, then for each new node n in (m+2)..N, attaches
			m edges to existing nodes with probability proportional to their
			current degree (rich-get-richer dynamics).

			For directed = true: each attachment edge points from the new node
			(source) to the chosen existing node (target). This is the canonical
			"citation network" formulation. The new node's out-degree at
			attachment is m; existing nodes accumulate in-degree.

			For undirected: edges have no directionality; we store them with
			src < dst convention.

			For weighted: each edge receives weight 1 + Poisson(λ).
		"""

		#	Validation
			if n_nodes <= m
				throw(ArgumentError("n_nodes ($n_nodes) must exceed m ($m) for PA"))
			end

		#	Initialize: Clique of (m+1) Nodes
			rng = Xoshiro(seed)
			n_initial = m + 1
			edge_srcs    = Int[]
			edge_dsts    = Int[]
			edge_weights = Float64[]
			degrees      = zeros(Int, n_nodes)
			weight_dist  = Poisson(weight_lambda)

			#	Add All Edges of Initial Clique
				for i in 1:(n_initial - 1)
					for j in (i + 1):n_initial
						push!(edge_srcs, i)
						push!(edge_dsts, j)
						w = weighted ? 1.0 + Float64(rand(rng, weight_dist)) : 1.0
						push!(edge_weights, w)
						degrees[i] += 1
						degrees[j] += 1
					end
				end

		#	Grow Network: Attach Each New Node n with m Edges
			for n in (n_initial + 1):n_nodes
				#	Sample m Existing Nodes Without Replacement
					chosen = Set{Int}()
					for _ in 1:m
						target = _pa_sample_existing!(degrees, chosen, n - 1, rng)
						push!(chosen, target)
					end

				#	Add Edges from New Node to Each Chosen Target
					for target in chosen
						#	For Undirected, Store with src < dst
							if directed
								push!(edge_srcs, n)
								push!(edge_dsts, target)
							else
								if n < target
									push!(edge_srcs, n)
									push!(edge_dsts, target)
								else
									push!(edge_srcs, target)
									push!(edge_dsts, n)
								end
							end

						w = weighted ? 1.0 + Float64(rand(rng, weight_dist)) : 1.0
						push!(edge_weights, w)
						degrees[n]      += 1
						degrees[target] += 1
					end
			end

		#	Build DataFrames
			edges_df = DataFrame(
				src    = edge_srcs,
				dst    = edge_dsts,
				weight = edge_weights
			)

			nodes_df = DataFrame(
				id    = 1:n_nodes,
				label = string.(1:n_nodes)
			)

		#	Build Metadata
			metadata = (
				network_name        = "Synthetic 2 - PA (high centralization)",
				source_format       = "synthetic_preferential_attachment",
				directed            = directed,
				weighted            = weighted,
				is_binary           = !weighted,
				n_nodes             = n_nodes,
				n_edges             = nrow(edges_df),
				m                   = m,
				weight_distribution = weighted ? "1 + Poisson(λ=$weight_lambda)" : "binary",
				generator_seed      = seed
			)

		#	Return Triple
			return (edges = edges_df, nodes = nodes_df, metadata = metadata)
	end

#	Build the Synthetic Half of the Corpus (8 GraphML Files)
	function build_synthetic_corpus(output_dir::String; seed::Int = 42)
		"""
		Args:
			output_dir::String: directory where GraphML files will be written
			seed::Int: RNG seed used consistently across all variants (default 42)
		Returns:
			NamedTuple: (synthetic_1::Dict, synthetic_2::Dict) listing the 4 files
				written for each network
		Notes:
			Generates all 8 synthetic network variants:
			- Synthetic 1 (SBM): {directed, undirected} × {weighted, unweighted}
			- Synthetic 2 (PA):  {directed, undirected} × {weighted, unweighted}

			Each variant uses the same seed so the underlying graph structure
			is comparable across weighted/unweighted forms of the same direction.
			Directed and undirected variants are independent samples since the
			sampling logic differs.
		"""

		#	Validation
			if !isdir(output_dir)
				mkpath(output_dir)
			end

		println("=" ^ 70)
		println("Building Synthetic Corpus")
		println("=" ^ 70)

		#	Synthetic 1: SBM
			println("\n  Synthetic 1: Stochastic Block Model (low centralization)")
			synthetic_1_files = Dict{String, String}()
			for directed in (false, true)
				for weighted in (false, true)
					net = generate_sbm_network(directed = directed, weighted = weighted, seed = seed)
					variant_name = "synthetic_1_sbm_$(directed ? "directed" : "undirected")_$(weighted ? "weighted" : "unweighted")"
					filepath = joinpath(output_dir, "$variant_name.graphml")
					write_graphml(net.edges, net.nodes, net.metadata, filepath)
					synthetic_1_files[variant_name] = filepath
					println("    Wrote $variant_name.graphml ($(nrow(net.edges)) edges)")
				end
			end

		#	Synthetic 2: Preferential Attachment
			println("\n  Synthetic 2: Preferential Attachment (high centralization)")
			synthetic_2_files = Dict{String, String}()
			for directed in (false, true)
				for weighted in (false, true)
					net = generate_pa_network(directed = directed, weighted = weighted, seed = seed)
					variant_name = "synthetic_2_pa_$(directed ? "directed" : "undirected")_$(weighted ? "weighted" : "unweighted")"
					filepath = joinpath(output_dir, "$variant_name.graphml")
					write_graphml(net.edges, net.nodes, net.metadata, filepath)
					synthetic_2_files[variant_name] = filepath
					println("    Wrote $variant_name.graphml ($(nrow(net.edges)) edges)")
				end
			end

		println("\n" * "=" ^ 70)
		println("Synthetic corpus complete: 8 files written to $output_dir")
		println("=" ^ 70)

		return (synthetic_1 = synthetic_1_files, synthetic_2 = synthetic_2_files)
	end

#   Generate Networks
    results = build_synthetic_corpus("/mnt/d/GitHub_Repositories/Network_Credible_Intervals/Data/GraphML_Test_Networks"; seed = 42)

#	Test: Round-Trip Integrity for One Synthetic Network Variant
	function test_synthetic_variant_integrity(filepath::String,
	                                            expected_n_nodes::Int,
	                                            expected_directed::Bool,
	                                            expected_weighted::Bool)
		"""
		Args:
			filepath::String: path to GraphML file to test
			expected_n_nodes::Int: expected number of nodes
			expected_directed::Bool: expected directed flag in metadata
			expected_weighted::Bool: expected weighted flag in metadata
		Returns:
			NamedTuple: (passed::Bool, checks::Dict{Symbol, Bool})
		Notes:
			Loads a single synthetic GraphML file and verifies counts, metadata,
			and weight properties. The expected counts and flags are passed in
			because each variant has different expected values.
		"""

		checks = Dict{Symbol, Bool}()

		#	Load
			loaded = load_graphml(filepath)

		#	Check Node Count
			checks[:n_nodes] = nrow(loaded.nodes) == expected_n_nodes

		#	Check Edge Count Is Positive
			checks[:n_edges_positive] = nrow(loaded.edges) > 0

		#	Check Metadata: Directed
			checks[:directed_flag] = haskey(loaded.metadata, :directed) &&
			                          loaded.metadata[:directed] === expected_directed

		#	Check Metadata: Weighted
			checks[:weighted_flag] = haskey(loaded.metadata, :weighted) &&
			                          loaded.metadata[:weighted] === expected_weighted

		#	Check Weight Properties Match Variant Type
			if expected_weighted
				#	All Weights Should Be ≥ 1 (Per 1 + Poisson Distribution)
					checks[:weights_min_1] = minimum(loaded.edges.weight) >= 1.0
				#	Weights Should Not All Be Equal (Variability Required)
					checks[:weights_variable] = length(unique(loaded.edges.weight)) > 1
			else
				#	Unweighted Should Have All Weights = 1
					checks[:weights_all_1] = all(loaded.edges.weight .== 1.0)
			end

		#	All-Passed Flag
			passed = all(values(checks))

		return (passed = passed, checks = checks)
	end

#	Helper Function: Compute Degree Distribution from Edges
	function _compute_degree_distribution(edges_df::DataFrame, n_nodes::Int, directed::Bool)
		"""
		Args:
			edges_df::DataFrame: edges with :src, :dst
			n_nodes::Int: total node count
			directed::Bool: whether to compute total or split in/out
		Returns:
			Vector{Int}: degree of each node (total degree for both undirected and directed)
		Notes:
			Returns total degree per node, ignoring weights. For directed graphs,
			total degree = in_degree + out_degree.
		"""

		degrees = zeros(Int, n_nodes)
		for row in eachrow(edges_df)
			degrees[row.src] += 1
			degrees[row.dst] += 1
		end
		return degrees
	end

#	Test: Verify Structural Properties of the Synthetic Corpus
	function test_synthetic_corpus_structure(file_paths::NamedTuple)
		"""
		Args:
			file_paths::NamedTuple: with fields :synthetic_1 and :synthetic_2, each a Dict
				mapping variant name → file path. Matches the return type of
				build_synthetic_corpus.
		Returns:
			NamedTuple: (all_passed::Bool, results::Dict)
		Notes:
			Verifies that the SBM has expected community structure and the PA
			has expected hub-skewed degree distribution. Provides quantitative
			evidence that "low centralization" and "high centralization" are
			actually realized by the generators.

			Centralization here uses the Gini coefficient of the degree
			distribution: 0 = perfectly equal degree (uniform graph), 1 = one
			node has all edges. SBM should have Gini in [0.10, 0.30]; PA should
			have Gini in [0.40, 0.65] for these parameters.

			Leiden is called on the unweighted variants of each network with
			weighted=false explicitly, since the loaded unweighted files have
			all weights = 1.0 and would be rejected as binary if treated as
			weighted.
		"""

		println("=" ^ 70)
		println("Synthetic Corpus Structural Tests")
		println("=" ^ 70)

		results = Dict{Symbol, Any}()

		#	Helper: Gini Coefficient
			gini = function(degrees::Vector{Int})
				n = length(degrees)
				n == 0 && return 0.0
				sorted = sort(degrees)
				cumsum_d = sum(sorted)
				cumsum_d == 0 && return 0.0
				weighted_sum = sum((2 * i - n - 1) * sorted[i] for i in 1:n)
				return weighted_sum / (n * cumsum_d)
			end

		#	Round-Trip Integrity Across All 8 Files
			println("\n  Step 1: Round-trip integrity for all 8 files")
			integrity_passed = true
			for (variant_name, filepath) in file_paths.synthetic_1
				directed = occursin("directed", variant_name) && !occursin("undirected", variant_name)
				weighted = occursin("_weighted", variant_name) && !occursin("_unweighted", variant_name)
				result = test_synthetic_variant_integrity(filepath, 600, directed, weighted)
				print("    $variant_name: ")
				if result.passed
					println("PASS")
				else
					println("FAIL")
					integrity_passed = false
					for (k, v) in result.checks
						v || println("       failed: $k")
					end
				end
			end
			for (variant_name, filepath) in file_paths.synthetic_2
				directed = occursin("directed", variant_name) && !occursin("undirected", variant_name)
				weighted = occursin("_weighted", variant_name) && !occursin("_unweighted", variant_name)
				result = test_synthetic_variant_integrity(filepath, 250, directed, weighted)
				print("    $variant_name: ")
				if result.passed
					println("PASS")
				else
					println("FAIL")
					integrity_passed = false
					for (k, v) in result.checks
						v || println("       failed: $k")
					end
				end
			end
			results[:integrity_all_passed] = integrity_passed

		#	SBM Structural Check
			println("\n  Step 2: SBM has community structure (low centralization)")
			sbm_path = file_paths.synthetic_1["synthetic_1_sbm_undirected_unweighted"]
			sbm = load_graphml(sbm_path)
			sbm_degrees = _compute_degree_distribution(sbm.edges, 600, false)
			sbm_gini = gini(sbm_degrees)
			sbm_low_centralization = 0.05 < sbm_gini < 0.35
			results[:sbm_gini] = sbm_gini
			results[:sbm_low_centralization] = sbm_low_centralization

			#	Run Leiden to Verify Community Structure
				println("    SBM degree Gini: $(round(sbm_gini, digits=4))")
				println("    Low centralization check (Gini < 0.35): $(sbm_low_centralization ? "PASS" : "FAIL")")
				println("    Running Leiden on SBM to verify communities exist...")
				sbm_leiden = leiden_community_detection(sbm.edges; weighted = false, n_runs = 5, seed = 42)
				println("    Leiden modularity: $(round(sbm_leiden.modularity, digits=4))")
				println("    Leiden communities: $(sbm_leiden.n_communities)")
				results[:sbm_modularity] = sbm_leiden.modularity
				results[:sbm_communities] = sbm_leiden.n_communities
				#	Modularity > 0.3 indicates real community structure
				sbm_has_structure = sbm_leiden.modularity > 0.3
				results[:sbm_has_community_structure] = sbm_has_structure
				println("    Has community structure (Q > 0.3): $(sbm_has_structure ? "PASS" : "FAIL")")

		#	PA Structural Check
			println("\n  Step 3: PA has hub-skewed degree distribution (high centralization)")
			pa_path = file_paths.synthetic_2["synthetic_2_pa_undirected_unweighted"]
			pa = load_graphml(pa_path)
			pa_degrees = _compute_degree_distribution(pa.edges, 250, false)
			pa_gini = gini(pa_degrees)
			pa_high_centralization = pa_gini > 0.35
			results[:pa_gini] = pa_gini
			results[:pa_high_centralization] = pa_high_centralization

			println("    PA degree Gini: $(round(pa_gini, digits=4))")
			println("    High centralization check (Gini > 0.35): $(pa_high_centralization ? "PASS" : "FAIL")")
			println("    Max degree: $(maximum(pa_degrees))")
			println("    Mean degree: $(round(sum(pa_degrees) / length(pa_degrees), digits=2))")
			println("    Top 5 nodes by degree: $(sort(pa_degrees, rev = true)[1:5])")

			#	Run Leiden on PA Network for Comparison
				println("    Running Leiden on PA to compare modularity...")
				pa_leiden = leiden_community_detection(pa.edges; weighted = false, n_runs = 5, seed = 42)
				results[:pa_modularity] = pa_leiden.modularity
				results[:pa_communities] = pa_leiden.n_communities
				println("    Leiden modularity: $(round(pa_leiden.modularity, digits=4))")
				println("    Leiden communities: $(pa_leiden.n_communities)")

		#	Centralization Contrast
			println("\n  Step 4: Centralization contrast")
			contrast = pa_gini - sbm_gini
			println("    PA Gini - SBM Gini = $(round(contrast, digits=4))")
			contrast_ok = contrast > 0.1
			results[:centralization_contrast] = contrast_ok
			println("    PA significantly more centralized than SBM: $(contrast_ok ? "PASS" : "FAIL")")

		#	Summary
			all_passed = integrity_passed && sbm_low_centralization && sbm_has_structure &&
			             pa_high_centralization && contrast_ok
			println("\n" * "=" ^ 70)
			println("SYNTHETIC CORPUS SUMMARY: $(all_passed ? "ALL PASSED" : "SOME FAILED")")
			println("  Round-trip integrity:           $(integrity_passed ? "PASS" : "FAIL")")
			println("  SBM low centralization:         $(sbm_low_centralization ? "PASS" : "FAIL") (Gini = $(round(sbm_gini, digits=3)))")
			println("  SBM has community structure:    $(sbm_has_structure ? "PASS" : "FAIL") (Q = $(round(sbm_leiden.modularity, digits=3)))")
			println("  PA high centralization:         $(pa_high_centralization ? "PASS" : "FAIL") (Gini = $(round(pa_gini, digits=3)))")
			println("  PA more centralized than SBM:   $(contrast_ok ? "PASS" : "FAIL") (Δ = $(round(contrast, digits=3)))")
			println("=" ^ 70)

		return (all_passed = all_passed, results = results)
	end

	test_results = test_synthetic_corpus_structure(results)