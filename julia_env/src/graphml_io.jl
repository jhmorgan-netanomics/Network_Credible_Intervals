module graphml_io

#   Module Packages
    using DataFrames
    using EzXML

#   ====================================================================
#   graphml_io submodule
#
#   GraphML I/O for the Network_Credible_Intervals package. Provides a
#   matched read/write pair plus the small helpers each side needs.
#
#   Public API:
#       write_graphml(edges_df, nodes_df, metadata, output_path)
#       load_graphml(filepath)
#
#   Both functions follow the same GraphML 1.0 schema: graph-level
#   metadata is emitted as <data> elements on the <graph> element;
#   node IDs are prefixed with "n" (e.g. id=42 becomes "n42"); edge
#   and node attribute columns from the supplied DataFrames are
#   declared as <key> elements and emitted as <data> children of
#   each <node> / <edge>. write_graphml followed by load_graphml is a
#   round-trip on (edges, nodes, metadata) modulo column ordering.
#
#   The four underscore-prefixed helpers below are not exported; they
#   are split out only to keep the two public functions readable.
#   ====================================================================

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

		#	Handle Missing (None of Empty String)
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
	@doc raw"""
	**Description**
	Writes a network (edges, nodes, and graph-level metadata) to a GraphML 1.0
	file. Designed as the inverse of `load_graphml`: writing and then loading
	round-trips the data faithfully, with attribute types preserved.

	**Usage**
	`write_graphml(edges_df::DataFrame, nodes_df::DataFrame, metadata, output_path::String)`

	**Arguments**
	- `edges_df::DataFrame`: Edge list. Must contain `:src` and `:dst`. All other
	  columns (e.g., `:weight`) are emitted as edge attributes.
	- `nodes_df::DataFrame`: Node list. Must contain `:id`. All other columns are
	  emitted as node attributes.
	- `metadata::Union{NamedTuple, AbstractDict}`: Graph-level attributes. Should
	  include at minimum `:network_name`, `:source_format`, `:directed`,
	  `:weighted`. Any additional fields are emitted as graph-level `<data>`
	  elements.
	- `output_path::String`: Destination file path.

	**Details**
	- Attribute types are auto-detected from each DataFrame column's non-missing
	  element type, producing GraphML `int`, `double`, `boolean`, or `string`.
	- Edge direction is set via the graph's `edgedefault` attribute, drawn from
	  `metadata.directed`. Per-edge directionality overrides are not emitted.
	- Node IDs are prefixed with `n` in the GraphML output (e.g., `id=42` becomes
	  `n42`) to satisfy GraphML's requirement that element IDs be valid XML names.
	  `load_graphml` strips this prefix on read.
	- Missing values are emitted as absent `<data>` elements rather than empty
	  strings, so they round-trip back to `missing`.
	- Reserved columns (`:id` on `nodes_df`; `:src`, `:dst` on `edges_df`) are not
	  duplicated as attributes.

	**Value**
	`Nothing`. The file is written to `output_path` as a side effect.

	**Examples**
	```julia
	using DataFrames
	edges = DataFrame(src=[1,2,3], dst=[2,3,1], weight=[1.0,2.0,3.0])
	nodes = DataFrame(id=[1,2,3], label=["A","B","C"])
	meta  = (network_name="triangle", source_format="manual",
	         directed=true, weighted=true)
	write_graphml(edges, nodes, meta, "triangle.graphml")
	```

	**See Also**
	`load_graphml`
	""" write_graphml

#	Helper Function for load_graphml: Parse a Single GraphML Value
	function _parse_graphml_value(text::AbstractString, attr_type::AbstractString)
		"""
		Args:
			text::AbstractString: raw text content from a <data> element
			attr_type::AbstractString: GraphML type ("int", "double", "boolean", "string")
		Returns:
			Union{Int, Float64, Bool, String, Missing}: parsed value
		Notes:
			For string-typed attributes, an empty (or whitespace-only) text content
			is treated as a legitimate empty-string value and returned as "". This
			preserves writer/reader symmetry: write_graphml emits "" as a present-
			but-empty <data> element (an absent attribute is emitted as no <data>
			element at all), and the reader must recover "" rather than coercing
			it to missing.

			For non-string types (int, double, boolean), empty text cannot be
			parsed and is returned as missing. The caller distinguishes this from
			"data element absent entirely," which is handled at the call site
			(load_graphml uses `get(this_node_data, attr_name, missing)` so an
			absent element never reaches this helper).

			Numeric and boolean parses use the declared attr_type. String parses
			run XML entity unescaping (&amp;, &lt;, &gt;, &quot;, &apos;).
		"""

		#	Strip Surrounding Whitespace
			t = strip(text)

		#	Handle Empty Text by Type
			if isempty(t)
				#	Empty string is a legitimate value only for string-typed attrs.
				#	For numeric/boolean attrs, empty text is unparseable → missing.
					return attr_type == "string" ? "" : missing
			end

		#	Parse by Declared Type
			if attr_type == "int"
				return parse(Int, t)
			elseif attr_type == "double"
				return parse(Float64, t)
			elseif attr_type == "boolean"
				return lowercase(t) in ("true", "1")
			else
				return _xml_unescape(String(t))
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
	@doc raw"""
	**Description**
	Reads a GraphML 1.0 file into Julia data structures: an edge list, a node
	list, and a graph-level metadata NamedTuple. Designed as the inverse of
	`write_graphml`, with attribute types preserved on read.

	**Usage**
	`load_graphml(filepath::String)`

	**Arguments**
	- `filepath::String`: Path to a GraphML file. Typically a file produced by
	  `write_graphml`, but the loader also accepts GraphML 1.0 files from other
	  sources (Gephi, igraph, NetworkX) provided they follow the standard schema.

	**Details**
	- Numeric attributes are returned as `Int` or `Float64`, booleans as `Bool`,
	  and everything else as `String`. Absent `<data>` elements come back as
	  `missing` in the corresponding DataFrame column.
	- Node IDs are expected to carry an `"n"` prefix (as emitted by
	  `write_graphml`) and are stripped on read. If no prefix is present, the
	  loader falls back to parsing the entire ID string as an integer; this
	  supports files from sources that use other ID conventions.
	- The `metadata` NamedTuple always contains `:directed`, derived from the
	  graph's `edgedefault` attribute. Any other graph-level `<data>` elements
	  declared via `<key for="graph">` are added as additional fields.
	- An `ArgumentError` is thrown if the file does not exist, if the root
	  element is not `<graphml>`, or if no `<graph>` element is found.

	**Value**
	A `NamedTuple` with three fields:
	- `edges::DataFrame`: columns `:src`, `:dst`, plus any other edge attributes.
	- `nodes::DataFrame`: column `:id`, plus any other node attributes.
	- `metadata::NamedTuple`: graph-level attributes including `:directed`.

	**Examples**
	```julia
	result = load_graphml("triangle.graphml")
	result.edges     # DataFrame of edges
	result.nodes     # DataFrame of nodes
	result.metadata  # NamedTuple of graph-level attributes
	```

	**See Also**
	`write_graphml`
	""" load_graphml

#   Exports (public API)
    export write_graphml,
           load_graphml

end # module graphml_io