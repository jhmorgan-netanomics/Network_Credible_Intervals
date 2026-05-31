__precompile__(true)

@doc raw"""
MIT License

Copyright (c) 2025 Jonathan H. Morgan, Ph.D., Netanomics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
""" 

module Network_Credible_Intervals

#   Load submodules
    include("graphml_io.jl")
    include("network_community_detection.jl")
    include("network_statistics.jl")
    include("network_degeneracy_functions.jl")
    include("network_reconstruction_functions.jl")

#   Pull their exports into the parent namespace
    using .graphml_io
    using .network_community_detection
    using .network_statistics
    using .network_degeneracy
    using .network_reconstruction

#   Re-export to users of Network_Credible_Intervals
    export write_graphml,
           load_graphml,
           calculate_modularity,
           delta_modularity_undirected_best!,
           delta_modularity_directed_best!,
           delta_modularity_best!,
           _leiden_single_run_preprocessed,
           leiden_community_detection,
           champ_community_detection,
           gini_coefficient,
           centralization,
           rand_index,
           in_degree,
           out_degree,
           total_degree,
           freeman_degree_normalization,
           freeman_degree_centralization,
           closeness_centrality,
           betweenness_centrality,
           mean_inverse_distance,
           bonacich_centrality,
           largest_component_proportion,
           reciprocity,
           local_weighted_reciprocity,
           local_clustering_coefficient,
           global_clustering_coefficient,
           recommend_L,
           _triad_census_layered,
           triad_census,
           largest_bicomponent_proportion,
           tau_statistic,
           structural_equivalence_blockmodel,
           generate_missingness_mask,
           apply_missingness,
           build_degeneration_corpus,
           SamplerSetup,
		   build_community_corpus,
		   compute_setup,
           feasible_rho_range

end # module Network_Credible_Intervals