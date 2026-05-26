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

#   Load the network_community_detection submodule
    include("network_community_detection.jl")

#   Pull its exports into the parent namespace
    using .network_community_detection

#   Re-export to users of Network_Credible_Intervals
    export calculate_modularity,
           delta_modularity_undirected_best!,
           delta_modularity_directed_best!,
           delta_modularity_best!,
           _leiden_single_run_preprocessed,
           leiden_community_detection,
           champ_community_detection

end # module Network_Credible_Intervals
