module HierarchicalMineralExploration

using Luxor
using ColorTypes
using ImageFiltering
using Parameters
using Distributions
using Random
using Turing
using LazySets
using AbstractGPs
using Plots

include("domains.jl")
export ThicknessBackground, GradeBackground
export GeochemicalDomainDistribution, draw_geochemical_domain
export GrabenDistribution, draw_graben

include("hypotheses.jl")
export Hypothesis, one_graben_one_geochem

include("visualization.jl")
export plot_model


end # module HierarchicalMineralExploration
