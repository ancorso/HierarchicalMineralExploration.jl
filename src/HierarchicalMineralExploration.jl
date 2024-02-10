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
using LogExpFunctions
using POMDPs
using POMDPTools

include("domains.jl")
export ThicknessBackground, GradeBackground
export GeochemicalDomainDistribution, draw_geochemical_domain
export GrabenDistribution, draw_graben

include("hypotheses.jl")
export Hypothesis, one_graben_one_geochem, loglikelihood

include("visualization.jl")
export plot_model

include("pomdp.jl")
export HierarchicalMinExPOMDP, HierarchicalMinExState

include("beliefs.jl")
export turing_model, particle_collection

end # module HierarchicalMineralExploration
