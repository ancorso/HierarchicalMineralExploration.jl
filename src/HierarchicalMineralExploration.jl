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
using ParticleFilters

include("domains.jl")
export ThicknessBackground, GradeBackground
export GeochemicalDomainDistribution, draw_geochemical_domain
export GrabenDistribution, draw_graben

include("hypotheses.jl")
export Hypothesis, MaxEntropyHypothesis, turing_model, default_alg, logprob
export one_graben_one_geochem
export one_graben_two_geochem
export two_graben_one_geochem
export two_graben_two_geochem

include("pomdp.jl")
export HierarchicalMinExPOMDP, HierarchicalMinExState

include("beliefs.jl")
export MCMCUpdater

include("visualization.jl")
export plot_model, plot_state

end # module HierarchicalMineralExploration
