module HierarchicalMineralExploration

using Luxor
using ColorTypes
using ImageFiltering
using Parameters
using Distributions
using Random
using Turing
using AdvancedMH
using Memoization
using LazySets
using AbstractGPs
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using LogExpFunctions
using POMDPs
using POMDPTools
using ParticleFilters
using StatsBase
using Meshes

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
export HierarchicalMinExPOMDP, HierarchicalMinExState, extraction_reward

include("beliefs.jl")
export MCMCUpdater, MultiHypothesisBelief

include("visualization.jl")
export plot_model, plot_state, plot_mineralization, plot_belief, plot_step

end # module HierarchicalMineralExploration
