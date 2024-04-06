using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using POMDPs
using ParticleFilters
using POMDPDiscretization
using Random
using NativeSARSOP
using POMDPTools
using JLD2
using DataStructures
using Dates
## Setup mineral system models and the POMDP solver helpers and runners
include("setup.jl")
include("pomdp_tools.jl")


# Set up the output directory
output_dir = "../outputs/all"
try
    mkdir(output_dir)
catch
end

# Define the pomdp
pomdp = HierarchicalMinExPOMDP()

# Ground truth hypothesis
h_gt = hypotheses[1]
m_gt = turing_model(h_gt)(Dict(), h_gt, true)

Nsamples = 150

Random.seed!(ceil(Int, datetime2unix(now())))
i = ceil(Int, rand()*1000000)

println("============== Trial ID: ", i, "=================")

results = Dict()

# Get the ground true
res_gt = m_gt()
s_gt = HierarchicalMinExState(res_gt.thickness, res_gt.grade)

# Run Grid search
history = run_gridsearch(pomdp, s_gt, hypotheses, max_ent_hypothesis; Nsamples)
# plot_gif(pomdp, history, "$(output_dir)/history_gridsearch$i.gif")
@save "$(output_dir)/history_gridsearch$i.jld2" history
results["gridsearch"] = history

# Run POMDP with correct hypothesis
history = run_trial(pomdp, s_gt, OrderedDict(1 => h_gt), max_ent_hypothesis; Nsamples)
# plot_gif(pomdp, history, "$(output_dir)/history_1correct$i.gif")
@save "$(output_dir)/history_1correct$i.jld2" history
results["1correct"] = history

# Run POMDP with all 4 hypotheses
history = run_trial(pomdp, s_gt, hypotheses, max_ent_hypothesis; Nsamples)
# plot_gif(pomdp, history, "$(output_dir)/history_4withcorrect$i.gif")
@save "$(output_dir)/history_4withcorrect$i.jld2" history
results["4withcorrect"] = history

# Run POMDP with incorrect hypotheses
history = run_trial(
    pomdp, s_gt, OrderedDict(i=>hypotheses[i] for i=2:4), max_ent_hypothesis; Nsamples
)
# plot_gif(pomdp, history, "$(output_dir)/history_3incorrect$i.gif")
@save "$(output_dir)/history_3incorrect$i.jld2" history
results["3incorrect"] = history

# Run POMDP with hypothesis reguvination
history = run_trial_rejuvination(
    pomdp, s_gt, OrderedDict(i=>hypotheses[i] for i=4:-1:1), max_ent_hypothesis; Nsamples
)
# plot_gif(pomdp, history, "$(output_dir)/history_rejuvination$i.gif")
@save "$(output_dir)/history_rejuvination$i.jld2" history
results["rejuvination"] = history

# Save the results
@save "$(output_dir)/results$i.jld2" results

