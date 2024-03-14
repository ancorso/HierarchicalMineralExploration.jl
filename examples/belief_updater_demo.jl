using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using AdvancedMH
using LogExpFunctions
using POMDPs

## Setup mineral system models
include("setup.jl")

# Set up the ground truth
h = hypotheses[1]
m_type = turing_model(h)
m = m_type(Dict(), h, true)
sgt = m()

# Visualize the model and observations
plot_model(; sgt...)

# Pre-select the points
pts = [[x, y] for x in 5:5:30 for y in 5:5:30]

# Setup the algorithms
alg = default_alg(h)

# Construct the belief updater
Nsamples=10
Nparticles=10
up = MCMCUpdater(Nsamples, [h], Nparticles)
b = initialize_belief(up, nothing)

# Target data to assimilate
i = 1
observations = Dict(
    p => (thickness=sgt.thickness[p...], grade=sgt.grade[p...]) for p in pts[1:i]
)

plot_model(; sgt..., observations)

# update the belief
up.observations = observations
b = update(up, b)

# Plot the results
anim = @animate for (i,p) in enumerate(particles(b))
    plot_state(p, observations)
    plot!(title="i=$i")
end
gif(anim, "particles.gif"; fps=2)
