using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using AdvancedMH
using LogExpFunctions
using POMDPs
using ParticleFilters

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
Nsamples=100
Nparticles=100
up = MCMCUpdater(Nsamples, hypotheses, Nparticles)
b = initialize_belief(up, nothing)

# Target data to assimilate
i = 36
obs = Dict(
    p => (thickness=sgt.thickness[p...], grade=sgt.grade[p...]) for p in pts[1:i]
)

plot_model(; sgt..., observations=obs)

# update the belief
up.observations = obs
b = update(up, b)

# mcond = m_type(obs, h)
# mcond_w_samples = m_type(obs, h, true)

# alg = default_alg(h)

# chn = sample(mcond, alg, 100; save_state=true, discard_initial=100)
# # ess(chn)
# # plot(chn)
plot(up.chains[4][:,:lp,:])


# os = generated_quantities(mcond_w_samples, chn[:, :, :])[:]
# ps = [HierarchicalMinExState(o.thickness, o.grade) for o in os]

# Plot the results
anim = @animate for (i, p) in enumerate(particles(b.particles))
    plot_state(p, obs)
    plot!(; title="i=$i")
end
gif(anim, "particles.gif"; fps=2)
