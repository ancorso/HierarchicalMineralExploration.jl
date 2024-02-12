using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using POMDPs
using ParticleFilters

## Setup mineral system models
include("setup.jl")

## Ground truth
h_gt = Hypothesis(
    N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains
)
m_gt = turing_model(h_gt)(Dict(), h_gt, true)
res_gt = m_gt()
s_gt = HierarchicalMinExState(res_gt.thickness, res_gt.grade)

# Plot the mineral model
plot_model(;res_gt...)

## Define the pomdp
pomdp = HierarchicalMinExPOMDP()

# Check the transition model
gen(pomdp, s_gt, (20,20))
gen(pomdp, s_gt, :mine)
gen(pomdp, s_gt, :abandon)

## Belief configuration
hypotheses = [
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains), # Two grabens, two geochem domains
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, [geochem_domains[1]]), # Two grabens, single geochem domain
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, geochem_domains), # Single graben, two geochem domains
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]), # Single graben, single geochem domain
]

# Contruct the initial belief with the hypotheses
Nsamples = 25
Nparticles = 12
up = MCMCUpdater(Nsamples, hypotheses, Nparticles, Dict())
b0 = initialize_belief(up, nothing)

# Show some initial particles
anim = @animate for i in 1:Nparticles
    plot_state(b0.particles[i])
end
gif(anim, "figures/initial_belief.gif", fps=2)

# Show the reward distribution
rewards = [HierarchicalMineralExploration.extraction_reward(pomdp, s) for s in b0.particles]
histogram(rewards)
savefig("figures/reward_distribution.png")

# Fill the updater with some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:10]
up.observations = Dict(
    p => (thickness=res_gt.thickness[p...], grade=res_gt.grade[p...]) for p in pts
)

# show the observations over the ground truth
plot_state(s_gt, up.observations)

# Belief update
b10 = update(up, b0, pts[end], up.observations[pts[end]])

# Show the posterior
anim = @animate for i in 1:Nparticles
    plot_state(b10[i], up.observations)
end
gif(anim, "figures/posterior_belief.gif", fps=2)

