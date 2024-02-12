using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs

## Setup mineral system models
include("setup.jl")

## Build out various hypotheses
hypotheses = [
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains), # Two grabens, two geochem domains
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, [geochem_domains[1]]), # Two grabens, single geochem domain
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, geochem_domains), # Single graben, two geochem domains
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]), # Single graben, single geochem domain
]
max_ent_hypothesis = MaxEntropyHypothesis(Normal(6, 7), Normal(10,10))

## Create some plots of the various hypotheses
sample1 = turing_model(hypotheses[1])(Dict(), hypotheses[1], true)()
plot_model(; sample1...)
savefig("figures/hypothesis1.png")

sample2 = turing_model(hypotheses[2])(Dict(), hypotheses[2], true)()
plot_model(; sample2...)
savefig("figures/hypothesis2.png")

sample3 = turing_model(hypotheses[3])(Dict(), hypotheses[3], true)()
plot_model(; sample3...)
savefig("figures/hypothesis3.png")

sample4 = turing_model(hypotheses[4])(Dict(), hypotheses[4], true)()
plot_model(; sample4...)
savefig("figures/hypothesis4.png")

## Choose a ground truth hypothesis
h_gt = hypotheses[1]
m_type_gt = turing_model(h_gt)
m = m_type_gt(Dict(), h_gt, true)
sample = m()

# Obtain some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:40]
observations = Dict(
    p => (thickness=sample.thickness[p...], grade=sample.grade[p...]) for p in pts
)

# Visualize the model and observations
plot_model(; sample..., observations=observations)
savefig("figures/observations.png")

## Show what conditioning a hypothesis to a given set of data looks like
h = hypotheses[1]
m_type = turing_model(h)

# Construct the conditional distribution
mcond = m_type(observations, h)
mcond_w_samples = m_type(observations, h, true)

# Sample from the posterior
Nsamples = 1000
Nchains = 1
alg = HierarchicalMineralExploration.default_alg(h)

# Run the chains and store the outcomes
chns = mapreduce(c -> Turing.sample(mcond, alg, Nsamples), chainscat, 1:Nchains)

# Compute the likelihood
outputs = generated_quantities(mcond, chns[end, :, :])[1]
plot(chns[:, :lp, 1]) # Show the likelihood progress over time

outputs = generated_quantities(mcond_w_samples, chns[end, :, :])
reshape(outputs, 1)
plots = []
for i in 1:size(outputs,2)
    push!(
        plots,
        plot_model(;
            structural=outputs[end, i].structural,
            thickness=outputs[end, i].thickness,
            grade=outputs[end, i].grade,
            geochemdomain=outputs[end, i].geochemdomain,
            observations,
        ),
    )
end
plot(plots...; size=(800, 800))
savefig("figures/conditioned_hypothesis.png")