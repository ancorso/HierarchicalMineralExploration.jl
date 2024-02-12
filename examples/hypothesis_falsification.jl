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
max_ent_hypothesis = MaxEntropyHypothesis(Normal(8, 8), Normal(8,8))

## Choose a ground truth hypothesis
h = hypotheses[1]
m_type = turing_model(h)
m = m_type(Dict(), h, true)
sample = m()

# Obtain some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:10]
observations = Dict(
    p => (thickness=sample.thickness[p...], grade=sample.grade[p...]) for p in pts
)

# Visualize the model and observations
plot_model(; sample..., observations=observations)

nobs_list = [1, 2, 5, 10] #, 20, 50, 100]
logprobs_h1 = []
logprobs_h2 = []
logprobs_h3 = []
logprobs_h4 = []
logprobs_max_ent = []

Nsamples = 100
Nchains = 10
for nobs in nobs_list
    println("computing likelihoods for $nobs observations")
    observations = Dict(
        p => (thickness=sample.thickness[p...], grade=sample.grade[p...]) for p in pts[1:nobs]
    )
    ## Compute likelihoods vs observations
    push!(logprobs_h1, logprob(hypotheses[1], observations; Nsamples, Nchains))
    push!(logprobs_h2, logprob(hypotheses[2], observations; Nsamples, Nchains))
    push!(logprobs_h3, logprob(hypotheses[3], observations; Nsamples, Nchains))
    push!(logprobs_h4, logprob(hypotheses[4], observations; Nsamples, Nchains))
    push!(logprobs_max_ent, logprob(max_ent_hypothesis, observations))
end

plot(nobs_list, logprobs_h1[end-3:end], label="Hypothesis 1 (correct)", xlabel="Number of observations", ylabel="Log likelihood", dpi=300)
plot!(nobs_list, logprobs_h2[end-3:end], label="Hypothesis 2")
plot!(nobs_list, logprobs_h3[end-3:end], label="Hypothesis 3")
plot!(nobs_list, logprobs_h4[end-3:end], label="Hypothesis 4")
plot!(nobs_list, logprobs_max_ent[end-3:end], label="Max Entropy Hypothesis")

savefig("figures/hypothesis_likelihoods.png")