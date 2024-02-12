using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs

## Setup mineral system models
K = Matern52Kernel() ∘ ScaleTransform(1.0 / 3.0)
N = 32 # Number of grid points
t₀ = ThicknessBackground(1.0, K) # Background thickness distribution (μ, kernel)
σₜ = 0.01 # Noise on thickness observation
grabens = [GrabenDistribution(; N, μ=6.0), GrabenDistribution(; N, μ=10.0)]
γ₀ = GradeBackground(0.0, K) # Background grade distribution
σᵧ = 0.01 # Standard deviation of the measurement noise on the grade
geochem_domains = [
    GeochemicalDomainDistribution(; N, μ=15.0, kernel=K),
    GeochemicalDomainDistribution(; N, μ=10.0, kernel=K),
]

## Build out various hypotheses
hypotheses = [
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains), # Two grabens, two geochem domains
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, [geochem_domains[1]]), # Two grabens, single geochem domain
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, geochem_domains), # Single graben, two geochem domains
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]), # Single graben, single geochem domain
]
max_ent_hypothesis = MaxEntropyHypothesis(Normal(6, 7), Normal(10,10))

## show the different models
h = hypotheses[1]
m_type = turing_model(h)
m = m_type(Dict(), h, true)
sample = m()

# Obtain some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:10]
observations = Dict(
    p => (thickness=sample.thickness[p...], grade=sample.grade[p...]) for p in pts
)

## Compute likelihoods:
Nsamples = 25
Nchains = 10

logprob(hypotheses[1], observations; Nsamples, Nchains)
logprob(hypotheses[2], observations; Nsamples, Nchains)
logprob(hypotheses[3], observations; Nsamples, Nchains)
logprob(hypotheses[4], observations; Nsamples, Nchains)
logprob(max_ent_hypothesism, observations)

# Visualize the model and observations
plot_model(; sample..., observations=observations)

# Construct the conditional distribution
mcond = one_graben_one_geochem(observations, h)
mcond_w_samples = one_graben_one_geochem(observations, h, true)


Nsamples = 100
N = 10
alg = Gibbs(
    (MH(:ltop), 1),
    (MH(:lwidth), 1),
    (MH(:rtop), 1),
    (MH(:rwidth), 1),
    (MH(:center), 1),
    (MH(:angle), 1),
    (MH(), 1),
)

chns = mapreduce(c -> sample(mcond, alg, Nsamples), chainscat, 1:N)

HierarchicalMineralExploration.l

outputs = generated_quantities(mcond_w_samples, chns[end, :, :])

plots = []
for i in 1:10
    push!(
        plots,
        plot_model(;
            graben=outputs[end, i].graben,
            thickness=outputs[end, i].thickness,
            grade=outputs[end, i].grade,
            geochem=outputs[end, i].geochem,
            observations,
        ),
    )
end
plot(plots...; size=(1600, 4000), layout=(5, 2))
