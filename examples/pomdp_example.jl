using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using POMDPs
using ParticleFilters


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

## Ground truth
h_gt = Hypothesis(
    N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains
)
m_gt = one_graben_one_geochem(Dict(), h_gt, true)
res_gt = m_gt()
s_gt = HierarchicalMinExState(res_gt.thickness, res_gt.grade)

## Define the pomdp
pomdp = HierarchicalMinExPOMDP()

## Belief configuration
h1 = Hypothesis(
    N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]
)
hypotheses = [h1]


Nsamples = 25
Nparticles = 100
alg = Gibbs(
    (MH(:ltop), 1),
    (MH(:lwidth), 1),
    (MH(:rtop), 1),
    (MH(:rwidth), 1),
    (MH(:center), 1),
    (MH(:angle), 1),
    (MH(), 1)
)
up = MCMCUpdater(alg, Nsamples, hypotheses, Nparticles, Dict())
b0 = initialize_belief(up, nothing)

pts = [[rand(1:N), rand(1:N)] for _ in 1:10]
up.observations = Dict(
    p => (thickness=res_gt.thickness[p...], grade=res_gt.grade[p...]) for p in pts
)

collect(keys(up.observations))
b10 = update(up, b0, pts[end], up.observations[pts[end]])

plot_state(b10[50], up.observations)


s = deepcopy(s_gt)
while !isterminal(pomdp, s)
    a = action(pomdp, s)
    o = generate_o(pomdp, s, a)
    s = generate_s(pomdp, s, a, o)
    b = update(up, s, a, o, b0)
    b0 = b
end
