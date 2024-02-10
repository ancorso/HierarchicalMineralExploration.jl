using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using POMDPs
using ParticleFilters


## Setup mineral system models
KERNEL = Matern52Kernel() ∘ ScaleTransform(1.0 / 3.0)
N = 32 # Number of grid points
thickness_background = ThicknessBackground(1.0, KERNEL) # Background thickness distribution (μ, kernel
σ_t = 0.01 # Noise on thickness observation
grabens = [GrabenDistribution(; N, μ=8.0), GrabenDistribution(; N, μ=8.0)]
grade_background = GradeBackground(0.0, KERNEL) # Background grade distribution
σ_grade = 0.01 # Standard deviation of the measurement noise on the grade
geochem_domains = [
    GeochemicalDomainDistribution(; N, μ=15.0, kernel=KERNEL),
    GeochemicalDomainDistribution(; N, μ=15.0, kernel=KERNEL),
]

## Ground truth
h_gt = Hypothesis(
    N, thickness_background, σ_t, grabens, grade_background, σ_grade, geochem_domains
)
m_gt = one_graben_one_geochem(Dict(), h_gt)
res_gt = m_gt()
s_gt = HierarchicalMinExState(res_gt.thickness, res_gt.grade)

## Define the pomdp
pomdp = HierarchicalMinExPOMDP()

## Belief configuration
h1 = Hypothesis(
    N, thickness_background, σ_t, grabens, grade_background, σ_grade, geochem_domains
)
hypotheses = [h1]

Nparticles = 100
b0 = particle_collection(hypotheses, Nparticles)
up = BootstrapFilter(pomdp, Nparticles)


s = deepcopy(s_gt)
while !isterminal(pomdp, s)
    a = action(pomdp, s)
    o = generate_o(pomdp, s, a)
    s = generate_s(pomdp, s, a, o)
    b = update(up, s, a, o, b0)
    b0 = b
end
```

