using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs

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

h = Hypothesis(
    N, thickness_background, σ_t, grabens, grade_background, σ_grade, geochem_domains
)

# Sample one model
m = one_graben_one_geochem(Dict(), h)
res = m()

# Obtain some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:10]
observations = Dict(
    p => (thickness=res.thickness[p...], grade=res.grade[p...]) for p in pts
)

# Visualize the model and observations
plot_model(;
    graben=res.graben,
    geochem=res.geochem,
    thickness=res.thickness,
    grade=res.grade,
    observations=observations,
)

# Construct the conditional distribution
mcond = one_graben_one_geochem(observations, h)

Nsamples = 100
specs = MH()
num_chains = 10
chains = mapreduce(c -> sample(mcond, specs, Nsamples), chainscat, 1:num_chains)

fieldnames(typeof(chains))
chains.value[:, 5, 1]

plot(chains.value[:, 5, 1])

chains.value
plot(chains)

outputs = generated_quantities(mcond, chains)

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
