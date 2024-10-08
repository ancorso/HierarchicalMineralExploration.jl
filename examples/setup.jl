K = 0.1 * Matern52Kernel() ∘ ScaleTransform(1.0 / 3.0)
N = 32 # Number of grid points
t₀ = ThicknessBackground(1.0, K) # Background thickness distribution (μ, kernel)
σₜ = 0.001 # Noise on thickness observation

graben_width = Normal(6, 2)
graben_top = Normal(21, 6)
graben_bottom = Normal(11, 6)
grabens = [
    GrabenDistribution(;
        N,
        μ=9.5,
        left_top=graben_bottom,
        left_width=graben_width,
        right_top=graben_bottom,
        right_width=graben_width,
    ),
    GrabenDistribution(;
        N,
        μ=7.5,
        left_top=graben_top,
        left_width=graben_width,
        right_top=graben_top,
        right_width=graben_width,
    ),
]
γ₀ = GradeBackground(0.0, K) # Background grade distribution
σᵧ = 0.001 # Standard deviation of the measurement noise on the grade
geochem_domains = [
    GeochemicalDomainDistribution(; N, μ=7.5, kernel=K),
    GeochemicalDomainDistribution(; N, μ=9.5, kernel=K),
]

hypotheses = OrderedDict(
    1 => Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains), # Two grabens, two geochem domains
    2 => Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, [geochem_domains[1]]), # Two grabens, single geochem domain
    3 => Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, geochem_domains), # Single graben, two geochem domains
    4 => Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]), # Single graben, single geochem domain
)
max_ent_hypothesis = MaxEntropyHypothesis(Normal(8, 8), Normal(8, 8))

max_ent_hypothesis2 = MaxEntropyHypothesis(
    MixtureModel([Normal(1.0, sqrt(0.1)), Normal(7.5, sqrt(0.1)), Normal(9.5, sqrt(0.1))], [0.5, 0.25, 0.25]), #thickness
    MixtureModel([Normal(0.0, sqrt(0.1)), Normal(7.5, sqrt(0.1)), Normal(9.5, sqrt(0.1))], [0.5, 0.25, 0.25]) # grade
)

# ## Visualize the different hypotheses
# pomdp = HierarchicalMinExPOMDP(; extraction_cost=0)
# hi = 2

# h = hypotheses[hi]
# m_type = turing_model(h)
# m = m_type(Dict(), h, true)
# sample_states = [m() for i in 1:100]

# rs = [extraction_reward(pomdp, s) for s in sample_states]
# mean(rs)
# extrema(rs)
# histogram(rs; title="Extraction reward distribution (hypothesis $hi)", xlabel="Reward")

# anim = @animate for (i, p) in enumerate(sample_states)
#     plot_state(p)
#     plot!(; title="i=$i")
# end
# gif(anim, "particles.gif"; fps=2)
