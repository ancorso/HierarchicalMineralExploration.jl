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
        μ=15.0,
        left_top=graben_top,
        left_width=graben_width,
        right_top=graben_top,
        right_width=graben_width,
    ),
    GrabenDistribution(;
        N,
        μ=8.0,
        left_top=graben_bottom,
        left_width=graben_width,
        right_top=graben_bottom,
        right_width=graben_width,
    ),
]
γ₀ = GradeBackground(0.0, K) # Background grade distribution
σᵧ = 0.001 # Standard deviation of the measurement noise on the grade
geochem_domains = [
    GeochemicalDomainDistribution(; N, μ=15.0, kernel=K),
    GeochemicalDomainDistribution(; N, μ=8.0, kernel=K),
]

hypotheses = [
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, geochem_domains), # Two grabens, two geochem domains
    Hypothesis(N, t₀, σₜ, grabens, γ₀, σᵧ, [geochem_domains[1]]), # Two grabens, single geochem domain
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, geochem_domains), # Single graben, two geochem domains
    Hypothesis(N, t₀, σₜ, [grabens[1]], γ₀, σᵧ, [geochem_domains[1]]), # Single graben, single geochem domain
]
max_ent_hypothesis = MaxEntropyHypothesis(Normal(8, 8), Normal(8, 8))


# ## Visualize the different hypotheses
# pomdp = HierarchicalMinExPOMDP()
# hi = 4

# h = hypotheses[hi]
# m_type = turing_model(h)
# m = m_type(Dict(), h, true)
# sample_states = [m() for i=1:100]

# rs = [extraction_reward(pomdp, s) for s in sample_states]
# histogram(rs, title="Extraction reward distribution (hypothesis $hi)", xlabel="Reward")

# anim = @animate for (i, p) in enumerate(sample_states)
#     plot_state(p)
#     plot!(; title="i=$i")
# end
# gif(anim, "particles.gif"; fps=2)
