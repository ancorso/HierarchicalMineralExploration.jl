K = 0.1 * Matern52Kernel() ∘ ScaleTransform(1.0 / 3.0) 
N = 32 # Number of grid points
t₀ = ThicknessBackground(1.0, K) # Background thickness distribution (μ, kernel)
σₜ = 0.001 # Noise on thickness observation
grabens = [GrabenDistribution(; N, μ=8.0), GrabenDistribution(; N, μ=15.0)]
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
max_ent_hypothesis = MaxEntropyHypothesis(Normal(8, 8), Normal(8,8))

# Functions to get the hypothesis class of a state (if that info wasn't stored)
structural1(s) = any((s.thickness[:] .> 5) .& (s.thickness[:] .< 10))
structural2(s) = any((s.thickness[:] .> 10) .& (s.thickness[:] .< 20))
geochem1(s) = any((s.grade[:] .> 10) .& (s.grade[:] .< 20))
geochem2(s) = any((s.grade[:] .> 5) .& (s.grade[:] .< 10))
function get_hypothesis(s)
    if structural1(s) && structural2(s) && geochem1(s) && geochem2(s)
        return 1
    elseif structural1(s) && structural2(s) && geochem1(s)
        return 2
    elseif structural1(s) && geochem1(s) && geochem2(s)
        return 3
    elseif structural1(s) && geochem1(s)
        return 4
    else
        error("not sure what this is")
    end
end