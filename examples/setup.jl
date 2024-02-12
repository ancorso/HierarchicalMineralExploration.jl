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
