"""
    Geological hypothesis that combines different structural and geochemical domains
"""
struct Hypothesis
    N::Int # Size of the grid
    thickness_background::ThicknessBackground # Background thickness distribution
    σ_thickness::Float64 # Standard deviation of the measurement noise on the thickness
    grabens::Vector{GrabenDistribution} # Graben shape distributions
    grade_background::GradeBackground # Background grade distribution
    σ_grade::Float64 # Standard deviation of the measurement noise on the grade
    geochem_domains::Vector{GeochemicalDomainDistribution} # Geochemical domain distributions
end

"""
    Constructs the model for 1 graben and 1 geochemical domain
"""
@model function one_graben_one_geochem(observations, h::Hypothesis)
    N = h.N
    # This defines the grid points (used multiple times below for GP sampling)
    x = hcat([[i, j] for i in 1:N, j in 1:N][:]...)

    # Sample the Graben
    graben_dist = h.grabens[1]
    ltop ~ graben_dist.left_top
    lwidth ~ graben_dist.left_width
    rtop ~ graben_dist.right_top
    rwidth ~ graben_dist.right_width
    graben = draw_graben(N, ltop, lwidth, rtop, rwidth)

    # Construct the thickness model
    μt(x) = graben[x...] == 1.0 ? graben_dist.μ : h.thickness_background.μ
    thicknessGP = GP(μt, h.thickness_background.kernel)
    if length(observations) > 0
        pts = collect(keys(observations)) # Pull out the observation pts
        obs = [v.thickness for v in values(observations)] # Pull out the thickness observations
        fx = thicknessGP(pts, h.σ_thickness) # Conditions the GP on the observation points
        Turing.@addlogprob! logpdf(fx, obs) # Adds the loglikelihood
        thicknessGP = posterior(fx, obs) # Creates the posterior disribution
    end
    # Now sample our GP to get the thickness distribution at all points
    thickness = reshape(rand(thicknessGP(x, h.σ_thickness)), N, N)

    # Sample the geochemical domain
    center ~ h.geochem_domains[1].center
    pts = fill(missing, length(h.geochem_domains[1].points))
    for i in eachindex(pts)
        pts[i] ~ h.geochem_domains[1].points[i]
    end
    angle ~ h.geochem_domains[1].angle
    geochem = draw_geochemical_domain(N, center, pts, angle)

    # Background model and sample
    backgroundGradeGP = GP((x) -> h.grade_background.μ, h.grade_background.kernel)
    if length(observations) > 0
        pts = collect(keys(observations)) # Pull out the observation pts
        obs = [v.grade for v in values(observations)] # Pull out the grade observations

        # Only keep the points that are in the background domain
        keep = [geochem[x...] == 0.0 for x in pts]
        pts = pts[keep]
        obs = obs[keep]

        if length(pts) > 0
            fx = backgroundGradeGP(pts, h.σ_grade) # Conditions the GP on the observation points
            Turing.@addlogprob! logpdf(fx, obs) # Adds the loglikelihood
            backgroundGradeGP = posterior(fx, obs) # Creates the posterior disribution
        end
    end
    g_background = reshape(rand(backgroundGradeGP(x, h.σ_grade)), N, N)

    # Inside geochemical domain model and sample
    geochem_dist = h.geochem_domains[1]
    geochemGP = GP((x) -> geochem_dist.μ, geochem_dist.kernel)
    if length(observations) > 0
        pts = collect(keys(observations)) # Pull out the observation pts
        obs = [v.grade for v in values(observations)] # Pull out the grade observations

        # Only keep the points that are in the geochem domain
        keep = [geochem[x...] == 1.0 for x in pts]
        pts = pts[keep]
        obs = obs[keep]

        if length(pts) > 0
            fx = geochemGP(pts, h.σ_grade) # Conditions the GP on the observation points
            Turing.@addlogprob! logpdf(fx, obs) # Adds the loglikelihood
            geochemGP = posterior(fx, obs) # Creates the posterior disribution
        end
    end
    g_geochem = reshape(rand(geochemGP(x, h.σ_grade)), N, N)

    # Compose the grade observation
    grade = g_background .* (1.0 .- geochem) .+ g_geochem .* geochem

    # mineralization
    mineralization = grade .* thickness

    return (; graben, geochem, thickness, grade, mineralization)
end
