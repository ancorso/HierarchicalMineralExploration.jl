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

# Return the logpdf of the domain variables
function domain_logpdf(h::Hypothesis, chainval)
    logpdf_val = 0.0
    for (i, g) in enumerate(h.grabens)
        ltop = chainval[1, Symbol("ltop$i"), 1]
        lwidth = chainval[1, Symbol("lwidth$i"), 1]
        rtop = chainval[1, Symbol("rtop$i"), 1]
        rwidth = chainval[1, Symbol("rwidth$i"), 1]
        logpdf_val += logpdf(g, ltop, lwidth, rtop, rwidth)
    end
    for (i, g) in enumerate(h.geochem_domains)
        cx = chainval[1, Symbol("cx$i"), 1]
        cy = chainval[1, Symbol("cy$i"), 1]
        r1 = chainval[1, Symbol("r1$i"), 1]
        r2 = chainval[1, Symbol("r2$i"), 1]
        r3 = chainval[1, Symbol("r3$i"), 1]
        r4 = chainval[1, Symbol("r4$i"), 1]
        r5 = chainval[1, Symbol("r5$i"), 1]
        r6 = chainval[1, Symbol("r6$i"), 1]
        r7 = chainval[1, Symbol("r7$i"), 1]
        r8 = chainval[1, Symbol("r8$i"), 1]
        r9 = chainval[1, Symbol("r9$i"), 1]
        r10 = chainval[1, Symbol("r10$i"), 1]

        logpdf_val += logpdf(g, cx, cy, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10)
    end
    return logpdf_val
end


"""
    A hypothesis that assumes the maximum entropy distribution over the observations
"""
struct MaxEntropyHypothesis
    thickness_dist
    grade_dist
end

"""
    turing_model(hypothesis)

    Return the appropriate Turing model function for the given `hypothesis`.
"""
function turing_model(hypothesis)
    if hypothesis isa Hypothesis
        if length(hypothesis.grabens) == 2 && length(hypothesis.geochem_domains) == 2
            return two_graben_two_geochem
        elseif length(hypothesis.grabens) == 2 && length(hypothesis.geochem_domains) == 1
            return two_graben_one_geochem
        elseif length(hypothesis.grabens) == 1 && length(hypothesis.geochem_domains) == 2
            return one_graben_two_geochem
        elseif length(hypothesis.grabens) == 1 && length(hypothesis.geochem_domains) == 1
            return one_graben_one_geochem
        else
            error("this configuration is not supported ") #TODO
        end
    else
        error("Unknown hypothesis type")
    end
end

"""
Produce the default sampling algorithm
"""
function default_alg(hypothesis)
    if hypothesis isa Hypothesis
        if length(hypothesis.grabens) == 2 && length(hypothesis.geochem_domains) == 2
            return Gibbs(ESS(:ltop1), ESS(:rtop1), ESS(:lwidth1), ESS(:rwidth1), ESS(:ltop2), ESS(:rtop2), ESS(:lwidth2), ESS(:rwidth2), ESS(:cx1), ESS(:cy1), ESS(:r11), ESS(:r21), ESS(:r31), ESS(:r41), ESS(:r51), ESS(:r61), ESS(:r71), ESS(:r81), ESS(:r91), ESS(:r101), ESS(:cx2), ESS(:cy2), ESS(:r12), ESS(:r22), ESS(:r32), ESS(:r42), ESS(:r52), ESS(:r62), ESS(:r72), ESS(:r82), ESS(:r92), ESS(:r102))
        elseif length(hypothesis.grabens) == 2 && length(hypothesis.geochem_domains) == 1
            return Gibbs(ESS(:ltop1), ESS(:rtop1), ESS(:lwidth1), ESS(:rwidth1), ESS(:ltop2), ESS(:rtop2), ESS(:lwidth2), ESS(:rwidth2), ESS(:cx1), ESS(:cy1), ESS(:r11), ESS(:r21), ESS(:r31), ESS(:r41), ESS(:r51), ESS(:r61), ESS(:r71), ESS(:r81), ESS(:r91), ESS(:r101))
        elseif length(hypothesis.grabens) == 1 && length(hypothesis.geochem_domains) == 2
            return Gibbs(ESS(:ltop1), ESS(:rtop1), ESS(:lwidth1), ESS(:rwidth1), ESS(:cx1), ESS(:cy1), ESS(:r11), ESS(:r21), ESS(:r31), ESS(:r41), ESS(:r51), ESS(:r61), ESS(:r71), ESS(:r81), ESS(:r91), ESS(:r101), ESS(:cx2), ESS(:cy2), ESS(:r12), ESS(:r22), ESS(:r32), ESS(:r42), ESS(:r52), ESS(:r62), ESS(:r72), ESS(:r82), ESS(:r92), ESS(:r102))
        elseif length(hypothesis.grabens) == 1 && length(hypothesis.geochem_domains) == 1
            return Gibbs(ESS(:ltop1), ESS(:rtop1), ESS(:lwidth1), ESS(:rwidth1), ESS(:cx1), ESS(:cy1), ESS(:r11), ESS(:r21), ESS(:r31), ESS(:r41), ESS(:r51), ESS(:r61), ESS(:r71), ESS(:r81), ESS(:r91), ESS(:r101))
        else
            error("this configuration is not supported ") #TODO
        end
    else
        error("Unknown hypothesis type")
    end
end

# """
#     loglikelihood(hypothesis, observations, alg, Nsamples, Nchains)

#     Compute the log-likelihood of the observations given the model
# """
# function logprob(h::Hypothesis, observations;alg=default_alg(h), Nsamples, Nchains)
#     m = turing_model(h)
#     mcond = m(observations, h)

#     # Run the chains and store the outcomes
#     c = mapreduce(c -> Turing.sample(mcond, alg, Nsamples), chainscat, 1:Nchains)
#     outputs = generated_quantities(mcond, c[end, :, :]) # NOTE: this is an overestimate by just computing p(o | d)
#     return maximum(outputs) # NOTE: this is an overestimate by taking the max
# end

"""
    loglikelihood(hypothesis, observations)

    Compute the log-likelihood under the Max Entropy Hypothesis
"""
function logprob(h::MaxEntropyHypothesis, observations)
    acc_logpdf = 0.0
    # Compute the log-likelihood of the observations given the model
    for (pt, obs) in observations
        acc_logpdf += logpdf(h.thickness_dist, obs.thickness)
        acc_logpdf += logpdf(h.grade_dist, obs.grade)
    end
    return acc_logpdf
end

"""
    pull out the points and observations where check evaluates to true
"""
function getobs(observations, sym, check=(x) -> true)
    length(observations) == 0 && return [], []

    pts = collect(keys(observations)) # Pull out the observation pts
    obs = [v[sym] for v in values(observations)] # Pull out the desired observations

    # Only keep the points that have check evaluated to true
    keep = [check(x) for x in pts]
    pts = pts[keep]
    obs = obs[keep]

    return pts, obs
end

@model function two_graben_two_geochem(observations, h::Hypothesis, return_samples=false)
    marginal_loglikelihood = 0.0

    N = h.N
    # This defines the grid points (used multiple times below for GP sampling)
    xs = [[i, j] for i in 1:N, j in 1:N][:]

    # Sample the Graben(s)
    graben_dist1 = h.grabens[1]
    ltop1 ~ graben_dist1.left_top
    lwidth1 ~ graben_dist1.left_width
    rtop1 ~ graben_dist1.right_top
    rwidth1 ~ graben_dist1.right_width
    graben1 = draw_graben(N, ltop1, lwidth1, rtop1, rwidth1)

    graben_dist2 = h.grabens[2]
    ltop2 ~ graben_dist2.left_top
    lwidth2 ~ graben_dist2.left_width
    rtop2 ~ graben_dist2.right_top
    rwidth2 ~ graben_dist2.right_width
    graben2 = draw_graben(N, ltop2, lwidth2, rtop2, rwidth2)

    # Construct the thickness model (Note that spatial correlations are shared across domains)
    μt(x) = begin
        graben1[x...] == 1.0 && return graben_dist1.μ
        graben2[x...] == 1.0 && return graben_dist2.μ
        return h.thickness_background.μ
    end
    thicknessGP = GP(μt, h.thickness_background.kernel)
    if length(observations) > 0
        pts, obs = getobs(observations, :thickness)
        thicknessGPx = thicknessGP(pts, h.σ_thickness) # Conditions the GP on the observation points
        lgprob = logpdf(thicknessGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        thicknessGP = posterior(thicknessGPx, obs)
    end

    # Sample the geochemical domain
    geochem_dist1 = h.geochem_domains[1]
    cx1 ~ geochem_dist1.cx
    cy1 ~ geochem_dist1.cy
    r11 ~ geochem_dist1.r1
    r21 ~ geochem_dist1.r2
    r31 ~ geochem_dist1.r3
    r41 ~ geochem_dist1.r4
    r51 ~ geochem_dist1.r5
    r61 ~ geochem_dist1.r6
    r71 ~ geochem_dist1.r7
    r81 ~ geochem_dist1.r8
    r91 ~ geochem_dist1.r9
    r101 ~ geochem_dist1.r10
    center1 = (cx1, cy1)
    rs1 = [r11, r21, r31, r41, r51, r61, r71, r81, r91, r101]
    geochem1 = draw_geochemical_domain(N, center1, rs1)

    geochem_dist2 = h.geochem_domains[2]
    cx2 ~ geochem_dist2.cx
    cy2 ~ geochem_dist2.cy
    r12 ~ geochem_dist2.r1
    r22 ~ geochem_dist2.r2
    r32 ~ geochem_dist2.r3
    r42 ~ geochem_dist2.r4
    r52 ~ geochem_dist2.r5
    r62 ~ geochem_dist2.r6
    r72 ~ geochem_dist2.r7
    r82 ~ geochem_dist2.r8
    r92 ~ geochem_dist2.r9
    r102 ~ geochem_dist2.r10
    center2 = (cx2, cy2)
    rs2 = [r12, r22, r32, r42, r52, r62, r72, r82, r92, r102]
    geochem2 = draw_geochemical_domain(N, center2, rs2)

    # Inside the first geochemical domain model
    geochemGP1 = GP((x) -> geochem_dist1.μ, geochem_dist1.kernel)
    ingeochem1not2(x) = (geochem1[x...] == 1.0) && !(geochem2[x...] == 1.0)
    pts, obs = getobs(observations, :grade, ingeochem1not2)
    if length(obs) > 0
        geochemGP1x = geochemGP1(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP1x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP1 = posterior(geochemGP1x, obs)
    end

    # Inside the second geochemical domain model
    geochemGP2 = GP((x) -> geochem_dist2.μ, geochem_dist2.kernel)
    ingeochem2(x) = geochem2[x...] == 1.0
    pts, obs = getobs(observations, :grade, ingeochem2)
    if length(obs) > 0
        geochemGP2x = geochemGP2(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP2x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP2 = posterior(geochemGP2x, obs)
    end

    # Background model and sample
    backgroundGradeGP = GP((x) -> h.grade_background.μ, h.grade_background.kernel)
    inbackground(x) = !(geochem1[x...] == 1.0) && !(geochem2[x...] == 1.0)
    pts, obs = getobs(observations, :grade, inbackground)
    if length(obs) > 0
        backgroundGradeGPx = backgroundGradeGP(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(backgroundGradeGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        backgroundGradeGP = posterior(backgroundGradeGPx, obs)
    end

    if return_samples
        # Create the structural domain
        structural_fn(x) = begin
            graben1[x...] == 1.0 && return 1.0
            graben2[x...] == 1.0 && return 2.0
            return 0.0
        end
        structural = reshape(structural_fn.(xs), N, N)

        # Sample the thickness
        thickness = reshape(rand(thicknessGP(xs, h.σ_thickness)), N, N)

        # Construct the geochemical domain
        ibackground = [inbackground(x) for x in xs]
        igeochem1 = [ingeochem1not2(x) for x in xs]
        igeochem2 = [ingeochem2(x) for x in xs]
        geochemdomain = zeros(N, N)
        geochemdomain[igeochem1] .= 1.0
        geochemdomain[igeochem2] .= 2.0

        g_geochem1 = rand(geochemGP1(xs[igeochem1], h.σ_grade))
        g_geochem2 = rand(geochemGP2(xs[igeochem2], h.σ_grade))
        g_background = rand(backgroundGradeGP(xs[ibackground], h.σ_grade))

        # Compose the grade observation
        grade = zeros(N, N)[:]
        grade[igeochem1] .= g_geochem1
        grade[igeochem2] .= g_geochem2
        grade[ibackground] .= g_background
        grade = reshape(grade, N, N)

        return (; structural, geochemdomain, thickness, grade)
    end
    return marginal_loglikelihood
end

@model function two_graben_one_geochem(observations, h::Hypothesis, return_samples=false)
    marginal_loglikelihood = 0.0
    N = h.N
    # This defines the grid points (used multiple times below for GP sampling)
    xs = [[i, j] for i in 1:N, j in 1:N][:]

    # Sample the Graben(s)
    graben_dist1 = h.grabens[1]
    ltop1 ~ graben_dist1.left_top
    lwidth1 ~ graben_dist1.left_width
    rtop1 ~ graben_dist1.right_top
    rwidth1 ~ graben_dist1.right_width
    graben1 = draw_graben(N, ltop1, lwidth1, rtop1, rwidth1)

    graben_dist2 = h.grabens[2]
    ltop2 ~ graben_dist2.left_top
    lwidth2 ~ graben_dist2.left_width
    rtop2 ~ graben_dist2.right_top
    rwidth2 ~ graben_dist2.right_width
    graben2 = draw_graben(N, ltop2, lwidth2, rtop2, rwidth2)

    # Construct the thickness model (Note that spatial correlations are shared across domains)
    μt(x) = begin
        graben1[x...] == 1.0 && return graben_dist1.μ
        graben2[x...] == 1.0 && return graben_dist2.μ
        return h.thickness_background.μ
    end
    thicknessGP = GP(μt, h.thickness_background.kernel)
    if length(observations) > 0
        pts, obs = getobs(observations, :thickness)
        thicknessGPx = thicknessGP(pts, h.σ_thickness) # Conditions the GP on the observation points
        lgprob = logpdf(thicknessGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        thicknessGP = posterior(thicknessGPx, obs)
    end

    # Sample the geochemical domain
    geochem_dist1 = h.geochem_domains[1]
    cx1 ~ geochem_dist1.cx
    cy1 ~ geochem_dist1.cy
    r11 ~ geochem_dist1.r1
    r21 ~ geochem_dist1.r2
    r31 ~ geochem_dist1.r3
    r41 ~ geochem_dist1.r4
    r51 ~ geochem_dist1.r5
    r61 ~ geochem_dist1.r6
    r71 ~ geochem_dist1.r7
    r81 ~ geochem_dist1.r8
    r91 ~ geochem_dist1.r9
    r101 ~ geochem_dist1.r10
    center1 = (cx1, cy1)
    rs1 = [r11, r21, r31, r41, r51, r61, r71, r81, r91, r101]
    geochem1 = draw_geochemical_domain(N, center1, rs1)

    # Inside the first geochemical domain model
    geochemGP1 = GP((x) -> geochem_dist1.μ, geochem_dist1.kernel)
    ingeochem1(x) = geochem1[x...] == 1.0
    pts, obs = getobs(observations, :grade, ingeochem1)
    if length(obs) > 0
        geochemGP1x = geochemGP1(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP1x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP1 = posterior(geochemGP1x, obs)
    end

    # Background model and sample
    backgroundGradeGP = GP((x) -> h.grade_background.μ, h.grade_background.kernel)
    inbackground(x) = !(geochem1[x...] == 1.0)
    pts, obs = getobs(observations, :grade, inbackground)
    if length(obs) > 0
        backgroundGradeGPx = backgroundGradeGP(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(backgroundGradeGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        backgroundGradeGP = posterior(backgroundGradeGPx, obs)
    end

    if return_samples
        # Create the structural domain
        structural_fn(x) = begin
            graben1[x...] == 1.0 && return 1.0
            graben2[x...] == 1.0 && return 2.0
            return 0.0
        end
        structural = reshape(structural_fn.(xs), N, N)

        # Sample the thickness
        thickness = reshape(rand(thicknessGP(xs, h.σ_thickness)), N, N)

        # Construct the geochemical domain
        ibackground = [inbackground(x) for x in xs]
        igeochem1 = [ingeochem1(x) for x in xs]
        geochemdomain = zeros(N, N)
        geochemdomain[igeochem1] .= 1.0

        g_geochem1 = rand(geochemGP1(xs[igeochem1], h.σ_grade))
        g_background = rand(backgroundGradeGP(xs[ibackground], h.σ_grade))

        # Compose the grade observation
        grade = zeros(N, N)[:]
        grade[igeochem1] .= g_geochem1
        grade[ibackground] .= g_background
        grade = reshape(grade, N, N)

        return (; structural, geochemdomain, thickness, grade)
    end
    return marginal_loglikelihood
end

@model function one_graben_two_geochem(observations, h::Hypothesis, return_samples=false)
    marginal_loglikelihood = 0.0
    N = h.N
    # This defines the grid points (used multiple times below for GP sampling)
    xs = [[i, j] for i in 1:N, j in 1:N][:]

    # Sample the Graben(s)
    graben_dist1 = h.grabens[1]
    ltop1 ~ graben_dist1.left_top
    lwidth1 ~ graben_dist1.left_width
    rtop1 ~ graben_dist1.right_top
    rwidth1 ~ graben_dist1.right_width
    graben1 = draw_graben(N, ltop1, lwidth1, rtop1, rwidth1)

    # Construct the thickness model (Note that spatial correlations are shared across domains)
    μt(x) = begin
        graben1[x...] == 1.0 && return graben_dist1.μ
        return h.thickness_background.μ
    end
    thicknessGP = GP(μt, h.thickness_background.kernel)
    if length(observations) > 0
        pts, obs = getobs(observations, :thickness)
        thicknessGPx = thicknessGP(pts, h.σ_thickness) # Conditions the GP on the observation points
        lgprob = logpdf(thicknessGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        thicknessGP = posterior(thicknessGPx, obs)
    end

    # Sample the geochemical domain
    geochem_dist1 = h.geochem_domains[1]
    cx1 ~ geochem_dist1.cx
    cy1 ~ geochem_dist1.cy
    r11 ~ geochem_dist1.r1
    r21 ~ geochem_dist1.r2
    r31 ~ geochem_dist1.r3
    r41 ~ geochem_dist1.r4
    r51 ~ geochem_dist1.r5
    r61 ~ geochem_dist1.r6
    r71 ~ geochem_dist1.r7
    r81 ~ geochem_dist1.r8
    r91 ~ geochem_dist1.r9
    r101 ~ geochem_dist1.r10
    center1 = (cx1, cy1)
    rs1 = [r11, r21, r31, r41, r51, r61, r71, r81, r91, r101]
    geochem1 = draw_geochemical_domain(N, center1, rs1)

    geochem_dist2 = h.geochem_domains[2]
    cx2 ~ geochem_dist2.cx
    cy2 ~ geochem_dist2.cy
    r12 ~ geochem_dist2.r1
    r22 ~ geochem_dist2.r2
    r32 ~ geochem_dist2.r3
    r42 ~ geochem_dist2.r4
    r52 ~ geochem_dist2.r5
    r62 ~ geochem_dist2.r6
    r72 ~ geochem_dist2.r7
    r82 ~ geochem_dist2.r8
    r92 ~ geochem_dist2.r9
    r102 ~ geochem_dist2.r10
    center2 = (cx2, cy2)
    rs2 = [r12, r22, r32, r42, r52, r62, r72, r82, r92, r102]
    geochem2 = draw_geochemical_domain(N, center2, rs2)

    # Inside the first geochemical domain model
    geochemGP1 = GP((x) -> geochem_dist1.μ, geochem_dist1.kernel)
    ingeochem1not2(x) = (geochem1[x...] == 1.0) && !(geochem2[x...] == 1.0)
    pts, obs = getobs(observations, :grade, ingeochem1not2)
    if length(obs) > 0
        geochemGP1x = geochemGP1(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP1x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP1 = posterior(geochemGP1x, obs)
    end

    # Inside the second geochemical domain model
    geochemGP2 = GP((x) -> geochem_dist2.μ, geochem_dist2.kernel)
    ingeochem2(x) = geochem2[x...] == 1.0
    pts, obs = getobs(observations, :grade, ingeochem2)
    if length(obs) > 0
        geochemGP2x = geochemGP2(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP2x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP2 = posterior(geochemGP2x, obs)
    end

    # Background model and sample
    backgroundGradeGP = GP((x) -> h.grade_background.μ, h.grade_background.kernel)
    inbackground(x) = !(geochem1[x...] == 1.0) && !(geochem2[x...] == 1.0)
    pts, obs = getobs(observations, :grade, inbackground)
    if length(obs) > 0
        backgroundGradeGPx = backgroundGradeGP(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(backgroundGradeGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        backgroundGradeGP = posterior(backgroundGradeGPx, obs)
    end

    if return_samples
        # Create the structural domain
        structural_fn(x) = begin
            graben1[x...] == 1.0 && return 1.0
            return 0.0
        end
        structural = reshape(structural_fn.(xs), N, N)

        # Sample the thickness
        thickness = reshape(rand(thicknessGP(xs, h.σ_thickness)), N, N)

        # Construct the geochemical domain
        ibackground = [inbackground(x) for x in xs]
        igeochem1 = [ingeochem1not2(x) for x in xs]
        igeochem2 = [ingeochem2(x) for x in xs]
        geochemdomain = zeros(N, N)
        geochemdomain[igeochem1] .= 1.0
        geochemdomain[igeochem2] .= 2.0

        g_geochem1 = rand(geochemGP1(xs[igeochem1], h.σ_grade))
        g_geochem2 = rand(geochemGP2(xs[igeochem2], h.σ_grade))
        g_background = rand(backgroundGradeGP(xs[ibackground], h.σ_grade))

        # Compose the grade observation
        grade = zeros(N, N)[:]
        grade[igeochem1] .= g_geochem1
        grade[igeochem2] .= g_geochem2
        grade[ibackground] .= g_background
        grade = reshape(grade, N, N)

        return (; structural, geochemdomain, thickness, grade)
    end
    return marginal_loglikelihood
end

@model function one_graben_one_geochem(observations, h::Hypothesis, return_samples=false)
    marginal_loglikelihood = 0.0
    N = h.N
    # This defines the grid points (used multiple times below for GP sampling)
    xs = [[i, j] for i in 1:N, j in 1:N][:]

    # Sample the Graben(s)
    graben_dist1 = h.grabens[1]
    ltop1 ~ graben_dist1.left_top
    lwidth1 ~ graben_dist1.left_width
    rtop1 ~ graben_dist1.right_top
    rwidth1 ~ graben_dist1.right_width
    graben1 = draw_graben(N, ltop1, lwidth1, rtop1, rwidth1)

    # Construct the thickness model (Note that spatial correlations are shared across domains)
    μt(x) = begin
        graben1[x...] == 1.0 && return graben_dist1.μ
        return h.thickness_background.μ
    end
    thicknessGP = GP(μt, h.thickness_background.kernel)
    if length(observations) > 0
        pts, obs = getobs(observations, :thickness)
        thicknessGPx = thicknessGP(pts, h.σ_thickness) # Conditions the GP on the observation points
        lgprob = logpdf(thicknessGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        thicknessGP = posterior(thicknessGPx, obs)
    end

    # Sample the geochemical domain
    geochem_dist1 = h.geochem_domains[1]
    cx1 ~ geochem_dist1.cx
    cy1 ~ geochem_dist1.cy
    r11 ~ geochem_dist1.r1
    r21 ~ geochem_dist1.r2
    r31 ~ geochem_dist1.r3
    r41 ~ geochem_dist1.r4
    r51 ~ geochem_dist1.r5
    r61 ~ geochem_dist1.r6
    r71 ~ geochem_dist1.r7
    r81 ~ geochem_dist1.r8
    r91 ~ geochem_dist1.r9
    r101 ~ geochem_dist1.r10
    center1 = (cx1, cy1)
    rs1 = [r11, r21, r31, r41, r51, r61, r71, r81, r91, r101]
    geochem1 = draw_geochemical_domain(N, center1, rs1)

    # Inside the first geochemical domain model
    geochemGP1 = GP((x) -> geochem_dist1.μ, geochem_dist1.kernel)
    ingeochem1(x) = geochem1[x...] == 1.0
    pts, obs = getobs(observations, :grade, ingeochem1)
    if length(obs) > 0
        geochemGP1x = geochemGP1(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(geochemGP1x, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        geochemGP1 = posterior(geochemGP1x, obs)
    end

    # Background model and sample
    backgroundGradeGP = GP((x) -> h.grade_background.μ, h.grade_background.kernel)
    inbackground(x) = !(geochem1[x...] == 1.0)
    pts, obs = getobs(observations, :grade, inbackground)
    if length(obs) > 0
        backgroundGradeGPx = backgroundGradeGP(pts, h.σ_grade) # Conditions the GP on the observation points
        lgprob = logpdf(backgroundGradeGPx, obs)
        marginal_loglikelihood += lgprob
        Turing.@addlogprob! lgprob # Adds the loglikelihood
        backgroundGradeGP = posterior(backgroundGradeGPx, obs)
    end

    if return_samples
        # Create the structural domain
        structural_fn(x) = begin
            graben1[x...] == 1.0 && return 1.0
            return 0.0
        end
        structural = reshape(structural_fn.(xs), N, N)

        # Sample the thickness
        thickness = reshape(rand(thicknessGP(xs, h.σ_thickness)), N, N)

        # Construct the geochemical domain
        ibackground = [inbackground(x) for x in xs]
        igeochem1 = [ingeochem1(x) for x in xs]
        geochemdomain = zeros(N, N)
        geochemdomain[igeochem1] .= 1.0

        g_geochem1 = rand(geochemGP1(xs[igeochem1], h.σ_grade))
        g_background = rand(backgroundGradeGP(xs[ibackground], h.σ_grade))

        # Compose the grade observation
        grade = zeros(N, N)[:]
        grade[igeochem1] .= g_geochem1
        grade[ibackground] .= g_background
        grade = reshape(grade, N, N)

        return (; structural, geochemdomain, thickness, grade)
    end
    return marginal_loglikelihood
end