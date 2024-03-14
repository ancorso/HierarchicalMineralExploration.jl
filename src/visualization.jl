"""
    plot_model(;structural, geochemical, thickness, grade, observations)

    Plot the structural, geochemical, thickness, and grade domains in a 2x2 grid.
"""
function plot_model(; structural, geochemdomain, thickness, grade, observations=Dict())
    # Collect the observations for easy plotting
    xpts = [p[1] for p in keys(observations)]
    ypts = [p[2] for p in keys(observations)]
    thickness_obs = [v.thickness for v in values(observations)]
    grade_obs = [v.grade for v in values(observations)]

    # Plot the structural domain
    p_graben = heatmap(structural'; cmap=:amp, colorbar=false)

    # Plot the thickness
    p_thickness = heatmap(thickness'; colorbar=false, clims=(0, 12))
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=thickness_obs,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
    else
        scatter!([],[], legend=false)
    end

    # Plot the geochemical domain
    p_geochem = heatmap(geochemdomain'; cmap=:algae, colorbar=false)

    # Plot the grade
    p_grade = heatmap(grade'; colorbar=false, clims=(0, 17))
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=grade_obs,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
    else
        scatter!([],[], legend=false)
    end
    # Combine the plots in 2x2 grid
    return plot(p_graben, p_thickness, p_geochem, p_grade; size=(800, 800), layout=(2, 2))
end

"""
Plot a state with observations
there is no domain information in the state
"""
function plot_state(s, observations=Dict())
    # Collect the observations for easy plotting
    xpts = [p[1] for p in keys(observations)]
    ypts = [p[2] for p in keys(observations)]
    thickness_obs = [v.thickness for v in values(observations)]
    grade_obs = [v.grade for v in values(observations)]

    # Plot the thickness
    p_thickness = heatmap(s.thickness'; colorbar=false, clims=(0, 12), title="Thickness")
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=thickness_obs,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:red,
        )
    else
        scatter!([],[], legend=false)
    end

    # Plot the grade
    p_grade = heatmap(s.grade'; colorbar=false, clims=(0, 17), title="Grade")
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=grade_obs,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:red,
            markerstrokewidth=2,
        )
    else
        scatter!([],[], legend=false)
    end
    # Combine the plots in 2x2 grid
    return plot(p_thickness, p_grade; size=(800, 400), layout=(1, 2))
end

function plot_mineralization(s, observations=Dict(); axis=false, kwargs...)
    # Collect the observations for easy plotting
    xpts = [p[1] for p in keys(observations)]
    ypts = [p[2] for p in keys(observations)]
    min_obs = [v.thickness * v.grade for v in values(observations)]

    mineralization = s.thickness .* s.grade

    # Plot the thickness
    pmin = heatmap(mineralization'; colorbar=false, clims=(0,250), cmap=:haline, title="Mineralization")
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            color=:red,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:red,
            markerstrokewidth=2,
        )
    else
        scatter!([],[], legend=false)
    end
    pstate = plot_state(s, observations)

    plot(pstate, pmin, layout=grid(2, 1, heights=[0.33 ,0.67]); margin=0Plots.mm, axis, kwargs...)
    # imgl = imfilter(s.thickness', ImageFiltering.Kernel.Laplacian());
    # imgl[abs.(imgl) .< 5] .= NaN
    # imgl[isfinite.(imgl)] .= 1000
    # heatmap!(imgl, colorbar=false, color=:grays, alpha=0.2)

    # imgl = imfilter(s.grade', ImageFiltering.Kernel.Laplacian());
    # imgl[abs.(imgl) .< 5] .= NaN
    # imgl[isfinite.(imgl)] .= 1000
    # heatmap!(imgl, colorbar=false, color=:grays, alpha=0.2)
end

function plot_belief(pomdp, step, hypothesis_fn, observations=Dict(); kwargs...)
    s, b = step.s, step.b

    # plot the belief (with ovbservations)
    xpts = [p[1] for p in keys(observations)]
    ypts = [p[2] for p in keys(observations)]
    min_obs = [v.thickness * v.grade for v in values(observations)]

    mineralization_maps = cat([s.thickness .* s.grade for s in particles(b)]..., dims=3)
    mean_mineralization = mean(mineralization_maps, dims=3)[:,:,1]
    std_mineralization = std(mineralization_maps, dims=3)[:,:,1]

    pmeanmin = heatmap(mean_mineralization'; colorbar=false, clims=(0,250), cmap=:haline, axis=false, title="Belief Mean", margin=Plots.mm)
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=min_obs,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:red,
            markerstrokewidth=2,
            cmap=:haline,
            margin=0Plots.mm
        )
    else
        scatter!([],[], legend=false)
    end

    pstdmin = heatmap(std_mineralization'; colorbar=false, clims=(0,60), cmap=:haline, axis=false, title="Belief Std", margin=0Plots.mm)

    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=0,
            markersize=7,
            legend=false,
            label="",
            markerstrokecolor=:red,
            markerstrokewidth=2,
            margin=0Plots.mm
        )
    else
        scatter!([],[], legend=false)
    end

    # Plot the hypothesis returns
    rs = [extraction_reward(pomdp, s) for s in particles(b)]
    b_hyps = [hypothesis_fn(s) for s in particles(b)]

    h1_returns = rs[findall(b_hyps .== 1)]
    h2_returns = rs[findall(b_hyps .== 2)]
    h3_returns = rs[findall(b_hyps .== 3)]
    h4_returns = rs[findall(b_hyps .== 4)]

    pret = histogram(h1_returns, label="", bins=-610:50:600, alpha=0.5, title="Returns", xlabel="Returns", linealpha=0., normalize=:probability, legend=:topleft, ylims=(0,0.5))
    histogram!(h2_returns, label="", bins=-620:50:600, alpha=0.5, linealpha=0, normalize=:probability)
    histogram!(h3_returns, label="", bins=-630:50:600, alpha=0.5, linealpha=0, normalize=:probability)
    histogram!(h4_returns, label="", bins=-640:50:600, alpha=0.5, linealpha=0, normalize=:probability)

    r = extraction_reward(pomdp, s)
    vline!([r], color=:red, label="Ground Truth")

    # Plot the distrribution over hypotheses
    phs = [sum(b_hyps .== i) for i in 1:4] ./ length(b_hyps)
    phypoth = bar([1], [phs[1]], ylims=(0,1), title="P(hypothesis)", xlabel="Hypothesis", label="", linealpha=0., alpha=0.5)
    bar!([2], [phs[2]], label="", linealpha=0., alpha=0.5)
    bar!([3], [phs[3]], label="", linealpha=0., alpha=0.5)
    bar!([4], [phs[4]], label="", linealpha=0., alpha=0.5)


    plot(pmeanmin, pstdmin, phypoth, pret; margin=0Plots.mm, layout=grid(2, 2, heights=[0.67, 0.33]), kwargs...)
end

function plot_step(pomdp, step, hypothesis_fn, observations=Dict(); kwargs...)
    pstate = plot_mineralization(step.s, observations;)
    pbel = plot_belief(pomdp, step, hypothesis_fn, observations;)
    plot(pstate, pbel; margin=0Plots.mm, layout=grid(1, 2, widths=[0.33 ,0.67]), kwargs...)
end