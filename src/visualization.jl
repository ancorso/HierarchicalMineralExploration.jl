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
            markersize=5,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
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
            markersize=5,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
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
    p_thickness = heatmap(s.thickness'; colorbar=false, clims=(0, 12))
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=thickness_obs,
            markersize=5,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
    end

    # Plot the grade
    p_grade = heatmap(s.grade'; colorbar=false, clims=(0, 17))
    if length(observations) > 0
        scatter!(
            xpts,
            ypts;
            zcolor=grade_obs,
            markersize=5,
            legend=false,
            label="",
            markerstrokecolor=:white,
        )
    end
    # Combine the plots in 2x2 grid
    return plot(p_thickness, p_grade; size=(800, 400), layout=(1, 2))
end