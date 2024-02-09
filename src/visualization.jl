"""
    plot_model(;structural, geochemical, thickness, grade, observations)

    Plot the structural, geochemical, thickness, and grade domains in a 2x2 grid.
"""
function plot_model(;graben, geochem, thickness, grade, observations)
    # Collect the observations for easy plotting
    xpts = [p[1] for p in keys(observations)]
    ypts = [p[2] for p in keys(observations)]
    thickness_obs = [v.thickness for v in values(observations)]
    grade_obs = [v.grade for v in values(observations)]

    # Plot the structural domain
    p_graben = heatmap(graben'; cmap=:amp, colorbar=false)

    # Plot the thickness
    p_thickness = heatmap(thickness', colorbar=false)
    scatter!(
        xpts,
        ypts;
        zcolor=thickness_obs,
        markersize=5,
        legend=false,
        label="",
        markerstrokecolor=:white,
    )

    # Plot the geochemical domain
    p_geochem = heatmap(geochem'; cmap=:algae, colorbar=false)

    # Plot the grade
    p_grade = heatmap(grade', colorbar=false)
    scatter!(
        xpts,
        ypts;
        zcolor=grade_obs,
        markersize=5,
        legend=false,
        label="",
        markerstrokecolor=:white,
    )
    # Combine the plots in 2x2 grid
    plot(p_graben, p_thickness, p_geochem, p_grade; size=(800, 800), layout=(2, 2))
end