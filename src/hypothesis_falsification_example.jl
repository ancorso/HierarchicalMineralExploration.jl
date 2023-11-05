using AbstractGPs
using LazySets
using Distributions
using Parameters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using LogExpFunctions
using Printf


## Setup the problem-specific modeling Parameters
KERNEL = Matern52Kernel() ∘ ScaleTransform(1.0 / 10.0)
DOMAIN_OBS_NOISE = 0.1

# Utility function for sampling from GP with a pre-determined mean
function sample_gp(domain)
    x = hcat([[i, j] for i in 1:size(domain)[1], j in 1:size(domain)[2]][:]...)
    f = GP((x) -> domain[x...], KERNEL)
    fx = f(x)
    return reshape(rand(fx), size(domain)...)
end

## Define the geochemical domain generator
@with_kw struct GeochemicalDomainGenerator
    Ng = (100, 100)
    grades = [5.0, 7.0]
    grade_out = 3.0
    locations = [[25,25], [55,55]]
    radii = [15., 15.]
    Npts = 10
end

# Some helpers for defining some default geochem domains
TwoGeochemDomain() = GeochemicalDomainGenerator(grades=[5.0, 7.0], locations=[[25,25], [55,55]], radii=[15., 15.], Npts=10)
OneGeochemDomain() = GeochemicalDomainGenerator(grades=[7.0], locations=[[35,35], ], radii=[20., 20.], Npts=10)

function Base.rand(g::GeochemicalDomainGenerator)
    regions = []
    for i in 1:length(g.grades)
        pts = rand(Normal(0, g.radii[i]), 2, g.Npts) .+ g.locations[i]
        r = VPolygon([pts[:,i] for i in 1:g.Npts])
        push!(regions, r)
    end
    # Fill the domain with the 
    domain = g.grade_out*ones(g.Ng...)
    for i in 1:g.Ng[1], j in 1:g.Ng[2]
        for (r, grade) in zip(regions, g.grades)
            if [i,j] in r
                domain[i, j] = grade
            end
        end
    end
    return domain
end

# Struct for defining the structural domain
@with_kw struct GrabenDomainGenerator
    Ng = (100,100)
    Ngrabens = 1
    μ_in = 7.0
    μ_out = 3.0
    left_top_dist = Uniform(0.15*Ng[2], Ng[2])
    left_width_dist = Uniform(0.05*Ng[2], 0.25*Ng[2])
    right_top_dist = Uniform(0.15*Ng[2], Ng[2])
    right_width_dist = Uniform(0.05*Ng[2], 0.25*Ng[2])
end

function Base.rand(g::GrabenDomainGenerator)
    domains = []
    if g.Ngrabens == 0
        return g.μ_out*ones(g.Ng...)
    end
    for i=1:g.Ngrabens
        offset = g.Ng[1]*(i-1) / g.Ngrabens
        ltop, lwidth = offset + (1.0/g.Ngrabens)*rand(g.left_top_dist), rand(g.left_width_dist)
        rtop, rwidth = offset + (1.0/g.Ngrabens)*rand(g.right_top_dist), rand(g.right_width_dist)

        # Compute the polygon representing the Graben
        v = VPolygon([[1.0, ltop], [1.0, ltop - lwidth], [g.Ng[1], rtop - rwidth], [g.Ng[2], rtop]])

        # Fill the domain with the 
        domain = g.μ_out*ones(g.Ng...)
        for i in 1:g.Ng[1], j in 1:g.Ng[2]
            if [i,j] in v
                domain[i, j] = g.μ_in
            end
        end
        push!(domains, domain)
    end
    return reduce((x, y) -> max.(x, y), domains)
end

# Combined domain generator for the whole geology
struct GeologicalDomainGenerator
    structural_domain
    geochemical_domain
end

function Base.rand(g::GeologicalDomainGenerator)
    return cat(rand(g.structural_domain), rand(g.geochemical_domain), dims=3)
end

function sample_geology(domains)
    s1 = sample_gp(domains[:,:,1])
    s2 = sample_gp(domains[:,:,2])
    return cat(s1, s2, dims=3)
end

## Utilities for computing likelihoods of data
function compute_logpdf(domain_ensemble, xobs, thickness_obs, grade_obs, structural_obs, geochem_obs)
    logpdfs_structural = []
    logpdfs_geochem = []
    for domain in domain_ensemble
        fx_thickness = GP((x) -> domain[:,:,1][x...], KERNEL)(xobs)
        fx_grade = GP((x) -> domain[:,:,2][x...], KERNEL)(xobs)
        logpdf_structural = sum([logpdf(Normal(domain[:,:,1][xobs[:,i]...], DOMAIN_OBS_NOISE), d) for (i, d) in enumerate(structural_obs)])
        logpdf_geochem_obs = sum([logpdf(Normal(domain[:,:,2][xobs[:,i]...], DOMAIN_OBS_NOISE), d) for (i, d) in enumerate(geochem_obs)])
        push!(logpdfs_structural, logpdf(fx_thickness, thickness_obs) + logpdf_structural)
        push!(logpdfs_geochem, logpdf(fx_grade, grade_obs) + logpdf_geochem_obs)
    end

    return logsumexp(logpdfs_structural) + log(1.0/length(logpdfs_structural)) + logsumexp(logpdfs_geochem) + log(1.0/length(logpdfs_geochem))
end

function compute_null_pdf(thickness_dist, thickness_obs, grade_dist, grade_obs, structural_dist, structural_obs, geochem_dist, geochem_obs)
    sumlogpdf = 0
    sumlogpdf += sum([logpdf(thickness_dist, d) for d in thickness_obs])
    sumlogpdf += sum([logpdf(grade_dist, d) for d in grade_obs])
    sumlogpdf += sum([logpdf(structural_dist, d) for d in structural_obs])
    sumlogpdf += sum([logpdf(geochem_dist, d) for d in geochem_obs])
    return sumlogpdf
end

## Plotting utilities
function plot_domain(domain, xobs=[], structural_obs=[], geochem_obs=[])
    ps = heatmap(domain[:,:,1]', title="Structural", c=:amp)
    if length(xobs) > 0
        scatter!(xobs[1,:], xobs[2,:], zcolor=structural_obs, label="", c=:amp, markerstrokecolor=:blue)
    end
    pg = heatmap(domain[:,:,2]', title="Geochemical", c=:algae)
    if length(xobs) > 0
        scatter!(xobs[1,:], xobs[2,:], zcolor=geochem_obs, label="", c=:algae, markerstrokecolor=:blue)
    end
    plot(ps, pg, colorbar=false, size=(800, 400))
end

function plot_mineralization(geology)
    heatmap(geology[:,:,1]' .* geology[:,:,2]' ./ 40., title="Mineralization", colorbar=false, size=(400,400))
end


####################################################################################
#             Problem setup and examples of the above utilities                    #
####################################################################################

# Define the hypotheses via a cross product
graben_hypotheses = [GrabenDomainGenerator(Ngrabens=i) for i in 1:2]
geochem_hypotheses = [OneGeochemDomain(), TwoGeochemDomain()]
hypotheses = [GeologicalDomainGenerator(g, c) for g in graben_hypotheses, c in geochem_hypotheses][:]
ensembles = [[rand(h) for _=1:10000] for h in hypotheses]

# Define the null hypothesis
thickness_null = Normal(5.0, 3.0)
grade_null = Normal(5.0, 3.0)
structural_null = Normal(5.0, 3.0)
geochem_null = Normal(5.0, 3.0)

# Plot a null hypothesis example
thickness_sample = rand(thickness_null, 100, 100)
grade_sample = rand(grade_null, 100, 100)
structural_sample = rand(structural_null, 100, 100)
geochem_sample = rand(geochem_null, 100, 100)
plot_domain(cat(structural_sample, geochem_sample, dims=3))
savefig("figures/null_domains.png")
plot_mineralization(cat(thickness_sample, grade_sample, dims=3))
savefig("figures/null_min.png")

# Plot the H1 example
d = rand(hypotheses[1])
s = sample_geology(d)
plot_domain(d)
savefig("figures/h1_domains.png")
plot_mineralization(s)
savefig("figures/h1_min.png")

# Construct a ground truth
domain_gt = rand(hypotheses[1])
geology_gt = sample_geology(domain_gt)

# Sample some data
x = hcat([[i, j] for i in 1:size(domain_gt)[1], j in 1:size(domain_gt)[2]][:]...)
r_indices = rand(1:size(x, 2), 30)
xobs = x[:, r_indices]

# Extract the observations
thickness_obs = domain_gt[:,:,1][:][r_indices]
grade_obs = geology_gt[:,:,2][:][r_indices]
structural_obs = domain_gt[:,:,1][:][r_indices] .+ rand(Normal(0.0, DOMAIN_OBS_NOISE), length(r_indices))
geochem_obs = domain_gt[:,:,2][:][r_indices] .+ rand(Normal(0.0, DOMAIN_OBS_NOISE), length(r_indices))

# Plot the ground truth and the data
plot_domain(domain_gt, xobs, structural_obs, geochem_obs)

# Compare the log probability of the data under each hypothesis
compute_logpdf(ensembles[1], xobs, thickness_obs, grade_obs, structural_obs, geochem_obs)
compute_logpdf(ensembles[2], xobs, thickness_obs, grade_obs, structural_obs, geochem_obs)
compute_logpdf(ensembles[3], xobs, thickness_obs, grade_obs, structural_obs, geochem_obs)
compute_logpdf(ensembles[4], xobs, thickness_obs, grade_obs, structural_obs, geochem_obs)
compute_null_pdf(thickness_null, thickness_obs, grade_null, grade_obs, structural_null, structural_obs, geochem_null, geochem_obs)


logpdfs_1 = [compute_logpdf(ensembles[1], xobs[:,1:N], thickness_obs[1:N], grade_obs[1:N], structural_obs[1:N], geochem_obs[1:N]) for N=1:length(grade_obs)]
logpdfs_2 = [compute_logpdf(ensembles[2], xobs[:,1:N], thickness_obs[1:N], grade_obs[1:N], structural_obs[1:N], geochem_obs[1:N]) for N=1:length(grade_obs)]
logpdfs_3 = [compute_logpdf(ensembles[3], xobs[:,1:N], thickness_obs[1:N], grade_obs[1:N], structural_obs[1:N], geochem_obs[1:N]) for N=1:length(grade_obs)]
logpdfs_4 = [compute_logpdf(ensembles[4], xobs[:,1:N], thickness_obs[1:N], grade_obs[1:N], structural_obs[1:N], geochem_obs[1:N]) for N=1:length(grade_obs)]
null_pdfs = [compute_null_pdf(thickness_null, thickness_obs[1:N], grade_null, grade_obs[1:N], structural_null, structural_obs[1:N], geochem_null, geochem_obs[1:N]) for N=1:length(grade_obs)]


y_formatted(y) = @sprintf("%.2e\n", y)
plots = []
anim = @animate for i=1:length(logpdfs_1)
    p1 = plot(1:i, logpdfs_2[1:i], label="H1", xlims=(0, length(logpdfs_1)), ylabel="Logpdf", legend=:bottomleft, yformatter = y_formatted, title="Logpdfs of Hypotheses", margin=5Plots.mm)
    plot!(1:i, logpdfs_3[1:i], label="H2")
    plot!(1:i, logpdfs_4[1:i], label="H3")
    plot!(1:i, null_pdfs[1:i], label="Null")
    plot!(1:i, logpdfs_1[1:i], label="Correct")

    p2 = plot_mineralization(geology_gt)
    scatter!(xobs[1,1:i], xobs[2,1:i], label="", c=:amp, markerstrokecolor=:blue, xlims=(0,100), ylims=(0,100), title="Observation Locations")
    p3 = plot(p1, p2, size=(800, 400), dpi=300)
    push!(plots, p3)
    p3
end
gif(anim, "figures/likelihood_w_all.gif", fps=1)
savefig(plots[end], "figures/likelihood_w_all.png")

# plot(1:2:length(grade_obs), logpdfs_1, label="H1")
plot(1:2:length(grade_obs), logpdfs_2, label="H2")
plot!(1:2:length(grade_obs), logpdfs_3, label="H3")
plot!(1:2:length(grade_obs), logpdfs_4, label="H4")
plot!(1:2:length(grade_obs), null_pdfs, label="Null")


## Make some simple figures regarding priors
heatmap(ensembles[1][1][:,:,1], padding = (0.0, 0.0), c=:amp, axis=false, margin=-8Plots.mm, size=(400,400), colorbar=false)
savefig("figures/example_hypothesis_1.png")

samples = [sample_geology(ensembles[1][i]) for i in 1:3]
for (i, s) in enumerate(samples)
    heatmap(s[:,:,1], padding = (0.0, 0.0), c=:amp, axis=false, margin=-8Plots.mm, size=(400,400), colorbar=false)
    savefig("figures/example_hypothesis_1_sample_$(i).png")
end

heatmap(ensembles[2][1][:,:,1], padding = (0.0, 0.0), c=:amp, axis=false, margin=-8Plots.mm, size=(400,400), colorbar=false)
savefig("figures/example_hypothesis_2.png")

samples = [sample_geology(ensembles[2][i]) for i in 1:3]
for (i, s) in enumerate(samples)
    heatmap(s[:,:,1], padding = (0.0, 0.0), c=:amp, axis=false, margin=-8Plots.mm, size=(400,400), colorbar=false)
    savefig("figures/example_hypothesis_2_sample_$(i).png")
end

# Plot the obs
scatter(xobs[1,:], xobs[2,:], zcolor=structural_obs, label="", c=:amp, xlims=(0,100), ylims=(0,100), title="Observation Locations", colorbar=false, size=(400,400))
savefig("figures/example_obs.png")

