using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using DataStructures
using Plots; default(fontfamily="Computer Modern", framestyle=:box, palette=:seaborn_dark, dpi=300)

## Setup mineral system models
include("setup.jl")

## Create some plots of the various hypotheses
sample1 = turing_model(hypotheses[1])(Dict(), hypotheses[1], true)()
plot_model(; sample1...)
savefig("figures/hypothesis1.png")

heatmap(sample1.structural'; cmap=:amp, colorbar=false)
savefig("outputs/figures/structural.png")
heatmap(sample1.thickness';  colorbar=false)
savefig("outputs/figures/thickness.png")


sample1
mineralization = sample1.thickness .* sample1.grade
pthickness = heatmap(sample1.thickness'; colorbar=false,title="Thickness")
pgrade = heatmap(sample1.grade'; colorbar=false,title="Grade")
pmin = heatmap(mineralization'; colorbar=false, cmap=:haline, title="Mineralization")

plot(pthickness, pgrade, pmin; layout=(1,3), axis=false, size=(900, 300), dpi=300)
savefig("outputs/figures/hypothesis1_components.png")
# imgl = imfilter(s.thickness', ImageFiltering.Kernel.Laplacian());

sample2 = turing_model(hypotheses[2])(Dict(), hypotheses[2], true)()
plot_model(; sample2...)
savefig("figures/hypothesis2.png")

sample3 = turing_model(hypotheses[3])(Dict(), hypotheses[3], true)()
plot_model(; sample3...)
savefig("figures/hypothesis3.png")

sample4 = turing_model(hypotheses[4])(Dict(), hypotheses[4], true)()
plot_model(; sample4...)
savefig("figures/hypothesis4.png")

p1 = plot_state(sample1)
p2 = plot_state(sample2)
p3 = plot_state(sample3)
p4 = plot_state(sample4)

plot(p1, p2, p3, p4, axis=false)
savefig("outputs/figures/hypothesies.png")

## Choose a ground truth hypothesis
h_gt = hypotheses[1]
m_type_gt = turing_model(h_gt)
m = m_type_gt(Dict(), h_gt, true)
sample0 = m()

# Obtain some observations
pts = [[rand(1:N), rand(1:N)] for _ in 1:40]
observations = Dict(
    p => (thickness=sample0.thickness[p...], grade=sample0.grade[p...]) for p in pts
)

# Visualize the model and observations
plot_model(; sample0..., observations=observations)
savefig("outputs/figures/observations.png")

## Show what conditioning a hypothesis to a given set of data looks like
h = hypotheses[1]
m_type = turing_model(h)

# Construct the conditional distribution
mcond = m_type(observations, h)
mcond_w_samples = m_type(observations, h, true)

# Sample from the posterior
Nsamples = 1000
Nchains = 1
alg = HierarchicalMineralExploration.default_alg(h)

# Run the chains and store the outcomes
chns = mapreduce(c -> Turing.sample(mcond, alg, Nsamples), chainscat, 1:Nchains)

# Compute the likelihood
outputs = generated_quantities(mcond, chns[end, :, :])[1]
plot(chns[:, :lp, 1]) # Show the likelihood progress over time

outputs = generated_quantities(mcond_w_samples, chns[end, :, :])
reshape(outputs, 1)
plots = []
for i in 1:size(outputs,2)
    push!(
        plots,
        plot_model(;
            structural=outputs[end, i].structural,
            thickness=outputs[end, i].thickness,
            grade=outputs[end, i].grade,
            geochemdomain=outputs[end, i].geochemdomain,
            observations,
        ),
    )
end
plot(plots...; size=(800, 800))
savefig("figures/conditioned_hypothesis.png")