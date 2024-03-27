using HierarchicalMineralExploration
using Turing
using Plots
using StatsPlots
using AbstractGPs
using POMDPs
using ParticleFilters
using POMDPDiscretization
using Random
using NativeSARSOP
using POMDPTools
using JLD2
using DataStructures
## Setup mineral system models and the POMDP solver helpers and runners
include("setup.jl")
include("pomdp_tools.jl")


# Set up the output directory
output_dir = "outputs/all/"
pomdp = HierarchicalMinExPOMDP()


results = Array{Any}(undef, 10)
for i=1:10
    try
        result = load("outputs/all/results$(i).jld2")["results"]
        # hist = load("outputs/all/history_gridsearch$(i).jld2")["history"]
        # hist = load("outputs/all/history_1correct$(i).jld2")["history"]
        # hist = load("outputs/all/history_3incorrect$(i).jld2")["history"]
        # hist = load("outputs/all/history_4withcorrect$(i).jld2")["history"]
        # hist = load("outputs/all/history_rejuvination$(i).jld2")["history"]
        results[i] = result
    catch
    end
end
# [isdefined(histories, i) for i in eachindex(histories)]
# results = results[[isassigned(results, i) for i in eachindex(results)]];

function iscorrect(hist, pomdp)
    if extraction_reward(pomdp, hist[1].s) < 0
        return hist[end].a == :abandon
    else
        return hist[end].a == :mine
    end
end
function get_mean_minmax_r(step)
    rs = [extraction_reward(pomdp, s) for s in particles(step.b.particles)]
    meanr = mean(rs)
    minmaxr = extrema(rs)
    meanr, minmaxr
end

algs = keys(results[1])
for k in algs
    println(k, ": ", mean([iscorrect(result[k], pomdp) for result in results]))
end

alg = "1correct"
plots = []
for i=1:length(results)
    !isassigned(results, i) && continue
    hist = results[i][alg]
    r = [extraction_reward(pomdp, h.s) for h in hist]
    meanrs = [get_mean_minmax_r(step)[1] for step in hist]
    minmaxr = [get_mean_minmax_r(step)[2] for step in hist]

    mins = abs.([e[1] for e in minmaxr] .- meanrs)
    maxs = abs.([e[2] for e in minmaxr]  .- meanrs)

    p = plot(meanrs, ribbon=(maxs, mins), title="Trial $i", label="")
    plot!(r, label="")

    push!(plots, p)
end
plot(plots...)

r = [extraction_reward(pomdp, h.s) for h in hist]
meanrs = [get_mean_std_r(step)[1] for step in hist]
stdsrs = [get_mean_std_r(step)[2] for step in hist]

plot(meanrs, ribbon=stdsrs)
plot!(r)


rs = [extraction_reward(pomdp, s) for s in particles(hist[end].b.particles)]

histogram!(rs)

## Plot the belief of a particular step
hist = results[4]["rejuvination"]
b = hist[end].b
observations = Dict()
for step in hist[1:end-1]
    a = step.a
    o = step.o
    observations[a] = (thickness=o[1], grade=o[2])
end
anim = @animate for (i, p) in enumerate(particles(b.particles))
    plot_state(p, observations)
    plot!(; title="i=$i")
end
gif(anim, "particles.gif"; fps=2)

plot_state(hist[1].s, observations)