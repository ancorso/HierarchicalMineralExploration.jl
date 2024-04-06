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
using LogExpFunctions
## Setup mineral system models and the POMDP solver helpers and runners
include("setup.jl")
include("pomdp_tools.jl")


# Set up the output directory
output_dir = "outputs/all/"
pomdp = HierarchicalMinExPOMDP()

files = [f for f in readdir(output_dir) if occursin("results", f)]
Nfiles = length(files)
results = Array{Any}(undef, Nfiles)
for (i, file) in enumerate(files)
    println("loading $file...")
    results[i] = load(joinpath(output_dir,file))["results"]
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

## Plot some gifs of each one
for (i, result) in enumerate(results)
    for (alg, hist) in result
        plot_gif(pomdp, hist, "$(output_dir)/history_$alg$i.gif")
    end
end

# Accuracies
algs = keys(results[1])
for k in algs
    println(k, ": ", mean([iscorrect(result[k], pomdp) for result in results]))
end


## Plot the reward of each hypothesis
alg = "gridsearch"
plots = []
for i=1:length(results)
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

## Plot the likelihood of each hypothesis
alg = "3incorrect"
plots = []
for i=1:length(results)
    hist = results[i][alg]
    hist[1].hyp_logprobs
    hyp_logprobs_dict = Dict(k => [] for k in keys(hist[1].hyp_logprobs))
    for (t, h) in enumerate(hist)
        for (k, v) in h.hyp_logprobs
            # k==0 && continue
            if t > 1 && k > 0 
                chn = h.up.chains[k]
                Nsamples = size(chn, 1)

                #TODO For bridge density, we need samples from the priors and the data likelihood as well. 
                g1s = [chn[i, :lp, 1] for i=1:Nsamples]
                g2s = [domain_logpdf(hypotheses[k], chn[i, :, 1]) for i=1:Nsamples]
                gbs = (g1s .+ g2s)/2
                v = (logsumexp(gbs .- g2s) .- log(length(gbs)))  - (logsumexp(gbs .- g1s) .- log(length(gbs)))
            end
            push!(hyp_logprobs_dict[k], v)
        end
    end

    p = plot()
    for (k, v) in hyp_logprobs_dict
        plot!(v, label=k, linewidth=2, xlims=(0,36))
    end
    push!(plots, p)
end
plot(plots..., size=(2000,1500))

## Plot the belief of a particular step
hist = results[1]["gridsearch"]

# Plot the state
plot_state(hist[4].s)

N = 36
b = hist[N].b
observations = Dict()
for step in hist[1:N-1]
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