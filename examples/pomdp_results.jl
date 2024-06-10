using HierarchicalMineralExploration
using Turing
using Plots; default(fontfamily="Computer Modern", framestyle=:box, palette=:seaborn_dark, dpi=300)
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

function G(hist)
    return sum([h.r for h in hist])
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
    println(k, ": Decision accuracy: ", mean([iscorrect(result[k], pomdp) for result in results]), " ± ", std([iscorrect(result[k], pomdp) for result in results]) / sqrt(length(results)),  " Return: ", mean([G(result[k]) for result in results for step in result[k]]), " Bore Holes: ", mean([length(result[k])-1 for result in results]), " ± ",  std([length(result[k])-1 for result in results])/sqrt(length(results)))
end


function mixedlength_means_stds(vectors)
    N = maximum(length.(vectors))
    means = zeros(N)
    stds = zeros(N)
    counts = zeros(N)
    for i in 1:N
        means[i] = mean([v[i] for v in vectors if length(v) >= i])
        stds[i] = std([v[i] for v in vectors if length(v) >= i]) / sqrt(sum([length(v) >= i for v in vectors]))
        counts[i] = sum([length(v) >= i for v in vectors])
    end
    return means, stds, counts
end


## Plot the reward of each hypothesis
algs = keys(results[1])
# plots = []
errors_dict = Dict(k => [] for k in algs)
for alg in algs
    for i=1:length(results)
        hist = results[i][alg]
        r = [extraction_reward(pomdp, h.s) for h in hist]
        meanrs = [get_mean_minmax_r(step)[1] for step in hist]
        minmaxr = [get_mean_minmax_r(step)[2] for step in hist]

        mins = abs.([e[1] for e in minmaxr] .- meanrs)
        maxs = abs.([e[2] for e in minmaxr]  .- meanrs)

        # p = plot(meanrs, ribbon=(maxs, mins), title="Trial $i", label="")
        # plot!(r, label="")

        # push!(plots, p)
        push!(errors_dict[alg], abs.(meanrs .- r))
    end
end
# plot(plots...)

gridsearch_means, gridsearch_stds, gridsearch_counts = mixedlength_means_stds(errors_dict["gridsearch"])
plot(gridsearch_means, ribbon=gridsearch_stds, label="Grid Search", xlabel="Number of Bore Holes", ylabel="Ore Estimate Error")

pomdp_means, pomdp_stds, pomdp_counts = mixedlength_means_stds(errors_dict["4withcorrect"])
plot!(pomdp_means[pomdp_counts .> 3], ribbon=pomdp_stds[pomdp_counts .> 3], label="POMDP")

vline!([mean(pomdp_counts)], label="Mean POMDP Decision Point", color=:black, linestyle=:dash)

savefig("outputs/figures/ore_estimate_error.png")

# TODO: 3 incorect - ID 3 is an example, where a second geochemical domain lead to an incorrect decision. 


## Plot the likelihood of each hypothesis
algs = ["gridsearch", "3incorrect"]
falsified_dict = Dict(k => [] for k in algs)
for alg in algs
    # plots = []
    for i=1:length(results)
        push!(falsified_dict[alg], [])  
        hist = results[i][alg]
        hist[1].hyp_logprobs
        hyp_logprobs_dict = Dict(k => [] for k in keys(hist[1].hyp_logprobs))
        for (t, h) in enumerate(hist)
            null_v = 0
            vals = zeros(3)
            for (k, v) in h.hyp_logprobs
                if k==0
                    observations = Dict()
                    for step in hist[1:t-1]
                        a = step.a
                        o = step.o
                        observations[a] = (thickness=o[1], grade=o[2])
                    end
                    # println(observations)
                    v = logprob(max_ent_hypothesis2, observations)
                    null_v = v
                else
                    if k != 1 # skip the real one
                        vals[k-1] = v
                    end
                end
                # k==0 && continue
                # if t > 1 && k > 0 
                #     chn = h.up.chains[k]
                #     Nsamples = size(chn, 1)

                #     #TODO For bridge density, we need samples from the priors and the data likelihood as well. 
                #     g1s = [chn[i, :lp, 1] for i=1:Nsamples]
                #     g2s = [domain_logpdf(hypotheses[k], chn[i, :, 1]) for i=1:Nsamples]
                #     gbs = (g1s .+ g2s)/2
                #     v = (logsumexp(gbs .- g2s) .- log(length(gbs)))  - (logsumexp(gbs .- g1s) .- log(length(gbs)))
                # end
                push!(hyp_logprobs_dict[k], v)
            end
            num_falsified = sum(null_v .> vals)
            push!(falsified_dict[alg][end], num_falsified)
        end

        # p = plot()
        # for (k, v) in hyp_logprobs_dict
        #     plot!(v, label=k, linewidth=2,)
        # end
        # push!(plots, p)
    end
end
# plot(plots..., size=(2000,1500))

gridsearch_faslified_means, gridsearch_faslified_stds, gridsearch_faslified_counts = mixedlength_means_stds(falsified_dict["gridsearch"])
plot(gridsearch_faslified_means, ribbon=gridsearch_faslified_stds, label="Grid Search", xlabel="Number of Bore Holes", ylabel="Number of Falsified Hypotheses")

three_faslified_means, three_faslified_stds, three_faslified_counts = mixedlength_means_stds(falsified_dict["3incorrect"])
plot!(three_faslified_means[three_faslified_counts .> 3], ribbon=three_faslified_stds[three_faslified_counts .> 3], label="POMDP")

savefig("outputs/figures/falsified_hypotheses.png")
three_faslified_counts


# Plot likelihoods
alg = "4withcorrect"
plots = []
for i=1:length(results)
    hist = results[i][alg]
    hyp_logprobs_dict = Dict(k => [] for k in keys(hist[1].hyp_logprobs))
    for (t, h) in enumerate(hist)
        for (k, v) in h.hyp_logprobs
            if k==0
                observations = Dict()
                for step in hist[1:t-1]
                    a = step.a
                    o = step.o
                    observations[a] = (thickness=o[1], grade=o[2])
                end
                # println(observations)
                v = logprob(max_ent_hypothesis2, observations)
            end
            # k==0 && continue
            # if t > 1 && k > 0 
            #     chn = h.up.chains[k]
            #     Nsamples = size(chn, 1)

            #     #TODO For bridge density, we need samples from the priors and the data likelihood as well. 
            #     g1s = [chn[i, :lp, 1] for i=1:Nsamples]
            #     g2s = [domain_logpdf(hypotheses[k], chn[i, :, 1]) for i=1:Nsamples]
            #     gbs = (g1s .+ g2s)/2
            #     v = (logsumexp(gbs .- g2s) .- log(length(gbs)))  - (logsumexp(gbs .- g1s) .- log(length(gbs)))
            # end
            push!(hyp_logprobs_dict[k], v)
        end
    end

    p = plot()
    for (k, v) in hyp_logprobs_dict
        plot!(v, label=k, linewidth=2,)
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