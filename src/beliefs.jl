"""
    Belief updater based on MCMC. 
    Note that the belief stores all the observations and doesn't actually use the prior belief to update
"""
mutable struct MCMCUpdater <: Updater
    Nsamples # Number of MCMC iterations
    Ndiscard # Number of MCMC iterations to discard
    hypotheses::OrderedDict{Int, Hypothesis} # Set of hypotheses
    hypothesis_loglikelihoods::OrderedDict{Int, Float64} # Likelihood of each hypothesis
    # N # Number of particles to produce (1 per chain)
    observations::Dict # Observations
    chains::OrderedDict # one chain per hypothesis
    algs::OrderedDict # one alg per hypothesis
    models::OrderedDict # on turing model per hypothesis
    function MCMCUpdater(Nsamples, hypotheses, observations=Dict(); Ndiscard=100)
        return new(
            Nsamples,
            Ndiscard,
            hypotheses,
            OrderedDict{Int, Float64}(i => 0 for i in keys(hypotheses)), # initialize with uniform likelihoods
            # N,
            observations,
            OrderedDict{Int, Any}(i => nothing for i in keys(hypotheses)),
            OrderedDict(hi => default_alg(h) for (hi, h) in hypotheses),
            OrderedDict(hi => turing_model(h) for (hi, h) in hypotheses),
        )
    end
end

# Belief type that also stores which hypothesis a particle belongs to
struct MultiHypothesisBelief
    particles::ParticleCollection
    hypotheses
end

ParticleFilters.particles(b::MultiHypothesisBelief) = particles(b.particles)

"""
    initialize_belief(up::MCMCUpdater, d)

    Samples a ParticleCollection from the set of hypotheses stored in the MCMCUpdater
"""
function POMDPs.initialize_belief(up::MCMCUpdater, d)
    @assert isempty(up.observations)
    N_particles_per_hypothesis = up.Nsamples ÷ length(up.hypotheses)
    particles = HierarchicalMinExState[]
    hypotheses = Int[]
    for hi in keys(up.hypotheses)
        m = up.models[hi](up.observations, up.hypotheses[hi], true)
        for _ in 1:N_particles_per_hypothesis
            res = m()
            push!(particles, HierarchicalMinExState(res.thickness, res.grade))
            push!(hypotheses, hi)
        end
    end
    return MultiHypothesisBelief(ParticleCollection(particles), hypotheses)
end

function POMDPs.update(up::MCMCUpdater, b::MultiHypothesisBelief)
    # Run the chains
    loglikelihoods = OrderedDict() # this holds all likelihoods for particles across hypotheses
    particles = OrderedDict() # this holds all state particles across hypotheses
    hypotheses = OrderedDict() # this holds the hypothesis index for each particle
    for hi in keys(up.hypotheses)
        m, alg, h = up.models[hi], up.algs[hi], up.hypotheses[hi]
        println("updating hypothesis: ", hi)

        # Create turing models to sample and use for MCMC
        mcond = m(up.observations, h)
        mcond_w_samples = m(up.observations, h, true) # this one produces state samples

        # For each particle, run (or continue the chain)
        if isnothing(up.chains[hi])
            up.chains[hi] = Turing.sample(
                mcond, alg, up.Nsamples; save_state=true, discard_initial=100
            )
        else
            up.chains[hi] = Turing.sample(
                mcond,
                alg,
                up.Nsamples;
                resume_from=up.chains[hi],
                save_state=true,
                discard_initial=100,
            )
        end
        # Compute observation likelihoods and add to master list
        loglikelihoods_h = generated_quantities(mcond, up.chains[hi][:, :, 1])[:]
        up.hypothesis_loglikelihoods[hi] = logsumexp(loglikelihoods_h) .- log(length(loglikelihoods_h))

        # Sample the actual state particles and add to master list
        os = generated_quantities(mcond_w_samples, up.chains[hi][:, :, 1])[:]
        ps = [HierarchicalMinExState(o.thickness, o.grade) for o in os]
        particles[hi] = ps
        hypotheses[hi] = fill(hi, length(ps))

        # # Resample the chains to retain high likelihood particles
        # weights = exp.(loglikelihoods_h .- logsumexp(loglikelihoods_h))
        # indices = rand(Categorical(weights), up.N)
        # up.chains[hi] = up.chains[hi][:, :, indices]

        # # Keep sampling some more
        # up.chains[hi] = Turing.sample(
        #     mcond,
        #     alg,
        #     MCMCThreads(),
        #     up.Nsamples,
        #     up.N;
        #     resume_from=up.chains[hi],
        #     save_state=true,
        # )

        # # Compute observation likelihoods and add to master list
        # loglikelihoods_h = generated_quantities(mcond, up.chains[hi][end, :, :])[:]
        # push!(loglikelihoods, loglikelihoods_h...)

        # # Sample the actual state particles and add to master list
        # os = generated_quantities(mcond_w_samples, up.chains[hi][end, :, :])[:]
        # ps = [HierarchicalMinExState(o.thickness, o.grade) for o in os]
        # push!(all_particles, ps...)

        # # Resample the chains to retain high likelihood particles
        # weights = exp.(loglikelihoods_h .- logsumexp(loglikelihoods_h))
        # indices = rand(Categorical(weights), up.N)
        # up.chains[hi] = up.chains[hi][:, :, indices]
    end

    # figure out how many particles to sample for each hypothesis
    loglikelihoods = collect(values(up.hypothesis_loglikelihoods))
    relative_weights = exp.(values(loglikelihoods) .- logsumexp(values(loglikelihoods)))
    @assert (sum(relative_weights) - 1) < 1e-6
    Nh = floor.(Int, up.Nsamples .* relative_weights)
    Nh[argmax(Nh)] += up.Nsamples - sum(Nh) # Make sure we add up to N
    Nh_dict = OrderedDict()
    for (i, hi) in enumerate(keys(up.hypotheses))
        Nh_dict[hi] = Nh[i]
    end
    @assert sum(Nh) == up.Nsamples
    @assert all(Nh .>= 0)

    # Sample particles without replacement from each hypothesis
    all_particles = []
    all_hypotheses = []
    for hi in keys(up.hypotheses)
        indices = StatsBase.sample(1:length(particles[hi]), Nh_dict[hi]; replace=false)
        push!(all_particles, particles[hi][indices]...)
        push!(all_hypotheses, hypotheses[hi][indices]...)
    end

    return MultiHypothesisBelief(ParticleCollection(all_particles), all_hypotheses)
end

"""
Produce a particle collection using N chains with a MCMC algorithm
"""
function POMDPs.update(up::MCMCUpdater, b, a, o)
    # If the action is terminal just return the last belief
    if a isa Symbol
        return b
    end

    # Update the observation list
    up.observations[[a...]] = o

    return update(up, b)
end
