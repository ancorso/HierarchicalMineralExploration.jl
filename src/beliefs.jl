"""
    Belief updater based on MCMC. 
    Note that the belief stores all the observations and doesn't actually use the prior belief to update
"""
mutable struct MCMCUpdater <: Updater
    Nsamples # Number of MCMC iterations
    hypotheses::Vector{Hypothesis} # Set of hypotheses
    hypothesis_loglikelihoods::Vector{Float64} # Likelihood of each hypothesis
    N # Number of particles to produce (1 per chain)
    observations::Dict # Observations
    chains::Vector # one chains per hypothesis
    algs::Vector # one alg per hypothesis
    models::Vector # on turing model per hypothesis
    function MCMCUpdater(Nsamples, hypotheses, N, observations=Dict())
        return new(
            Nsamples,
            hypotheses,
            zeros(length(hypotheses)), # initialize with uniform likelihoods
            N,
            observations,
            Any[nothing for i in 1:length(hypotheses)],
            [default_alg(h) for h in hypotheses],
            [turing_model(h) for h in hypotheses],
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
    N_particles_per_hypothesis = up.N ÷ length(up.hypotheses)
    particles = HierarchicalMinExState[]
    hypotheses = Int[]
    for (hi, h) in enumerate(up.hypotheses)
        m = turing_model(h)(Dict(), h, true)
        for i in 1:N_particles_per_hypothesis
            res = m()
            push!(particles, HierarchicalMinExState(res.thickness, res.grade))
            push!(hypotheses, hi)
        end
    end
    return MultiHypothesisBelief(ParticleCollection(particles), hypotheses)
end

function POMDPs.update(up::MCMCUpdater, b::MultiHypothesisBelief)
    # Run the chains
    loglikelihoods = [] # this holds all likelihoods for particles across hypotheses
    particles = [] # this holds all state particles across hypotheses
    hypotheses = [] # this holds the hypothesis index for each particle
    for (hi, (m, alg, h)) in enumerate(zip(up.models, up.algs, up.hypotheses))
        println("updating hypothesis: ", hi)

        # Create turing models to sample and use for MCMC
        mcond = m(up.observations, h)
        mcond_w_samples = m(up.observations, h, true) # this one produces state samples

        # For each particle, run (or continue the chain)
        if isnothing(up.chains[hi])
            up.chains[hi] = Turing.sample(
                mcond, alg, MCMCThreads(), up.Nsamples, up.N; save_state=true
            )
        else
            up.chains[hi] = Turing.sample(
                mcond, alg, MCMCThreads(), up.Nsamples, up.N; resume_from=up.chains[hi], save_state=true
            )
        end
        # Compute observation likelihoods and add to master list
        loglikelihoods_h = generated_quantities(mcond, up.chains[hi][end, :, :])[:]
        push!(loglikelihoods, logsumexp(loglikelihoods_h) .- log(length(loglikelihoods_h)))

        # Sample the actual state particles and add to master list
        os = generated_quantities(mcond_w_samples, up.chains[hi][end, :, :])[:]
        ps = [HierarchicalMinExState(o.thickness, o.grade) for o in os]
        push!(particles, ps)
        push!(hypotheses, fill(hi, length(ps)))

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
    relative_weights = exp.(loglikelihoods .- logsumexp(loglikelihoods))
    @assert (sum(relative_weights) - 1) < 1e-6
    Nh = floor.(Int, up.N .* relative_weights)
    Nh[argmax(Nh)] += up.N - sum(Nh) # Make sure we add up to N
    @assert sum(Nh) == up.N
    @assert all(Nh .>= 0)

    # Sample particles without replacement from each hypothesis
    all_particles = []
    all_hypotheses = []
    for i = 1:length(up.hypotheses)
        indices = StatsBase.sample(1:length(particles[i]), Nh[i], replace=false)
        push!(all_particles, particles[i][indices]...)
        push!(all_hypotheses, hypotheses[i][indices]...)
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
