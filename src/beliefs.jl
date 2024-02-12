"""
    Belief updater based on MCMC. 
    Note that the belief stores all the observations and doesn't actually use the prior belief to update
"""
mutable struct MCMCUpdater <: Updater
    alg # MCMC algorithm
    Nsamples # Number of MCMC iterations
    hypotheses::Vector{Hypothesis} # Set of hypotheses
    N # Number of particles to produce (1 per chain)
    observations::Dict # Observations
end

"""
    initialize_belief(up::MCMCUpdater, d)

    Samples a ParticleCollection from the set of hypotheses stored in the MCMCUpdater
"""
function POMDPs.initialize_belief(up::MCMCUpdater, d)
    @assert isempty(up.observations)
    N_particles_per_hypothesis = up.N ÷ length(up.hypotheses)
    particles = HierarchicalMinExState[]
    for h in up.hypotheses
        m = turing_model(h)(Dict(), h, true)
        for i in 1:N_particles_per_hypothesis
            res = m()
            push!(particles, HierarchicalMinExState(res.thickness, res.grade))
        end
    end
    return ParticleCollection(particles)
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

    # Run the chains
    chns = []
    loglikelihoods = []
    particle_lists = []
    for h in up.hypotheses
        # Create turing models to sample and use for MCMC
        m = turing_model(h)
        mcond = m(up.observations, h)
        mcond_w_samples = m(up.observations, h, true)

        # Run the chains and store the outcomes
        c = mapreduce(c -> Turing.sample(mcond, up.alg, up.Nsamples), chainscat, 1:up.N)
        push!(chns, c)
        push!(loglikelihoods, loglikelihood(c))
        os = generated_quantities(mcond_w_samples, c[end, :, :])
        ps = [HierarchicalMinExState(os[1,i].thickness, os[1,i].grade) for i=1:up.N]
        push!(particle_lists, ps)
    end

    return particle_lists[1] # TODO: resample based on likelihood
end
