"""
    turing_model(hypothesis)

    Return the appropriate Turing model function for the given `hypothesis`.
"""
function turing_model(hypothesis)
    if hypothesis isa Hypothesis
        if length(hypothesis.grabens) == 1 && length(hypothesis.geochem_domains) == 1
            return one_graben_one_geochem
        else
            error("this configuration is not supported ") #TODO
        end
    else
        error("Unknown hypothesis type")
    end
end

# struct MCMCUpdater <: Updater
#     chains::Chains
#     hypotheses::Vector{Hypothesis}
# end

"""
    particle_collection(N, hypotheses)

    Create a particle collection of size `N` from a set of `hypotheses`.
    The prior on the hypotheses is assumed to be uniform.
"""
function particle_collection(N, hypotheses)
    N_particles_per_hypothesis = N ÷ length(hypotheses)
    particles = HierarchicalMinExState[]
    for h in hypotheses
        m = turing_model(h)()
        for i=1:N_particles_per_hypothesis
            res = m()
            push!(particles, HierarchicalMinExState(res.thickness, res.grade))
        end
    end
    return ParticleCollection(particles)
end

"""
    particle_collection(N, hypotheses, observations)

    Create a particle collection of size `N` from a set of `hypotheses` and `observations`.
    The hypotheses are assumed to have a uniform prior and are updated using the obserations.
"""
function particle_collection(N, hypotheses, observations; Nsamples=1000)
    # Create posterior chains for each hypothesis
    chains = Chains[]
    for h in hypotheses
        mcond = turing_model(h)(observations, h)
        Nsamples = 1000
        specs = MH()
        num_chains = 10
        # chains = sample(mcond, specs, Nsamples), chainscat, 1:num_chains)
        push!(chains, chains)
    end

    # Compute likelihoods
    likelihoods = [loglikelihood(chains[i]) for i in 1:length(chains)]

    # Return particles with appropriate frequency

    return ParticleCollection(particles)
end
