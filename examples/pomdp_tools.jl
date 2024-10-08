## define the discretization and solver functions
function discretize_fn(pomdp, b::MultiHypothesisBelief; Nobs=5)
    sts = particles(b)
    state_abstraction = NearestNeighborAbstraction(
        sts; convert=(s) -> vcat(s.thickness[:], s.grade[:])
    )

    # Define discrete observations
    os = [
        gen(pomdp, rand(sts), rand(pomdp.drill_locations), Random.GLOBAL_RNG).o for
        i in 1:10000
    ]
    discrete_obs = [nothing, KMeansDiscretizer(Nobs)(os)...]
    observation_abstraction = NearestNeighborAbstraction(
        discrete_obs; convert=(o) -> isnothing(o) ? [-1000.0, -1000] : o
    )

    # Generate samples for filling in the observations
    hist = sample_transitions(pomdp, sts, 1; rng=Random.GLOBAL_RNG)
    return DiscretizedPOMDP(pomdp, hist, state_abstraction, observation_abstraction)
end

function solver_fn(discrete_pomdp; solver_time=25.0)
    return POMDPs.solve(SARSOPSolver(; max_time=solver_time), discrete_pomdp)
end

## run the pomdp trial
function run_trial(
    pomdp,
    s,
    hypotheses,
    max_ent_hyp;
    Nsamples=100,
    max_steps=37,
    verbose=true,
)
    verbose && println("reward: ", extraction_reward(pomdp, s))

    hist = []
    steps = 0

    # Initialize belief and updater
    up = MCMCUpdater(Nsamples, hypotheses)
    b0 = initialize_belief(up, nothing)
    b = b0

    while !isterminal(pomdp, s)
        # Manage the steps
        steps += 1
        println("step: ", steps)
        steps > max_steps && break

        # compute hypothesis likelihoods
        hyp_logprobs = deepcopy(up.hypothesis_loglikelihoods)
        hyp_logprobs[0] = logprob(max_ent_hyp, up.observations)

        # Solve the pomdp 
        discrete_pomdp = discretize_fn(pomdp, b)
        solver = solver_fn(discrete_pomdp)
        discrete_belief = initialstate(discrete_pomdp)
        discrete_updater = DiscreteUpdater(discrete_pomdp)

        # Take the action step
        a_int = action(solver, discrete_belief)
        a = actions(pomdp)[a_int]
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        o_int = discrete_pomdp.observation_abstraction(o)
        verbose && println("a: ", a, " o: ", o, " r: ", r)

        # Store the history
        push!(hist, (; s, a, o, r, b, hyp_logprobs, up=deepcopy(up)))

        # Check if terminal and break before updating the belief
        isterminal(pomdp, sp) && break

        # Belief updates
        discrete_belief = update(discrete_updater, discrete_belief, a_int, o_int)
        b = update(up, b, a, (thickness=o[1], grade=o[2]))

        s = sp
    end
    return hist
end

# Run a pomdp trial with hypothesis rejuvination
function run_trial_rejuvination(
    pomdp,
    s,
    hypotheses,
    max_ent_hyp;
    Nsamples=100,
    max_steps=37,
    verbose=true,
)
    verbose && println("reward: ", extraction_reward(pomdp, s))

    hist = []
    steps = 0

    # Initialize belief and updater
    updaters = [
        MCMCUpdater(Nsamples, OrderedDict(j => hypotheses[j] for j in 1:i)) for
        i in 1:length(hypotheses)
    ]
    beliefs = [initialize_belief(up, nothing) for up in updaters]
    bi = 1

    while !isterminal(pomdp, s)
        b = beliefs[bi]
        up = updaters[bi]

        # Manage the steps
        steps += 1
        println("step: ", steps)
        steps > max_steps && break

        # compute hypothesis likelihoods
        hyp_logprobs = deepcopy(up.hypothesis_loglikelihoods)
        hyp_logprobs[0] = logprob(max_ent_hyp, up.observations)

        # add hypotheses until we match the data
        if steps > 1
            while all(hyp_logprobs[0] .> values(up.hypothesis_loglikelihoods))
                println("ADDING HYPOTHESES")
                bi += 1
                b = beliefs[bi]
                up = updaters[bi]
                hyp_logprobs = deepcopy(up.hypothesis_loglikelihoods)
                hyp_logprobs[0] = logprob(max_ent_hyp, up.observations)
            end
        end

        # Solve the pomdp 
        discrete_pomdp = discretize_fn(pomdp, b)
        solver = solver_fn(discrete_pomdp)
        discrete_belief = initialstate(discrete_pomdp)
        discrete_updater = DiscreteUpdater(discrete_pomdp)

        # Take the action step
        a_int = action(solver, discrete_belief)
        a = actions(pomdp)[a_int]
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        o_int = discrete_pomdp.observation_abstraction(o)
        verbose && println("a: ", a, " o: ", o, " r: ", r)

        # Store the history
        push!(hist, (; s, a, o, r, b, hyp_logprobs, up=deepcopy(up)))

        # Check if terminal and break before updating the belief
        isterminal(pomdp, sp) && break

        # Belief updates
        discrete_belief = update(discrete_updater, discrete_belief, a_int, o_int)
        for i in bi:length(updaters)
            beliefs[i] = update(updaters[i], beliefs[i], a, (thickness=o[1], grade=o[2]))
        end
        s = sp
    end
    return hist
end

# run grid search trial
function run_gridsearch(
    pomdp,
    s,
    hypotheses,
    max_ent_hyp;
    Nsamples=100,
    max_steps=37,
    verbose=true,
)
    verbose && println("reward: ", extraction_reward(pomdp, s))

    hist = []
    steps = 0

    # Initialize belief and updater
    up = MCMCUpdater(Nsamples, hypotheses)
    b0 = initialize_belief(up, nothing)
    b = b0
    println("length of belief: ", length(particles(b)))

    # Loop over actions
    for a in [(i, j) for i in 5:5:30 for j in 5:5:30]
        # Manage the steps
        steps += 1
        println("step: ", steps)
        steps > max_steps && break

        # compute hypothesis likelihoods
        hyp_logprobs = deepcopy(up.hypothesis_loglikelihoods)
        hyp_logprobs[0] = logprob(max_ent_hyp, up.observations)

        # Take the action step
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        verbose && println("a: ", a, " o: ", o, " r: ", r)

        # Store the history
        push!(hist, (; s, a, o, r, b, hyp_logprobs, up=deepcopy(up)))

        # Check if terminal and break before updating the belief
        isterminal(pomdp, sp) && error("shouldn't be terminal")

        # Belief updates
        b = update(up, b, a, (thickness=o[1], grade=o[2]))
        s = sp
    end

    rs = [extraction_reward(pomdp, s) for s in particles(b)]
    if mean(rs) > 0
        a = :mine
    else
        a = :abandon
    end

    sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
    verbose && println("a: ", a, " o: ", o, " r: ", r)

    # compute hypothesis likelihoods
    hyp_logprobs = deepcopy(up.hypothesis_loglikelihoods)
    hyp_logprobs[0] = logprob(max_ent_hyp, up.observations)

    # Store the history
    push!(hist, (; s, a, o, r, b, hyp_logprobs, up=deepcopy(up)))
    return hist
end

function plot_gif(pomdp, history, filename)
    anim = @animate for i in eachindex(history)
        observations = Dict(
            step.a => (thickness=step.o[1], grade=step.o[2]) for step in history[1:(i - 1)]
        )
        plot_step(pomdp, history[i], observations; size=(1200, 600), bottom_margin=5Plots.mm)
    end
    return gif(anim, filename; fps=2)
end
