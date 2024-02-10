"""
State of a hierarchical mineral exploration problem
"""
mutable struct HierarchicalMinExState
    thickness::Matrix{Float32}
    grade::Matrix{Float32}
    drill_locations::Vector{Tuple{Int,Int}}
    function HierarchicalMinExState(thickness, grade, drill_locations=[])
        return new(thickness, grade, drill_locations)
    end
end

# Ensure HierarchicalMinExState can be compared when adding to a dictionary
function Base.hash(s::HierarchicalMinExState, h::UInt)
    return hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
end
function Base.isequal(s1::HierarchicalMinExState, s2::HierarchicalMinExState)
    return all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
end
Base.:(==)(s1::HierarchicalMinExState, s2::HierarchicalMinExState) = isequal(s1, s2)

const TERMINAL_LOCATION = (-1, -1)

## Definition of the POMDP
@with_kw struct HierarchicalMinExPOMDP <: POMDP{HierarchicalMinExState,Any,Any}
    grid_dims = (32, 32)
    grade_threshold = 10.0
    extraction_cost = 400 # this is an eyball from 100 samples
    drill_cost = 0.1
    drill_locations = [(i, j) for i in 5:5:30 for j in 5:5:30]
    terminal_actions = [:abandon, :mine]
    σ_thickness = 0.01
    σ_grade = 0.01
    γ = 0.999
end

POMDPs.discount(m::HierarchicalMinExPOMDP) = m.γ
function POMDPs.isterminal(m::HierarchicalMinExPOMDP, s)
    return any(loc -> loc == TERMINAL_LOCATION, s.drill_locations)
end
undrilled_locations(m::HierarchicalMinExPOMDP, b) = undrilled_locations(m, rand(b))
function undrilled_locations(m::HierarchicalMinExPOMDP, s::HierarchicalMinExState)
    return setdiff(m.drill_locations, s.drill_locations)
end
function POMDPs.actions(m::HierarchicalMinExPOMDP, s_or_b)
    return [m.terminal_actions..., undrilled_locations(m, s_or_b)...]
end
POMDPs.actions(m::HierarchicalMinExPOMDP) = [m.terminal_actions..., m.drill_locations...]

function calc_massive(m::HierarchicalMinExPOMDP, s::HierarchicalMinExState)
    return sum(s.thickness .* (s.grade .> m.grade_threshold))
end
function extraction_reward(m::HierarchicalMinExPOMDP, s::HierarchicalMinExState)
    return calc_massive(m, s) - m.extraction_cost
end
function calibrate_extraction_cost(m::HierarchicalMinExPOMDP, ds0)
    return mean(calc_massive(m, s) for s in ds0)
end
# ^ Note: Change extraction cost so state distribution is centered around 0.

# This gen function is for passing multiple drilling actions
function POMDPs.gen(
    m::HierarchicalMinExPOMDP, s, as::Vector{Tuple{Int,Int}}, rng=Random.GLOBAL_RNG
)
    sp = deepcopy(s)
    rtot = 0
    os = Float64[]
    for a in as
        push!(sp.drill_locations, a)
        _, o, r = gen(m, s, a, rng)
        push!(os, o)
        rtot += r
    end
    return (; sp, o=os, r=rtot)
end

function POMDPs.reward(m::HierarchicalMinExPOMDP, s, a)
    if a == :abandon || isterminal(m, s)
        r = 0
    elseif a == :mine
        r = extraction_reward(m, s)
    else
        r = -m.drill_cost
    end
    return r
end

function POMDPs.observation(m::HierarchicalMinExPOMDP, a, s)
    if isterminal(m, s) || a in m.terminal_actions
        o = (thickness=-Inf32, grade=-Inf32)
    else
        o = (thickness=s.thickness[a...], grade=s.grade[a...])
    end
    return o
end

function POMDPs.gen(m::HierarchicalMinExPOMDP, s, a, rng=Random.GLOBAL_RNG)
    # Compute the next state
    sp = deepcopy(s)

    # Compute the reward
    r = reward(m, s, a)
    if r == -m.drill_cost
        push!(sp.drill_locations, a)
    end

    # observation
    o = observation(m, a, s)

    if (a in m.terminal_actions || isterminal(m, s))
        push!(sp.drill_locations, TERMINAL_LOCATION)
    end

    return (; sp, o, r)
end

# Function for handling vector of actions (and therefore vector of observations)
function POMDPTools.obs_weight(
    m::HierarchicalMinExPOMDP, s, a::Vector{Tuple{Int64,Int64}}, sp, o::Vector{Float64}
)
    w = 1.0
    for (a_i, o_i) in zip(a, o)
        w *= obs_weight(m, s, a_i, sp, o_i)
    end
    return w
end

function POMDPTools.obs_weight(m::HierarchicalMinExPOMDP, s, a, sp, o)
    if (isterminal(m, s) || a in m.terminal_actions)
        w = Float64(isinf(o.thickness))
    else
        w =
            pdf(Normal(s.thickness[a...], m.σ_thickness), o.thickness) *
            pdf(Normal(s.grade[a...], m.σ_grade), o.grade)
    end
    return w
end