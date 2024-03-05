"""
State of a hierarchical mineral exploration problem
"""
mutable struct HierarchicalMinExState
    thickness::Matrix{Float32}
    grade::Matrix{Float32}
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
@with_kw struct HierarchicalMinExPOMDP <: POMDP{Any,Any,Any}
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
POMDPs.isterminal(m::HierarchicalMinExPOMDP, s) = s == :terminal
POMDPs.actions(m::HierarchicalMinExPOMDP) = [m.terminal_actions..., m.drill_locations...]
POMDPs.actionindex(m::HierarchicalMinExPOMDP, a) = findfirst([a] .== actions(m))
function calc_massive(m::HierarchicalMinExPOMDP, s)
    return sum(s.thickness .* (s.grade .> m.grade_threshold))
end
function extraction_reward(m::HierarchicalMinExPOMDP, s)
    return calc_massive(m, s) - m.extraction_cost
end
function calibrate_extraction_cost(m::HierarchicalMinExPOMDP, ds0)
    return mean(calc_massive(m, s) for s in ds0)
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
        o = SparseCat([nothing], [1.0])
    else
        o = product_distribution(
            Normal(s.thickness[a...], m.σ_thickness), Normal(s.grade[a...], m.σ_grade)
        )
    end
    return o
end

function POMDPs.gen(m::HierarchicalMinExPOMDP, s, a, rng=Random.GLOBAL_RNG)
    # Compute the next state
    sp = (a in m.terminal_actions || isterminal(m, s)) ? :terminal : s #copy(s)

    # Compute the reward
    r = reward(m, s, a)

    # observation
    o = rand(observation(m, a, s))

    return (; sp, o, r)
end

# Function for handling vector of actions (and therefore vector of observations)
# function POMDPTools.obs_weight(
#     m::HierarchicalMinExPOMDP, s, a::Vector{Tuple{Int64,Int64}}, sp, o::Vector{Float64}
# )
#     w = 1.0
#     for (a_i, o_i) in zip(a, o)
#         w *= obs_weight(m, s, a_i, sp, o_i)
#     end
#     return w
# end

# function POMDPTools.obs_weight(m::HierarchicalMinExPOMDP, s, a, sp, o)
#     if (isterminal(m, s) || a in m.terminal_actions)
#         w = Float64(isinf(o.thickness))
#     else
#         w =
#             pdf(Normal(s.thickness[a...], m.σ_thickness), o.thickness) *
#             pdf(Normal(s.grade[a...], m.σ_grade), o.grade)
#     end
#     return w
# end