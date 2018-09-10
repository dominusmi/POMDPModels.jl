const Vec2 = SVector{2,Int}
const StateTypes = Vec2

abstract type GridWorld <: MDP{StateTypes, Symbol} end

"""
    SimpleGridWorld creates an MDP problem with four allowed movements: up, down, left, right
"""
@with_kw struct SimpleGridWorld <: GridWorld
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{Vec2, Float64}    = Dict(Vec2(4,3)=>-10.0, Vec2(4,6)=>-5.0, Vec2(9,3)=>10.0, Vec2(8,8)=>3.0)
    terminate_in::Set{Vec2}         = Set((Vec2(4,3), Vec2(4,6), Vec2(9,3), Vec2(8,8)))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
end

@with_kw struct DiagonalGridWorld <: GridWorld
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{Vec2, Float64}    = Dict(Vec2(4,3)=>-10.0, Vec2(4,6)=>-5.0, Vec2(9,3)=>10.0, Vec2(8,8)=>3.0)
    terminate_in::Set{Vec2}         = Set((Vec2(4,3), Vec2(4,6), Vec2(9,3), Vec2(8,8)))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
end

function POMDPs.states(mdp::GridWorld)
    vec(StateTypes[Vec2(x, y) for x in 1:mdp.size[1], y in mdp.size[2]])
end

POMDPs.actions(mdp::SimpleGridWorld) = SVector(:up, :down, :left, :right)
POMDPs.actions(mdp::DiagonalGridWorld) = SVector(:n, :nw, :w, :sw, :s, :se, :e, :ne)

POMDPs.n_states(mdp::SimpleGridWorld) = prod(mdp.size)
POMDPs.n_states(mdp::DiagonalGridWorld) = prod(mdp.size)

POMDPs.n_actions(mdp::SimpleGridWorld) = 4
POMDPs.n_actions(mdp::DiagonalGridWorld) = 8

POMDPs.discount(mdp::SimpleGridWorld) = mdp.discount

POMDPs.stateindex(mdp::GridWorld, s::Vec2) = LinearIndices(mdp.size)[s...]


function POMDPs.actionindex(mdp::SimpleGridWorld, a::Symbol)
    aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)
    aind[a]
end
function POMDPs.actionindex(mdp::DiagonalGridWorld, a::Symbol)
    aind = Dict(:n=>1, :nw=>2, :w=>3, :sw=>4, :s=>5, :se=>6, :e=>7, :ne=>8)
    aind[a]
end

function directions(mdp::SimpleGridWorld)
    Dict(:up=>Vec2(0,1), :down=>Vec2(0,-1), :left=>Vec2(-1,0), :right=>Vec2(1,0))
end
function directions(mdp::DiagonalGridWorld)
    Dict(:n=>Vec2(0,1), :nw=>Vec2(-1,1), :w=>Vec2(-1,0), :sw=>Vec2(-1,-1), :s=>Vec2(0,-1), :se=>Vec2(1,-1), :e=>Vec2(1,0), :ne=>Vec2(1,1))
end

POMDPs.reward(mdp::GridWorld, s::Vec2, a::Symbol) = get(mdp.rewards, s, 0.0)

POMDPs.initialstate_distribution(mdp::GridWorld) = uniform_belief(mdp)
POMDPs.initialstate(mdp::GridWorld, rng::AbstractRNG) = Vec2(rand(rng, 1:mdp.size[1]), rand(rng, 1:mdp.size[2]))

POMDPs.isterminal(mdp::GridWorld, s::Vec2) = s ∈ mdp.terminate_in

inbounds(mdp::GridWorld, nb::Vec2) = ( 0 < nb[1] < mdp.size[1] ) && ( 0 < nb[2] < mdp.size[2] )

function POMDPs.transition(mdp::GridWorld, s::Vec2, a::Symbol)
    dir = directions(mdp)
    if s in mdp.terminate_in
        return SparseCat([s], [1.0])
    end

    neighbors = map(actions(mdp)) do act
        nb = s+dir[act]
        if !inbounds(mdp, nb)
            # If not inbounds, don't move
            nb = s
        end
        nb
    end

    probs = map(actions(mdp)) do act
        if act == a
            return mdp.tprob # probability of transitioning to the desired cell
        else
            return (1.0 - mdp.tprob)/( length(dir)-1 ) # probability of transitioning to another cell
        end
    end

    return SparseCat(neighbors, probs)
end


POMDPs.convert_a(::Type{A}, a::Symbol, mdp::SimpleGridWorld) where A<:AbstractArray = convert(A, SVector(actionindex(mdp)[a]))
POMDPs.convert_a(::Type{Symbol}, a::AbstractArray, mdp::SimpleGridWorld) = actions(mdp)[first(a)]
