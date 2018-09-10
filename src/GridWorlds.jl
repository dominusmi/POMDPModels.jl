#################################################################
# This file implements the grid world problem as an MDP.
# In the problem, the agent is tasked with navigating in a
# stochatic environemnt. For example, when the agent chooses
# to go right, it may not always go right, but may go up, down
# or left with some probability. The agent's goal is to reach the
# reward states. The states with a positive reward are terminal,
# while the states with a negative reward are not.
#################################################################

#################################################################
# States and Actions
#################################################################
# state of the agent in grid world
struct GridWorldState # this is not immutable because of how it is used in transition(), but maybe it should be
    x::Int64 # x position
    y::Int64 # y position
    done::Bool # entered the terminal reward state in previous step - there is only one terminal state
    GridWorldState(x,y,done) = new(x,y,done)
    GridWorldState() = new()
end

# simpler constructors
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)
# for state comparison
function ==(s1::GridWorldState,s2::GridWorldState)
    if s1.done && s2.done
        return true
    elseif s1.done || s2.done
        return false
    else
        return posequal(s1, s2)
    end
end
# for hashing states in dictionaries in Monte Carlo Tree Search
posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y
function hash(s::GridWorldState, h::UInt64 = zero(UInt64))
    if s.done
        return hash(s.done, h)
    else
        return hash(s.x, hash(s.y, h))
    end
end
Base.copy!(dest::GridWorldState, src::GridWorldState) = (dest.x=src.x; dest.y=src.y; dest.done=src.done; return dest)

# action taken by the agent indicates desired travel direction
const GridWorldAction = Symbol # deprecated - this is here so that other people's code won't break

#################################################################
# Grid World MDP
#################################################################
# the grid world mdp type
abstract type GridWorld <: MDP{GridWorldState, Symbol} end

# SimpleGridWorld only allows the four directions :up,:down,:left,:right
mutable struct SimpleGridWorld <: GridWorld
    size_x::Int64 # x size of the grid
    size_y::Int64 # y size of the grid
    reward_states::Vector{GridWorldState} # the states in which agent recieves reward
    reward_values::Vector{Float64} # reward values for those states
    bounds_penalty::Float64 # penalty for bumping the wall (will be added to reward)
    tprob::Float64 # probability of transitioning to the desired state
    terminals::Set{GridWorldState}
    discount_factor::Float64 # disocunt factor
end

# DiagonalGridWorld allows for the eight actions :n,:nw,:w,..:e, :ne representing
# north, north-west, ..
# SimpleGridWorld only allows the four directions :up,:down,:left,:right
mutable struct DiagonalGridWorld <: GridWorld
    size_x::Int64 # x size of the grid
    size_y::Int64 # y size of the grid
    reward_states::Vector{GridWorldState} # the states in which agent recieves reward
    reward_values::Vector{Float64} # reward values for those states
    bounds_penalty::Float64 # penalty for bumping the wall (will be added to reward)
    tprob::Float64 # probability of transitioning to the desired state
    terminals::Set{GridWorldState}
    discount_factor::Float64 # disocunt factor
end


# we use key worded arguments so we can change any of the values we pass in
function SimpleGridWorld(sx::Int64, # size_x
                   sy::Int64; # size_y
                   rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)],
                   rv::Vector{Float64}=[-10.,-5,10,3],
                   penalty::Float64=0.0, # penalty for trying to go out of bounds  (will be added to reward)
                   tp::Float64=0.7, # tprob
                   discount_factor::Float64=0.95,
                   terminals=Set{GridWorldState}([rs[i] for i in filter(i->rv[i]>0.0, 1:length(rs))]))
    return SimpleGridWorld(sx, sy, rs, rv, penalty, tp, Set{GridWorldState}(terminals), discount_factor)
end

function DiagonalGridWorld(sx::Int64, # size_x
                   sy::Int64; # size_y
                   rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)],
                   rv::Vector{Float64}=[-10.,-5,10,3],
                   penalty::Float64=0.0, # penalty for trying to go out of bounds  (will be added to reward)
                   tp::Float64=0.7, # tprob
                   discount_factor::Float64=0.95,
                   terminals=Set{GridWorldState}([rs[i] for i in filter(i->rv[i]>0.0, 1:length(rs))]))
    return DiagonalGridWorld(sx, sy, rs, rv, penalty, tp, Set{GridWorldState}(terminals), discount_factor)
end


SimpleGridWorld(;sx::Int64=10, sy::Int64=10, kwargs...) = SimpleGridWorld(sx, sy; kwargs...)
DiagonalGridWorld(;sx::Int64=10, sy::Int64=10, kwargs...) = DiagonalGridWorld(sx, sy; kwargs...)


#################################################################
# State and Action Spaces
#################################################################
# This could probably be implemented more efficiently without vectors

function states(mdp::GridWorld)
    s = vec(collect(GridWorldState(x, y, false) for x in 1:mdp.size_x, y in 1:mdp.size_y))
    push!(s, GridWorldState(0, 0, true))
    return s
end

actions(mdp::SimpleGridWorld) = [:up, :down, :left, :right]
actions(mdp::DiagonalGridWorld) = [:n, :nw, :w, :sw, :s, :se, :e, :ne]


n_states(mdp::GridWorld) = mdp.size_x*mdp.size_y+1
n_actions(mdp::SimpleGridWorld) = 4
n_actions(mdp::DiagonalGridWorld) = 8


function reward(mdp::GridWorld, state::GridWorldState, action::Symbol)
    if state.done
        return 0.0
    end
    r = static_reward(mdp, state)
    if !inbounds(mdp, state, action)
        r += mdp.bounds_penalty
    end
  return r
end

"""
    static_reward(mdp::GridWorld, state::GridWorldState)

Return the reward for being in the state (the reward not including bumping)
"""
function static_reward(mdp::GridWorld, state::GridWorldState)
    r = 0.0
    n = length(mdp.reward_states)
    for i = 1:n
        if posequal(state, mdp.reward_states[i])
            r += mdp.reward_values[i]
        end
    end
    return r
end

#checking boundries- x,y --> points of current state
inbounds(mdp::GridWorld,x::Int64,y::Int64) = 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
inbounds(mdp::GridWorld,state::GridWorldState) = inbounds(mdp, state.x, state.y)

"""
    inbounds(mdp::GridWorld, s::GridWorldState, a::Symbol)

Return false if `a` is trying to go out of bounds, true otherwise.
"""
function inbounds(mdp::GridWorld, s::GridWorldState, a::Symbol)
    xdir = s.x
    ydir = s.y
    if a == :right
        xdir += 1
    elseif a == :left
        xdir -= 1
    elseif a == :up
        ydir += 1
    else
        # @assert a == :down
        ydir -= 1
    end
    return inbounds(mdp, GridWorldState(xdir, ydir, s.done))
end

function inbounds(mdp::GridWorld, s::GridWorldState, a::Symbol)
    xdir = s.x
    ydir = s.y
    if a == :n
        ydir += 1
    elseif a == :nw
        ydir += 1
        xdir -= 1
    elseif a == :w
        xdir -= 1
    elseif a == :sw
        xdir -= 1
        ydir -= 1
    elseif a == :s
        ydir -= 1
    elseif a == :se
        xdir += 1
        ydir -= 1
    elseif a == :e
        xdir += 1
    elseif a == :ne
        xdir += 1
        ydir += 1
    end
    return inbounds(mdp, GridWorldState(xdir, ydir, s.done))
end


function fill_probability!(p::AbstractVector{Float64}, val::Float64, index::Int64)
    for i = 1:length(p)
        if i == index
            p[i] = val
        else
            p[i] = 0.0
        end
    end
end

"""
    Returns the set of GridWorld neighbours given current location (x,y)
"""
function neighbors_states(mdp::SimpleGridWorld, x::Integer, y::Integer)
    MVector(
        GridWorldState(x+1, y, false), # right
        GridWorldState(x-1, y, false), # left
        GridWorldState(x, y-1, false), # down
        GridWorldState(x, y+1, false), # up
        GridWorldState(x, y, false)    # stay
    )
end

function neighbors_states(mdp::DiagonalGridWorld, x::Integer, y::Integer)
    MVector(
        GridWorldState(x, y+1, false), # north
        GridWorldState(x+1, y+1, false), # north east
        GridWorldState(x+1, y, false), # east
        GridWorldState(x+1, y-1, false), # south east
        GridWorldState(x, y-1, false), # south
        GridWorldState(x-1, y-1, false), # south west
        GridWorldState(x-1, y, false), # south west
        GridWorldState(x-1, y+1, false), # north west
        GridWorldState(x, y, false)    # stay
       )
end

"""
    Returns the respective action for neighbors as indexed in neighbors_states
"""
function target_neighbor_index(mdp::SimpleGridWorld, a::Symbol)
    target_neighbor = 0
    if a == :right
        target_neighbor = 1
    elseif a == :left
        target_neighbor = 2
    elseif a == :down
        target_neighbor = 3
    elseif a == :up
        target_neighbor = 4
    end
    target_neighbor
end

function target_neighbor_index(mdp::DiagonalGridWorld, a::Symbol)
    target_neighbor = 0
    if a == :n
        target_neighbor = 1
	elseif a == :ne
        target_neighbor = 2
	elseif a == :e
        target_neighbor = 3
	elseif a == :se
        target_neighbor = 4
    elseif a == :s
        target_neighbor = 5
    elseif a == :sw
        target_neighbor = 6
    elseif a == :w
        target_neighbor = 7
    elseif a == :nw
        target_neighbor = 8
	end
    target_neighbor
end

function transition(mdp::GridWorld, state::GridWorldState, action::Symbol)

    a = action
    x = state.x
    y = state.y

    terminal_idx = n_actions(mdp)+1
    neighbors = neighbors_states(mdp, x, y)

    probability = MVector{terminal_idx, Float64}(undef)
    fill!(probability, 0.0)

    if state.done
        fill_probability!(probability, 1.0, terminal_idx)
        neighbors[ terminal_idx ] = GridWorldState(x, y, true)
        return SparseCat(neighbors, probability)
    end

    reward_states = mdp.reward_states
    reward_values = mdp.reward_values
    n = length(reward_states)
    if state in mdp.terminals
        fill_probability!(probability, 1.0, terminal_idx)
        neighbors[ terminal_idx ] = GridWorldState(x, y, true)
        return SparseCat(neighbors, probability)
    end

    # The following match the definition of neighbors
    # given above
    target_neighbor = target_neighbor_index(mdp, a)
    @assert target_neighbor > 0

    if !inbounds(mdp, neighbors[target_neighbor])
        # If would transition out of bounds, stay in
        # same cell with probability 1
        fill_probability!(probability, 1.0, terminal_idx)
    else
        probability[target_neighbor] = mdp.tprob

        oob_count = 0 # number of out of bounds neighbors

        for i = 1:length(neighbors)
            if !inbounds(mdp, neighbors[i])
                oob_count += 1
                @assert probability[i] == 0.0
            end
        end

        new_probability = (1.0 - mdp.tprob)/( (n_actions(mdp)-1)-oob_count)

        for i = 1:n_actions(mdp) # do not include neighbor 5
            if inbounds(mdp, neighbors[i]) && i != target_neighbor
                probability[i] = new_probability
            end
        end
    end

    return SparseCat(neighbors, probability)
end


function action_index(mdp::SimpleGridWorld, a::Symbol)
    # lazy, replace with switches when they arrive
    if a == :up
        return 1
    elseif a == :down
        return 2
    elseif a == :left
        return 3
    elseif a == :right
        return 4
    else
        error("Invalid action symbol $a")
    end
end

function action_index(mdp::DiagonalGridWorld, a::Symbol)
    # lazy, replace with switches when they arrive
    if a == :n
        return 1
    elseif a == :nw
        return 2
    elseif a == :w
        return 3
    elseif a == :sw
        return 4
    elseif a == :s
        return 5
    elseif a == :se
        return 6
    elseif a == :e
        return 7
    elseif a == :ne
        return 8
    else
        error("Invalid action symbol $a")
    end
end



function state_index(mdp::GridWorld, s::GridWorldState)
    return s2i(mdp, s)
end

function s2i(mdp::GridWorld, state::GridWorldState)
    if state.done
        return mdp.size_x*mdp.size_y + 1
    else
        return LinearIndices((mdp.size_x, mdp.size_y))[state.x, state.y]
    end
end

#=
function i2s(mdp::GridWorld, i::Int)
end
=#

isterminal(mdp::GridWorld, s::GridWorldState) = s.done

discount(mdp::GridWorld) = mdp.discount_factor

convert_s(::Type{A}, s::GridWorldState, mdp::GridWorld) where A<:AbstractArray = Float64[s.x, s.y, s.done]
convert_s(::Type{GridWorldState}, s::AbstractArray, mdp::GridWorld) = GridWorldState(s[1], s[2], s[3])

function a2int(a::Symbol, mdp::GridWorld)
    if a == :up
        return 0
    elseif a == :down
        return 1
    elseif a == :left
        return 2
    elseif a == :right
        return 3
    else
        throw("Action $a is invalid")
    end
end

function int2a(a::Int, mdp::GridWorld)
    if a == 0
        return :up
    elseif a == 1
        return :down
    elseif a == 2
        return :left
    elseif a == 3
        return :right
    else
        throw("Action $a is invalid")
    end
end

function a2int(a::Symbol, mdp::DiagonalGridWorld)
    action_index(mdp, a)
end

function int2a(a::Int, mdp::DiagonalGridWorld)
    if a == 1
        return :n
    elseif a == 2
        return :nw
    elseif a == 3
        return :w
    elseif a == 4
        return :sw
    elseif a == 5
        return :s
    elseif a == 6
        return :se
    elseif a == 7
        return :e
    elseif a == 8
        return :ne
    else
        throw("Action $a is invalid")
    end
end



convert_a(::Type{A}, a::Symbol, mdp::GridWorld) where A<:AbstractArray = [Float64(a2int(a, mdp))]
convert_a(::Type{Symbol}, a::A, mdp::GridWorld) where A<:AbstractArray = int2a(Int(a[1]), mdp)

initialstate(mdp::GridWorld, rng::AbstractRNG) = GridWorldState(rand(rng, 1:mdp.size_x), rand(rng, 1:mdp.size_y))

# Visualization
#=
function colorval(val, brightness::Real = 1.0)
    val = convert(Vector{Float64}, val)
    x = 255 .- min.(255, 255 * (abs.(val) ./ 10.0) .^ brightness)
    r = 255 * ones(size(val))
    g = 255 * ones(size(val))
    b = 255 * ones(size(val))
    r[val .>= 0] = x[val .>= 0]
    b[val .>= 0] = x[val .>= 0]
    g[val .< 0] = x[val .< 0]
    b[val .< 0] = x[val .< 0]
    return (r, g, b)
end

function plot(g::GridWorld, f::Function)
    V = map(f, states(g))
    plot(g, V)
end

function plot(mdp::GridWorld, V::Vector, state=GridWorldState(0,0,true))
    o = IOBuffer()
    sqsize = 1.0
    twid = 0.05
    (r, g, b) = colorval(V)
    for s in states(mdp)
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval
            println(o, "\\definecolor{currentcolor}{RGB}{$(r[i]),$(g[i]),$(b[i])}")
            println(o, "\\fill[currentcolor] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            if s == state
                println(o, "\\fill[orange] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            end
            vs = @sprintf("%0.2f", V[i])
            println(o, "\\node[above right] at ($((xval-1) * sqsize), $((yval) * sqsize)) {\$$(vs)\$};")
        end
    end
    println(o, "\\draw[black] grid(10,10);")
    tikzDeleteIntermediate(false)
    TikzPicture(String(take!(o)), options="scale=1.25")
end

function plot(mdp::GridWorld, state=GridWorldState(0,0,true))
    plot(mdp, zeros(n_states(mdp)), state)
end

function plot(g::GridWorld, f::Function, policy::Policy, state=GridWorldState(0,0,true))
    V = map(f, states(g))
    plot(g, V, policy, state)
end

function plot(mdp::GridWorld, V::Vector, policy::Policy, state=GridWorldState(0,0,true))
    o = IOBuffer()
    sqsize = 1.0
    twid = 0.05
    (r, g, b) = colorval(V)
    for s in states(mdp)
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval
            println(o, "\\definecolor{currentcolor}{RGB}{$(r[i]),$(g[i]),$(b[i])}")
            println(o, "\\fill[currentcolor] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            if s == state
                println(o, "\\fill[orange] ($((xval-1) * sqsize),$((yval) * sqsize)) rectangle +($sqsize,$sqsize);")
            end
        end
    end
    println(o, "\\begin{scope}[fill=gray]")
    for s in states(mdp)
        if !s.done
            (xval, yval) = (s.x, mdp.size_y-s.y+1)
            i = state_index(mdp, s)
            yval = 10 - yval + 1
            c = [xval, yval] * sqsize .- sqsize / 2
            C = [c'; c'; c']'
            RightArrow = [0 0 sqsize/2; twid -twid 0]
            dir = action(policy, s)
            if dir == :left
                A = [-1 0; 0 -1] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :right
                A = RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :up
                A = [0 -1; 1 0] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end
            if dir == :down
                A = [0 1; -1 0] * RightArrow + C
                println(o, "\\fill ($(A[1]), $(A[2])) -- ($(A[3]), $(A[4])) -- ($(A[5]), $(A[6])) -- cycle;")
            end

            vs = @sprintf("%0.2f", V[i])
            println(o, "\\node[above right] at ($((xval-1) * sqsize), $((yval-1) * sqsize)) {\$$(vs)\$};")
        end
    end
    println(o, "\\end{scope}");
    println(o, "\\draw[black] grid(10,10);");
    TikzPicture(String(take!(o)), options="scale=1.25")
end
=#
