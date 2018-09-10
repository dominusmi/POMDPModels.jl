using POMDPs
using POMDPModels
using Test
# using NBInclude

let


    #### GridWorldState tests ####

    @test GridWorldState(1,1,false) == GridWorldState(1,1,false)
    @test hash(GridWorldState(1,1,false)) == hash(GridWorldState(1,1,false))
    @test GridWorldState(1,2,false) != GridWorldState(1,1,false)
    @test GridWorldState(1,2,true) == GridWorldState(1,1,true)
    @test hash(GridWorldState(1,2,true)) == hash(GridWorldState(1,1,true))

    #### SimpleGridWorld tests ####

    problem = SimpleGridWorld()

    # XXX simulation
    # policy = RandomPolicy(problem)

    # sim = HistoryRecorder(rng=MersenneTwister(1), max_steps=1000)

    # hist = simulate(sim, problem, policy, GridWorldState(1,1))

    # for i in 1:length(hist.action_hist)
    #     td = transition(problem, hist.state_hist[i], hist.action_hist[i])
    #     @test sum(td.probs) ≈ 1.0 atol=0.01
    #     for p in td.probs
    #         @test p >= 0.0
    #     end
    # end


    sv = convert_s(Array{Float64}, GridWorldState(1, 1, false), problem)
    @test sv == [1.0, 1.0, 0.0]
    sv = convert_s(Array{Float64}, GridWorldState(5, 3, false), problem)
    @test sv == [5.0, 3.0, 0.0]
    s = convert_s(GridWorldState, sv, problem)
    @test s == GridWorldState(5, 3, false)

    av = convert_a(Array{Float64}, :up, problem)
    @test av == [0.0]
    a = convert_a(Symbol, av, problem)
    @test a == :up

    trans_prob_consistency_check(problem)

    #### DiagonalGridWorld tests ####
    
    problem = DiagonalGridWorld()

    sv = convert_s(Array{Float64}, GridWorldState(1, 1, false), problem)
    @test sv == [1.0, 1.0, 0.0]
    sv = convert_s(Array{Float64}, GridWorldState(5, 3, false), problem)
    @test sv == [5.0, 3.0, 0.0]
    s = convert_s(GridWorldState, sv, problem)
    @test s == GridWorldState(5, 3, false)

    av = convert_a(Array{Float64}, :e, problem)
    @test av == [7.0]
    a = convert_a(Symbol, av, problem)
    @test a == :e

    trans_prob_consistency_check(problem)
end

# XXX simulation
# let
#     @nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "GridWorld Visualization.ipynb"))
# end
