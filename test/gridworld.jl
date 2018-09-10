using POMDPModels
using Test
# using NBInclude

let
    problem = SimpleGridWorld()

	s = Vec2(1,1)
	@test POMDPModels.inbounds(problem, s) == true

	T = transition(problem, s, :down)
	@test T.vals == [ Vec2(1,2), Vec2(1,1), Vec2(1,1), Vec2(2,1)]
	@test T.probs ≈ [ 0.1, 0.7, 0.1, 0.1 ]

	@test isterminal(problem, first(problem.terminate_in) ) == true

	@test reward(problem, Vec2(4,3), :up) == -10.

	for s in states(problem)
		for a in actions(problem)
			T = transition(problem, s, a)
			@assert sum(T.probs) ≈ 1.00
		end
	end

	problem = DiagonalGridWorld()

	s = Vec2(1,1)
	@test POMDPModels.inbounds(problem, s) == true

	T = transition(problem, s, :n)
	@test T.vals == [ Vec2(1,2), Vec2(1,1), Vec2(1,1), Vec2(1,1), Vec2(1,1), Vec2(1,1), Vec2(2,1), Vec2(2,2)]
	@test T.probs ≈ [ 0.7, 0.3/7, 0.3/7, 0.3/7, 0.3/7, 0.3/7, 0.3/7, 0.3/7 ]

	@test isterminal(problem, first(problem.terminate_in) ) == true

	@test reward(problem, Vec2(4,3), :n) == -10.
	@test reward(problem, Vec2(4,2), :n) == 0.


	for s in states(problem)
		for a in actions(problem)
			T = transition(problem, s, a)
			@assert sum(T.probs) ≈ 1.00
		end
	end
end
