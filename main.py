from hho import HHO
from problems.tsp import TSPProblem

problem = TSPProblem()

optimizer = HHO(
    objective=problem.evaluate,
    dim=5,
    population_size=20,
    max_iterations=100,
)

best_route, best_cost = optimizer.optimize()

print(best_route)
print(best_cost)