from hho import HHO
from benshmarks.sphere import shpere


optimizer = HHO(
    objective=shpere,
    dim=30,
    lb=-100,
    ub=100,
    population_size=30,
    max_iterations=500
)

best_position, best_fitness = optimizer.optimize()

print("Best Fitness :", best_fitness)
print("Best Position :", best_position)