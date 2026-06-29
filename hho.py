import numpy as np

from population import Population
from context import Context
from transformation import Transformation
from repair import Repair
from operators.exploration import exploration
from strategies.selector import StrategySelector


class HHO:

    def __init__(
        self,
        objective,       
        dim,
        population_size,
        max_iterations,
        lb=-1,
        ub=1,
    ):
        self.objective = objective
        self.max_iterations = max_iterations
        self.transformation = Transformation(lb, ub)
        self.repair = Repair()

        self.population = Population(population_size, dim)
        # Evaluate initial population with 1-based permutations
        self.population.positions = np.array([
            np.random.permutation(dim) + 1 for _ in range(population_size)
        ])
        self.population.evaluate(self.objective)


    def _continuous_objective(self, x: np.ndarray) -> float:
        x_clipped = self.transformation.clip(x)
        permutation = self.transformation.backward_transform(x_clipped)
        permutation = self.repair.repair(permutation)
        return float(self.objective(permutation))

    def _continuous_population(self) -> np.ndarray:
        return np.array([
            self.transformation.forward_transform(p)
            for p in self.population.positions
        ])

    def optimize(self):
        for t in range(self.max_iterations):

            continuous_pop = self._continuous_population()

            rabbit_idx = int(np.argmin(self.population.fitness))
            rabbit = continuous_pop[rabbit_idx]
            Xmean = np.mean(continuous_pop, axis=0)

            for i in range(self.population.size):

                Xi = continuous_pop[i].copy()

                E0 = 2 * np.random.rand() - 1
                E  = 2 * E0 * (1 - t / self.max_iterations)
                J  = 2 * (1 - np.random.rand())
                r  = np.random.rand()

                ctx = Context(
                    Xi=Xi,
                    rabbit=rabbit,
                    population=self.population.positions,
                    continuous_population=continuous_pop,
                    Xmean=Xmean,
                    E=E,
                    J=J,
                    lb=self.transformation.lb,
                    ub=self.transformation.ub,
                    continuous_objective=self._continuous_objective,
                )

                if abs(E) >= 1:
                    new_real = exploration(ctx)
                else:
                    strategy = StrategySelector.select(r, E)
                    new_real = strategy.besiege(ctx)

               
                trial_fitness = self._continuous_objective(new_real)

                if trial_fitness < self.population.fitness[i]:
                    self.population.positions[i] = self.repair.repair(
                        self.transformation.backward_transform(
                            self.transformation.clip(new_real)
                        )
                    )
                    self.population.fitness[i] = trial_fitness

        return (
            self.population.best_position(),
            self.population.best_fitness(),
        )