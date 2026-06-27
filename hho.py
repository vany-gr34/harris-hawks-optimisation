import numpy as np
from population import Population
from context import Context
from operators.exploration import exploration
from strategies.selector import StrategySelector


class HHO:

    def __init__(
        self,
        objective,
        dim,
        lb,
        ub,
        population_size,
        max_iterations,
    ):

        self.objective = objective
        self.max_iterations = max_iterations

        self.population = Population(
            population_size,
            dim,
            lb,
            ub
        )

    def optimize(self):

        
        self.population.evaluate(self.objective)

        rabbit = self.population.best_position()

        for t in range(self.max_iterations):

            rabbit = self.population.best_position()
            mean = self.population.mean_position()

            for i in range(self.population.size):

                Xi = self.population.positions[i]

                E0 = 2 * np.random.rand() - 1
                E = 2 * E0 * (1 - t / self.max_iterations)

                J = 2 * (1 - np.random.rand())
                r = np.random.rand()

                ctx = Context(
                    Xi=Xi,
                    rabbit=rabbit,
                    population=self.population.positions,
                    Xmean=mean,
                    lb=self.population.lb,
                    ub=self.population.ub,
                    E=E,
                    J=J,
                    objective=self.objective,
                )
        

                if abs(E) >= 1:
                    new_position = exploration(ctx)

                else:
                    strategy = StrategySelector.select_strategy(r, E)
                    new_position = strategy.besiege(ctx)

                self.population.positions[i] = new_position

            self.population.clip()

            self.population.evaluate(self.objective)

        return (
            self.population.best_position(),
            self.population.best_fitness()
        )