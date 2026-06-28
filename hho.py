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
        ub=1
    ):

        self.objective = objective
        self.max_iterations = max_iterations

        self.population = Population(
            population_size,
            dim
        )

        self.transformation = Transformation(lb, ub)
        self.repair = Repair()


    def _continuous_population(self):

        return np.array([
            self.transformation.forward_transform(p)
            for p in self.population.positions
        ])


    def optimize(self):


        self.population.evaluate(self.objective)

        for t in range(self.max_iterations):

            continuous_population = self._continuous_population()

            rabbit_index = np.argmin(self.population.fitness)

            rabbit = continuous_population[rabbit_index]

            Xmean = np.mean(continuous_population, axis=0)

            for i in range(self.population.size):

                Xi = continuous_population[i].copy()

                E0 = 2 * np.random.rand() - 1
                E = 2 * E0 * (1 - t / self.max_iterations)

                J = 2 * (1 - np.random.rand())
                r = np.random.rand()

                ctx = Context(
                    Xi=Xi,
                    rabbit=rabbit,
                    population=self.population.positions,
                    continuous_population=continuous_population,
                    Xmean=Xmean,
                    
                    E=E,
                    J=J,
                    lb=self.transformation.lb,
                    ub=self.transformation.ub,
                    objective=self.objective,
                )

            

                if abs(E) >= 1:

                    new_real = exploration(ctx)

                else:

                    strategy = StrategySelector.select_strategy(r, E)

                    new_real = strategy.besiege(ctx)

           
                new_real = self.transformation.clip(new_real)

                trial = self.transformation.backward_transform(new_real)

                trial = self.repair.repair(trial)



                trial_fitness = self.objective(trial)

                if trial_fitness < self.population.fitness[i]:

                    self.population.positions[i] = trial
                    self.population.fitness[i] = trial_fitness

        return (
            self.population.best_position(),
            self.population.best_fitness()
        )