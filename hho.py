import numpy as np

from population import Population
from context import Context
from transformation import Transformation
from repair import Repair
from operators.exploration import exploration
from strategies.selector import StrategySelector


class HHO:
    """Harris Hawks Optimisation for permutation problems.

    The population lives in permutation space. At each iteration it is mapped to
    the continuous box [lb, ub]^dim, moved by the standard HHO equations, then
    decoded back to a valid permutation (rounding + repair, or rank decoding).

    After ``optimize()``:
        history        best fitness after each iteration
        history_evals  number of objective evaluations used at that point
        n_evals        total objective evaluations (including the extra
                       evaluations made by the dive strategies)
    """

    def __init__(
        self,
        objective,
        dim,
        population_size=20,
        max_iterations=100,
        lb=-1,
        ub=1,
        decoding="round",
        use_dive=True,
        persistent=False,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.objective = objective
        self.dim = dim
        self.use_dive = use_dive
        self.persistent = persistent
        self.max_iterations = max_iterations
        self.transformation = Transformation(lb, ub, decoding)
        self.repair = Repair()

        self.n_evals = 0
        self.history = []
        self.history_evals = []

        self.population = Population(population_size, dim)
        if persistent:
            # Random-key mode: the continuous vectors are the real state and
            # persist across iterations; permutations are only their decoding.
            self.continuous = np.random.uniform(lb, ub, (population_size, dim))
            self.population.positions = np.array([self._decode(x) for x in self.continuous])
        self.population.evaluate(self._evaluate)

    # ------------------------------------------------------------------
    def _evaluate(self, permutation) -> float:
        self.n_evals += 1
        return float(self.objective(permutation))

    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Continuous vector -> valid 1-based permutation (deterministic)."""
        x = self.transformation.clip(x)
        return self.repair.repair(self.transformation.backward_transform(x))

    def _continuous_objective(self, x: np.ndarray) -> float:
        return self._evaluate(self._decode(x))

    def _continuous_population(self) -> np.ndarray:
        return np.array([
            self.transformation.forward_transform(p)
            for p in self.population.positions
        ])

    # ------------------------------------------------------------------
    def optimize(self):
        for t in range(self.max_iterations):

            if self.persistent:
                continuous_pop = self.continuous.copy()  # snapshot for this iteration
            else:
                continuous_pop = self._continuous_population()

            rabbit_idx = int(np.argmin(self.population.fitness))
            rabbit = continuous_pop[rabbit_idx]
            Xmean = np.mean(continuous_pop, axis=0)

            for i in range(self.population.size):

                Xi = continuous_pop[i].copy()

                E0 = 2 * np.random.rand() - 1
                E = 2 * E0 * (1 - t / self.max_iterations)
                J = 2 * (1 - np.random.rand())
                r = np.random.rand()

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
                    strategy = StrategySelector.select(r, E, self.use_dive)
                    new_real = strategy.besiege(ctx)

                # Decode once, so the stored permutation is exactly the one
                # that was evaluated.
                trial = self._decode(new_real)
                trial_fitness = self._evaluate(trial)

                if trial_fitness < self.population.fitness[i]:
                    self.population.positions[i] = trial
                    self.population.fitness[i] = trial_fitness
                    if self.persistent:
                        self.continuous[i] = self.transformation.clip(new_real)

            self.history.append(self.population.best_fitness())
            self.history_evals.append(self.n_evals)

        return (
            self.population.best_position(),
            self.population.best_fitness(),
        )
