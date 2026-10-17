import numpy as np


class Population:
    """A set of 1-based permutations together with their fitness values."""

    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.positions = np.array([np.random.permutation(dim) + 1 for _ in range(size)])
        self.fitness = np.full(size, np.inf)

    def evaluate(self, objective):
        for i in range(self.size):
            self.fitness[i] = objective(self.positions[i])

    def best_position(self):
        return self.positions[int(np.argmin(self.fitness))].copy()

    def best_fitness(self):
        return float(np.min(self.fitness))
