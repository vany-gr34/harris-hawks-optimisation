import numpy as np 


class Population:
    def __init__(self, size, dim, lb, ub):
        self.size = size
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.positions= np.random.uniform(lb, ub, (size, dim))

        self.fitness = np.full(size, np.inf)

    def evaluate(self, objective):
        for i in range(self.size):
            self.fitness[i] = objective(self.positions[i])
    def best_position(self):
        best_index = np.argmin(self.fitness)
        return self.positions[best_index].copy()
    
    def best_fitness(self):
        return np.min(self.fitness)
    def mean_position(self):
        return np.mean(self.positions, axis=0)
    def clip_positions(self):
        self.positions = np.clip(self.positions, self.lb, self.ub)
    