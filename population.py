import numpy as np 


class Population:
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.positions=np.array([np.random.permutation(dim) for _ in range(size)])
        self.fitness = np.full(size, np.inf)

    def evaluate(self, objective):
        for i in range(self.size):
            self.fitness[i] = objective(self.positions[i])
    def best_position(self):
        best_index = np.argmin(self.fitness)
        return self.positions[best_index].copy()
    
    def best_fitness(self):
        return np.min(self.fitness)
    

    #i think this is not gonna be neede d

    #def mean_position(self):
        #return np.mean(self.positions, axis=0)
   # def clip(self):
        #self.positions = np.clip(self.positions, self.lb, self.ub)
#population = Population(size=10, dim=5)
#print("Initial Positions:\n", population.positions)