import numpy as np 



class Transformation:

    def __init__(self, ub=1, lb=-1):
        self.ub = ub
        self.lb = lb

    def forward_transform(self, permutation):
        z=np.empty(len(permutation))
        for i , x in enumerate(permutation):
            z[i] = self.lb + (self.ub - self.lb) * x / (len(permutation) - 1)
        return z

    def backward_transform(self, transformation):
        permutation=np.empty(len(transformation), dtype=int)
        for i , z in enumerate(transformation):
            x = round((z - self.lb) / (self.ub - self.lb) * (len(transformation) - 1)) +1

            permutation[i] = x  
        return permutation    
    def clip(self, transformation):
        return np.clip(transformation, self.lb, self.ub)