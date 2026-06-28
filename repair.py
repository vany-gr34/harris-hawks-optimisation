import numpy as np 
class Repair:


    def repair(self, permuation):
        n = len(permuation)
        permutation = np.clip(permuation, 1, n)  
        missing= list(set(range(1, n + 1)) - set(permutation))
        seen = set()
        for i in range(n):
            if permutation[i] in seen :
                permutation[i] = missing.pop()
            else:
                seen.add(permutation[i])
        return permutation