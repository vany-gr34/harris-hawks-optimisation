import numpy as np


class TSPProblem:
    """Symmetric TSP. Solutions are 1-based permutations of the cities.

    With no argument, the original 5-city toy instance is used. Pass a
    ``distance`` matrix (or use ``problems.tsplib.load_tsp``) for real instances.
    """

    def __init__(self, distance=None):
        if distance is None:
            distance = [
                [0, 2, 9, 10, 7],
                [2, 0, 6, 4, 3],
                [9, 6, 0, 8, 5],
                [10, 4, 8, 0, 6],
                [7, 3, 5, 6, 0],
            ]
        self.distance = np.asarray(distance, dtype=float)
        self.n = len(self.distance)

    def evaluate(self, permutation) -> float:
        p = np.asarray(permutation) - 1
        return float(self.distance[p, np.roll(p, -1)].sum())
