import numpy as np


class TSPProblem:

    def __init__(self):
        self.distance = np.array([
            [0, 2, 9, 10, 7],
            [2, 0, 6, 4, 3],
            [9, 6, 0, 8, 5],
            [10, 4, 8, 0, 6],
            [7, 3, 5, 6, 0],
        ])

    def evaluate(self, permutation):

        total = 0

        for i in range(len(permutation) - 1):
            a = permutation[i] - 1
            b = permutation[i + 1] - 1

            total += self.distance[a][b]

        total += self.distance[permutation[-1] - 1][permutation[0] - 1]

        return total