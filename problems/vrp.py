import numpy as np


class VRPProblem:
    """
    Capacitated VRP with a single depot (node 0).

    Encoding: a permutation of customers [1..n_customers].
    Routes are decoded greedily: scan left to right and start a new vehicle
    whenever adding the next customer would exceed the capacity. The number of
    vehicles is therefore not fixed in advance.

    With no argument, the original 6-customer toy instance is used. Pass
    ``distance``, ``demands`` and ``capacity`` (or use
    ``problems.tsplib.load_cvrp``) for real instances. ``demands[0]`` is the
    depot and must be 0.
    """

    def __init__(self, distance=None, demands=None, capacity=None):
        if distance is None:
            distance = [
                # 0   1   2   3   4   5   6
                [0,  12, 18, 10, 22, 15, 20],  # 0 depot
                [12,  0,  8, 14, 10, 18, 25],  # 1
                [18,  8,  0,  9, 16, 12, 14],  # 2
                [10, 14,  9,  0, 20,  7, 17],  # 3
                [22, 10, 16, 20,  0, 13,  8],  # 4
                [15, 18, 12,  7, 13,  0, 11],  # 5
                [20, 25, 14, 17,  8, 11,  0],  # 6
            ]
            demands = [0, 10, 15, 20, 12, 18, 14]
            capacity = 40

        self.distance = np.asarray(distance, dtype=float)
        self.demands = np.asarray(demands)
        self.vehicle_capacity = capacity
        self.n_customers = len(self.distance) - 1

    def _decode_routes(self, permutation):
        """Split permutation into feasible vehicle routes."""
        routes = []
        current_route = []
        current_load = 0

        for customer in permutation:
            demand = self.demands[customer]
            if current_load + demand > self.vehicle_capacity:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_load = demand
            else:
                current_route.append(customer)
                current_load += demand

        if current_route:
            routes.append(current_route)

        return routes

    def evaluate(self, permutation) -> float:
        """Total distance over all decoded routes (permutation is 1-based)."""
        total = 0.0
        for route in self._decode_routes(permutation):
            total += self.distance[0][route[0]]
            for i in range(len(route) - 1):
                total += self.distance[route[i]][route[i + 1]]
            total += self.distance[route[-1]][0]
        return float(total)
