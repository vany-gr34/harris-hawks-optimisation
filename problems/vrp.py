import numpy as np


class VRPProblem:
    """
    Capacitated VRP with a single depot (node 0).

    Encoding: a permutation of customers [1..n_customers].
    Routes are decoded greedily — we scan left to right and start a
    new vehicle whenever adding the next customer would exceed capacity.

    Example with 6 customers, 2 vehicles, capacity=40:
        permutation [3,1,5,2,6,4]
        → vehicle 1: depot→3→1→5→depot  (if demand fits)
        → vehicle 2: depot→2→6→4→depot
    """

    def __init__(self):
       
        self.distance = np.array([
            # 0   1   2   3   4   5   6
            [0,  12, 18, 10, 22, 15, 20],  # 0 depot
            [12,  0,  8, 14, 10, 18, 25],  # 1
            [18,  8,  0,  9, 16, 12, 14],  # 2
            [10, 14,  9,  0, 20,  7, 17],  # 3
            [22, 10, 16, 20,  0, 13,  8],  # 4
            [15, 18, 12,  7, 13,  0, 11],  # 5
            [20, 25, 14, 17,  8, 11,  0],  # 6
        ])

        # Demand of each customer (index 0 = depot, demand 0)
        self.demands = np.array([0, 10, 15, 20, 12, 18, 14])

        self.vehicle_capacity = 40
        self.n_customers = 6      # must match dim in HHO

    def _decode_routes(self, permutation):
        """Split permutation into feasible vehicle routes."""
        routes = []
        current_route = []
        current_load = 0

        for customer in permutation:
            demand = self.demands[customer]
            if current_load + demand > self.vehicle_capacity:
                # Current vehicle is full — save route and start new one
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

    def evaluate(self, permutation):
        """
        Compute total distance for all vehicle routes.
        permutation: 1-based array of customer indices, e.g. [3,1,5,2,6,4]
        """
        routes = self._decode_routes(permutation)
        total = 0

        for route in routes:
            # depot → first customer
            total += self.distance[0][route[0]]
            # between customers
            for i in range(len(route) - 1):
                total += self.distance[route[i]][route[i + 1]]
            # last customer → depot
            total += self.distance[route[-1]][0]

        return total