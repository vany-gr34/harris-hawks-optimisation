from hho import HHO
from problems.tsp import TSPProblem
from problems.vrp import VRPProblem


# ── TSP ────────────────────────────────────────────────────────────────
print("=" * 40)
print("TSP — 5 cities")
print("=" * 40)

tsp = TSPProblem()

tsp_optimizer = HHO(
    objective=tsp.evaluate,
    dim=5,
    population_size=20,
    max_iterations=100,
)

best_route, best_cost = tsp_optimizer.optimize()
print("Best route:", best_route)
print("Best cost: ", best_cost)


# ── VRP ────────────────────────────────────────────────────────────────
print()
print("=" * 40)
print("VRP — 6 customers, capacity=40")
print("=" * 40)

vrp = VRPProblem()

vrp_optimizer = HHO(
    objective=vrp.evaluate,
    dim=vrp.n_customers,
    population_size=50,
    max_iterations=100,
)

best_route, best_cost = vrp_optimizer.optimize()
print("Best route (customer order):", best_route)
print("Decoded vehicle routes:")
for i, route in enumerate(vrp._decode_routes(best_route), 1):
    print(f"  Vehicle {i}: depot → {' → '.join(str(c) for c in route)} → depot")
print("Best cost: ", best_cost)