"""Run with:  python -m pytest -q   (or: python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hho import HHO
from repair import Repair
from transformation import Transformation
from problems.tsp import TSPProblem
from problems.tsplib import load_tsp, load_cvrp
import baselines

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def is_perm(p, n):
    return sorted(np.asarray(p).tolist()) == list(range(1, n + 1))


def test_transform_roundtrip():
    for decoding in ("round", "rank"):
        t = Transformation(-1, 1, decoding)
        for n in (2, 5, 30):
            p = np.random.permutation(n) + 1
            assert (t.backward_transform(t.forward_transform(p)) == p).all()


def test_repair_always_valid():
    r = Repair()
    for _ in range(200):
        n = np.random.randint(2, 40)
        raw = np.random.randint(1, n + 1, size=n)
        assert is_perm(r.repair(raw), n)
    assert (r.repair(np.array([1, 2, 3, 4])) == np.array([1, 2, 3, 4])).all()


def test_repair_is_deterministic():
    raw = np.array([3, 3, 3, 1, 1])
    assert (Repair().repair(raw) == Repair().repair(raw)).all()


def test_hho_returns_consistent_solution():
    p = TSPProblem()
    for kw in (dict(), dict(use_dive=False), dict(decoding="rank"),
               dict(decoding="rank", persistent=True)):
        h = HHO(p.evaluate, 5, population_size=10, max_iterations=20, seed=0, **kw)
        perm, cost = h.optimize()
        assert is_perm(perm, 5)
        assert cost == p.evaluate(perm)
        assert len(h.history) == 20 and h.n_evals >= 10 + 20 * 10


def test_hho_solves_toy_tsp():
    p = TSPProblem()
    _, cost = HHO(p.evaluate, 5, population_size=20, max_iterations=100, seed=0).optimize()
    assert cost == 26.0  # optimum of the 5-city instance


def test_tsplib_loaders():
    tsp = load_tsp(os.path.join(DATA, "berlin52.tsp"))
    assert tsp.n == 52
    cvrp = load_cvrp(os.path.join(DATA, "A-n32-k5.vrp"))
    assert cvrp.n_customers == 31 and cvrp.vehicle_capacity == 100


def test_baselines_respect_budget():
    p = TSPProblem()
    for fn in (baselines.random_search, baselines.simulated_annealing, baselines.genetic_algorithm):
        perm, cost, curve = fn(p.evaluate, 5, 300, seed=0)
        assert is_perm(perm, 5) and len(curve) == 300 and curve[-1] == cost


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
