"""Benchmark HHO (and variants) against GA / SA / random search.

Usage:
    python benchmark.py                      # full benchmark (20 runs / instance)
    python benchmark.py --runs 5 --instances eil51 berlin52

For every (instance, seed) the full HHO is run first; its number of objective
evaluations is then used as the evaluation budget of GA, SA and random search,
so the comparison is at an equal budget. Raw per-run results (including
convergence curves) are saved to results/raw.json; use report.py to turn them
into the summary table and plots.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import baselines
from hho import HHO
from problems.tsplib import load_cvrp, load_tsp

# instance -> (loader, file, best known / optimal value)
INSTANCES = {
    "eil51":     (load_tsp,  "data/eil51.tsp",     426),
    "berlin52":  (load_tsp,  "data/berlin52.tsp",  7542),
    "st70":      (load_tsp,  "data/st70.tsp",      675),
    "eil76":     (load_tsp,  "data/eil76.tsp",     538),
    "kroA100":   (load_tsp,  "data/kroA100.tsp",   21282),
    "A-n32-k5":  (load_cvrp, "data/A-n32-k5.vrp",  784),
    "A-n45-k6":  (load_cvrp, "data/A-n45-k6.vrp",  944),
}

HHO_VARIANTS = {
    "HHO":            dict(),
    "HHO (no dive)":  dict(use_dive=False),
    "HHO (rank dec)": dict(decoding="rank"),
    "HHO (rand key)": dict(decoding="rank", persistent=True),
}
GRID = np.linspace(0.02, 1.0, 50)  # fractions of the reference budget


def _dim(problem):
    return problem.n if hasattr(problem, "n") else problem.n_customers


def _curve_from_history(history, history_evals, grid_evals):
    history, history_evals = np.asarray(history), np.asarray(history_evals)
    idx = np.searchsorted(history_evals, grid_evals, side="right") - 1
    return history[np.clip(idx, 0, len(history) - 1)]


def run_one(task):
    name, seed, pop, iters = task
    loader, path, _ = INSTANCES[name]
    problem = loader(path)
    dim = _dim(problem)
    out = {}

    ref_evals = None
    for label, kw in HHO_VARIANTS.items():
        t0 = time.time()
        h = HHO(problem.evaluate, dim, population_size=pop, max_iterations=iters,
                seed=seed, **kw)
        perm, cost = h.optimize()
        assert sorted(perm.tolist()) == list(range(1, dim + 1))
        if ref_evals is None:
            ref_evals = h.n_evals
        grid = GRID * ref_evals
        out[label] = dict(cost=cost, evals=h.n_evals, time=time.time() - t0,
                          curve=_curve_from_history(h.history, h.history_evals, grid).tolist())

    for label, fn in [("GA", baselines.genetic_algorithm),
                      ("SA", baselines.simulated_annealing),
                      ("Random", baselines.random_search)]:
        t0 = time.time()
        perm, cost, curve = fn(problem.evaluate, dim, ref_evals, seed=seed)
        assert sorted(perm.tolist()) == list(range(1, dim + 1))
        grid = np.maximum((GRID * ref_evals).astype(int) - 1, 0)
        out[label] = dict(cost=cost, evals=ref_evals, time=time.time() - t0,
                          curve=curve[grid].tolist())
    return name, seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--instances", nargs="+", default=list(INSTANCES))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default="results/raw.json")
    args = ap.parse_args()

    tasks = [(n, s, args.pop, args.iters) for n in args.instances for s in range(args.runs)]
    results = {n: {"best_known": INSTANCES[n][2], "runs": []} for n in args.instances}

    t0 = time.time()
    with Pool(args.procs) as pool:
        for k, (name, seed, out) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            results[name]["runs"].append(dict(seed=seed, methods=out))
            print(f"[{k}/{len(tasks)}] {name} seed={seed}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), grid=GRID.tolist(), results=results), f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
