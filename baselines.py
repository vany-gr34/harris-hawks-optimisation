"""Reference metaheuristics used to put HHO's results in context.

All of them work on 1-based permutations, take the same ``objective`` as HHO,
and stop after exactly ``max_evals`` objective evaluations, so comparisons are
made at an equal evaluation budget.

Each returns (best_permutation, best_cost, curve) where ``curve[k]`` is the
best cost found after k+1 evaluations.
"""

import numpy as np


class _Tracker:
    """Counts evaluations and records the best-so-far curve."""

    def __init__(self, objective, max_evals):
        self.objective = objective
        self.max_evals = max_evals
        self.curve = np.empty(max_evals)
        self.n = 0
        self.best = np.inf
        self.best_perm = None

    @property
    def done(self):
        return self.n >= self.max_evals

    def __call__(self, perm):
        cost = float(self.objective(perm))
        if cost < self.best:
            self.best, self.best_perm = cost, perm.copy()
        if self.n < self.max_evals:
            self.curve[self.n] = self.best
        self.n += 1
        return cost


def random_search(objective, dim, max_evals, seed=None):
    rng = np.random.default_rng(seed)
    f = _Tracker(objective, max_evals)
    while not f.done:
        f(rng.permutation(dim) + 1)
    return f.best_perm, f.best, f.curve


def simulated_annealing(objective, dim, max_evals, seed=None):
    """SA with random segment-reversal (2-opt style) moves and geometric cooling."""
    rng = np.random.default_rng(seed)
    f = _Tracker(objective, max_evals)

    def neighbour(p):
        i, j = sorted(rng.choice(dim, size=2, replace=False))
        q = p.copy()
        q[i:j + 1] = q[i:j + 1][::-1]
        return q

    current = rng.permutation(dim) + 1
    current_cost = f(current)

    # Initial temperature from the typical size of a worsening move.
    deltas = []
    for _ in range(min(100, max_evals // 10)):
        deltas.append(abs(f(neighbour(current)) - current_cost))
    t0 = max(float(np.mean(deltas)), 1e-9)
    t_end = t0 * 1e-3
    steps = max(max_evals - f.n, 1)
    alpha = (t_end / t0) ** (1.0 / steps)

    temperature = t0
    while not f.done:
        cand = neighbour(current)
        cand_cost = f(cand)
        delta = cand_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            current, current_cost = cand, cand_cost
        temperature *= alpha
    return f.best_perm, f.best, f.curve


def genetic_algorithm(objective, dim, max_evals, seed=None, pop_size=50,
                      crossover_rate=0.9, mutation_rate=0.2, tournament=3):
    """GA: tournament selection, order crossover (OX), segment-reversal
    mutation, 1-elitism."""
    rng = np.random.default_rng(seed)
    f = _Tracker(objective, max_evals)

    def order_crossover(a, b):
        i, j = sorted(rng.choice(dim, size=2, replace=False))
        child = np.zeros(dim, dtype=int)
        child[i:j + 1] = a[i:j + 1]
        taken = set(a[i:j + 1])
        fill = [x for x in np.concatenate([b[j + 1:], b[:j + 1]]) if x not in taken]
        positions = list(range(j + 1, dim)) + list(range(0, i))
        for pos, val in zip(positions, fill):
            child[pos] = val
        return child

    def mutate(p):
        i, j = sorted(rng.choice(dim, size=2, replace=False))
        q = p.copy()
        q[i:j + 1] = q[i:j + 1][::-1]
        return q

    pop = [rng.permutation(dim) + 1 for _ in range(pop_size)]
    fit = np.array([f(p) for p in pop if not f.done] + [np.inf] * 0)
    pop = pop[:len(fit)]

    def pick():
        idx = rng.choice(len(pop), size=min(tournament, len(pop)), replace=False)
        return pop[idx[np.argmin(fit[idx])]]

    while not f.done:
        elite = int(np.argmin(fit))
        new_pop, new_fit = [pop[elite].copy()], [fit[elite]]
        while len(new_pop) < pop_size and not f.done:
            a, b = pick(), pick()
            child = order_crossover(a, b) if rng.random() < crossover_rate else a.copy()
            if rng.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
            new_fit.append(f(child))
        pop, fit = new_pop, np.array(new_fit)
    return f.best_perm, f.best, f.curve
