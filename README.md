# Harris Hawks Optimisation (HHO) for Combinatorial Problems

A Python implementation of the Harris Hawks Optimisation algorithm adapted for discrete combinatorial problems such as the Travelling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (VRP).

---

## Overview

Harris Hawks Optimisation is a nature-inspired metaheuristic that mimics the cooperative hunting behaviour of harris hawks. The algorithm transitions between exploration (searching for prey) and exploitation (attacking prey) based on an escaping energy parameter `E`.

This implementation adapts HHO — which natively operates in continuous space — to work on permutation-based combinatorial problems through a **continuous ↔ discrete transformation layer**.

---

## How It Works

Because HHO's math requires continuous vectors, solutions are represented in two spaces simultaneously:

```
Population (permutations)
    │
    │  forward_transform        ← integers → floats
    ▼
Continuous space  ──────────────────────────────────┐
    │                                               │
    │  HHO phases (explore / besiege)               │ continuous_objective
    │                                               │ (used by dive strategies)
    ▼                                               │
New continuous vector                               │
    │  backward_transform → repair → objective ◄────┘
    ▼
Fitness (float) → update population if improved
```

The `Transformation` class maps a permutation `[1..n]` to the continuous interval `[lb, ub]` (default `[-1, 1]`) and back. The `Repair` class ensures the decoded result is always a valid permutation (no duplicates, no missing cities).

---

## Project Structure

```
harris-hawks-optimisation/
│
├── hho.py                  # Main optimiser
├── context.py              # Shared state passed to strategies
├── transformation.py       # Forward/backward transform between spaces
├── repair.py               # Fixes invalid permutations after decoding
├── population.py           # Population initialisation and tracking
├── main.py                 # Entry point — runs TSP and VRP examples
│
├── operators/
│   ├── exploration.py      # Exploration phase (|E| ≥ 1)
│   └── levy.py             # Lévy flight for dive strategies
│
├── strategies/
│   ├── base.py             # Abstract BesiegeStrategy
│   ├── selector.py         # Picks strategy based on r and E
│   ├── soft_besiege.py     # |E| ≥ 0.5, r ≥ 0.5
│   ├── hard_besiege.py     # |E| < 0.5, r ≥ 0.5
│   ├── soft_dive.py        # |E| ≥ 0.5, r < 0.5  (with Lévy flight)
│   ├── hard_dive.py        # |E| < 0.5, r < 0.5  (with Lévy flight)
│   └── dive.py             # Shared dive logic — evaluates Y vs Z vs Xi
│
└── problems/
    ├── tsp.py              # Travelling Salesman Problem
    └── vrp.py              # Capacitated Vehicle Routing Problem
```

---

## Installation

```bash
git clone https://github.com/your-username/harris-hawks-optimisation.git
cd harris-hawks-optimisation
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install numpy scipy
```

---

## Usage

```bash
python main.py
```

Expected output:

```
========================================
TSP — 5 cities
========================================
Best route: [1 2 4 3 5]
Best cost:  26.0

========================================
VRP — 6 customers, capacity=40
========================================
Best route (customer order): [5 2 6 4 1 3]
Decoded vehicle routes:
  Vehicle 1: depot → 5 → 2 → depot
  Vehicle 2: depot → 6 → 4 → 1 → depot
  Vehicle 3: depot → 3 → depot
Best cost:  115.0
```

---

## Adding Your Own Problem

Implement a class with a single `evaluate` method that takes a 1-based permutation array and returns a float:

```python
class MyProblem:
    def evaluate(self, permutation: np.ndarray) -> float:
        # permutation is a valid 1-based array, e.g. [3, 1, 4, 2]
        ...
        return total_cost
```

Then plug it into HHO:

```python
from hho import HHO
from problems.my_problem import MyProblem

problem = MyProblem()

optimizer = HHO(
    objective=problem.evaluate,
    dim=10,               # number of elements in the permutation
    population_size=50,
    max_iterations=300,
)

best_route, best_cost = optimizer.optimize()
```

No changes to the optimiser, strategies, or transformation layer are needed.

---

## HHO Parameters

| Parameter | Description | Default |
|---|---|---|
| `objective` | Your problem's evaluate function | required |
| `dim` | Number of elements in the permutation | required |
| `population_size` | Number of hawks | 20 |
| `max_iterations` | Number of iterations | 100 |
| `lb` | Lower bound of continuous space | -1 |
| `ub` | Upper bound of continuous space | 1 |

For harder instances, increasing `population_size` and `max_iterations` improves solution quality at the cost of runtime.

---

## Strategy Selection

Exploitation strategies are selected based on escaping energy `E` and a random value `r`:

| Condition | Strategy |
|---|---|
| `r ≥ 0.5` and `\|E\| ≥ 0.5` | Soft Besiege |
| `r ≥ 0.5` and `\|E\| < 0.5` | Hard Besiege |
| `r < 0.5` and `\|E\| ≥ 0.5` | Soft Dive (Lévy flight) |
| `r < 0.5` and `\|E\| < 0.5` | Hard Dive (Lévy flight) |

When `|E| ≥ 1` the algorithm is in the **exploration** phase and hawks move based on random flock positions.

---

## Reference

> Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H. (2019).
> **Harris hawks optimization: Algorithm and applications.**
> *Future Generation Computer Systems*, 97, 849–872.
> https://doi.org/10.1016/j.future.2019.02.028