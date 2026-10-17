"""Minimal TSPLIB / CVRPLIB readers (EUC_2D instances only)."""

import re

import numpy as np

from .tsp import TSPProblem
from .vrp import VRPProblem


def _read(path):
    header, coords, demands = {}, [], {}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            if line.endswith("SECTION") or re.match(r"^[A-Z_]+_SECTION", line):
                section = line.split()[0]
                continue
            if section is None or ":" in line:
                key, _, value = line.partition(":")
                header[key.strip()] = value.strip()
                if ":" in line:
                    section = None
                continue
            parts = line.split()
            if section == "NODE_COORD_SECTION":
                coords.append((float(parts[1]), float(parts[2])))
            elif section == "DEMAND_SECTION":
                demands[int(parts[0])] = int(parts[1])
    return header, np.array(coords), demands


def _euc2d(coords):
    """TSPLIB EUC_2D: Euclidean distance rounded to the nearest integer."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.floor(np.sqrt((diff ** 2).sum(-1)) + 0.5)


def load_tsp(path):
    header, coords, _ = _read(path)
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("only EUC_2D instances are supported")
    problem = TSPProblem(_euc2d(coords))
    problem.name = header.get("NAME", path)
    return problem


def load_cvrp(path):
    header, coords, demands = _read(path)
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("only EUC_2D instances are supported")
    n = len(coords)
    demand_vec = [demands[i + 1] for i in range(n)]  # node 1 is the depot
    problem = VRPProblem(_euc2d(coords), demand_vec, int(header["CAPACITY"]))
    problem.name = header.get("NAME", path)
    return problem
