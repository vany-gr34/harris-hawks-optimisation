from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class Context:
    Xi: np.ndarray
    rabbit: np.ndarray
    population: np.ndarray
    Xmean: np.ndarray
    E: float
    J: float
    lb :float
    ub : float
   
    objective: Callable