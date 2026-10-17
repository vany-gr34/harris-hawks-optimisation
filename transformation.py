import numpy as np


class Transformation:
    """Maps between 1-based permutations and the continuous box [lb, ub]^n.

    forward_transform : label x in {1..n}  ->  lb + (ub - lb) * (x - 1) / (n - 1)
    backward_transform: continuous value   ->  nearest label (``decoding="round"``)
                        or rank of the value (``decoding="rank"``)

    ``decoding="round"`` can produce duplicates, so the result must go through
    ``Repair``. ``decoding="rank"`` (Smallest Position Value rule) always yields
    a valid permutation and needs no repair.
    """

    def __init__(self, lb=-1, ub=1, decoding="round"):
        if decoding not in ("round", "rank"):
            raise ValueError("decoding must be 'round' or 'rank'")
        self.lb = lb
        self.ub = ub
        self.decoding = decoding

    def forward_transform(self, permutation):
        permutation = np.asarray(permutation, dtype=float)
        n = len(permutation)
        if n == 1:
            return np.array([float(self.lb)])
        return self.lb + (self.ub - self.lb) * (permutation - 1) / (n - 1)

    def backward_transform(self, z):
        z = np.asarray(z, dtype=float)
        n = len(z)
        if self.decoding == "rank":
            return np.argsort(np.argsort(z, kind="stable"), kind="stable") + 1
        if n == 1:
            return np.array([1])
        x = np.rint((z - self.lb) / (self.ub - self.lb) * (n - 1)).astype(int) + 1
        return np.clip(x, 1, n)

    def clip(self, z):
        return np.clip(z, self.lb, self.ub)
