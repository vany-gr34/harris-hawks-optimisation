import numpy as np


class Repair:
    """Turns an array with duplicates / missing labels into a valid 1-based
    permutation, deterministically.

    Scanning left to right, the first occurrence of a label is kept. Each later
    duplicate is replaced by the *closest* label that is still missing, so the
    repaired solution stays near the continuous point it was decoded from.
    """

    def repair(self, permutation):
        perm = np.clip(np.asarray(permutation, dtype=int), 1, len(permutation)).copy()
        n = len(perm)

        seen = np.zeros(n + 1, dtype=bool)
        duplicate_positions = []
        for i in range(n):
            v = perm[i]
            if seen[v]:
                duplicate_positions.append(i)
            else:
                seen[v] = True

        if not duplicate_positions:
            return perm

        missing = [v for v in range(1, n + 1) if not seen[v]]
        for i in duplicate_positions:
            v = perm[i]
            j = min(range(len(missing)), key=lambda k: (abs(missing[k] - v), missing[k]))
            perm[i] = missing.pop(j)
        return perm
