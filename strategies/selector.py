from .soft_besiege import SoftBesiege
from .hard_besiege import HardBesiege
from .soft_dive import SoftDive
from .hard_dive import HardDive


class StrategySelector:
    @staticmethod
    def select(r: float, E: float, dive: bool = True):
        if not dive:  # ablation: plain besiege only, no Lévy-flight dives
            r = 1.0
        if r >= 0.5 and abs(E) >= 0.5:
            return SoftBesiege()
        elif r >= 0.5 and abs(E) < 0.5:
            return HardBesiege()
        elif r < 0.5 and abs(E) >= 0.5:
            return SoftDive()
        return HardDive()
