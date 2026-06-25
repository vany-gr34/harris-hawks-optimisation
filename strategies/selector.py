from soft_besiege import softbeiege
from hard_besiege import hardbesiege
from soft_dive import softdive
from hard_dive import harddive


class StrategySelector:
    @staticmethod
    def select_strategy(r:float, E: float):
        if r >= 0.5 and abs(E) >= 0.5:
            return softbeiege()

        elif r >= 0.5 and abs(E) < 0.5:
            return hardbesiege()
        elif r < 0.5 and abs(E) >= 0.5:
            return softdive()
        else:
            return harddive()