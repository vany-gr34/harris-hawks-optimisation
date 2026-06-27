from .dive import DiveStrategy
import numpy as np
class softdive(DiveStrategy):
    def compute_Y(self, ctx):
        return ctx.rabbit -ctx.E * np.abs(ctx.J * ctx.rabbit - ctx.Xi)
    