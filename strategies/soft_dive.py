import numpy as np

from .dive import DiveStrategy


class SoftDive(DiveStrategy):
    def compute_Y(self, ctx):
        return ctx.rabbit - ctx.E * np.abs(ctx.J * ctx.rabbit - ctx.Xi)
