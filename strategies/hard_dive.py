from .dive import  DiveStrategy
import numpy as np
class harddive(DiveStrategy):
    def compute_Y(self, ctx):
        return ctx.rabbit -ctx.E * np.abs(ctx.J * ctx.rabbit - ctx.Xmean)