from .base import BesiegeStrategy
import numpy as np
class hardbesiege(BesiegeStrategy):
    def besiege(self, ctx):
        return ctx.rabbit -ctx.E * np.abs(ctx.rabbit - ctx.Xi)