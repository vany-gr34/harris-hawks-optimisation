import numpy as np

from .base import BesiegeStrategy


class SoftBesiege(BesiegeStrategy):
    def besiege(self, ctx):
        return ctx.rabbit - ctx.Xi - ctx.E * np.abs(ctx.J * ctx.rabbit - ctx.Xi)
