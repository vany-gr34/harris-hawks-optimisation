import numpy as np

from .base import BesiegeStrategy


class HardBesiege(BesiegeStrategy):
    def besiege(self, ctx):
        return ctx.rabbit - ctx.E * np.abs(ctx.rabbit - ctx.Xi)
