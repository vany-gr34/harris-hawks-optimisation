from base import BesiegeStrategy
import numpy as np 
class softbeiege(BesiegeStrategy):
    def besiege(self, ctx):
        return ctx.rabbit - ctx.Xi -ctx.E * np.abs(ctx.J * ctx.rabbit - ctx.Xi)