from .base import BesiegeStrategy
from abc import ABC, abstractmethod
import numpy as np
from operators.levy import levy_flight
class DiveStrategy(BesiegeStrategy, ABC):
    @abstractmethod
    def compute_Y(self, ctx):
        pass


    def besiege(self, ctx):
        Y = self.compute_Y(ctx)
        Z=(Y + np.random.rand(len(ctx.Xi))*levy_flight(len(ctx.Xi)))
        fx=ctx.objective(ctx.Xi)
        fz=ctx.objective(Z)
        fy=ctx.objective(Y)

        if fy<fx :
            return Y
        elif fz<fx:
            return Z
        else:
            return ctx.Xi
        