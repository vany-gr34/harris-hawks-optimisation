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

        Y = np.clip(Y, ctx.lb, ctx.ub)
        Z = np.clip(Z, ctx.lb, ctx.ub)
 
        f_Xi = ctx.continuous_objective(ctx.Xi)
        f_Y  = ctx.continuous_objective(Y)
        f_Z  = ctx.continuous_objective(Z)
 
        if f_Y < f_Xi and f_Y <= f_Z:
            return Y
        elif f_Z < f_Xi:
            return Z
        else:
            return ctx.Xi 
























        #fx=ctx.objective(ctx.Xi)
        #fz=ctx.objective(Z)
        #fy=ctx.objective(Y)
        #if fy<fx :
        #    return Y
        #elif fz<fx:
        #    return Z
        #else:
         #   return ctx.Xi
  