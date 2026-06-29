import  numpy as np 
from context import Context

def exploration(ctx: Context):

    q = np.random.rand()
    if q < 0.5:

        rand_hawk=ctx.continuous_population[np.random.randint(len(ctx.continuous_population)), :]
        return rand_hawk -np.random.rand() * np.abs(rand_hawk - 2 * np.random.rand() * ctx.Xi)
    else :

       return ((ctx.rabbit - ctx.Xmean) - np.random.rand() * (ctx.lb + np.random.rand() * (ctx.ub - ctx.lb)))
