import numpy as np 
from scipy.special import gamma
def levy_flight(dim, beta=1.5):
   
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1 / beta))
    return 0.01 * step




