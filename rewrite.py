import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_snap


# initialize variables and cells
#primative values:
    # rho = density
    # v   = velocity
    # P   = pressure
# conserved variables:
    # m   = mass
    # p   = momentum
    # E   = energy
# gamma   = ideal gas value
# dx      = cell area

def getConserved(rho, v, P, gamma, dx):
    m = rho * dx
    p = m * v 
    E = P * dx/ (gamma - 1) + 0.5 * m * v**2

    return m, p, E

def getPrimitive(m, p, E, gamma, dx):
    rho = m / dx
    v = p / m 
    P = (E / dx - 0.5 * rho * v**2) * (gamma - 1)

    return rho, v, P

def getGradient(f, dx): 
    f_dx = np.empty_like(f)
    f_dx[1:-1] = (f[2:] - f[:-2]) / (2*dx) # slope formula
    f_dx[0] = (f[1] - f[0]) / dx # slope for left boundary
    f_dx[-1] = (f[-1] - f[-2]) / dx # slope for right boundary

def slope_limit(f, dx, f_dx): 
    # applying limiter to slopes 
    # f    = array of the field 
    # dx   = cell size 
    # f_dx = array of derivatives of f in the x-direction 
    alpha = np.minimum(f - np.roll)