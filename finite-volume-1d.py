import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_snap

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Kelvin Helmholtz Instability
In the compressible Euler equations

"""

def getConserved(rho, v, P, gamma, vol):
    """
    Calculate the conserved variable from the primitive
    rho      is matrix of cell densities
    v        is matrix of cell velocity
    P        is matrix of cell pressures
    gamma    is ideal gas gamma
    vol      is cell volume
    Mass     is matrix of mass in cells
    Momemtum is matrix of momentum in cells
    Energy   is matrix of energy in cells
    """
    Mass   = rho * vol
    Momemtum   = rho * v * vol
    Energy = (P/(gamma-1) + 0.5*rho*v**2) * vol

    return Mass, Momemtum, Energy


def getPrimitive(Mass, Momemtum, Energy, gamma, vol):
    """
    Calculate the primitive variable from the conservative
    Mass     is matrix of mass in cells
    Momemtum is matrix of momentum in cells
    Energy   is matrix of energy in cells
    gamma    is ideal gas gamma
    vol      is cell volume
    rho      is matrix of cell densities
    v        is matrix of cell velocity
    P        is matrix of cell pressures
    """
    rho = Mass / vol
    v  = Momemtum / rho / vol
    P   = (Energy/vol - 0.5*rho * v**2) * (gamma-1)

    return rho, v, P


def getGradient(f, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    """
    # directions for np.roll()
    R = -1   # right
    L = 1    # left

    f_dx = (np.roll(f,R) - np.roll(f,L)) / (2*dx)

    return f_dx


def slopeLimit(f, dx, f_dx):
    """
    Apply slope limiter to slopes
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    """
    # directions for np.roll()
    R = -1   # right
    L = 1    # left

    f_dx = np.maximum(0., np.minimum(1., ((f-np.roll(f,L))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx

    return f_dx


def extrapolateInSpaceToFace(f, f_dx, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    dx       is the cell size
    f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis
    f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis
    """
    # directions for np.roll()
    R = -1   # right
    # L = 1    # left

    f_XL = f - f_dx * dx/2
    f_XL = np.roll(f_XL,R)
    f_XR = f + f_dx * dx/2

    return f_XL, f_XR


def applyFluxes(F, flux_F_X, dx, dt):
    """
    Apply fluxes to conserved variables
    F        is a matrix of the conserved variable field
    flux_F_X is a matrix of the x-dir fluxes
    dx       is the cell size
    dt       is the timestep
    """
    # directions for np.roll()
    # R = -1   # right
    L = 1    # left

    # update solution
    F += - dt * dx * flux_F_X
    F +=   dt * dx * np.roll(flux_F_X,L)

    return F


def getFlux(rho_L, rho_R, v_L, v_R, P_L, P_R, gamma):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    v_L          is a matrix of left-state  velocity
    v_R          is a matrix of right-state velocity
    P_L          is a matrix of left-state  pressure
    P_R          is a matrix of right-state pressure
    gamma        is the ideal gas gamma
    flux_Mass    is the matrix of mass fluxes
    flux_Momemtum is the matrix of momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_L = P_L/(gamma-1)+0.5*rho_L * (v_L**2)
    en_R = P_R/(gamma-1)+0.5*rho_R * (v_R**2)

    # compute star (averaged) states
    rho_star  = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * v_L + rho_R * v_R)
    en_star   = 0.5*(en_L + en_R)

    P_star = (gamma-1)*(en_star-0.5*(momx_star**2)/rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass   = momx_star
    flux_Momemtum   = momx_star**2/rho_star + P_star
    flux_Energy = (en_star+P_star) * momx_star/rho_star

    # find wavespeeds
    C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(v_L)
    C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(v_R)
    C = np.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass   -= C * 0.5 * (rho_L - rho_R)
    flux_Momemtum -= C * 0.5 * (rho_L * v_L - rho_R * v_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momemtum, flux_Energy




def main():
    """ Finite Volume simulation """

    # Simulation parameters
    N                      = 128 # resolution
    boxsize                = 1.
    gamma                  = 1.4 # 5/3 # ideal gas gamma
    courant_fac            = 0.4
    t                      = 0
    tEnd                   = 0.2
    tOut                   = 0.01 # draw frequency
    useSlopeLimiting       = True
    plotRealTime           = False # switch on for plotting as the simulation goes along

    # Mesh
    dx = boxsize / N
    vol = dx
    # x_edges = np.linspace(0., boxsize, N+1)
    x = np.linspace(0.5*dx, boxsize-0.5*dx, N)

    # Generate Initial Conditions - different values for left and right states
    rho_L, P_L, v_L = 1., 1., 0.    # Initial state (left)
    rho_R, P_R, v_R = .125, .1, 0.  # Initial state (right)
    mask = x > 0.5 # Separates left/right regions
    rho = np.empty(N); rho[~mask] = rho_L; rho[mask] = rho_R # Density
    P = np.empty(N); P[~mask] = P_L; P[mask] = P_R # Pressure
    v = np.empty(N); v[~mask] = v_L; v[mask] = v_R # Velocity

    # Get conserved variables
    Mass, Momemtum, Energy = getConserved(rho, v, P, gamma, vol)

    # prep figure
    # fig = plt.figure(figsize=(8,6), dpi=80)
    outputCount = 1

    # if plotRealTime:
    #     plt.cla()
    #     plt.plot(x, rho)
    #     plt.xlim(0., 1.)
    #     # plt.ylim(0., 2.2)
    #     plt.pause(3)
    #     1/0

    # Simulation Main Loop
    while t < tEnd:

        # get Primitive variables
        rho, v, P = getPrimitive(Mass, Momemtum, Energy, gamma, vol)

        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (np.sqrt(gamma*P/rho) + np.sqrt(v**2)))
        plotThisTurn = False
        if t + dt > outputCount*tOut:
            dt = outputCount*tOut - t
            plotThisTurn = True

        # calculate gradients
        rho_dx = getGradient(rho, dx)
        v_dx = getGradient(v, dx)
        P_dx = getGradient(P, dx)

        # slope limit gradients
        if useSlopeLimiting:
            rho_dx = slopeLimit(rho, dx, rho_dx)
            v_dx = slopeLimit(v, dx, v_dx)
            P_dx = slopeLimit(P, dx, P_dx)

        # extrapolate half-step in time
        rho_prime = rho - 0.5*dt * (v * rho_dx + rho * v_dx)
        v_prime = v - 0.5*dt * (v * v_dx + P_dx / rho)
        P_prime = P - 0.5*dt * (gamma * P * v_dx + v * P_dx)

        # extrapolate in space to face centers
        rho_XL, rho_XR = extrapolateInSpaceToFace(rho_prime, rho_dx, dx)
        v_XL, v_XR = extrapolateInSpaceToFace(v_prime, v_dx, dx)
        P_XL, P_XR = extrapolateInSpaceToFace(P_prime, P_dx, dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momemtum_X, flux_Energy_X = getFlux(rho_XL, rho_XR, v_XL, v_XR, P_XL, P_XR, gamma)

        # update solution
        Mass = applyFluxes(Mass, flux_Mass_X, dx, dt)
        Momemtum = applyFluxes(Momemtum, flux_Momemtum_X, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, dx, dt)

        # update time
        t += dt

        # plot in real time
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plot_snap(gamma=gamma, t=t, x=x, rho=rho)
            #plt.cla()
            #plt.plot(x, rho)
            #plt.xlim(0., 1.)
            # plt.ylim(0., 2.2)
            #plt.title('t = %0.2f' % t)
            #plt.pause(0.001)
            #outputCount += 1

    # Save figure
    #plt.savefig('density.pdf',dpi=240)
    #plt.show()

    return


if __name__== "__main__":
  main()
