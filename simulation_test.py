import matplotlib.pyplot as plt
import numpy as np

def getConserved(rho, v, P, gamma, dx):
    m = rho * dx
    p = m * v 
    E = (P * dx/ (gamma - 1) + 0.5 * m * v**2)

    return m, p, E

def getPrimitive(m, p, E, gamma, dx):
    rho = m / dx
    v = p / m 
    P = (E / dx - 0.5 * rho * v**2) * (gamma - 1)

    return rho, v, P

def getGradient(f, dx):
    f_dx = np.gradient(f, dx)
    return f_dx

def slopeLimit(f, dx, f_dx):
    f_dx = np.maximum(0., np.minimum(1., (np.roll(f, -1) - f) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (f - np.roll(f, 1)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))) * f_dx
    return f_dx

def extrapolateInSpaceToFace(f, f_dx, dx):
    f_L = f - f_dx * dx / 2
    f_R = f + f_dx * dx / 2
    return f_L, f_R

def applyFluxes(F, flux_F, dx, dt):
    R = -1   # right
    L = 1    # left

    F += -dt * dx * flux_F # Flow out to the right
    F += dt * dx * np.roll(flux_F, L) # Flow in from the right

    return F

def getFlux(rho_L, rho_R, v_L, v_R, P_L, P_R, gamma):
    # left and right energies
    e_L = P_L / (gamma - 1) + 0.5 * rho_L * v_L**2
    e_R = P_R / (gamma - 1) + 0.5 * rho_R * v_R**2

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    p_star = 0.5 * (rho_L * v_L + rho_R * v_R)
    e_star = 0.5 * (e_L + e_R)
    P_star = (gamma - 1) * (e_star - 0.5 * (p_star**2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_m = p_star
    flux_p = p_star**2 / rho_star + P_star
    flux_E = (e_star + P_star) * p_star / rho_star

    # find wavespeeds
    C_L = np.sqrt(gamma * P_L / rho_L) + np.abs(v_L)
    C_R = np.sqrt(gamma * P_R / rho_R) + np.abs(v_R)
    C = np.maximum(C_L, C_R)

    flux_m -= C * 0.5 * (rho_L - rho_R)
    flux_p -= C * 0.5 * (rho_L * v_L - rho_R * v_R)
    flux_E += C * 0.5 * (e_R - e_L)

    return flux_m, flux_p, flux_E

def main():
    """ Finite Volume simulation """
    # Simulation parameters
    N = 128
    boxsize = 1.
    gamma = 5/3
    courant_fac = 0.4
    t = 0
    tEnd = 2
    tOut = 0.02
    useSlopeLimiting = False
    plotRealTime = True

    # Mesh
    dx = boxsize / N
    vol = dx
    x = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)

    # Generate Initial Conditions - different values for left and right states
    rho = 1. + (np.abs(x - 0.5) < 0.25)
    v = -0.5 + (np.abs(x - 0.5) < 0.25)
    P = 2.5 * np.ones(N)

    m, p, E = getConserved(rho, v, P, gamma, dx)

    #fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    while t < tEnd:
        rho, v, P = getPrimitive(m, p, E, gamma, dx)

        dt = courant_fac * np.min(dx / (np.sqrt(gamma * P / rho) + np.abs(v)))
        plotThisTurn = False

        if t + dt > outputCount * tOut:
            dt = outputCount * tOut - t
            plotThisTurn = True

        rho_dx = getGradient(rho, dx)
        v_dx = getGradient(v, dx)
        P_dx = getGradient(P, dx)

        if useSlopeLimiting:
            rho_dx = slopeLimit(rho, dx, rho_dx)
            v_dx = slopeLimit(v, dx, v_dx)
            P_dx = slopeLimit(P, dx, P_dx)

        rho_prime = rho - 0.5 * dt * (v * rho_dx + rho * v_dx)
        v_prime = v - 0.5 * dt * (v * v_dx + (1 / rho) * P_dx)
        P_prime = P - 0.5 * dt * (gamma * P * v_dx + v * P_dx)

        rho_L, rho_R = extrapolateInSpaceToFace(rho_prime, rho_dx, dx)
        v_L, v_R = extrapolateInSpaceToFace(v_prime, v_dx, dx)
        P_L, P_R = extrapolateInSpaceToFace(P_prime, P_dx, dx)

        flux_m, flux_p, flux_E = getFlux(rho_L, rho_R, v_L, v_R, P_L, P_R, gamma)

        m = applyFluxes(m, flux_m, dx, dt)
        p = applyFluxes(p, flux_p, dx, dt)

        t += dt

        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            plt.plot(x, rho)
            plt.ylim(0.8, 2.2)
            plt.xlabel('Position')
            plt.ylabel('Density')
            plt.pause(0.001)
            outputCount += 1

    plt.savefig('finitevolume.png', dpi=240)
    plt.show()

    return 0

if __name__ == "__main__":
    main()