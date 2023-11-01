import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_snap

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Kelvin Helmholtz Instability
In the compressible Euler equations

"""

def get_conserved(rho, v, P, gamma, dx):
    """
    Calculate the conserved variable from the primitive
    rho      array of cell densities
    v        array of cell velocity
    P        array of cell pressures
    gamma    ideal gas gamma
    dx       cell volume
    m        array of mass in cells
    p        array of momentum in cells
    E        array of energy in cells
    """
    m = rho * dx
    p = m * v
    E = P*dx/(gamma-1) + 0.5*m*v**2
    return m, p, E


def get_primitive(m, p, E, gamma, dx):
    """
    Calculate the primitive variable from the conservative
    m        array of mass in cells
    p        array of momentum in cells
    E        array of energy in cells
    gamma    ideal gas gamma
    dx       cell volume
    rho      array of cell densities
    v        array of cell velocity
    P        array of cell pressures
    """
    rho = m / dx
    v = p / m
    P = (E/dx - 0.5*rho*v**2) * (gamma-1)
    return rho, v, P


def get_gradient(f, dx):
    """
    Calculate the gradients of a field
    f        array of the field
    dx       cell size
    f_dx     return array of derivative of f in the x-direction
    """
    return (np.roll(f,-1) - np.roll(f,1)) / (2*dx)


def slope_limit(f, dx, f_dx):
    """
    Apply slope limiter to slopes
    f        array of the field
    dx       cell size
    f_dx     array of derivative of f in the x-direction
    """
    alpha = np.minimum(f - np.roll(f,1), np.roll(f,-1) - f) / (f_dx*dx + 1e-8) # min(f_i - f_i-1, f_i+1 - f_i) / (f_dx*dx)
    f_dx = np.maximum(0., np.minimum(1., alpha)) * f_dx # Limit alpha in [0,1]
    return f_dx


def extrapolate_to_face(f, f_dx, dx):
    """
    Calculate the gradients of a field
    f        array of the field
    f_dx     array of the field x-derivatives
    dx       cell size
    f_L      array of spatial-extrapolated values on `left' face along x-axis
    f_R      array of spatial-extrapolated values on `right' face along x-axis
    """
    f_L = f - f_dx * dx/2
    f_R = f + f_dx * dx/2
    f_L = np.roll(f_L,-1) # Why is this here?
    return f_L, f_R


def apply_fluxes(F, flux_F, dx, dt):
    """
    Apply fluxes to conserved variables
    F        array of the conserved variable field
    flux_F   array of the x-dir fluxes
    dx       cell size
    dt       timestep
    """
    # Update solution: note the face grid uses right faces
    F -= dt * dx * flux_F # Flow out to the right
    F += dt * dx * np.roll(flux_F,1) # Flow in from the right
    return F


def get_flux(rho_L, rho_R, v_L, v_R, P_L, P_R, gamma):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L    array of left-state  density
    rho_R    array of right-state density
    v_L      array of left-state  velocity
    v_R      array of right-state velocity
    P_L      array of left-state  pressure
    P_R      array of right-state pressure
    gamma    ideal gas gamma
    flux_m   array of mass fluxes
    flux_p   array of momentum fluxes
    flux_E   array of energy fluxes
    """
    # left and right energies
    e_L = P_L/(gamma-1) + 0.5*rho_L*v_L**2
    e_R = P_R/(gamma-1) + 0.5*rho_R*v_R**2

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    p_star = 0.5 * (rho_L * v_L + rho_R * v_R)
    e_star = 0.5 * (e_L + e_R)
    P_star = (gamma-1)*(e_star - 0.5*p_star**2/rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_m = p_star
    flux_p = p_star**2/rho_star + P_star # Why is this here?
    flux_E = (e_star + P_star) * p_star / rho_star

    # find wavespeeds
    C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(v_L)
    C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(v_R)
    C = np.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_m -= C * 0.5 * (rho_L - rho_R)
    flux_p -= C * 0.5 * (rho_L * v_L - rho_R * v_R)
    flux_E -= C * 0.5 * (e_L - e_R)

    return flux_m, flux_p, flux_E


def main():
    """ Finite Volume simulation """

    # Simulation parameters
    useSlopeLimiting       = True
    plotRealTime           = False # switch on for plotting as the simulation goes along
    N                      = 128 # resolution
    boxsize                = 1.
    gamma                  = 1.4 # 5/3 # ideal gas gamma
    courant_fac            = 0.4
    t                      = 0
    tEnd                   = 0.02
    t_out                   = 0.01 if plotRealTime else tEnd # plot frequency

    # Mesh
    dx = boxsize / N
    dx = dx
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
    m, p, E = get_conserved(rho, v, P, gamma, dx)

    # prep figure
    # fig = plt.figure(figsize=(8,6), dpi=80)
    output_count = 1

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
        rho, v, P = get_primitive(m, p, E, gamma, dx)

        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (np.sqrt(gamma*P/rho) + np.sqrt(v**2)))
        plotThisTurn = False
        if t + dt > output_count * t_out:
            dt = output_count * t_out - t
            plotThisTurn = True

        # calculate gradients
        rho_dx = get_gradient(rho, dx)
        v_dx = get_gradient(v, dx)
        P_dx = get_gradient(P, dx)

        # slope limit gradients
        if useSlopeLimiting:
            rho_dx = slope_limit(rho, dx, rho_dx)
            v_dx = slope_limit(v, dx, v_dx)
            P_dx = slope_limit(P, dx, P_dx)

        # extrapolate half-step in time
        rho_prime = rho - 0.5*dt * (v * rho_dx + rho * v_dx)
        v_prime = v - 0.5*dt * (v * v_dx + P_dx / rho)
        P_prime = P - 0.5*dt * (gamma * P * v_dx + v * P_dx)

        # extrapolate in space to face centers
        rho_L, rho_R = extrapolate_to_face(rho_prime, rho_dx, dx)
        v_L, v_R = extrapolate_to_face(v_prime, v_dx, dx)
        P_L, P_R = extrapolate_to_face(P_prime, P_dx, dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_m, flux_p, flux_E = get_flux(rho_L, rho_R, v_L, v_R, P_L, P_R, gamma)

        # update solution
        m = apply_fluxes(m, flux_m, dx, dt)
        p = apply_fluxes(p, flux_p, dx, dt)
        E = apply_fluxes(E, flux_E, dx, dt)

        # update time
        t += dt
        print(f't = {t}')

        # plot in real time
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            print(x)
            print(rho)
            plot_snap(gamma=gamma, t=t, x=x, rho=rho, p=P, u=v, num=output_count)
            #plt.cla()
            #plt.plot(x, rho)
            #plt.xlim(0., 1.)
            # plt.ylim(0., 2.2)
            #plt.title('t = %0.2f' % t)
            #plt.pause(0.001)
            output_count += 1

    # Save figure
    #plt.savefig('density.pdf',dpi=240)
    #plt.show()

    return


if __name__== "__main__":
  main()
