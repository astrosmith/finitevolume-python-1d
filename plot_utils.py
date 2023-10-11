###############
### plot.py ###
###############

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sod

def plot_snap(gamma=1.4, t=0.2, x=None, rho=None):
    plotname = 'sod.pdf'

    # left_state and right_state set p, rho and u
    # geometry = (left boundary, right boundary, initial position of the shock
    pos, reg, d = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0),
                            geometry=(0, 1, 0.5), t=t, gamma=gamma)
    # Printing positions
    print('\nPositions:')
    for desc, vals in pos.items():
        print('{0:10} : {1}'.format(desc, vals))

    # Printing p, rho and u for regions
    print('\nRegions:')
    for region, vals in sorted(reg.items()):
        print('{0:10} : {1}'.format(region, vals))

    # Create energy and temperature
    d['e'] = d['p']/d['rho']/(gamma-1.)
    d['E'] = d['p']/(gamma-1.) + 0.5*d['u']**2
    d['T'] = d['p']/d['rho']
    d['M'] = d['u']/np.sqrt(d['p']/d['rho'])     # Mach number: u/c

    ## Create pyplot figure and axes objects ##
    fig = plt.figure(figsize=(5,10), dpi=300)
    ax1 = fig.add_axes([0,0.75,1,0.25])
    ax2 = fig.add_axes([0,0.5 ,1,0.25])
    ax3 = fig.add_axes([0,0.25,1,0.25])
    ax4 = fig.add_axes([0,   0,1,0.25])
    axs = [ax1, ax2, ax3, ax4]
    yas = [d['rho'], d['p'], d['u'], d['e']]

    ## Plot data ##
    for ax,ya in zip(axs,yas):
        ax.plot(d['x'], ya, lw=0.5)
    if x is not None: 
        ax1.scatter(x, rho, marker='.', s=10, edgecolor='none', facecolor=[0.5, 0.5, 0.5])

    ## Plot labels ##
    ax4.set_xlabel(r'$x\;{\rm [cm]}$', fontsize=15)
    ax1.set_ylabel(r'$\rm{Density\;[g/cm^3]}$', fontsize=15)
    ax2.set_ylabel(r'$\rm{Pressure\;[erg/cm^3]}$', fontsize=15)
    ax3.set_ylabel(r'$\rm{Velocity\;[cm/s]}$', fontsize=15)
    ax4.set_ylabel(r'$\rm{Specific\,Int.\,Energy\;[ergs/g]}$', fontsize=15)
    t_str = r'$t\,=\,%g\,\rm{s}$' % t
    ax1.text(0.75, 0.75, t_str, fontsize=15, transform=ax1.transAxes)

    ## LaTeX formatting for x and y ticks ##
    for ax,y in zip(axs,yas):
        ax.set_xlim([0, 1])
        xlocs = np.linspace(0, 1, 6)
        ax.set_xticks(xlocs);
        ymin, ymax = np.min(y), np.max(y)
        dy = 0.101*(ymax - ymin)
        ax.set_ylim([ymin-dy, ymax+dy])
        ylocs = ax.get_yticks()
        ax.set_yticklabels([r"$%g$" %val for val in ylocs], fontsize=12)
        ax.minorticks_on()
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4.set_xticklabels([r"$%g$" % val for val in xlocs], fontsize=12)

    ## Save the figure ##
    print("Saving plot as '"+plotname+"'")
    fig.savefig(plotname, bbox_inches="tight")

    ## Close the figure so we don't leak memory ##
    plt.close(fig)
## end plot_snap(base='sod', num=0) ##

plot_snap()
