from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np
from scipy.integrate import solve_ivp
from solver import Solver


if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau
    
    Q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a3, Q0, epsilon)


    ###################################################################################
    # FINDING SOLUTION

    eta_f_guess = 0.2
    f_start = 1e-4
    
    
    sol = solver.find_etaf(eta_f_guess,f_start=f_start)
    print(f"Solved f0:  {sol.f0:.6f}")
    print(f"etaf:       {sol.eta_f:.6f}")
    print(f"Residual:   {sol.Res:.6e}")
    print()

    
    # fig, ax = plt.subplots(1,2, figsize = (14,8))
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax = [ax1,ax2]
    # ax[0].plot(sol.eta, sol.x[0], 'r', zorder  = 1e6)
    # ax[1].plot(sol.eta, sol.x[2], 'r', zorder  = 1e6)

    ee = np.linspace(0, sol.eta[-1], 100)
    f, fp , q = solver.evaluate_power_series(ee, sol.f0)
    ax0.plot(ee, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    ax[0].plot(ee, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    ax[1].plot(ee, q,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    
    n_lines = 500
    n_lines = 100

    mask_func = lambda eta: eta < 0.05
    mask = mask_func(sol.eta)

    # iterable = np.linspace(sol.eta[0], sol.eta[mask][-1], n_lines)
    iterable = np.logspace(-9, np.log10(sol.eta[mask][-1]), n_lines)
    iterable = np.logspace(-5, np.log10(sol.eta[mask][-1]), n_lines)

    cmap = cm.viridis
    norm = mcolors.LogNorm(vmin=iterable.min(), vmax=iterable.max())
    colors = cmap(norm(iterable))   # color per line

    
    target = 0.0015888505608369391

    f_lines = []
    q_lines = []
    
    flag = True
    for i,e in enumerate(iterable):
        eta,x = solver.forward_integration(sol.f0, e, state_space=1)
        f_lines.append(np.column_stack([eta, x[0]]))
        q_lines.append(np.column_stack([eta, x[2]]))
        if flag and x[0][-1] < 1e-2:
            flag = False
            ax0.plot(eta, x[0], color = colors[i])
    #         ax0.legend(fontsize = 12)
    
    lc = LineCollection(f_lines, colors=colors)
    lc.set_linewidth(1.25)
    ax[0].add_collection(lc)

    lc = LineCollection(q_lines, colors=colors)
    lc.set_linewidth(1.25)
    ax[1].add_collection(lc)

    # Colorbar (log scale automatically applied)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax[0])
    cbar.set_label("$\\delta \\eta$", rotation = 0)

    cbar = fig0.colorbar(sm, ax=ax0)
    cbar.set_label("$\\delta \\eta$", rotation = 0)
    cbar.remove()

    for a in [ax0,ax[0]]:
        a.set_xlabel("$\\eta$")
        a.set_ylim(0,sol.f0*1.2)
        a.set_xlim(0,0.26)
        a.set_ylabel("$f(\\eta)$", rotation = 0, labelpad=20)
        a.grid()
        a.legend(fontsize = 12)

    ax[1].set_ylabel("$q(\\eta)$", rotation = 0, labelpad=20)
    
    
    fig0.tight_layout()
    fig1.tight_layout()

    plt.show()