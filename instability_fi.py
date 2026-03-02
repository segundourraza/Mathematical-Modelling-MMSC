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

    eta_f_guess = 0.1
    f_start = 1e-4
    
    
    sol_etaf, (sol_eta, sol_x) = solver.find_etaf(eta_f_guess,f_start=f_start)
    sol_f0 = sol_x[0][0]
    Res = solver._check_integral_condition(sol_eta, sol_x)
    print(f"Solved f0:  {sol_f0:.6f}")
    print(f"etaf:       {sol_etaf:.6f}")
    print(f"Residual:   {Res:.6e}")
    print()

    
    fig, ax = plt.subplots(1,2, figsize = (14,8))
    ax[0].plot(sol_eta, sol_x[0], 'r', zorder  = 1e6)
    ax[1].plot(sol_eta, sol_x[2], 'r', zorder  = 1e6)

    ee = np.linspace(0, sol_eta[-1], 100)
    f, fp , q = solver.evaluate_power_series(ee, sol_f0)
    ax[0].plot(ee, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    ax[1].plot(ee, q,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    
    n_lines = 500
    n_lines = 100

    mask_func = lambda eta: eta < 0.05
    mask = mask_func(sol_eta)

    iterable = np.linspace(sol_eta[0], sol_eta[mask][-1], n_lines)
    iterable = np.logspace(-9, np.log10(sol_eta[mask][-1]), n_lines)

    f_lines = []
    q_lines = []
    for e in tqdm(iterable):
        eta,x = solver.forward_integration(sol_f0,e)
        mask = mask_func(eta)
        f_lines.append(np.column_stack([eta, x[0]]))
        q_lines.append(np.column_stack([eta, x[2]]))
    
    
    cmap = cm.viridis
    norm = mcolors.LogNorm(vmin=iterable.min(), vmax=iterable.max())
    colors = cmap(norm(iterable))   # color per line
    
    lc = LineCollection(f_lines, colors=colors)
    lc.set_linewidth(1.25)
    ax[0].add_collection(lc)

    lc = LineCollection(q_lines, colors=colors)
    lc.set_linewidth(1.25)
    ax[1].add_collection(lc)

    for a in ax:
        a.set_xlabel("$\\eta$")
        a.grid()
    
    ax[0].set_ylim(0,)
    ax[0].set_xlim(0,)    
    ax[0].set_ylabel("$f(\\eta)$", rotation = 0, labelpad=20)

    ax[1].set_ylabel("$q(\\eta)$", rotation = 0, labelpad=20)
    
    
    # Colorbar (log scale automatically applied)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.set_label("$\\Delta \\eta$", rotation = 0)
    
    fig.tight_layout()

    plt.show()