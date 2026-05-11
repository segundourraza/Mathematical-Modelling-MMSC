import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    
    
    sol_backward = solver.find_etaf(eta_f_guess,f_start=f_start)
    print(f"Solved f0:  {sol_backward.f0:.6f}")
    print(f"etaf:       {sol_backward.eta_f:.6f}")
    print(f"Residual:   {sol_backward.Res:.6e}")
    print()


    f0_guess = sol_backward.f0
    f0_guess = (0.1, 10)
    eta_start = 1e-6
    eta_transition = 1e-5
    # # eta_transition = 1e-1
    # # eta_transition = 1
    # sol_forward = solver.find_f0_desingularized(f0_guess, eta_start, eta_transition, method='brentq')
    # print(f"Solved f0:  {sol_forward.f0:.6f}")
    # print(f"etaf:       {sol_forward.eta_f:.6f}")
    # print(f"Residual:   {sol_forward.Res:.6e}")
    # print()

    # fig, ax = plt.subplots(1,2, figsize = (14,8)).
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()


    
    ee = np.linspace(0, sol_backward.eta[-1], 100)
    f, fp , q = solver.evaluate_power_series(ee, sol_backward.f0)
    for a in [ax1,ax2]:
        a.plot(sol_backward.eta, sol_backward.x[0], '-r', zorder = -1e8,label = "$f(\\eta)$")
        # a.plot(sol_forward.eta, sol_forward.x[0], '-b')
        a.plot(ee, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    
    
    n_lines = 500
    n_lines = 100
    n_lines = 10

    mask_func = lambda eta: eta < 0.05
    mask = mask_func(sol_backward.eta)

    # iterable = np.linspace(sol.eta[0], sol.eta[mask][-1], n_lines)
    iterable = np.logspace(-9, np.log10(sol_backward.eta[mask][-1]), n_lines)
    iterable = np.logspace(-5, np.log10(sol_backward.eta[mask][-1]), n_lines)

    cmap = cm.viridis
    norm = mcolors.LogNorm(vmin=iterable.min(), vmax=iterable.max())
    colors = cmap(norm(iterable))   # color per line

    f_lines_old = []
    q_lines_old = []
    
    f_lines_new = []
    q_lines_new = []
    
    flag = True
    for i,delta in enumerate(tqdm(iterable)):
        # --- POWER SERIES APPROACH ---
        eta,x = solver.forward_integration(sol_backward.f0, delta, state_space=1)
        f_lines_old.append(np.column_stack([eta, x[0]]))
        q_lines_old.append(np.column_stack([eta, x[2]]))
    
        # --- TAYLOR SERIES APPROACH ---
        eta,x = solver.desingularized_forward_integration(sol_backward.f0, delta, 
                                                        #   eta_transition=sol_backward.eta_f/2,
                                                          eta_transition=eta_transition,
                                                        #   eta_transition=sol_backward.eta_f
                                                          )
        f_lines_new.append(np.column_stack([eta, x[0]]))
        q_lines_new.append(np.column_stack([eta, x[2]]))
        
    lc = LineCollection(f_lines_old, colors=colors, zorder=1e7)
    lc.set_linewidth(1.25)
    ax1.add_collection(lc)
    
    lc = LineCollection(f_lines_new, colors=colors, zorder=1e7)
    lc.set_linewidth(1.25)
    ax2.add_collection(lc)

    # Colorbar (log scale automatically applied)
    for a in [ax1,ax2]:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig1.colorbar(sm, ax=a)
        cbar.set_label("$\\delta \\eta$", rotation = 0)

        a.set_xlabel("$\\eta$")
        a.set_ylim(0,sol_backward.f0*1.2)
        a.set_xlim(0,0.26)
        a.set_ylabel("$f(\\eta)$", rotation = 0, labelpad=20)
        a.grid()
    leg = ax1.legend(loc = 'lower left')
    leg.set_zorder(1e10)
    ax1.set_box_aspect(1)  # force square axes regardless of figsize
    fig1.savefig(f'forward_instability.pdf', bbox_inches='tight', pad_inches=0.02)
    
    fig1.tight_layout()

    plt.show()