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
from solver import Solver, desingularized_ode, odeV1


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
    eta_transition = 1e-2
    # eta_transition = 0.2
    # eta_transition = 

    # fig, ax = plt.subplots(1,2, figsize = (14,8)).
    fig1, ax1 = plt.subplots()

    ax1.plot(sol_backward.eta, sol_backward.x[0], '-r', zorder = -1e8)
    
    n_lines = 500
    n_lines = 100
    # n_lines = 10

    # iterable = np.linspace(sol.eta[0], sol.eta[mask][-1], n_lines)
    iterable = np.logspace(-9, np.log10(eta_transition), n_lines, endpoint=False)
    # iterable = np.logspace(-5, np.log10(sol_backward.eta[mask][-1]), n_lines)[::-1]

    iterable = np.logspace(np.log10(eta_start), np.log10(sol_backward.eta_f*0.9), n_lines)[1:]
    iterable = np.logspace(np.log10(eta_start), np.log10(0.1), n_lines)[1:]
    


    cmap = cm.viridis
    norm = mcolors.LogNorm(vmin=iterable.min(), vmax=iterable.max())
    colors = cmap(norm(iterable))   # color per line




    ####################################################################
    # PROFILES SENSITIVITY TO delta eta

    f_hybrid_lines = []
    f_full_lines = []
    
    flag = True
    # for i,delta in enumerate(tqdm(iterable)):
    for i,delta in enumerate(iterable):
        # --- EARLY TERMINATION ---
        eta,x = solver.desingularized_forward_integration(sol_backward.f0, 
                                                          eta_start=delta, eta_transition=eta_transition,
                                                        #   eta_start=eta_start, eta_transition=delta,
                                                          )
        f_hybrid_lines.append(np.column_stack([eta, x[0]]))
        
        # # --- LATE TERMINATION ---
        # eta,x = solver.desingularized_forward_integration(sol_backward.f0, delta, 
        #                                                 #   eta_transition=sol_backward.eta_f/2,
        #                                                   eta_transition=1.0,
        #                                                 #   eta_transition=sol_backward.eta_f
        #                                                   )
        # f_full_lines.append(np.column_stack([eta, x[0]]))
        

        
    lc = LineCollection(f_hybrid_lines, colors=colors, zorder=1e7)
    lc.set_linewidth(1.25)
    ax1.add_collection(lc)

    
    lc = LineCollection(f_full_lines, colors=colors, zorder=1e7, linestyle = '--')
    lc.set_linewidth(1.25)
    ax1.add_collection(lc)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label("$\\delta \\eta$", rotation = 0)

    ax1.set_xlabel("$\\eta$")
    ax1.set_ylim(0,sol_backward.f0*1.2)
    ax1.set_xlim(0,0.26)
    ax1.set_ylabel("$f(\\eta)$", rotation = 0, labelpad=20)
    ax1.grid()
    fig1.tight_layout()








plt.show()