from typing import List

from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

from solver import Solver, Solution
    
if __name__ == '__main__':

    epsilon = 1                 
    params = [2, 1/4,   3/4]

    etaf_guess = 0.3
    fstart = 1e-2

    iterable = np.linspace(0.1, 1, 50)
    
    sols:List[Solution] = []
    igs = []
    res = []
    fs = []
    for Q0 in ([0.01, 0.1, 0.2, 0.5]):
        a1, gamma, omega = params
        a2 = a3 = a1 + 1
        print(a1, a2, a3)
        solver = Solver(a1, a2, a3, Q0, epsilon)
        sols.append(solver.find_etaf(etaf_guess,f_start=fstart, state_space=1, fp_condition=0.1))
        igs.append(solver.backward_integrator(etaf_guess, f_start=fstart))
        
        
        ee = np.linspace(0, sols[-1].eta_f, 100)
        fs.append((ee,solver.evaluate_power_series(ee, sols[-1].f0)[0]))
            
        _temp = []
        for e in tqdm(iterable,position=1, leave=False):
            _e, _x = solver.backward_integrator(e, f_start=fstart)
            _temp.append(solver._check_integral_condition(_e,_x))
        res.append(_temp)

    fig1, ax1 = plt.subplots()
    for sol,ig,(ee,f) in zip(sols, igs, fs):
        l, = ax1.plot(sol.eta, sol.x[0], label = "$Q_0 = {:.2f}$".format(sol.Q0))
        ax1.plot(ig[0], ig[1][0], '--', color = l.get_color(), linewidth = l.get_linewidth()*0.5)
        
        ax1.plot(ee, f,   '-.',  color = 'k', zorder = 1e7*2, linewidth = 1.25)
        
    ax1.set_xlabel('$\\eta$')
    ax1.set_ylabel('$f(\\eta)$')
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()


    fig2, ax2 = plt.subplots()
    for r,sol in zip(res, sols):
        l, = ax2.plot(iterable, r, '-')
        ax2.plot(sol.eta_f, sol.Res, 'x', color = l.get_color(), ms = 10, markeredgewidth = 2)
    ax2.set_xlabel('$\\eta_f$')
    ax2.set_ylabel('Residual')
    ax2.grid()
    # ax2.legend()
    fig2.tight_layout()

    plt.show()