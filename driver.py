from tqdm import tqdm
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from solver import Solver, Solution

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})


if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau

    a1 = 2
    a2 = a3 = 3

    # a1 = 4.5
    # a2 = 2
    # a3 = 3
    
    
    # a1 = 3   # Power of D
    # a2 = 4   # Power of K
    # a3 = 1   # Power of tau

    # a1 = 4   # Power of D
    # a2 = 5   # Power of K
    # a3 = 2   # Power of tau

    epsilon = 1

    etaf_guess = 0.3

    sols:List[Solution] = []
    for Q0 in tqdm(np.linspace(0.01, 0.3, 5)):
    # for Q0 in tqdm(np.logspace(-1, 1, 5)):
        solver = Solver(a1, a2, a3, Q0, epsilon)
        sols.append(solver.find_etaf(etaf_guess, method='brentq'))
    

    fig1, ax1 = plt.subplots()
    for sol in sols:
        ax1.plot(sol.eta, sol.x[0], label = "$Q_0 = {:.2e}$".format(sol.Q0))
    ax1.set_xlabel('$\\eta$')
    ax1.set_ylabel('$f(\\eta)$')
    ax1.grid()
    ax1.legend()

    t = np.linspace(1e-1,10, 1000)
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for sol in sols:
        ax2.semilogy(sol.xf(t), t, label = "$Q_0 = {:.2e}$".format(sol.Q0))
        ax3.semilogy(sol.theta(t, 0.5), t, label = "$Q_0 = {:.2e}$".format(sol.Q0))
    
    ax2.legend()
    ax2.set_ylim(t[0])
    ax2.set_xlabel("$x_f$")
    ax2.set_ylabel("$t$", rotation = 0)
    ax2.grid(which='major', linestyle='-', linewidth=0.8)
    ax2.grid(which='minor', linestyle='-', linewidth=0.25)
    
    fig2.tight_layout()

    

    # T,E = np.meshgrid(t, sols[0].eta)
    
    # interp = interp1d(sols[0].eta, sols[0].x[0], fill_value="extrapolate")
    
    # X = E*T**sols[0].omega
    # Z =T**sols[0].gamma*interp(X/T**sols[0].gamma)
    # print(Z)

    # fig, ax = plt.subplots()
    # cf = ax.contourf(X, T, Z, levels = 100)
    # ax.set_yscale('log')
    # fig.colorbar(cf, ax=ax)

    plt.show()
