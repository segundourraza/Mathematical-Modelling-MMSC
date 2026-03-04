from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

from solver import Solver

def driver(a1, a2, a3, etaf_guess, f_start = 1e-4):
    solver = Solver(a1, a2, a3, Q0, epsilon)
    return solver.find_etaf(etaf_guess,f_start=f_start, state_space=1, fp_condition=10)
    
if __name__ == '__main__':

    
    epsilon = 1
    Q0 = 0.1

    param_set = [
                 [2, 1/4,   3/4],
                 [4, 1/6,   5/6],
                 [6, 1/8,   7/8],
                 [8, 11/10, 9/10]
                 ]


    etaf_guess = 0.3
    fstart = 1e-2

    fig, ax = plt.subplots(1,2, figsize=(14, 8), layout = 'constrained')
    
    fig, ax = plt.subplots(1,1)
    for params in tqdm(param_set):
        a1, gamma, omega = params
        a2 = a3 = a1 + 1
        sol = driver(a1, a2, a3, etaf_guess=etaf_guess, f_start = fstart)
        print(sol.a1, sol.a2, sol.a3, sol.eta_f, sol.x[0][0])
        ax.plot(sol.eta, sol.x[0])
        ax.text(0, sol.x[0][0], "$f(0) = {:.4f}$".format(sol.x[0][0]))
        ax.text(sol.eta[-1], fstart, "$\\eta_f = {:.4f}$".format(sol.eta_f))

    ax.set_ylabel("$f(\\eta)$")
    ax.set_ylabel("$q(\\eta)$")
    ax.grid()
    ax.set_xlabel("$\\eta$")

    
    plt.show()