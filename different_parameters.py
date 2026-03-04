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
    return solver.find_etaf(etaf_guess,f_start=f_start)

def parametric_study(a1, a2, a3, eta_f_guess, f_start = 1e-4):
    solver = Solver(a1, a2, a3, Q0, epsilon)
    eta_data = []
    x_data = []
    res_data = []
    for efg in tqdm(eta_f_guess):
        _eta, _x = solver.backward_integrator(efg, f_start=f_start)
        res_data.append(solver._check_integral_condition(_eta, _x))
        eta_data.append(_eta)
        x_data.append(_x)
    
    sol = solver.find_etaf(eta_f_guess[0], f_start=f_start)
    return sol
    
if __name__ == '__main__':

    
    epsilon = 0.1
    Q0 = 0.1

    param_sets = [[2, 3, 0.8],
                  [3, 4, 1  ],
                #   [4, 5, 2]
                  ]
    
    etaf_guess = 0.2
    fstart = 1e-4

    fig, ax = plt.subplots(1,2, figsize=(14, 8), layout = 'constrained')
    for params in param_sets:
        sol = driver(*params, etaf_guess=etaf_guess, f_start = fstart)
        print(sol.a1, sol.a2, sol.a3, sol.eta_f, sol.x[0][0])
        ax[0].plot(sol.eta, sol.x[0])
        ax[1].plot(sol.eta, sol.x[2], label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(*params))
        ax[0].text(0, sol.x[0][0], "$f(0) = {:.4f}$".format(sol.x[0][0]))
        ax[0].text(sol.eta[-1], fstart, "$\\eta_f = {:.4f}$".format(sol.eta_f))

    ax[0].set_ylabel("$f(\\eta)$")
    ax[1].set_ylabel("$q(\\eta)$")
    for a in ax:
        a.grid()
        a.set_xlabel("$\\eta$")

    
    ###################################################################################
    # PARAMETRIC SEARCH STUDY

    


    # eta_f_guess = np.linspace(0.1, 1, 100)
    # f_start = 1e-4
    # fig, ax =  plt.subplots()
    # for params in param_sets:
    #     (eta_f, (sol_eta, sol_x ), res), (eta_data, x_data, res_data) = parametric_study(*params, eta_f_guess, f_start)
    #     ax.plot(eta_f_guess, res_data)
    #     ax.plot(eta_f, res, 'xr')

    
    # ax.set_xlabel("$\\eta_f$")
    # ax.set_ylabel("Residual")
    # ax.grid()


    plt.show()