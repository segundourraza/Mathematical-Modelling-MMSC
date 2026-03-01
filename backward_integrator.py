from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np

from solver import Solver



if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau

    
    # a1 = 1.8   # Power of D
    # a2 = 3   # Power of K
    # a3 = 0.8 # Power of tau
    
    epsilon = 0.1
    Q0 = 0.1


    solver = Solver(a1, a2, a3, Q0, epsilon)


    ###################################################################################
    # FINDING SOLUTION

    eta_f_guess = 0.1
    f_start = 1e-7
    sol1_eta, sol1_x = solver.backward_integrator(eta_f_guess, f_start=f_start)
    sol1_dx = solver._ode(sol1_eta, sol1_x)
    sol1_f0 = sol1_x[0][0]
    sol1_etaf = sol1_eta[-1]
    Res1 = solver._check_integral_condition(sol1_eta, sol1_x)
    print(f"Solved f0:  {sol1_f0:.6f}")
    print(f"etaf:       {sol1_etaf:.6f}")
    print(f"Residual:   {Res1:.6e}")
    print()

    
    sol2_etaf, (sol2_eta, sol2_x) = solver.find_etaf(eta_f_guess,f_start=f_start)
    sol2_dx = solver._ode(sol2_eta, sol2_x)
    sol2_f0 = sol2_x[0][0]
    Res2 = solver._check_integral_condition(sol2_eta, sol2_x)
    print(f"Solved f0:  {sol2_f0:.6f}")
    print(f"etaf:       {sol2_etaf:.6f}")
    print(f"Residual:   {Res2:.6e}")
    print()


    fig, ax = plt.subplots(2,3, figsize=(14, 8))
    for i in range(3):
        ax[0][i].plot(sol1_eta, sol1_x[i], label = "$\\eta = {:.4f}$".format(sol1_etaf))
        ax[1][i].semilogy(sol1_eta, abs(sol1_dx[i]))
        
        ax[0][i].plot(sol2_eta, sol2_x[i], label = "$\\eta = {:.4f}$".format(sol2_etaf))
        ax[1][i].semilogy(sol2_eta, abs(sol2_dx[i]))

    # # for line in ax[0][1].get_lines():
    # #     y = line.get_ydata()
    # #     line.set_ydata(abs(y))
    # ax[0][1].set_yscale('log')
    ax[0][0].set_xlim(0)
    ax[0][0].set_ylim(0)


    ax[0][0].set_ylabel("$f(\\eta)$")
    ax[0][1].set_ylabel("$f^\\prime(\\eta)$")
    ax[0][2].set_ylabel("$q(\\eta)$")
    ax[1][0].set_ylabel("$|f^\\prime(\\eta)|$")
    ax[1][1].set_ylabel("$|f^{\\prime\\prime}(\\eta)|$")
    ax[1][2].set_ylabel("$|q^\\prime(\\eta)|$")
    ax[0][1].legend()
    for a in ax.flatten():
        a.grid()
        a.set_xlabel("$\\eta$")

    fig.tight_layout()


    ###################################################################################
    # PARAMETRIC SEARCH STUDY

    


    eta_f_guess = np.linspace(0.1, 1, 100)
    f_start = 1e-4
    eta_data = []
    x_data = []
    res_data = []
    fig, ax = plt.subplots(1,2)
    for efg in tqdm(eta_f_guess):
        _eta, _x = solver.backward_integrator(efg, f_start=f_start)
        ax[0].plot(_eta, _x[0], linewidth = 0.25)
        res_data.append(solver._check_integral_condition(_eta, _x))
        eta_data.append(_eta)
        x_data.append(_x)
    ax[1].plot(eta_f_guess, res_data)

    ax[0].plot(sol2_eta, sol2_x[0], '-r')
    ax[1].plot(sol2_etaf, Res2, 'xr')

    
    ax[0].set_ylabel("$f(\\eta)$")
    ax[0].set_xlabel("$\\eta$")
    ax[1].set_xlabel("$f_0$")
    ax[1].set_ylabel("Residual")
    ax[0].grid()
    ax[1].grid()


    plt.show()