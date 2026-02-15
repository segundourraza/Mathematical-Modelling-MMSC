import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

from solver import Solver


if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a3, q0, epsilon)

    f0_guess = 0.1
    # f0_guess = 0.6
    eta0 = 1e-2
    eta0 = 1e-6

    
    sol_eta, sol_x = solver.solve(f0=f0_guess, eta0 = eta0)
    Res = solver._check_integral_condition(sol_eta, sol_x)
    print(f"Guessed f0: {f0_guess:.4f}")
    print(f"Residual:   {Res:.4e}")

    f0, (sol_eta, sol_x) = solver.find_f0(f0_guess)
    Res = solver._check_integral_condition(sol_eta, sol_x)
    print(f"Solved f0:  {f0:.4f}")
    print(f"Residual:   {Res:.4e}")
    
    # inverted_sol = solver.inverted_solve(sol_actual)
    # print(sol_actual.y[:,-1])
    # print(inverted_sol.y[:,-1])
    ################################################################
    # PLOTTING

    
    fig, ax = plt.subplots(1,2)
    
    # # PLOT [f, g, q] solver    
    # eta = sol_guessed.t
    # f, g, q = sol_guessed.y
    # fp = (solver.gamma*f - g/(f**(solver.a3)))/(solver.omega*eta)
    # ax[0].plot(eta, f, '--',  label =f"Guessed f(0) = {f0_guess:.4f}")
    # ax[1].plot(eta, q, '--',  label =f"Guessed f(0) = {f0_guess:.4f}")
    
    # ACTUAL PROFILE
    f, g, q = sol_x
    fp = (solver.gamma*f - g/(f**(solver.a3)))/(solver.omega*sol_eta)
    ax[0].plot(sol_eta, f, '-',  label =f"Actual f(0) = {f0:.4f}")
    ax[1].plot(sol_eta, q, '-',  label =f"Actual f(0) = {f0:.4f}")
    
    # PLOT POWER SERIES
    f, fp , q = solver.evaluate_power_series(sol_eta, f0)
    ax[0].plot(sol_eta, f, '-.', label = "Power series")
    ax[1].plot(sol_eta, q, '-.', label = "Power series")

    ax[0].set_xlim(0, sol_eta[-1]*1.05)
    ax[1].set_xlim(0, sol_eta[-1]*1.05)


    
    ax[0].legend()
    [_.set_xlabel('$\\eta$') for _ in ax]
    [_.grid() for _ in ax]
    ax[0].set_ylabel('f($\\eta$)', rotation = 0, labelpad = 20)
    ax[1].set_ylabel('q($\\eta$)', rotation = 0, labelpad = 20)
    fig.tight_layout()

    
    
    
    
    
    
    plt.show()