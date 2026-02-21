import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver

def save(f0_search, eta_data, x_data, prefix = None, directory = Path.cwd()):

    filename = f'f0span_{f0_search[0]:.3f}_{f0_search[-1]:.3f}_solution.pkl'
    if prefix is not None:
        filename = prefix + '_' + filename
    
    # Create directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename

    with open(filepath, "wb") as f:
        pickle.dump((f0_search, eta_data, x_data), f)
    print("Simulation successfully saved as : {}\n".format(filepath))
    return filepath

def execute_residual_scan(solver:Solver, f0_search, prefix = None):
    sol_eta, sol_x = solver.solve(f0=f0_search, eta0 = eta0)
    return save(f0_search, sol_eta, sol_x, prefix), (f0_search, sol_eta, sol_x)
    
def plot_data(f0_search, sol_eta, sol_x, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,2)
    
    xl, yl = ax[0].get_xlim(), ax[0].get_ylim()
    for eta, x in zip(sol_eta, sol_x):
        ax[0].plot(eta, x[0], linewidth = 0.2)

    Res = solver._check_integral_condition(sol_eta, sol_x)
    ax[1].plot(f0_search, Res)

    
    return ax

if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau
    
    
    a1 = 3   # Power of D
    a2 = 4   # Power of K
    a3 = 1   # Power of tau

    # a1 = 4   # Power of D
    # a2 = 5   # Power of K
    # a3 = 2   # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a3, q0, epsilon)

    eta0 = 1e-2
    # eta0 = 1e-6
    
    f0_guess = 0.3
    updated_f0_guess = f0_guess
    # updated_f0_guess = solver._update_f0(f0_guess, eta0, step=1e-2)
    # print(f0_guess, updated_f0_guess)
    # # actual_eta, actual_x = solver.solve(f0=f0_guess,eta0=eta0)
    
    # sol1_f0, (sol1_eta, sol1_x) = solver.find_f0(updated_f0_guess) 
    # Res = solver._check_integral_condition(sol1_eta, sol1_x)
    # print(f"Solved f0:  {sol1_f0:.6f}")
    # print(f"etaf:       {sol1_eta[-1]:.6f}")
    # print(f"Residual:   {Res:.6e}")
    # print()

    # sol2_f0, (sol2_eta, sol2_x) = solver.find_f0([updated_f0_guess, 1], method='brentq') 
    # Res = solver._check_integral_condition(sol2_eta, sol2_x)
    # print(f"Solved f0:  {sol2_f0:.6f}")
    # print(f"etaf:       {sol2_eta[-1]:.6f}")
    # print(f"Residual:   {Res:.6e}")
    # print()
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_ylabel("$f(\\eta)$")
    ax[0].set_xlabel("$\\eta$")
    ax[1].set_xlabel("$f_0$")
    ax[1].set_ylabel("Residual")
    ax[0].grid()
    ax[1].grid()

    
    
    # # PLOT POWER SERIES
    
    # f, fp , q = solver.evaluate_power_series(sol1_eta, sol1_f0)
    # l1, = ax[0].plot(sol1_eta, sol1_x[0], 'r', zorder = 1e7)
    # ax[0].plot(sol1_eta, f, '-.',  color = 'k', label = "Power series", zorder = 1e7*2)
    
    # f, fp , q = solver.evaluate_power_series(sol2_eta, sol2_f0)
    # l2, = ax[0].plot(sol2_eta, sol2_x[0], 'b', zorder = 1e7)
    # ax[0].plot(sol2_eta, f, '-.',  color = 'k', label = "Power series", zorder = 1e7*2)
    
    
    f0 = np.linspace(updated_f0_guess, 0.8,1000)
    # f0 = np.linspace(updated_f0_guess, 0.48,100)
    # f0 = np.sort(np.concatenate([f0, [f0_actual]]))
    filepath, (f0_search, sol_eta, sol_x) = execute_residual_scan(solver, f0)
    # filepath = 'f0span_0.372_0.400_solution.pkl'
    # with open(filepath, "rb") as f:
    #     f0_search, sol_eta, sol_x = pickle.load(f)

    plot_data(f0_search, sol_eta, sol_x, ax = ax)
    
    # ax[0].set_xlim(0, max(sol1_eta[-1], sol2_eta[-1]))
    # ax[0].set_ylim(0, max(max(l1.get_ydata()), max(l2.get_ydata())))



    
    plt.show()
