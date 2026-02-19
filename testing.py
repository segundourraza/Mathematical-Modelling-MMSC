import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver

def save(f0_search, eta_data, x_data, prefix = None, directory = Path.cwd()):

    filename = f'f0span_{f0_search[0]:.3f}_{f0_search[0]:.3f}_solution.pkl'
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
    return save(f0_search, sol_eta, sol_x, prefix)
    
def plot_data(f0_search, sol_eta, sol_x, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,2)
    
    ax[0].plot(actual_eta, actual_x[0], 'r')
    xl, yl = ax[0].get_xlim(), ax[0].get_ylim()
    for eta, x in zip(sol_eta, sol_x):
        ax[0].plot(eta, x[0], linewidth = 0.2)
    ax[0].plot(actual_eta, actual_x[0], 'r')
    ax[0].grid()
    ax[0].set_ylabel("$f_0$")
    ax[0].set_xlabel("$\\eta$")
    

    Res = solver._check_integral_condition(sol_eta, sol_x)
    ax[1].plot(f0_search, Res)
    ax[1].axvline(f0_actual, linestyle = '--', color = 'k')
    ax[1].set_xlabel("$f_0$")
    ax[1].set_ylabel("Residual")
    ax[1].grid()
    
    return ax

if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a3, q0, epsilon)

    eta0 = 1e-2
    # eta0 = 1e-6
    
    f0_guess = 0.3924

    # f0_actual = f0_guess
    # actual_eta, actual_x = solver.solve(f0=f0_actual,eta0=eta0)
    f0_actual, (actual_eta, actual_x) = solver.find_f0(f0_guess) 
    Res = solver._check_integral_condition(actual_eta, actual_x)
    print(f"Solved f0:  {f0_actual:.4f}")
    print(f"Residual:   {Res:.4e}")
    
    
    f0 = np.linspace(f0_actual-0.02, 0.4,100)
    f0 = np.sort(np.concatenate([f0, [f0_actual]]))
    filepath = execute_residual_scan(solver, f0)
    # filepath = 'f0span_3.724e-01_3.724e-01_solution.pkl'
    
    with open(filepath, "rb") as f:
        f0_search, sol_eta, sol_x = pickle.load(f)


    ax = plot_data(f0_search, sol_eta, sol_x)
    
    test_f0, (test_eta, test_x) =solver.find_f0([0.378, 0.38], method = 'brentq')
    Res = solver._check_integral_condition(test_eta, test_x)
    line, = ax[0].plot(test_eta, test_x[0], 'b')
    print(f"Solved f0:  {test_f0:.4f}")
    print(f"Residual:   {Res:.4e}")

    ax[0].set_xlim(0, max(line.get_xdata())*1.2)
    ax[0].set_ylim(0, max(f0_search))
    





    f0 = np.linspace(0.4, 0.55,1000)
    f0 = np.linspace(0.4, 0.446,1000)
    filepath = execute_residual_scan(solver, f0)
    # filepath = 'f0span_3.724e-01_3.724e-01_solution.pkl'
    
    with open(filepath, "rb") as f:
        f0_search, sol_eta, sol_x = pickle.load(f)


    ax = plot_data(f0_search, sol_eta, sol_x, ax = ax)
    


    try:
        # f0_next, (actual_eta, actual_x) = solver.find_f0(0.4, method='forward') 
        f0_next, (actual_eta, actual_x) = solver.find_f0([0.4, 0.44595], method='brentq') 
        Res = solver._check_integral_condition(actual_eta, actual_x)
        print(f"Solved f0:  {f0_next:.4f}")
        print(f"Residual:   {Res:.4e}")
        line, = ax[0].plot(actual_eta, actual_x[0], 'lime')

        ax[0].set_xlim(0, max(line.get_xdata())*1.2)
        ax[0].set_ylim(0, max(max(f0_search), f0_next))
    except Exception as e:
        print(e)



    
    
    
    plt.show()
