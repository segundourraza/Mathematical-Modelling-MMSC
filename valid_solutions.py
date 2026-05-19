import pickle, re, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

from solver import Solver

def get_next_filename(directory, base_name, extension):
    """
    directory: folder where files are stored
    base_name: file prefix (e.g., "report")
    extension: file extension without dot (e.g., "txt")
    """
    pattern = re.compile(rf"^{re.escape(base_name)}(\d+)\.{re.escape(extension)}$")

    existing_numbers = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1

    return os.path.join(directory, f"{base_name}{next_number}.{extension}")

def save(f0_search, eta_data, x_data, directory = Path.cwd(), base_filename= 'sol'):
    
    # Create directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)
    filepath = get_next_filename(directory,base_filename, 'pkl')
    
    with open(filepath, "wb") as f:
        pickle.dump((f0_search, eta_data, x_data), f)
    print("Simulation successfully saved as : {}\n".format(filepath))
    return filepath

    
def plot_data(f0_search, sol_eta, sol_x, ax, steps = 1):
    
    xl, yl = ax[0].get_xlim(), ax[0].get_ylim()
    for eta, x in zip(sol_eta[::steps], sol_x[::steps]):
        ax[0].plot(eta, x[0], linewidth = 0.2)

    Res = solver._check_integral_condition(sol_eta, sol_x)
    ax[1].plot(f0_search, Res, '.-')

    
    return ax

if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau
    
    # a1 = 3   # Power of D
    # a2 = 2   # Power of K
    # a3 = 0.8 # Power of tau
    
    
    # a1 = 3   # Power of D
    # a2 = 4   # Power of K
    # a3 = 1   # Power of tau

    # a1 = 4   # Power of D
    # a2 = 5   # Power of K
    # a3 = 2   # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a3, q0, epsilon)

    eta0 = 1e-2
    # eta0 = 1e-6
    
    f0_guess = 0.2
    f0_low, f0_high = solver._find_valid_sol(f0_guess, eta0)
    print(f0_low, f0_high)
    
    f0_low, f0_high  = f0_guess, 3*f0_guess

    f0_search = np.linspace(f0_low, f0_high,100)
    sol_eta, sol_x = solver.forward_integration(f0=f0_search, deta = eta0)
    save(f0_search, sol_eta, sol_x)
    filename = 'sol2.pkl'
    with open(filename, 'rb') as f:
        f0_search, sol_eta, sol_x = pickle.load(f)



    sol1_f0, (sol1_eta, sol1_x) = solver.find_f0(f0_guess, eta0=eta0) 
    sol1_f0, (sol1_eta, sol1_x) = solver.find_f0(0.5*(f0_high + f0_low), eta0=eta0) 
    Res1 = solver._check_integral_condition(sol1_eta, sol1_x)
    print(f"Solved f0:  {sol1_f0:.6f}")
    print(f"etaf:       {sol1_eta[-1]:.6f}")
    print(f"Residual:   {Res1:.6e}")
    print()

    # sol2_f0, (sol2_eta, sol2_x) = solver.find_f0([f0_low, sol1_f0-1e-3],  eta0=eta0, method = 'brentq')
    # Res2 = solver._check_integral_condition(sol2_eta, sol2_x)
    # print(f"Solved f0:  {sol2_f0:.6f}")
    # print(f"etaf:       {sol2_eta[-1]:.6f}")
    # print(f"Residual:   {Res2:.6e}")
    # print()
    
    # # sol3_f0, (sol3_eta, sol3_x) = solver.find_f0([sol1_f0, f0_high], method = 'brentq')
    # sol3_f0, (sol3_eta, sol3_x) = solver.find_f0([f0_high-1e-3, f0_high], method = 'brentq')
    # Res3 = solver._check_integral_condition(sol3_eta, sol3_x)
    # print(f"Solved f0:  {sol3_f0:.6f}")
    # print(f"etaf:       {sol3_eta[-1]:.6f}")
    # print(f"Residual:   {Res3:.6e}")
    # print()
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_ylabel("$f(\\eta)$")
    ax[0].set_xlabel("$\\eta$")
    ax[1].set_xlabel("$f_0$")
    ax[1].set_ylabel("Residual")
    ax[0].grid()
    ax[1].grid()

    ax[1].axvline(f0_low,  linestyle = '--', color = 'r')
    ax[1].axvline(f0_high, linestyle = '--', color = 'r')
    

    steps = 1
    xl, yl = ax[0].get_xlim(), ax[0].get_ylim()
    for eta, x in zip(sol_eta[::steps], sol_x[::steps]):
        ax[0].plot(eta, x[0], linewidth = 0.2)

    Res = solver._check_integral_condition(sol_eta, sol_x)
    ax[1].plot(f0_search, Res, '-')


    # # PLOT POWER SERIES
    # ee = np.linspace(0, sol1_eta[-1], 100)
    # f, fp , q = solver.evaluate_power_series(ee, sol1_f0)
    # l, = ax[0].plot(sol1_eta, sol1_x[0], 'r', zorder = 1e7, linewidth = 2, label = f"f(0) = {sol1_f0:.4f}")
    # ax[0].plot(ee, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2, linewidth = 1.25)
    # ax[1].plot(sol1_f0,  Res1, 'o',  color = l.get_color(), zorder = 1e7)
    
    # f, fp , q = solver.evaluate_power_series(sol2_eta, sol2_f0)
    # l, = ax[0].plot(sol2_eta, sol2_x[0], 'm', zorder = 1e7, label = f"f(0) = {sol2_f0:.4f}")
    # ax[0].plot(sol2_eta, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2)
    # ax[1].plot(sol2_f0,  Res2, 'o',  color = l.get_color(), zorder = 1e7)
    
    # f, fp , q = solver.evaluate_power_series(sol3_eta, sol3_f0)
    # l, = ax[0].plot(sol3_eta, sol3_x[0], 'lime', zorder = 1e7, label = f"f(0) = {sol3_f0:.4f}")
    # ax[0].plot(sol3_eta, f,   '-.',  color = 'k', label = "Power series", zorder = 1e7*2)
    # ax[1].plot(sol3_f0,  Res3, 'o',  color = l.get_color(), zorder = 1e7)
    # ax[0].legend()


    fig.tight_layout()
    # ax[0].set_xlim(0, max(sol1_eta[-1], sol2_eta[-1]))
    # ax[0].set_ylim(0, max(max(l1.get_ydata()), max(l2.get_ydata())))



    
    plt.show()
