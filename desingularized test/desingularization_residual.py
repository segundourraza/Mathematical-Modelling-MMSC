import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
np.seterr(over='raise')

from tqdm import tqdm
from solver import Solver
    
if __name__ == '__main__':
    
    epsilon = 1

    Q0 = 0.1

    a1 = 2
    a2 = a3 = 3

    etaf_guess = 0.3

    solver = Solver(a1, a2, a3, Q0, epsilon)
    
    eta0 = 1e-6
    eta_transition = 1e-2
    
    ################################################
    # RESIDUAL ITERATOR
    data = []
    iterable = np.linspace(2, 0.15, 10)
    for f0_guess in tqdm(iterable):
        eta,x = solver.desingularized_forward_integration(f0_guess, deta=eta0, eta_transition=eta_transition)
        data.append([f0_guess, solver._check_integral_condition(eta,x)])
    data = np.array(data)

    fig, ax = plt.subplots()
    ax.plot(*data.T)
    
    sol_backward = solver.find_etaf(etaf_guess=etaf_guess)
    print(f"Solved f0:  {sol_backward.f0:.6f}")
    print(f"etaf:       {sol_backward.eta_f:.6f}")
    print(f"Residual:   {sol_backward.Res:.6e}")
    print()
    

    f0_guess = (0.2, 2)
    eta0 = 1e-6
    eta_transition = 1e-2
    sol_forward = solver.find_f0_desingularized(f0_guess=f0_guess, eta0=eta0, eta_transition=eta_transition, method='brentq')
    print(f"Solved f0:  {sol_forward.f0:.6f}")
    print(f"etaf:       {sol_forward.eta_f:.6f}")
    print(f"Residual:   {sol_forward.Res:.6e}")
    print()
    

    fig1, ax1 = plt.subplots()
    ax1.plot(sol_backward.eta, sol_backward.x[0])
    ax1.plot(sol_forward.eta, sol_forward.x[0])
    
    ax.grid()
    
    plt.show()