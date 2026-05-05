import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt


from solver import Solver, odeV1, desingularized_ode
    
if __name__ == '__main__':
    
    epsilon = 1

    Q0 = 0.1

    a1 = 2
    a2 = a3 = 3

    etaf_guess = 0.3

    solver = Solver(a1, a2, a3, Q0, epsilon)
    sol = solver.find_etaf(etaf_guess=etaf_guess)

    ######################################################
    # TRY DESINGULARIZATIOM STRATEGY
    mu = a2+a3
    gamma = 1/(mu - a1)
    omega = mu/(2*(mu-a1))
    

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    ode_old = odeV1(sol.eta, sol.x, sol.gamma, sol.omega, sol.a1, sol.a2, sol.a3, sol.epsilon)
    ode_new = desingularized_ode(sol.eta, sol.x, sol.gamma, sol.omega, sol.a1, sol.a2, sol.a3, sol.epsilon)
    for e,ax in zip(range(3), [ax1,ax2,ax3]):
        # l, =ax.plot(sol.eta, ode_old[e], '-')
        # ax.plot(sol.eta, ode_new[e], '-.', )
        
        l, =ax.plot(sol.eta, abs(ode_old[e]), '-')
        ax.plot(sol.eta, abs(ode_new[e]), '-.', )
        

        ax.grid()
        # ax.set_ylim(-10, min(max(ode_new[e])+abs(max(ode_new[e])), 10))

        ax.set_yscale('log')


    # fig2, ax2 = plt.subplots()
    # ax2.plot(sol.eta, sol.x[0])
    # ax2.grid()

    plt.show()