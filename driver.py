import numpy as np
import matplotlib.pyplot as plt

from solver import execute_solver


if __name__ == '__main__':
    a1 = 2   # Power of D
    a2 = 3   # Power of K
    a3 = 0.8 # Power of tau
    
    
    # a1 = 3   # Power of D
    # a2 = 4   # Power of K
    # a3 = 1   # Power of tau

    # a1 = 4   # Power of D
    # a2 = 5   # Power of K
    # a3 = 2   # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1


    f0_guess = 0.2
    eta0 = 1e-4

    
    fig, ax = plt.subplots()
    f0, (eta, x) = execute_solver(a1, a2, a3, q0, epsilon, f0_guess, eta0)
    ax.plot(eta, x[0])
    ax.set_xlabel('$\\eta$')
    ax.set_ylabel('$f(\\eta)$')
    ax.grid()
    plt.show()


    # a1_list = [2, 3, 4]
    # a2_list = [3, 4, 5]
    # a3_list = [0.8, 1, 2]

    # fig, ax = plt.subplots()
    # ls = ['-', '-.', '--']
    # for i, (a1, a2, a3) in enumerate(zip(a1_list, a2_list, a3_list)):
    #     eta, x = execute_solver(a1, a2, a3, q0, epsilon, f0_guess, eta0)
    #     ax.plot(eta, x[0], linestyle = ls[i])
    # ax.set_xlabel('$\\eta$')
    # ax.set_ylabel('$f(\\eta)$')
    # ax.grid()



    plt.show()
