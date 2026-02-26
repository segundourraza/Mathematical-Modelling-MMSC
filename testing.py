import numpy as np 
import matplotlib.pyplot as plt
np.set_printoptions(linewidth = 270)
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})
from solver import Solver


def compute_values(eta, x):
    f0, fp0, q0 = solver.evaluate_power_series(eta0, f0_guess)

    f, g, q = x[:,0]
    fp = (solver.gamma*f - g/(f**a3))/(solver.omega*eta[0])
    gp = (-q - f**a1*fp)/(epsilon*f**(a2))
    qp = -solver.gamma*f + solver.omega*eta[0]*fp
    return (f0, fp0, q0), (f, g, q), (fp, gp, qp), (solver.gamma*f - g/(f**a3), solver.omega*eta[0])

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

    solver = Solver(a1, a2, a3, q0, epsilon)

    f0_guess = 0.36
    
    eta0 = 1e-2
    
    fig, ax = plt.subplots()
    fig, ax2 = plt.subplots(1,3)
    eta, x = solver.solve(f0=f0_guess, deta = eta0, state_space=1)
    ax.plot(eta, x[0])
    mask = eta < 0.1
    xdot = solver._ode(eta, x)
    for i,a in enumerate(ax2):
        a.plot(eta[mask], xdot[i][mask])
    eta, x = solver.solve(f0=f0_guess, deta = eta0, state_space = 2)
    ax.plot(eta, x[0])
    mask = eta < 0.1
    xdot = solver._ode(eta, x)
    for i,a in enumerate(ax2):
        a.plot(eta[mask], xdot[i][mask])
    
    
    # fig, ax = plt.subplots(2,3, layout = 'tight')


    # fact = 1/4
    # for f in np.append([1],np.linspace(fact, 1/fact, 4)):
    # # for f in np.linspace(1, 1/fact, 5):
    # # for f in [1, 0.75, 0.5, 0.25]:
    #     eta, x = solver.solve(f0=f0_guess, deta = f*eta0)
    #     mask = eta < 0.15
    #     mask = eta < 0.2
    #     # mask = eta > 0
    #     sol = solver._ode(eta, x)
    #     [ax[0][i].plot(eta[mask], x[i][mask], label = "$\\eta_{{start}}$ = {}".format(f*eta0)) for i in range(3)]
    #     [ax[1][i].plot(eta[mask], sol[i][mask]) for i in range(3)]
    # ax[0][0].set_ylabel("$f(\\eta)$")
    # ax[0][1].set_ylabel("$g(\\eta)$")
    # ax[0][2].set_ylabel("$q(\\eta)$")
    # ax[1][0].set_ylabel("$f^\\prime(\\eta)$")
    # ax[1][1].set_ylabel("$g^\\prime(\\eta)$")
    # ax[1][2].set_ylabel("$q^\\prime(\\eta)$")
    # ax[0][0].legend()
    # for a in ax.flatten():
    #     a.grid()
    #     a.set_xlabel("$\\eta$")



    # fact = 1/4
    # for e in np.append([1],np.linspace(fact, 1/fact, 4)):
    #     eta, x = solver.solve(f0=f0_guess, deta = f*eta0)
    #     mask = eta < 0.15
    #     mask = eta < 0.2
    #     # mask = eta > 0
    #     sol = solver._ode(eta, x)
    #     [ax[0][i].plot(eta[mask], x[i][mask], label = "$\\eta_{{start}}$ = {}".format(f*eta0)) for i in range(3)]
    #     [ax[1][i].plot(eta[mask], sol[i][mask]) for i in range(3)]
    # ax[0][0].set_ylabel("$f(\\eta)$")
    # ax[0][1].set_ylabel("$g(\\eta)$")
    # ax[0][2].set_ylabel("$q(\\eta)$")
    # ax[1][0].set_ylabel("$f^\\prime(\\eta)$")
    # ax[1][1].set_ylabel("$g^\\prime(\\eta)$")
    # ax[1][2].set_ylabel("$q^\\prime(\\eta)$")
    # ax[0][0].legend()
    # for a in ax.flatten():
    #     a.grid()
    #     a.set_xlabel("$\\eta$")

    
    plt.show()