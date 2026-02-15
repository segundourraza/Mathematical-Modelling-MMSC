import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from expansion_coefficients import evaluate_power_series
np.seterr(all='raise')


def ode_old(eta, x, gamma, omega, a1, a2, a3, epsilon):
    """_summary_

    Parameters
    ----------
    eta : _type_
        _description_
    x : _type_
        _description_

    Note:
    x[0] = f
    x[1] = f'
    x[2] = q
    """
    f, fp, q = x[0], x[1], x[2]
    fpp = ((f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma-omega)*fp)/(omega*eta) - a3*fp**2/f
    qp = -gamma*f+ omega*eta*fp
    return fp, fpp, qp

def ode(eta, x, gamma, omega, a1, a2, a3, epsilon):
    """_summary_

    Parameters
    ----------
    eta : _type_
        _description_
    x : _type_
        _description_

    Note:
    x[0] = f
    x[1] = g
    x[2] = q
    """
    f, g, q = x
    fp = (gamma*f - g/(f**a3))/(omega*eta)
    gp = (-q - f**a1*fp)/(epsilon*f**(a2))
    qp = -gamma*f + omega*eta*fp
    return fp, gp, qp

def inverted_ode(eta, x, gamma, omega, a1, a2, a3, epsilon):
    """_summary_

    Parameters
    ----------
    eta : _type_
        _description_
    x : _type_
        _description_

    Note:
    x[0] = eta
    x[1] = g
    x[2] = q
    """
    etap = (omega*eta)/(gamma*x[0] - x[1]/x[0]**(a3))
    fp = 1/etap
    gp = ((x[2] - x[0]**a1*fp)/(epsilon*x[0]**a2))*etap
    qp = (gamma*x[0] - omega*eta*x[1])*etap
    return x[1], gp, qp

class Solver:

    def __init__(self, a1, a2, a3, Q0, epsilon):
        
        # Problem parameters
        self.a1:float = a1
        self.a2:float = a2
        self.a3:float = a3
        
        self.Q0:float = Q0
        
        self.epsilon:float = epsilon    
        
        
        # Self-similarity parameters
        self.gamma: float = 1/(a2 + a3 - a1)
        self.omega: float  = 0.5*(self.a2 + self.a3)*self.gamma
        self.beta: float = (2 + 2*a1 - a2 - a3)/(2*(a2+a3-a1))

        self.__check_conditions()

        # Setting up params in ODE
        self.evaluate_power_series = lambda eta, f0: evaluate_power_series(eta, self.xi, self.C, self.G, f0, self.Q0, self.epsilon)
    
    def __check_conditions(self):
        if self.a3 + self.a2 - self.a1 < 0:
            raise ValueError("Coefficients do not satisfy inequality")

    @staticmethod
    def evaluate_power_series(eta, f0)->OdeResult:...

    @staticmethod
    def event(eta0):
        def func(eta, x):
            return eta-eta0 > 1e-2
        func.terminal = False
        # func.terminal = True
        return func
    
    def solve(self, f0, eta0):
        # Compute quantities at eta = deta
        f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
        g0 = f0**(self.a3)*(self.gamma*f0 - self.omega*eta0*fp0)
        x0 = [f0, 
              g0,
              q0]
    
        fode = lambda eta, x: ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        finverted_ode = lambda eta, x: inverted_ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        
        sol = self.integrate(fode, x0, eta0)
        return sol, None
        i = -1
        inverted_sol = solve_ivp(self.inverted_ode, [sol.y[0,i], 0], [sol.t[i], sol.y[1,i], sol.y[2,i]],  rtol = 1e-10, atol = 1e-10, first_step = sol.y[0,i]/100)
        return sol, inverted_sol
    
    def solve_old(self, f0, eta0 = 1e-6):
        # Compute quantities at eta = deta
        f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
        x0 = [f0, 
              fp0,
              q0]
    
        fode = lambda eta, x: ode_old(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        sol = self.integrate(fode, x0, eta0)
        return sol, None
        i = -1
        inverted_sol = solve_ivp(self.inverted_ode, [sol.y[0,i], 0], [sol.t[i], sol.y[1,i], sol.y[2,i]],  rtol = 1e-10, atol = 1e-10, first_step = sol.y[0,i]/100)
        return sol, inverted_sol
    
    def integrate(self, ode, x0, eta0):
        return solve_ivp(ode, [eta0, 1], x0, rtol = 1e-10, atol = 1e-10, events=self.event(eta0))


    
    def check_integral_condition(self, sol:OdeResult):
        I = np.trapezoid(sol.y[0], sol.t)
        return I - self.Q0/(self.beta+1)

    def find_f0(self, f0_guess):
        def func(f0):
            sol = self.solve(f0=f0, eta0 = eta0)[0]
            return self.check_integral_condition(sol)
        f0 = newton(func, f0_guess)
        sol , inverted_sol = self.solve(f0=f0, eta0 = eta0)
        return f0, sol
    

    def evaluate_fp(self, sol:OdeResult):
        return 1/(self.omega*sol.t)*(self.gamma*sol.y[0] - sol.y[1]/sol.y[0]**self.a3)


    @property
    def xi(self): return self.a1
    
    @property
    def C(self): return self.a2
    
    @property
    def G(self): return self.a3

if __name__ == '__main__':
    a1 = 2 # Power of D
    a2 = 3 # Power of K
    a3 = 0.8 # Power of tau

    q0 = 0.1 # Flux Pre-factor
    epsilon = 1

    solver = Solver(a1, a2, a2, q0, epsilon)

    f0_guess = 0.4
    eta0 = 1e-2

    sol1, _ = solver.solve(f0=f0_guess, eta0 = eta0)
    sol2, _ = solver.solve_old(f0=f0_guess, eta0 = eta0)
    # print(sol1)
    # print(sol2)
    
    R1 = solver.check_integral_condition(sol1)
    R2 = solver.check_integral_condition(sol2)
    # print(R1)
    # print(R2)
    
    f0 = f0_guess
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].plot(sol1.t, sol1.y[0])
    ax[0].plot(sol2.t, sol2.y[0], '--')

    ax[1].plot(sol1.t, sol1.y[2])
    ax[1].plot(sol2.t, sol2.y[2], '--')
    
    fig, ax = plt.subplots()
    ax.plot(sol2.t, sol2.y[1])
    
    # f0, sol = solver.find_f0(f0_guess)
    # print(f0_guess, f0)
    
    # etaf = newton(lambda x: solver.evaluate_power_series(x, f0)[0], eta0)
    # mask = sol1.t<etaf
    # print(etaf)
    # xx = np.linspace(0, etaf, 100)

    # fig1, ax1 = plt.subplots(1,2)
    
    # ax1[0].plot(sol.t[mask], sol.y[0, mask])
    # ax1[1].plot(sol.t[mask], solver.evaluate_fp(sol))
    
    
    # # xx = sol.t
    # f, fp , q = solver.evaluate_power_series(xx, f0)
    # # ax[0].plot(xx, f, '--')
    # # ax[1].plot(xx, q, '--')
    
    # ax1[0].plot(xx, f, '--')
    # ax1[1].plot(xx, fp, '--')
    
    # # ax[0].set_yscale('log')
    # # ax[0].set_xscale('log')
    # # ax.plot(sol.y[0], sol.t)

    # # [a.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) for a in np.concatenate([ax, [ax1]])]
    
    # [a.grid() for a in ax]






    plt.show()

