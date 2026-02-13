import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

    
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
    fpp = 1/(omega*eta)*((x[0]**a1*x[1] - x[2])/(epsilon*x[0]**(a2+a3)) + (gamma*a3 + gamma - omega)*x[1]) - a3*x[1]**2/x[0]
    qp = gamma*x[0] - omega*eta*x[1]
    return x[1], fpp, qp

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
    fp = 1/(omega*eta)*(gamma*x[0] - x[1]/x[0]**(a3))
    gp = (x[2] - x[0]**a1*fp)/(epsilon*x[0]**a2)
    qp = gamma*x[0] - omega*eta*x[1]
    return x[1], gp, qp

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

    def __init__(self, a1, a2, a3, Q0, epsilon = 1):
        
        # Problem parameters
        self.a1:float = a1
        self.a2:float = a2
        self.a3:float = a3
        
        self.Q0:float = Q0
        
        self.epsilon:float = epsilon    
        
        
        # Self-similarity parameters
        self.gamma: float = 1/(a2 + a3 - a1)
        self.omega: float  = 0.5*(self.a1*self.gamma + 1)
        self.beta: float = (2 + 2*a1 - a2 - a3)/(2*(a2+a3-a1))

        self.__check_conditions()

        # Setting up params in ODE
        self.ode = lambda eta, x: ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        self.inverted_ode = lambda eta, x: inverted_ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        
    def __check_conditions(self):
        if self.a3 + self.a2 - self.a1 > 0:
            pass
        else:
            raise ValueError("Coefficients do not satisfy inequality")

    @staticmethod
    def ode(self, eta, x)->OdeResult:
        pass

    
    @staticmethod
    def inverted_ode(self, eta, x)->OdeResult:
        pass

    def solve(self, f0, eta0 = 1e-6):
        # Compute quantities at eta = deta
        fp0 = self.fp_powerseries(eta0, f0)
        g0 = f0**(self.a3)*fp0*(self.gamma*f0 - self.beta*eta0*fp0)
        x0 = [f0, 
              g0,
              self.q_powerseries(eta0, f0)]
    
        sol = solve_ivp(self.ode, [eta0, 1], x0, rtol = 1e-10, atol = 1e-10)
        i = -100
        print([sol.y[0,i], 0]), 
        inverted_sol = solve_ivp(self.inverted_ode, [sol.y[0,i], 0], [sol.t[i], sol.y[1,i], sol.y[2,i]],  rtol = 1e-10, atol = 1e-10, first_step = sol.y[0,i]/100)
        return sol, inverted_sol
    
    
    def f_powerseries(self, eta, f0):
        """Power series expansion  fo f(eta)"""
        return f0 + self.f1(f0)*eta
        pass

    def fp_powerseries(self, eta, f0):
        """Power series expansion  fo f'(eta)"""
        return f0*0.1
        pass

    def q_powerseries(self, eta, f0):
        """Power series expansion  fo q(eta)"""
        return f0*0.1
        pass

    def f1(self, f0):
        D = -(self.gamma*self.epsilon*f0**(self.C + self.G) + self.G *self.gamma*self.epsilon*f0**(self.C+self.G) + f0**self.xi)
        return self.Q0/D

    
    def check_integral_condition(self, sol:OdeResult, Q):
        I = np.trapezoid(sol.y[0], sol.t)
        return I - Q/(self.beta+1)

    @property
    def xi(self): return self.a1
    
    @property
    def C(self): return self.a2
    
    @property
    def G(self): return self.a3

if __name__ == '__main__':
    eps = np.finfo(float).eps
    print(eps)
    a1 = 3 # Power of D
    a2 = 4 # Power of K
    a3 = 0.8 # Power of tau

    Q = 0.1 # Flux Pre-factor
    epsilon = 1e-2

    solver = Solver(a1, a2, a2, epsilon)

    f0 = 0.1
    deta = 1e-12
    sol, inverted_sol = solver.solve(f0=f0, eta0 = deta)
    R = solver.check_integral_condition(sol, Q)
    print(inverted_sol)
    fig, ax = plt.subplots(1,3)
    
    [ax[_].plot(sol.t, sol.y[_]) for _ in range(3)]
    
    print(sol.t[-1])
    print(inverted_sol.y[0])
    print(sol.y[0,-1])
    print(inverted_sol.t)
    ax[0].plot(inverted_sol.y[0], inverted_sol.t)
    
    # ax[0].plot(sol.t, solver.f_powerseries(sol.t, f0), '--')
    # ax[0].set_yscale('log')
    print(len(sol.t))
    # ax.plot(sol.y[0], sol.t)

    [a.grid() for a in ax]
    plt.show()

