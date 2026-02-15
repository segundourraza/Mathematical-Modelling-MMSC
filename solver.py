import numpy as np
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from expansion_coefficients import coeffs_fq
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
    fpp = ((f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma + gamma-omega)*fp)/(omega*eta) - a3*fp**2/f
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
    return etap, gp, qp

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
        self.omega: float  = 0.5*(self.a1*self.gamma + 1)
        self.beta: float = self.gamma + self.omega - 1

        self.__check_conditions()

    def __check_conditions(self):
        if self.a3 + self.a2 - self.a1 < 0:
            raise ValueError("Coefficients do not satisfy inequality")

    def evaluate_power_series(self,eta, f0):
        f, q = coeffs_fq(self.gamma, self.omega, self.xi, self.C, self.G, f0, self.Q0, self.epsilon)
        f_poly = np.polynomial.Polynomial(f)
        q_poly = np.polynomial.Polynomial(q)
        return f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta)
    
    @staticmethod
    def event(eta0):
        def func(eta, x):
            return x[0]
            # return (eta/eta0) < 1.5
        func.terminal = True
        func.direction = -1
        return func
    
    def solve(self, f0, eta0, tol_f = 1e-8):
        # Compute quantities at eta = deta
        f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
        if f0 < 0:
            raise RuntimeError("try a larger value of f(0)")
        g0 = f0**(self.a3)*(self.gamma*f0 - self.omega*eta0*fp0)
        x0 = [f0, 
              g0,
              q0]

        # Normal solve
        fode = lambda eta, x: ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        sol = self.__integrate(fode, x0, eta0)
        return sol.t, sol.y
        if sol.y[0,-1] > 1 or np.isclose(sol.y[0,-1], 0, atol= tol_f):
            return sol.t, sol.y
        else:
            inverted_sol = self.inverted_solve(sol,tol_f = tol_f)
            print(np.shape(sol.y), np.shape(inverted_sol.y))
            return np.concatenate([sol.t[:-1], inverted_sol.t]), np.column_stack([sol.y[:,:-1], inverted_sol.y])

    
    def inverted_solve(self, sol:OdeResult, tol_f = 1e-9):
        # Inverted Solve
        finverted_ode = lambda eta, x: inverted_ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        i = -1
        return solve_ivp(finverted_ode, [sol.y[0,i], tol_f], [sol.t[i], sol.y[1,i], sol.y[2,i]],  
                         rtol = 1e-10, atol = 1e-10, 
                        first_step = sol.y[0,i]/100)
        
    
    def solve_old(self, f0, eta0 = 1e-6):
        # Compute quantities at eta = deta
        f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
        x0 = [f0, 
              fp0,
              q0]
        
        fode = lambda eta, x: ode_old(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        sol = self.__integrate(fode, x0, eta0)
        return sol
        
    def __integrate(self, ode, x0, eta0):
        return solve_ivp(ode, [eta0, 1], x0, rtol = 1e-10, atol = 1e-10, first_step = 1e-6, events=self.event(eta0))


    def find_f0(self, f0_span, eta0 = 1e-2):
        def func(f0):
            eta, x = self.solve(f0=f0, eta0 = eta0)
            return self._check_integral_condition(eta, x)
        if isinstance(f0_span, float):
            try:
                f0 = newton(func, f0_span)
            except RuntimeError:
                return self.find_f0(f0_span*2)
        elif len(f0_span) == 2:
            try:
                f0 = brentq(func, *f0_span)
            except RuntimeError:
                return self.find_f0([f0_span[0]*2, f0_span[1]], eta0=eta0)
        else:
            raise ValueError("'f0_span' must be a float to initialize a 'newton' root finder, or a list of length 2 to initialize a 'brentq' root finder.")
        return f0, self.solve(f0=f0, eta0 = eta0)


    def _check_integral_condition(self, eta, x):
        I = np.trapezoid(x[0], eta)
        return I - self.Q0/(self.beta+1)



    @property
    def xi(self): return self.a1
    
    @property
    def C(self): return self.a2
    
    @property
    def G(self): return self.a3
