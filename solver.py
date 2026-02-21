import numpy as np
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from expansion_coefficients import coeffs_fq
from tqdm import tqdm

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
    return np.array([fp, gp, qp])

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
    return [etap, gp, qp]

class Solver:
    ZERO_F = 1e-9

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
            raise ValueError(f"Coefficients do not satisfy inequality, a3 + a2 - a1 ({self.a3 + self.a2 - self.a1}) >0")

    def evaluate_power_series(self,eta, f0):
        f, q = coeffs_fq(self.gamma, self.omega, self.xi, self.C, self.G, f0, self.Q0, self.epsilon)
        f_poly = np.polynomial.Polynomial(f)
        q_poly = np.polynomial.Polynomial(q)
        return f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta)
          
    def solve_old(self, f0, eta0 = 1e-6):
        # Compute quantities at eta = deta
        f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
        x0 = [f0, 
              fp0,
              q0]
        
        fode = lambda eta, x: ode_old(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        sol = self.__integrate(fode, x0, eta0)
        return sol
    
    def solve(self, f0, eta0, ftol = 1e-2):
        if isinstance(f0, (list, tuple, np.ndarray)):
            nf0 = len(f0)
            a = [0]*nf0
            b = [0]*nf0
            for i in tqdm(range(nf0)):
                a[i], b[i] = self.solve(f0[i], eta0, ftol)
            return a, b
        else:
            # Compute quantities at eta = deta
            f0, fp0, q0 = self.evaluate_power_series(eta0, f0)
            if f0 < 0:
                raise FloatingPointError("try a larger value of f(0)")
            g0 = f0**(self.a3)*(self.gamma*f0 - self.omega*eta0*fp0)
            x0 = [f0, g0, q0]

            # Normal solve
            fode = lambda eta, x: ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
            sol = self.__integrate(fode, x0, eta0, ftol)
            if sol.status == 1:
                inverted_sol = self.inverted_solve(sol)
                return np.concatenate([sol.t[:-1], inverted_sol.y[0]]), np.column_stack([sol.y[:,:-1],np.vstack([inverted_sol.t, inverted_sol.y[1:]])])
            else:
                return sol.t, sol.y

        
    def inverted_solve(self, sol:OdeResult):
        # Inverted Solve
        finverted_ode = lambda eta, x: inverted_ode(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        i = -1
        return solve_ivp(finverted_ode, [sol.y[0,i], self.ZERO_F], [sol.t[i], sol.y[1,i], sol.y[2,i]],  
                         rtol = 1e-10, atol = 1e-10,
                         first_step = 1e-8
                        )
        
    #########################################################################
    # ROOT FINDERS
    def _update_f0(self, f0_guess, eta0, step = 0.1, max_iter = 50, direction = 1):
        
        def func(f0):
            eta, x = self.solve(f0=f0, eta0 = eta0)
            return self._check_integral_condition(eta, x)

        try:        
            f_old, R_old = f0_guess, func(f0_guess)
        except FloatingPointError:
            for _ in range(max_iter):
                f0_guess = f0_guess+step*direction
                try:
                    f_old, R_old = f0_guess, func(f0_guess)
                    break
                except:
                    continue
            else:
                raise RuntimeError()
            
        for iter in range(max_iter):
            f_new = f0_guess + direction*step*iter
            try:
                R_new = func(f_new)
            except:
                continue
            if R_old*R_new <= 0:
                return brentq(func, f_old, f_new)
            # update anchor in same direction
            f_old, R_old = f_new, R_new
        raise ValueError("Failed to find a bracket after scanning.")


    def find_f0(self, f0, eta0 = 1e-2, method = 'newton'):
        def func(f0):
            eta, x = self.solve(f0=f0, eta0 = eta0)
            return self._check_integral_condition(eta, x)
        
        if method == 'newton':        
            if isinstance(f0, (float,int)):
                f0 = newton(func, f0)
            else:
                raise ValueError("For 'newton' method 'f0' must be a scalar")
        elif method == 'brentq':
            if isinstance(f0, (list,tuple, np.ndarray)) and len(f0) == 2:
                f0 = brentq(func, *f0)
            else:
                raise ValueError("For 'brentq' method 'f0' must be an interval")
        elif method == 'forward':
            if isinstance(f0, (float,int)):
                f0 = find_root_forward(func, f0, step = eta0, xmax = 1)
            else:
                raise ValueError("For 'newton' method 'f0' must be a scalar")

        else:
            raise ValueError("'f0_span' must be a float to initialize a 'newton' root finder, or a list of length 2 to initialize a 'brentq' root finder.")
        return f0, self.solve(f0=f0, eta0 = eta0)

    ###########################################################################
    # AUXILIARY METHODS
    def __event(self, f0, ftol):
        def zero_f(eta, x):
            return x[0] > ftol
        zero_f.terminal = True
        zero_f.direction = -1
        
        return (zero_f,)
        
        def blowup_f(eta, x):
            return x[0] < f0*10
        blowup_f.terminal = True
        blowup_f.direction = -1

        return (zero_f, blowup_f)
        
    def __integrate(self, ode, x0, eta0, ftol):
        return solve_ivp(ode, [eta0, 1], x0, rtol = 1e-10, atol = 1e-10, first_step = 1e-6, events=self.__event(x0[0], ftol), vectorized=True)

    def _check_integral_condition(self, eta, x):
        if all(isinstance(_, (list, np.ndarray)) for _ in eta):
            return [self._check_integral_condition(i,j) for i,j in zip(eta, x)]
        else:
            I = np.trapezoid(x[0], eta)
            res = I - self.Q0/(self.beta+1)
            return res 



    @property
    def xi(self): return self.a1
    
    @property
    def C(self): return self.a2
    
    @property
    def G(self): return self.a3




def execute_solver(a1, a2, a3, q0, epsilon, f0_guess, eta0):
    solver = Solver(a1, a2, a3, q0, epsilon)
    
    f0, fp0, q0 = solver.evaluate_power_series(eta0, f0)
    if f0 < 0:
        try:
            f0_guess = solver._update_f0(f0_guess, eta0)
            print("Updated ")
        except ValueError:
            print("Failed to update f0 to physcial bracket")
            raise RuntimeError
    return solver.find_f0(f0_guess, eta0)
    return solver.solve(f0_guess,eta0)



def find_root_forward(f, x0, step=0.1, xmax=100):
    x_left = x0
    f_left = f(x_left)

    x_right = x_left + step
    while x_right <= xmax:
        f_right = f(x_right)
        if f_left * f_right <= 0:
            return brentq(f, x_left, x_right)
        x_left, f_left = x_right, f_right
        x_right += step

    raise ValueError("No root found in positive direction.")