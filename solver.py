import numpy as np
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from expansion_coefficients import coeffs_fq
from tqdm import tqdm

np.seterr(all='raise')


def odeV1(eta, x, gamma, omega, a1, a2, a3, epsilon):
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
    fpp = +((f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma + gamma-omega)*fp)/(omega*eta) - a3*fp**2/f
    qp = -gamma*f+ omega*eta*fp
    return fp, fpp, qp


def inverted_odeV1(f, x, gamma, omega, a1, a2, a3, epsilon):
    """Inverted ode for state-space [f,g,q]

    Parameters
    ----------
    f : _type_
        _description_
    x : _type_
        _description_

    Note:
    x[0] = eta
    x[1] = f'
    x[2] = q
    """
    eta, fp, q = x[0], x[1], x[2]
    etap = 1/fp
    fpp = -( (f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma + gamma-omega)*fp ) / (omega*eta) - a3*fp**2/f
    qp = -gamma*f+ omega*eta*fp
    return np.array([etap, fpp*etap, qp*etap])


def odeV2(eta, x, gamma, omega, a1, a2, a3, epsilon):
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
    gp = -(q + f**a1*fp)/(epsilon*f**(a2))
    qp = -gamma*f + omega*eta*fp
    return np.array([fp, gp, qp])

def inverted_odeV2(f, x, gamma, omega, a1, a2, a3, epsilon):
    """Inverted ode for state-space [f,g,q]

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
    eta, g, q = x
    fp = (gamma*f - g/(f**a3))/(omega*eta)
    etap = 1/fp
    gp = (q + f**a1*fp)/(epsilon*f**(a2))
    qp = -gamma*f + omega*eta*fp    
    return np.array([etap, gp*etap, qp*etap])



class Solver:
    ZERO_F = 0

    def __init__(self, a1, a2, a3, Q0, epsilon):
        
        # Problem parameters
        self.a1:float = a1
        self.a2:float = a2
        self.a3:float = a3
        
        self.Q0:float = Q0
        
        self.epsilon:float = epsilon    
        
        
        # Self-similarity parameters
        self.gamma: float = 1/(a2 + a3 - a1)
        # self.omega: float  = 0.5*(self.a1*self.gamma + 1)
        self.omega: float  = 0.5*(self.a2+ self.a3)*self.gamma
        self.beta: float = self.gamma + self.omega - 1
        self.beta = (self.a1 + 1)*self.gamma - self.omega
        # self.beta: float =(self.a2 + 1)*self.gamma - self.omega

        self.__check_conditions()

    def __check_conditions(self):
        if self.a3 + self.a2 - self.a1 < 0:
            raise ValueError(f"Coefficients do not satisfy inequality, a3 + a2 - a1 ({self.a3 + self.a2 - self.a1}) >0")

    def get_coeffs(self, f0):
        return coeffs_fq(self.gamma, self.omega, self.xi, self.C, self.G, f0, self.Q0, self.epsilon)
    
    def evaluate_power_series(self, eta, f0):
        f, q = coeffs_fq(self.gamma, self.omega, self.xi, self.C, self.G, f0, self.Q0, self.epsilon)
        f_poly = np.polynomial.Polynomial(f)
        q_poly = np.polynomial.Polynomial(q)
        return f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta)
    
    def solve(self, f0, deta, invert_fraction = 0.1, state_space = 1):
        if isinstance(f0, (list, tuple, np.ndarray)):
            nf0 = len(f0)
            a = [0]*nf0
            b = [0]*nf0
            for i in tqdm(range(nf0)):
                a[i], b[i] = self.solve(f0[i], deta, invert_fraction, state_space=state_space)
            return a, b
        else:
            # Compute quantities at eta = deta
            f_deta, fp_deta, q_deta = self.evaluate_power_series(deta, f0)
            x_deta = [f_deta, fp_deta, q_deta]
            if f_deta < 0:
                raise FloatingPointError("try a larger value of f(0)")
            if state_space == 1:
                self._ode = lambda eta, x: odeV1(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
                inverted_solve = self.inverted_solveV1
            elif state_space == 2:
                x_deta[1] = f_deta**(self.a3)*(self.gamma*f_deta - self.omega*deta*fp_deta)
                self._ode = lambda eta, x: odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
                inverted_solve = self.inverted_solveV2
            else:
                raise ValueError()
            
            # Normal solve
            sol = self._integrate(self._ode, x_deta, deta, invert_fraction)
            # return sol.t, sol.y
            
            if sol.status == 1:
                inverted_sol = inverted_solve(sol)
                print(inverted_sol.t[[0,-1]])
                return np.concatenate([sol.t[:-1], inverted_sol.y[0]]), np.column_stack([sol.y[:,:-1],np.vstack([inverted_sol.t, inverted_sol.y[1:]])])
            else:
                return sol.t, sol.y

        
    
    def inverted_solveV1(self, sol:OdeResult):
        # Inverted Solve
        self._inverted_ode = lambda f, x: inverted_odeV1(f, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        return solve_ivp(self._inverted_ode, [sol.y[0,-1], self.ZERO_F], [sol.t[-1], sol.y[1,-1], sol.y[2,-1]],
                         rtol = 1e-10, atol = 1e-10,
                         first_step = 1e-8
                        )

    def inverted_solveV2(self, sol:OdeResult):
        # Inverted Solve
        self._inverted_ode = lambda eta, x: inverted_odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        return solve_ivp(self._inverted_ode, [sol.y[0,-1], self.ZERO_F], [sol.t[-1], sol.y[1,-1], sol.y[2,-1]],  
                         rtol = 1e-10, atol = 1e-10,
                         first_step = 1e-8
                        )
    


    #########################################################################
    # ROOT FINDERS

    def _find_valid_sol(self, f0_start, eta0):
        def func(f0):
            eta, x = self.solve(f0=f0, deta = eta0)
            return x[0][-1] - self.ZERO_F
        
        f0_low, f0_high = find_bracket_forward(func, f0_start)
        return f0_low, f0_high
        


    def _update_f0(self, f0_guess, eta0, step = 0.1, max_iter = 50, direction = 1):
        
        def func(f0):
            eta, x = self.solve(f0=f0, deta = eta0)
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
            eta, x = self.solve(f0=f0, deta = eta0)
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
                f0 = find_bracket_forward(func, f0, step = eta0, xmax = 1)
            else:
                raise ValueError("For 'newton' method 'f0' must be a scalar")

        else:
            raise ValueError("'f0_span' must be a float to initialize a 'newton' root finder, or a list of length 2 to initialize a 'brentq' root finder.")
        return f0, self.solve(f0=f0, deta = eta0)

    ###########################################################################
    # AUXILIARY METHODS
    def __event(self, f0, invert_fraction):
        def zero_f(eta, x):
            return x[0]/f0 > invert_fraction
        zero_f.terminal = True
        zero_f.direction = -1
        
        return (zero_f,)
        
    def _integrate(self, ode, x0, eta0, invert_fraction):
        return solve_ivp(ode, [eta0, 3], x0, rtol = 1e-10, atol = 1e-10, first_step = 1e-6, events=self.__event(x0[0], invert_fraction), vectorized=True)

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




def find_bracket_forward(f, x0, step=0.01, xmax=1, tol = 1e-10, max_iter = 100):
    x_left = x0
    x_right = x_left + step
    while x_right <= xmax:
        f_right = f(x_right)
        print(x_right, f_right)
        if f_right == 0:
            x_low = binary_search_left(f, x_left, x_right, tol = tol, max_iter=max_iter)
            break
        x_left = x_right
        x_right += step
    else:
        raise ValueError("No root found in negative direction.")

    while x_right <= xmax:
        f_right = f(x_right)
        if f_right > 0:
            x_high = binary_search_right(f, x_left, x_right, tol = tol, max_iter=max_iter)
            break
        x_left = x_right
        x_right += step
    else:
        raise ValueError("No root found in positive direction.")

    return x_low, x_high

def binary_search_left(f, x_left, x_right, tol = 1e-10, max_iter = 100):
    for i in range(max_iter):
        x_inter = 0.5*(x_left + x_right)
        if f(x_inter) == 0:
            x_right = x_inter
        else:
            x_left = x_inter

        if x_right - x_left < tol:
            return x_right
    else:
        raise ValueError("No root found in positive direction.")

def binary_search_right(f, x_left, x_right, tol = 1e-10, max_iter = 100):
    for i in range(max_iter):
        x_inter = 0.5*(x_left + x_right)
        if f(x_inter) > 0:
            x_right = x_inter
        else:
            x_left = x_inter

        if x_right - x_left < tol:
            return x_left
    else:
        raise ValueError("No root found in positive direction.")