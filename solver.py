from dataclasses import dataclass
from typing import Callable
from enum import Enum, auto

import numpy as np
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
from expansion_coefficients import coeffs_fq
from tqdm import tqdm

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
    fpp = ((f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma + gamma-omega)*fp)/(omega*eta) - a3*fp**2/f
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
    fpp = ((f**(a1)*fp + q)/(epsilon*f**(a2+a3)) + (a3*gamma + gamma-omega)*fp ) / (omega*eta) - a3*fp**2/f
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
    gp = -(q + f**a1*fp)/(epsilon*f**(a2))
    qp = -gamma*f + omega*eta*fp
    print(f, x, etap)
    return np.array([etap, gp*etap, qp*etap])


class ConditionViolationError(Exception):
    def __init__(self, *args):
        super().__init__(*args)
class CaseType(Enum):
    CaseA1 = auto()
    CaseA2 = auto()
    CaseA3 = auto()

def compute_a_case_A1(a1:float, a2:float, a3:float):
    if (a1 - a3 == 1) and (a2 > 1):
        return a1
    else:
        raise ConditionViolationError(f"Case A1 invalid: (a1 - a3 == 1) and (a2 > 1), got '{a1 - a3}' and '{a2}'")

def compute_a_case_A2(a1:float, a2:float, a3:float):
    if a1 - a2 - a3 < 0:
        return a1
    else:
        raise ConditionViolationError(f"Case A2 invalid: (a1 - a2 - a3 < 0), got '{a1 - a2 - a3}'")

def compute_a_case_A3(a1:float, a2:float, a3:float):
    if ((a2 - a3) == 2) and (a1 - a2 == - 1):
        return a1
    else:
        raise ConditionViolationError(f"Case A3 invalid: (a2 - a3 == 2) and (a1 - a2 == - 1), got '{a2 - a3}' and '{a1 - a2}'")

# DISPATCH TABLE
A_COMPUTERS: dict[CaseType, Callable[[float, float, float], float]] = {
    CaseType.CaseA1: compute_a_case_A1,
    CaseType.CaseA2: compute_a_case_A2,
    CaseType.CaseA3: compute_a_case_A3,
}


@dataclass
class Solution:

    a1: float
    a2: float
    a3: float
    
    omega: float
    gamma: float
    beta: float
    
    Q0: float
    epsilon:float

    eta: np.ndarray
    x: np.ndarray
    
    def __post_init__(self):
        self.eta_f = self.eta[-1]
        self.f0 = self.x[0][0]
        
        I = np.trapezoid(self.x[0], self.eta)
        self.res = I - self.Q0/(self.beta+1)

    def xf(self,t):
        return self.eta_f*t**self.omega
    
class Solver:
    ZERO_F = 1e-10

    def __init__(self, a1, a2, a3, Q0, epsilon):
        
        # Problem parameters
        self.a1:float = a1
        self.a2:float = a2
        self.a3:float = a3
        
        self.Q0:float = Q0
        
        self.epsilon:float = epsilon    
        
        
        # Self-similarity parameters
        self.gamma: float = 1/(a2 + a3 - a1)
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
    
    
    #####################################################################################
    # FORWARD INTEGRATION

    def forward_integration(self, f0, deta, invert_fraction = 0.1, state_space = 1):
        if isinstance(f0, (list, tuple, np.ndarray)):
            nf0 = len(f0)
            a = [0]*nf0
            b = [0]*nf0
            for i in tqdm(range(nf0)):
                a[i], b[i] = self.forward_integration(f0[i], deta, invert_fraction, state_space=state_space)
            return a, b
        else:
            # Compute quantities at eta = deta
            f_deta, fp_deta, q_deta = self.evaluate_power_series(deta, f0)
            x_deta = [f_deta, fp_deta, q_deta]
            if f_deta < 0:
                raise FloatingPointError("try a larger value of f(0)")
            if state_space == 1:
                self._ode = lambda eta, x: odeV1(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
                self._inverted_ode = lambda eta, x: inverted_odeV1(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
            elif state_space == 2:
                x_deta[1] = f_deta**(self.a3)*(self.gamma*f_deta - self.omega*deta*fp_deta)
                self._ode = lambda eta, x: odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
                self._inverted_ode = lambda eta, x: inverted_odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
            else:
                raise ValueError()
            
            # Normal solve
            sol = solve_ivp(self._ode, [deta, 3], x_deta, events=self.__event(f0, invert_fraction),
                            rtol = 1e-10, atol = 1e-10, first_step = 1e-6,
                            vectorized=True)
            # return sol.t, sol.y
            
            if sol.status == 1:
                inverted_sol = solve_ivp(self._inverted_ode, [sol.y[0,-1], self.ZERO_F], [sol.t[-1], sol.y[1,-1], sol.y[2,-1]],
                                         rtol = 1e-10, atol = 1e-10, first_step = 1e-8,
                                         vectorized=True)
                return np.concatenate([sol.t[:-1], inverted_sol.y[0]]), np.column_stack([sol.y[:,:-1],np.vstack([inverted_sol.t, inverted_sol.y[1:]])])
            else:
                return sol.t, sol.y

    
    
    #########################################################################
    # BACKWARD INTEGRATOR

    def find_etaf(self, etaf_guess, case:CaseType = CaseType.CaseA2, f_start = 1e-3, eta_floor = 1e-9, fp_condition = 1,
                         method = 'newton')->Solution:
        def func(eta_f):
            eta, x = self.backward_integrator(eta_f=eta_f,case=case, f_start=f_start, eta_floor=eta_floor, fp_condition=fp_condition)
            return self._check_integral_condition(eta, x)
        
        if method == 'newton':        
            if isinstance(etaf_guess, (float,int)):
                eta_f = newton(func, etaf_guess)
            else:
                raise ValueError("For 'newton' method 'f0' must be a scalar")
        elif method == 'brentq':
            if isinstance(etaf_guess, (list,tuple, np.ndarray)) and len(etaf_guess) == 2:
                eta_f = brentq(func, *etaf_guess)
            else:
                raise ValueError("For 'brentq' method 'f0' must be an interval")
        else:
            raise ValueError("'f0_span' must be a float to initialize a 'newton' root finder, or a list of length 2 to initialize a 'brentq' root finder.")
        eta, x = self.backward_integrator(eta_f=eta_f,case=case, f_start=f_start, eta_floor=eta_floor, fp_condition=fp_condition)
        return Solution(self.a1, self.a2, self.a3, self.omega, self.gamma, self.beta, self.Q0, self.epsilon, eta, x)
        

    def backward_integrator(self, eta_f, case:CaseType = CaseType.CaseA2, f_start = 1e-3, eta_floor = 1e-9, fp_condition = 1, state_space = 1):

        
        if state_space == 1:
            self._ode = lambda eta, x: odeV1(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
            self._inverted_ode = lambda eta, x: inverted_odeV1(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        # elif state_space == 2:
        #     x_deta[1] = f_deta**(self.a3)*(self.gamma*f_deta - self.omega*deta*fp_deta)
        #     self._ode = lambda eta, x: odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        #     self._inverted_ode = lambda eta, x: inverted_odeV2(eta, x, self.gamma, self.omega, self.a1, self.a2, self.a3, self.epsilon)
        else:
            raise ValueError()
            
    

        # # ASSUMING CASE 1.1 => a = a1 
        # # Check: a3 + 1 = a1
        # if (self.a3 + 1 != self.a1):
        #     raise ValueError("Condition 'a3 + 1 == a1' must be satisfied, currently {} != {}".format(self.a3+1, self.a1))
        
        # ASSUMING CASE 1.2 => a = a1
        # Check a2 + a3 > a1
        a = A_COMPUTERS[case](self.a1, self.a2, self.a3)
        
        
        b = a*self.omega*eta_f
        def f_func(eta): return b**(1/a)*(eta_f-eta)**(1/a)
        def fp_func(eta): return -b/a*f_func(eta)**(1-a)
        def fpp_func(eta): return b*(a-1)/a*f_func(eta)**(-a)*fp_func(eta)



        # STARTING CONDITIONS
        f_s = f_start
        eta_s = eta_f - f_s**(a)/b
        fp_s = fp_func(eta_s)
        fpp_s = fpp_func(eta_s)
        qs = -f_s**(self.a1)*fp_s - self.epsilon*f_s**(self.a2+self.a3)*(((self.a3+1)*self.gamma - self.omega)*fp_s\
                                                                        - self.omega*eta_s*(fpp_s + self.a3*(fp_s)**2/f_s))
        # INTEGRATE
        def event_flip(f,x):
            return abs(x[1]) > fp_condition
        event_flip.terminal = True
        
        def event_zero(f,x):
            return x[0] > eta_floor
        event_zero.terminal = True

        xs = [eta_s, fp_s, qs]
        inverted_sol = solve_ivp(self._inverted_ode, [f_s,10], xs, events=(event_flip, event_zero, ),
                                rtol = 1e-10, atol = 1e-10, first_step = 1e-8,
                                vectorized=True,)
        if inverted_sol.status == 1 and inverted_sol.y[0,-1] < eta_floor:
            return inverted_sol.y[0][::-1], np.flip(np.vstack([inverted_sol.t, inverted_sol.y[1:]]), axis = 1)
        else:
            sol = solve_ivp(self._ode, [inverted_sol.y[0,-1], eta_floor], [inverted_sol.t[-1], inverted_sol.y[1,-1], inverted_sol.y[2,-1]],
                            rtol = 1e-10, atol = 1e-10, first_step = 1e-8,
                            vectorized=True)
            return np.concatenate([inverted_sol.y[0], sol.t[1:]])[::-1],\
                    np.column_stack([np.vstack([inverted_sol.t, inverted_sol.y[1:]]), sol.y[:,1:]])[:,::-1]
            

    #########################################################################
    # ROOT FINDERS

    def _find_valid_sol(self, f0_start, eta0):
        def func(f0):
            eta, x = self.forward_integration(f0=f0, deta = eta0)
            return x[0][-1] - self.ZERO_F
        
        f0_low, f0_high = find_bracket_forward(func, f0_start)
        return f0_low, f0_high
        


    def _update_f0(self, f0_guess, eta0, step = 0.1, max_iter = 50, direction = 1):
        
        def func(f0):
            eta, x = self.forward_integration(f0=f0, deta = eta0)
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
            eta, x = self.forward_integration(f0=f0, deta = eta0)
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
        return f0, self.forward_integration(f0=f0, deta = eta0)

    ###########################################################################
    # AUXILIARY METHODS
    def __event(self, f0, invert_fraction):
        def zero_f(eta, x):
            return x[0]/f0 > invert_fraction
        zero_f.terminal = True
        zero_f.direction = -1
        
        return (zero_f,)
        

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


    def xf(self,t,eta_f):
        return eta_f*t**self.omega




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