#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy as sc
def Q0_expr(f, gamma, omega, xi, C, G, epsilon):
    return -((gamma*epsilon*f[0]**(C+G)
                 + G*gamma*epsilon*f[0]**(C+G)
                 - epsilon*omega*f[0]**(C+G)
                 + f[0]**xi) * f[1])
def Q1_expr(f, gamma, omega, xi, C, G, epsilon):
    return -(f[0]**(-1+xi)*(xi*f[1]**2 + 2*f[0]*f[2])
                 + epsilon*(C*(gamma+G*gamma-omega)
                 *f[0]**(-1+C+G)*f[1]**2
                 + (gamma+G*gamma-2*omega)
                 *f[0]**(-1+C+G)
                 *(G*f[1]**2 + 2*f[0]*f[2])))


def Q2_expr(f, gamma, omega, xi, C, G, epsilon):
    return -(0.5*f[0]**(-2+xi)
                *(-xi*f[1]**3 + xi**2*f[1]**3
                  + 6*xi*f[0]*f[1]*f[2]
                  + 6*f[0]**2*f[3])
                + epsilon*(
                    0.5*C*(gamma+G*gamma-omega)
                    *f[0]**(-2+C+G)*f[1]
                    *(-f[1]**2 + C*f[1]**2 + 2*f[0]*f[2])
                    + C*(gamma+G*gamma-2*omega)
                    *f[0]**(-2+C+G)*f[1]
                    *(G*f[1]**2 + 2*f[0]*f[2])
                    + 0.5*(gamma+G*gamma-3*omega)
                    *f[0]**(-2+C+G)
                    *(-G*f[1]**3 + G**2*f[1]**3
                      + 6*G*f[0]*f[1]*f[2]
                      + 6*f[0]**2*f[3])
                ))


def Q3_expr(f, gamma, omega, xi, C, G, epsilon):
    return -(1/6)*f[0]**(-3+xi)*(
            2*xi*f[1]**4
            - 3*xi**2*f[1]**4
            + xi**3*f[1]**4
            - 12*xi*f[0]*f[1]**2*f[2]
            + 12*xi**2*f[0]*f[1]**2*f[2]
            + 12*xi*f[0]**2*f[2]**2
            + 24*xi*f[0]**2*f[1]*f[3]
            + 24*f[0]**3*f[4]
        ) - epsilon*(

            0.5*C*(gamma+G*gamma-2*omega)
            *f[0]**(-3+C+G)
            *(-f[1]**2 + C*f[1]**2 + 2*f[0]*f[2])
            *(G*f[1]**2 + 2*f[0]*f[2])

            + (1/6)*C*(gamma+G*gamma-omega)
            *f[0]**(-3+C+G)
            *f[1]*(2*f[1]**3
                     - 3*C*f[1]**3
                     + C**2*f[1]**3
                     - 6*f[0]*f[1]*f[2]
                     + 6*C*f[0]*f[1]*f[2]
                     + 6*f[0]**2*f[3])

            + 0.5*C*(gamma+G*gamma-3*omega)
            *f[0]**(-3+C+G)
            *f[1]*(-G*f[1]**3
                     + G**2*f[1]**3
                     + 6*G*f[0]*f[1]*f[2]
                     + 6*f[0]**2*f[3])

            + (1/6)*f[0]**(-3+C)*(
                2*G*gamma*f[0]**G*f[1]**4
                - G**2*gamma*f[0]**G*f[1]**4
                - 2*G**3*gamma*f[0]**G*f[1]**4
                + G**4*gamma*f[0]**G*f[1]**4
                - 8*G*omega*f[0]**G*f[1]**4
                + 12*G**2*omega*f[0]**G*f[1]**4
                - 4*G**3*omega*f[0]**G*f[1]**4
                - 12*G*gamma*f[0]**(1+G)*f[1]**2*f[2]
                + 12*G**3*gamma*f[0]**(1+G)*f[1]**2*f[2]
                + 48*G*omega*f[0]**(1+G)*f[1]**2*f[2]
                - 48*G**2*omega*f[0]**(1+G)*f[1]**2*f[2]
                + 12*G*gamma*f[0]**(2+G)*f[2]**2
                + 12*G**2*gamma*f[0]**(2+G)*f[2]**2
                - 48*G*omega*f[0]**(2+G)*f[2]**2
                + 24*G*gamma*f[0]**(2+G)*f[1]*f[3]
                + 24*G**2*gamma*f[0]**(2+G)*f[1]*f[3]
                - 96*G*omega*f[0]**(2+G)*f[1]*f[3]
                + 24*gamma*f[0]**(3+G)*f[4]
                + 24*G*gamma*f[0]**(3+G)*f[4]
                - 96*omega*f[0]**(3+G)*f[4]
            )
        )
def Q4_expr(f, gamma, omega, xi, C, G, epsilon):
    return (
        -(1/24)*f[0]**(-4+xi)*(
            -6*xi*f[1]**5
            + 11*xi**2*f[1]**5
            - 6*xi**3*f[1]**5
            + xi**4*f[1]**5
            + 40*xi*f[0]*f[1]**3*f[2]
            - 60*xi**2*f[0]*f[1]**3*f[2]
            + 20*xi**3*f[0]*f[1]**3*f[2]
            - 60*xi*f[0]**2*f[1]*f[2]**2
            + 60*xi**2*f[0]**2*f[1]*f[2]**2
            - 60*xi*f[0]**2*f[1]**2*f[3]
            + 60*xi**2*f[0]**2*f[1]**2*f[3]
            + 120*xi*f[0]**3*f[2]*f[3]
            + 120*xi*f[0]**3*f[1]*f[4]
            + 120*f[0]**4*f[5]
        )
        - epsilon * (
            (1/6)*C*(gamma + G*gamma - 2*omega)
            *f[0]**(-4+C+G)
            *(G*f[1]**2 + 2*f[0]*f[2])
            *(2*f[1]**3
              - 3*C*f[1]**3
              + C**2*f[1]**3
              - 6*f[0]*f[1]*f[2]
              + 6*C*f[0]*f[1]*f[2]
              + 6*f[0]**2*f[3])
            +
            (1/4)*C*(gamma + G*gamma - 3*omega)
            *f[0]**(-4+C+G)
            *(-f[1]**2 + C*f[1]**2 + 2*f[0]*f[2])
            *(-G*f[1]**3
              + G**2*f[1]**3
              + 6*G*f[0]*f[1]*f[2]
              + 6*f[0]**2*f[3])
            +
            (1/24)*C*(gamma + G*gamma - omega)
            *f[0]**(-4+C+G)
            *f[1]*(
                -6*f[1]**4
                + 11*C*f[1]**4
                - 6*C**2*f[1]**4
                + C**3*f[1]**4
                + 24*f[0]*f[1]**2*f[2]
                - 36*C*f[0]*f[1]**2*f[2]
                + 12*C**2*f[0]*f[1]**2*f[2]
                - 12*f[0]**2*f[2]**2
                + 12*C*f[0]**2*f[2]**2
                - 24*f[0]**2*f[1]*f[3]
                + 24*C*f[0]**2*f[1]*f[3]
                + 24*f[0]**3*f[4]
            )
            +
            (1/6)*C*f[0]**(-4+C)*f[1]*(
                2*G*gamma*f[0]**G*f[1]**4
                - G**2*gamma*f[0]**G*f[1]**4
                - 2*G**3*gamma*f[0]**G*f[1]**4
                + G**4*gamma*f[0]**G*f[1]**4
                - 8*G*omega*f[0]**G*f[1]**4
                + 12*G**2*omega*f[0]**G*f[1]**4
                - 4*G**3*omega*f[0]**G*f[1]**4
                - 12*G*gamma*f[0]**(1+G)*f[1]**2*f[2]
                + 12*G**3*gamma*f[0]**(1+G)*f[1]**2*f[2]
                + 48*G*omega*f[0]**(1+G)*f[1]**2*f[2]
                - 48*G**2*omega*f[0]**(1+G)*f[1]**2*f[2]
                + 12*G*gamma*f[0]**(2+G)*f[2]**2
                + 12*G**2*gamma*f[0]**(2+G)*f[2]**2
                - 48*G*omega*f[0]**(2+G)*f[2]**2
                + 24*G*gamma*f[0]**(2+G)*f[1]*f[3]
                + 24*G**2*gamma*f[0]**(2+G)*f[1]*f[3]
                - 96*G*omega*f[0]**(2+G)*f[1]*f[3]
                + 24*gamma*f[0]**(3+G)*f[4]
                + 24*G*gamma*f[0]**(3+G)*f[4]
                - 96*omega*f[0]**(3+G)*f[4]
            )
            +
            (1/24)*f[0]**(-4+C)*(
                -6*G*gamma*f[0]**G*f[1]**5
                + 5*G**2*gamma*f[0]**G*f[1]**5
                + 5*G**3*gamma*f[0]**G*f[1]**5
                - 5*G**4*gamma*f[0]**G*f[1]**5
                + G**5*gamma*f[0]**G*f[1]**5
                + 30*G*omega*f[0]**G*f[1]**5
                - 55*G**2*omega*f[0]**G*f[1]**5
                + 30*G**3*omega*f[0]**G*f[1]**5
                - 5*G**4*omega*f[0]**G*f[1]**5
                + 40*G*gamma*f[0]**(1+G)*f[1]**3*f[2]
                - 20*G**2*gamma*f[0]**(1+G)*f[1]**3*f[2]
                - 40*G**3*gamma*f[0]**(1+G)*f[1]**3*f[2]
                + 20*G**4*gamma*f[0]**(1+G)*f[1]**3*f[2]
                - 200*G*omega*f[0]**(1+G)*f[1]**3*f[2]
                + 300*G**2*omega*f[0]**(1+G)*f[1]**3*f[2]
                - 100*G**3*omega*f[0]**(1+G)*f[1]**3*f[2]
                - 60*G*gamma*f[0]**(2+G)*f[1]*f[2]**2
                + 60*G**3*gamma*f[0]**(2+G)*f[1]*f[2]**2
                + 300*G*omega*f[0]**(2+G)*f[1]*f[2]**2
                - 300*G**2*omega*f[0]**(2+G)*f[1]*f[2]**2
                - 60*G*gamma*f[0]**(2+G)*f[1]**2*f[3]
                + 60*G**3*gamma*f[0]**(2+G)*f[1]**2*f[3]
                + 300*G*omega*f[0]**(2+G)*f[1]**2*f[3]
                - 300*G**2*omega*f[0]**(2+G)*f[1]**2*f[3]
                + 120*G*gamma*f[0]**(3+G)*f[2]*f[3]
                + 120*G**2*gamma*f[0]**(3+G)*f[2]*f[3]
                - 600*G*omega*f[0]**(3+G)*f[2]*f[3]
                + 120*G*gamma*f[0]**(3+G)*f[1]*f[4]
                + 120*G**2*gamma*f[0]**(3+G)*f[1]*f[4]
                - 600*G*omega*f[0]**(3+G)*f[1]*f[4]
                + 120*gamma*f[0]**(4+G)*f[5]
                + 120*G*gamma*f[0]**(4+G)*f[5]
                - 600*omega*f[0]**(4+G)*f[5]
            )
        )
    )

def coeffs_fq(xi, C, G, f0, Q0, epsilon):
    "Returns column vector with coefficients of constant to 5th order expansions about eta= 0"
    f = np.zeros(6)
    q = np.zeros(6)
    f[0] = f0
    q[0] = Q0
    gamma = 1 / (C+G-xi)
    omega = (C+G)*gamma / 2
    Q_expressions = [Q0_expr,Q1_expr,Q2_expr,Q3_expr,Q4_expr]
    #Relations -(gamma -omega k )/ (i+1) f_k = q_k+1
    #Second relation Q_{k} = A + Bf_{k+1},where A and B only depend on lower order terms
    for k in range(0,5):
        q[k+1] = -(gamma - (k)* omega) * f[k] / (k+1)
        #finding slope and intercept to get f_k+1
        f[k+1] = 0
        A = Q_expressions[k](f, gamma, omega, xi, C, G, epsilon)
        f[k+1] = 1
        B = (Q_expressions[k](f, gamma, omega, xi, C, G, epsilon) - A)
        f[k+1] = (q[k] - A) / B
    return f, q

def evaluate_power_series(eta, xi, C, G, f0, q0, epsilon):
    f, q = coeffs_fq(xi, C, G, f0, q0, epsilon)
    f_poly = np.polynomial.Polynomial(f)
    q_poly = np.polynomial.Polynomial(q)
    print(f)
    print(q)
    print(f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta))
    return f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta)

#Coeffs backward direction. solved for f0,q0,Coeffs backward direction,starting
# from some f and q evaulated at delta eta
def coeffs_backwards(f0_guess,q0_guess,delt_eta,f_delt_eta,q_delt_eta,xi,C,G,epsilon):
#Returns f0,q0,and column vector for coeffs f and q. Requires an initial guess
#of f0,q0, f_delt_eta/q_delt_eta, and delt_eta values,and usual expoents and epsilon
    def residual(x):
            f0_guess,q0_guess = x
            f,q = coeffs_fq(xi, C, G, f0_guess, q0_guess, epsilon)
            f_taylor= np.sum([f[i]*delt_eta**i for i in range(6)])
            q_taylor= np.sum([q[i]*delt_eta**i for i in range(6)])
            return np.array([f_delt_eta- f_taylor,
                            q_delt_eta-q_taylor],dtype=float)
    x0 =np.array([f0_guess,q0_guess])
    sol= sc.optimize.root(residual,x0 =x0 )
    if not sol.success:
        raise RuntimeError(sol.message)
    f0_sol,q0_sol = sol.x
    #coeffs for the solution:
    f_coeffs,q_coeffs = coeffs_fq(xi,C,G,f0_sol,q0_sol,epsilon)
    return f0_sol, q0_sol,f_coeffs,q_coeffs
#Polynomial Backward direction
def evaluate_power_series_backwards(eta,f0,q0,delt_eta,f_delt_eta,q_delt_eta, xi, C, G,epsilon):
    f0,q0,f,q = coeffs_backwards(f0,q0,delt_eta,f_delt_eta,q_delt_eta,xi, C, G,epsilon)
    f_poly = np.polynomial.Polynomial(f)
    q_poly = np.polynomial.Polynomial(q)
    print(f0," calculated f0")
    print(q0,"calculated q0")
    print(f)
    print(q)
    print(f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta))
    return f_poly(eta), f_poly.deriv(1)(eta), q_poly(eta),f0,q0
