from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

import numpy as np
from solver import Solver, Solution,odeV1, inverted_odeV1
    
if __name__ == '__main__':

    
    epsilon = 1
    Q0 = 0.1

    param_set = [
                 [2, 1/4,   3/4],
                 [4, 1/6,   5/6],
                 [6, 1/8,   7/8],
                 [8, 1/10, 9/10]
                 ]
    
    # a1 = 4.5
    # a2 = 2
    # a3 = 3
    # gamma = 1/(a2 + a3-a1)
    # omega = (a2 + a3/(2*(a2+a3-a1)))
    # param_set = [[a1, gamma, omega]]


    etaf_guess = 0.3
    # etaf_guess = 0.8
    # etaf_guess = 0.8
    fstart = 1e-2


    sols:List[Solution] = []
    solvers = []
    # igs = []
    fs = []
    lam2_list = []
    for params in tqdm(param_set):
        a1, gamma, omega = params
        a2 = a3 = a1 + 1
        solver = Solver(a1, a2, a3, Q0, epsilon)
        solvers.append(solver)
        sols.append(solver.find_etaf(etaf_guess,f_start=fstart, state_space=1))
        lam2_list.append(solver.compute_lambda2(sols[-1]))
        # igs.append(solver.backward_integrator(etaf_guess, f_start=fstart))
        
        eta,x = solver.desingularized_forward_integration(sols[-1].eta_f, 
                                                          eta_start=1e-5, eta_transition=1e-2,
                                                        #   eta_start=eta_start, eta_transition=delta
                                                        )
        ee = np.linspace(0, sols[-1].eta_f, 100)
        fs.append((ee,solver.evaluate_power_series(ee, sols[-1].f0)[0]))
      
    print()
    print(lam2_list)
    print([[_.a1, _.a2, _.a3] for _ in sols])
    print([[_.gamma, _.omega] for _ in sols])
    print([_.Q0 for _ in sols])
    print([_.eta_f for _ in sols])
    print([_.f0 for _ in sols])
    print([_.Res for _ in sols])
    
    
    
    
    #########################################################################################
    # --- STATE SPACE PLOTS ---
    name = ['solution_f', 'solution_f_prime', 'solution_J']    
    for e,label in zip(range(3), ['$f(\\eta)$', '$|f^\\prime(\\eta)|$', '$J(\\eta)$']):
        fig1, ax1 = plt.subplots()
        for sol,(ee,f) in zip(sols, fs):
            if e == 1:
                l, = ax1.plot(sol.eta, abs(sol.x[e]), '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
                # l, = ax1.plot(sol.eta, abs(sol.x[e]), '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
            elif e == 2:
                l, = ax1.plot(sol.eta, -sol.x[e], '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
            else:
                l, = ax1.plot(sol.eta, sol.x[e], '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))

            # ax1.plot(ig[0], ig[1][0], '--', color = l.get_color(), linewidth = l.get_linewidth()*0.5)
            if e == 0:
                ax1.plot(ee, f,   '-.',  color = 'k', zorder = 1e7*2, linewidth = 1.25)
        
        if e == 1:
            ax1.set_yscale('log')
            ax1.set_ylim(3e-1, 1e3)
            ax1.grid(which='minor', linewidth=0.3, linestyle='--')
        elif e == 0:
            # ax1.legend(fontsize= 'small')
            ax1.set_xlim(0)
            ax1.set_ylim(0)
        ax1.set_ylabel(label)
        ax1.set_xlabel('$\\eta$')
        ax1.grid()
        ax1.set_box_aspect(1)  # force square axes regardless of figsize
        fig1.savefig(f'{name[e]}.pdf', bbox_inches='tight', pad_inches=0.02)

        
        fig1.tight_layout()    


      
    #########################################################################################
    # RESIDUAL PLOTS

    

    # iterable = np.linspace(0.1, 0.4, 10)
    # res = []
    # for params in tqdm(param_set):
    #     a1, gamma, omega = params
    #     a2 = a3 = a1 + 1
    #     solver = Solver(a1, a2, a3, Q0, epsilon)
    #     solvers.append(solver)
    #     _temp = []
    #     for e in tqdm(iterable, position=1, leave=False):
    #         _e, _x = solver.backward_integrator(e, f_start=fstart)
    #         _temp.append(solver._check_integral_condition(_e,_x))
    #     res.append(_temp)


    # fig2, ax2 = plt.subplots()
    # for r,sol in zip(res, sols):
    #     l, = ax2.plot(iterable, r, '-',label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
    #     ax2.plot(sol.eta_f, sol.Res, 'x', color = l.get_color(), ms = 10, markeredgewidth = 2)
    # ax2.set_xlabel('$\\eta_f$')
    # ax2.set_ylabel('Residual')
    # ax2.grid()
    # ax2.legend()
    # fig2.tight_layout()
    # ax2.set_box_aspect(1)  # force square axes regardless of figsize
    # fig2.savefig(f'residual.pdf', bbox_inches='tight', pad_inches=0.02)


    # plt.show()
    # plt.close('all')
    

    ##########################################################################################
    # # --- STATE SPACE PLOTS ---
    # for e,label in zip(range(3), ['$f(\\eta)$', '$|f^\\prime(\\eta)|$', '$q(\\eta)$']):
    #     fig1, ax1 = plt.subplots()
    #     for sol,(ee,f) in zip(sols, fs):
    #         if e == 1:
    #             l, = ax1.plot(sol.eta, abs(sol.x[e]), '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
    #         else:
    #             l, = ax1.plot(sol.eta, sol.x[e], '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))

    #         # ax1.plot(ig[0], ig[1][0], '--', color = l.get_color(), linewidth = l.get_linewidth()*0.5)
    #         if e == 0:
    #             ax1.plot(ee, f,   '-.',  color = 'k', zorder = 1e7*2, linewidth = 1.25)
        
    #     if e == 1:
    #         ax1.set_yscale('log')
        
    #     ax1.set_ylabel(label)
    #     ax1.set_xlabel('$\\eta$')
    #     ax1.grid()
    #     ax1.legend()
    #     fig1.tight_layout()
    
    
    

    #########################################################################################
    # DERIVATIVES PLOTS 
    # fig21, ax21 = plt.subplots()
    # fig22, ax22 = plt.subplots()
    # fig23, ax23 = plt.subplots()
    
    # axs = [ax21, ax22, ax23]
    # figs = [fig21, fig22, fig23]

    # for i,sol in enumerate(sols):        
    #     y = odeV1(sol.eta, sol.x, sol.gamma, sol.omega, sol.a1, sol.a2, sol.a3, sol.epsilon)
    #     for e in range(3):
    #         ax = axs[e]
    #         l, = ax.plot(sol.eta, abs(y[e]), '-', label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a2, sol.a3))
    
    # # axs[1].legend()
    # axs[2].legend(loc='lower right', fontsize='small')

    # # # ax[0].set_ylim()
    # axs[0].set_ylim(3e-1, 5e1)
    # axs[1].set_ylim(3e-1, 1e5)
    # axs[2].set_ylim(3e-2, 1e1)

    # names = ['u1', 'u2', 'u3']
    # for e, label in zip(range(3), ['$\\left|\\frac{\\mathrm{d}u_1}{\\mathrm{d}\\eta}\\right|$',
    #                                '$\\left|\\frac{\\mathrm{d}u_2}{\\mathrm{d}\\eta}\\right|$',
    #                                '$\\left|\\frac{\\mathrm{d}u_3}{\\mathrm{d}\\eta}\\right|$']):
    #     a = axs[e]; f = figs[e]
    #     a.set_yscale('log')
    #     a.grid(which='major', linewidth=0.8)
    #     a.grid(which='minor', linewidth=0.3, linestyle='--')
    #     a.minorticks_on()
    #     a.set_ylabel(label, rotation = 0, labelpad=20, fontsize = 18)
    #     a.set_xlabel('$\\eta$')
    #     a.set_box_aspect(1)  # force square axes regardless of figsize
    #     # f.savefig(f'{names[e]}.pdf', bbox_inches='tight', pad_inches=0.02)
    

    # # plt.close('all')

    # fig21, ax21 = plt.subplots()
    # fig22, ax22 = plt.subplots()
    # fig23, ax23 = plt.subplots()
    
    # axs = [ax21, ax22, ax23]
    # figs = [fig21, fig22, fig23]
    # for c,sol in enumerate(sols):
    #     y = inverted_odeV1(sol.x[0], np.vstack((sol.eta, sol.x[1:])), sol.gamma, sol.omega, sol.a1, sol.a2, sol.a3, sol.epsilon)
    #     for e in range(3):
    #         ax = axs[e]        
    #         l, = ax.plot(sol.x[0], abs(y[e]), '-', color = f'C{c}',label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a1, sol.a3))
    
    # names = ['y1', 'y2', 'y3']
    # for e, label in zip(range(3), ['$\\left|\\frac{\\mathrm{d}\\eta}{\\mathrm{d}f}\\right|$',
    #                                '$\\left|\\frac{\\mathrm{d}g}{\\mathrm{d}f}\\right|$',
    #                                '$\\left|\\frac{\\mathrm{d}J}{\\mathrm{d}f}\\right|$']):
    #     a = axs[e]; f = figs[e]
    #     a.set_yscale('log')
    #     a.grid(which='major', linewidth=0.8)
    #     a.grid(which='minor', linewidth=0.3, linestyle='--')
    #     a.minorticks_on()
    #     a.set_ylabel(label, rotation = 0, labelpad=20, fontsize = 18)
    #     a.set_xlabel('$f$')
    #     a.set_box_aspect(1)  # force square axes regardless of figsize
        
    #     axs[0].legend(loc='lower right')

    #     axs[2].set_yscale('linear')


    #     f.savefig(f'{names[e]}.pdf', bbox_inches='tight', pad_inches=0.02)
    











    # PHYSICAL VARIABLES
    t = np.linspace(1e-1,10, 1000)
    t = np.logspace(-1,1, 1000)
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for sol in sols:
        ax2.semilogy(sol.xf(t), t, label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a1, sol.a3))
        ax3.semilogy(sol.theta(t, 0.5), t,  label = "$\\alpha_1 = {}, \\alpha_2 = {}, \\alpha_3 = {}$".format(sol.a1, sol.a1, sol.a3))
    
    for a in [ax2, ax3]:
        a.set_ylim(t[0])
        a.set_ylabel("$t$", rotation = 0)
        a.grid(which='major', linestyle='-', linewidth=0.8)
        a.grid(which='minor', linestyle='-', linewidth=0.25)
        a.set_box_aspect(1)
    ax2.legend()
    ax2.set_xlabel("$x_f$")
    ax3.set_xlabel("$\\theta$")
    
    fig2.tight_layout()
    fig2.savefig(f'time_history_front.pdf', bbox_inches='tight', pad_inches=0.02)

    fig3.tight_layout()
    fig3.savefig(f'time_history_theta.pdf', bbox_inches='tight', pad_inches=0.02)



    plt.show()