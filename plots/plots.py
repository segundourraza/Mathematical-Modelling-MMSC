import sys
import os

root_dir = os.path.abspath("")   # adjust as needed
sys.path.insert(0, root_dir)

import pickle
from tqdm import tqdm
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from solver import Solver, Solution

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({
    "mathtext.fontset": "cm",   # Computer Modern
    "font.family": "serif"
})

with open(r"plots\sol1.pkl", "rb") as f:   # binary read mode
    sols:List[Solution] = pickle.load(f)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for sol in sols:
    print(len(sol.eta))
    ax2.semilogy(sol.eta, abs(sol.x[1]), label = "$\\alpha_1 = {}, \\gamma = {}, \\omega = {}$".format(sol.a1, sol.a2, sol.a3))
    ax1.plot(sol.eta, sol.x[0], label = "$\\alpha_1 = {}, \\gamma = {}, \\omega = {}$".format(sol.a1, sol.a2, sol.a3))
    # ax1.plot(sol.eta, sol.x[0], label = "$Q_0 = {:.2e}$".format(sol.Q0))
ax1.set_xlabel('$\\eta$')
ax1.set_ylabel('$f(\\eta)$')
ax1.grid()
ax1.legend()

t = np.linspace(1e-1,10, 1000)
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
for sol in sols:
    ax2.semilogy(sol.xf(t), t,       label = "$\\alpha_1 = {}, \\gamma = {}, \\omega = {}$".format(sol.a1, sol.a2, sol.a3))
    ax3.semilogy(sol.theta(t, 1), t, label = "$\\alpha_1 = {}, \\gamma = {}, \\omega = {}$".format(sol.a1, sol.a2, sol.a3))

ax2.legend()
ax2.set_ylim(t[0])
ax2.set_xlabel("$x_f$")
ax2.set_ylabel("$t$", rotation = 0)
ax2.grid(which='major', linestyle='-', linewidth=0.8)
ax2.grid(which='minor', linestyle='-', linewidth=0.25)

fig2.tight_layout()



# T,E = np.meshgrid(t, sols[0].eta)

# interp = interp1d(sols[0].eta, sols[0].x[0], fill_value="extrapolate")
# X = E*T**sols[0].omega
# Z =T**sols[0].gamma*interp(X/T**sols[0].gamma)
# print(X/T**sols[0].gamma)

# fig, ax = plt.subplots()
# cf = ax.contourf(X, T, Z, levels = 100)
# ax.set_yscale('log')
# fig.colorbar(cf, ax=ax)

plt.show()
