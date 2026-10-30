import numpy as np
import matplotlib.pyplot as plt
from module_vfp import integ_vfp, init_vfp, init_WF
t1 = 0.0
t2 = 3.18 #63.118 #pertiod 3.18
alpha_unfold = 1e-6
I = 8.044e-5*2.
nt = 1000
#Define dt instead of giving ntint. Keep in mind the CFL condition.
ntint=10
n_q = 200
n_p = 200
Lq = 20.
Lp = 20.
grand_L = 40.
rms_gauss_p = 1.2
rms_gauss_q = 1.2
#Initialize the VFP distribution using the init_vfp function
init_WF(n_q, Lq, grand_L)

U_init = init_vfp(n_q, n_p, Lq, Lp, rms_gauss_q, rms_gauss_p)

U0 = U_init.copy()
# sol = integ_vfp(U0, t1, t2 ,I, nt, ntint, n_q, n_p, Lq, Lp, grand_L)

dq = Lq/n_q
dp = Lp/n_p
w = np.ones(n_q)
w[[0,-1]] = 0.5
H = dp*dq*np.kron(w,w)

sol = integ_vfp(U0,H, t1, t2 ,I, alpha_unfold, nt, ntint, n_q, n_p, Lq, Lp, grand_L)
mass = H @ sol.y
print("Mass using trapzoid rule", mass)
print("Mass from the solver ", sol.mass[0], sol.mass[-1])
# v = np.random.default_rng(0).random(n_q * n_p)
# direction = U_init * v
# epsilons = np.logspace(-1, -10, 10)
# MV = []

# for epsilon in epsilons:
#     U_pert = U_init + epsilon * direction
#     sol_pert = integ_vfp(U_pert.copy(), t1, t2, nt, ntint, n_q, n_p, Lq, Lp, grand_L)
#     MV.append((sol_pert.y - sol.y) / epsilon)

# MV = np.asarray(MV)

# errors = np.linalg.norm(np.diff(MV, axis=0), ord=np.inf, axis=1)

# scale = max(np.linalg.norm(MV[-1], ord=np.inf), np.finfo(float).tiny)

# relative_errors = errors / scale

# relative_errors = np.maximum(relative_errors, np.finfo(float).tiny)

# rates = np.log(relative_errors[:-1] / relative_errors[1:]) / np.log(
#     epsilons[:-2] / epsilons[1:-1]
# )

# plt.figure(figsize=(22/2.54, 16/2.54), dpi=200)
# plt.loglog(epsilons[:-1], relative_errors, marker="o", label="Successive errors")
# plt.loglog(
#     epsilons[:-1],
#     relative_errors[0] * epsilons[:-1] / epsilons[0],
#     "--",
#     label="Slop P1",
# )
# plt.xlabel(r"$\varepsilon$")
# plt.ylabel(r"relative $\|Mv_\varepsilon-Mv_{\varepsilon/10}\|_\infty$")
# plt.title("Mv computation using first-order FD: Convergence test")
# plt.grid(True, which="both")
# plt.legend()
# plt.savefig("convergence_plot.png")
# plt.show()
# print("Observed convergence rates:", rates)