import numpy as np
import matplotlib.pyplot as plt
from module_vfp import integ_vfp, init_vfp
t1 = 0.0
t2 = 20.
nt = 800
ntint=10
n_q = 80
n_p = 80
Lq = 20.
Lp = 20.
grand_L = 40.
rms_gauss_p = 1.2
rms_gauss_q = 1.2
#Initialize the VFP distribution using the init_vfp function
U0 = init_vfp(n_q, n_p, Lq, Lp, rms_gauss_q, rms_gauss_p)
# print("Type of the initial distribution:  ", type(U0))
# print("Length of the initial distribution:  ", len(U0))

sol = integ_vfp(U0, t1, t2 , nt, ntint, n_q, n_p, Lq, Lp, grand_L)
print("Type of the solution:  ", type(sol.y))
print("Mass ini and mass fin: ", sol.mass[0], sol.mass[-1])
print("density length", len(sol.density))

#Computing the action of the monodromy matrix over a vector v using the flow sol

v = np.zeros(n_q * n_p, dtype=np.float64)
v[0] = 1.0  #
epsilon = 1e-6
U_pert = U0 + epsilon * v
sol_pert = integ_vfp(U_pert, t1, t2 , nt, ntint, n_q, n_p, Lq, Lp, grand_L)
Mv =  (sol_pert.y - sol.y)/epsilon

print("Type of the monodromy matrix action:  ", type(Mv))
print("Size of the monodromy matrix action: ", len(Mv))

#Test of the finite difference approximation of the monodromy matrix action
Epsi = [1e-6, 1e-7, 1e-8, 1e-9]
Norm_dif = np.zeros(len(Epsi))
for k, epsilon in enumerate(Epsi):
    U_pert = U0 + epsilon * v
    sol_pert = integ_vfp(U_pert, t1, t2 , nt, ntint, n_q, n_p, Lq, Lp, grand_L)
    Mv =  (sol_pert.y - sol.y)/epsilon
    Norm_dif[k] = np.linalg.norm(epsilon * Mv)
    print(f"epsilon: {epsilon}, norm of the difference: {Norm_dif[k]}")

plt.figure()
plt.loglog(Epsi, Norm_dif, marker='o')
plt.xlabel('epsilon')
plt.ylabel('Norm of the difference')
plt.title('Convergence of the finite difference approximation of the monodromy matrix action')
plt.grid()
plt.show()
plt.savefig('convergence_plot.png')


