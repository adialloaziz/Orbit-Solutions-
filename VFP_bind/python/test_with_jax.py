import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from module_vfp import integ_vfp, init_vfp, init_WF

jax.config.update("jax_enable_x64", True)
t1 = 0.0
t2 = 3.18 #63.118 #pertiod 3.18
nt = 100
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

U = jnp.asarray(U_init, dtype=jnp.float64)
solver_args = (t1, t2, nt, ntint, n_q, n_p, Lq, Lp, grand_L)
output_spec = jax.ShapeDtypeStruct(U.shape, U.dtype)

def _solver_callback(u):
    result = integ_vfp(np.asarray(u, dtype=np.float64).copy(), *solver_args)
    return np.asarray(result.y, dtype=np.float64)

def finite_difference_jax(u, du, primal, epsilon):
    perturbed = jax.pure_callback(_solver_callback, output_spec, u + epsilon * du)
    return (perturbed - primal) / epsilon

@jax.custom_jvp
def solver_jax(u):
    return jax.pure_callback(_solver_callback, output_spec, u)

@solver_jax.defjvp
def _solver_jvp(primals, tangents):
    (u,), (du,) = primals, tangents
    primal = solver_jax(u)
    epsilon = jnp.asarray(1e-5, dtype=u.dtype)
    tangent = finite_difference_jax(u, du, primal, epsilon)
    return primal, tangent

key = jax.random.PRNGKey(0)
V = jax.random.uniform(key, shape=U.shape, dtype=U.dtype)
direction = V * U
sol_jax, Mv_jax = jax.jvp(solver_jax, (U,), (direction,))
epsilons = np.logspace(-1, -6, 6)
fd_values = [
    finite_difference_jax(U, direction, sol_jax, jnp.asarray(eps, dtype=U.dtype))
    for eps in epsilons
]
reference = fd_values[-1]
reference_norm = jnp.maximum(jnp.linalg.norm(reference), jnp.finfo(U.dtype).tiny)
errors = np.array([
    float(jax.device_get(jnp.linalg.norm(value - reference) / reference_norm))
    for value in fd_values[:-1]
])
errors = np.maximum(errors, np.finfo(np.float64).tiny)
rates = np.log(errors[:-1] / errors[1:]) / np.log(epsilons[:-2] / epsilons[1:-1])

print("JVP norm:", float(jax.device_get(jnp.linalg.norm(Mv_jax))))
print("Observed convergence rates:", rates)

fig, (error_ax, rate_ax) = plt.subplots(1, 2, figsize=(10, 4))
error_ax.loglog(epsilons[:-1], errors, "o-", label="relative error")
error_ax.loglog(epsilons[:-1], errors[0] * epsilons[:-1] / epsilons[0], "--", label="first order")
error_ax.set(xlabel="epsilon", ylabel="relative error", title="Finite-difference convergence")
error_ax.legend()
rate_ax.semilogx(np.sqrt(epsilons[:-2] * epsilons[1:-1]), rates, "o-")
rate_ax.axhline(1, color="black", linestyle="--", label="first order")
rate_ax.set(xlabel="epsilon", ylabel="observed order", title="Convergence rate")
rate_ax.legend()
fig.tight_layout()
plt.show()
