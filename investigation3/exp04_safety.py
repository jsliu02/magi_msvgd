"""
Exp 4: does a properly converged MAP change the safety picture?

investigation2 concluded that sigma=0.5 is a weakly identified regime where the Laplace metric
is unusable -- min eig(H) = 0.0078, trace(H^-1) = 133, and every sampler built on it failed. But
that H was evaluated at a MAP with gradient norm 1763, i.e. not a stationary point at all. Since
the Hessian is only meaningful AT a stationary point, that diagnosis has to be re-run.

Gauss-Newton, available for free from the least-squares form, converges these problems properly.
"""
import numpy as np, jax, jax.numpy as jnp, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from magi import MAGI
from lsq import LSQ
from jac import AnalyticJac

def build(stride, sigma):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")[::stride]
    g = np.arange(0, 20.001, 0.125)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], d[:, 0])] = d
    m = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    m.put(dtype=jnp.float64, device=jax.devices()[0])
    return m

print(f'{"setting":>26} {"||grad|| prodigy":>17} {"||grad|| GN":>12} {"min eig(H)":>12} '
      f'{"cond(H)":>10} {"tr(H^-1)":>10}')
import optax
for name, stride, sig in [("baseline s=0.2", 1, 0.2), ("half-obs", 2, 0.2),
                          ("quarter-obs", 4, 0.2), ("noisy s=0.5", 1, 0.5)]:
    m = build(stride, sig); l = LSQ(m); aj = AnalyticJac(l)
    m.particles = None
    m.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=30000, atol=1e-7, rtol=0.0,
            random_seed=0, monitor_convergence=-1, optimizer=optax.contrib.prodigy,
            optimizer_kwargs={})
    x_p = jnp.asarray(np.asarray(m.particles[0], np.float64))
    g_p = float(jnp.linalg.norm(m.gradient(x_p[None, :], m.data)))
    x = x_p
    for _ in range(60):
        x = x - jnp.linalg.lstsq(aj(x), l.residual(x), rcond=None)[0]
    g_n = float(jnp.linalg.norm(m.gradient(x[None, :], m.data)))
    Hs = np.asarray(jax.hessian(l.neglogp)(x), np.float64); Hs = .5 * (Hs + Hs.T)
    w = np.linalg.eigvalsh(Hs)
    print(f'{name:>26} {g_p:>17.3e} {g_n:>12.3e} {w.min():>12.4e} '
          f'{w.max()/max(w.min(),1e-30):>10.2e} {np.sum(1/np.maximum(w,1e-14)):>10.4e}')
    np.savez(f"map_{name.split()[0]}.npz", x_map=np.asarray(x), H=Hs)
