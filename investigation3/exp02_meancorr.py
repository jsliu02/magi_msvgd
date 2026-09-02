"""
Exp 2: the deterministic mean correction.

investigation2 established that the ENTIRE error of N(x*, H^-1) is the mean: swapping in the
true mean takes the energy distance from 1.446 to 0.048, the sampling floor. So a deterministic
estimate of the posterior mean is, on this problem, a deterministic estimate of the whole
posterior.

The classical Laplace expansion supplies one in closed form. For p ∝ exp(-U) with H = grad^2 U
at the mode,
        E[x] = x* - (1/2) H^-1 grad_x tr(H^-1 grad^2 U(x)) |_{x*} + O(higher),
derived by expanding U to cubic order and taking Gaussian expectations. Nothing is sampled.

With U = ||R||^2/2 the Hessian is grad^2 U = J^T J + sum_a R_a grad^2 R_a, and for MAGI the
second term is nonzero only on the ODE rows and only in the local (X_m, theta) block, because f
acts pointwise. So the whole object is assembled from the analytic Jacobian plus n small local
Hessians, and the outer gradient is one autodiff pass over that.
"""
import numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
import harness as H
from lsq import LSQ
from jac import AnalyticJac

G = H.Gold()
m = H.build_magi(dtype=jnp.float64)
l = LSQ(m); aj = AnalyticJac(l)
n, D, P, nD = l.n, l.D, l.P, l.nD if hasattr(l, "nD") else l.n * l.D
b = np.sqrt(l.b)

def f_local(z, t):
    return m.ode(z[:D][None, :], z[D:], t[None])[0]
hess_local = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))   # (n, D, D+P, D+P)

# global indices of the local coordinate block [X_m (D), theta (P)]
idx = np.concatenate([ (P + np.arange(n)[:, None] * D + np.arange(D)[None, :]),
                        np.broadcast_to(np.arange(P)[None, :], (n, P)) ], axis=1)   # (n, D+P)
IDX = jnp.asarray(idx)

def hess_U(x):
    J = aj(x)
    Hu = J.T @ J
    R = l.residual(x)
    R_ode = R[2 * nD:].reshape(n, D)
    c = b * jnp.einsum('nd,dmn->md', R_ode, aj.Lk)            # weight of each local Hessian
    th = x[:P]; X = x[P:P + nD].reshape(n, D)
    Z = jnp.concatenate([X, jnp.broadcast_to(th, (n, P))], axis=1)
    Hf = hess_local(Z, aj.I)                                   # (n, D, D+P, D+P)
    Sloc = jnp.einsum('md,mdij->mij', c, Hf)                   # (n, D+P, D+P)
    add = jnp.zeros_like(Hu)
    rows = IDX[:, :, None]; cols = IDX[:, None, :]
    add = add.at[rows, cols].add(Sloc)
    return Hu + add

x0 = jnp.asarray(np.load("laplace_cache.npz")["x_map"])
for _ in range(30):
    x0 = x0 - jnp.linalg.lstsq(aj(x0), l.residual(x0), rcond=None)[0]
Hn = np.asarray(hess_U(x0)); Ha = np.asarray(jax.hessian(l.neglogp)(x0))
print(f"hess_U check vs jax.hessian: relative error {np.linalg.norm(Hn-Ha)/np.linalg.norm(Ha):.3e}")

Sig = jnp.asarray(np.linalg.inv(0.5 * (Hn + Hn.T)))
g = lambda x: jnp.sum(Sig * hess_U(x))
t0 = time.time(); v = jax.grad(g)(x0); v.block_until_ready()
corr = -0.5 * (Sig @ v)
print(f"third-order mean correction computed in {time.time()-t0:.1f}s;  "
      f"||correction|| = {float(jnp.linalg.norm(corr)):.4f}")

bias = lambda mu: float(np.sqrt((G.whiten(np.asarray(mu)[None, :]) ** 2).mean()))
print(f"\n{'mean estimate':>34} {'bias':>8}")
print(f"{'MAP (Laplace centre)':>34} {bias(x0):>8.4f}")
print(f"{'MAP + deterministic correction':>34} {bias(x0 + corr):>8.4f}")
print(f"{'(reference) SAV-VI, 2.3 s':>34} {0.0513:>8.4f}")
print(f"{'(reference) mean of an SVGD run':>34} {0.0680:>8.4f}")
print(f"{'(reference) sampling floor':>34} {0.0340:>8.4f}")

rng = np.random.default_rng(0)
Sh = np.linalg.cholesky(np.asarray(Sig))
print(); print(H.HDR); print("-" * len(H.HDR))
for tag, mu in [("N(MAP, H^-1)", np.asarray(x0)),
                ("N(MAP + correction, H^-1)", np.asarray(x0 + corr))]:
    s = mu[None, :] + rng.standard_normal((800, H.DIM)) @ Sh.T
    H.show(H.evaluate(jnp.asarray(s), m, tag=tag))
H.show(H.gold_row())
np.save("mean_correction.npy", np.asarray(x0 + corr))
