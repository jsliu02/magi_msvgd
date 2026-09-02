"""
Exp 2: the deterministic route, refined and self-certifying.

Two ingredients beyond the third-order Laplace correction.

(1) A REFERENCE-FREE CERTIFICATE. For any density with the usual decay, E_p[grad log p] = 0
    exactly. So for a candidate Gaussian q = N(mu, Sigma), the quantity

        delta = Sigma * E_q[grad log p]

    vanishes iff mu already satisfies the stationarity condition the true mean satisfies, and to
    leading order equals the remaining mean error. It needs no reference chain: it is both a
    refinement step and a computable measure of what is left. Reported in posterior standard
    deviations per dimension, tau = ||H^(1/2) delta|| / sqrt(d), so it is scale free.

(2) DETERMINISTIC INTEGRATION. E_q[grad log p] can be evaluated on a spherical cubature rule --
    the 2d points mu +- sqrt(d) Sigma^(1/2) e_i with equal weights -- which integrates cubics
    exactly and involves no randomness at all. Compared here against fixed random base samples
    (a sample-average approximation), which is stochastic only in the draw.
"""
import numpy as np, jax, jax.numpy as jnp, time, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from setup4 import cache
from gauss_newton import GaussNewtonMAP

G = H.Gold()
m, x_map, Hs, Sig, L = cache("baseline")
d = H.DIM
bias = lambda mu: float(np.sqrt((G.whiten(np.asarray(mu)[None, :]) ** 2).mean()))
tau = lambda dl: float(np.sqrt(np.asarray(dl) @ Hs @ np.asarray(dl) / d))
grad = jax.jit(lambda P: m.gradient(P, m.data))

# ------------------------------------------------------------------ third-order correction
gn = GaussNewtonMAP(m)
n, D, p, nD = gn.n, gn.D, gn.p, gn.nD
def f_local(z, t): return m.ode(z[:D][None, :], z[D:], t[None])[0]
hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
IDX = jnp.asarray(np.concatenate([(p + np.arange(n)[:, None]*D + np.arange(D)[None, :]),
                  np.broadcast_to(np.arange(p)[None, :], (n, p))], axis=1))
def hess_U(x):
    J = gn.jacobian(x[:p], x[p:p+nD].reshape(n, D), m.sigmas)
    Hu = J.T @ J
    c = gn.b * jnp.einsum('nd,dmn->md',
        gn.residual(x[:p], x[p:p+nD].reshape(n, D), m.sigmas)[2*nD:].reshape(n, D), gn.Lk)
    Z = jnp.concatenate([x[p:p+nD].reshape(n, D), jnp.broadcast_to(x[:p], (n, p))], axis=1)
    S = jnp.einsum('md,mdij->mij', c, hl(Z, gn.I))
    return Hu + jnp.zeros_like(Hu).at[IDX[:, :, None], IDX[:, None, :]].add(S)
Sj = jnp.asarray(Sig)
t0 = time.time()
mu3 = np.asarray(x_map) - 0.5 * np.asarray(Sj @ jax.grad(lambda z: jnp.sum(Sj * hess_U(z)))(
    jnp.asarray(x_map)))
t_corr = time.time() - t0

# ------------------------------------------------------------------ the two integration rules
Sh = np.linalg.cholesky(Sig + 1e-14 * np.eye(d))
sph = np.concatenate([np.sqrt(d) * Sh.T, -np.sqrt(d) * Sh.T], axis=0)   # 2d cubature offsets
rnd = np.random.default_rng(0).standard_normal((2 * d, d)) @ Sh.T       # fixed random offsets
def vi_step(mu, off):
    g = np.asarray(grad(jnp.asarray(mu[None, :] + off))).mean(0)
    return Sig @ g

print(f'{"mean estimate":>40} {"bias":>8} {"tau (certificate)":>18}')
print(f'{"MAP":>40} {bias(x_map):>8.4f} {tau(vi_step(x_map, sph)):>18.4f}')
print(f'{"+ third-order correction":>40} {bias(mu3):>8.4f} {tau(vi_step(mu3, sph)):>18.4f}')
for lbl, off in [("spherical cubature (deterministic)", sph), ("fixed random samples", rnd)]:
    mu = mu3.copy()
    for k in range(1, 5):
        dl = vi_step(mu, off); mu = mu + dl
        if k in (1, 2, 4):
            print(f'{f"  + VI step {k}, {lbl}":>40} {bias(mu):>8.4f} {tau(vi_step(mu, off)):>18.4f}')
    if lbl.startswith("spherical"):
        mu_final = mu
print(f'{"(sampling floor at k=800)":>40} {0.0340:>8.4f}')

rng = np.random.default_rng(1)
print(); print(H.HDR); print("-" * len(H.HDR))
for tag, mu in [("N(MAP, H^-1)", x_map), ("N(MAP+3rd order, H^-1)", mu3),
                ("N(+VI refined, H^-1)", mu_final)]:
    H.show(H.evaluate(jnp.asarray(mu[None, :] + rng.standard_normal((800, d)) @ Sh.T), m, tag=tag))
H.show(H.gold_row())
np.save("mu_determ.npy", mu_final)
print(f'\nthird-order correction took {t_corr:.2f}s; each VI step is {2*d} gradient evaluations')
