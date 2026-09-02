"""
Exp 15: marginal SHAPE fidelity -- the case for particles over a Gaussian resample.

The gold standard has real, highly significant marginal skew (theta_b skew -0.219, 23 SE from
zero; max over 325 marginals 0.811). Energy distance in Mahalanobis geometry aggregates this to
something small, so exp02's "Gaussian to sampling resolution" is a statement about the joint in
that metric -- NOT a claim that individual marginals are Gaussian.

A Gaussian resample (gauss-hybrid) cannot reproduce skew by construction. A particle method can.
This measures whether the whitened-IMQ ensemble actually does, using 1-D Wasserstein distance
and skew on the theta marginals -- the quantities a MAGI user reports.
"""
import numpy as np, jax, jax.numpy as jnp, optax
from scipy import stats
import harness as H
from msvgd import MSVGD

G = H.Gold()
z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
evc = np.maximum(ev, 1e-8 * ev.max())
Sig_h = (V / np.sqrt(evc)) @ V.T
Lw = jnp.asarray(Sig_h, jnp.float32); xm = jnp.asarray(x_map, jnp.float32)
m = H.build_magi()
s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)
rng = np.random.default_rng(0)

@jax.jit
def step(y):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1. + L2sq / h) ** -.5; Kg = (1. + L2sq / h) ** -1.5
    dx = (Kg.sum(1, keepdims=True) * y - Kg @ y) * (1. / h)
    return y + 1e-2 * ((Kx @ s.gradient(y, m.data) + dx) / y.shape[0])

ens = {}
y = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
for it in range(1, 3001):
    y = step(y)
    if it % 50 == 0:
        X = xm + y @ Lw.T
        if float(-jnp.sum((X - X.mean(0)) * m.gradient(X, m.data)) / X.size) <= 1.0:
            break
ens["whitened IMQ"] = np.asarray(xm + y @ Lw.T, np.float64)

H.patch_split()
mm = H.build_magi()
mm.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
         optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=0,
         monitor_convergence=-1, reweighted_kernel=False)
mu = np.asarray(mm.particles, np.float64).mean(0)
ens["gauss-hybrid"] = mu[None, :] + rng.standard_normal((800, H.DIM)) @ Sig_h.T
mm2 = H.build_magi()
mm2.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
          optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=0,
          monitor_convergence=-1, reweighted_kernel=True)
ens["x-space reweighted"] = np.asarray(mm2.particles, np.float64)
ens["NUTS k=800 (floor)"] = G.pos[rng.choice(len(G.pos), 800, replace=False)]

names = ["a", "b", "c"]
print(f'{"method":>22} | ' + " | ".join(f'{n:>24}' for n in names))
print(f'{"":>22} | ' + " | ".join(f'{"W1(x1e3)  skew   sd%":>24}' for _ in names))
print("-" * 100)
print(f'{"NUTS 64k reference":>22} | ' + " | ".join(
    f'{"":>10}{stats.skew(G.pos[:,j]):>7.3f}{100.0:>7.0f}' for j in range(3)))
for tag, P in ens.items():
    cells = []
    for j in range(3):
        w1 = stats.wasserstein_distance(P[:, j], G.pos[:, j]) / G.sd[j] * 1e3
        cells.append(f'{w1:>10.1f}{stats.skew(P[:,j]):>7.3f}{100*P[:,j].std()/G.sd[j]:>7.0f}')
    print(f'{tag:>22} | ' + " | ".join(cells))
print("\nW1 is 1-D Wasserstein in units of 1e-3 posterior sd; lower is better.")
np.savez("exp15_ensembles.npz", **{k.replace(" ", "_"): v for k, v in ens.items()})
