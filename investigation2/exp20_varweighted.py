"""
Exp 20: re-rank every method by a VARIANCE-WEIGHTED criterion.

exp15/exp20 analysis showed the energy distance used throughout is computed in Mahalanobis-
whitened coordinates, which gives all 325 directions equal weight -- while the top 5 directions
carry 78.5% of the posterior variance. A constructed control confirmed the blindness: shrinking
the top 3 eigendirections to 0.3x leaves energy AT THE FLOOR (0.048) while collapsing theta
widths (dev 4.7 -> 35.9).

So energy alone cannot rank these methods. This re-scores the main candidates on the ratio of
ensemble variance to reference variance along the NUTS principal axes, reported per eigen-block
and as a variance-weighted mean -- the view that matters for functionals dominated by the
high-variance directions (trajectory uncertainty bands, theta intervals).
"""
import numpy as np, jax, jax.numpy as jnp, optax
from functools import partial
import harness as H
from msvgd import MSVGD

G = H.Gold(); order = np.argsort(G.evals)[::-1]
z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
evc = np.maximum(ev, 1e-8 * ev.max())
Sig_h = (V / np.sqrt(evc)) @ V.T
Lw = jnp.asarray(Sig_h, jnp.float32); xm = jnp.asarray(x_map, jnp.float32)
m = H.build_magi(); H.patch_split()
s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)
rng = np.random.default_rng(0)

@partial(jax.jit, static_argnums=2)
def ula(y, key, n):
    def b(c, _):
        y, k = c; k, sk = jax.random.split(k)
        return (y + 1e-2 * s.gradient(y, m.data)
                + jnp.sqrt(2e-2) * jax.random.normal(sk, y.shape, y.dtype), k), None
    (y, _), _ = jax.lax.scan(b, (y, key), None, length=n)
    return y

@jax.jit
def imq(y):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1. + L2sq / h) ** -.5; Kg = (1. + L2sq / h) ** -1.5
    dx = (Kg.sum(1, keepdims=True) * y - Kg @ y) * (1. / h)
    return y + 1e-2 * ((Kx @ s.gradient(y, m.data) + dx) / y.shape[0])

ens = {}
for seed in [0, 1, 2]:
    y = jax.random.normal(jax.random.key(seed), (800, H.DIM), dtype=jnp.float32)
    ens.setdefault("whitened ULA", []).append(np.asarray(xm + ula(y, jax.random.key(9 + seed), 2000) @ Lw.T, np.float64))
    y2 = jax.random.normal(jax.random.key(seed), (800, H.DIM), dtype=jnp.float32)
    for thr, name in [(1.05, "whitened IMQ @R<=1.05"), (1.00, "whitened IMQ @R<=1.0")]:
        yy = y2
        for it in range(1, 3001):
            yy = imq(yy)
            if it % 50 == 0:
                X = xm + yy @ Lw.T
                if float(-jnp.sum((X - X.mean(0)) * m.gradient(X, m.data)) / X.size) <= thr:
                    break
        ens.setdefault(name, []).append(np.asarray(xm + yy @ Lw.T, np.float64))
    mm = H.build_magi()
    mm.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
             optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=seed,
             monitor_convergence=-1, reweighted_kernel=False)
    mu = np.asarray(mm.particles, np.float64).mean(0)
    ens.setdefault("gauss-hybrid", []).append(mu[None, :] + rng.standard_normal((800, H.DIM)) @ Sig_h.T)
    ens.setdefault("NUTS k=800 (floor)", []).append(G.pos[rng.choice(len(G.pos), 800, replace=False)])

print(f'{"method":>24} {"varwtd":>7} {"top-1":>6} {"top-5":>6} {"top-20":>7} {"rest":>6} '
      f'{"sdrat":>6} {"energy":>7} {"dev":>5}')
res = {}
for k, Ps in ens.items():
    rows = []
    for P in Ps:
        proj = (P - G.mean) @ G.evecs
        r = proj.var(0) / G.evals
        vw = float(np.sum(r * G.evals) / np.sum(G.evals))
        sc = H.evaluate(jnp.asarray(P, jnp.float32), m, tag="")
        rows.append([vw, r[order[0]], np.median(r[order[:5]]), np.median(r[order[:20]]),
                     np.median(r[order[20:]]), sc["sd_ratio_med"], sc["energy"], sc["width_dev"]])
    a = np.mean(rows, 0); res[k] = a.tolist()
    print(f'{k:>24} {a[0]:>7.3f} {a[1]:>6.3f} {a[2]:>6.3f} {a[3]:>7.3f} {a[4]:>6.3f} '
          f'{a[5]:>6.3f} {a[6]:>7.3f} {a[7]:>5.1f}')
H.save(res, "exp20_varweighted_results")
