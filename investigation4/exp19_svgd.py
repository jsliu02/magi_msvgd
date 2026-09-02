"""
Exp 19: does mSVGD still have a role now that a faithful Gaussian is available in 0.5 s?

Investigations 1-3 tried to make mSVGD represent this posterior faithfully and found anisotropic
collapse that no kernel variant fully cured. The situation has changed: the corrected Gaussian is
at 1.2x the energy-distance floor at baseline. So the question is no longer whether SVGD can get
there on its own, but whether, STARTED there, it moves toward the target or away from it. Started
from a correct answer, a faithful sampler should stay put.

Both kernels are tested. Iterations are fixed (atol = rtol = 0) because a shared absolute tolerance
otherwise reports false convergence after a couple of steps.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time, optax
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache

d = H.DIM
name = "baseline"
z = np.load(f"ref4_{name}.npz"); D = np.load(f"determ_{name}.npz")
m, x_map, Hs, Sig, L = cache(name)
rng = np.random.default_rng(0)
Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
Wi = np.linalg.inv(np.linalg.cholesky(z["cov"] + 1e-12 * np.trace(z["cov"]) / d * np.eye(d)))
wh = lambda X: (np.asarray(X, np.float64) - z["mean"]) @ Wi.T
def ed(X, Y, n=1500):
    r = np.random.default_rng(1)
    X = X[r.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[r.choice(len(Y), min(n, len(Y)), replace=False)]
    md = lambda A, B: np.sqrt(np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2*A@B.T, 0)).mean()
    return 2 * md(X, Y) - md(X, X) - md(Y, Y)

K = 400
mu = 0.5 * (D["mu3"] + D["muvi"])
X0 = mu[None, :] + rng.standard_normal((K, d)) @ Ch.T
sub = z["sub"]
fl = ed(wh(sub[:2000]), wh(sub[2000:]))
print(f'{"start / after mSVGD":>34} {"energy":>9} {"sd ratio":>9} {"Stein R":>9} {"sec":>7}')
print(f'{"corrected Gaussian (start)":>34} {ed(wh(X0), wh(sub)):>9.4f} '
      f'{np.median(X0.std(0)/np.sqrt(np.diag(z["cov"]))):>9.4f} {"":>9}')

for rw in (False, True):
    for iters in (200, 1000):
        m.particles = jnp.asarray(X0, m.mu.dtype)
        t0 = time.time()
        try:
            m.solve(k=K, atol=0.0, rtol=0.0, max_iter=iters, monitor_convergence=-1,
                    optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                    reweighted_kernel=rw, random_seed=0)
            P = np.asarray(m.particles, np.float64)
            g = np.asarray(m.gradient(m.particles, m.data), np.float64)
            R = float(-np.sum((P - P.mean(0)) * g) / P.size)
            print(f'{f"  {'reweighted' if rw else 'standard'} kernel, {iters} iters":>34} '
                  f'{ed(wh(P), wh(sub)):>9.4f} '
                  f'{np.median(P.std(0)/np.sqrt(np.diag(z["cov"]))):>9.4f} {R:>9.4f} '
                  f'{time.time()-t0:>7.1f}')
        except Exception as e:
            print(f'{f"  {'reweighted' if rw else 'standard'} kernel, {iters} iters":>34} FAILED {type(e).__name__}')
        m.particles = None
print(f'{"reference subsample floor":>34} {fl:>9.4f} {1.0:>9.4f} {1.0:>9.4f}')
