"""Exp 1: does RTO work on MAGI, and what does it certify?"""
import numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
import harness as H
from rto import RTO, kappa_bound

G = H.Gold()
m = H.build_magi(dtype=jnp.float64)
r = RTO(m, np.load("laplace_cache.npz")["x_map"])
d = H.DIM

def ess(lw):
    w = np.exp(lw - lw.max()); w /= w.sum()
    return float(1.0 / np.sum(w ** 2)), w

print(f'{"k":>6} {"iters":>6} {"final ||F-xi||":>15} {"sec":>6}')
for k in [800]:
    for n_it in [1, 2, 3, 5, 8]:
        xi = jnp.asarray(np.random.default_rng(0).standard_normal((k, d)))
        x0 = jnp.tile(r.x_map, (k, 1))
        t0 = time.time()
        X, hist = r.solve(xi, x0, n_it); X.block_until_ready()
        print(f'{k:>6} {n_it:>6} {float(hist[-1]):>15.3e} {time.time()-t0:>6.1f}')

k, n_it = 800, 8
xi = jnp.asarray(np.random.default_rng(0).standard_normal((k, d)))
X, hist = r.solve(xi, jnp.tile(r.x_map, (k, 1)), n_it)
print(f"\nconvergence of the fixed-Jacobian solve (max over chains of ||Qb^T R - xi||):")
print("   " + "  ".join(f"{float(v):.2e}" for v in hist))

t0 = time.time()
lw, kap = r.log_weights(X, xi)
lw = np.asarray(lw); kap = np.asarray(kap)
e, w = ess(lw)
print(f"\nweights computed in {time.time()-t0:.1f}s")
print(f"  log-weight sd  = {lw.std():.4f}   range {lw.max()-lw.min():.4f}")
print(f"  ESS            = {e:.1f} / {k}   ({100*e/k:.1f}%)")
kb, dev, L2 = kappa_bound(m, r.lsq, r, np.asarray(X))
print(f"  kappa measured = {kap.max():.4f}  (max over samples)   <- condition (C) needs < 1")
print(f"  kappa bound    = {kb:.4f}   from ||x-x*||_inf = {dev:.3f} and L2 = {L2:.2f}")

print(); print(H.HDR); print("-"*len(H.HDR))
H.show(H.evaluate(X, m, tag="RTO, unweighted"))
idx = np.random.default_rng(1).choice(k, k, replace=True, p=w)
H.show(H.evaluate(X[idx], m, tag="RTO, weighted (resampled)"))
H.show(H.gold_row())
pr = ((np.asarray(X, np.float64) - G.mean) @ G.evecs).var(0) / G.evals
print(f'\n  varwtd (unweighted) = {float(np.sum(pr*G.evals)/np.sum(G.evals)):.3f}')
np.savez("exp01_rto_out.npz", X=np.asarray(X), lw=lw, kap=kap)
