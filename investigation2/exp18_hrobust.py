"""
Exp 18: how accurate must the whitening metric be?

The recipe hinges on H, and H is the expensive part. Practically it matters whether H must be
the exact Hessian at the exact MAP, or whether a sloppy version suffices -- that determines
whether the MAP solve can be truncated, whether H can be reused across nearby datasets, and how
much a low-rank approximation could get away with.

Perturbations: H evaluated away from the MAP; H from a partially-converged MAP; and H shrunk
toward its diagonal by a factor alpha (H_a = alpha*H + (1-alpha)*diag(H)), which interpolates
between the exact metric and the diagonal one that exp11 showed fails.
"""
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD

z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
Hfull = (V * ev) @ V.T
m32 = H.build_magi()

def sqrt_inv(A):
    w, U = np.linalg.eigh(0.5 * (A + A.T))
    w = np.maximum(w, 1e-8 * w.max())
    return (U / np.sqrt(w)) @ U.T, w.max() / w.min()

def run(Amat, tag, xc, iters=2500):
    Lw = jnp.asarray(Amat, jnp.float32); xm = jnp.asarray(xc, jnp.float32)
    s = MSVGD(lambda y, db: m32.logdensity(xm + Lw @ y, db), data=m32.data)
    y = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
    best = None
    for it in range(1, iters + 1):
        L2sq, h = s.pairwise_distance(y, -1)
        Kx = (1. + L2sq / h) ** -.5; Kg = (1. + L2sq / h) ** -1.5
        dx = (Kg.sum(1, keepdims=True) * y - Kg @ y) * (1. / h)
        y = y + 1e-2 * ((Kx @ s.gradient(y, m32.data) + dx) / y.shape[0])
        if not bool(jnp.all(jnp.isfinite(y))):
            print(f'{tag:>34}  DIVERGED at {it}'); return None
        if it % 100 == 0:
            X = xm + y @ Lw.T
            best = X
            if float(-jnp.sum((X - X.mean(0)) * m32.gradient(X, m32.data)) / X.size) <= 1.05:
                break
    return H.evaluate(best, m32, tag=tag)

out = []
print(H.HDR); print("-" * len(H.HDR))

# exact
A, c = sqrt_inv(Hfull); r = run(A, "H at MAP (exact)", x_map); out.append(r); H.show(r)

# H evaluated away from the MAP
jax.config.update("jax_enable_x64", True)
m64 = H.build_magi(dtype=jnp.float64)
hess = jax.jit(jax.hessian(lambda x: m64.logdensity(x, m64.data)))
rng = np.random.default_rng(0)
Sig_h0 = (V / np.sqrt(np.maximum(ev, 1e-8 * ev.max()))) @ V.T
for nsd in [1.0, 3.0]:
    xp = x_map + nsd * (rng.standard_normal(H.DIM) @ Sig_h0.T)
    Hp = -np.asarray(hess(jnp.asarray(xp)), np.float64); Hp = .5 * (Hp + Hp.T)
    rel = np.linalg.norm(Hp - Hfull) / np.linalg.norm(Hfull)
    jax.config.update("jax_enable_x64", False)
    A, c = sqrt_inv(Hp)
    r = run(A, f"H at MAP+{nsd:g}sd (rel err {rel:.3f})", x_map)
    if r: out.append(r); H.show(r)
    jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", False)

# shrink toward the diagonal
for a in [0.99, 0.9, 0.5]:
    Ha = a * Hfull + (1 - a) * np.diag(np.diag(Hfull))
    A, c = sqrt_inv(Ha)
    r = run(A, f"shrink to diag, alpha={a:g} (cond {c:.1e})", x_map)
    if r: out.append(r); H.show(r)

r = H.gold_row(); out.append(r); H.show(r)
H.save(out, "exp18_hrobust_results")
