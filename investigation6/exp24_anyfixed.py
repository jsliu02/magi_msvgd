"""
Exp 24: could any fixed metric sample Hes1, or is a reference only reachable another way?

HMC with a mass matrix whitens by one fixed matrix. That succeeds when the Hessians at different
points in the posterior are close to each other, and fails when they are not, regardless of which
fixed matrix is chosen. The obstruction is therefore not cond(H) at any single point but the
mutual disagreement between Hessians, cond(H_i^-1 H_j) over pairs of draws: if that is large, no
fixed metric exists that whitens both, and the question is settled without trying any of them.

Also tried: the averaged Hessian over draws, which is the cheapest alternative metric and the one
worth ruling in or out before reaching for anything harder.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS

def whiten_from(Hm, floor=1e-12):
    Hm = 0.5 * (Hm + Hm.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(Hm)), 1e-300))
    w, V = np.linalg.eigh(Hm / np.outer(d, d))
    w = np.maximum(w, floor * max(w.max(), 1.0))
    return ((V / np.sqrt(w)) @ V.T) / d[:, None]

print(f'{"system":>8} {"pairwise cond(Hi^-1 Hj)":>24} {"cond(M): mode metric":>21} '
      f'{"averaged metric":>17}')
print("-" * 76)
for name in SYSTEMS:
    m, ds = build(name); m.map_solve(verbose=False)
    x0 = np.asarray(m.map_particle, np.float64)
    H0 = np.asarray(m.hessian(x0), np.float64); H0 = 0.5 * (H0 + H0.T)
    L0 = whiten_from(H0)
    rng = np.random.default_rng(0)
    pts = [x0 + rng.standard_normal(len(x0)) @ L0.T for _ in range(6)]
    Hs = []
    for xi in pts:
        Hi = np.asarray(m.hessian(xi), np.float64)
        Hs.append(0.5 * (Hi + Hi.T))
    # mutual disagreement: generalised eigenvalues of (H_i, H_j)
    pair = []
    for i in range(len(Hs)):
        for j in range(i + 1, len(Hs)):
            Li = whiten_from(Hs[i])
            e = np.linalg.eigvalsh(Li.T @ Hs[j] @ Li)
            pair.append(np.abs(e).max() / max(np.abs(e).min(), 1e-300))
    Havg = np.mean(np.stack(Hs + [H0]), axis=0)
    Lavg = whiten_from(Havg)
    cm = lambda Lw: float(np.median([
        (lambda e: np.abs(e).max() / max(np.abs(e).min(), 1e-300))(
            np.linalg.eigvalsh(Lw.T @ Hi @ Lw)) for Hi in Hs]))
    print(f'{name:>8} {np.median(pair):>24.2e} {cm(L0):>21.2e} {cm(Lavg):>17.2e}')
print()
print('A large pairwise number means the Hessians disagree with each other, so no single mass')
print('matrix whitens them and the averaged metric cannot help either.')
