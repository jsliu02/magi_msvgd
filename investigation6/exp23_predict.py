"""
Exp 23: can the funnel be detected before sampling?

Exp 21 used draws from the reference chain, which is circular as a diagnostic -- if a reference
exists the difficulty is already known. But the quantity being measured is how far the curvature
departs from the mode's, and Laplace draws probe the same neighbourhood. If cond(M) over Laplace
draws reproduces the ordering, it is a pre-inference predictor of how hard the posterior will be to
sample, costing a handful of Hessians and no chain at all.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")

print(f'{"system":>8} {"cond(M): Laplace draws":>23} {"reference draws":>17} | '
      f'{"ref R-hat":>10} {"div %":>7} {"leapfrog/chain":>15}')
print("-" * 92)
for name in SYSTEMS:
    m, ds = build(name); m.map_solve(verbose=False)
    x0 = np.asarray(m.map_particle, np.float64)
    H0 = np.asarray(m.hessian(x0), np.float64); H0 = 0.5 * (H0 + H0.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H0)), 1e-300))
    ws, Vs = np.linalg.eigh(H0 / np.outer(d, d))
    wsc = np.maximum(ws, 1e-12 * ws.max())
    L = (Vs / np.sqrt(wsc)) @ Vs.T / d[:, None]
    rng = np.random.default_rng(0)
    def cond_over(pts):
        out = []
        for x in pts:
            Hi = np.asarray(m.hessian(np.asarray(x, np.float64)), np.float64)
            Hi = 0.5 * (Hi + Hi.T)
            e = np.linalg.eigvalsh(L.T @ Hi @ L)
            out.append(abs(e).max() / max(abs(e).min(), 1e-300))
        return float(np.median(out))
    lap = x0[None, :] + rng.standard_normal((10, len(x0))) @ L.T      # draws from N(x0, H^-1)
    c_lap = cond_over(lap)
    c_ref, rh, dv, lf = np.nan, np.nan, np.nan, "--"
    if os.path.exists(R(name)):
        z = np.load(R(name)); sub = np.asarray(z["sub"], np.float64)
        c_ref = cond_over(sub[rng.choice(len(sub), 10, replace=False)])
        rh, dv = float(z["rhat"].max()), 100 * int(z["div"]) / int(z["ndraw"])
    print(f'{name:>8} {c_lap:>23.2e} {c_ref:>17.2e} | {rh:>10.4f} {dv:>7.2f}')
print()
print('If the Laplace column tracks the reference column, the difficulty is knowable in advance')
print('from about ten extra Hessian evaluations.')
