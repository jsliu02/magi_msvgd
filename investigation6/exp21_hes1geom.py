"""
Exp 21: why did Hes1 become hard for NUTS after the GP fit was corrected?

Before the fix its reference converged (R-hat 1.005) at 366k leapfrog steps per chain; after, it
takes 2.79M and does not converge (R-hat 1.76). The parameters went from wholly unidentified to
mostly identified, so the posterior is now concentrated where before it was diffuse, and a
concentrated posterior can be far harder to traverse than a flat one.

The reference samples in coordinates whitened by the exact Hessian AT THE MODE, which is exact only
if the curvature is the same everywhere. The direct test is therefore to evaluate the Hessian at
draws from the posterior and ask how well the mode's metric whitens them: if M = L' H(x) L is close
to the identity the metric is adequate, and if its spectrum wanders the geometry is varying and no
fixed metric will do. fn, hiv and lorenz act as controls.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")

print(f'{"system":>8} {"cond(H)":>10} {"cond(DHD)":>10} | '
      f'{"cond(M) at MAP":>15} {"median over draws":>18} {"worst":>10} | {"min eig(M)":>11}')
print("-" * 104)
for name in SYSTEMS:
    if not os.path.exists(R(name)):
        continue
    z = np.load(R(name))
    m, ds = build(name)
    m.map_solve(verbose=False)
    x0 = np.asarray(m.map_particle, np.float64)
    H0 = np.asarray(m.hessian(x0), np.float64); H0 = 0.5 * (H0 + H0.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H0)), 1e-300))
    ws, Vs = np.linalg.eigh(H0 / np.outer(d, d))
    wsc = np.maximum(ws, 1e-12 * ws.max())
    L = (Vs / np.sqrt(wsc)) @ Vs.T / d[:, None]          # x = x0 + L y, the reference's metric
    conds, mins = [], []
    sub = z["sub"]
    rng = np.random.default_rng(0)
    for i in rng.choice(len(sub), 12, replace=False):
        Hi = np.asarray(m.hessian(np.asarray(sub[i], np.float64)), np.float64)
        Hi = 0.5 * (Hi + Hi.T)
        M = L.T @ Hi @ L
        e = np.linalg.eigvalsh(0.5 * (M + M.T))
        conds.append(abs(e).max() / max(abs(e).min(), 1e-300)); mins.append(e.min())
    M0 = L.T @ H0 @ L
    e0 = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    print(f'{name:>8} {np.linalg.cond(H0):>10.2e} '
          f'{abs(ws).max()/max(abs(ws).min(),1e-300):>10.2e} | '
          f'{abs(e0).max()/max(abs(e0).min(),1e-300):>15.2e} {np.median(conds):>18.2e} '
          f'{np.max(conds):>10.2e} | {np.min(mins):>11.2e}')
print()
print('cond(M) = 1 would mean the mode metric whitens the target exactly. Growth away from the')
print('mode means the curvature varies over the posterior, which a fixed mass matrix cannot track;')
print('a negative min eig(M) means the density is not even log-concave where the chain is going.')
