"""
Exp 6: is HIV's posterior actually improper, or merely flat to second order?

The scaled Hessian has one exactly-null direction lying entirely in theta. A null direction of the
QUADRATIC model is not by itself an improper posterior -- the density can still decay at fourth
order, or the flatness can be an artefact of the mode's location. The question decides what any
method should output, and it decides whether a reference chain can exist at all: along a genuinely
flat direction NUTS random-walks without limit and never converges, so waiting for one is futile.

Walk along the null direction and along the two worst-determined parameter directions, re-profiling
the states at each step so the walk follows the ridge rather than cutting across it, and watch the
profiled log-density.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build
from profiled2 import ProfiledPosterior2

m, ds = build("hiv")
m.map_solve(verbose=False, tol=1e-9, max_iter=300)
x0 = np.asarray(m.map_particle, np.float64)
p = m.p
H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
w, V = np.linalg.eigh(H / np.outer(d, d))
sc = max(abs(w).max(), 1e-300)
null = V[:, 0] / d                                   # back to raw coordinates
null = null / np.linalg.norm(null)
pp = ProfiledPosterior2(m, n_nodes=8, seed=0)
lp0 = pp.logp(x0[None, :p])[0][0]

print(f'smallest scaled eigenvalue: {w[0]/sc:.3e};  theta mass of that direction: '
      f'{float((V[:p,0]**2).sum()):.1%}')
print(f'\nprofiled log-density along the null direction (states re-profiled at each step)')
print(f'{"t (raw units)":>16} {"theta":>52} {"log p_hat - max":>16}')
dirs = {"null direction": null[:p] / max(np.linalg.norm(null[:p]), 1e-300)}
for j, nm in ((0, "lam axis"), (1, "rho axis")):
    e = np.zeros(p); e[j] = 1.0
    dirs[nm] = e
for nm, v in dirs.items():
    print(f'  -- {nm} --')
    ts = np.array([0.0, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5])
    TH = x0[None, :p] + ts[:, None] * v[None, :]
    lp, _, ok = pp.logp(TH)
    lp = np.where(ok, lp, -np.inf)
    for t, l, th in zip(ts, lp, TH):
        print(f'{t:>16.0e} {str(np.round(th, 4)):>52} {l - np.nanmax(lp):>16.4f}')
print(f'\nA density that stops decaying as t grows is improper: the parameter is unbounded and no')
print(f'posterior mean or variance exists for it, so no reference chain can converge either.')
