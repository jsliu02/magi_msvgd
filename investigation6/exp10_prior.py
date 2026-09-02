"""
Exp 10: does a proper prior on theta fix HIV?

The paper's posterior is proportional to pi_Theta(theta) times the three MAGI terms; the
implementation omitted pi_Theta, leaving a flat improper prior. On HIV that is not academic --
lambda is unbounded, its posterior has no mean or variance, the Hessian has an exact null
direction, importance sampling collapses to ESS = 1, and a reference chain cannot converge.

A Gaussian prior centred on the user's theta_guess is tested at several strengths, measured as the
prior standard deviation as a fraction of the guess. Four things should follow if the diagnosis was
right: the null direction disappears, the ridge walk starts falling, the effective sample size
recovers, and the parameters the data already determines do not move -- a prior that shifts
delta, N and c is too strong regardless of what it does for lambda.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from magi import MAGI
import tests as T
from profiled2 import ProfiledPosterior2
import mode

NM = ["lam", "rho", "delta", "N", "c"]
GUESS = np.array([30.0, 0.1, 0.5, 1000.0, 3.0])
ds = T.HIV
data = ds.reset().dataset(seed=0, step=1e-3)
tru = np.asarray(ds.hyperparams["theta"], np.float64)

print(f'{"prior sd":>12} {"null dirs":>10} {"min eig/max":>12} {"lam fall":>10} '
      f'{"ESS":>7} {"khat":>7} | ' + " ".join(f'{n:>10}' for n in NM))
print("-" * 132)
for frac in (None, 1.0, 0.5, 0.2):
    conf = np.zeros(5) if frac is None else 1.0 / (frac * np.abs(GUESS)) ** 2
    m = MAGI(ds.ode, data, GUESS, conf,
             sigmas=np.asarray(ds.hyperparams["sigma"], np.float64))
    m.put(dtype=jnp.float64)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    x = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    w = np.linalg.eigvalsh(H / np.outer(d, d))
    sc = max(abs(w).max(), 1e-300)
    nnull = int((w < 1e-10 * sc).sum())
    pp8 = ProfiledPosterior2(m, n_nodes=8, seed=0)
    pr = mode.properness(m, pp8, x)
    pp = ProfiledPosterior2(m, n_nodes=256, seed=0).build(verbose=False)
    lbl = "flat (none)" if frac is None else f'{frac:.0%} of guess'
    print(f'{lbl:>12} {nnull:>10} {w.min()/sc:>12.2e} {pr[0]["fall"]:>10.3g} '
          f'{pp.ess/pp.n_nodes:>7.1%} {pp.khat:>7.2f} | '
          + " ".join(f'{v:>10.4g}' for v in x[:5]))
print(f'{"true":>12} {"":>10} {"":>12} {"":>10} {"":>7} {"":>7} | '
      + " ".join(f'{v:>10.4g}' for v in tru))
