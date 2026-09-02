"""Exp 8: does the implicit-function warm start let the inner iteration count come down?"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build
from profiled2 import ProfiledPosterior2

z = np.load("../investigation5/ref5_fn.npz"); rm = z["mean"]; rs = np.sqrt(np.diag(z["cov"]))
hm, hc = z["half_mean"], z["half_cov"]
m, ds = build("fn"); m.map_solve(verbose=False, tol=1e-9, max_iter=300); p = m.p
print(f'{"predictor":>10} {"inner":>6} {"max|err|":>10} {"max|sd err|":>12} {"ESS":>7} {"sec":>7}')
for predict in (False, True):
    for it in (2, 3, 4, 6, 8):
        pp = ProfiledPosterior2(m, n_nodes=512, seed=0, inner_iters=it)
        if not predict:
            orig = pp.logp
            pp.logp = lambda TH, X0=None, _o=orig: _o(TH, X0, predict=False)
        t0 = time.time(); pp.build(verbose=False); dt = time.time() - t0
        e = np.abs((pp.theta_mean - rm[:p]) / rs[:p]).max()
        se = np.abs(np.sqrt(np.maximum(np.diag(pp.theta_cov), 0)) / rs[:p] - 1).max()
        print(f'{str(predict):>10} {it:>6} {e:>10.4f} {se:>12.2%} {pp.ess/pp.n_nodes:>7.1%} {dt:>7.1f}')
print(f'{"floor":>10} {"":>6} '
      f'{np.abs((hm[0][:p]-hm[1][:p])/rs[:p]).max():>10.4f} '
      f'{np.abs(np.sqrt(np.diag(hc[0])[:p]/np.diag(hc[1])[:p])-1).max():>12.2%}')
