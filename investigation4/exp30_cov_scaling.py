"""
Exp 30: as a problem becomes more non-Gaussian, does the covariance error grow as fast as the
mean error?

The theoretical expectation, from the exact identity: the leading mean shift is driven by the
CUBIC term of the potential and is O(Lambda), whereas a correction to the covariance needs either
the quartic term or the square of the cubic, both O(Lambda^2). So the mean error should dominate
across the whole range in which fitting a Gaussian is sensible at all -- and the ratio between the
two errors should widen as the problem gets harder, not narrow.

The four settings already span a factor of 200 in measured non-Gaussianity (kappa_S from 10 to
2121), so the test needs no new reference chains. Each error is reported as a MULTIPLE OF ITS OWN
FLOOR, taken from the reference's half-vs-half agreement, because the two quantities have very
different noise levels in d = 325 and comparing them raw would be meaningless.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from pipeline import metrics

d, P, I = H.DIM, 3, np.eye(H.DIM)
REF = {"baseline": "ref4_baseline.npz", "half": "ref5_half.npz",
       "noisy": "ref5_noisy.npz", "quarter": "ref5_quarter.npz"}

print(f'{"setting":>9} {"kappa_S":>9} | {"MEAN err":>9} {"floor":>7} {"xfloor":>7} | '
      f'{"COV forst":>10} {"floor":>7} {"xfloor":>7} | {"med|var-1|":>11} {"floor":>7} {"xfloor":>7}')
print("-" * 116)
rows = []
for name in ["baseline", "half", "noisy", "quarter"]:
    z = np.load(REF[name]); sc = metrics(z["mean"], z["cov"], d)
    hm, hc = z["half_mean"], z["half_cov"]
    fl = metrics(hm[1], hc[1], d)(hm[0], hc[0])
    m = build(*SETTINGS[name], dtype=jnp.float64)
    post = m.fit(n_pairs=1024, verbose=False, tol=1e-8, max_iter=200)
    Hs = np.asarray(m.hessian(post.mu_map), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T

    me = sc(post.mu_map, I)["bias"]                       # mean error of the Laplace/MAP centre
    cf = sc(post.mu_map, Sig)["forst"]                    # covariance error, affine invariant
    mv = float(np.median(np.abs(np.diag(Sig) / np.diag(z["cov"]) - 1)))
    mvf = float(np.median(np.abs(np.diag(hc[0]) / np.diag(hc[1]) - 1)))
    r = (name, float(post.certificates["kappa_S"]), me, fl["bias"], cf, fl["forst"], mv, mvf)
    rows.append(r)
    print(f'{name:>9} {r[1]:>9.1f} | {me:>9.4f} {fl["bias"]:>7.4f} {me/fl["bias"]:>7.1f} | '
          f'{cf:>10.4f} {fl["forst"]:>7.4f} {cf/fl["forst"]:>7.1f} | '
          f'{mv:>11.4f} {mvf:>7.4f} {mv/mvf:>7.1f}')

print(f'\nratio of (mean error / its floor) to (covariance error / its floor):')
for nm, k, me, mef, cf, cff, mv, mvf in rows:
    print(f'  {nm:>9}  kappa_S {k:>8.1f}   affine-invariant {(me/mef)/(cf/cff):>7.2f}x   '
          f'marginal-variance {(me/mef)/(mv/mvf):>7.2f}x')

print(f'\nwhat a user reads: largest |theta| error vs largest |theta| sd error, Laplace')
for name in ["baseline", "half", "noisy", "quarter"]:
    z = np.load(REF[name]); rm, rs = z["mean"], np.sqrt(np.diag(z["cov"]))
    hm, hc = z["half_mean"], z["half_cov"]
    m = build(*SETTINGS[name], dtype=jnp.float64)
    post = m.fit(n_pairs=256, verbose=False, tol=1e-8, max_iter=200)
    Hs = np.asarray(m.hessian(post.mu_map), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T
    em = np.abs((post.mu_map[:P] - rm[:P]) / rs[:P]).max()
    e3 = np.abs((post.mu3[:P] - rm[:P]) / rs[:P]).max()
    es = np.abs(np.sqrt(np.diag(Sig)[:P]) / rs[:P] - 1).max()
    esf = np.abs(np.sqrt(np.diag(hc[0])[:P] / np.diag(hc[1])[:P]) - 1).max()
    print(f'  {name:>9}: mean err {em:>6.3f} sd (mu3 {e3:>6.3f})   sd err {100*es:>6.1f}%   '
          f'(gold-half sd err {100*esf:>5.1f}%)')
