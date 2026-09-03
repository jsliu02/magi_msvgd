"""
exp05b: why preconditioning does not remove the anisotropy.

exp05 predicted that whitening by H^-1 at the MAP would turn the target into something close to
N(0, I), where exp02d shows a fixed bandwidth has a genuine attractor. It diverges instead, worse
than any other configuration measured. The obvious candidate explanation is that the Laplace
metric does not actually whiten this posterior, so preconditioning trades one anisotropy for
another. That is directly checkable without running SVGD at all: whiten the REFERENCE covariance
by the Laplace factor and look at the spectrum. Under a successful whitening it is all ones.

  spread before   eigenvalue spread of the reference covariance itself (the anisotropy SVGD faces)
  spread after    eigenvalue spread of L^-1 Sigma_ref L^-T  (the anisotropy it faces preconditioned)

If "after" is not much better than "before", the diagnosis is confirmed and the preconditioned
route is closed for the reason given rather than by a bug.
"""
import numpy as np, sys
import harness8 as H
import msvgd8 as M7

print(f'{"system":>8} {"d":>5} {"cond(Sigma_ref)":>16} {"cond(L^-1 S L^-T)":>18} '
      f'{"5-95% before":>22} {"5-95% after":>22} {"floored eigs of H":>18}', flush=True)
for name in H.USABLE:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    m.fit(verbose=False)
    x_map, L = M7.laplace_metric(m)
    Sig = 0.5 * (S.cov + S.cov.T)
    w0 = np.linalg.eigvalsh(Sig)
    Li = np.linalg.inv(L)
    Wt = Li @ Sig @ Li.T
    w1 = np.linalg.eigvalsh(0.5 * (Wt + Wt.T))
    Hm = np.asarray(m.hessian(), np.float64); Hm = 0.5 * (Hm + Hm.T)
    dg = np.sqrt(np.maximum(np.abs(np.diag(Hm)), np.finfo(float).tiny))
    ws = np.linalg.eigvalsh(Hm / np.outer(dg, dg))
    nfloor = int((ws <= 1e-10 * max(ws.max(), 1.0)).sum())
    q0 = np.percentile(np.maximum(w0, 0), [5, 95])
    q1 = np.percentile(np.maximum(w1, 0), [5, 95])
    print(f'{name:>8} {d:>5} {w0.max()/max(w0.min(),1e-300):>16.3e} '
          f'{w1.max()/max(w1.min(),1e-300):>18.3e} '
          f'{f"{q0[0]:.2e} .. {q0[1]:.2e}":>22} {f"{q1[0]:.2e} .. {q1[1]:.2e}":>22} '
          f'{nfloor:>18}', flush=True)
