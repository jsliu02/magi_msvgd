"""
Exp 2: which parameters are actually identified?

Exp 1 called 205 of HIV's 608 directions flat, using a threshold on the raw Hessian's spectrum.
That is the scale-naive mistake investigation 5 was about: an eigenvalue of H has units, and HIV's
coordinates span 30 to 1e5, so "small eigenvalue" and "flat" are not the same thing. Two scale-free
readings instead.

    scaled spectrum   eigenvalues of D H D with D = diag(H)^(-1/2). Unit diagonal, so an
                      eigenvalue near zero really is a direction with no curvature relative to the
                      curvature the coordinates do have.
    relative sd       sqrt((H^-1)_jj) / |theta_j|, the marginal posterior standard deviation of
                      each parameter as a fraction of its own value. This is what a practitioner
                      reads, and it needs no threshold to interpret: 0.05 is a tight estimate and
                      20 means the data says nothing.

The second is reported against the Laplace covariance, which requires H to be invertible; where it
is not, the pseudo-inverse over the non-null subspace is used and the null dimension is reported
alongside, since that is the honest answer.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup6 import build, SYSTEMS

NAMES = {"fn": ["a", "b", "c"], "hes1": list("abcdefg"),
         "hiv": ["lam", "rho", "delta", "N", "c"], "lorenz": ["beta", "rho", "sigma"]}

for name in SYSTEMS:
    m, ds = build(name)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    x = np.asarray(m.map_particle, np.float64)
    p = m.p
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    Hs = H / np.outer(d, d)
    ws = np.linalg.eigvalsh(0.5 * (Hs + Hs.T))
    sc = max(abs(ws).max(), 1e-300)
    nflat = int((np.abs(ws) < 1e-10 * sc).sum())
    # pseudo-inverse over the non-null subspace
    w, V = np.linalg.eigh(Hs)
    keep = w > 1e-10 * sc
    Sig_s = (V[:, keep] / w[keep]) @ V[:, keep].T
    Sig = Sig_s / np.outer(d, d)
    sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
    th = x[:p]; tru = np.asarray(ds.hyperparams["theta"], np.float64)
    print(f'--- {name}  (dim {len(x)}, cond(DHD) {sc/max(abs(ws).min(),1e-300):.2e}, '
          f'{nflat} null direction(s) of {len(x)}) ---')
    print(f'    {"param":>7} {"MAP":>13} {"true":>13} {"post sd":>13} {"sd/|MAP|":>10} {"verdict":>16}')
    for j in range(p):
        r = sd[j] / max(abs(th[j]), 1e-300)
        v = "identified" if r < 0.5 else ("weak" if r < 5 else "NOT identified")
        print(f'    {NAMES[name][j]:>7} {th[j]:>13.5g} {tru[j]:>13.5g} {sd[j]:>13.5g} '
              f'{r:>10.3g} {v:>16}')
    if nflat:
        vN = V[:, ~keep]
        mass = (vN[:p] ** 2).sum(0) / np.maximum((vN ** 2).sum(0), 1e-300)
        print(f'    null directions with >50% of their mass on theta: '
              f'{int((mass > 0.5).sum())} of {nflat}')
    print()
