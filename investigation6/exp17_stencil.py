"""
Exp 17: is a fixed finite-difference stencil safe, and what would a dynamic one have to see?

The central second difference has error ~ (h^2/12)|f''''| from truncation and ~sqrt(6)*sigma/h^2
from round-off, so the usable range of h is bounded on both sides and both bounds are properties of
the problem: sigma comes from the working precision and f'''' from how non-Gaussian the profiled
marginal is. A constant is therefore a guess. This measures the diagonal curvature of the profiled
log-density as a function of h, per system and per precision, in units of the joint Laplace
standard deviation. A trustworthy stencil shows a PLATEAU -- a range of h over which the estimate
does not move -- and its absence, or its location, is what a dynamic rule would have to detect.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
from profiled import ProfiledPosterior

LAD = np.array([0.05, 0.2, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2])
for name in SYSTEMS:
    for dt in (jnp.float64, jnp.float32):
        m, ds = build(name, dtype=dt)
        m.map_solve(verbose=False, tol=1e-9, max_iter=300)
        x = np.asarray(m.map_particle, np.float64); p = m.p
        H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
        d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
        w, V = np.linalg.eigh(H / np.outer(d, d)); k = w > 1e-10 * abs(w).max()
        Sig = ((V[:, k] / w[k]) @ V[:, k].T) / np.outer(d, d)
        sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 1e-300))
        pp = ProfiledPosterior(m, n_nodes=8, seed=0)
        f0 = pp.logp(x[None, :p])[0][0]
        # diagonal curvature at each h, normalised by the Laplace curvature 1/sd^2 so that a
        # perfectly Gaussian profile would read 1.0 at every h
        curv = np.full((len(LAD), p), np.nan)
        for i, fr in enumerate(LAD):
            h = fr * sd
            TH = np.repeat(x[None, :p], 2 * p, axis=0)
            for j in range(p):
                TH[2 * j, j] += h[j]; TH[2 * j + 1, j] -= h[j]
            lp, _, ok = pp.logp(TH)
            for j in range(p):
                if ok[2 * j] and ok[2 * j + 1] and np.all(np.isfinite(lp[2*j:2*j+2])):
                    c = -(lp[2 * j] - 2 * f0 + lp[2 * j + 1]) / h[j] ** 2
                    curv[i, j] = c * sd[j] ** 2
        print(f'--- {name} / {dt.__name__} --- diagonal curvature x sd^2 (1.0 = Laplace)')
        print(f'{"fd_rel":>8} ' + " ".join(f'{f"th{j}":>9}' for j in range(min(p, 7))))
        for i, fr in enumerate(LAD):
            print(f'{fr:>8} ' + " ".join(f'{curv[i, j]:>9.3f}' if np.isfinite(curv[i, j])
                                         else f'{"--":>9}' for j in range(min(p, 7))))
        # plateau: relative change between consecutive rungs, maxed over parameters
        rel = np.full(len(LAD) - 1, np.nan)
        for i in range(len(LAD) - 1):
            a, b = curv[i], curv[i + 1]
            g = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
            if g.any():
                rel[i] = float(np.max(np.abs(b[g] - a[g]) / np.abs(a[g])))
        print(f'{"chg":>8} ' + " ".join(f'{r:>9.3f}' if np.isfinite(r) else f'{"--":>9}'
                                        for r in rel) + "   <- consecutive relative change")
        print()
