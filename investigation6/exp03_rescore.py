"""
Exp 3: re-score the existing methods on correctly integrated data, separating what is identified
from what is not.

Reporting "max |theta error| = 2.5 posterior sd" on a system where nothing is identified measures
the reference chain's wandering, not the method. Errors are therefore split: over parameters the
data determines (relative posterior sd below 0.5) and over the rest, where the only meaningful
question is whether the method reproduces the reference's WIDTH rather than its location.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build, SYSTEMS
from profiled import ProfiledPosterior

REF = lambda n: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "investigation5", f"ref5_{n}.npz")
for name in SYSTEMS:
    if not os.path.exists(REF(name)):
        print(f'--- {name}: reference not built yet ---\n'); continue
    z = np.load(REF(name))
    rm, rc = z["mean"], z["cov"]
    rs = np.sqrt(np.maximum(np.diag(rc), 0))
    hm, hc = z["half_mean"], z["half_cov"]
    m, ds = build(name)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    p = m.p
    x0 = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    w, V = np.linalg.eigh(H / np.outer(d, d))
    keep = w > 1e-10 * max(abs(w).max(), 1e-300)
    Sig = ((V[:, keep] / w[keep]) @ V[:, keep].T) / np.outer(d, d)
    lap_sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
    ident = (lap_sd / np.maximum(np.abs(x0[:p]), 1e-300)) < 0.5

    post = m.fit(verbose=False, tol=1e-9, max_iter=300)
    t0 = time.time()
    pp = ProfiledPosterior(m, n_nodes=512, seed=0).adapt(rounds=3, verbose=False)
    t_pp = time.time() - t0

    def rep(lbl, mu, sd, sec):
        e = np.abs((np.asarray(mu)[:p] - rm[:p]) / np.maximum(rs[:p], 1e-300))
        se = np.abs(np.asarray(sd) / np.maximum(rs[:p], 1e-300) - 1)
        ei = e[ident].max() if ident.any() else np.nan
        si = se[ident].max() if ident.any() else np.nan
        su = se[~ident].max() if (~ident).any() else np.nan
        print(f'    {lbl:>24} {ei:>12.4f} {si:>13.2%} {su:>16.2%} {sec:>8}')

    fl_m = np.abs((hm[0][:p] - hm[1][:p]) / np.maximum(rs[:p], 1e-300))
    fl_s = np.abs(np.sqrt(np.diag(hc[0])[:p] / np.maximum(np.diag(hc[1])[:p], 1e-300)) - 1)
    print(f'--- {name}  (ref Rhat {float(z["rhat"].max()):.4f}, div {int(z["div"])}, '
          f'{int(ident.sum())}/{p} identified) ---')
    print(f'    {"estimate":>24} {"|err| ident":>12} {"|sd err| id":>13} {"|sd err| unid":>16} {"sec":>8}')
    rep("MAP", x0, lap_sd, "-")
    rep("third order", post.mu3, lap_sd, f'{sum(post.timings.values()):.1f}')
    rep("profiled", np.concatenate([pp.theta_mean, np.zeros(0)]),
        np.sqrt(np.maximum(np.diag(pp.theta_cov), 0)), f'{t_pp:.1f}')
    print(f'    {"reference half-vs-half":>24} '
          f'{(fl_m[ident].max() if ident.any() else np.nan):>12.4f} '
          f'{(fl_s[ident].max() if ident.any() else np.nan):>13.2%} '
          f'{(fl_s[~ident].max() if (~ident).any() else np.nan):>16.2%}')
    print(f'    profiled ESS {pp.ess/pp.n_nodes:.0%}, khat {pp.khat:.2f}   '
          f'gate {"apply" if post.applied else "WARN"}\n')
