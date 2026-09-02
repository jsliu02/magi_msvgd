"""
Exp 6: score every candidate against the reference, on the parameters, per system.

Candidates: the mode; the shipped third-order corrected Gaussian; and the profiled posterior.
Errors are in reference standard deviations, and the floor is the reference's own half-vs-half
agreement, so "0.1" and "at the floor" mean different things and both are shown.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map
from profiled import ProfiledPosterior

N = {"fn": 512, "hes1": 512, "lorenz": 512, "hiv": 256}
for name in SYSTEMS:
    f = f"ref5_{name}.npz"
    if not os.path.exists(f):
        print(f'--- {name}: reference not built yet ---\n'); continue
    z = np.load(f)
    rm, rc = z["mean"], z["cov"]
    rs = np.sqrt(np.maximum(np.diag(rc), 0))
    hm, hc = z["half_mean"], z["half_cov"]
    m, ds = build(name)
    scaled_map(m, tol=1e-10, max_iter=400)
    p = m.p
    x0 = np.asarray(m.map_particle, np.float64)
    post = m.fit(verbose=False)
    t0 = time.time()
    pp = ProfiledPosterior(m, n_nodes=N[name], seed=0).adapt(rounds=4, verbose=False)
    t_pp = time.time() - t0

    err = lambda v: np.abs((np.asarray(v)[:p] - rm[:p]) / np.maximum(rs[:p], 1e-300))
    sderr = lambda s: np.abs(np.asarray(s) / np.maximum(rs[:p], 1e-300) - 1)
    Sg = np.asarray(post.cov, np.float64)
    lap_sd = np.sqrt(np.maximum(np.diag(Sg)[:p], 0))
    pr_sd = np.sqrt(np.maximum(np.diag(pp.theta_cov), 0))
    fl_m = np.abs((hm[0][:p] - hm[1][:p]) / np.maximum(rs[:p], 1e-300)).max()
    fl_s = np.abs(np.sqrt(np.diag(hc[0])[:p] / np.maximum(np.diag(hc[1])[:p], 1e-300)) - 1).max()
    print(f'--- {name}  (ref: Rhat {float(z["rhat"].max()):.3f}, div {int(z["div"])}, '
          f'{int(z["ndraw"])} draws) ---')
    print(f'    {"estimate":>22} {"max |theta err| (sd)":>22} {"max |sd err|":>14} {"sec":>7}')
    print(f'    {"MAP":>22} {err(x0).max():>22.4f} {sderr(lap_sd).max():>14.2%} {"-":>7}')
    print(f'    {"third order (shipped)":>22} {err(post.mu3).max():>22.4f} '
          f'{sderr(lap_sd).max():>14.2%} {sum(post.timings.values()):>7.2f}')
    print(f'    {"profiled":>22} {err(pp.theta_mean).max():>22.4f} '
          f'{sderr(pr_sd).max():>14.2%} {t_pp:>7.2f}')
    print(f'    {"reference half-vs-half":>22} {fl_m:>22.4f} {fl_s:>14.2%}')
    print(f'    profiled ESS {pp.ess:.0f}/{pp.n_nodes} ({pp.ess/pp.n_nodes:.0%}), khat {pp.khat:.2f}'
          f'   |   fit gate: {"apply" if post.applied else "WARN"}')
    print(f'    {"":>10} ' + " ".join(f'{f"th{j}":>10}' for j in range(min(p, 6))))
    for lbl, v in [("reference", rm[:p]), ("MAP", x0[:p]), ("third order", post.mu3[:p]),
                   ("profiled", pp.theta_mean)]:
        print(f'    {lbl:>10} ' + " ".join(f'{np.asarray(v)[j]:>10.5f}' for j in range(min(p, 6))))
    print()
