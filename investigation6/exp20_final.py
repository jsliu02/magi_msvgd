"""
Exp 20: the full comparison, re-run after the GP hyperparameter fit.

Every reference-based number in investigations 5 and 6 predates that fix and was measuring a
posterior whose states were unconstrained between observations. This regenerates all of them: the
identifiability table, and the mode-versus-profiled comparison against freshly built references.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")
NM = {"fn": ["a", "b", "c"], "hes1": list("abcdefg"),
      "hiv": ["lam", "rho", "delta", "N", "c"], "lorenz": ["beta", "rho", "sigma"]}

print("=== identifiability and properness (no reference needed) ===")
print(f'{"system":>8} {"param":>7} {"MAP":>12} {"true":>12} {"sd/|MAP|":>9} {"log p fall":>11} {"verdict":>11}')
for n in SYSTEMS:
    m, ds = build(n)
    o = m.diagnose(n_starts=0, verbose=False)
    tru = np.asarray(ds.hyperparams["theta"], np.float64)
    for j in range(m.p):
        print(f'{n if j == 0 else "":>8} {NM[n][j]:>7} {o["theta"][j]:>12.5g} {tru[j]:>12.5g} '
              f'{o["rel_sd"][j]:>9.3g} {o["fall"][j]:>11.4g} {o["verdict"][j]:>11}')
    print()

print("=== posterior accuracy against the rebuilt references ===")
print(f'{"system":>8} {"Rhat":>6} {"div%":>6} | {"estimate":>10} {"max|th err|":>12} '
      f'{"max|sd err|":>12} {"traj err":>9} {"sec":>6}')
for n in SYSTEMS:
    if not os.path.exists(R(n)):
        print(f'{n:>8}  reference not available'); continue
    z = np.load(R(n)); rm = z["mean"]; rs = np.sqrt(np.maximum(np.diag(z["cov"]), 0))
    hm, hc = z["half_mean"], z["half_cov"]
    m, ds = build(n); p = m.p
    t0 = time.time(); post = m.fit(verbose=False); el = time.time() - t0
    x0 = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    w, V = np.linalg.eigh(H / np.outer(d, d)); k = w > 1e-10 * abs(w).max()
    Sig = ((V[:, k] / w[k]) @ V[:, k].T) / np.outer(d, d)
    lap = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
    I = np.asarray(m.I).ravel(); Tg = np.asarray(ds.T); sol = np.asarray(ds.solution)
    truth = sol[np.clip(np.searchsorted(Tg, I), 0, len(Tg) - 1)]
    err = lambda v: np.abs((np.asarray(v)[:p] - rm[:p]) / np.maximum(rs[:p], 1e-300)).max()
    sde = lambda v: np.abs(np.asarray(v) / np.maximum(rs[:p], 1e-300) - 1).max()
    tr = lambda v: np.linalg.norm(np.asarray(v)[p:].reshape(m.n, m.D) - truth) / np.linalg.norm(truth)
    hd = f'{n:>8} {float(z["rhat"].max()):>6.3f} {100*int(z["div"])/int(z["ndraw"]):>6.2f} | '
    print(hd + f'{"mode":>10} {err(x0):>12.4f} {sde(lap):>12.2%} {tr(x0):>9.1%} {"-":>6}')
    lbl = "profiled" if post.reliable else "Laplace*"
    sd_post = np.sqrt(np.maximum(np.diag(post.theta_cov), 0))
    print(f'{"":>8} {"":>6} {"":>6} | {lbl:>10} {err(post.mean):>12.4f} {sde(sd_post):>12.2%} '
          f'{tr(post.mean):>9.1%} {el:>6.1f}')
    flm = np.abs((hm[0][:p] - hm[1][:p]) / np.maximum(rs[:p], 1e-300)).max()
    fls = np.abs(np.sqrt(np.diag(hc[0])[:p] / np.maximum(np.diag(hc[1])[:p], 1e-300)) - 1).max()
    print(f'{"":>8} {"":>6} {"":>6} | {"floor":>10} {flm:>12.4f} {fls:>12.2%}')
    print(f'{"":>8} ESS {post.diagnostics["ess"]/post.diagnostics["n_nodes"]:.0%}, '
          f'khat {post.diagnostics["khat"]:.2f}\n')
