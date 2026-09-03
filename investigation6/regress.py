"""Numerical regression harness: freeze fit()/diagnose() outputs so refactors can be checked."""
import numpy as np, jax, jax.numpy as jnp, sys, os, json, time, argparse
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
from setup6 import build, SYSTEMS

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dev", default="cpu")
a = ap.parse_args()
dev = jax.devices(a.dev)[0]
rec = {}
for name in SYSTEMS:
    for tag, dt in (("f32", jnp.float32), ("f64", jnp.float64)):
        m, ds = build(name, dtype=dt, device=dev)
        t0 = time.time(); post = m.fit(verbose=False); tfit = time.time() - t0
        t0 = time.time(); o = m.diagnose(n_starts=2, n_curv=4, verbose=False); tdiag = time.time() - t0
        rec[f"{name}/{tag}"] = dict(
            theta=np.asarray(post.mean)[:m.p].tolist(),
            theta_sd=np.sqrt(np.maximum(np.diag(np.asarray(post.theta_cov, np.float64)), 0)).tolist(),
            ess=float(post.diagnostics["ess"]), khat=float(post.diagnostics["khat"]),
            failed=int(post.diagnostics["failed"]), reliable=bool(post.reliable),
            mode_dist=float(o["mode_dist"]), cond=float(o["cond"]), cond_M=float(o["cond_M"]),
            n_null=int(o["n_null"]), fd=float(getattr(post.profiled, "fd_pick", float("nan"))),
            t_fit=tfit, t_diag=tdiag)
        r = rec[f"{name}/{tag}"]
        print(f'{name:>7} {tag} fit {tfit:6.1f}s diag {tdiag:6.1f}s  ESS {r["ess"]:6.1f} '
              f'khat {r["khat"]:6.2f} mode_dist {r["mode_dist"]:.2e} fd {r["fd"]:g}', flush=True)
json.dump(rec, open(a.out, "w"), indent=1)
print("wrote", a.out)
