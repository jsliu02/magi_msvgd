"""Baseline benchmark: full fit() per system, dtype and device, cold and warm compile cache."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time, argparse
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
from setup6 import build, SYSTEMS

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu"); ap.add_argument("--systems", default="fn,hes1,hiv,lorenz")
ap.add_argument("--dtypes", default="f32,f64"); ap.add_argument("--tag", default="")
a = ap.parse_args()
DT = {"f32": jnp.float32, "f64": jnp.float64}
dev = jax.devices(a.dev)[0]
print(f'# device {dev} {a.tag}')
print(f'{"system":>7} {"dt":>4} {"fit":>7} {"map":>6} {"hess":>6} {"setup":>6} {"mode":>6} '
      f'{"nodes":>6} {"ESS":>6} {"gate":>5} {"maxsd":>8}')
for name in a.systems.split(","):
    for tag in a.dtypes.split(","):
        m, ds = build(name, dtype=DT[tag], device=dev)
        t0 = time.time(); post = m.fit(verbose=False); el = time.time() - t0
        t = post.timings
        th = np.asarray(post.mean)[:m.p]
        sd = np.sqrt(np.maximum(np.diag(np.asarray(post.theta_cov, np.float64)), 0))
        print(f'{name:>7} {tag:>4} {el:>7.1f} ' +
              " ".join(f'{t.get(k, float("nan")):>6.1f}' for k in ("map","hessian","setup","mode","nodes")) +
              f' {post.diagnostics["ess"]/post.diagnostics["n_nodes"]:>5.1%} '
              f'{"OK" if post.reliable else "FALL":>5} {np.max(sd/np.maximum(np.abs(th),1e-30)):>8.4f}', flush=True)
