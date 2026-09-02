import jax, jax.numpy as jnp, numpy as np, time, sys
jax.config.update("jax_enable_x64", True)
import harness as H
from rto2 import RTO2, ess_of
m = H.build_magi(dtype=jnp.float64)
t0 = time.time(); r = RTO2(m, np.load("laplace_cache.npz")["x_map"])
print(f"setup {time.time()-t0:.1f}s  ||Qb^T R(x*)||={r.resid_map:.2e}", flush=True)
xi = np.random.default_rng(0).standard_normal((20, H.DIM))
t0 = time.time(); X, lw, kap, res = r.run(xi, n_it=4, chunk=20)
print(f"k=20 n_it=4: {time.time()-t0:.1f}s incl compile | max||F-xi||={res:.3e} | "
      f"log-w sd={lw.std():.3f} | kappa_max={kap.max():.3f}", flush=True)
t0 = time.time(); r.run(xi, n_it=4, chunk=20)
print(f"   compiled re-run: {time.time()-t0:.2f}s", flush=True)
