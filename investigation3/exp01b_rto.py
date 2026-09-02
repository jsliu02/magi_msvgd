import jax, jax.numpy as jnp, numpy as np, time
jax.config.update("jax_enable_x64", True)
import harness as H
from rto2 import RTO2, ess_of
G = H.Gold()
m = H.build_magi(dtype=jnp.float64)
r = RTO2(m, np.load("laplace_cache.npz")["x_map"])
print(f"||Qb^T R(x*)|| = {r.resid_map:.2e}", flush=True)
k = 800
xi = np.random.default_rng(0).standard_normal((k, H.DIM))
t0 = time.time(); X, lw, kap, res = r.run(xi, n_it=4, chunk=200); dt = time.time() - t0
e, w = ess_of(lw)
print(f"k={k}: {dt:.0f}s | max||F-xi||={res:.2e} | log-w sd={lw.std():.3f} "
      f"range={lw.max()-lw.min():.2f} | ESS={e:.1f}/{k} ({100*e/k:.1f}%) | "
      f"kappa max={kap.max():.3f} mean={kap.mean():.3f}", flush=True)
np.savez("rto_out.npz", X=X, lw=lw, kap=kap, xi=xi)
print(); print(H.HDR); print("-" * len(H.HDR))
H.show(H.evaluate(jnp.asarray(X), m, tag="RTO unweighted"))
idx = np.random.default_rng(1).choice(k, k, replace=True, p=w)
H.show(H.evaluate(jnp.asarray(X[idx]), m, tag="RTO weighted (resampled)"))
H.show(H.gold_row())
for tag, A in [("unweighted", X), ("weighted", X[idx])]:
    pr = ((A - G.mean) @ G.evecs).var(0) / G.evals
    print(f'  varwtd {tag:>10} = {float(np.sum(pr*G.evals)/np.sum(G.evals)):.3f}')
