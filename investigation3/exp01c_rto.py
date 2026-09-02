"""RTO with enough Gauss-Newton iterations that every solve converges, for an honest ESS."""
import jax, jax.numpy as jnp, numpy as np, time
jax.config.update("jax_enable_x64", True)
import harness as H
from rto2 import RTO2, ess_of
G = H.Gold(); m = H.build_magi(dtype=jnp.float64)
r = RTO2(m, np.load("laplace_cache.npz")["x_map"])
k = 400
xi = np.random.default_rng(0).standard_normal((k, H.DIM))
X, lw, kap, res = r.run(xi, n_it=10, chunk=200)
# per-particle convergence, so non-converged solves can be excluded rather than poison the ESS
F = np.linalg.norm(np.asarray(r.Rv(jnp.asarray(X)) @ r.Q) - xi, axis=1)
ok = F < 1e-6
print(f"k={k}, 10 GN iterations: converged {ok.sum()}/{k}   max residual {F.max():.2e}")
e_all, _ = ess_of(lw); e_ok, w_ok = ess_of(lw[ok])
print(f"  log-w sd  all={lw.std():.3f}   converged-only={lw[ok].std():.3f}")
print(f"  ESS       all={e_all:.1f}/{k}   converged-only={e_ok:.1f}/{ok.sum()} "
      f"({100*e_ok/max(ok.sum(),1):.1f}%)")
print(f"  kappa     max={kap[ok].max():.3f}  mean={kap[ok].mean():.3f}   (condition (C) needs <1)")
print(f"  fraction of converged samples with kappa<1: {np.mean(kap[ok]<1):.1%}")
print(); print(H.HDR); print("-"*len(H.HDR))
H.show(H.evaluate(jnp.asarray(X[ok]), m, tag="RTO unweighted (converged)"))
idx = np.random.default_rng(1).choice(int(ok.sum()), int(ok.sum()), replace=True, p=w_ok)
H.show(H.evaluate(jnp.asarray(X[ok][idx]), m, tag="RTO weighted"))
H.show(H.gold_row())
