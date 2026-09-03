"""
exp05d: why does preconditioned SVGD diverge, when the whitening demonstrably works?

State of the evidence:
  exp05   precond (H^-1 at the MAP) + large fixed h diverges on fn and lorenz, worse than any
          other configuration measured (energy 1192x the floor, Stein R 5e5).
  exp05b  the whitening is not at fault: L^-1 Sigma_ref L^-T has cond 47 on fn and 1.6 on hiv,
          5-95% eigenvalue range 0.82-1.22. The whitened target IS isotropic in second moment.
  exp05c  an exact ANISOTROPIC Gaussian at fixed h has a perfectly stable attractor, so anisotropy
          alone does not cause divergence either.

Two candidates remain: non-Gaussianity of the real posterior, and the optimizer. Three cells,
changing one thing at a time, all preconditioned by the same L and all at the same bandwidth:

  real posterior + Prodigy         reproduce the divergence
  real posterior + SGD, small lr   if this is stable, the divergence is a step-size artefact of
                                   Prodigy in whitened coordinates, not a property of the flow
  exact N(mean_ref, cov_ref) + Prodigy, preconditioned by the SAME L
                                   the whitened target is then Gaussian AND isotropic; if this is
                                   stable while the real posterior is not, non-Gaussianity is the
                                   cause and it is the only thing left
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

NAME = sys.argv[1] if len(sys.argv) > 1 else "fn"
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 20000))
MULT = float(os.environ.get("MULT", 10))
CHECK = [int(x) for x in os.environ.get("CHECK", "1000,5000,10000,20000").split(",")]


class GaussLike:
    """Exact N(mean, cov) presented with the same interface as a MAGI instance."""
    def __init__(self, mean, cov, dt):
        d = mean.shape[0]
        self.mean = jnp.asarray(mean, dt)
        self.P = jnp.asarray(np.linalg.inv(cov + 1e-12 * np.trace(cov) / d * np.eye(d)), dt)
        self.mu = jnp.zeros((1,), dt)
        self.data = None
        self.p = 0
        self.logdensity = lambda x, data: -0.5 * (x - self.mean) @ (self.P @ (x - self.mean))
        self.gradient = jax.jit(jax.vmap(lambda x, data: -(self.P @ (x - self.mean)),
                                         in_axes=(0, None)))


m, ds = H.build(NAME)
S = H.Scorer(NAME)
d = S.mean.shape[0]
post = m.fit(verbose=False)
X0 = np.asarray(post.sample(K, unpack=False), np.float64)
x_map, L = M7.laplace_metric(m)
hstar = 2.0 * d / np.log(K)
flk = S.energy_floor_k(K)
sv_fl, ks_fl, _ = MM.floors(S, K)
gauss = GaussLike(S.mean, S.cov, m.mu.dtype)

print(f"{NAME}: d={d} K={K} precond by H^-1, h={MULT:g}*h* ({MULT*hstar:.1f}), "
      f"floors energy {flk:.4f} stiffvar {sv_fl:.3f}", flush=True)
print(f'{"cell":>34}  ' + " ".join(f'{c:>9}' for c in CHECK), flush=True)

CELLS = [("real posterior + Prodigy", m, optax.contrib.prodigy, {}),
         ("real posterior + SGD 1e-3", m, optax.sgd, {"learning_rate": 1e-3}),
         ("real posterior + SGD 1e-5", m, optax.sgd, {"learning_rate": 1e-5}),
         ("exact Gaussian + Prodigy", gauss, optax.contrib.prodigy, {}),
         ("exact Gaussian + SGD 1e-3", gauss, optax.sgd, {"learning_rate": 1e-3})]
res = {}
for lab, tgt, opt, okw in CELLS:
    t0 = time.time()
    try:
        P, _, hist = M7.run_svgd(tgt, X0, MAXIT, kernel="standard", precond=(x_map, L),
                                 bandwidth=MULT * hstar, optimizer=opt, optimizer_kwargs=okw,
                                 record_every=min(CHECK))
        hd = dict(hist)
        tr = [dict(it=c, energy=S.energy(hd[c]), stiff_var=MM.stiff_var(S, hd[c]),
                   whsd=S.mahalanobis_sd(hd[c])) for c in CHECK if c in hd]
        print(f'{lab:>34}  energy/floor ' + " ".join(f'{t["energy"]/flk:>9.2f}' for t in tr)
              + f'   ({time.time()-t0:.0f}s)', flush=True)
        print(f'{"":>34}  whsd^2       ' + " ".join(f'{t["whsd"]**2:>9.3f}' for t in tr),
              flush=True)
        print(f'{"":>34}  stiffvar     ' + " ".join(f'{t["stiff_var"]:>9.4f}' for t in tr),
              flush=True)
        res[lab] = tr
    except Exception as e:
        print(f'{lab:>34}  FAILED {type(e).__name__}: {str(e)[:80]}', flush=True)
    json.dump(res, open(f"exp05d_results_{NAME}.json", "w"), indent=1)
