"""
exp04: the control. Is the drift a property of MAGI's posterior, or of SVGD in d ~ 300?

Everything in exp01 could be blamed on MAGI's posterior being nasty -- non-Gaussian tails,
a stiff GP block, whatever. This removes the posterior entirely and keeps only the dimension and
the covariance: the target is an exact multivariate Gaussian N(ref_mean, ref_cov) built from the
reference itself, so the score is analytic, the gradient is exact, and the correct answer is
known to machine precision. The ensemble starts at exact draws from that Gaussian.

If SVGD moves away from exact draws HERE, the failure is SVGD's, not MAGI's, and no amount of
work on the MAGI side can fix it. If it stays here but moves in exp01, the posterior's
non-Gaussianity is the culprit and that is a different (and more hopeful) problem.

The Gaussian is also run in whitened coordinates (cov = I), which is the best case a
preconditioner could ever deliver -- an isotropic target of the same dimension.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness7 as H
import msvgd7 as M7

SYS = sys.argv[1:] or list(H.USABLE)
MAXIT = int(os.environ.get("MAXIT", 1000))
KS = [int(x) for x in os.environ.get("KS", "400").split(",")]
KERNELS = os.environ.get("KERNELS", "standard,reweighted,matrix").split(",")
out = {}


class GaussTarget:
    """Minimal stand-in for a MAGI instance: logdensity(x, data), gradient(P, data), mu, data."""

    def __init__(self, mean, cov, dtype=jnp.float64):
        self.mean = jnp.asarray(mean, dtype)
        d = mean.shape[0]
        self.P = jnp.asarray(np.linalg.inv(cov + 1e-12 * np.trace(cov) / d * np.eye(d)), dtype)
        self.mu = jnp.zeros((1,), dtype)
        self.data = None
        self.p = 0
        self.logdensity = lambda x, data: -0.5 * (x - self.mean) @ (self.P @ (x - self.mean))
        self.gradient = jax.jit(jax.vmap(lambda x, data: -(self.P @ (x - self.mean)),
                                         in_axes=(0, None)))


def band(cov, mean, X, nbands=5):
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-300), V[:, o]
    r = ((np.asarray(X, np.float64) - mean) @ V).var(0) / w
    return np.array([r[b].mean() for b in np.array_split(np.arange(len(w)), nbands)])


print(f'{"target / variant":>34} {"K":>5} {"energy":>8} {"e/floor":>8} {"SteinR":>8} '
      f'{"whsd":>6}   band profile (soft -> stiff)     {"sec":>7}', flush=True)
for name in SYS:
    S = H.Scorer(name)
    d = S.mean.shape[0]
    rec = {}
    for iso in (False, True):
        mean = np.zeros(d) if iso else S.mean
        cov = np.eye(d) if iso else S.cov
        g = GaussTarget(mean, cov)
        C = np.linalg.cholesky(cov + 1e-12 * np.trace(cov) / d * np.eye(d))
        lab = "isotropic N(0,I)" if iso else "N(ref_mean, ref_cov)"
        for K in KS:
            rng = np.random.default_rng(0)
            X0 = mean[None, :] + rng.standard_normal((K, d)) @ C.T
            Xr = mean[None, :] + rng.standard_normal((4000, d)) @ C.T
            Wi = np.linalg.inv(C)
            wh = lambda X: (np.asarray(X, np.float64) - mean) @ Wi.T
            sc = object.__new__(H.Scorer)
            sc.name, sc.mean, sc.cov, sc.Wi = lab, mean, cov, Wi
            sc.sub, sc.n_energy, sc.seed = Xr, 1500, 1
            sc.sd = np.sqrt(np.diag(cov))
            fl = sc.energy_floor_k(K)

            def row(l, P, dt):
                r = dict(energy=sc.energy(P), steinR=H.stein_R(g, P),
                         whsd=sc.mahalanobis_sd(P), band=band(cov, mean, P).tolist(),
                         sec=dt, floor=fl)
                print(f'{l:>34} {K:>5} {r["energy"]:>8.4f} {r["energy"]/fl:>8.2f} '
                      f'{r["steinR"]:>8.4f} {r["whsd"]:>6.3f}   '
                      + " ".join(f'{v:>6.3f}' for v in r["band"]) + f'   {dt:>7.1f}', flush=True)
                return r

            rec[f"{lab}_K{K}"] = {"start": row(f"{name} {lab} EXACT draws", X0, 0.0)}
            for kern in KERNELS:
                t0 = time.time()
                P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern,
                                      optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                rec[f"{lab}_K{K}"][kern] = row(f"  {kern}, {MAXIT} it", P, time.time() - t0)
    out[name] = rec
    json.dump(out, open("exp04_results.json", "w"), indent=1)
