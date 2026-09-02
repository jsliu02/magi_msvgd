"""
Preconditioned (matrix-valued) SVGD vs the density-reweighted kernel, scored by the
Stein-identity diagnostic R rather than by theta interval widths.

Motivation (see investigation.md sec. 4.4): at the best-calibrated reweighted setting the
ensemble matches NUTS on all 325 marginal sds (~0.84x) yet satisfies Stein's identity at only
~4% of the required level. Since R = tr(A Sigma^-1)/dim for a Gaussian, that isolates the
failure as ANISOTROPIC collapse -- the ensemble is deficient along the stiff (small-eigenvalue)
directions of Sigma, which a scalar-bandwidth RBF kernel cannot resolve. Preconditioning is the
mechanism aimed at exactly that, and the earlier matrix-SVGD comparison was scored on theta
widths, now known to be uninformative about this failure.

Matrix-valued SVGD: Wang, Tang, Bajaj & Liu, NeurIPS 2019, Eq. 12-15.

    K_Q(x,y) = Q^-1 exp(-||x-y||^2_Q / h)

The divergence term is Q^-1 grad k_Q = -2(x-y)/h k_Q, so Q CANCELS in the repulsion and
survives only as an elementwise factor on the drift. Hence _combine is reused directly with
drift = Q^-1 (broadcast over dim), and the Q-metric distances are obtained by feeding
sqrt(diag Q)-scaled particles to pairwise_distance.

Q is the diagonal empirical Fisher, NORMALIZED TO MEAN 1. An overall scale on Q cancels in the
kernel (the median-heuristic h absorbs it) but not in the drift, so normalizing makes this a
pure anisotropy correction and removes a confound with the overall step scale.

All variants run a FIXED iteration budget (atol=0), because Q^-1-rescaled gradients otherwise
trip a shared absolute tolerance and report false convergence after ~1 iteration.

Run:  CUDA_VISIBLE_DEVICES=0 python "matrix_vs_reweighted_stein_R.py"
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "..", "msvgd", "msvgd"))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from magi import MAGI

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
MAX_ITER = 1000
RIDGE = 1e-6


# ----------------------------------------------------------------------------- kernel variants
def _fisher_diag(raw_grad):
    """Diagonal empirical Fisher from the score, normalized to mean 1 (pure anisotropy)."""
    Qd = jnp.mean(raw_grad ** 2, axis=0)              # raw_grad = -s, squared so sign is moot
    Qd = Qd + RIDGE * jnp.mean(Qd)
    return Qd / jnp.mean(Qd)


def _matrix_update(self, particles, raw_grad, data_batch, h=-1, clip_exponent=20.0):
    Qd = _fisher_diag(raw_grad)
    L2sq, hh = self.pairwise_distance(particles * jnp.sqrt(Qd), h)
    K = jnp.exp(-L2sq / hh)
    return self._combine(particles, raw_grad, K, hh, drift=1.0 / Qd), jnp.float32(1.0)


def _matrix_reweighted_update(self, particles, raw_grad, data_batch, h=-1, clip_exponent=20.0):
    Qd = _fisher_diag(raw_grad)
    L2sq, hh = self.pairwise_distance(particles * jnp.sqrt(Qd), h)
    ld = jax.vmap(lambda x: self.logdensity(x, data_batch).sum())(particles)
    ld = ld - jnp.max(ld)
    reweight = jnp.exp(jnp.clip(-0.5 * (ld[:, None] + ld[None, :]), max=clip_exponent))
    K = reweight * jnp.exp(-L2sq / hh)
    return self._combine(particles, raw_grad, K, hh, drift=0.5 / Qd), jnp.mean(reweight)


_PRODUCTION_REWEIGHT = MAGI._reweighted_svgd_update

VARIANTS = {
    "standard":          (None,                      False),
    "reweighted":        (_PRODUCTION_REWEIGHT,      True),
    "matrix":            (_matrix_update,            True),
    "matrix_reweighted": (_matrix_reweighted_update, True),
}


# ----------------------------------------------------------------------------------- diagnostic
def stein_R(particles, score, cols=None):
    """-(1/(k*dim)) sum_i (x_i - xbar).s(x_i);  -> 1 under the target."""
    p, s = (particles, score) if cols is None else (particles[:, cols], score[:, cols])
    return float(-jnp.sum((p - p.mean(axis=0)) * s) / p.size)


def spectral_profile(particles, evals, evecs, n_bins=5):
    """
    Ensemble variance / NUTS eigenvalue, along the NUTS principal axes, binned softest -> stiffest.
    This decomposes R: for a Gaussian, R is exactly the mean of these ratios over all directions.
    """
    proj = np.asarray(particles) @ evecs                      # (k, dim) in the eigenbasis
    ratio = proj.var(axis=0) / evals                          # 1 = correctly dispersed
    order = np.argsort(evals)[::-1]                           # softest (largest eval) first
    return [float(np.median(ratio[b])) for b in np.array_split(order, n_bins)]


# ----------------------------------------------------------------------------------------- data
def fn_ode(X, theta, t=None):
    V, R = X.T
    a, b, c = theta
    return jnp.stack([c * (V - V ** 3 / 3 + R), -1 / c * (V - a + b * R)])


def build_solver():
    d = np.loadtxt(os.path.join(HERE, "..", "y.csv"), delimiter=",")
    grid = np.arange(0, 20.001, 0.125)
    full = np.full((grid.shape[0], d.shape[1]), np.nan)
    full[:, 0] = grid
    full[np.isin(full[:, 0], d[:, 0])] = d
    s = MAGI(fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    s.put(dtype=jnp.float32, device=jax.devices()[0])
    return s


def main():
    gold = np.load(os.path.join(HERE, "nuts_gold_standard.npz"))
    pos = gold["positions"].reshape(-1, 325)
    gold_w = gold["theta_ci_hi"] - gold["theta_ci_lo"]
    gold_sd = pos.std(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pos, rowvar=False))
    evals = np.maximum(evals, 1e-12)
    print(f"NUTS covariance condition number: {evals.max() / evals.min():.3e}\n")

    results = {}
    hdr = (f'{"variant":>18} {"width % of NUTS":>22} {"|dev|":>6} {"R_global":>9} {"R_theta":>8} '
           f'{"sd ratio":>9}  {"spectral profile soft->stiff":>34}')
    print(hdr)
    print("-" * len(hdr))

    for name, (update_fn, reweighted) in VARIANTS.items():
        if update_fn is not None:
            MAGI._reweighted_svgd_update = update_fn
        W, Rg, Rt, SD, SP, T = [], [], [], [], [], []
        for seed in SEEDS:
            m = build_solver()
            t0 = time.time()
            _, th, _ = m.solve(k=200, sigma_init=0.01, k_schedule=800,
                               optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                               atol=0.0, rtol=0.0, max_iter=MAX_ITER, random_seed=seed,
                               monitor_convergence=-1, reweighted_kernel=reweighted)
            jax.block_until_ready(th)
            T.append(time.time() - t0)
            if not bool(jnp.all(jnp.isfinite(th))):
                W.append([np.nan] * 3); Rg.append(np.nan); Rt.append(np.nan)
                SD.append(np.nan); SP.append([np.nan] * 5)
                continue
            p = m.particles
            score = m.gradient(p, m.data)                     # magi.gradient returns +grad log p
            lo = np.array(jnp.quantile(th, 0.025, axis=0))
            hi = np.array(jnp.quantile(th, 0.975, axis=0))
            W.append(100 * (hi - lo) / gold_w)
            Rg.append(stein_R(p, score))
            Rt.append(stein_R(p, score, np.arange(m.p)))
            SD.append(float(np.median(np.array(p.std(axis=0)) / gold_sd)))
            SP.append(spectral_profile(p, evals, evecs))

        W = np.array(W); mw = np.nanmean(W, axis=0)
        prof = np.nanmean(np.array(SP), axis=0)
        results[name] = {
            "width_pct_nuts": mw.tolist(), "width_sd": np.nanstd(W, axis=0).tolist(),
            "dev": float(np.abs(mw - 100).mean()), "R_global": float(np.nanmean(Rg)),
            "R_theta": float(np.nanmean(Rt)), "sd_ratio_median": float(np.nanmean(SD)),
            "spectral_profile": prof.tolist(), "elapsed": float(np.mean(T)),
        }
        print(f'{name:>18} {str(np.round(mw, 1)):>22} {np.abs(mw - 100).mean():6.1f} '
              f'{np.nanmean(Rg):9.3f} {np.nanmean(Rt):8.3f} {np.nanmean(SD):9.3f}  '
              f'{str(np.round(prof, 3)):>34}')

    print(f'{"NUTS (target)":>18} {"[100. 100. 100.]":>22} {0.0:6.1f} {1.0:9.3f} {1.0:8.3f} '
          f'{1.0:9.3f}  {"[1. 1. 1. 1. 1.]":>34}')

    out = os.path.join(HERE, "matrix_vs_reweighted_stein_R_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
