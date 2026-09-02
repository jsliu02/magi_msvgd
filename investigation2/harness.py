"""
Shared harness for investigation2: build the FHN/MAGI problem, run a sampler, score it
against the 64k-draw NUTS gold standard.

Scoring deliberately uses several complementary views, because investigation.md showed that
theta interval widths alone cannot distinguish overshoot from genuine improvement:

  R_global   Stein-identity dispersion diagnostic; 1 = correct. Gaussian-exact reading
             R = tr(A Sigma^-1)/dim, so it is sensitive to stiff-direction collapse.
  profile    R decomposed along NUTS principal axes, binned softest -> stiffest.
  energy     Energy distance to the NUTS sample in NUTS-whitened coords. A proper metric
             between distributions, no Gaussian assumption, no reliance on Stein's identity.
  bias       Mahalanobis distance between ensemble mean and NUTS mean, per-dim normalized.
  width%     theta 95% CI widths as % of NUTS (the old criterion, kept for continuity).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "magi_msvgd"))
sys.path.insert(0, os.path.join(os.path.dirname(REPO), "msvgd", "msvgd"))

import jax
import jax.numpy as jnp
import optax
from magi import MAGI
from msvgd import MSVGD

GOLD_PATH = os.path.join(REPO, "magi_msvgd", "dev tests", "nuts_gold_standard.npz")
DIM, P = 325, 3


# --------------------------------------------------------------------------------- problem
def fn_ode(X, theta, t=None):
    V, R = X.T
    a, b, c = theta
    return jnp.stack([c * (V - V ** 3 / 3 + R), -1 / c * (V - a + b * R)])


def build_magi(dtype=jnp.float32):
    d = np.loadtxt(os.path.join(REPO, "magi_msvgd", "y.csv"), delimiter=",")
    grid = np.arange(0, 20.001, 0.125)
    full = np.full((grid.shape[0], d.shape[1]), np.nan)
    full[:, 0] = grid
    full[np.isin(full[:, 0], d[:, 0])] = d
    m = MAGI(fn_ode, full, [1, 1, 1], [0.0, 0.0, 0.0], sigmas=[0.2, 0.2])
    m.put(dtype=dtype, device=jax.devices()[0])
    return m


# ------------------------------------------------------------------------------ gold standard
class Gold:
    _cache = None

    def __new__(cls):
        if cls._cache is None:
            g = np.load(GOLD_PATH)
            pos = g["positions"].reshape(-1, DIM).astype(np.float64)
            self = super().__new__(cls)
            self.pos = pos
            self.mean = pos.mean(axis=0)
            self.sd = pos.std(axis=0)
            self.cov = np.cov(pos, rowvar=False)
            ev, V = np.linalg.eigh(self.cov)
            self.evals = np.maximum(ev, 1e-14)
            self.evecs = V
            self.theta_w = g["theta_ci_hi"] - g["theta_ci_lo"]
            self.theta_lo, self.theta_hi = g["theta_ci_lo"], g["theta_ci_hi"]
            rng = np.random.default_rng(0)
            self.ref = pos[rng.choice(len(pos), 2000, replace=False)]
            cls._cache = self
        return cls._cache

    def whiten(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.evecs / np.sqrt(self.evals)


def _energy_distance(X, Y, rng, n=1500):
    """2 E|X-Y| - E|X-X'| - E|Y-Y'|; 0 iff equal in distribution."""
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
    def md(A, B):
        return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0)).mean()
    return float(2 * md(X, Y) - md(X, X) - md(Y, Y))


def stein_R(particles, score, cols=None):
    p, s = (particles, score) if cols is None else (particles[:, cols], score[:, cols])
    return float(-jnp.sum((p - p.mean(axis=0)) * s) / p.size)


def evaluate(particles, magi, n_bins=5, tag=""):
    """particles : (k, DIM) in the ORIGINAL parameterization."""
    G = Gold()
    p = jnp.asarray(particles)
    out = {"tag": tag, "k": int(p.shape[0])}
    if not bool(jnp.all(jnp.isfinite(p))):
        return {**out, "failed": True}

    score = magi.gradient(p, magi.data)                  # magi.gradient returns +grad log p
    out["R_global"] = stein_R(p, score)
    out["R_theta"] = stein_R(p, score, np.arange(P))

    pn = np.asarray(p, dtype=np.float64)
    th = pn[:, :P]
    lo, hi = np.quantile(th, 0.025, axis=0), np.quantile(th, 0.975, axis=0)
    wpct = 100 * (hi - lo) / G.theta_w
    out["width_pct"] = wpct.tolist()
    out["width_dev"] = float(np.abs(wpct - 100).mean())
    out["theta_mean"] = th.mean(axis=0).tolist()

    # spectral profile: ensemble variance / NUTS eigenvalue along NUTS principal axes
    proj = (pn - G.mean) @ G.evecs
    ratio = proj.var(axis=0) / G.evals
    order = np.argsort(G.evals)[::-1]                     # softest first
    out["profile"] = [float(np.median(ratio[b])) for b in np.array_split(order, n_bins)]
    out["profile_mean"] = float(ratio.mean())

    Z, Zr = G.whiten(pn), G.whiten(G.ref)
    rng = np.random.default_rng(1)
    out["energy"] = _energy_distance(Z, Zr, rng)
    out["bias"] = float(np.sqrt(((Z.mean(axis=0)) ** 2).mean()))
    out["sd_ratio_med"] = float(np.median(pn.std(axis=0) / G.sd))
    return out


HDR = (f'{"variant":>26} {"width%NUTS":>21} {"dev":>5} {"R_glob":>7} {"R_th":>6} '
       f'{"energy":>7} {"bias":>6} {"sdrat":>6}  {"profile soft->stiff":>32}')


def show(r):
    if r.get("failed"):
        print(f'{r["tag"]:>26} {"FAILED (non-finite)":>21}')
        return
    print(f'{r["tag"]:>26} {str(np.round(r["width_pct"],1)):>21} {r["width_dev"]:5.1f} '
          f'{r["R_global"]:7.3f} {r["R_theta"]:6.3f} {r["energy"]:7.3f} {r["bias"]:6.3f} '
          f'{r["sd_ratio_med"]:6.3f}  {str(np.round(r["profile"],3)):>32}')


def gold_row():
    """Self-consistency floor: score an independent NUTS subsample the same way."""
    G = Gold()
    rng = np.random.default_rng(7)
    sub = G.pos[rng.choice(len(G.pos), 800, replace=False)]
    m = build_magi(dtype=jnp.float64)
    r = evaluate(jnp.asarray(sub, dtype=jnp.float64), m, tag="NUTS k=800 (floor)")
    return r


def save(results, name):
    with open(os.path.join(HERE, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {name}.json")


# ------------------------------------------------------------------- robustness monkeypatches
def patch_split():
    """
    MSVGD._mitotic_split regularizes the ensemble covariance with an ABSOLUTE 1e-6*I ridge, then
    Choleskys it. With k < dim that covariance is EXACTLY singular (rank <= k-1), so the ridge is
    the only thing making the factorization succeed -- and being absolute, it only works at the
    particle scale it was tuned for (~1e-2, where 1e-6 is ~1% relative). Under any
    reparameterization that changes the particle scale it breaks: in exp01's whitened coordinates
    (spread ~50) the ridge is negligible and cholesky returns NaN; shrinking it to stay relative
    makes it negligible in x-space instead. There is no single ridge that works for both.

    Fix: drop the ridge and sample the covariance-matched jitter inside the ensemble's own span
    via SVD, which is what a rank-deficient Gaussian actually is. Exact, scale-invariant, and
    it removes the isotropic 325-dim noise floor the ridge was silently injecting.

    Applied uniformly to every variant in investigation2 so comparisons stay internally
    consistent.
    """
    import jax.random as jr
    from functools import partial

    @partial(jax.jit, static_argnames=['self', 'is_MAP', 'k_target'])
    def _split(self, particles, key, is_MAP, k_target):
        k, dim = particles.shape
        n_new = k_target - k
        budget = (0.01 if is_MAP else self.pairwise_distance(particles, -1)[1]) / 2
        key_parents, key_jitter = jr.split(key)
        centered = particles - particles.mean(axis=0)
        _, S, Wt = jnp.linalg.svd(centered, full_matrices=False)   # cov = Wt.T diag(S^2/k) Wt
        trace = jnp.sum(S ** 2) / k
        n_each, n_rem = divmod(n_new, k)
        idx = jnp.concatenate([jnp.repeat(jnp.arange(k), n_each),
                               jr.choice(key_parents, k, shape=(n_rem,), replace=False)])
        z = jr.normal(key_jitter, shape=(n_new, S.shape[0]), dtype=particles.dtype)
        jitter = (z * (S / jnp.sqrt(k))) @ Wt
        offspring = particles[idx] + jnp.sqrt(budget / trace) * jitter
        return jnp.concatenate([particles, offspring], axis=0)

    MSVGD._mitotic_split = _split
