"""
investigation7 harness: scoring an ensemble against the post-GP-fix references.

Scoring rules, all of them the product of earlier mistakes:

* NEVER score on marginal standard deviations. An SVGD ensemble can hold every marginal sd at
  0.99 of the reference while sitting 45x the energy-distance floor away from it, because the
  collapse is anisotropic and no marginal sees it.
* Energy distance is computed in Mahalanobis coordinates (whitened by the REFERENCE covariance),
  so every direction counts equally regardless of its posterior scale.
* Every number is reported with its floor -- the same statistic between two disjoint subsamples
  of the reference itself. An energy distance without a floor is meaningless.
* float64 everywhere that touches a reference.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "magi_msvgd"))
sys.path.insert(0, "/home/jamie/storage-1/github-repos/msvgd/msvgd")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from setup7 import build, SYSTEMS          # noqa: E402

REFDIR = os.path.join(REPO, "investigation5")

# reference quality, read off the npz files. hes1 is unusable; nothing quantitative is drawn
# from it anywhere in investigation 7.
REF_QUALITY = {"fn": (1.0064, 0.014), "hes1": (1.7599, 0.130),
               "hiv": (1.0001, 0.000), "lorenz": (1.0072, 0.011)}
USABLE = ("fn", "hiv", "lorenz")


def load_ref(name):
    return dict(np.load(os.path.join(REFDIR, f"ref5_{name}.npz")))


class Scorer:
    """Whitening + energy distance + floors against one system's reference."""

    def __init__(self, name, n_energy=1500, seed=1):
        self.name = name
        self.z = load_ref(name)
        self.mean = self.z["mean"]
        self.cov = self.z["cov"]
        d = self.mean.shape[0]
        C = np.linalg.cholesky(self.cov + 1e-12 * np.trace(self.cov) / d * np.eye(d))
        self.Wi = np.linalg.inv(C)
        self.sub = self.z["sub"]
        self.n_energy = n_energy
        self.seed = seed
        self.sd = np.sqrt(np.diag(self.cov))

    def wh(self, X):
        return (np.asarray(X, np.float64) - self.mean) @ self.Wi.T

    def _ed(self, X, Y):
        r = np.random.default_rng(self.seed)
        n = self.n_energy
        if len(X) > n: X = X[r.choice(len(X), n, replace=False)]
        if len(Y) > n: Y = Y[r.choice(len(Y), n, replace=False)]
        def md(A, B):
            D2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
            return np.sqrt(np.maximum(D2, 0)).mean()
        return 2 * md(X, Y) - md(X, X) - md(Y, Y)

    def energy(self, X):
        """Energy distance of ensemble X to the reference subsample, in whitened coordinates."""
        return self._ed(self.wh(X), self.wh(self.sub))

    def energy_floor(self):
        """Two disjoint halves of the reference subsample against each other."""
        h = len(self.sub) // 2
        return self._ed(self.wh(self.sub[:h]), self.wh(self.sub[h:]))

    def energy_floor_k(self, k, reps=3):
        """
        Floor at the ensemble's own size k: k reference draws against the reference subsample.
        A k-particle ensemble cannot beat the Monte-Carlo error of k exact draws, and at
        k = 400 in 325 dimensions that error is NOT negligible.

        The k draws come from the first half of `sub` and are scored against the second half, so
        the two sets are disjoint -- drawing them from the same pool that is being scored against
        makes the floor spuriously small (0.021 instead of 0.055 on fn at k=2000).
        """
        h = len(self.sub) // 2
        A, B = self.sub[:h], self.sub[h:]
        r = np.random.default_rng(7)
        out = []
        for _ in range(reps):
            idx = r.choice(len(A), min(k, len(A)), replace=(k > len(A)))
            out.append(self._ed(self.wh(A[idx]), self.wh(B)))
        return float(np.mean(out))

    def cov_spectrum_floor(self, k, reps=2):
        """
        The whitened covariance spectrum of k EXACT reference draws. Needed because at k = 400
        in d = 325 the Marchenko-Pastur spread alone runs roughly (1 +/- sqrt(d/k))^2, i.e. 0.01
        to 4 -- so a raw min/max eigenvalue of an ensemble says nothing without this.
        """
        r = np.random.default_rng(11)
        los, his = [], []
        for _ in range(reps):
            idx = r.choice(len(self.sub), min(k, len(self.sub)), replace=(k > len(self.sub)))
            w = self.cov_spectrum(self.sub[idx])
            los.append(w.min()); his.append(w.max())
        return float(np.mean(los)), float(np.mean(his))

    def theta_err(self, X, p):
        """max |mean(theta) - ref mean| in reference sd."""
        mu = np.asarray(X, np.float64).mean(0)[:p]
        return float(np.max(np.abs(mu - self.mean[:p]) / self.sd[:p]))

    def theta_err_floor(self, p):
        hm = self.z["half_mean"]
        return float(np.max(np.abs(hm[0][:p] - hm[1][:p]) / self.sd[:p]))

    def sd_ratio(self, X):
        """Reported ONLY to show it is uninformative -- never used as a score."""
        return float(np.median(np.asarray(X, np.float64).std(0) / self.sd))

    def mahalanobis_sd(self, X):
        """Median sd of the whitened ensemble per whitened axis. 1 under the target.
        Unlike sd_ratio this sees the stiff directions."""
        return float(np.median(self.wh(X).std(0)))

    def band_profile(self, X, nbands=5):
        """
        Variance ratio (ensemble / reference) along the reference covariance's own eigenvectors,
        averaged within `nbands` equal-count bands ordered from the LARGEST reference eigenvalue
        (softest direction) to the smallest (stiffest).

        This is the direct picture of anisotropic collapse: a scalar-bandwidth RBF kernel has one
        length scale, so it cannot simultaneously fill a direction of posterior sd 10 and one of
        sd 1e-3. Under the target every band is 1. Unlike the raw eigenvalue spectrum this is
        computed along FIXED directions, so it carries no Marchenko-Pastur inflation --
        k independent draws give an unbiased ratio at any k.
        """
        w, V = np.linalg.eigh(0.5 * (self.cov + self.cov.T))
        o = np.argsort(w)[::-1]
        w, V = np.maximum(w[o], 1e-300), V[:, o]
        Z = (np.asarray(X, np.float64) - self.mean) @ V
        ratio = Z.var(0) / w
        return np.array([ratio[b].mean()
                         for b in np.array_split(np.arange(len(w)), nbands)])

    def theta_scorer(self, p):
        """A Scorer restricted to the first p coordinates, sharing this reference."""
        t = object.__new__(Scorer)
        t.name = self.name + f"[theta:{p}]"
        t.z = self.z
        t.mean = self.mean[:p]
        t.cov = self.cov[:p, :p]
        C = np.linalg.cholesky(t.cov + 1e-12 * np.trace(t.cov) / p * np.eye(p))
        t.Wi = np.linalg.inv(C)
        t.sub = self.sub[:, :p]
        t.n_energy = self.n_energy
        t.seed = self.seed
        t.sd = np.sqrt(np.diag(t.cov))
        return t

    def cov_spectrum(self, X):
        """Eigenvalues of the whitened ensemble covariance. All 1 under the target.
        This is the direct read of anisotropic collapse."""
        Y = self.wh(X)
        S = np.cov(Y, rowvar=False)
        w = np.linalg.eigvalsh(0.5 * (S + S.T))
        return np.maximum(w, 0)


def stein_R(m, P):
    """
    R = -(1/(k*dim)) sum_i (x_i - xbar) . s(x_i), s = grad log p.  -> 1 under the target.

    MAGI.gradient returns +grad log p, so the minus sign belongs here. MSVGD._stein_R takes
    raw_grad = -gradient and omits it; the two agree. Verified in exp00.
    """
    P = jnp.asarray(P, m.mu.dtype)
    g = np.asarray(m.gradient(P, m.data), np.float64)
    Pn = np.asarray(P, np.float64)
    return float(-np.sum((Pn - Pn.mean(0)) * g) / Pn.size)


def fmt_row(label, w=34):
    return f"{label:>{w}}"
