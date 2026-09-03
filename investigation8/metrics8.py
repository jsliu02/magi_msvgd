"""
Scores with different sensitivity from the energy distance, added after exp01b showed the energy
distance is nearly blind to the one error SVGD actually makes.

exp01b: an ensemble with 1.95x the reference variance in the softest half of the reference's
eigendirections and 0.05x in the stiffest half -- a 40-fold misallocation -- scores Stein R = 1.000
(exactly, by construction) and a Mahalanobis energy distance of 1.06x the K-particle floor. Both
of investigation 7's headline diagnostics are trace statistics, and this distortion preserves the
trace. In d ~ 300 the energy distance is dominated by the radial distribution, and this
transformation leaves ||y||^2 essentially unchanged.

So "below the floor" does not establish "correct". These do not have that blind spot:

  stiff_var  variance ratio averaged over the stiffest `frac` of reference directions, i.e. the
             ones the posterior constrains hardest and the ones a collapsing sampler loses first.
             Reads 0.050 on the a = 0.95 ensemble above.
  worst_band variance ratio of the single worst reference direction (5th/95th percentile, to stay
             robust at finite K).
  ks_mean    mean over coordinates of the two-sample Kolmogorov-Smirnov statistic against
             reference draws. Contains no covariance at all -- purely marginal quantile
             calibration -- so it is sensitive to shape errors the second-moment scores miss.
             Reads 2.2x its floor on the same ensemble.

All three need a floor from K exact draws, as everything else here does; `floors()` returns it.
"""
import numpy as np


def _eig(S):
    w, V = np.linalg.eigh(0.5 * (S.cov + S.cov.T))
    return np.maximum(w, 1e-300), V


def stiff_var(S, X, frac=0.10):
    w, V = _eig(S)
    o = np.argsort(w)                       # ascending -> stiffest first
    sel = o[: max(1, int(frac * len(w)))]
    Z = (np.asarray(X, np.float64) - S.mean) @ V[:, sel]
    return float(np.mean(Z.var(0) / w[sel]))


def dir_ratios(S, X):
    """Variance ratio along every reference eigendirection, ordered soft -> stiff."""
    w, V = _eig(S)
    o = np.argsort(w)[::-1]
    Z = (np.asarray(X, np.float64) - S.mean) @ V[:, o]
    return Z.var(0) / w[o]


def worst_band(S, X, q=5.0):
    r = dir_ratios(S, X)
    return float(np.percentile(r, q)), float(np.percentile(r, 100 - q))


def ks_mean(S, X, ref):
    X = np.asarray(X, np.float64)
    ref = np.asarray(ref, np.float64)
    n, m = len(X), len(ref)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        a = np.sort(X[:, j]); b = np.sort(ref[:, j])
        allv = np.concatenate([a, b])
        out[j] = np.abs(np.searchsorted(a, allv, side="right") / n
                        - np.searchsorted(b, allv, side="right") / m).max()
    return float(out.mean())


def ref_split(S):
    """Disjoint halves of the reference subsample: (pool to draw K from, pool to score against)."""
    h = len(S.sub) // 2
    return S.sub[:h], S.sub[h:]


def floors(S, K, reps=3):
    """(stiff_var, ks_mean, (lo, hi) worst_band) on K genuine exact draws."""
    A, B = ref_split(S)
    rng = np.random.default_rng(11)
    sv, ks, lo, hi = [], [], [], []
    for _ in range(reps):
        X = A[rng.choice(len(A), min(K, len(A)), replace=(K > len(A)))]
        sv.append(stiff_var(S, X))
        ks.append(ks_mean(S, X, B))
        a, b = worst_band(S, X)
        lo.append(a); hi.append(b)
    return (float(np.mean(sv)), float(np.mean(ks)),
            (float(np.mean(lo)), float(np.mean(hi))))


def score(S, X, ref=None, K=None):
    if ref is None:
        ref = ref_split(S)[1]
    lo, hi = worst_band(S, X)
    return dict(stiff_var=stiff_var(S, X), ks=ks_mean(S, X, ref),
                wb_lo=lo, wb_hi=hi)
