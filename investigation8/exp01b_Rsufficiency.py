"""
exp01b: Stein R = 1 is necessary, not sufficient -- and neither is energy distance.

exp01 uses R as a reference-free stand-in for the energy distance. That is only safe if R cannot
be satisfied by a wrong ensemble. For a Gaussian target R = tr(A Sigma^-1)/dim, so ANY ensemble
whose whitened covariance has eigenvalues averaging 1 scores R = 1 however they are distributed --
which is exactly SVGD's failure mode, deficient along some directions and excessive along others.

Construction: take K exact reference draws, express them in the reference covariance's own
eigenbasis, multiply the softest half of the directions by sqrt(1+a) and the stiffest half by
sqrt(1-a), map back. The mean is untouched and tr(A Sigma^-1)/d = 1 exactly for every a, so R
cannot see it at all. Sweeping a from 0 to 0.95 then asks which of the available scores CAN.

Four candidate scores:
  energy        Mahalanobis energy distance (this report's primary metric)
  band profile  variance ratio along the reference eigenvectors, 5 bands
  stiff var     variance ratio averaged over the stiffest 10% of reference directions
  KS            mean over coordinates of the Kolmogorov-Smirnov statistic between the ensemble's
                marginal and the reference's -- a quantile-calibration check with no covariance
                in it at all
Reported against the value each takes on K genuine exact draws.
"""
import numpy as np, sys
import harness8 as H


def ks_mean(S, X, ref):
    """Mean per-coordinate KS statistic between ensemble X and reference draws `ref`."""
    n, m = len(X), len(ref)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        a = np.sort(X[:, j]); b = np.sort(ref[:, j])
        allv = np.concatenate([a, b])
        ca = np.searchsorted(a, allv, side="right") / n
        cb = np.searchsorted(b, allv, side="right") / m
        out[j] = np.abs(ca - cb).max()
    return float(out.mean())


def stiff_var(S, X, frac=0.10):
    w, V = np.linalg.eigh(0.5 * (S.cov + S.cov.T))
    o = np.argsort(w)                      # ascending: stiffest first
    k = max(1, int(frac * len(w)))
    sel = o[:k]
    Z = (np.asarray(X, np.float64) - S.mean) @ V[:, sel]
    return float(np.mean(Z.var(0) / np.maximum(w[sel], 1e-300)))


print(f'{"system":>8} {"a":>6} {"R":>7} {"energy":>9} {"x floor":>8} {"stiff var":>10} '
      f'{"KS":>7} {"KS/floor":>9} {"sd ratio":>9}   band profile (soft -> stiff)', flush=True)
for name in H.USABLE:
    S = H.Scorer(name)
    d = S.mean.shape[0]
    K = 400
    rng = np.random.default_rng(0)
    half = len(S.sub) // 2
    X = S.sub[:half][rng.choice(half, K, replace=False)]
    ref = S.sub[half:]
    fl = S.energy_floor_k(K)
    ksfl = ks_mean(S, S.sub[:half][rng.choice(half, K, replace=False)], ref)

    w, V = np.linalg.eigh(0.5 * (S.cov + S.cov.T))
    o = np.argsort(w)[::-1]                                  # soft -> stiff
    V = V[:, o]
    Z = (X - S.mean) @ V
    for a in (0.0, 0.2, 0.5, 0.8, 0.95):
        s = np.ones(d)
        s[: d // 2] = np.sqrt(1 + a)
        s[d // 2:] = np.sqrt(1 - a)
        Q = S.mean + (Z * s) @ V.T
        R = float(np.mean(s ** 2))
        print(f'{name:>8} {a:>6.2f} {R:>7.4f} {S.energy(Q):>9.4f} {S.energy(Q)/fl:>8.2f} '
              f'{stiff_var(S, Q):>10.4f} {ks_mean(S, Q, ref):>7.4f} '
              f'{ks_mean(S, Q, ref)/ksfl:>9.2f} {S.sd_ratio(Q):>9.4f}   '
              + " ".join(f'{v:>6.3f}' for v in S.band_profile(Q)), flush=True)
    print(f'{"":>8} {"FLOOR":>6} {1.0:>7.4f} {fl:>9.4f} {1.00:>8.2f} {1.0:>10.4f} '
          f'{ksfl:>7.4f} {1.00:>9.2f} {1.0:>9.4f}', flush=True)
