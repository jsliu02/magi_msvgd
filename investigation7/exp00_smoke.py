"""
exp00: sanity checks before anything is measured.

1. Does MSVGD._stein_R agree with the harness's own, and is 1 really the target value?
   Checked by evaluating both on a REFERENCE subsample, which is by construction distributed
   as the target: R must come out at 1.
2. Does fit() still reach the reference on fn / hiv / lorenz, so that its draws are a
   legitimate "known-good" start for the fixed-point test?
3. What are the floors?
"""
import numpy as np, jax.numpy as jnp, time, sys
import harness7 as H
from msvgd import MSVGD

for name in H.USABLE:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]

    # ---- 1. Stein R on the reference itself
    sub = S.sub
    Rh = H.stein_R(m, sub)
    g = np.asarray(m.gradient(jnp.asarray(sub, m.mu.dtype), m.data), np.float64)
    Rm = float(MSVGD._stein_R(jnp.asarray(sub), jnp.asarray(-g)))

    # ---- 2. fit()
    t0 = time.time()
    post = m.fit(verbose=False)
    t_fit = time.time() - t0
    P = np.asarray(post.sample(2000, unpack=False), np.float64)

    fl = S.energy_floor()
    flk = S.energy_floor_k(2000)
    print(f"\n=== {name}  dim={d} p={m.p} ===")
    print(f"  Stein R on reference subsample : harness {Rh:.4f} | MSVGD._stein_R {Rm:.4f}"
          f"   (target 1)")
    print(f"  reliable={post.reliable}  ess={post.diagnostics['ess']:.0f}/"
          f"{post.diagnostics['n_nodes']} khat={post.diagnostics['khat']:.2f} "
          f"nnull={post.diagnostics['n_null']}")
    print(f"  fit() {t_fit:.1f}s  energy {S.energy(P):.4f} (floor {fl:.4f}, "
          f"k=2000 floor {flk:.4f})")
    print(f"  fit() max|theta err| {S.theta_err(P, m.p):.4f} sd "
          f"(floor {S.theta_err_floor(m.p):.4f})")
    print(f"  fit() Stein R {H.stein_R(m, P):.4f} | sd ratio {S.sd_ratio(P):.4f} | "
          f"whitened median sd {S.mahalanobis_sd(P):.4f}")
    w = S.cov_spectrum(P)
    print(f"  fit() whitened cov eigs: min {w.min():.3e} med {np.median(w):.3f} "
          f"max {w.max():.3f}")
    wr = S.cov_spectrum(S.sub)
    print(f"  ref  whitened cov eigs: min {wr.min():.3e} med {np.median(wr):.3f} "
          f"max {wr.max():.3f}")
    sys.stdout.flush()
