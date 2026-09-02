"""
exp02: mSVGD from a cold start, driven the way a user would drive it, using the SHIPPED code.

The historical MAGI.solve() (commit 834ebf8, before MAGI stopped inheriting from MSVGD) built its
initial ensemble as

    particles_init + sigma_init * N(0, I),   sigma_init = 0.2,  k = 200

and then ran MSVGD.solve with `mitosis_splits` doublings. That is reproduced here against the
current `msvgd.MSVGD` (whose `k_schedule` replaced `mitosis_splits`), so what is measured is the
library, not my reimplementation of it.

Step 0 first checks that my own driver (msvgd7.run_svgd, used for the kernel variants in exp01,
which the shipped code no longer supports) reproduces MSVGD.solve on the standard kernel. If it
does not, nothing in exp01 can be trusted.

Wall clock is the point of comparison as much as accuracy: fit() costs 9-29 s and the NUTS
reference 2-20 min.
"""
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax, time, sys, os, json
import harness7 as H
import msvgd7 as M7
from msvgd import MSVGD

SYS = sys.argv[1:] or list(H.USABLE)
TAG = os.environ.get("TAG", "")
out = {}

CONFIGS = [
    # label, k0, k_schedule, max_iter/phase, optimizer, kwargs
    ("adam lr .1   k200, 2000 it",      200, None,       2000, optax.adam, {"learning_rate": 0.1}),
    ("prodigy      k200, 2000 it",      200, None,       2000, optax.contrib.prodigy, {}),
    ("prodigy      k50->200, 2 splits", 50,  [100, 200], 1000, optax.contrib.prodigy, {}),
    ("prodigy      k400, 5000 it",      400, None,       5000, optax.contrib.prodigy, {}),
]

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    print(f"\n===== {name}  dim={d}  p={m.p} =====", flush=True)

    # ---- step 0: does my driver reproduce the shipped one?
    key = jr.key(8)
    xchk = m.particles_init + jr.normal(key, (64, d), dtype=m.mu.dtype) * 0.2
    s = MSVGD(m.logdensity, data=m.data)
    Pa = np.asarray(s.solve(x0=xchk, max_iter=50, atol=0.0, rtol=0.0, optimizer=optax.adam,
                            optimizer_kwargs={"learning_rate": 0.1},
                            monitor_convergence=-1, random_seed=8), np.float64)
    Pb, _, _ = M7.run_svgd(m, xchk, 50, kernel="standard", optimizer=optax.adam,
                           optimizer_kwargs={"learning_rate": 0.1})
    rel = float(np.max(np.abs(Pa - Pb)) / max(np.max(np.abs(Pa)), 1e-300))
    print(f"  driver check vs MSVGD.solve, 50 adam steps: max rel diff {rel:.3e}", flush=True)

    t0 = time.time(); post = m.fit(verbose=False); t_fit = time.time() - t0
    Pf = np.asarray(post.sample(400, unpack=False), np.float64)
    flk = S.energy_floor_k(400)
    print(f'{"variant":>34} {"energy":>8} {"thEnrgy":>8} {"SteinR":>8} {"thErr":>8} '
          f'{"whsd":>6}   band profile (soft -> stiff)     {"sec":>7}')

    def row(label, P, dt):
        r = dict(energy=S.energy(P), thenergy=T.energy(P[:, :m.p]), steinR=H.stein_R(m, P),
                 therr=S.theta_err(P, m.p), sdrat=S.sd_ratio(P), whsd=S.mahalanobis_sd(P),
                 band=S.band_profile(P).tolist(), sec=dt)
        print(f'{label:>34} {r["energy"]:>8.4f} {r["thenergy"]:>8.4f} {r["steinR"]:>8.4f} '
              f'{r["therr"]:>8.4f} {r["whsd"]:>6.3f}   '
              + " ".join(f'{v:>6.3f}' for v in r["band"]) + f'   {dt:>7.1f}', flush=True)
        return r

    rec = {"driver_check": rel, "fit": row("fit()  [the incumbent]", Pf, t_fit),
           "floor_k400": flk, "theta_floor_k400": T.energy_floor_k(400),
           "therr_floor": S.theta_err_floor(m.p), "ref_sec": float(S.z["sec"]), "runs": {}}

    for label, k0, ks, mi, opt, okw in CONFIGS:
        x0 = m.particles_init + jr.normal(jr.key(8), (k0, d), dtype=m.mu.dtype) * 0.2
        s = MSVGD(m.logdensity, data=m.data)
        t0 = time.time()
        try:
            P = np.asarray(s.solve(x0=x0, k_schedule=ks, max_iter=mi, atol=0.0, rtol=0.0,
                                   optimizer=opt, optimizer_kwargs=okw,
                                   monitor_convergence=-1, random_seed=8), np.float64)
            rec["runs"][label] = row("  " + label, P, time.time() - t0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec["runs"][label] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f'{"  " + label:>34} FAILED {type(e).__name__}', flush=True)

    print(f'{"FLOOR: 400 exact draws":>34} {flk:>8.4f} {rec["theta_floor_k400"]:>8.4f} '
          f'{1.0:>8.4f} {rec["therr_floor"]:>8.4f} {1.0:>6.3f}   '
          + " ".join(f'{v:>6.3f}' for v in S.band_profile(S.sub[:400]))
          + f'   {rec["ref_sec"]:>7.0f} (NUTS)', flush=True)
    out[name] = rec
    json.dump(out, open(f"exp02_results{TAG}.json", "w"), indent=1)
