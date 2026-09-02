"""
exp03: does the fixed-point drift shrink as the particle count grows?

SVGD's bias is known to fall with the particle count, and there is a specific reason to expect
trouble at K ~ d: the ensemble's empirical covariance has rank at most K - 1, so with K below the
dimension it cannot represent the target's covariance at all, whatever the kernel does. fn is
d = 325, hiv d = 608, lorenz d = 306, and investigation 4 ran K = 400. This sweeps K.

If the drift is a finite-particle artefact it must shrink like some power of 1/K. If it is a
fixed-point property it will not.

Scored against the K-dependent floor, which grows as K falls -- so the comparison that matters is
energy(K) / floor(K), not energy(K) alone.
"""
import numpy as np, optax, time, sys, os, json
import harness7 as H
import msvgd7 as M7

SYS = sys.argv[1:] or ["fn"]
KS = [int(x) for x in os.environ.get("KS", "50,100,200,400,800,1600").split(",")]
MAXIT = int(os.environ.get("MAXIT", 1000))
KERNELS = os.environ.get("KERNELS", "standard,reweighted,precond").split(",")
TAG = os.environ.get("TAG", "")
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    x_map, L = M7.laplace_metric(m)
    print(f"\n===== {name}  dim={d}  p={m.p}  maxit={MAXIT} =====", flush=True)
    print(f'{"K":>6} {"variant":>14} {"energy":>8} {"e/floor":>8} {"thEnrgy":>8} {"te/flr":>8} '
          f'{"SteinR":>8} {"thErr":>8} {"whsd":>6}   band profile (soft -> stiff)     {"sec":>7}')
    rec = {}
    for K in KS:
        X0 = np.asarray(post.sample(K, seed=0, unpack=False), np.float64)
        fl, tfl = S.energy_floor_k(K), T.energy_floor_k(K)

        def row(lab, P, dt):
            r = dict(energy=S.energy(P), thenergy=T.energy(P[:, :m.p]), steinR=H.stein_R(m, P),
                     therr=S.theta_err(P, m.p), whsd=S.mahalanobis_sd(P),
                     band=S.band_profile(P).tolist(), sec=dt, floor=fl, theta_floor=tfl)
            print(f'{K:>6} {lab:>14} {r["energy"]:>8.4f} {r["energy"]/fl:>8.2f} '
                  f'{r["thenergy"]:>8.4f} {r["thenergy"]/max(tfl,1e-12):>8.2f} '
                  f'{r["steinR"]:>8.4f} {r["therr"]:>8.4f} {r["whsd"]:>6.3f}   '
                  + " ".join(f'{v:>6.3f}' for v in r["band"]) + f'   {dt:>7.1f}', flush=True)
            return r

        rec[K] = {"start": row("START fit()", X0, 0.0)}
        for kern in KERNELS:
            pc = (x_map, L) if kern == "precond" else None
            kk = "standard" if kern == "precond" else kern
            t0 = time.time()
            try:
                P, _, _ = M7.run_svgd(m, X0, MAXIT, kernel=kk, precond=pc,
                                      optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                rec[K][kern] = row(kern, P, time.time() - t0)
            except Exception as e:
                rec[K][kern] = dict(error=f"{type(e).__name__}: {str(e)[:150]}")
                print(f'{K:>6} {kern:>14} FAILED {type(e).__name__}', flush=True)
        rec[K]["floor_row"] = row("EXACT K draws", S.sub[:K], 0.0)
    out[name] = rec
    json.dump(out, open(f"exp03_results{TAG}.json", "w"), indent=1)
