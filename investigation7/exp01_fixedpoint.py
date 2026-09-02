"""
exp01: THE FIXED-POINT TEST, redone on the corrected data.

Started AT a known-good posterior sample, does mSVGD stay there? A sampler whose fixed point is
the posterior must. This is the cleanest form of the question, because the starting point is
known correct independently of anything SVGD does.

Differences from investigation4/exp19_svgd.py, which produced the earlier negative result:
  * the GP hyperparameter fit is fixed, so the posterior geometry is different
  * the data comes from RK4 rather than forward Euler
  * three systems, including HIV (cond 4e2, reference with zero divergences), not fn alone
  * four kernels, not two: standard RBF, density-reweighted (Huang/Dong/Fang 2023), diagonal
    matrix-valued (Wang et al. 2019), and SVGD run in coordinates whitened by the EXACT Hessian
  * the whole iteration trajectory is scored, not just the endpoint, so "converges slowly" and
    "converges elsewhere" can be told apart
  * floors are reported at the ensemble's own particle count K, which for K = 400 in d = 325 is
    0.082 rather than the 0.037 of a 2000-draw comparison
  * a band profile: variance ratio along the reference covariance's own eigenvectors, banded from
    the softest direction to the stiffest. Under the target every band is 1, and it carries no
    Marchenko-Pastur inflation at small K, unlike the raw ensemble eigenvalue spectrum.

The driver runs a FIXED iteration count; there is no tolerance test to trip.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness7 as H
import msvgd7 as M7

SYS = sys.argv[1:] or list(H.USABLE)
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 1000))
CHECK = [c for c in (100, 200, 500, 1000, 2000, 5000) if c <= MAXIT]
KERNELS = os.environ.get("KERNELS", "standard,reweighted,matrix,precond").split(",")
TAG = os.environ.get("TAG", "")
out = {}

hdr = (f'{"variant":>26} {"energy":>8} {"thEnrgy":>8} {"moved":>8} {"SteinR":>8} {"thErr":>8} '
       f'{"sdrat":>6} {"whsd":>6}   band profile (soft -> stiff)     {"sec":>7}')


def score(S, T, m, P, X0, dt):
    b = S.band_profile(P)
    return dict(energy=S.energy(P), thenergy=T.energy(P[:, :m.p]),
                moved=S._ed(S.wh(P), S.wh(X0)), steinR=H.stein_R(m, P),
                therr=S.theta_err(P, m.p), sdrat=S.sd_ratio(P),
                whsd=S.mahalanobis_sd(P), band=b.tolist(), sec=dt)


def show(label, r):
    print(f'{label:>26} {r["energy"]:>8.4f} {r["thenergy"]:>8.4f} {r["moved"]:>8.4f} '
          f'{r["steinR"]:>8.4f} {r["therr"]:>8.4f} {r["sdrat"]:>6.3f} {r["whsd"]:>6.3f}   '
          + " ".join(f'{v:>6.3f}' for v in r["band"]) + f'   {r["sec"]:>7.1f}', flush=True)


for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    x_map, L = M7.laplace_metric(m)

    flk = S.energy_floor_k(K)
    tflk = T.energy_floor_k(K)
    print(f"\n===== {name}  dim={d}  p={m.p}  K={K}  maxit={MAXIT} =====", flush=True)
    print(hdr)
    r0 = score(S, T, m, X0, X0, 0.0)
    show("START fit() sample", r0)
    rec = {"start": r0, "floor": S.energy_floor(), "floor_k": flk, "theta_floor_k": tflk,
           "therr_floor": S.theta_err_floor(m.p), "dim": d, "p": int(m.p), "K": K,
           "band_floor": S.band_profile(S.sub[:K]).tolist(), "runs": {}}

    for kern in KERNELS:
        pc = (x_map, L) if kern == "precond" else None
        kk = "standard" if kern == "precond" else kern
        try:
            t0 = time.time()
            P, Rs, hist = M7.run_svgd(m, X0, MAXIT, kernel=kk, precond=pc,
                                      optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                      record_every=min(CHECK))
            dt_tot = time.time() - t0
            hd = dict(hist)
            for c in CHECK:
                if c in hd:
                    r = score(S, T, m, hd[c], X0, dt_tot * c / MAXIT)
                    r["R_traj"] = float(Rs[c - 1])
                    rec["runs"][f"{kern}_{c}"] = r
                    show(f"  {kern}, {c} it", r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec["runs"][kern] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f'{f"  {kern}":>26} FAILED {type(e).__name__}: {str(e)[:120]}', flush=True)

    print(f'{f"FLOOR: K={K} exact draws":>26} {flk:>8.4f} {tflk:>8.4f} {"":>8} {1.0:>8.4f} '
          f'{S.theta_err_floor(m.p):>8.4f} {1.0:>6.3f} {1.0:>6.3f}   '
          + " ".join(f'{v:>6.3f}' for v in rec["band_floor"]), flush=True)
    out[name] = rec
    json.dump(out, open(f"exp01_results{TAG}.json", "w"), indent=1)
