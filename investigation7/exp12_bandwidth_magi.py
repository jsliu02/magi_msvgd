"""
exp12: does the fixed-bandwidth fix survive on the real posteriors?

exp11 changed the conclusion. On N(0, I_325) with K = 400, holding the bandwidth at 100x the
median heuristic's target value gives a variance ratio of 0.97, and -- this is the part that
matters -- it converges there from starting ensembles spanning 0.05x to 2x the correct spread,
expanding by a factor of 388 from the narrowest. So it is a genuine attractor, not a frozen
ensemble, and the collapse is an artefact of the ADAPTIVE bandwidth rule rather than of SVGD.

Two things that does not yet establish, both tested here.

1. Getting the variance right is not getting the distribution right. This scores the
   fixed-bandwidth runs by energy distance against the reference, with its floor, plus the band
   profile -- an isotropic h can only match one scale, and the real posteriors are anisotropic
   over several orders of magnitude.
2. A user has no reference from which to compute h*. The bandwidth here is set from the
   median heuristic evaluated on the fit() ensemble, h0 = median||x-y||^2 / ln K, which IS
   available in practice, and swept over multiples of it.

Both starts are run: the fit() ensemble (correct) and the same ensemble shrunk to 0.25x about its
mean (wrong), so "attractor" and "does not move" stay distinguishable.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness7 as H
import msvgd7 as M7

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 2000))
MULTS = [float(x) for x in os.environ.get("MULTS", "1,10,100,1000").split(",")]
KERNELS = os.environ.get("KERNELS", "standard").split(",")
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
    flk, tflk = S.energy_floor_k(K), T.energy_floor_k(K)
    print(f"\n===== {name}  dim={d}  K={K}  maxit={MAXIT}  "
          f"h0 (median heuristic at fit()) = {h0:.4g} =====", flush=True)
    print(f'{"variant":>34} {"energy":>8} {"thEnrgy":>8} {"SteinR":>8} {"thErr":>8} {"whsd":>6}'
          f'   band profile (soft -> stiff)     {"sec":>7}', flush=True)
    rec = {"h0": h0, "floor": flk, "theta_floor": tflk,
           "therr_floor": S.theta_err_floor(m.p), "runs": {}}

    def row(lab, P, dt):
        r = dict(energy=S.energy(P), thenergy=T.energy(P[:, :m.p]), steinR=H.stein_R(m, P),
                 therr=S.theta_err(P, m.p), whsd=S.mahalanobis_sd(P),
                 band=S.band_profile(P).tolist(), sec=dt)
        print(f'{lab:>34} {r["energy"]:>8.4f} {r["thenergy"]:>8.4f} {r["steinR"]:>8.4f} '
              f'{r["therr"]:>8.4f} {r["whsd"]:>6.3f}   '
              + " ".join(f'{v:>6.3f}' for v in r["band"]) + f'   {dt:>7.1f}', flush=True)
        return r

    rec["start"] = row("START fit() draws", X0, 0.0)
    Xn = X0.mean(0)[None, :] + 0.25 * (X0 - X0.mean(0))
    rec["start_narrow"] = row("START fit() draws shrunk 0.25x", Xn, 0.0)

    for kern in KERNELS:
        for mult in MULTS:
            for lab, Xs in (("wide start", X0), ("0.25x start", Xn)):
                t0 = time.time()
                try:
                    P, _, _ = M7.run_svgd(m, Xs, MAXIT, kernel=kern, bandwidth=mult * h0,
                                          optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                    rec["runs"][f"{kern}|{mult}|{lab}"] = row(
                        f"  {kern} h={mult:g}*h0, {lab}", P, time.time() - t0)
                except Exception as e:
                    print(f'  {kern} h={mult:g}*h0 {lab}: FAILED {type(e).__name__}', flush=True)
                    rec["runs"][f"{kern}|{mult}|{lab}"] = dict(error=str(e)[:150])
    print(f'{f"FLOOR: {K} exact draws":>34} {flk:>8.4f} {tflk:>8.4f} {1.0:>8.4f} '
          f'{S.theta_err_floor(m.p):>8.4f} {1.0:>6.3f}   '
          + " ".join(f'{v:>6.3f}' for v in S.band_profile(S.sub[:K])), flush=True)
    out[name] = rec
    json.dump(out, open(f"exp12_results_{name}.json", "w"), indent=1)
