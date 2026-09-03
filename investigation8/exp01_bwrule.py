"""
exp01: is "raise h until Stein R ~ 1" a usable reference-free bandwidth rule?

Investigation 7 sec. 9 showed that a fixed bandwidth of 1000*h0 turns mSVGD from a 60x-the-floor
failure into a below-the-floor sampler on fn and lorenz -- but 1000 was found by sweeping against
a reference, which no user has. Sec. 13 proposed tuning h upward until Stein R ~ 1, since R needs
no reference, and never tested it. This tests it.

At each rung of a five-decade sweep, record R (reference-free, what the rule would see) and the
Mahalanobis energy distance (what the rule is trying to minimise, needs the reference). Then ask:
does argmin|R-1| coincide with argmin energy, and what does taking the former cost?

PREDICTION, recorded before running (see investigation8.md sec. 1):
  (a) With a fit() start the rule is DEGENERATE. R(start) is already ~1, and as h -> infinity the
      update tends to a rigid translation that cannot change the ensemble, so R -> R(start) ~ 1
      for free. "Raise h until R ~ 1" would then select h = infinity and return the starting
      ensemble unchanged -- passing its own test while doing nothing.
  (b) With a mis-scaled start (0.25x) the rule has content, because R(start) ~ 0.06 and only a
      bandwidth that actually moves the ensemble can raise it. Here I expect argmax R to be within
      a decade of argmin energy, because both are driven by the same expansion.
If (a) holds, the rule as stated in investigation 7 sec. 13 is wrong and needs the start to be
part of it.

Two starts per system, therefore: the fit() ensemble, and the same ensemble shrunk 4x about its
mean. K = 400, 2000 iterations, standard RBF kernel, Prodigy, float64.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7

SYS = sys.argv[1:] or list(H.USABLE)
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 2000))
MULTS = [float(x) for x in os.environ.get(
    "MULTS", "1,3,10,30,100,300,1000,3000,10000,30000,100000").split(",")]
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    Xn = X0.mean(0)[None, :] + 0.25 * (X0 - X0.mean(0))
    h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
    flk, tflk = S.energy_floor_k(K), T.energy_floor_k(K)
    print(f"\n===== {name} dim={d} p={m.p} K={K} it={MAXIT}  h0={h0:.5g}  "
          f"floor={flk:.4f} =====", flush=True)
    rec = {"h0": h0, "floor": flk, "theta_floor": tflk,
           "therr_floor": S.theta_err_floor(m.p), "starts": {}}

    for slab, Xs in (("fit() start", X0), ("0.25x start", Xn)):
        r0 = dict(steinR=H.stein_R(m, Xs), energy=S.energy(Xs),
                  thenergy=T.energy(Xs[:, :m.p]), therr=S.theta_err(Xs, m.p),
                  whsd=S.mahalanobis_sd(Xs))
        print(f'-- {slab}: Stein R {r0["steinR"]:.4f}  energy {r0["energy"]:.4f} '
              f'({r0["energy"]/flk:.1f}x floor)', flush=True)
        print(f'{"h/h0":>10} {"SteinR":>9} {"|R-1|":>8} {"energy":>9} {"x floor":>8} '
              f'{"thEnrgy":>9} {"thErr":>8} {"whsd":>6} {"sec":>6}', flush=True)
        rows = {}
        for mult in MULTS:
            t0 = time.time()
            P, _, _ = M7.run_svgd(m, Xs, MAXIT, kernel="standard", bandwidth=mult * h0,
                                  optimizer=optax.contrib.prodigy, optimizer_kwargs={})
            R = H.stein_R(m, P)
            e = S.energy(P)
            r = dict(steinR=R, energy=e, thenergy=T.energy(P[:, :m.p]),
                     therr=S.theta_err(P, m.p), whsd=S.mahalanobis_sd(P),
                     band=S.band_profile(P).tolist(), sec=time.time() - t0)
            rows[mult] = r
            print(f'{mult:>10.0f} {R:>9.4f} {abs(R-1):>8.4f} {e:>9.4f} {e/flk:>8.2f} '
                  f'{r["thenergy"]:>9.4f} {r["therr"]:>8.4f} {r["whsd"]:>6.3f} '
                  f'{r["sec"]:>6.1f}', flush=True)
        # the rule's pick vs the oracle's
        mR = min(rows, key=lambda k: abs(rows[k]["steinR"] - 1))
        mE = min(rows, key=lambda k: rows[k]["energy"])
        pen = rows[mR]["energy"] / rows[mE]["energy"]
        print(f'   RULE picks h={mR:g}*h0 (R={rows[mR]["steinR"]:.4f}, '
              f'energy {rows[mR]["energy"]:.4f} = {rows[mR]["energy"]/flk:.2f}x floor)', flush=True)
        print(f'   ORACLE picks h={mE:g}*h0 (energy {rows[mE]["energy"]:.4f} = '
              f'{rows[mE]["energy"]/flk:.2f}x floor)   penalty {pen:.2f}x  '
              f'h ratio {mR/mE:g}x', flush=True)
        rec["starts"][slab] = dict(start=r0, rows={str(k): v for k, v in rows.items()},
                                   rule_h=mR, oracle_h=mE, penalty=pen)
        out[name] = rec
        json.dump(out, open(f"exp01_results_{name}.json", "w"), indent=1)
