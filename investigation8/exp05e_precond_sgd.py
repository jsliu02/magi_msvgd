"""
exp05e: preconditioned SVGD with a fixed bandwidth and a fixed step -- does it have a fixed point
AT the target?

exp05d isolated the cause of exp05's divergence, and it was not the posterior. Preconditioned by
H^-1 at the MAP, bandwidth 10*h*, 20,000 iterations on fn:

    real posterior + Prodigy      energy 287 -> 785x floor      DIVERGES
    exact Gaussian + Prodigy      energy 97x floor              DIVERGES
    real posterior + SGD 1e-3     energy 1.69 -> 1.43x floor    stable, stiff-var 1.06 -> 1.10
    exact Gaussian + SGD 1e-3     energy 1.62 -> 1.47x floor    stable, stiff-var 1.06 -> 1.10

Prodigy diverges on the exact Gaussian too, so this is an optimizer artefact in whitened
coordinates and nothing to do with the target: Prodigy estimates its step from the gradient norm,
the preconditioned gradient is uniformly O(1) in every direction, and it runs the step up until
the ensemble leaves the region where the quadratic model holds. Every other experiment in
investigations 7 and 8 used Prodigy; in the UNwhitened metric the coordinate scales differ by
orders of magnitude and that evidently keeps it in check.

So the configuration that investigation 7 sec. 3 rejected and section 2.5 predicts should work has
never actually been run: whitened metric, fixed bandwidth, fixed step. This runs it properly --
step-size sweep, several bandwidths, both starts, long.

PREDICTION: exp05c says an isotropic target at fixed h has a stable attractor at 0.58 / 0.82 / 0.93
for h = 3 / 10 / 30 h*. Whitening makes the MAGI posteriors isotropic in second moment (exp05b),
so I expect the same numbers here -- a genuine fixed point, correct to within the h-dependent
deficit, from either start. If so this is a real sampler and not a transient, which is a different
and better result than section 1's stopping rule.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or ["fn"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 200000))
MULTS = [float(x) for x in os.environ.get("MULTS", "10,30,100").split(",")]
LRS = [float(x) for x in os.environ.get("LRS", "0.01").split(",")]
CHECK = [int(x) for x in os.environ.get("CHECK", "5000,25000,50000,100000,200000").split(",")]
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    mu0 = X0.mean(0)
    Xn = mu0 + 0.25 * (X0 - mu0)
    Xw = mu0 + 4.00 * (X0 - mu0)
    x_map, L = M7.laplace_metric(m)
    hstar = 2.0 * d / np.log(K)
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, _ = MM.floors(S, K)
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} d={d} K={K} PRECOND + fixed h + fixed step | h*={hstar:.1f} | "
          f"floors energy {flk:.4f} stiffvar {sv_fl:.3f} KS {ks_fl:.4f} =====", flush=True)
    print(f'   start energies: correct {S.energy(X0)/flk:.2f}x  narrow {S.energy(Xn)/flk:.2f}x  '
          f'wide {S.energy(Xw)/flk:.2f}x   checkpoints {CHECK}', flush=True)
    rec = {"hstar": hstar, "floor": flk, "sv_floor": sv_fl, "ks_floor": ks_fl, "runs": {}}
    for mult in MULTS:
        for lr in LRS:
            for slab, Xs in (("correct", X0), ("narrow4x", Xn), ("wide4x", Xw)):
                t0 = time.time()
                # lr < 0 selects Prodigy, so the adaptive and fixed-step cases share one sweep
                opt, okw = ((optax.contrib.prodigy, {}) if lr < 0
                            else (optax.sgd, {"learning_rate": lr}))
                P, _, hist = M7.run_svgd(m, Xs, MAXIT, kernel="standard", precond=(x_map, L),
                                         bandwidth=mult * hstar, optimizer=opt,
                                         optimizer_kwargs=okw, record_every=min(CHECK))
                dt = time.time() - t0
                hd = dict(hist)
                tr = [dict(it=c, energy=S.energy(hd[c]), stiff_var=MM.stiff_var(S, hd[c]),
                           whsd=S.mahalanobis_sd(hd[c]), steinR=H.stein_R(m, hd[c]),
                           therr=S.theta_err(hd[c], m.p), ks=MM.ks_mean(S, hd[c], ref))
                      for c in CHECK if c in hd]
                lrlab = "prodigy" if lr < 0 else f"{lr:g}"
                print(f'  h={mult:>5.0f}h* {lrlab:<8} {slab:>9} e/floor  '
                      + " ".join(f'{t["energy"]/flk:>8.2f}' for t in tr) + f'  ({dt:.0f}s)',
                      flush=True)
                print(f'{"":>28} {"":>9} stiffvar ' +
                      " ".join(f'{t["stiff_var"]:>8.3f}' for t in tr), flush=True)
                print(f'{"":>28} {"":>9} whsd^2   ' +
                      " ".join(f'{t["whsd"]**2:>8.3f}' for t in tr), flush=True)
                print(f'{"":>28} {"":>9} Stein R  ' +
                      " ".join(f'{t["steinR"]:>8.3f}' for t in tr), flush=True)
                print(f'{"":>28} {"":>9} KS/floor ' +
                      " ".join(f'{t["ks"]/ks_fl:>8.2f}' for t in tr), flush=True)
                rec["runs"][f"{mult}|{lr}|{slab}"] = dict(traj=tr, sec=dt)
                out[name] = rec
                json.dump(out, open(f"exp05e_results_{name}.json", "w"), indent=1)
