"""
exp05: the one untried configuration -- preconditioning AND a large fixed bandwidth.

Section 2's diagnosis: a scalar bandwidth has no correct fixed point on the MAGI posteriors
because they span several orders of magnitude in scale, while on N(0, I) it has a perfectly stable
one (exp02d, 0.93 at h = 30*h*, unchanged over 200,000 iterations from either start). If that
diagnosis is right, removing the anisotropy should restore the attractor.

Investigation 7 sec. 3 tested exact-Hessian preconditioning, but only at the ADAPTIVE bandwidth,
where it failed (on fn it was the worst of four kernels). The combination -- run SVGD in
coordinates whitened by H^-1 at the MAP, with a large FIXED bandwidth in those coordinates -- was
never run, and it is the configuration the diagnosis predicts should work.

PREDICTION, recorded before the run: in whitened coordinates the target is approximately N(0, I),
so exp02d's table should apply directly -- a stable attractor at roughly 0.81 (h = 10*h*), 0.93
(h = 30*h*), 0.98 (h = 100*h*), reached from both starts and NOT decaying. The failure mode I
expect if it does not work is that the Laplace metric is only a local whitening: the posteriors
are non-Gaussian, so H^-1 whitens the bulk but not the tails, and residual anisotropy could be
enough to drain the stiff directions anyway.

h* = 2d/ln K is the median heuristic evaluated at the whitened target, so it is computable without
a reference -- the same quantity used throughout investigation 7 sec. 6.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or ["fn", "lorenz"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 100000))
MULTS = [float(x) for x in os.environ.get("MULTS", "10,30,100").split(",")]
CHECK = [int(x) for x in os.environ.get("CHECK", "2000,10000,50000,100000").split(",")]
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    mu0 = X0.mean(0)
    Xn = mu0 + 0.25 * (X0 - mu0)
    x_map, L = M7.laplace_metric(m)
    hstar = 2.0 * d / np.log(K)
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, _ = MM.floors(S, K)
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} d={d} K={K} preconditioned by H^-1 at the MAP | "
          f"h* = 2d/lnK = {hstar:.1f} | floors energy {flk:.4f} stiffvar {sv_fl:.3f} =====",
          flush=True)
    print(f'   checkpoints {CHECK}', flush=True)
    rec = {"hstar": hstar, "floor": flk, "sv_floor": sv_fl, "runs": {}}
    for mult in MULTS:
        for slab, Xs in (("correct", X0), ("narrow4x", Xn)):
            t0 = time.time()
            P, _, hist = M7.run_svgd(m, Xs, MAXIT, kernel="standard", precond=(x_map, L),
                                     bandwidth=mult * hstar,
                                     optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                     record_every=min(CHECK))
            dt = time.time() - t0
            hd = dict(hist)
            traj = [dict(it=c, energy=S.energy(hd[c]), stiff_var=MM.stiff_var(S, hd[c]),
                         whsd=S.mahalanobis_sd(hd[c]), steinR=H.stein_R(m, hd[c]),
                         therr=S.theta_err(hd[c], m.p), ks=MM.ks_mean(S, hd[c], ref))
                    for c in CHECK if c in hd]
            print(f'  h={mult:>5.0f}*h*  {slab:>9}  stiffvar ' +
                  " ".join(f'{t["stiff_var"]:>7.3f}' for t in traj) + f'   ({dt:.0f}s)',
                  flush=True)
            print(f'{"":>18} {"":>9}  e/floor  ' +
                  " ".join(f'{t["energy"]/flk:>7.2f}' for t in traj), flush=True)
            print(f'{"":>18} {"":>9}  Stein R  ' +
                  " ".join(f'{t["steinR"]:>7.3f}' for t in traj), flush=True)
            print(f'{"":>18} {"":>9}  KS/floor ' +
                  " ".join(f'{t["ks"]/ks_fl:>7.2f}' for t in traj), flush=True)
            rec["runs"][f"{mult}|{slab}"] = dict(traj=traj, sec=dt)
            out[name] = rec
            json.dump(out, open(f"exp05_results_{name}.json", "w"), indent=1)
