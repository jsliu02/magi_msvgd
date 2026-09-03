"""
exp02c: where does the fixed-h flow actually end up?

exp02b's four bandwidths give the SAME trajectory in rescaled time: stiff-var at h = 10^4*h0 after
100k iterations (0.746) equals h = 10^3*h0 after 10k (0.741); h = 10^5*h0 after 100k (0.963)
equals h = 10^4*h0 after 10k (0.963). Ten times the bandwidth buys exactly ten times the delay and
nothing else. So a large fixed h does not move the fixed point -- it slows the approach -- and the
below-the-floor results of investigation 7 sec. 9 and of exp01 are early stopping.

That rescaling is also a tool: to see 1000x further into the future, divide h by 1000 rather than
multiplying the iteration count. This runs h = 10, 100, 1000 times h0 out to 1,000,000 iterations,
which at the rescaling reaches an effective time equivalent to h = 10^5*h0 for 10^10 iterations.
Two starts that bracket the target, so the equilibrium is identified by them meeting rather than
by either one flattening.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 1000000))
MULTS = [float(x) for x in os.environ.get("MULTS", "10,100,1000").split(",")]
CHECK = [int(x) for x in os.environ.get(
    "CHECK", "5000,20000,50000,100000,200000,500000,1000000").split(",")]
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    mu0 = X0.mean(0)
    Xn = mu0 + 0.25 * (X0 - mu0)
    h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, _ = MM.floors(S, K)
    print(f"\n===== {name} K={K} h0={h0:.5g} | floors energy {flk:.4f} stiffvar {sv_fl:.3f} "
          f"| checkpoints {CHECK} =====", flush=True)
    rec = {"h0": h0, "floor": flk, "sv_floor": sv_fl, "runs": {}}
    for mult in MULTS:
        for slab, Xs in (("correct", X0), ("narrow4x", Xn)):
            t0 = time.time()
            P, _, hist = M7.run_svgd(m, Xs, MAXIT, kernel="standard", bandwidth=mult * h0,
                                     optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                     record_every=min(CHECK))
            dt = time.time() - t0
            hd = dict(hist)
            traj = [dict(it=c, energy=S.energy(hd[c]), stiff_var=MM.stiff_var(S, hd[c]),
                         whsd=S.mahalanobis_sd(hd[c]), steinR=H.stein_R(m, hd[c]),
                         therr=S.theta_err(hd[c], m.p)) for c in CHECK if c in hd]
            print(f'  h={mult:>7.0f}*h0 {slab:>9}  stiffvar ' +
                  " ".join(f'{t["stiff_var"]:>7.3f}' for t in traj) + f'   ({dt:.0f}s)',
                  flush=True)
            print(f'{"":>20} {"":>9}  e/floor  ' +
                  " ".join(f'{t["energy"]/flk:>7.2f}' for t in traj), flush=True)
            print(f'{"":>20} {"":>9}  Stein R  ' +
                  " ".join(f'{t["steinR"]:>7.3f}' for t in traj), flush=True)
            rec["runs"][f"{mult}|{slab}"] = dict(traj=traj, sec=dt)
            out[name] = rec
            json.dump(out, open(f"exp02c_results_{name}.json", "w"), indent=1)
