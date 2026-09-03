"""
exp02b: is there a fixed point at the target for ANY fixed bandwidth, or only a transient?

exp02 at h = 1000*h0 on fn shows the "correct" start decaying monotonically for 20,000 iterations
-- stiff-var 0.968 -> 0.654, Stein R 1.03 -> 0.78, energy 0.73x -> 1.95x the floor -- while the
4x-narrow start crawls upward, stiff-var 0.065 -> 0.073. They are converging toward each other,
which is the signature of a common equilibrium BELOW the target. If so, exp01's 2000-iteration
numbers are transients caught on the way down, and investigation 7 sec. 9's below-the-floor result
is an early-stopping artefact rather than a fixed point.

That is cheap to settle, because 20,000 iterations cost 10 s on the V100. Two starts that bracket
the target (correct, and 4x narrow), four bandwidths spanning three decades, 100,000 iterations,
checkpointed logarithmically. If the two starts meet at v = 1 the bandwidth is a genuine fix; if
they meet below it, the fix is early stopping and must be described as such.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 100000))
MULTS = [float(x) for x in os.environ.get("MULTS", "1000,10000,100000,1000000").split(",")]
CHECK = [int(x) for x in os.environ.get(
    "CHECK", "1000,2000,5000,10000,20000,50000,100000").split(",")]
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
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} K={K} h0={h0:.5g} | floors energy {flk:.4f} stiffvar {sv_fl:.3f} "
          f"=====", flush=True)
    rec = {"h0": h0, "floor": flk, "sv_floor": sv_fl, "runs": {}}
    step = min(CHECK)
    for mult in MULTS:
        for slab, Xs in (("correct", X0), ("narrow4x", Xn)):
            t0 = time.time()
            P, _, hist = M7.run_svgd(m, Xs, MAXIT, kernel="standard", bandwidth=mult * h0,
                                     optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                     record_every=step)
            dt = time.time() - t0
            hd = dict(hist)
            traj = []
            cells = []
            for c in CHECK:
                if c in hd:
                    Q = hd[c]
                    r = dict(it=c, energy=S.energy(Q), stiff_var=MM.stiff_var(S, Q),
                             whsd=S.mahalanobis_sd(Q), steinR=H.stein_R(m, Q),
                             therr=S.theta_err(Q, m.p))
                    traj.append(r)
                    cells.append(f'{r["stiff_var"]:>7.3f}')
            print(f'  h={mult:>9.0f}*h0 {slab:>9} stiffvar@[{",".join(str(c) for c in CHECK)}]: '
                  + " ".join(cells) + f'   ({dt:.0f}s)', flush=True)
            print(f'{"":>22} {"":>9} energy  : '
                  + " ".join(f'{t["energy"]/flk:>7.2f}' for t in traj), flush=True)
            print(f'{"":>22} {"":>9} Stein R : '
                  + " ".join(f'{t["steinR"]:>7.3f}' for t in traj), flush=True)
            rec["runs"][f"{mult}|{slab}"] = dict(traj=traj, sec=dt)
            out[name] = rec
            json.dump(out, open(f"exp02b_results_{name}.json", "w"), indent=1)
