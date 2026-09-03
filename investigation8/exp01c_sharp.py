"""
exp01c: re-score the bandwidth sweep with metrics that are not trace statistics.

exp01b showed that Stein R and the Mahalanobis energy distance are both nearly blind to a
trace-preserving misallocation of variance across directions -- the exact error SVGD makes. Since
exp01 selects h by R and validates by energy, both of its instruments share a blind spot, so the
sweep is repeated here with `metrics8`:

  stiff var   variance ratio over the stiffest 10% of reference directions (1 = correct)
  wb lo/hi    5th and 95th percentile of the per-direction variance ratio (both 1 = correct)
  KS          mean per-coordinate Kolmogorov-Smirnov statistic vs reference draws, over its floor

This also answers, for the first time with an instrument that can see it, investigation 7's open
question of whether "energy below the K-particle floor" means the ensemble is right or merely
quasi-uniform.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or list(H.USABLE)
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 2000))
MULTS = [float(x) for x in os.environ.get(
    "MULTS", "1,10,100,300,1000,3000,10000,30000,100000").split(",")]
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, wb_fl = MM.floors(S, K)
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} K={K} h0={h0:.5g} | floors: energy {flk:.4f} "
          f"stiffvar {sv_fl:.3f} KS {ks_fl:.4f} wb [{wb_fl[0]:.3f}, {wb_fl[1]:.3f}] =====",
          flush=True)
    print(f'{"h/h0":>9} {"SteinR":>8} {"energy":>8} {"x flr":>7} {"stiffvar":>9} '
          f'{"wb lo":>7} {"wb hi":>7} {"KS":>7} {"KS/flr":>7} {"thErr":>8}', flush=True)

    def row(lab, P):
        sc = MM.score(S, P, ref)
        r = dict(steinR=H.stein_R(m, P), energy=S.energy(P), therr=S.theta_err(P, m.p), **sc)
        print(f'{lab:>9} {r["steinR"]:>8.4f} {r["energy"]:>8.4f} {r["energy"]/flk:>7.2f} '
              f'{r["stiff_var"]:>9.4f} {r["wb_lo"]:>7.3f} {r["wb_hi"]:>7.3f} {r["ks"]:>7.4f} '
              f'{r["ks"]/ks_fl:>7.2f} {r["therr"]:>8.4f}', flush=True)
        return r

    rec = {"h0": h0, "floors": dict(energy=flk, stiff_var=sv_fl, ks=ks_fl, wb=wb_fl), "rows": {}}
    rec["start"] = row("start", X0)
    for mult in MULTS:
        P, _, _ = M7.run_svgd(m, X0, MAXIT, kernel="standard", bandwidth=mult * h0,
                              optimizer=optax.contrib.prodigy, optimizer_kwargs={})
        rec["rows"][str(mult)] = row(f"{mult:g}", P)
    rec["exact"] = row("EXACT K", MM.ref_split(S)[0][:K])
    out[name] = rec
    json.dump(out, open(f"exp01c_results_{name}.json", "w"), indent=1)
