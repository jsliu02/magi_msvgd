"""
exp01: is the theta drift in investigation 8 sec. 2.6 mode-seeking?

Investigation 8 sec. 2.6 found a reference-free recipe -- whiten by H^-1, fixed bandwidth, small
fixed step, run to equilibrium, rescale by 1/sqrt(Stein R) -- that reaches 0.57-0.96x the
K-particle energy floor on all three systems, but degrades the parameter mean on two of them:

    max|theta err|   fn 0.077 -> 0.242   lorenz 0.071 -> 0.401   hiv 0.096 -> 0.067

The coordinator's hypothesis: SVGD's drift term is a kernel-weighted average of the score, which
pulls the ensemble mean toward high density, so the equilibrium hands back exactly the
mode-versus-mean correction that the profiled posterior exists to supply. The independently
measured joint-mode-to-profile-mode distances order identically to the degradation:

    hiv 0.16 sd      fn 1.07 sd      lorenz 1.42 sd

This tests it directly. For a coordinate block with reference mean b, joint MAP a, and reference
covariance Sigma_block, whiten by Sigma_block and project the ensemble mean m onto the segment:

    t = (m - b).(a - b) / |a - b|^2        0 at the reference mean, 1 at the joint MAP
    r = |(m - b) - t (a - b)| / |a - b|    orthogonal drift, in units of the gap

If the hypothesis holds, t should move from ~0 (the fit() start sits at the reference mean) toward
~1, and should be SIMILAR ACROSS SYSTEMS even though the absolute errors differ fivefold -- because
what differs between systems is the length of the segment, not the fraction travelled.

Computed for theta and, separately, for the state block, where the mode-versus-mean gap is small.
If only theta drifts, that localises it.

Trajectory recorded, not just the endpoint, so "moving toward the mode" is distinguishable from
"happens to sit near it".
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness9 as H
import msvgd9 as M7
import metrics9 as MM

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 100000))
MULT = float(os.environ.get("MULT", 10))
LR = float(os.environ.get("LR", 0.01))
CHECK = [int(x) for x in os.environ.get("CHECK", "5000,25000,50000,100000").split(",")]
out = {}


def proj(cov_blk, b, a, m):
    """(t, r) for block mean m, whitened by the block's reference covariance."""
    n = b.shape[0]
    C = np.linalg.cholesky(cov_blk + 1e-12 * np.trace(cov_blk) / n * np.eye(n))
    Ci = np.linalg.inv(C)
    u = Ci @ (a - b)
    v = Ci @ (m - b)
    nu2 = float(u @ u)
    if nu2 <= 0:
        return float("nan"), float("nan")
    t = float(v @ u / nu2)
    r = float(np.linalg.norm(v - t * u) / np.sqrt(nu2))
    return t, r


for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    p = m.p
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    x_map = np.asarray(m.map_particle, np.float64)
    x_map2, L = M7.laplace_metric(m)
    hstar = 2.0 * d / np.log(K)
    flk = S.energy_floor_k(K)
    ref = MM.ref_split(S)[1]

    # the two segments this experiment is about
    gap_th = np.abs(x_map[:p] - S.mean[:p]) / S.sd[:p]
    gap_X = np.abs(x_map[p:] - S.mean[p:]) / S.sd[p:]
    print(f"\n===== {name} d={d} p={p} K={K} precond h={MULT:g}h* lr={LR:g} =====", flush=True)
    print(f'  |joint MAP - reference mean|, in reference sd:'
          f'  theta max {gap_th.max():.3f} mean {gap_th.mean():.3f}   '
          f'states max {gap_X.max():.3f} mean {gap_X.mean():.3f}', flush=True)
    print(f'{"iter":>8} {"t_theta":>9} {"r_theta":>9} {"t_states":>9} {"r_states":>9} '
          f'{"thErr":>8} {"SteinR":>8} {"e/floor":>8}', flush=True)

    rec = {"gap_theta_max": float(gap_th.max()), "gap_theta_mean": float(gap_th.mean()),
           "gap_X_max": float(gap_X.max()), "gap_X_mean": float(gap_X.mean()),
           "floor": flk, "traj": []}

    def show(it, Q):
        mu = Q.mean(0)
        tt, rt = proj(S.cov[:p, :p], S.mean[:p], x_map[:p], mu[:p])
        # states: use a random 200-coordinate subset, so the Cholesky stays cheap and the
        # projection is not dominated by one stiff direction
        rng = np.random.default_rng(0)
        sel = p + rng.choice(d - p, min(200, d - p), replace=False)
        ts, rs = proj(S.cov[np.ix_(sel, sel)], S.mean[sel], x_map[sel], mu[sel])
        r = dict(it=it, t_theta=tt, r_theta=rt, t_states=ts, r_states=rs,
                 therr=S.theta_err(Q, p), steinR=H.stein_R(m, Q), energy=S.energy(Q))
        print(f'{it:>8} {tt:>9.3f} {rt:>9.3f} {ts:>9.3f} {rs:>9.3f} {r["therr"]:>8.4f} '
              f'{r["steinR"]:>8.4f} {r["energy"]/flk:>8.2f}', flush=True)
        return r

    rec["traj"].append(show(0, X0))
    t0 = time.time()
    P, _, hist = M7.run_svgd(m, X0, MAXIT, kernel="standard", precond=(x_map2, L),
                             bandwidth=MULT * hstar, optimizer=optax.sgd,
                             optimizer_kwargs={"learning_rate": LR},
                             record_every=min(CHECK))
    dt = time.time() - t0
    hd = dict(hist)
    for c in CHECK:
        if c in hd:
            rec["traj"].append(show(c, hd[c]))
    # the rescaled ensemble, to confirm the rescaling leaves t untouched
    Rq = H.stein_R(m, P)
    mu = P.mean(0)
    Pc = mu[None, :] + (P - mu) / np.sqrt(max(Rq, 1e-12))
    rec["rescaled"] = show(-1, Pc)
    print(f'  (row -1 is the 1/sqrt(R) rescaling: it moves nothing, by construction)', flush=True)
    rec["sec"] = dt
    np.savez_compressed(f"eq_{name}.npz", P=P, Pc=Pc, X0=X0, x_map=x_map, R=Rq)
    out[name] = rec
    json.dump(out, open(f"exp01_results_{name}.json", "w"), indent=1)
