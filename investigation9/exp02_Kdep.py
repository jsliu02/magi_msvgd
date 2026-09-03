"""
exp02: does the theta displacement shrink with the particle count?

exp01 refutes the mode-seeking DIRECTION and confirms the mode-seeking SCALE. At the equilibrium
of the preconditioned dynamics the theta mean is displaced along the joint-MAP-to-reference-mean
axis by a fraction t of the gap:

    fn      t = +0.186   (toward the MAP)      gap 1.029 sd   theta error 0.077 -> 0.242
    lorenz  t = -0.221   (AWAY from the MAP)   gap 1.803 sd   theta error 0.071 -> 0.401

Same magnitude, opposite sign, and in both cases the displacement is 92% and 99% aligned with that
axis (against 58% for a random direction in p = 3). So |drift| ~= 0.2 x gap explains the ordering
the mode-versus-mean distances predicted, without the direction the hypothesis assumed.

The question that decides whether the recipe is usable is whether that 0.2 is a finite-K bias. If
it is O(1/K) it can be bought off; if it is a property of the fixed point at any K it cannot.
Investigation 7 sec. 4 measured the finite-K variance bias as exactly O(1/K) at the adaptive
bandwidth, and found the MEAN unbiased there -- but the preconditioned dynamics runs at a fixed
bandwidth ~10^3 times larger, which weakens the repulsion by 1/h while leaving the drift at O(1),
so the mean has no reason to stay unbiased.

K = 100, 400, 1600 at the same bandwidth-per-h*, same step, same length.

PREDICTION: if t ~ 1/K the three values should fall by 16x across the sweep; if t is a fixed-point
property it should be flat. I expect flat-to-weakly-decreasing, because the displacement is
strongly aligned with a specific direction rather than looking like noise -- an O(1/K) sampling
bias would not pick out the mode-mean axis so cleanly.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness9 as H
import msvgd9 as M7

SYS = sys.argv[1:] or ["fn", "lorenz"]
KS = [int(x) for x in os.environ.get("KS", "100,400,1600").split(",")]
MAXIT = int(os.environ.get("MAXIT", 100000))
MULT = float(os.environ.get("MULT", 10))
LR = float(os.environ.get("LR", 0.01))
out = {}


def proj(cov_blk, b, a, m):
    n = b.shape[0]
    C = np.linalg.cholesky(cov_blk + 1e-12 * np.trace(cov_blk) / n * np.eye(n))
    Ci = np.linalg.inv(C)
    u, v = Ci @ (a - b), Ci @ (m - b)
    nu2 = float(u @ u)
    t = float(v @ u / nu2)
    return t, float(np.linalg.norm(v - t * u) / np.sqrt(nu2))


for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d, p = S.mean.shape[0], m.p
    post = m.fit(verbose=False)
    x_map = np.asarray(m.map_particle, np.float64)
    x_map2, L = M7.laplace_metric(m)
    gap = np.abs(x_map[:p] - S.mean[:p]) / S.sd[:p]
    print(f"\n===== {name} p={p} gap(theta, max) {gap.max():.3f} sd  "
          f"precond h={MULT:g}h* lr={LR:g} it={MAXIT} =====", flush=True)
    print(f'{"K":>6} {"h*":>9} {"t_theta":>9} {"r_theta":>9} {"align":>7} {"thErr0":>8} '
          f'{"thErr":>8} {"SteinR":>8} {"sec":>7}', flush=True)
    rec = {"gap_max": float(gap.max()), "rows": {}}
    for K in KS:
        X0 = np.asarray(post.sample(K, seed=0, unpack=False), np.float64)
        hstar = 2.0 * d / np.log(K)
        t0 = time.time()
        P, _, _ = M7.run_svgd(m, X0, MAXIT, kernel="standard", precond=(x_map2, L),
                              bandwidth=MULT * hstar, optimizer=optax.sgd,
                              optimizer_kwargs={"learning_rate": LR})
        dt = time.time() - t0
        t, r = proj(S.cov[:p, :p], S.mean[:p], x_map[:p], P.mean(0)[:p])
        al = abs(t) / np.sqrt(t * t + r * r) if (t or r) else float("nan")
        row = dict(K=K, hstar=hstar, t=t, r=r, align=al, therr0=S.theta_err(X0, p),
                   therr=S.theta_err(P, p), steinR=H.stein_R(m, P), sec=dt)
        print(f'{K:>6} {hstar:>9.1f} {t:>9.3f} {r:>9.3f} {al:>7.2f} {row["therr0"]:>8.4f} '
              f'{row["therr"]:>8.4f} {row["steinR"]:>8.4f} {dt:>7.0f}', flush=True)
        rec["rows"][str(K)] = row
        out[name] = rec
        json.dump(out, open(f"exp02_results_{name}.json", "w"), indent=1)
    print(f'  random-direction alignment in p={p} would be {1/np.sqrt(p):.2f}', flush=True)
