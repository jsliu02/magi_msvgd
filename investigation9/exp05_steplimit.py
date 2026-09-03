"""
exp05: does the theta drift survive the lr -> 0 limit?

exp03 leaves the most consequential question open. The real-posterior drift falls with the step
size (theta error 0.242 -> 0.184 on fn and 0.401 -> 0.219 on lorenz for a 3.3x smaller step) while
the Gaussian control shows no step dependence at all (0.0045 -> 0.0042). If the limit is zero then
investigation 8 sec. 2.6's recipe needs neither importance reweighting nor 10^5 particles -- only a
smaller step and proportionally more iterations, which is a far better answer.

The comparison has to be at matched FLOW time, so lr * iterations is held constant:

    lr 0.01   x 100,000     (investigation 8 sec. 2.6's setting)
    lr 0.003  x 333,000
    lr 0.001  x 1,000,000

If the drift is O(step) it should fall roughly 10x across that sweep. If it converges to a nonzero
limit it should flatten. Either answer is decisive, and the second would mean the flow itself --
not its discretisation -- displaces the parameter mean.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness9 as H
import msvgd9 as M7

SYS = sys.argv[1:] or ["fn"]
K = int(os.environ.get("K", 400))
MULT = float(os.environ.get("MULT", 10))
FLOW = float(os.environ.get("FLOW", 1000.0))     # lr * iters, held constant
LRS = [float(x) for x in os.environ.get("LRS", "0.01,0.003,0.001").split(",")]
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
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    x_map = np.asarray(m.map_particle, np.float64)
    x_map2, L = M7.laplace_metric(m)
    hstar = 2.0 * d / np.log(K)
    gap = np.abs(x_map[:p] - S.mean[:p]) / S.sd[:p]
    flk = S.energy_floor_k(K)
    print(f"\n===== {name} p={p} K={K} h={MULT:g}h* | flow time lr*iters = {FLOW:g} | "
          f"gap {gap.max():.3f} sd | theta floor {S.theta_err_floor(p):.4f} =====", flush=True)
    print(f'{"lr":>8} {"iters":>9} {"t_theta":>9} {"r_theta":>9} {"align":>7} '
          f'{"max|th err|":>12} {"SteinR":>8} {"e/floor":>8} {"sec":>7}', flush=True)
    rec = {"gap_max": float(gap.max()), "floor": flk,
           "therr_floor": S.theta_err_floor(p), "rows": {}}
    for lr in LRS:
        iters = int(round(FLOW / lr))
        t0 = time.time()
        P, _, _ = M7.run_svgd(m, X0, iters, kernel="standard", precond=(x_map2, L),
                              bandwidth=MULT * hstar, optimizer=optax.sgd,
                              optimizer_kwargs={"learning_rate": lr})
        dt = time.time() - t0
        t, r = proj(S.cov[:p, :p], S.mean[:p], x_map[:p], P.mean(0)[:p])
        al = abs(t) / np.sqrt(t * t + r * r)
        row = dict(lr=lr, iters=iters, t=t, r=r, align=al, therr=S.theta_err(P, p),
                   steinR=H.stein_R(m, P), energy=S.energy(P), sec=dt)
        print(f'{lr:>8g} {iters:>9} {t:>9.3f} {r:>9.3f} {al:>7.2f} {row["therr"]:>12.4f} '
              f'{row["steinR"]:>8.4f} {row["energy"]/flk:>8.2f} {dt:>7.0f}', flush=True)
        rec["rows"][str(lr)] = row
        out[name] = rec
        json.dump(out, open(f"exp05_results_{name}.json", "w"), indent=1)
