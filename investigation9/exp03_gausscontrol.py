"""
exp03: is the theta displacement a non-Gaussianity effect at all?

exp01, at the equilibrium of the preconditioned dynamics:

    system   t(theta)   gap(sd)   |t|*gap   measured final max|theta err|
    fn        +0.186     1.029     0.191     0.242
    lorenz    -0.221     1.803     0.399     0.401
    hiv       +0.454     0.148     0.067     0.067

The displacement IS the error -- |t|*gap predicts the final theta error to 0.05 sd on all three --
and it is 0.92 to 0.999 aligned with the joint-MAP-to-reference-mean axis against 0.58 for a random
direction in p = 3. But its sign is positive on fn and hiv and NEGATIVE on lorenz, so it is not
simply mode-seeking, and the fraction is not universal (0.19, 0.22, 0.45).

The decisive control: on an exact Gaussian the mode IS the mean, so the gap is zero and a
mode-related displacement has nowhere to go. Run the identical dynamics -- same whitening matrix L,
same bandwidth, same step, same start -- on N(mean_ref, cov_ref), and measure the theta mean's
displacement in reference sd. If it vanishes, the drift is a non-Gaussianity effect and the
mode-mean axis is meaningful. If it does not, the alignment in exp01 is a coincidence of three
systems and something else is going on.

Also swept: the step size, since a fixed-step discretisation has its own O(step) fixed-point bias
and that would be a much duller explanation than any of the above.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness9 as H
import msvgd9 as M7

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 100000))
MULT = float(os.environ.get("MULT", 10))
LRS = [float(x) for x in os.environ.get("LRS", "0.01,0.003").split(",")]
out = {}


class GaussLike:
    def __init__(self, mean, cov, dt):
        n = mean.shape[0]
        self.mean = jnp.asarray(mean, dt)
        self.P = jnp.asarray(np.linalg.inv(cov + 1e-12 * np.trace(cov) / n * np.eye(n)), dt)
        self.mu = jnp.zeros((1,), dt)
        self.data = None
        self.p = 0
        self.logdensity = lambda x, data: -0.5 * (x - self.mean) @ (self.P @ (x - self.mean))
        self.gradient = jax.jit(jax.vmap(lambda x, data: -(self.P @ (x - self.mean)),
                                         in_axes=(0, None)))


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
    gauss = GaussLike(S.mean, S.cov, m.mu.dtype)
    print(f"\n===== {name} p={p} K={K} precond h={MULT:g}h* | gap(theta,max) {gap.max():.3f} sd "
          f"=====", flush=True)
    print(f'{"target":>22} {"lr":>7} {"|dtheta| (sd)":>14} {"max|dtheta|":>12} '
          f'{"max|theta err|":>15} {"|dX| (sd, rms)":>15} {"SteinR":>8} {"sec":>6}', flush=True)
    rec = {"gap_max": float(gap.max()), "rows": {}}
    for tlab, tgt in (("real posterior", m), ("N(mean_ref, cov_ref)", gauss)):
        for lr in LRS:
            t0 = time.time()
            P, _, _ = M7.run_svgd(tgt, X0, MAXIT, kernel="standard", precond=(x_map2, L),
                                  bandwidth=MULT * hstar, optimizer=optax.sgd,
                                  optimizer_kwargs={"learning_rate": lr})
            dt = time.time() - t0
            dmu = (P.mean(0) - X0.mean(0)) / S.sd            # displacement of the mean, in ref sd
            row = dict(target=tlab, lr=lr,
                       dth_norm=float(np.linalg.norm(dmu[:p])),
                       dth_max=float(np.abs(dmu[:p]).max()),
                       therr=S.theta_err(P, p),
                       dX_rms=float(np.sqrt(np.mean(dmu[p:] ** 2))),
                       steinR=H.stein_R(m, P), sec=dt)
            print(f'{tlab:>22} {lr:>7g} {row["dth_norm"]:>14.4f} {row["dth_max"]:>12.4f} '
                  f'{row["therr"]:>15.4f} {row["dX_rms"]:>15.4f} {row["steinR"]:>8.4f} '
                  f'{dt:>6.0f}', flush=True)
            rec["rows"][f"{tlab}|{lr}"] = row
            out[name] = rec
            json.dump(out, open(f"exp03_results_{name}.json", "w"), indent=1)
