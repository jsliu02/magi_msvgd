"""
exp11: the control for exp08 Part 2.

Part 2 found that fixing the bandwidth at 100x the median heuristic's target value lets SVGD hold
97.6% of the correct variance on N(0, I_325) with only 400 particles, against 1.8% under the
median heuristic. Taken at face value that would say the collapse is an artefact of the bandwidth
rule and is fixable.

It is very likely not that. As h -> infinity the RBF kernel tends to the constant 1, and the SVGD
update degenerates to

    phi(x_i) -> mean_j s(x_j) + (2/h)(x_i - xbar)  ->  mean_j s(x_j),

a rigid translation that cannot change the ensemble's shape at all. Since exp08 Part 2 starts at
EXACT draws, an update that does nothing scores perfectly. "Holds the variance" and "cannot move
the variance" are indistinguishable from that starting point.

The control: run the same bandwidth sweep from starts whose variance is deliberately WRONG --
0.25x, 4x, and a near-point-mass -- and see whether the ensemble converges to variance 1. A
sampler must; a frozen ensemble will simply keep whatever it started with.

Also reported: the fraction of the initial variance retained, which separates "converged to 1"
from "did not move".
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd7 as M7

D = int(os.environ.get("D", 325))
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 5000))
KERNELS = os.environ.get("KERNELS", "standard").split(",")
SCALES = [float(x) for x in os.environ.get("SCALES", "0.05,0.5,1.0,2.0").split(",")]
MULTS = [float(x) for x in os.environ.get("MULTS", "1.0,10.0,100.0,1000.0").split(",")]


class Iso:
    def __init__(self):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


g = Iso()
hstar = 2.0 * D / np.log(K)
res = {}
print(f'target N(0, I_{D}), K={K}, {MAXIT} iters, h* = 2d/lnK = {hstar:.1f}', flush=True)
rng0 = np.random.default_rng(99)
REF = rng0.standard_normal((2000, D))       # exact draws, for the energy distance


def energy(X, Y=REF, n=1500):
    r = np.random.default_rng(1)
    if len(X) > n: X = X[r.choice(len(X), n, replace=False)]
    if len(Y) > n: Y = Y[r.choice(len(Y), n, replace=False)]
    md = lambda A, B: np.sqrt(np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :]
                                         - 2 * A @ B.T, 0)).mean()
    return 2 * md(X, Y) - md(X, X) - md(Y, Y)


FL = energy(rng0.standard_normal((K, D)))
print(f'energy floor at K={K} exact draws: {FL:.4f}', flush=True)
print(f'{"kernel":>12} {"h/h*":>8} {"start sd":>9} {"var ratio":>10} {"var/var0":>9} '
      f'{"energy":>9} {"x floor":>8} {"mean |mu|":>10} {"sec":>6}', flush=True)
for kern in KERNELS:
    for mult in MULTS:
        for s in SCALES:
            rng = np.random.default_rng(0)
            X0 = s * rng.standard_normal((K, D))
            v0 = float(np.mean(X0.var(0)))
            t0 = time.time()
            P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern, bandwidth=mult * hstar,
                                  optimizer=optax.contrib.prodigy, optimizer_kwargs={})
            vr = float(np.mean(P.var(0)))
            e = energy(P)
            print(f'{kern:>12} {mult:>8.0f} {s:>9.2f} {vr:>10.5f} {vr/v0:>9.3f} '
                  f'{e:>9.4f} {e/FL:>8.2f} '
                  f'{float(np.abs(P.mean(0)).mean()):>10.4f} {time.time()-t0:>6.1f}', flush=True)
            res[f"{kern}|{mult}|{s}"] = dict(var_ratio=vr, var_over_var0=vr / v0, v0=v0,
                                             energy=e, floor=FL)
            json.dump(res, open("exp11_results.json", "w"), indent=1)
