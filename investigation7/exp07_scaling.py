"""
exp07: how the collapse scales with dimension and particle count, on the cleanest possible
target.

exp04 found that SVGD started at exact draws from N(0, I) in d = 325 collapses to a variance
ratio of 0.018 in every direction. That is a statement about SVGD and dimension, not about MAGI,
and it is worth pinning down as a law rather than a single number, because the law is what tells
a user when mSVGD is usable at all.

Target N(0, I_d) throughout -- isotropic, so a preconditioner has nothing to correct, and the
equilibrium variance ratio is directly the number SVGD gets wrong. Started at exact draws, so
there is no burn-in to confuse with the equilibrium.

Two optimizers, because the whole result would be worthless if it were a Prodigy artefact:
Prodigy (adaptive, what exp01 uses) and plain gradient descent at a small fixed step.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd7 as M7

DS = [int(x) for x in os.environ.get("DS", "2,5,10,25,50,100,200,325,608").split(",")]
KS = [int(x) for x in os.environ.get("KS", "50,100,400,1600").split(",")]
MAXIT = int(os.environ.get("MAXIT", 2000))
KERNELS = os.environ.get("KERNELS", "standard,reweighted").split(",")


class Iso:
    def __init__(self, d):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


OPTS = [("prodigy", optax.contrib.prodigy, {}),
        ("sgd 1e-2", optax.sgd, {"learning_rate": 1e-2})]

res = {}
print(f'{"kernel":>12} {"opt":>10} {"d":>5} {"K":>6} {"K/d":>6} {"var ratio":>10} '
      f'{"x d":>8} {"SteinR":>8} {"sec":>6}', flush=True)
for kern in KERNELS:
    for oname, opt, okw in OPTS:
        for d in DS:
            g = Iso(d)
            for K in KS:
                rng = np.random.default_rng(0)
                X0 = rng.standard_normal((K, d))
                t0 = time.time()
                P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern, optimizer=opt,
                                      optimizer_kwargs=okw)
                dt = time.time() - t0
                vr = float(np.mean(P.var(0)))
                R = float(-np.sum((P - P.mean(0)) * (-P)) / P.size)
                print(f'{kern:>12} {oname:>10} {d:>5} {K:>6} {K/d:>6.2f} {vr:>10.5f} '
                      f'{vr*d:>8.3f} {R:>8.4f} {dt:>6.1f}', flush=True)
                res[f"{kern}|{oname}|{d}|{K}"] = dict(var_ratio=vr, times_d=vr * d, steinR=R)
                json.dump(res, open("exp07_results.json", "w"), indent=1)
