"""
exp06: turn `K >~ e^p` from a two-point fit into a test.

Section 3 proposed that profiled SVGD needs K >~ e^p, from investigation 7 sec. 5's law
Var_SVGD/Var_target ~= ln(K)/d applied at d = p. The evidence was two distinct values of p: p = 3
(fn, lorenz) works at K = 16, p = 5 (hiv) does not and needs K = 256. That is a consistency check,
not a test.

The law itself can be tested at as many p as one likes, because it is a statement about SVGD on a
target of dimension p and says nothing about where the target came from. Isotropic Gaussians
N(0, I_p) for p = 2..10, K swept, and for each p the smallest K whose equilibrium variance ratio
exceeds a threshold. The law predicts K_crit(f) = exp(f*p) for threshold f, i.e. a straight line of
slope f on a log-K vs p plot.

This tests the law, not the profiled machinery. The three profiled marginals remain the only
evidence that the law transfers to them, and that is stated as such.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd8 as M7

PS = [int(x) for x in os.environ.get("PS", "2,3,4,5,6,7,8,10").split(",")]
KS = [int(x) for x in os.environ.get(
    "KS", "4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,2048,4096").split(",")]
MAXIT = int(os.environ.get("MAXIT", 4000))
THRESH = [float(x) for x in os.environ.get("THRESH", "0.8,0.9").split(",")]


class Iso:
    def __init__(self):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


g = Iso()
res = {}
print(f'{"p":>4} ' + " ".join(f'{K:>7}' for K in KS), flush=True)
for p in PS:
    row = {}
    vals = []
    for K in KS:
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((K, p))
        P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel="standard",
                              optimizer=optax.contrib.prodigy, optimizer_kwargs={})
        v = float(np.mean(P.var(0)))
        row[K] = v
        vals.append(v)
    print(f'{p:>4} ' + " ".join(f'{v:>7.3f}' for v in vals), flush=True)
    res[p] = row
    json.dump(res, open("exp06_results.json", "w"), indent=1)

print(f'\n{"p":>4} ' + " ".join(f'{"Kcrit@" + str(f):>12} {"exp(" + str(f) + "p)":>12}'
                                for f in THRESH), flush=True)
for p in PS:
    cells = []
    for f in THRESH:
        ok = [K for K in KS if res[p][K] >= f]
        cells.append(f'{(min(ok) if ok else -1):>12} {np.exp(f * p):>12.0f}')
    print(f'{p:>4} ' + " ".join(cells), flush=True)
