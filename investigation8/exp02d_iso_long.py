"""
exp02d: was investigation 7 sec. 6's isotropic result also a transient?

exp02c finds that on the MAGI posteriors every fixed bandwidth ends in the same collapsed state
and larger h only delays it. Investigation 7 sec. 6 concluded the opposite on N(0, I_325) -- at
h = 100*h*, 5000 iterations, the ensemble converged to a variance ratio of 0.976 from starts
spanning 0.05x to 2x the correct spread, and that was read as a genuine attractor. On the evidence
of exp02c, 5000 iterations may simply have been early.

Same target, same K, same bandwidths, 2,000,000 iterations instead of 5,000. If the variance ratio
holds at 0.97 the isotropic case really is different and the difference is worth understanding;
if it decays, investigation 7 sec. 6 needs the same correction as sec. 9.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd8 as M7

D = int(os.environ.get("D", 325))
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 2000000))
MULTS = [float(x) for x in os.environ.get("MULTS", "3,10,30,100").split(",")]
CHECK = [int(x) for x in os.environ.get(
    "CHECK", "5000,20000,100000,500000,1000000,2000000").split(",")]
SCALES = [float(x) for x in os.environ.get("SCALES", "0.25,1.0").split(",")]


class Iso:
    def __init__(self):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


g = Iso()
hstar = 2.0 * D / np.log(K)
res = {}
print(f'N(0, I_{D}), K={K}, h* = 2d/lnK = {hstar:.1f}, checkpoints {CHECK}', flush=True)
for mult in MULTS:
    for s in SCALES:
        rng = np.random.default_rng(0)
        X0 = s * rng.standard_normal((K, D))
        t0 = time.time()
        P, _, hist = M7.run_svgd(g, X0, MAXIT, kernel="standard", bandwidth=mult * hstar,
                                 optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                 record_every=min(CHECK))
        hd = dict(hist)
        vals = [float(np.mean(hd[c].var(0))) for c in CHECK if c in hd]
        print(f'  h={mult:>5.0f}*h*  start sd {s:>5.2f}  var ratio  '
              + " ".join(f'{v:>8.4f}' for v in vals) + f'   ({time.time()-t0:.0f}s)', flush=True)
        res[f"{mult}|{s}"] = dict(check=[c for c in CHECK if c in hd], var=vals)
        json.dump(res, open("exp02d_results.json", "w"), indent=1)
