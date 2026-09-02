"""
Exp 11: control -- is the prior fixing the improperness, or just injecting the right answer?

Exp 10's guess happened to sit exactly on the true delta and N, so the improvement there is
confounded: a prior centred on the truth will flatter any method. The test is to centre the prior
somewhere deliberately wrong and see whether the identified parameters still move toward the truth.
If they do, the prior is buying properness, and the identified parameters are recovering because
they are no longer being dragged by an unbounded direction. If they simply follow the prior
centre, it is buying nothing.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from magi import MAGI
import tests as T
from profiled2 import ProfiledPosterior2

NM = ["lam", "rho", "delta", "N", "c"]
ds = T.HIV
data = ds.reset().dataset(seed=0, step=1e-3)
tru = np.asarray(ds.hyperparams["theta"], np.float64)

GUESSES = {
    "on truth   (0.5, 1000)": np.array([30.0, 0.1, 0.5, 1000.0, 3.0]),
    "delta,N x2 (1.0, 2000)": np.array([30.0, 0.1, 1.0, 2000.0, 3.0]),
    "delta,N /2 (0.25, 500)": np.array([30.0, 0.1, 0.25, 500.0, 3.0]),
    "all x3               ": np.array([90.0, 0.3, 1.5, 3000.0, 9.0]),
}
print(f'{"prior centre":>24} {"prior sd":>10} | ' + " ".join(f'{n:>9}' for n in NM) +
      f' | {"ESS":>6}')
print("-" * 104)
print(f'{"flat (no prior)":>24} {"-":>10} | ' + " ".join(f'{"":>9}' for n in NM) + f' |')
for gname, g in GUESSES.items():
    for frac in (0.5, 0.2):
        conf = 1.0 / (frac * np.abs(g)) ** 2
        m = MAGI(ds.ode, data, g, conf,
                 sigmas=np.asarray(ds.hyperparams["sigma"], np.float64))
        m.put(dtype=jnp.float64)
        m.map_solve(verbose=False, tol=1e-9, max_iter=300)
        x = np.asarray(m.map_particle, np.float64)
        pp = ProfiledPosterior2(m, n_nodes=128, seed=0).build(verbose=False)
        print(f'{gname:>24} {frac:>10.0%} | ' + " ".join(f'{v:>9.4g}' for v in x[:5]) +
              f' | {pp.ess/pp.n_nodes:>6.1%}')
    print(f'{"  (prior centre)":>24} {"":>10} | ' + " ".join(f'{v:>9.4g}' for v in g) + ' |')
print(f'{"TRUE":>24} {"":>10} | ' + " ".join(f'{v:>9.4g}' for v in tru) + ' |')
print(f'{"flat-prior MAP":>24} {"":>10} | ' +
      " ".join(f'{v:>9.4g}' for v in [-11.9, -0.2448, 0.259, 1788, 2.76]) + ' |')
