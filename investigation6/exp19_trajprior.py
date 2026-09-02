"""
Exp 19: do the trajectories recover when theta gets a proper prior?

Exp 18 showed the fitted trajectory bands match the reference chain closely on every system, so the
approximation is not the problem -- but the mean sits 15 sigma from the truth on Hes1 and 176 on
HIV, and the reference sits there too. That is the flat prior of section 3: with no pi_Theta the
parameters drift to values the data cannot exclude, the states follow them, and the posterior is
confidently in the wrong place. FitzHugh-Nagumo is the exception because its three parameters are
all identified.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from magi import MAGI
from setup6 import SYSTEMS
import tests as T

print(f'{"system":>8} {"prior":>16} {"comp":>5} {"band/scale":>11} {"mean err/scale":>15} '
      f'{"traj rel err":>13}   scale = sigma, or the range where unobserved')
print("-" * 104)
for name in SYSTEMS:
    spec = SYSTEMS[name]; ds = spec["sys"]
    data = ds.reset().dataset(seed=0)
    sig = np.asarray(ds.hyperparams["sigma"], np.float64)
    g = np.asarray(spec["guess"], np.float64)
    I = np.asarray(ds.hyperparams["I"], np.float64)
    sol = np.asarray(ds.solution); Tg = np.asarray(ds.T)
    truth = sol[np.clip(np.searchsorted(Tg, I), 0, len(Tg) - 1)]
    scale = np.where(np.isfinite(sig), sig, np.nan)
    for j in range(len(scale)):                       # unobserved: use the state's own range
        if not np.isfinite(scale[j]):
            scale[j] = max(truth[:, j].max() - truth[:, j].min(), 1e-12)
    for lbl, prec in [("flat", np.zeros(len(g))), ("100% of guess", 1.0 / np.abs(g) ** 2)]:
        m = MAGI(ds.ode, data, g, prec, sigmas=sig)
        m.put(jnp.float64)
        post = m.fit(verbose=False)
        Xs = np.asarray(post.sample(k=400, seed=0)[0], np.float64)
        band, mean = Xs.std(0), Xs.mean(0)
        for j in range(len(scale)):
            rel = np.linalg.norm(mean[:, j] - truth[:, j]) / max(np.linalg.norm(truth[:, j]), 1e-30)
            print(f'{name if (j == 0 and lbl == "flat") else "":>8} {lbl if j == 0 else "":>16} '
                  f'{j:>5} {np.median(band[:, j]) / scale[j]:>11.2f} '
                  f'{np.max(np.abs(mean[:, j] - truth[:, j])) / scale[j]:>15.2f} {rel:>13.1%}')
    print()
