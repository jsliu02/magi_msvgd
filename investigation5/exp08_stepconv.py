"""
Exp 8: how fine an Euler step does each system need, and can it be afforded?

tests.py integrates with forward Euler inside a lax.scan that retains every state, so the memory
cost is (t_max / step) * D * 8 bytes: the new 1e-6 default is 5.8 GB for Hes1, whose horizon is
240. The step that matters is the one at which the integration error at the OBSERVATION times is
well below the observation noise, since beyond that point the data is noise-limited and a finer
grid buys nothing. Measured by Richardson comparison between successive halvings, in units of
sigma, together with the exact-grid-membership that sample() requires.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, gc
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import tests as T

CASES = {"fn": (T.FitzHughNagumo, [1e-3, 1e-4, 1e-5, 1e-6]),
         "hes1": (T.Hes1, [1e-3, 1e-4, 1e-5]),
         "hiv": (T.HIV, [1e-4, 1e-5, 1e-6]),
         "lorenz": (T.Lorenz, [1e-4, 1e-5, 1e-6])}

for name, (ds, steps) in CASES.items():
    hp = dict(ds.hyperparams)
    if "x0" not in hp: hp["x0"] = jnp.asarray(hp["X0"], jnp.float64)
    ds.hyperparams = hp
    sg = np.asarray(hp["sigma"], np.float64)
    tobs = np.asarray(jnp.round(jnp.unique(jnp.concat([jnp.asarray(x) for x in hp["tau"]])), 4))
    tmax = float(np.asarray(hp["I"]).max())
    prev = None
    print(f'--- {name}  (horizon {tmax:g}, D={len(sg)}) ---')
    print(f'{"step":>9} {"grid pts":>11} {"mem MB":>8} {"exact match":>12} '
          f'{"max |change| / sigma vs previous step":>38}')
    for st in steps:
        npts = int(tmax / st) + 2
        mem = npts * len(sg) * 8 / 1e6
        Tg = jnp.arange(0.0, tmax + st, st)
        match = bool(jnp.isin(jnp.asarray(tobs), Tg).all())
        if mem > 2500:
            print(f'{st:>9.0e} {npts:>11,} {mem:>8.0f} {str(match):>12} {"skipped (memory)":>38}')
            continue
        ds.solution = None
        ds.ground_truth(step=st)
        sol = np.asarray(ds.solution); Tn = np.asarray(ds.T)
        idx = np.clip(np.searchsorted(Tn, tobs), 0, len(Tn) - 1)
        cur = sol[idx]
        rel = "-" if prev is None else \
            f'{np.nanmax(np.abs(cur - prev) / np.where(np.isfinite(sg), sg, np.nan)):.3f}'
        print(f'{st:>9.0e} {npts:>11,} {mem:>8.0f} {str(match):>12} {rel:>38}')
        prev = cur
        del sol, Tn; ds.solution = None; ds.T = None; gc.collect()
    print()
