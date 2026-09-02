"""
Exp 25: regenerate the chosen-stencil / ESS table AFTER the GP hyperparameter fix.

The table in the writeup (tab:chosen) was measured before Section "GP" corrected the
hyperparameter fit, so its ESS column describes a different posterior -- HIV in particular
reported 2.6%, which cannot be reconciled with HIV now passing the gate. This re-runs the
full fit() per system and precision and reports the step actually chosen, the plateau it was
chosen from, the effective sample size, and the wall time.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS

print(f'{"system":>8} {"prec":>5} {"h/sd":>6} {"plateau":>12} {"ESS":>8} {"khat":>7} '
      f'{"failed":>7} {"gate":>6} {"secs":>7}')
for name in SYSTEMS:
    for dt, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        m, ds = build(name, dtype=dt)
        t0 = time.time()
        post = m.fit(verbose=False)
        el = time.time() - t0
        pp = post.profiled
        run = f'{pp.fd_run[0]:g}-{pp.fd_run[1]:g}' if pp.fd_plateau else "none"
        print(f'{name:>8} {tag:>5} {getattr(pp,"fd_pick",np.nan):>6g} {run:>12} '
              f'{pp.ess/pp.n_nodes:>7.1%} {pp.khat:>7.2f} {int((~pp.ok).sum()):>7} '
              f'{("OK" if post.reliable else "FALL"):>6} {el:>7.1f}', flush=True)
