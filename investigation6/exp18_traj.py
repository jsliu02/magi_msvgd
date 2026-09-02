"""
Exp 18: are the posterior TRAJECTORIES sensible, on every system?

Everything measured so far has been about theta. The states are 300 to 600 of the coordinates and
are what a user actually plots, and nothing in this work has checked them. Compared here against
the reference chain where one exists, and against the observation noise otherwise, since a
trajectory band far wider than sigma at observed times is wrong regardless of what the reference
says.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")

print(f'{"system":>8} {"path":>9} {"comp":>5} {"band/sigma":>11} {"ref band/sigma":>15} '
      f'{"mean err/sigma":>15} {"ref mean err":>13}')
print("-" * 92)
for name in SYSTEMS:
    m, ds = build(name)
    post = m.fit(verbose=False)
    Xs, th, sg = post.sample(k=400, seed=0)
    Xs = np.asarray(Xs, np.float64)
    n, D = m.n, m.D
    sig = np.asarray(ds.hyperparams["sigma"], np.float64)
    I = np.asarray(m.I).ravel()
    sol = np.asarray(ds.solution); T = np.asarray(ds.T)
    truth = sol[np.clip(np.searchsorted(T, I), 0, len(T) - 1)]
    band = Xs.std(axis=0)                                    # (n, D) pointwise sd
    mean = Xs.mean(axis=0)
    ref_band = ref_mean_err = None
    if os.path.exists(R(name)):
        z = np.load(R(name))
        rc = np.sqrt(np.maximum(np.diag(z["cov"]), 0))[m.p:].reshape(n, D)
        rm = z["mean"][m.p:].reshape(n, D)
        ref_band, ref_mean_err = rc, rm
    path = "profiled" if post.reliable else "Laplace"
    for j in range(D):
        s = sig[j] if np.isfinite(sig[j]) else np.nan
        rb = f'{np.median(ref_band[:, j]) / s:>15.2f}' if ref_band is not None else f'{"--":>15}'
        rme = (f'{np.max(np.abs(ref_mean_err[:, j] - truth[:, j])) / s:>13.2f}'
               if ref_band is not None else f'{"--":>13}')
        print(f'{name if j == 0 else "":>8} {path if j == 0 else "":>9} {j:>5} '
              f'{np.median(band[:, j]) / s:>11.2f} {rb} '
              f'{np.max(np.abs(mean[:, j] - truth[:, j])) / s:>15.2f} {rme}')
    print()
