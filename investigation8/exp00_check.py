"""
exp00: before anything else, confirm the cached references still describe the current model.

tests.py changed under investigation 7 (commit cbf5088): the system definitions are numpy now, so
the grids are float64 regardless of the x64 flag, and _locate's tolerance scales with the grid
spacing. Hes1's output grid used to have 45 points instead of 33 with x64 off. Everything in
investigation 7 ran with x64 ON, so the grids ought to be unchanged for fn/hiv/lorenz -- but
"ought to" is not a measurement, and every number in investigation 8 is scored against
investigation5/ref5_*.npz, which was built before the change.

Three checks per system: the particle dimension must equal the reference's, the reference mean
must still be a high-density point of the CURRENT log-density, and fit() must still land where
investigation 7 says it does.
"""
import numpy as np, jax.numpy as jnp, time, sys
import harness8 as H

print(f'{"system":>8} {"dim":>6} {"ref dim":>8} {"match":>6} {"logp(ref mean)":>15} '
      f'{"|grad| at ref mean":>19} {"Stein R on sub":>15} {"fit energy":>11} {"floor":>8}',
      flush=True)
for name in H.USABLE:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d_model = int(m.p + m.n * m.D + int(np.sum(np.asarray(m.unknown_sigmas))))
    d_ref = S.mean.shape[0]
    ok = d_model == d_ref
    if ok:
        x = jnp.asarray(S.mean, m.mu.dtype)
        lp = float(m.magi_logdensity(x))
        g = float(np.abs(np.asarray(m.gradient(x[None, :], m.data), np.float64)).max())
        R = H.stein_R(m, S.sub)
        post = m.fit(verbose=False)
        P = np.asarray(post.sample(400, unpack=False), np.float64)
        e, fl = S.energy(P), S.energy_floor_k(400)
        print(f'{name:>8} {d_model:>6} {d_ref:>8} {"OK":>6} {lp:>15.4f} {g:>19.3e} '
              f'{R:>15.4f} {e:>11.4f} {fl:>8.4f}', flush=True)
    else:
        print(f'{name:>8} {d_model:>6} {d_ref:>8} {"MISMATCH -- reference is stale":>6}',
              flush=True)
