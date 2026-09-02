"""
Exp 1: is the returned point a mode, is it the right mode, and does the posterior have flat
directions?

Everything the pipeline does is built on the MAP. Now that the MAP solve is cheap these questions
are answerable rather than assumed.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup6 import build, SYSTEMS
import mode

print(f'{"system":>8} {"log p":>12} {"||grad||":>10} {"cond":>10} {"min eig/max":>12} '
      f'{"neg":>4} {"flat":>5} | {"escape":>18} | {"multistart":>26} | {"homotopy":>12}')
print("-" * 132)
res = {}
for name in SYSTEMS:
    m, ds = build(name)
    t0 = time.time()
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    x0 = np.asarray(m.map_particle, np.float64)
    lp0 = float(m.logdensity(m.map_particle, m.data))
    g0 = float(jnp.linalg.norm(m.gradient(jnp.asarray(x0)[None, :], m.data)))
    s = mode.spectrum(m, x0)

    xe, lpe, moved = mode.escape(m)
    uniq = mode.multistart(m, n=8, seed=0)
    best_ms = uniq[0][0] if uniq else np.nan
    nd = len(uniq)
    try:
        xh, lph = mode.homotopy(m)
    except Exception as e:
        xh, lph = None, np.nan
    dt = time.time() - t0
    res[name] = dict(lp0=lp0, s=s, lpe=lpe, moved=moved, best_ms=best_ms, nd=nd, lph=lph)
    print(f'{name:>8} {lp0:>12.4f} {g0:>10.2e} {s["cond"]:>10.2e} {s["min_rel"]:>12.2e} '
          f'{s["n_neg"]:>4} {s["n_flat"]:>5} | '
          f'{("moved to " + format(lpe, ".4f")) if moved else "no negative dir":>18} | '
          f'{f"{nd} distinct, best {best_ms:.4f}":>26} | {lph:>12.4f}')

print(f'\nflat / near-flat directions (relative eigenvalue below 1e-8), where present:')
for name, r in res.items():
    w, V, sc = r["s"]["w"], r["s"]["V"], r["s"]["scale"]
    k = int((np.abs(w) < 1e-8 * sc).sum())
    if k == 0:
        print(f'  {name:>8}: none'); continue
    m, ds = build(name)
    p = m.p
    print(f'  {name:>8}: {k} direction(s); mass on theta vs states, and the theta combination:')
    for j in range(min(k, 3)):
        v = V[:, j]
        mt = float((v[:p] ** 2).sum())
        comb = v[:p] / (np.abs(v[:p]).max() + 1e-300)
        print(f'{"":>12} eig/max {w[j]/sc:>10.2e}  theta mass {mt:>6.1%}  '
              f'theta dir {np.round(comb, 3)}')
