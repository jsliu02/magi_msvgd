"""Exp 7: gauss_newton.py with Jacobi-scaled normal equations, against the prototype and float32."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map

print(f'{"system":>8} {"dtype":>8} | {"map_solve ||grad||":>19} {"log p":>13} {"steps":>6} {"sec":>6} '
      f'| {"prototype ||grad||":>19} {"log p":>13} | {"jac check":>10}')
print("-" * 122)
for name in SYSTEMS:
    for dtype in (jnp.float64, jnp.float32):
        m, ds = build(name, dtype=dtype)
        t0 = time.time()
        m.map_solve(verbose=False, tol=1e-8, max_iter=400)
        dt = time.time() - t0
        g = float(jnp.linalg.norm(m.gradient(jnp.asarray(m.map_particle)[None, :], m.data)))
        lp = float(m.logdensity(m.map_particle, m.data))
        rel = m._gn.check_jacobian()
        if dtype is jnp.float64:
            m2, _ = build(name, dtype=dtype)
            _, g2, _ = scaled_map(m2, tol=1e-8, max_iter=400)
            lp2 = float(m2.logdensity(m2.map_particle, m2.data))
            extra = f'{g2:>19.3e} {lp2:>13.4f}'
        else:
            extra = f'{"":>19} {"":>13}'
        print(f'{name:>8} {dtype.__name__:>8} | {g:>19.3e} {lp:>13.4f} {m._gn.n_steps:>6} '
              f'{dt:>6.2f} | {extra} | {rel:>10.2e}')
    print()
