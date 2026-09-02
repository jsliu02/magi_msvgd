"""Exp 3: does Jacobi preconditioning of the solve recover the MAP on the hard systems?"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map

print(f'{"system":>8} | {"stock ||grad||":>15} {"log p":>14} {"sec":>6} | '
      f'{"precond ||grad||":>17} {"log p":>14} {"steps":>6} {"sec":>6} | {"nats gained":>12}')
print("-" * 116)
for name in SYSTEMS:
    m, ds = build(name)
    t0 = time.time(); m.map_solve(verbose=False, tol=1e-10, max_iter=400); t1 = time.time() - t0
    g0 = float(jnp.linalg.norm(m.gradient(jnp.asarray(m.map_particle)[None, :], m.data)))
    lp0 = float(m.logdensity(m.map_particle, m.data))
    m2, _ = build(name)
    t0 = time.time(); _, g1, ns = scaled_map(m2, tol=1e-10, max_iter=400); t2 = time.time() - t0
    lp1 = float(m2.logdensity(m2.map_particle, m2.data))
    print(f'{name:>8} | {g0:>15.3e} {lp0:>14.4f} {t1:>6.2f} | {g1:>17.3e} {lp1:>14.4f} '
          f'{ns:>6} {t2:>6.2f} | {lp1 - lp0:>+12.4f}')
