"""Exp 9: choose lm_init now that lam is a relative damping on a unit-diagonal matrix."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS

print(f'{"system":>8} ' + " ".join(f'{f"lm_init {v:.0e}":>16}' for v in (1e-8, 1e-10, 1e-12, 1e-14)))
print("-" * 76)
for name in SYSTEMS:
    row = []
    for lm in (1e-8, 1e-10, 1e-12, 1e-14):
        m, ds = build(name)
        m.map_solve(verbose=False, tol=1e-9, max_iter=400, lm_init=lm)
        g = float(jnp.linalg.norm(m.gradient(jnp.asarray(m.map_particle)[None, :], m.data)))
        lp = float(m.logdensity(m.map_particle, m.data))
        row.append(f'{g:.1e}/{lp:.2f}')
    print(f'{name:>8} ' + " ".join(f'{r:>16}' for r in row))
print('\n(cell = final ||grad log p|| / log p)')
