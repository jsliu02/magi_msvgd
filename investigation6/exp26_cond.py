"""
Exp 26: re-measure the Gauss-Newton conditioning table after the GP hyperparameter fix.

The table in the writeup gives HIV kappa(A) = 4.1e17, which was the headline motivating the
Jacobi scaling. That figure was measured before the hyperparameter fit was made scale-free, and
Section "gp" reports the same condition number falling to 4e2 afterwards -- so the motivating
example may no longer exist. This re-measures kappa(J^T J) and kappa(D J^T J D) at the post-fix
mode, per system and precision.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
from gauss_newton import GaussNewtonMAP

print(f'{"system":>8} {"prec":>5} {"iters":>6} {"decrement":>10} {"|grad|":>10} '
      f'{"cond(A)":>10} {"cond(DAD)":>10} {"th-curv range":>14}')
for name in SYSTEMS:
    for dt, tag in ((jnp.float64, "f64"), (jnp.float32, "f32")):
        m, ds = build(name, dtype=dt)
        gn = m._gn_solver()
        gn.solve(verbose=False, check=False)
        x = np.asarray(gn.map_particle, np.float64); p = m.p
        u = jnp.asarray(x, gn.dtype)
        J = np.asarray(gn.jacobian(u[:p], u[p:p + gn.nD].reshape(gn.n, gn.D), m.sigmas), np.float64)
        A = J.T @ J
        d = np.sqrt(np.maximum(np.diag(A), np.finfo(float).tiny))
        kA = np.linalg.cond(A); kD = np.linalg.cond(A / np.outer(d, d))
        thc = np.diag(A)[:p]
        g = float(np.linalg.norm(np.asarray(m.gradient(jnp.asarray(x, gn.dtype)[None, :], m.data))))
        print(f'{name:>8} {tag:>5} {gn.n_steps:>6} '
              f'{gn.decrement:>10.2e} {gn.grad_norm:>10.2e} {kA:>10.2e} {kD:>10.2e} '
              f'{f"{thc.min():.0e}-{thc.max():.0e}":>14}', flush=True)
