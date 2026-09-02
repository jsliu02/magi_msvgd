"""
Exp 11: the conditioning story, re-measured on correctly integrated data.

The original table was built on forward-Euler data whose integration error exceeded the
observation noise on HIV, so it has to be redone. "Stock" reproduces the pre-change solver exactly
-- unscaled normal equations with a uniform ridge lam*trace(A)/dim*I -- so the comparison is
against what was actually there, not against a reconstruction from memory.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS

def stock_solve(m, lam=1e-8, iters=300):
    """The solver as it was: unscaled A, uniform ridge, bounds 1e-12 / 1e3."""
    gn = m._gn_solver(); p, nD, n, D = gn.p, gn.nD, gn.n, gn.D
    dt = m.mu.dtype; sig = m.sigmas
    eye = jnp.eye(gn.dim, dtype=dt); sc = jnp.asarray(1.0 / gn.dim, dt)
    u = jnp.asarray(m.particles_init, dt); th, X = u[:p], u[p:p + nD].reshape(n, D)
    @jax.jit
    def step(th, X, lam):
        A, g, r2 = gn._normal_equations(th, X, sig)
        du = jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(A + lam * jnp.trace(A) * sc * eye), g)
        return du, r2
    for _ in range(iters):
        du, r2 = step(th, X, lam)
        tn, Xn = th - du[:p], X - du[p:].reshape(n, D)
        if bool(jnp.all(jnp.isfinite(du))) and \
           float(jnp.sum(gn.residual(tn, Xn, sig) ** 2)) < float(r2):
            th, X, lam = tn, Xn, max(lam * 0.3, 1e-12)
        else:
            lam = min(lam * 10.0, 1e3)
    part = jnp.concatenate([th, X.ravel(), sig[m.unknown_sigmas]])
    return float(jnp.linalg.norm(m.gradient(part[None, :], m.data))), \
           float(m.logdensity(part, m.data))

print(f'{"system":>8} {"cond(H)":>10} {"cond(DHD)":>10} {"cond(A)":>10} {"cond(DAD)":>10} | '
      f'{"stock ||g||":>12} {"scaled ||g||":>13} {"stock sec":>10} {"scaled sec":>11}')
print("-" * 118)
for name in SYSTEMS:
    m, ds = build(name)
    t0 = time.time(); m.map_solve(verbose=False, tol=1e-9, max_iter=300); ts = time.time() - t0
    gs = float(jnp.linalg.norm(m.gradient(jnp.asarray(m.map_particle)[None, :], m.data)))
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    dh = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    cH, cDH = np.linalg.cond(H), np.linalg.cond(H / np.outer(dh, dh))
    gn = m._gn_solver(); p, nD = gn.p, gn.nD
    xm = jnp.asarray(m.map_particle, m.mu.dtype)
    A = np.asarray(gn._normal_equations(xm[:p], xm[p:p + nD].reshape(m.n, m.D), m.sigmas)[0],
                   np.float64)
    da = np.sqrt(np.maximum(np.diag(A), 1e-300))
    cA, cDA = np.linalg.cond(A), np.linalg.cond(A / np.outer(da, da))
    m2, _ = build(name)
    t0 = time.time(); g0, lp0 = stock_solve(m2); t2 = time.time() - t0
    print(f'{name:>8} {cH:>10.2e} {cDH:>10.2e} {cA:>10.2e} {cDA:>10.2e} | '
          f'{g0:>12.2e} {gs:>13.2e} {t2:>10.2f} {ts:>11.2f}')
