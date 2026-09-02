"""
Jacobi-preconditioned Gauss-Newton, and a scale-free wrapper for the whole pipeline.

gauss_newton.py forms the normal equations A = J^T J and Choleskys them, which is 110x faster than
a QR least-squares solve but squares the condition number. On FitzHugh-Nagumo cond(A) = 7e3 and
this is free; on HIV cond(A) = 4e16, the factorisation is meaningless and the solve stalls three
orders of magnitude short of the mode. Exp 2 showed the cause is the parameter units -- HIV's
theta-block curvature spans 3e-6 to 5e2 -- and that symmetric diagonal scaling recovers 1.4e9 of
the 4.4e16.

The fix is one line of linear algebra applied in the right place. Marquardt's damping in the
existing solver already scales by diag(A); this scales the SOLVE as well, which is what actually
matters:

    D = diag(A)^(-1/2),  (D A D + lam I) s = -D g,  delta = D s.

Exactly equivalent in real arithmetic and completely different in floating point.
"""
import numpy as np, jax, jax.numpy as jnp
from gauss_newton import GaussNewtonMAP


def scaled_map(m, tol=1e-10, max_iter=300, lam0=1e-10, verbose=False):
    """Gauss-Newton to the MAP with Jacobi-preconditioned normal equations."""
    gn = m._gn_solver()
    p, nD, n, D = gn.p, gn.nD, gn.n, gn.D
    dt = m.mu.dtype
    u = jnp.asarray(m.particles_init, dt)
    th, X = u[:p], u[p:p + nD].reshape(n, D)
    sig = m.sigmas
    lam = lam0
    best = None

    @jax.jit
    def step(th, X, lam):
        A, g, r2 = gn._normal_equations(th, X, sig)
        d = jnp.sqrt(jnp.maximum(jnp.diag(A), jnp.finfo(A.dtype).tiny))
        Di = 1.0 / d
        As = A * Di[:, None] * Di[None, :]
        As = As + lam * jnp.eye(As.shape[0], dtype=A.dtype)
        s = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(As), -(g * Di))
        return s * Di, r2, jnp.linalg.norm(g)

    for it in range(max_iter):
        delta, r2, gnorm = step(th, X, lam)
        if not bool(jnp.all(jnp.isfinite(delta))):
            lam *= 10.0
            continue
        th_n = th + delta[:p]
        X_n = X + delta[p:].reshape(n, D)
        r2n = jnp.sum(gn.residual(th_n, X_n, sig) ** 2)
        if bool(r2n < r2):                       # accept and relax the damping
            th, X, lam = th_n, X_n, max(lam * 0.3, 1e-14)
            if float(gnorm) < tol:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break
    particle = jnp.concatenate([th, X.ravel(), sig[m.unknown_sigmas]])
    m.map_particle = particle
    m._gn.map_particle = particle
    g = float(jnp.linalg.norm(m.gradient(particle[None, :], m.data)))
    if verbose:
        print(f'preconditioned GN: {it + 1} steps | ||grad|| = {g:.3e} | '
              f'log p = {float(m.logdensity(particle, m.data)):.6f}')
    return particle, g, it + 1
