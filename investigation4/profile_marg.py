"""
Laplace marginals (Tierney & Kadane 1986) along chosen directions, via bordered Gauss-Newton.

For a unit direction v the exact 1-d marginal of z = v^T (x - mu) is, to second order,

    log p_marg(z) = -U_prof(z) - 0.5 log det H_perp(z) + const,
    U_prof(z) = min { U(x) : v^T (x - mu) = z },

with H_perp the exact Hessian restricted to v's orthogonal complement at the profile point. Two
identities make this cheap. The constrained Gauss-Newton step is the unconstrained step plus a
multiple of A^-1 v, so it costs one extra triangular solve against the SAME Cholesky factor. And
for unit v, det(N^T H N) = det(H) * (v^T H^-1 v) for any orthonormal basis N of v's complement,
so the restricted determinant never needs the 324-dimensional basis to be formed.

This is the ESTIMATOR half of a screen-then-correct scheme: the slice curvature screen says WHICH
directions the quadratic model fails on, and is far too pessimistic about how badly (it holds the
complement fixed, while the profile lets it relax); the profile says by how much.
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial


class Profiler:
    def __init__(self, gn, magi):
        self.gn, self.m = gn, magi
        n, D, p = gn.n, gn.D, gn.p
        self.d = gn.dim
        def f_local(z, t): return magi.ode(z[:D][None, :], z[D:], t[None])[0]
        self.hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
        self.IDX = jnp.asarray(np.concatenate(
            [p + np.arange(n)[:, None] * D + np.arange(D)[None, :],
             np.broadcast_to(np.arange(p)[None, :], (n, p))], axis=1))
        self._step = jax.jit(self._step_impl)
        self._hess = jax.jit(self._hess_impl)

    def _hess_impl(self, x):
        gn, n, D, p = self.gn, self.gn.n, self.gn.D, self.gn.p
        th, X = x[:p], x[p:].reshape(n, D)
        J = gn.jacobian(th, X, self.m.sigmas)
        c = gn.b * jnp.einsum('nd,dmn->md', gn.residual(th, X, self.m.sigmas)[2 * gn.nD:]
                              .reshape(n, D), gn.Lk)
        Z = jnp.concatenate([X, jnp.broadcast_to(th, (n, p))], axis=1)
        S = jnp.einsum('md,mdij->mij', c, self.hl(Z, gn.I))
        return J.T @ J + jnp.zeros((self.d, self.d), x.dtype).at[
            self.IDX[:, :, None], self.IDX[:, None, :]].add(S)

    def _step_impl(self, x, v, c, lam):
        """One damped Gauss-Newton step constrained to v^T x = c."""
        gn, n, D, p = self.gn, self.gn.n, self.gn.D, self.gn.p
        A, g, _ = gn._normal_equations(x[:p], x[p:].reshape(n, D), self.m.sigmas)
        A = A + lam * jnp.diag(jnp.diag(A))
        cf = jax.scipy.linalg.cho_factor(A + 1e-12 * jnp.trace(A) / self.d * jnp.eye(self.d, dtype=x.dtype))
        df = jax.scipy.linalg.cho_solve(cf, -g)
        Av = jax.scipy.linalg.cho_solve(cf, v)
        return x + df + (c - v @ (x + df)) / (v @ Av) * Av

    def profile(self, v, zs, x0, iters=6, lam=1e-8):
        """U_prof and 0.5 log det H_perp on a grid, warm-started outward from the mode."""
        v = jnp.asarray(v / np.linalg.norm(v))
        out, order = {}, sorted(range(len(zs)), key=lambda i: abs(zs[i]))
        c0 = float(v @ jnp.asarray(x0))
        warm = {}
        for i in order:
            near = min(warm, key=lambda j: abs(zs[j] - zs[i])) if warm else None
            x = jnp.asarray(x0 if near is None else warm[near])
            for _ in range(iters):
                x = self._step(x, v, c0 + zs[i], lam)
            warm[i] = x
            Hm = self._hess(x)
            Hm = 0.5 * (Hm + Hm.T)
            cf = jax.scipy.linalg.cho_factor(Hm)
            ld = 2 * jnp.sum(jnp.log(jnp.diag(cf[0]))) + jnp.log(v @ jax.scipy.linalg.cho_solve(cf, v))
            out[i] = (-float(self.m.logdensity(x, self.m.data)), float(0.5 * ld), np.asarray(x))
        U = np.array([out[i][0] for i in range(len(zs))])
        ld = np.array([out[i][1] for i in range(len(zs))])
        return U, ld, np.stack([out[i][2] for i in range(len(zs))])


def moments(zs, logdens):
    """Normalised mean and variance of a 1-d log-density sampled on a grid (Simpson-free trapz)."""
    w = np.exp(logdens - logdens.max())
    Z = np.trapezoid(w, zs)
    mu = np.trapezoid(w * zs, zs) / Z
    return mu, np.trapezoid(w * (zs - mu) ** 2, zs) / Z
