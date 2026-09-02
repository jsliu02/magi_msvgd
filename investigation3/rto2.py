"""
RTO with full Gauss-Newton solves, using the analytic Jacobian.

The fixed-Jacobian variant fails here: its contraction factor is
||I - Rb^-1 Qb^T J(x)||, and Rb^-1 amplifies by 1/sigma_min, so it is governed by the
worst-conditioned direction (measured kappa = 6.5 >> 1) even though J itself barely moves in the
dominant directions. Recomputing the Jacobian removes that dependence entirely: Gauss-Newton on
Qb^T R(x) = xi converges at a rate set by the curvature of R alone.

Per particle per iteration this needs Qb^T J(x), assembled without ever forming J:
only the ODE rows of J depend on x, so Qb^T J = const + Qb_ode^T * ode_block(x).
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from lsq import LSQ
from jac import AnalyticJac


class RTO2:
    def __init__(self, m, x_map, gn_map=40):
        self.m = m
        self.l = LSQ(m)
        self.aj = AnalyticJac(self.l)
        self.R = jax.jit(self.l.residual)
        self.Rv = jax.jit(jax.vmap(self.l.residual))
        # tighten the MAP as a stationary point of ||R||^2
        x = jnp.asarray(x_map, jnp.float64)
        for _ in range(gn_map):
            x = x - jnp.linalg.lstsq(self.aj(x), self.R(x), rcond=None)[0]
        self.x_map = x
        J = np.asarray(self.aj(x), np.float64)
        Q, Rb = np.linalg.qr(J)
        sgn = np.sign(np.diag(Rb)); Q, Rb = Q * sgn, Rb * sgn[:, None]
        self.Q, self.Rb = jnp.asarray(Q), jnp.asarray(Rb)
        self.Rb_np, self.J0 = Rb, J
        self.resid_map = float(np.linalg.norm(Q.T @ np.asarray(self.R(x))))
        self.nD = self.l.n * self.l.D
        self.Q_ode = self.Q[-self.nD:, :]                       # ODE rows of Qb
        self.QtJconst = self.Q.T @ self.aj.Jconst               # constant part of Qb^T J

    @partial(jax.jit, static_argnums=(0,))
    def _QtJ(self, x):
        """Qb^T J(x), assembled from the pointwise ODE derivatives only."""
        n, D, P, nD = self.l.n, self.l.D, self.l.P, self.nD
        th = x[:P]; X = x[P:P + nD].reshape(n, D)
        Z = jnp.concatenate([X, jnp.broadcast_to(th, (n, P))], axis=1)
        dl = self.aj.dloc(Z, self.aj.I)
        blk = self.aj.b * jnp.einsum('djn,jde->ndje', self.aj.Lk, dl[:, :, :D]).reshape(nD, nD)
        bth = self.aj.b * jnp.einsum('dmn,mdq->ndq', self.aj.Lk, dl[:, :, D:]).reshape(nD, P)
        return self.QtJconst + self.Q_ode.T @ jnp.concatenate([bth, blk], axis=1)

    @partial(jax.jit, static_argnums=(0, 3))
    def _gn_chunk(self, xi, x0, n_it):
        """Gauss-Newton on Qb^T R(x) = xi, with backtracking on ||Qb^T R - xi||."""
        def body(x, _):
            F = self.Rv(x) @ self.Q - xi
            A = jax.vmap(self._QtJ)(x)
            dx = jnp.linalg.solve(A, F[:, :, None])[:, :, 0]
            def trial(t):
                xt = x - t * dx
                return xt, jnp.linalg.norm(self.Rv(xt) @ self.Q - xi, axis=1)
            cands = [trial(t) for t in (1.0, 0.5, 0.25)]
            base = jnp.linalg.norm(F, axis=1)
            xs = jnp.stack([c[0] for c in cands] + [x])
            ns = jnp.stack([c[1] for c in cands] + [base])
            pick = jnp.argmin(ns, axis=0)
            x = jnp.take_along_axis(xs, pick[None, :, None], 0)[0]
            return x, jnp.max(jnp.min(ns, axis=0))
        return jax.lax.scan(body, x0, None, length=n_it)

    @partial(jax.jit, static_argnums=(0,))
    def _diag_chunk(self, X, xi):
        """exact log-weights and the local contraction factor, per particle"""
        A = jax.vmap(self._QtJ)(X)
        _, ld = jnp.linalg.slogdet(A)
        r2 = jnp.sum(self.Rv(X) ** 2, axis=1)
        M = jax.vmap(lambda a: jnp.eye(a.shape[0], dtype=a.dtype)
                     - jax.scipy.linalg.solve_triangular(self.Rb, a, lower=False))(A)
        kap = jnp.linalg.norm(M, ord=2, axis=(1, 2))
        return -ld - 0.5 * r2 + 0.5 * jnp.sum(xi ** 2, axis=1), kap

    def run(self, xi, n_it=6, chunk=100):
        X, res, lw, kap = [], [], [], []
        for s in range(0, len(xi), chunk):
            xc = jnp.asarray(xi[s:s + chunk])
            x0 = jnp.tile(self.x_map, (len(xc), 1))
            xs, hist = self._gn_chunk(xc, x0, n_it)
            l, k = self._diag_chunk(xs, xc)
            X.append(np.asarray(xs)); res.append(float(hist[-1]))
            lw.append(np.asarray(l)); kap.append(np.asarray(k))
        return np.concatenate(X), np.concatenate(lw), np.concatenate(kap), max(res)


def ess_of(lw):
    w = np.exp(lw - lw.max()); w /= w.sum()
    return float(1.0 / np.sum(w ** 2)), w
