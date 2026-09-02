"""
Analytic residual Jacobian for MAGI.

Only the ODE block of R depends on x, and only through the POINTWISE derivatives of the vector
field, df/d(X_i, theta) -- a (D, D+P) matrix at each of n grid points. Everything else is a
constant matrix fixed once. So the per-particle Jacobian costs a handful of einsums instead of
d=325 JVPs, which is what makes full Gauss-Newton affordable inside RTO.

Rows of R: [ GP (n*D) ; observation (n*D) ; ODE (n*D) ],  columns: [ theta (P) ; X (n*D) ].
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial


class AnalyticJac:
    def __init__(self, lsq):
        self.l = lsq
        m, n, D, P = lsq.m, lsq.n, lsq.D, lsq.P
        b = np.sqrt(lsq.b)
        Lc = np.asarray(lsq.Lc, np.float64); Lk = np.asarray(lsq.Lk, np.float64)
        ms = np.asarray(lsq.ms, np.float64); w = np.asarray(lsq.w_obs, np.float64)
        nD = n * D
        # --- constant blocks -------------------------------------------------------------
        Jgp = np.zeros((nD, nD))
        Jobs = np.zeros((nD, nD))
        Jode_lin = np.zeros((nD, nD))                 # the -m(X-mu) part, pushed through Lk^T
        for d in range(D):
            r = np.arange(n) * D + d                  # rows/cols carrying component d
            Jgp[np.ix_(r, r)] = b * Lc[d].T           # dR_gp[n,d]/dX[j,d] = b Lc[d,j,n]
            Jode_lin[np.ix_(r, r)] = -b * (Lk[d].T @ ms[d])
        Jobs[np.arange(nD), np.arange(nD)] = w.ravel()
        self.Jconst = jnp.asarray(np.block([
            [np.zeros((nD, P)), Jgp],
            [np.zeros((nD, P)), Jobs],
            [np.zeros((nD, P)), Jode_lin]]))
        self.Lk = jnp.asarray(Lk); self.b = b
        self.n, self.D, self.P, self.nD = n, D, P, nD
        self.I = jnp.asarray(m.I).ravel()

        def f_local(z, t):
            return m.ode(z[:D][None, :], z[D:], t[None])[0]
        self.dloc = jax.vmap(jax.jacfwd(f_local))     # (n, D, D+P)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        n, D, P, nD = self.n, self.D, self.P, self.nD
        th = x[:P]; X = x[P:P + nD].reshape(n, D)
        Z = jnp.concatenate([X, jnp.broadcast_to(th, (n, P))], axis=1)
        dl = self.dloc(Z, self.I)                     # (n, D, D+P)
        dfdx, dfdth = dl[:, :, :D], dl[:, :, D:]
        # dR_ode[n,d]/dX[j,e] = b * Lk[d,j,n] * dfdx[j,d,e]
        blk = self.b * jnp.einsum('djn,jde->ndje', self.Lk, dfdx).reshape(nD, nD)
        # dR_ode[n,d]/dtheta_q = b * sum_m Lk[d,m,n] * dfdth[m,d,q]
        bth = self.b * jnp.einsum('dmn,mdq->ndq', self.Lk, dfdth).reshape(nD, P)
        top = jnp.zeros((2 * nD, nD + P), dtype=x.dtype)
        ode = jnp.concatenate([bth, blk], axis=1)
        return self.Jconst + jnp.concatenate([top, ode], axis=0)


def verify(lsq, aj, x, tol=1e-9):
    Ja = np.asarray(aj(jnp.asarray(x)), np.float64)
    Jn = np.asarray(jax.jacfwd(lsq.residual)(jnp.asarray(x)), np.float64)
    return float(np.linalg.norm(Ja - Jn) / np.linalg.norm(Jn))
