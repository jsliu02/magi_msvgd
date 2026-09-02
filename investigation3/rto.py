"""
Randomize-then-Optimize for MAGI.

Write the posterior as p(x) ∝ exp(-||R(x)||^2/2) (verified exactly in lsq.py). Let
J = grad R(x*) at the MAP and J = Qb Rb its thin QR. RTO draws xi ~ N(0, I_d) and SOLVES

    Qb^T R(x) = xi                                                            (*)

for x. Three properties make this the right shape for MAGI:

1. EXACT FOR AFFINE R. Then Qb^T J is constant and (I - Qb Qb^T)R(x) is constant, so the
   importance weights below are constant and RTO samples the posterior exactly. Our posterior is
   nearly Gaussian, so the weights should be nearly constant -- unlike the Laplace independence
   sampler, whose log-weights had sd 14 nats.

2. SELF-CERTIFYING. The pushforward density is available in closed form,
       log w(x) = -log|det(Qb^T J(x))| - ||R(x)||^2/2 + ||xi||^2/2,
   so the sample can be exactly reweighted and the ESS reports how nonlinear the problem was.
   Nothing is assumed; the deviation from Gaussianity is measured.

3. NO MARKOV CHAIN. Each sample is an independent deterministic solve, so none of the
   transience or stickiness that defeated ULA/MALA/HMC can occur, and it is embarrassingly
   parallel.

Solving (*) with the FIXED Jacobian from the MAP gives the quasi-Newton iteration

    x <- x - Rb^{-1} (Qb^T R(x) - xi),

whose iteration matrix is I - Rb^{-1} Qb^T J(x). Hence the safety condition

    kappa = sup_x || I - Rb^{-1} Qb^T J(x) ||_2  <  1                          (C)

which simultaneously gives (i) geometric convergence of every solve, (ii) invertibility of
Qb^T J(x) so the weights are well defined, and (iii) a diffeomorphism xi -> x. For a polynomial
ODE, kappa is bounded by the field's second derivative times the state excursion, so (C) is an
explicit and checkable condition on the specified ODE -- see kappa_bound().
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import harness as H
from lsq import LSQ


class RTO:
    def __init__(self, m, x_map, dtype=jnp.float64):
        self.m, self.lsq = m, LSQ(m)
        self.R = jax.jit(self.lsq.residual)
        self.Rv = jax.jit(jax.vmap(self.lsq.residual))
        self.Jf = jax.jit(jax.jacfwd(self.lsq.residual))
        # tighten the MAP as a stationary point of ||R||^2 (Gauss-Newton); RTO needs Qb^T R(x*)=0
        x = jnp.asarray(x_map, dtype)
        for _ in range(30):
            J = self.Jf(x); r = self.R(x)
            x = x - jnp.linalg.lstsq(J, r, rcond=None)[0]
        self.x_map = x
        J = np.asarray(self.Jf(x), np.float64)
        self.J0 = J
        Q, Rb = np.linalg.qr(J)
        sgn = np.sign(np.diag(Rb)); Q, Rb = Q * sgn, Rb * sgn[:, None]      # fix QR sign
        self.Q, self.Rb = jnp.asarray(Q), jnp.asarray(Rb)
        self.Rb_np = Rb
        self.logdet_Rb = float(np.sum(np.log(np.abs(np.diag(Rb)))))
        self.resid_map = float(np.linalg.norm(np.asarray(self.Q.T @ self.R(x))))

    # ---------------------------------------------------------------- the deterministic map
    @partial(jax.jit, static_argnums=(0, 3))
    def solve(self, xi, x0, n_iter):
        """xi (k,d) -> x (k,d) by the fixed-Jacobian quasi-Newton iteration."""
        def step(x, _):
            F = self.Rv(x) @ self.Q                                   # (k,d)
            dx = jax.scipy.linalg.solve_triangular(self.Rb, (F - xi).T, lower=False).T
            return x - dx, jnp.max(jnp.linalg.norm(F - xi, axis=1))
        x, hist = jax.lax.scan(step, x0, None, length=n_iter)
        return x, hist

    # ---------------------------------------------------------------- exact importance weights
    def log_weights(self, X, xi, chunk=40):
        """log w = -log|det(Qb^T J(x))| - ||R(x)||^2/2 + ||xi||^2/2, up to a constant."""
        out, kap = [], []
        for s in range(0, len(X), chunk):
            xb = X[s:s + chunk]
            Jb = jax.vmap(self.Jf)(xb)                                # (c,N,d)
            A = jnp.einsum('nd,cne->cde', self.Q, Jb)                 # Qb^T J  (c,d,d)
            sgn, ld = jnp.linalg.slogdet(A)
            r2 = jnp.sum(self.Rv(xb) ** 2, axis=1)
            out.append(-ld - 0.5 * r2)
            M = jax.vmap(lambda a: jnp.eye(a.shape[0], dtype=a.dtype)
                         - jax.scipy.linalg.solve_triangular(self.Rb, a, lower=False))(A)
            kap.append(jnp.linalg.norm(M, ord=2, axis=(1, 2)))
        lw = jnp.concatenate(out) + 0.5 * jnp.sum(xi ** 2, axis=1)
        return lw, jnp.concatenate(kap)


def local_L2(m, X, P=3):
    """
    Sup over the visited region of the operator norm of d^2 f / d(state, params)^2, computed
    pointwise in time -- the ONLY ODE-dependent quantity in the safety bound. Zero exactly when
    f is affine in (state, params), which is the case in which RTO is exact.
    """
    n, D = m.n, m.D
    Xs = jnp.asarray(X)
    st = Xs[:, P:P + n * D].reshape(-1, n, D)
    th = Xs[:, :P]

    def f_local(z, t):                      # z = (state, params) at one time point
        return m.ode(z[:D][None, :], z[D:], t[None])[:, 0]

    def hess_norm(z, t):
        Hs = jax.jacfwd(jax.jacfwd(f_local))(z, t)          # (D, D+P, D+P)
        return jnp.max(jnp.linalg.norm(Hs, ord=2, axis=(1, 2)))

    Z = jnp.concatenate([st, jnp.repeat(th[:, None, :], n, 1)], axis=2)   # (k,n,D+P)
    T = jnp.broadcast_to(jnp.asarray(m.I).ravel()[None, :], Z.shape[:2])
    return float(jnp.max(jax.vmap(jax.vmap(hess_norm))(Z, T)))


def kappa_bound(m, lsq, rto, X):
    """
    Explicit bound on the contraction factor (C).

        kappa <= ||Rb^-1|| * sqrt(b) * ||Lk|| * L2 * ||x - x*||
                 \________ depends only on the GP/data setup ________/   \__ODE__/

    The first factor is a property of the MAGI discretization alone; the ODE enters only through
    L2. Affine f gives L2 = 0, hence kappa = 0 and RTO exact. Returned with its pieces so the
    bound's tightness against the measured kappa can be judged.
    """
    Xs = np.asarray(X, np.float64)
    dev = float(np.linalg.norm(Xs - np.asarray(rto.x_map, np.float64), axis=1).max())
    L2 = local_L2(m, Xs, lsq.P)
    c_magi = float(1.0 / np.linalg.svd(rto.Rb_np, compute_uv=False).min()) * np.sqrt(lsq.b) * \
             float(max(np.linalg.norm(np.asarray(lsq.Lk[j]), 2) for j in range(lsq.D)))
    return c_magi * L2 * dev, dict(c_magi=c_magi, L2=L2, rho=dev)
