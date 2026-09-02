"""
MAGI as nonlinear least squares.

    -2 log p(x) = ||R(x)||^2 + const,     R(x) = [ sqrt(b) Lc^T (X - mu)          GP prior
                                                   (X - y) / sigma  on observed   likelihood
                                                   sqrt(b) Lk^T r(X, theta) ]     ODE

with Lc Lc^T = C^-1, Lk Lk^T = K^-1 and b = beta_inv. The first two blocks are exactly LINEAR
in X; every nonlinearity lives in r, and r is POINTWISE in time -- f(X_i, theta) at each grid
point. Both facts are what the rest of investigation3 is built on, so this module verifies the
identity to machine precision before anything is built on top of it.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H


class LSQ:
    def __init__(self, m):
        self.m = m
        n, D, P = m.n, m.D, m.p
        self.n, self.D, self.P = n, D, P
        self.b = float(m.beta_inv)
        Cinv = np.asarray(m.C_invs, np.float64)
        Kinv = np.asarray(m.K_invs, np.float64)
        jit = lambda A: np.linalg.cholesky(0.5 * (A + A.T) + 1e-12 * np.trace(A) / len(A) * np.eye(len(A)))
        self.Lc = jnp.asarray(np.stack([jit(Cinv[j]) for j in range(D)]))   # (D,n,n) lower
        self.Lk = jnp.asarray(np.stack([jit(Kinv[j]) for j in range(D)]))
        self.mu, self.mu_dot = m.mu, m.mu_dot
        self.ms, self.I = m.ms, m.I
        self.tau = m.tau
        sig = jnp.where(m.Ns > 0, m.sigmas, 1.0)
        self.w_obs = jnp.where(m.tau, 1.0 / sig[None, :], 0.0)              # (n,D) mask/sigma
        self.x_obs = m.x_init
        self.N = 3 * n * D

    def residual(self, particle):
        """(dim,) -> (N,);  ||residual||^2 = -2 log p + const"""
        m, n, D, P = self.m, self.n, self.D, self.P
        theta = particle[:P]
        X = particle[P:P + n * D].reshape(n, D)
        diff = X - self.mu
        ode_res = m.ode(X, theta, self.I) - self.mu_dot - jnp.einsum('dnm,md->nd', self.ms, diff)
        r_gp = jnp.einsum('dmn,md->nd', self.Lc, diff)                      # Lc^T diff
        r_ode = jnp.einsum('dmn,md->nd', self.Lk, ode_res)
        r_obs = (X - self.x_obs) * self.w_obs
        return jnp.concatenate([jnp.sqrt(self.b) * r_gp.ravel(),
                                r_obs.ravel(),
                                jnp.sqrt(self.b) * r_ode.ravel()])

    def neglogp(self, particle):
        return 0.5 * jnp.sum(self.residual(particle) ** 2)


def verify(m, lsq, n_test=40, seed=0):
    """||R||^2/(-2 log p) must differ by a CONSTANT, to machine precision."""
    rng = np.random.default_rng(seed)
    z = np.load("laplace_cache.npz")
    pts = z["x_map"][None, :] + rng.standard_normal((n_test, H.DIM)) * 0.05
    lhs = np.array([float(lsq.neglogp(jnp.asarray(p))) for p in pts])
    rhs = np.array([-float(m.logdensity(jnp.asarray(p), m.data)) for p in pts])
    d = lhs - rhs
    return float(np.std(d) / max(np.std(lhs), 1e-30)), float(np.mean(d))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    m = H.build_magi(dtype=jnp.float64)
    lsq = LSQ(m)
    rel, off = verify(m, lsq)
    print(f"residual dimension N = {lsq.N}, parameter dimension d = {H.DIM}")
    print(f"||R||^2/2  vs  -log p :  relative sd of the difference = {rel:.3e}"
          f"   (constant offset {off:.6f})")
    print("  -> the identity holds exactly" if rel < 1e-10 else "  -> MISMATCH, do not proceed")
