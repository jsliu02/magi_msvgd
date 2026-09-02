"""
Profiled inference restricted to the identifiable subspace.

HIV's diagnosis says lam is improper -- the profiled log-density falls 0.07 nats while lam moves
five orders of magnitude -- while delta, N and c are pinned to under 3%. Integrating over lam is
not merely hard, it is undefined, and it is what destroys the importance sampling: the proposal
must cover a direction with no finite width, so almost every node lands where the weight is
negligible and the effective sample size collapses to one.

The fix follows from what profiling already is. The states are handled by maximisation, not
integration, and nothing stops the same treatment being applied to parameter directions the data
does not determine. Choose an orthonormal basis B (p x k) for the directions worth integrating,
and profile over everything else:

    log p_hat(z) = -U_prof(z) - 0.5 [ log det H(z) + log det (C^T H(z)^-1 C) ],   C = [B; 0],
    U_prof(z) = min { U(x) : C^T x = z }.

The determinant is the Hessian restricted to the constraint's complement, via
det(N^T H N) = det(H) det(C^T H^-1 C) for orthonormal C -- the k-dimensional generalisation of the
rank-one identity used in investigation 4, and it again avoids ever forming the complement basis.
The constrained Gauss-Newton step is the unconstrained step plus a correction in span(A^-1 C), so
it reuses one Cholesky and costs k extra triangular solves.

What is integrated out and what is maximised over becomes a modelling choice driven by the
identifiability diagnosis, rather than being fixed by which symbols happen to be parameters.
"""
import numpy as np, jax, jax.numpy as jnp, time


class RestrictedProfile:
    def __init__(self, m, B, n_nodes=512, seed=0, inner_iters=6, damp=1e-10, batch=32,
                 jitter=1e-10, inflate=1.3):
        self.m = m
        self.B = np.asarray(B, np.float64)                     # (p, k), orthonormal columns
        self.k = self.B.shape[1]
        self.n_nodes, self.seed = n_nodes, seed
        self.inner_iters, self.damp, self.batch = inner_iters, damp, batch
        self.jitter, self.inflate = jitter, inflate
        self._inner = None

    def _make_inner(self):
        if self._inner is not None:
            return self._inner
        m = self.m
        gn = m._gn_solver()
        p, n, D, dim = m.p, m.n, m.D, gn.dim
        sig, hess, dt = m.sigmas, m._hessian_fn(), m.mu.dtype
        C = jnp.asarray(np.vstack([self.B, np.zeros((dim - p, self.k))]), dt)   # (dim, k)
        eye = jnp.eye(dim, dtype=dt)

        def inner(z, x0):
            def body(x, _):
                A, g, _r = gn._normal_equations(x[:p], x[p:].reshape(n, D), sig)
                A = A + self.damp * jnp.trace(A) / dim * eye
                # Jacobi scaling, as in the MAP solve. With S = diag(A)^(-1/2) and As = S A S,
                # A^-1 g = S As^-1 S g and A^-1 C = S As^-1 S C, so both the free step and the
                # constraint correction come from the one well-conditioned factorisation.
                dg = jnp.diag(A)
                dg = jnp.where(dg > jnp.finfo(dg.dtype).tiny, dg, jnp.ones_like(dg))
                Di = jax.lax.rsqrt(dg)
                cf = jax.scipy.linalg.cho_factor(A * Di[:, None] * Di[None, :])
                dfree = -Di * jax.scipy.linalg.cho_solve(cf, g * Di)
                AiC = Di[:, None] * jax.scipy.linalg.cho_solve(cf, Di[:, None] * C)
                M = C.T @ AiC
                rhs = z - C.T @ (x + dfree)
                t = jnp.linalg.solve(M + 1e-14 * jnp.eye(self.k, dtype=dt) * jnp.trace(M), rhs)
                return x + dfree + AiC @ t, None
            x, _ = jax.lax.scan(body, x0, None, length=self.inner_iters)
            H = hess(x)
            H = 0.5 * (H + H.T) + self.jitter * jnp.trace(H) / dim * eye
            cf = jax.scipy.linalg.cho_factor(H)
            dd = jnp.diag(cf[0])
            logdetH = 2.0 * jnp.sum(jnp.log(jnp.abs(dd)))
            M = C.T @ jax.scipy.linalg.cho_solve(cf, C)
            sgn, logdetM = jnp.linalg.slogdet(M)
            ok = (jnp.all(jnp.isfinite(x)) & (jnp.min(dd) > 0) & (sgn > 0))
            lw = m.logdensity(x, m.data) - 0.5 * (logdetH + logdetM)
            return jnp.where(ok, lw, -jnp.inf), x, ok
        self._inner = jax.jit(jax.vmap(inner, in_axes=(0, 0)))
        return self._inner

    def logp(self, Z, x0=None):
        m = self.m
        dt = m.mu.dtype
        inner = self._make_inner()
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        x0 = np.asarray(m.map_particle if x0 is None else x0, np.float64)
        X0j = jnp.asarray(x0, dt)
        lw, xs, oks = [], [], []
        for s in range(0, len(Z), self.batch):
            zz = jnp.asarray(Z[s:s + self.batch], dt)
            a, b, c = inner(zz, jnp.broadcast_to(X0j, (zz.shape[0],) + X0j.shape))
            lw.append(np.asarray(a, np.float64)); xs.append(np.asarray(b, np.float64))
            oks.append(np.asarray(c))
        return np.concatenate(lw), np.concatenate(xs), np.concatenate(oks)

    def build(self, verbose=True):
        m = self.m
        p = m.p
        t0 = time.time()
        x0 = np.asarray(m.map_particle, np.float64)
        H = np.asarray(m.hessian(x0), np.float64); H = 0.5 * (H + H.T)
        d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
        w, V = np.linalg.eigh(H / np.outer(d, d))
        keep = w > 1e-10 * max(abs(w).max(), 1e-300)
        Sig = ((V[:, keep] / w[keep]) @ V[:, keep].T) / np.outer(d, d)
        Sz = self.B.T @ Sig[:p, :p] @ self.B                    # covariance of z under Laplace
        Sz = 0.5 * (Sz + Sz.T)
        wz, Vz = np.linalg.eigh(Sz)
        wz = np.maximum(wz, 1e-14 * max(wz.max(), 1e-300))
        L = (Vz * np.sqrt(wz)) @ Vz.T * self.inflate
        z0 = self.B.T @ x0[:p]
        try:
            from scipy.stats import qmc, norm
            Zs = norm.ppf(np.clip(qmc.Sobol(d=self.k, scramble=True, seed=self.seed)
                                  .random(self.n_nodes), 1e-12, 1 - 1e-12))
        except Exception:
            Zs = np.random.default_rng(self.seed).standard_normal((self.n_nodes, self.k))
        Z = z0[None, :] + Zs @ L.T
        lp, xs, ok = self.logp(Z)
        lq = -0.5 * np.sum(Zs ** 2, axis=1)
        lr = np.where(np.isfinite(lp) & ok, lp - lq, -np.inf)
        lr -= lr.max()
        wts = np.exp(lr); wts /= wts.sum()
        self.Z, self.log_p, self.xs, self.ok, self.w = Z, lp, xs, ok, wts
        self.ess = float(1.0 / np.sum(wts ** 2))
        self.sec = time.time() - t0
        if verbose:
            print(f'    RestrictedProfile(k={self.k}/{p}, n={self.n_nodes}, '
                  f'ESS={self.ess:.0f} ({self.ess/self.n_nodes:.1%}), '
                  f'failed={int((~self.ok).sum())})  {self.sec:.1f}s')
        return self

    @property
    def z_mean(self):
        return self.w @ self.Z

    @property
    def theta_mean(self):
        """Full-theta mean: integrated directions weighted, profiled directions at their optimum."""
        return np.einsum('i,ij->j', self.w, self.xs[:, :self.m.p])
