'''
Gauss-Newton MAP solver for MAGI.

With sigma held fixed the MAGI log-density is exactly a sum of squares,

    -2 log p(theta, X) = ||R(theta, X)||^2 + const,
    R = [ sqrt(b) Lc^T (X - mu) ;  (X - x_obs)/sigma on observed ;
          sqrt(b) Lk^T (f(X, theta) - mudot - m(X - mu)) ;
          P^(1/2) (theta - theta_mean) ]      with P the theta prior precision,

for Lc Lc^T = C^-1, Lk Lk^T = K^-1 and b = beta_inv, so the mode is a nonlinear least-squares
problem and Gauss-Newton applies with its quadratic local convergence. This matters because the
posterior Hessian is conditioned around 1e4, where a first-order optimizer stalls far from the
mode while appearing to converge: on FitzHugh-Nagumo, solve(is_MAP=True) with Prodigy terminates
at ||grad|| = 1.3e3, while Gauss-Newton reaches 1e-6 -- a log-density 8.7 nats higher -- in a
fraction of the time. Anything built on the mode or the Hessian there (a Laplace approximation,
a preconditioner, a starting ensemble) inherits that error.

Two structural facts keep it cheap.

  * Only the ODE block of R depends on (theta, X), and only through the POINTWISE derivatives
    df/d(X_i, theta) -- a (D, D+p) matrix at each grid point. So the residual Jacobian is
    assembled from n small autodiff calls on the user's ODE plus constant precomputed blocks,
    rather than the dim forward-mode passes jax.jacfwd would take through the whole residual.
  * The GP and observation blocks contribute a CONSTANT term to J^T J, so the normal equations
    are formed without ever materialising J.

The price of the normal equations is that they square the condition number, so the linear solve
is Jacobi-scaled; see _run. Without it, a problem whose only sin is heterogeneous units -- rate
constants spanning orders of magnitude, which is most real ODE models -- silently returns a
non-stationary point.

Usage
-----
    from gauss_newton import GaussNewtonMAP
    gn = GaussNewtonMAP(magi)          # precompute + compile once
    Xs, thetas, sigmas = gn.solve()    # reuse across calls, no recompilation
'''
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


class GaussNewtonMAP:
    def __init__(self, magi):
        m = self.m = magi
        if not m._put_called:
            m.put(dtype=jnp.float32)
        dt = self.dtype = m.mu.dtype
        n, D, p = m.n, m.D, m.p
        self.n, self.D, self.p = n, D, p
        self.nD = nD = n * D
        self.dim = p + nD
        self.n_unknown = int(jnp.sum(m.unknown_sigmas))

        jit_eye = lambda k: jnp.eye(k, dtype=dt)
        # These factors define the residual, so they are ALWAYS computed in float64 and only then
        # cast to the working dtype. Factoring them in float32 does not merely lose accuracy, it
        # changes the problem: the ridge needed to make a float32 Cholesky of C^-1 succeed is
        # ~5e-4 relative, and a residual built from that is the residual of a different model than
        # magi_logdensity scores. Every float32 mode then fails the gradient check by whole
        # standard deviations while the solver reports convergence, because it has converged --
        # to the wrong objective. In float64 the same matrices need ~1e-12 and the two agree.
        # m._C_invs64/_K_invs64 are snapshots taken before put() downcast anything; fall back to
        # promoting whatever is on the model for a caller that built one by hand.
        C64 = np.asarray(getattr(m, "_C_invs64", m.C_invs), np.float64)
        K64 = np.asarray(getattr(m, "_K_invs64", m.K_invs), np.float64)
        self.chol_ridge = {}
        def chol(A, tag=None):
            A = 0.5 * (A + A.T)
            scale = np.trace(A) / A.shape[0]
            eye = np.eye(A.shape[0])
            r = 4096.0 * float(np.finfo(np.float64).eps)
            for _ in range(16):
                try:
                    L = np.linalg.cholesky(A + r * scale * eye)
                    if np.all(np.isfinite(L)):
                        break
                except np.linalg.LinAlgError:
                    pass
                r *= 10.0
            else:
                raise RuntimeError(f'Cholesky of the GP precision {tag} failed up to a ridge of '
                                   f'{r:.1e} relative; the matrix is not usable.')
            self.chol_ridge[tag] = r
            return jnp.asarray(L, dt)
        self.Lc = jnp.stack([chol(C64[j], ("C", j)) for j in range(D)])   # (D, n, n) lower
        self.Lk = jnp.stack([chol(K64[j], ("K", j)) for j in range(D)])
        self.b = jnp.sqrt(jnp.asarray(m.beta_inv, dt))
        # Gaussian prior on theta as a residual block: ||P^(1/2)(theta - mean)||^2 is exactly the
        # prior's contribution to -2 log p, so the least-squares form survives a full precision
        # matrix, not just a diagonal one. The root comes from an eigendecomposition rather than a
        # Cholesky because P is allowed to be singular -- a zero precision is a flat prior, and
        # Cholesky would fail on it.
        self.theta_prec = jnp.asarray(m.theta_prec, dt)
        self.theta_mean = jnp.asarray(m.theta_mean, dt)
        pw, pv = jnp.linalg.eigh(0.5 * (self.theta_prec + self.theta_prec.T))
        self.theta_root = (jnp.sqrt(jnp.maximum(pw, 0.0))[:, None] * pv.T)   # root^T root = P
        self.I = m.I.ravel()

        # Constant Jacobian blocks. Index convention: residual row (i,d) -> i*D+d, and the X
        # column (j,e) -> j*D+e, so both live in an (n, D, n, D) tensor before reshaping.
        eyeD = jit_eye(D)
        Jgp = (self.b * jnp.einsum('dji,de->idje', self.Lc, eyeD)).reshape(nD, nD)
        LkT_ms = jnp.einsum('dmi,dmj->dij', self.Lk, m.ms)               # Lk[d]^T @ ms[d]
        self.Jode_lin = (-self.b * jnp.einsum('dij,de->idje', LkT_ms, eyeD)).reshape(nD, nD)
        self.Jgp = Jgp
        with jax.default_matmul_precision('highest'):                    # see _normal_equations
            self.JgpTJgp = Jgp.T @ Jgp                                   # constant X-X block

        def f_local(z, t):                                               # R^(D+p) -> R^D
            return m.ode(z[:D][None, :], z[D:], t[None])[0]
        # reverse mode: the local map has D outputs against D+p inputs, so it costs D passes
        # rather than D+p. Only the user's ODE is ever differentiated, and only pointwise.
        self.dloc = jax.vmap(jax.jacrev(f_local))

        self._run_jit = jax.jit(self._run)

    # ------------------------------------------------------------------ pieces
    def _obs_weight(self, sigmas):
        return jnp.where(self.m.tau,
                         1.0 / jnp.where(self.m.Ns > 0, sigmas, 1.0)[None, :], 0.0).ravel()

    def residual(self, theta, X, sigmas):
        '''(p,), (n,D), (D,) -> (3*n*D + p,);  ||residual||^2 = -2 log p + const at fixed sigma.

        The trailing p entries are the Gaussian prior on theta. They are last so that the three
        original blocks keep their offsets; anything slicing the ODE block must therefore ask for
        [2nD : 3nD] rather than [2nD :].'''
        m = self.m
        diff = X - m.mu
        ode_res = (m.ode(X, theta, m.I) - m.mu_dot - jnp.einsum('dnm,md->nd', m.ms, diff))
        return jnp.concatenate([
            (self.b * jnp.einsum('dmn,md->nd', self.Lc, diff)).ravel(),
            ((X - m.x_init).ravel() * self._obs_weight(sigmas)),
            (self.b * jnp.einsum('dmn,md->nd', self.Lk, ode_res)).ravel(),
            self.theta_root @ (theta - self.theta_mean)])

    def _ode_jac(self, theta, X):
        '''ODE rows of dR/d(theta, X): (n*D, p + n*D). The only x-dependent part.'''
        n, D, p, nD = self.n, self.D, self.p, self.nD
        Z = jnp.concatenate([X, jnp.broadcast_to(theta, (n, p))], axis=1)
        dl = self.dloc(Z, self.I)                                        # (n, D, D+p)
        blk = self.b * jnp.einsum('dji,jde->idje', self.Lk, dl[:, :, :D]).reshape(nD, nD)
        bth = self.b * jnp.einsum('dmi,mdq->idq', self.Lk, dl[:, :, D:]).reshape(nD, p)
        return jnp.concatenate([bth, self.Jode_lin + blk], axis=1)

    def jacobian(self, theta, X, sigmas):
        '''Full dR/d(theta, X), (3*n*D, p + n*D). Only needed for verification; the solver
        never forms it.'''
        nD, p = self.nD, self.p
        Z0 = jnp.zeros((nD, p), self.dtype)
        return jnp.concatenate([
            jnp.concatenate([Z0, self.Jgp], axis=1),
            jnp.concatenate([Z0, jnp.diag(self._obs_weight(sigmas))], axis=1),
            self._ode_jac(theta, X),
            jnp.concatenate([self.theta_root,
                             jnp.zeros((p, nD), self.dtype)], axis=1)], axis=0)

    def _normal_equations(self, theta, X, sigmas):
        '''A = J^T J, g = J^T R and ||R||^2. See _normal_equations_r, which this wraps.'''
        A, g, f, _ = self._normal_equations_r(theta, X, sigmas)
        return A, g, f

    def _normal_equations_r(self, theta, X, sigmas):
        '''
        A = J^T J, g = J^T R, ||R||^2 and the ODE residual block, without materialising J.

        The three-value _normal_equations is kept as the public form because recorded experiments
        unpack it; the exact Hessian wants r_ode as well and would otherwise recompute the whole
        residual to get it.

        Forced to full precision throughout. JAX defaults to reduced-precision float32 matmuls on
        hardware with tensor cores, which costs about four significant digits on a J^T J of this
        size (measured: 1.5e-4 relative against 8.9e-8). Since cond(J^T J) ~ 1e4 here, that
        perturbation is amplified straight into the Gauss-Newton step: in float32 the guard makes
        the recovered mode 2.75x closer to the float64 answer for about 2% more time. It is a
        no-op in float64, where matmuls are already computed at full precision.

        The guard covers the assembly of the residual and of Jode as well as the products that
        follow. Both are built from einsums against the GP factors, which XLA is equally free to
        run on tensor cores, and reduced precision there corrupts J before J^T J is ever formed.
        '''
        nD, p = self.nD, self.p
        with jax.default_matmul_precision('highest'):
            r = self.residual(theta, X, sigmas)
            r_gp, r_obs, r_ode, r_pri = r[:nD], r[nD:2 * nD], r[2 * nD:3 * nD], r[3 * nD:]
            w = self._obs_weight(sigmas)
            Jode = self._ode_jac(theta, X)
            A = Jode.T @ Jode
            A = A.at[p:, p:].add(self.JgpTJgp)                           # constant GP block
            diag = jnp.arange(p, p + nD)
            A = A.at[diag, diag].add(w ** 2)                             # diagonal obs block
            g = Jode.T @ r_ode
            g = g.at[p:].add(self.Jgp.T @ r_gp + w * r_obs)
            A = A.at[:p, :p].add(self.theta_prec)                         # theta prior block
            g = g.at[:p].add(self.theta_root.T @ r_pri)
        # r_ode comes back too: the exact Hessian needs it for the second-derivative term and
        # would otherwise recompute the whole residual.
        return A, g, jnp.sum(r ** 2), r_ode

    # ------------------------------------------------------------------ the loop, all on device
    def _run(self, theta, X, sigmas, lam, tol, max_iter):
        """
        Levenberg-Marquardt on Jacobi-scaled normal equations.

        The scaling is not cosmetic. Forming A = J^T J SQUARES the condition number, so a Jacobian
        conditioned at 1e8 -- reachable purely by mixing units, with no ill-posedness at all --
        leaves A at the edge of float64 and its Cholesky meaningless. That is not hypothetical:
        across the four systems in tests.py the parameter blocks carry rate constants spanning
        many orders of magnitude, and on HIV, whose theta = (36, 0.108, 0.5, 1000, 3) gives a
        theta-block curvature spread of 3e-6 to 5e2,

            cond(A) = 4.4e16   ->   cond(D A D) = 3.2e7,

        which is the difference between a solve that stalls at ||grad log p|| = 1.3e-2 and one
        that reaches 7.9e-11. Hes1 gains 2.8e5. The softest direction of the unscaled Hessian sits
        100% on theta in both cases, so this is a units artefact and nothing more.

        Symmetric diagonal scaling with D = diag(A)^(-1/2),

            (D A D + lam I) s = -D g,      delta = D s,

        is identical in exact arithmetic and completely different in floating point. It also fixes
        the damping: on a matrix with unit diagonal, lam I IS Marquardt's lam diag(A), so the
        damping becomes per-coordinate and dimensionless instead of a uniform ridge scaled by the
        mean curvature -- which on a badly scaled problem damps the stiff and soft directions by
        wildly inappropriate amounts. Well-scaled problems are unaffected: FitzHugh-Nagumo gains a
        factor of 1.2 in conditioning and reaches the same mode.

        Only exactly-degenerate coordinates are excluded from the scaling, by leaving their
        entry at 1. A floor set relative to the largest diagonal is tempting and wrong: the
        legitimate curvature range on HIV spans about 1e16, so any relative floor big enough to
        matter clips real curvature and costs eight orders of magnitude in the recovered gradient.
        A coordinate whose diagonal is not even representable carries no information and no
        off-diagonal either, since A is positive semidefinite, so leaving it unscaled is safe.
        """
        p, nD, dt = self.p, self.nD, self.dtype
        eye = jnp.eye(self.dim, dtype=dt)
        tiny = jnp.asarray(jnp.finfo(dt).tiny, dt)
        lam_min = jnp.asarray(jnp.finfo(dt).eps, dt)

        def body(c):
            theta, X, sigmas, lam, it, _, stall = c
            A, g, f0, _ = self._normal_equations_r(theta, X, sigmas)
            dg = jnp.diag(A)
            dg = jnp.where(dg > tiny, dg, jnp.ones_like(dg))
            Di = jax.lax.rsqrt(dg)                                       # D = diag(A)^(-1/2)
            As = A * Di[:, None] * Di[None, :]                           # unit diagonal
            du = Di * jax.scipy.linalg.cho_solve(
                jax.scipy.linalg.cho_factor(As + lam * eye), g * Di)
            # Newton decrement, sqrt(g' A^-1 g), as the convergence measure. It is the same
            # scale-free quantity diagnose() reports: to second order it is the distance from here
            # to the mode in posterior standard deviations, so a tolerance on it means the same
            # thing on every problem. ||g|| cannot serve, because it carries the units of the
            # log-density and of theta and its floor is set by the working precision -- in float32
            # it bottoms out near 1e-2 on FitzHugh-Nagumo, so any absolute tolerance below that is
            # unreachable and the solver silently runs to max_iter. And it costs nothing: du is
            # already A^-1 g.
            # A non-positive or non-finite g'du is a failed solve, not a converged one. Clamping
            # it to zero reads as convergence and stops the iteration on its first step: on HIV in
            # float32, where the normal equations are conditioned near the limit of the format,
            # that returned a point 246 nats below the mode and called it done.
            gd = jnp.dot(g, du)
            dec = jnp.where(jnp.isfinite(gd) & (gd > 0), jnp.sqrt(jnp.abs(gd)), jnp.inf)
            tn, Xn = theta - du[:p], X - du[p:].reshape(self.n, self.D)
            f1 = jnp.sum(self.residual(tn, Xn, sigmas) ** 2)
            better = f1 < f0
            # A scale-free tolerance still has a precision floor, so stop also when the iteration
            # stops making progress: without this, float32 runs every remaining iteration for
            # nothing whenever tol happens to sit below the achievable decrement.
            stall = jnp.where(better & (f1 < f0 * (1.0 - 1e-12)), 0, stall + 1)
            theta = jnp.where(better, tn, theta)
            X = jnp.where(better, Xn, X)
            # Scaling makes lam RELATIVE to the local curvature, so the old absolute bounds are
            # the wrong size. The floor especially: on HIV the scaled normal equations are
            # conditioned at 3e10, whose smallest eigenvalue is 3e-11, so a floor of 1e-12 keeps
            # perturbing the solve at the level of the answer and the iteration stalls four
            # orders short. Machine epsilon is the meaningful floor -- damping below it cannot
            # change a unit-diagonal matrix -- but it must stay strictly positive, or a rejected
            # step at lam = 0 could never grow the damping back.
            lam = jnp.where(better, jnp.maximum(lam * 0.3, lam_min),
                            jnp.minimum(lam * 10.0, 1e6))
            if self.n_unknown:            # exact conditional maximiser; static branch
                sq = jnp.sum(jnp.where(self.m.tau, (X - self.m.x_init) ** 2, 0.0), axis=0)
                sigmas = jnp.where(self.m.unknown_sigmas,
                                   jnp.clip(jnp.sqrt(sq / jnp.maximum(self.m.Ns, 1)), min=1e-5),
                                   sigmas)
            return theta, X, sigmas, lam, it + 1, dec, stall

        def cond(c):
            return (c[5] > tol) & (c[4] < max_iter) & (c[6] < 8)

        init = (theta, X, sigmas, lam, jnp.zeros((), jnp.int32),
                jnp.asarray(jnp.inf, dt), jnp.zeros((), jnp.int32))
        theta, X, sigmas, lam, it, dec, _ = jax.lax.while_loop(cond, body, init)
        A, g, _, _ = self._normal_equations_r(theta, X, sigmas)          # at the answer
        dg = jnp.diag(A)
        dg = jnp.where(dg > tiny, dg, jnp.ones_like(dg))
        Di = jax.lax.rsqrt(dg)
        du = Di * jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(A * Di[:, None] * Di[None, :] + lam_min * eye), g * Di)
        gd = jnp.dot(g, du)
        dec = jnp.where(jnp.isfinite(gd) & (gd > 0), jnp.sqrt(jnp.abs(gd)), jnp.inf)
        return theta, X, sigmas, it, jnp.linalg.norm(g), dec

    # ------------------------------------------------------------------ public entry point
    def check_jacobian(self, theta=None, X=None, sigmas=None):
        '''Relative disagreement between the analytic Jacobian and jax.jacfwd. The analytic form
        encodes the layout of magi_logdensity, so this is what catches a change to it.'''
        m, p, n, D = self.m, self.p, self.n, self.D
        if theta is None:
            u = jnp.asarray(m.particles_init, self.dtype)
            theta, X = u[:p], u[p:p + self.nD].reshape(n, D)
        sigmas = m.sigmas if sigmas is None else sigmas
        Ja = self.jacobian(theta, X, sigmas)
        Jn = jax.jacfwd(lambda u: self.residual(u[:p], u[p:].reshape(n, D), sigmas))(
            jnp.concatenate([theta, X.ravel()]))
        return float(jnp.linalg.norm(Ja - Jn) / jnp.linalg.norm(Jn))

    def solve(self, x0=None, max_iter=100, tol=1e-3, lm_init=1e-10, verbose=True, check=True):
        '''
        Gauss-Newton descent on ||R||^2, Levenberg-Marquardt damped so the iteration is monotone,
        with the whole loop inside one jitted lax.while_loop -- no host synchronisation per step
        and one compilation per solver instance, reused across calls.

        Unknown sigmas are handled by block coordinate descent: Gauss-Newton on (theta, X) at
        fixed sigma alternating with the exact conditional maximiser
        sigma_d^2 = sum_obs (X - y)^2 / N_d. Both blocks decrease the same objective.

        x0       : (dim,) starting particle; defaults to magi.particles_init
        max_iter : cap on Gauss-Newton steps
        tol      : stop when the Newton decrement sqrt(g' A^-1 g) falls below this. Scale free:
            to second order it is the remaining distance to the mode in posterior standard
            deviations, so 1e-3 means the mode is located to a thousandth of one on any problem in
            any precision. A tolerance on ||grad log p|| cannot do this -- it carries units and its
            attainable floor depends on the dtype. The iteration also stops if it stagnates, which
            is what happens when the requested tolerance sits below the precision floor.
        lm_init  : initial Levenberg-Marquardt damping. Dimensionless: the normal equations
            are Jacobi-scaled to a unit diagonal before damping, so this is a fraction of the
            local curvature in every coordinate rather than a ridge in absolute units. Starting
            small is close to free, since a rejected step raises it tenfold, whereas starting too
            large can leave the iteration oscillating short of the mode: on HIV, 1e-8 stalls at
            ||grad log p|| = 1.5e-3 where anything at or below 1e-10 reaches 1.2e-7. The other
            three test systems are insensitive across 1e-8 to 1e-14.
        check    : verify the analytic Jacobian against jax.jacfwd before iterating

        Returns the unpacked (Xs, thetas, sigmas) at the mode and leaves the particle on
        self.map_particle. Does not touch magi.particles.
        '''
        m, p, nD = self.m, self.p, self.nD
        u = jnp.asarray(m.particles_init if x0 is None else x0, self.dtype)
        theta, X = u[:p], u[p:p + nD].reshape(self.n, self.D)

        if check:
            rel = self.check_jacobian(theta, X, m.sigmas)
            # a genuine layout error shows up at O(0.1); float32 roundoff on this assembly is
            # O(1e-4), so the tolerance has to follow the dtype
            tol_j = 1e-5 if jnp.dtype(self.dtype) == jnp.float64 else 1e-3
            if not rel < tol_j:
                raise RuntimeError(
                    f'analytic residual Jacobian disagrees with jax.jacfwd (relative {rel:.2e}, '
                    f'tolerance {tol_j:.0e}). The analytic form encodes the layout of '
                    'magi_logdensity; if that changed it must be updated. Pass check=False to '
                    'bypass.')

        theta, X, sigmas, it, gnorm, dec = self._run_jit(
            theta, X, m.sigmas, jnp.asarray(lm_init, self.dtype),
            jnp.asarray(tol, self.dtype), jnp.asarray(max_iter, jnp.int32))
        particle = jnp.concatenate([theta, X.ravel(), sigmas[m.unknown_sigmas]])
        self.map_particle = particle
        self.n_steps, self.grad_norm, self.decrement = int(it), float(gnorm), float(dec)
        if verbose:
            print(f'Gauss-Newton MAP: {int(it)} steps | distance to mode = {float(dec):.2e} sd | '
                  f'||grad log p|| = {float(gnorm):.3e} | '
                  f'log p = {float(m.logdensity(particle, m.data)):.6f}')
        return m.unpack_particles(particle[None, :])
