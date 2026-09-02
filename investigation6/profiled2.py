"""
Profiled posterior, second generation: build the proposal from the profile's own geometry.

The first version drew theta from the joint Laplace marginal N(theta*, (H^-1)_theta) and adapted.
That proposal is derived from the wrong object. (H^-1)_theta accounts for the theta-X coupling
only to first order, whereas the profile accounts for it exactly, and when the coupling is
strongly nonlinear the two differ enormously: on Hes1 the log importance weights span 8400 nats
and the effective sample size collapses to 0.6%, recovering to 16% only after ten rounds of
adaptation or by shrinking the proposal by 3x by hand.

The profiled marginal's own mode and curvature are directly computable, and cheaply, because the
parameter space is small -- p is 3 to 7 across every test system while the state space is 300 to
600. So:

    1. joint Gauss-Newton MAP                                    (~1 s, gives X*(theta) warm start)
    2. Newton on log p_hat(theta) in p dimensions, by central
       differences through the profile                           (~p^2 profile solves per step)
    3. importance sampling from N(theta_hat, H_prof^-1) on a
       scrambled Sobol set, or Gauss-Hermite when p is small
    4. ESS and Pareto k-hat as reference-free diagnostics

Step 2 is worth stating plainly: theta_hat is NOT the joint MAP's theta. The joint mode maximises
U(theta, X); the profiled mode maximises U(theta, X*(theta)) + 0.5 log det H_XX(theta), and the
determinant term moves it. That term is the volume of the state space consistent with theta, and
ignoring it is what makes the joint mode a biased estimate of the parameters in the first place.

Finite differences rather than autodiff for step 2: differentiating log p_hat means
differentiating through an argmin and through a log-determinant, which needs third derivatives of
U, while p is small enough that O(p^2) extra profile solves is cheaper than assembling them.
"""
import numpy as np, jax, jax.numpy as jnp, time


class ProfiledPosterior2:
    def __init__(self, m, n_nodes=512, seed=0, inner_iters=3, damp=1e-10, batch=64,
                 jitter=1e-10, fd_rel=0.02, inflate=1.3):
        self.m, self.n_nodes, self.seed = m, n_nodes, seed
        self.inner_iters, self.damp, self.batch = inner_iters, damp, batch
        self.jitter, self.fd_rel, self.inflate = jitter, fd_rel, inflate
        self._inner = None
        self._S = None
        self.t = {}

    # ------------------------------------------------------------------ profile evaluation
    def _make_inner(self):
        if self._inner is not None:
            return self._inner
        m = self.m
        gn = m._gn_solver()
        p, n, D, nD = m.p, m.n, m.D, gn.nD
        sig, hess, dt = m.sigmas, m._hessian_fn(), m.mu.dtype
        eyeX = jnp.eye(nD, dtype=dt)

        def inner(theta, X0):
            def body(X, _):
                # Jacobi-scaled exactly as the MAP solve is, and for the same reason: this is a
                # normal-equations Cholesky, so it carries the SQUARE of the Jacobian's condition
                # number. On HIV that is 4e17 unscaled, and half the inner solves simply fail.
                A, g, _r = gn._normal_equations(theta, X, sig)
                Axx = A[p:, p:] + self.damp * jnp.trace(A[p:, p:]) / nD * eyeX
                dg = jnp.diag(Axx)
                dg = jnp.where(dg > jnp.finfo(dg.dtype).tiny, dg, jnp.ones_like(dg))
                Di = jax.lax.rsqrt(dg)
                As = Axx * Di[:, None] * Di[None, :]
                dX = Di * jax.scipy.linalg.cho_solve(
                    jax.scipy.linalg.cho_factor(As), -(g[p:] * Di))
                return X + dX.reshape(n, D), None
            X, _ = jax.lax.scan(body, X0, None, length=self.inner_iters)
            x = jnp.concatenate([theta, X.ravel()])
            Hxx = hess(x)[p:, p:]
            Hxx = 0.5 * (Hxx + Hxx.T) + self.jitter * jnp.trace(Hxx) / nD * eyeX
            c = jax.scipy.linalg.cho_factor(Hxx)
            dd = jnp.diag(c[0])
            ok = jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(dd)) & (jnp.min(dd) > 0)
            lw = m.logdensity(x, m.data) - jnp.sum(jnp.log(jnp.abs(dd)))
            return jnp.where(ok, lw, -jnp.inf), X, ok
        self._inner = jax.jit(jax.vmap(inner, in_axes=(0, 0)))
        return self._inner

    def _sensitivity(self):
        """
        dX*/dtheta at the joint mode, by the implicit function theorem.

        X*(theta) is defined by grad_X U(theta, X*) = 0, so differentiating gives
        H_XX dX*/dtheta + H_Xtheta = 0. Both blocks are already available from the exact Hessian,
        and the solve reuses one Cholesky for all p right-hand sides, so the whole sensitivity
        costs about as much as a single inner iteration. Starting each node from
        X_MAP + (dX*/dtheta)(theta - theta_MAP) instead of from X_MAP alone puts it a full order
        closer, which is what lets the inner iteration count come down.
        """
        if getattr(self, "_S", None) is not None:
            return self._S
        m = self.m
        p = m.p
        x0 = np.asarray(m.map_particle, np.float64)
        H = np.asarray(m.hessian(x0), np.float64); H = 0.5 * (H + H.T)
        Hxx, Hxt = H[p:, p:], H[p:, :p]
        w, V = np.linalg.eigh(0.5 * (Hxx + Hxx.T))
        w = np.maximum(w, 1e-12 * max(w.max(), 1e-300))
        self._S = -((V / w) @ (V.T @ Hxt))                      # (nD, p)
        self._th0 = x0[:p].copy()
        self._X0 = x0[p:].reshape(m.n, m.D).copy()
        return self._S

    def logp(self, TH, X0=None, predict=True):
        """Profiled log marginal at a batch of theta. Returns (logp, X*, ok)."""
        m = self.m
        dt = m.mu.dtype
        inner = self._make_inner()
        TH = np.atleast_2d(np.asarray(TH, np.float64))
        if X0 is not None:
            starts = np.broadcast_to(np.asarray(X0, np.float64),
                                     (len(TH), m.n, m.D))
        elif predict:
            S = self._sensitivity()
            starts = (self._X0.ravel()[None, :]
                      + (TH - self._th0[None, :]) @ S.T).reshape(len(TH), m.n, m.D)
        else:
            starts = np.broadcast_to(np.asarray(m.map_particle, np.float64)[m.p:]
                                     .reshape(m.n, m.D), (len(TH), m.n, m.D))
        lw, Xs, oks = [], [], []
        for s in range(0, len(TH), self.batch):
            a, b, c = inner(jnp.asarray(TH[s:s + self.batch], dt),
                            jnp.asarray(starts[s:s + self.batch], dt))
            lw.append(np.asarray(a, np.float64)); Xs.append(np.asarray(b, np.float64))
            oks.append(np.asarray(c))
        return np.concatenate(lw), np.concatenate(Xs), np.concatenate(oks)

    # ------------------------------------------------------------------ profile mode
    def profile_mode(self, th0, sd0, max_steps=8, tol=1e-4, verbose=False):
        """
        Newton on log p_hat(theta) using central differences on a fixed relative stencil.

        The step is damped and rejected unless it improves log p_hat, so a bad curvature estimate
        costs an iteration rather than the answer. Returns (theta_hat, H_prof, log p_hat).
        """
        p = self.m.p
        th = np.asarray(th0, np.float64).copy()
        sd0 = np.maximum(np.asarray(sd0, np.float64), 1e-12)
        # The stencil must sit inside the region where the profile is locally quadratic, and the
        # JOINT Laplace sd is the wrong yardstick for that: on Hes1 it overstates the profile's
        # own width so badly that a 0.25-sd stencil straddles a cliff and the differences are
        # meaningless. Start small and let the measured curvature set the scale.
        h = self.fd_rel * sd0
        f0 = self.logp(th[None, :])[0][0]
        th_best, f_best, H_best = th.copy(), f0, None
        for it in range(max_steps):
            pts, idx = [th], {}
            for i in range(p):                                  # gradient + diagonal
                for s in (+1, -1):
                    idx[(i, i, s)] = len(pts)
                    e = np.zeros(p); e[i] = s * h[i]; pts.append(th + e)
            for i in range(p):                                  # off-diagonal
                for j in range(i + 1, p):
                    for si in (+1, -1):
                        for sj in (+1, -1):
                            idx[(i, j, si, sj)] = len(pts)
                            e = np.zeros(p); e[i] = si * h[i]; e[j] = sj * h[j]
                            pts.append(th + e)
            vals = self.logp(np.array(pts))[0]
            f = vals[0]
            g = np.array([(vals[idx[(i, i, +1)]] - vals[idx[(i, i, -1)]]) / (2 * h[i])
                          for i in range(p)])
            Hp = np.zeros((p, p))
            for i in range(p):
                Hp[i, i] = (vals[idx[(i, i, +1)]] - 2 * f + vals[idx[(i, i, -1)]]) / h[i] ** 2
            for i in range(p):
                for j in range(i + 1, p):
                    v = (vals[idx[(i, j, +1, +1)]] - vals[idx[(i, j, +1, -1)]]
                         - vals[idx[(i, j, -1, +1)]] + vals[idx[(i, j, -1, -1)]]) / (4 * h[i] * h[j])
                    Hp[i, j] = Hp[j, i] = v
            Hn = -0.5 * (Hp + Hp.T)                             # curvature of -log p_hat
            if not np.all(np.isfinite(Hn)) or not np.all(np.isfinite(g)):
                break
            if np.all(np.linalg.eigvalsh(Hn) > 0):
                H_best = Hn
            w, V = np.linalg.eigh(Hn)
            w = np.where(w > 0, w, np.maximum(np.abs(w), 1e-8 * max(abs(w).max(), 1e-300)))
            step = (V / w) @ V.T @ g
            done = False
            for a in (1.0, 0.5, 0.2, 0.05, 0.01):
                cand = th + a * step
                fc = self.logp(cand[None, :])[0][0]
                if np.isfinite(fc) and fc > f + 1e-10:
                    th, f0, done = cand, fc, True
                    if fc > f_best:
                        th_best, f_best = cand.copy(), fc
                    break
            if verbose:
                print(f'      newton {it}: log p_hat {f:.4f} -> {f0:.4f}  |step/sd| '
                      f'{np.max(np.abs(step) / np.maximum(sd0, 1e-300)):.3f}')
            if not done or np.max(np.abs(step) / np.maximum(sd0, 1e-300)) < tol:
                break
            # re-scale the stencil to the curvature just measured, clipped to the joint Laplace
            # width so a bad Hessian cannot send it off to infinity
            hnew = 1.0 / np.sqrt(np.maximum(np.diag(Hn), 1e-300))
            h = np.clip(self.fd_rel * hnew, 1e-4 * sd0, sd0)
        if H_best is None:
            H_best = np.diag(1.0 / sd0 ** 2)          # no PD curvature found: fall back
            th_best = np.asarray(th0, np.float64).copy()
            self.mode_ok = False
        else:
            self.mode_ok = True
        return th_best, H_best, f_best

    # ------------------------------------------------------------------ build
    def build(self, verbose=True):
        m = self.m
        p = m.p
        t0 = time.time()
        if getattr(m, "map_particle", None) is None:
            m.map_solve(verbose=False)
        x0 = np.asarray(m.map_particle, np.float64)
        H = np.asarray(m.hessian(x0), np.float64); H = 0.5 * (H + H.T)
        dsc = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
        w_, V_ = np.linalg.eigh(H / np.outer(dsc, dsc))
        keep = w_ > 1e-10 * max(abs(w_).max(), 1e-300)
        Sig = ((V_[:, keep] / w_[keep]) @ V_[:, keep].T) / np.outer(dsc, dsc)
        sd0 = np.sqrt(np.maximum(np.diag(Sig)[:p], 1e-300))
        self.t["setup"] = time.time() - t0

        t0 = time.time()
        th_hat, Hn, f_hat = self.profile_mode(x0[:p], sd0, verbose=verbose)
        self.t["mode"] = time.time() - t0
        self.theta_hat, self.H_prof = th_hat, Hn

        wq, Vq = np.linalg.eigh(0.5 * (Hn + Hn.T))
        wq = np.maximum(wq, 1e-12 * max(wq.max(), 1e-300))
        L = (Vq / np.sqrt(wq)) @ Vq.T * self.inflate

        t0 = time.time()
        try:
            from scipy.stats import qmc, norm
            Z = norm.ppf(np.clip(qmc.Sobol(d=p, scramble=True, seed=self.seed)
                                 .random(self.n_nodes), 1e-12, 1 - 1e-12))
        except Exception:
            Z = np.random.default_rng(self.seed).standard_normal((self.n_nodes, p))
        TH = th_hat[None, :] + Z @ L.T
        lp, Xs, ok = self.logp(TH)
        self.t["nodes"] = time.time() - t0

        lq = -0.5 * np.sum(Z ** 2, axis=1)
        lr = np.where(np.isfinite(lp) & ok, lp - lq, -np.inf)
        lr = lr - lr.max()
        wts = np.exp(lr); wts /= wts.sum()
        self.TH, self.log_p, self.Xstar, self.ok, self.w = TH, lp, Xs, ok, wts
        self.ess = float(1.0 / np.sum(wts ** 2))
        self.khat = _pareto_k(np.exp(lr[np.isfinite(lr)]))
        self.log_ratio = lr
        # Reference-free gate. Importance sampling with an effective sample size of a few percent
        # is not an estimate, and on Hes1 -- where no parameter is identified and the mode already
        # sits at the reference mean -- the profiled answer is worse than doing nothing. Below the
        # threshold the caller is told to use the Laplace approximation instead.
        self.reliable = bool(self.ess / self.n_nodes >= 0.10
                             and np.isfinite(self.khat) and self.khat < 0.7
                             and getattr(self, "mode_ok", True))
        if verbose:
            print(f'    {self}')
        return self

    @property
    def theta_mean(self):
        return self.w @ self.TH

    @property
    def theta_cov(self):
        d = self.TH - self.theta_mean
        return (d * self.w[:, None]).T @ d

    def __repr__(self):
        return (f'ProfiledPosterior2(p={self.m.p}, n={self.n_nodes}, '
                f'ESS={self.ess:.0f} ({self.ess/self.n_nodes:.1%}), khat={self.khat:.2f}, '
                f'failed={int((~self.ok).sum())}, '
                f'{"RELIABLE" if self.reliable else "NOT reliable -- use the Laplace"})'
                f'  cost {sum(self.t.values()):.2f}s {self.t}')


def _pareto_k(w):
    w = np.sort(np.asarray(w)[np.isfinite(w)])
    n = len(w)
    if n < 50:
        return np.nan
    M = max(int(min(0.2 * n, 3 * np.sqrt(n))), 10)
    x = w[-M:] - w[-M]
    x = x[x > 0]
    if len(x) < 5:
        return np.nan
    mu, s2 = x.mean(), x.var()
    return float(0.5 * (1 - mu ** 2 / s2)) if s2 > 0 else np.nan
