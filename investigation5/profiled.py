"""
Rao-Blackwellised MAGI posterior: profile the states out, solve the parameter problem properly.

Investigation 4 spent its effort correcting a d-dimensional Gaussian and kept running into the
same wall: the mean error is first order in the ODE's nonlinearity, the corrections are
perturbative, and they stop working exactly when the problem stops being nearly Gaussian. But
every metric in that investigation also said the error that matters lives in theta, and p is
small -- 3, 5, 7 across the test systems -- while the 300 to 600 state dimensions are what make
the problem hard. So do not approximate in the hard directions at all. Integrate them out:

    p(theta) = int exp(-U(theta, X)) dX
             ~ exp(-U(theta, X*(theta))) det H_XX(theta, X*(theta))^(-1/2),     X* = argmin_X U

This is the Laplace approximation applied ONLY to the inner integral, and it is a different
approximation from the joint Laplace in a way that matters:

  * it is EXACT whenever f is affine in the state at fixed theta (condition A), where the joint
    Laplace still carries an O(Lambda) mean error, because the joint version linearises the
    theta-X coupling and this one does not;
  * what remains is the non-Gaussianity of p(X | theta) alone, which the GP prior and the data
    constrain far more tightly than they constrain theta;
  * it makes no Gaussian assumption whatsoever about theta -- skew, heavy tails and curvature in
    the parameter marginals all survive.

The p-dimensional integral over theta is then done by self-normalised importance sampling from
the Laplace theta-marginal, on a fixed randomised-QMC point set so the result is deterministic
given a seed. That proposal is the right one to use: it already has the correct location and
approximately the correct spread, and where it is wrong it is typically too WIDE (the Laplace
standard deviation for one FitzHugh-Nagumo parameter is 1.7x the truth), which is the harmless
direction for importance sampling. Effective sample size and the Pareto k-hat of the weights are
reported and need no reference chain.

The output is not a Gaussian. It is a MIXTURE, one Gaussian in X per theta node,

    p(x) ~ sum_i w_i  delta(theta - theta_i) N(X; X*(theta_i), H_XX(theta_i)^-1),

from which every moment follows in closed form -- including Cov(theta, X), which no Gaussian
approximation centred at the mode gets right.
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial


class ProfiledPosterior:
    def __init__(self, m, n_nodes=512, seed=0, inner_iters=6, damp=1e-10, batch=64,
                 scale=1.0, jitter=1e-10, inner_pairs=0):
        self.m, self.n_nodes, self.seed = m, n_nodes, seed
        self.inner_iters, self.damp, self.batch = inner_iters, damp, batch
        self.scale, self.jitter = scale, jitter
        self.inner_pairs = inner_pairs
        self.mu_prop = self.L_prop = None
        self._built = False

    # ------------------------------------------------------------------ inner problem
    def _make_inner(self):
        m = self.m
        gn = m._gn_solver()
        p, n, D, nD = m.p, m.n, m.D, gn.nD
        sig = m.sigmas
        hess = m._hessian_fn()
        dt = m.mu.dtype
        eyeX = jnp.eye(nD, dtype=dt)

        npair = int(self.inner_pairs)
        # Common random numbers across theta nodes. What the weights need is the RATIO of inner
        # integrals between nodes, so sharing the draws cancels most of the Monte Carlo error
        # rather than adding it independently at every node.
        Zc = (jnp.asarray(np.random.default_rng(12345).standard_normal((npair, nD)), dt)
              if npair else jnp.zeros((0, nD), dt))

        def inner(theta, X0):
            """Profile: minimise U over X at fixed theta, then the Laplace weight."""
            def body(X, _):
                A, g, _r = gn._normal_equations(theta, X, sig)
                Axx, gx = A[p:, p:], g[p:]
                Axx = Axx + self.damp * jnp.trace(Axx) / nD * eyeX
                dX = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Axx), -gx)
                return X + dX.reshape(n, D), None
            X, _ = jax.lax.scan(body, X0, None, length=self.inner_iters)
            x = jnp.concatenate([theta, X.ravel()])
            # the profile is only the profile if the inner problem actually converged; an
            # unconverged X* biases U_prof and log det H_XX together, and at three inner
            # iterations instead of six that alone costs an order of magnitude of accuracy
            _A, _g, _r = gn._normal_equations(theta, X, sig)
            gres = jnp.linalg.norm(_g[p:]) / jnp.sqrt(jnp.asarray(nD, _g.dtype))
            Hxx = hess(x)[p:, p:]
            Hxx = 0.5 * (Hxx + Hxx.T) + self.jitter * jnp.trace(Hxx) / nD * eyeX
            c, low = jax.scipy.linalg.cho_factor(Hxx), True
            logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(c[0]))))
            ok = jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(jnp.diag(c[0]))) \
                 & (jnp.min(jnp.diag(c[0])) > 0)
            lw = m.logdensity(x, m.data) - 0.5 * logdet

            # Correct the INNER Laplace approximation. Writing the exact inner integral as
            #     log int e^-U dX = -U* - 0.5 logdet H_XX + const + log E_xi[e^-Delta],
            # with xi ~ N(0, H_XX^-1) and Delta the non-quadratic remainder, the pure Laplace
            # weight drops the last term. Antithetic pairs cancel Delta's cubic part exactly, so
            # a handful of draws resolve the quartic term that is actually left. This is exactly
            # zero when f is affine in the state, and is the only error the profile construction
            # carries there.
            def dlt(z):
                xi = jax.scipy.linalg.solve_triangular(c[0], z, lower=True, trans='T')
                xp = jnp.concatenate([theta, (X.ravel() + xi)])
                xm = jnp.concatenate([theta, (X.ravel() - xi)])
                q = 0.5 * jnp.sum(z ** 2)
                dp = -m.logdensity(xp, m.data) + m.logdensity(x, m.data) - q
                dm = -m.logdensity(xm, m.data) + m.logdensity(x, m.data) - q
                return 0.5 * (dp + dm)
            corr = jnp.where(npair > 0, -jnp.mean(jax.vmap(dlt)(Zc)) if npair else 0.0, 0.0)
            lw = lw + corr
            return jnp.where(ok, lw, -jnp.inf), X, ok, gres
        return jax.jit(jax.vmap(inner, in_axes=(0, 0)))

    # ------------------------------------------------------------------ build
    def check(self):
        """At theta = theta_MAP the inner solve must reproduce the MAP trajectory."""
        m = self.m
        p = m.p
        x0 = np.asarray(m.map_particle, np.float64)
        inner = self._make_inner()
        dt = m.mu.dtype
        th = jnp.asarray(x0[None, :p], dt)
        X0 = jnp.asarray(x0[p:].reshape(m.n, m.D), dt)[None]
        lw, Xs, ok = inner(th, X0)
        err = float(np.linalg.norm(np.asarray(Xs[0], np.float64).ravel() - x0[p:]) /
                    max(np.linalg.norm(x0[p:]), 1e-300))
        return err, float(lw[0]), bool(ok[0])

    def adapt(self, rounds=3, inflate=1.2, min_ess_frac=0.02, ladder=True, verbose=True):
        """
        Population Monte Carlo on the proposal.

        The Laplace theta-marginal is centred at the mode, and the mode is exactly what is wrong:
        on FitzHugh-Nagumo it sits about 1.9 standard deviations from the posterior mean, and a
        shift of that size costs a factor exp(-1.9^2/2) ~ 0.17 in effective sample size all by
        itself. So the low ESS of the first pass is not a defect of the estimator, it is a
        measurement of the bias -- and re-centring the proposal on the weighted moments removes
        it. The covariance is inflated slightly each round so the tails stay covered, and a round
        that loses effective sample size is rejected rather than accepted.
        """
        best = None
        full = self.n_nodes
        # Early rounds only have to relocate the proposal, which needs far fewer nodes than the
        # final estimate does; spending the full budget on them is most of the cost for none of
        # the accuracy.
        sizes = ([max(full // (2 ** (rounds - 1 - r)), 64) for r in range(rounds)]
                 if ladder else [full] * rounds)
        sizes[-1] = full
        for r in range(rounds):
            self.n_nodes = sizes[r]
            self.build(verbose=False)
            rec = (self.ess, self.mu_prop.copy(), self.L_prop.copy(), self.khat)
            if verbose:
                print(f'    round {r}: n {self.n_nodes:>5}  ESS {self.ess:>7.1f} '
                      f'({self.ess/self.n_nodes:>5.1%})  khat {self.khat:>5.2f}  '
                      f'failed {int((~self.ok).sum()):>4}')
            if best is None or self.ess > best[0]:
                best = (self.ess, self.mu_prop.copy(), self.L_prop.copy(),
                        dict(TH=self.TH, log_p=self.log_p, log_q=self.log_q,
                             Xstar=self.Xstar, ok=self.ok, w=self.w, khat=self.khat,
                             ess=self.ess))
            if self.ess / self.n_nodes < min_ess_frac:
                self.mu_prop = self.theta_mean if np.all(np.isfinite(self.theta_mean)) else self.mu_prop
                self.L_prop = self.L_prop * 2.0          # too few survivors: widen, do not chase
                continue
            C = self.theta_cov * inflate ** 2
            C = 0.5 * (C + C.T)
            wv, Vv = np.linalg.eigh(C)
            wv = np.maximum(wv, 1e-14 * max(wv.max(), 1.0))
            self.mu_prop = self.theta_mean
            self.L_prop = (Vv * np.sqrt(wv)) @ Vv.T
        st = best[3]
        self.TH, self.log_p, self.log_q = st["TH"], st["log_p"], st["log_q"]
        self.Xstar, self.ok, self.w = st["Xstar"], st["ok"], st["w"]
        self.khat, self.ess = st["khat"], st["ess"]
        self.mu_prop, self.L_prop = best[1], best[2]
        self.n_nodes = len(self.w)
        self._built = True
        if verbose:
            print(f'    {self}')
        return self

    def build(self, verbose=True):
        import time
        m = self.m
        p = m.p
        t = {}
        t0 = time.time()
        if getattr(m, "map_particle", None) is None:
            m.map_solve(verbose=False)
        x0 = np.asarray(m.map_particle, np.float64)
        H = np.asarray(m.hessian(x0), np.float64); H = 0.5 * (H + H.T)
        d = np.sqrt(np.maximum(np.diag(H), np.finfo(float).tiny))
        Hs = H * np.outer(1 / d, 1 / d)
        w, V = np.linalg.eigh(0.5 * (Hs + Hs.T))
        w = np.maximum(w, 1e-12 * max(w.max(), 1.0))
        Sig = ((V / w) @ V.T) / np.outer(d, d)              # Jacobi-stabilised H^-1
        Sth = Sig[:p, :p]
        Sth = 0.5 * (Sth + Sth.T)
        wt, Vt = np.linalg.eigh(Sth)
        wt = np.maximum(wt, 1e-14 * max(wt.max(), 1.0))
        Lth = (Vt * np.sqrt(wt)) @ Vt.T * self.scale        # proposal sd, symmetric root
        if getattr(self, "mu_prop", None) is None:
            self.mu_prop = x0[:p].copy()
        if getattr(self, "L_prop", None) is None:
            self.L_prop = Lth
        t["setup"] = time.time() - t0

        # randomised QMC in the proposal's whitened coordinates
        rng = np.random.default_rng(self.seed)
        try:
            from scipy.stats import qmc, norm
            u = qmc.Sobol(d=p, scramble=True, seed=self.seed).random(self.n_nodes)
            Z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        except Exception:
            Z = rng.standard_normal((self.n_nodes, p))
        TH = self.mu_prop[None, :] + Z @ self.L_prop.T
        self.log_q = -0.5 * np.sum(Z ** 2, axis=1)          # up to a constant

        inner = self._make_inner()
        dt = m.mu.dtype
        X0 = jnp.asarray(x0[p:].reshape(m.n, m.D), dt)
        lw, Xs, oks, grs = [], [], [], []
        t0 = time.time()
        for s in range(0, self.n_nodes, self.batch):
            th = jnp.asarray(TH[s:s + self.batch], dt)
            a, b, c, gg = inner(th, jnp.broadcast_to(X0, (th.shape[0],) + X0.shape))
            lw.append(np.asarray(a, np.float64)); Xs.append(np.asarray(b, np.float64))
            oks.append(np.asarray(c)); grs.append(np.asarray(gg, np.float64))
        t["profile"] = time.time() - t0
        self.log_p = np.concatenate(lw); self.Xstar = np.concatenate(Xs)
        self.ok = np.concatenate(oks); self.inner_grad = np.concatenate(grs)
        self.TH, self.x0, self.Sig, self.H, self.t = TH, x0, Sig, H, t

        lr = self.log_p - self.log_q
        lr = np.where(np.isfinite(lr) & self.ok, lr, -np.inf)
        lr -= lr.max()
        wts = np.exp(lr); wts /= wts.sum()
        self.w, self.log_ratio = wts, lr
        self.ess = float(1.0 / np.sum(wts ** 2))
        self.khat = _pareto_k(np.exp(lr[np.isfinite(lr)]))
        self._built = True
        if verbose:
            print(self)
        return self

    # ------------------------------------------------------------------ moments
    @property
    def theta_mean(self):
        return self.w @ self.TH

    @property
    def theta_cov(self):
        d = self.TH - self.theta_mean
        return (d * self.w[:, None]).T @ d

    @property
    def X_mean(self):
        return np.einsum('i,ijk->jk', self.w, self.Xstar)

    def sample(self, k=1000, seed=0, unpack=True):
        """Draw from the mixture: pick a theta node, then X from its conditional Gaussian."""
        m = self.m
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.w), size=k, p=self.w)
        p = m.p
        out = np.empty((k, self.x0.shape[0]))
        hess = m._hessian_fn()
        dtj = m.mu.dtype
        for j in np.unique(idx):
            sel = np.where(idx == j)[0]
            x = np.concatenate([self.TH[j], self.Xstar[j].ravel()])
            Hxx = np.asarray(hess(jnp.asarray(x, dtj)), np.float64)[p:, p:]
            Hxx = 0.5 * (Hxx + Hxx.T)
            ww, VV = np.linalg.eigh(Hxx)
            ww = np.maximum(ww, 1e-12 * max(ww.max(), 1.0))
            L = (VV / np.sqrt(ww)) @ VV.T
            out[sel, :p] = self.TH[j]
            out[sel, p:] = self.Xstar[j].ravel() + rng.standard_normal((len(sel), L.shape[0])) @ L.T
        return m.unpack_particles(jnp.asarray(out)) if unpack else out

    def __repr__(self):
        if not self._built:
            return "ProfiledPosterior(unbuilt)"
        g = self.inner_grad[self.ok]
        return (f'ProfiledPosterior(n={self.n_nodes}, ESS={self.ess:.0f} '
                f'({self.ess/self.n_nodes:.1%}), khat={self.khat:.2f}, '
                f'failed nodes={int((~self.ok).sum())})\n'
                f'  inner solve: max ||grad_X||/sqrt(nD) = '
                f'{(g.max() if len(g) else np.nan):.2e} (weighted mean '
                f'{float(self.w[self.ok] @ g / max(self.w[self.ok].sum(), 1e-300)):.2e})\n'
                f'  cost {sum(self.t.values()):.2f}s  {self.t}')


def _pareto_k(w):
    """Crude Pareto k-hat of the largest importance weights (Vehtari et al.)."""
    w = np.sort(np.asarray(w)[np.isfinite(w)])
    n = len(w)
    if n < 50:
        return np.nan
    M = max(int(min(0.2 * n, 3 * np.sqrt(n))), 10)
    tail = w[-M:]
    u = tail[0]
    x = tail - u
    x = x[x > 0]
    if len(x) < 5:
        return np.nan
    # method of moments on the generalised Pareto
    mu, s2 = x.mean(), x.var()
    if s2 <= 0:
        return np.nan
    k = 0.5 * (1 - mu ** 2 / s2)
    return float(k)
