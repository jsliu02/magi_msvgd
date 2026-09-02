"""
exp09: the constructive consequence. Run mSVGD where it can work -- on the p-dimensional
profiled marginal.

exp07 measures the collapse as Var_SVGD/Var_target ~= ln(K)/d. Nothing in that is specific to
MAGI; it says SVGD is a usable sampler at d of a few and useless at d of a few hundred. MAGI's
joint dimension is p + n*D = 306-608, which is hopeless. But its PARAMETER dimension is p = 3-7,
which is exactly the regime where exp07 says SVGD reaches 0.83-0.99 of the target variance -- and
`profiled.ProfiledPosterior` already supplies the p-dimensional profiled log marginal

    log p_hat(theta) = log p(theta, X*(theta)) - 1/2 log det H_XX(theta),

with X*(theta) obtained by a fixed number of Gauss-Newton steps warm-started from the
implicit-function predictor. That whole computation is a `jax.lax.scan` of Cholesky solves, so it
is differentiable, and SVGD can be run on it directly.

This measures whether that works: same scoring as exp01 but restricted to theta, against the
reference's theta marginal and its K-particle floor. The comparison is against `fit()`, which
integrates the same p-dimensional marginal by importance sampling on a Sobol set.

Cost note: every gradient backpropagates through `inner_iters` Cholesky factorisations of the
(nD x nD) state block per particle, so K is deliberately small -- which is the point, since at
p = 3 exp07 says K = 50 is already enough.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness7 as H
import msvgd7 as M7
from profiled import ProfiledPosterior

SYS = sys.argv[1:] or ["fn", "lorenz"]
K = int(os.environ.get("K", 64))
MAXIT = int(os.environ.get("MAXIT", 400))
INNER = int(os.environ.get("INNER", 3))
KERNELS = os.environ.get("KERNELS", "standard,reweighted").split(",")
out = {}


class ProfiledTarget:
    """A p-dimensional MAGI target: log p_hat(theta), differentiable through the inner solve."""

    def __init__(self, m, inner_iters=3, damp=1e-10, jitter=1e-10):
        self.m = m
        p, n, D = m.p, m.n, m.D
        nD = n * D
        gn = m._gn_solver()
        hess = m._hessian_fn()
        sig = m.sigmas
        dt = m.mu.dtype
        eyeX = jnp.eye(nD, dtype=dt)
        pp = ProfiledPosterior(m, n_nodes=8, inner_iters=inner_iters)
        S = jnp.asarray(pp._sensitivity(), dt)          # (nD, p)
        th0 = jnp.asarray(pp._th0, dt)
        X0 = jnp.asarray(pp._X0, dt)

        def logphat(theta, data):
            Xs = (X0.ravel() + S @ (theta - th0)).reshape(n, D)

            def body(X, _):
                A, g, _r = gn._normal_equations(theta, X, sig)
                Axx = A[p:, p:] + damp * jnp.trace(A[p:, p:]) / nD * eyeX
                dg = jnp.diag(Axx)
                dg = jnp.where(dg > jnp.finfo(dg.dtype).tiny, dg, jnp.ones_like(dg))
                Di = jax.lax.rsqrt(dg)
                As = Axx * Di[:, None] * Di[None, :]
                dX = Di * jax.scipy.linalg.cho_solve(
                    jax.scipy.linalg.cho_factor(As), -(g[p:] * Di))
                return X + dX.reshape(n, D), None

            X, _ = jax.lax.scan(body, Xs, None, length=inner_iters)
            x = jnp.concatenate([theta, X.ravel()])
            Hxx = hess(x)[p:, p:]
            Hxx = 0.5 * (Hxx + Hxx.T) + jitter * jnp.trace(Hxx) / nD * eyeX
            c = jax.scipy.linalg.cho_factor(Hxx)
            return m.logdensity(x, m.data) - jnp.sum(jnp.log(jnp.abs(jnp.diag(c[0]))))

        self.logdensity = logphat
        self.mu = m.mu
        self.data = m.data
        self.gradient = jax.jit(jax.vmap(
            lambda th, d: jax.grad(lambda z: logphat(z, d))(th), in_axes=(0, None)))


for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    t0 = time.time()
    post = m.fit(verbose=False)
    t_fit = time.time() - t0
    p = m.p
    print(f"\n===== {name}  p={p}  K={K}  maxit={MAXIT} =====", flush=True)
    tfl = T.energy_floor_k(K)
    rec = {"floor_k": tfl, "therr_floor": S.theta_err_floor(p), "runs": {}}

    def row(lab, TH, dt):
        r = dict(energy=T.energy(TH), therr=float(np.max(np.abs(TH.mean(0) - S.mean[:p])
                                                         / S.sd[:p])),
                 sdrat=float(np.median(TH.std(0) / S.sd[:p])), sec=dt)
        print(f'{lab:>32} {r["energy"]:>9.5f} {r["energy"]/max(tfl,1e-12):>8.2f} '
              f'{r["therr"]:>8.4f} {r["sdrat"]:>7.3f} {dt:>7.1f}', flush=True)
        return r

    print(f'{"variant":>32} {"thEnergy":>9} {"x floor":>8} {"thErr":>8} {"sdrat":>7} {"sec":>7}')
    THfit = np.asarray(post.sample(K, unpack=False), np.float64)[:, :p]
    rec["fit"] = row("fit() [the incumbent]", THfit, t_fit)

    g = ProfiledTarget(m, inner_iters=INNER)
    # cold start: Laplace draws around the joint MAP, i.e. no knowledge of the answer
    Hm = np.asarray(m.hessian(), np.float64)
    Hm = 0.5 * (Hm + Hm.T)
    dsc = np.sqrt(np.maximum(np.abs(np.diag(Hm)), 1e-300))
    wv, Vv = np.linalg.eigh(Hm / np.outer(dsc, dsc))
    keep = wv > 1e-10 * max(abs(wv).max(), 1e-300)
    Sig = ((Vv[:, keep] / wv[keep]) @ Vv[:, keep].T) / np.outer(dsc, dsc)
    Sth = Sig[:p, :p]
    rng = np.random.default_rng(0)
    TH0 = (np.asarray(m.map_particle, np.float64)[:p][None, :]
           + rng.standard_normal((K, p)) @ np.linalg.cholesky(Sth + 1e-14 * np.eye(p)).T)
    rec["start"] = row("START: Laplace theta draws", TH0, 0.0)

    for kern in KERNELS:
        t0 = time.time()
        try:
            TH, Rs, _ = M7.run_svgd(g, TH0, MAXIT, kernel=kern,
                                    optimizer=optax.contrib.prodigy, optimizer_kwargs={})
            rec["runs"][kern] = row(f"  profiled SVGD, {kern}", TH, time.time() - t0)
            rec["runs"][kern]["steinR"] = float(Rs[-1])
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec["runs"][kern] = dict(error=f"{type(e).__name__}: {str(e)[:150]}")

    print(f'{f"FLOOR: {K} exact draws":>32} {tfl:>9.5f} {1.0:>8.2f} '
          f'{S.theta_err_floor(p):>8.4f} {1.0:>7.3f}', flush=True)
    out[name] = rec
    json.dump(out, open("exp09_results.json", "w"), indent=1)
