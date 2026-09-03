"""
exp03: broaden investigation 7 sec. 11 -- SVGD on the profiled p-dimensional marginal.

That was the strongest positive result in investigation 7 (0.30x and 0.36x the 64-draw floor on
fn and lorenz, beating fit() by 2.5-8x) and it was two systems at one particle count. Here: HIV
added (p = 5, nD = 603, so the inner Cholesky is ~8x the cost), K swept over {16, 64, 256}, and
wall clock reported honestly -- including the first-call JIT compile, which for a backward pass
through a scan of Cholesky solves is not negligible and which sec. 11 hid inside a warmed process.

The question that matters is whether the advantage over fit() survives at the K a user would
actually pick given the runtime, so fit() is timed the same way on the same device and scored at
the same K.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness8 as H
import msvgd8 as M7
from profiled import ProfiledPosterior

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
KS = [int(x) for x in os.environ.get("KS", "16,64,256").split(",")]
MAXIT = int(os.environ.get("MAXIT", 400))
INNER = int(os.environ.get("INNER", 3))
KERNELS = os.environ.get("KERNELS", "standard,reweighted").split(",")
out = {}


class ProfiledTarget:
    """p-dimensional MAGI target log p_hat(theta), differentiable through the inner solve."""

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
        S = jnp.asarray(pp._sensitivity(), dt)
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
    p = m.p
    t0 = time.time()
    post = m.fit(verbose=False)
    t_fit = time.time() - t0
    print(f"\n===== {name}  p={p}  nD={m.n*m.D}  maxit={MAXIT}  "
          f"fit() {t_fit:.1f}s =====", flush=True)
    print(f'{"K":>5} {"variant":>26} {"thEnergy":>10} {"x floor":>8} {"thErr":>8} '
          f'{"sdrat":>7} {"sec":>8} {"vs fit()":>9}', flush=True)
    rec = {"fit_sec": t_fit, "p": p, "nD": int(m.n * m.D),
           "therr_floor": S.theta_err_floor(p), "runs": {}}

    # Laplace theta covariance at the joint MAP, the cold start
    Hm = np.asarray(m.hessian(), np.float64)
    Hm = 0.5 * (Hm + Hm.T)
    dsc = np.sqrt(np.maximum(np.abs(np.diag(Hm)), 1e-300))
    wv, Vv = np.linalg.eigh(Hm / np.outer(dsc, dsc))
    keep = wv > 1e-10 * max(abs(wv).max(), 1e-300)
    Sig = ((Vv[:, keep] / wv[keep]) @ Vv[:, keep].T) / np.outer(dsc, dsc)
    Lth = np.linalg.cholesky(Sig[:p, :p] + 1e-14 * np.eye(p))

    t0 = time.time()
    g = ProfiledTarget(m, inner_iters=INNER)
    t_build = time.time() - t0

    for K in KS:
        tfl = T.energy_floor_k(K)

        def row(lab, TH, dt_, extra=""):
            r = dict(K=K, thenergy=T.energy(TH),
                     therr=float(np.max(np.abs(TH.mean(0) - S.mean[:p]) / S.sd[:p])),
                     sdrat=float(np.median(TH.std(0) / S.sd[:p])), sec=dt_, floor=tfl)
            print(f'{K:>5} {lab:>26} {r["thenergy"]:>10.5f} {r["thenergy"]/tfl:>8.2f} '
                  f'{r["therr"]:>8.4f} {r["sdrat"]:>7.3f} {dt_:>8.1f} {extra:>9}', flush=True)
            return r

        rec["runs"][f"fit_{K}"] = row("fit() sample", np.asarray(
            post.sample(K, unpack=False), np.float64)[:, :p], t_fit)
        rng = np.random.default_rng(0)
        TH0 = (np.asarray(m.map_particle, np.float64)[:p][None, :]
               + rng.standard_normal((K, p)) @ Lth.T)
        rec["runs"][f"start_{K}"] = row("START Laplace theta", TH0, 0.0)
        for kern in KERNELS:
            t0 = time.time()
            try:
                TH, Rs, _ = M7.run_svgd(g, TH0, MAXIT, kernel=kern,
                                        optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                dt_ = time.time() - t0
                rec["runs"][f"{kern}_{K}"] = row(f"profiled SVGD, {kern}", TH, dt_,
                                                 f"{dt_/t_fit:.1f}x")
            except Exception as e:
                import traceback; traceback.print_exc()
                rec["runs"][f"{kern}_{K}"] = dict(error=f"{type(e).__name__}: {str(e)[:150]}")
        print(f'{K:>5} {"FLOOR: K exact draws":>26} {tfl:>10.5f} {1.0:>8.2f} '
              f'{S.theta_err_floor(p):>8.4f} {1.0:>7.3f}', flush=True)
    rec["target_build_sec"] = t_build
    print(f'   (ProfiledTarget construction {t_build:.1f}s; the first SVGD call in each '
          f'row includes its own JIT compile)', flush=True)
    out[name] = rec
    json.dump(out, open(f"exp03_results_{name}.json", "w"), indent=1)
