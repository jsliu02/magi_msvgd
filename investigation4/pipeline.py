"""
The deterministic pipeline, as one reusable function, plus properly calibrated metric floors.

build(name, M) returns the corrected Gaussian (mu, Sigma) and a cost breakdown:
    Gauss-Newton MAP -> exact Hessian -> third-order mean correction
    -> slice-curvature screen over all d directions (4 batched log-density evals)
    -> bordered-GN Laplace marginal on the top M, replacing those marginal variances.

floors(gold, d) calibrates every covariance metric. The half-vs-half split conflates TWO noise
sources (both halves are estimates); what is needed is the floor for a NOISELESS approximation
scored against the full chain. So gold's effective sample size is first inferred by finding the n
at which synthetic draws from N(mu_g, Sigma_g) reproduce the observed half-vs-half Forstner
distance, and the floor is then read off at that n with one side exact.
"""
import numpy as np, jax, jax.numpy as jnp, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler, moments


def metrics(ref_mu, ref_cov, d):
    eg, Vg = np.linalg.eigh(ref_cov); eg = np.maximum(eg, 1e-14)
    Wg = Vg / np.sqrt(eg); ldg = float(np.sum(np.log(eg))); trg = float(np.trace(ref_cov))
    def f(mu, S):
        M = Wg.T @ S @ Wg
        w = np.maximum(np.linalg.eigvalsh(0.5 * (M + M.T)), 1e-12)
        dl = Wg.T @ (mu - ref_mu)
        return dict(bias=float(np.linalg.norm(dl) / np.sqrt(d)), trace=float(np.trace(S) / trg),
                    forst=float(np.linalg.norm(np.log(w)) / np.sqrt(d)),
                    kl=float(0.5 * (w.sum() - d + dl @ dl - np.linalg.slogdet(S)[1] + ldg)))
    return f


def floors(gold, d, nch=8, reps=4):
    """(observed half-vs-half, inferred n_eff, floor for a noiseless q vs the full chain)."""
    ch = gold.reshape(nch, -1, d)
    obs = np.mean([metrics(B.mean(0), np.cov(B, rowvar=False), d)(A.mean(0), np.cov(A, rowvar=False))
                   ["forst"] for A, B in
                   [(ch[i[:4]].reshape(-1, d), ch[i[4:]].reshape(-1, d))
                    for i in (np.random.default_rng(r).permutation(nch) for r in range(reps))]])
    mu_g, Sg = gold.mean(0), np.cov(gold, rowvar=False)
    Lg = np.linalg.cholesky(Sg + 1e-14 * np.trace(Sg) / d * np.eye(d))
    rng = np.random.default_rng(0)
    def sim(n, both):
        out = []
        for r in range(3):
            A = mu_g + rng.standard_normal((n, d)) @ Lg.T
            ref = (mu_g, Sg) if not both else (
                (lambda B: (B.mean(0), np.cov(B, rowvar=False)))(mu_g + rng.standard_normal((n, d)) @ Lg.T))
            out.append(metrics(ref[0], ref[1], d)(A.mean(0), np.cov(A, rowvar=False)))
        return {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    lo, hi = 400, 200000                                  # bisect n_eff on the half-vs-half stat
    for _ in range(18):
        mid = int(np.sqrt(lo * hi))
        if sim(mid, True)["forst"] > obs: lo = mid
        else: hi = mid
    n_eff = int(np.sqrt(lo * hi))
    return obs, n_eff, sim(2 * n_eff, False)


def build(magi, x_map, Hs, Sig, M=20, nz=17, span=4.5):
    d = Hs.shape[0]
    t = {}
    gn = GaussNewtonMAP(magi); pr = Profiler(gn, magi)
    lp = jax.jit(lambda P: jax.vmap(lambda z: magi.logdensity(z, magi.data))(P))
    Sj = jnp.asarray(Sig)
    t0 = time.time()
    mu3 = np.asarray(x_map) - 0.5 * np.asarray(
        Sj @ jax.grad(lambda z: jnp.sum(Sj * pr._hess(z)))(jnp.asarray(x_map)))
    t["third_order"] = time.time() - t0

    ev, V = np.linalg.eigh(Hs); sd = 1.0 / np.sqrt(ev)
    mu0 = np.asarray(x_map)
    t0 = time.time()
    lp0 = float(lp(jnp.asarray(mu0[None, :]))[0])
    P = np.concatenate([mu0[None, :] + s * (sd[:, None] * V.T) for s in (-2, -1, 1, 2)])
    U = -(np.asarray(lp(jnp.asarray(P))) - lp0).reshape(4, d)
    qscr = np.abs(U / np.array([2.0, 0.5, 0.5, 2.0])[:, None] - 1).mean(0)
    t["screen"] = time.time() - t0

    order = np.argsort(-qscr)[:M]
    t0 = time.time()
    vprof, mprof = np.full(M, np.nan), np.zeros(M)
    for i, j in enumerate(order):
        try:
            zs = np.linspace(-span * sd[j], span * sd[j], nz)
            Up, ldt, _ = pr.profile(V[:, j], zs, x_map)
            if np.all(np.isfinite(Up)) and np.all(np.isfinite(ldt)):
                mprof[i], vprof[i] = moments(zs, -Up - ldt)
        except Exception:
            pass
    t["profiles"] = time.time() - t0
    ok = np.isfinite(vprof) & (vprof > 0.25 * sd[order] ** 2) & (vprof < 4 * sd[order] ** 2)
    return dict(mu0=mu0, mu3=mu3, ev=ev, V=V, sd=sd, qscr=qscr, order=order,
                vprof=vprof, mprof=mprof, ok=ok, t=t)


def cov_of(b, Sig, M):
    sel, vp = b["order"][:M][b["ok"][:M]], b["vprof"][:M][b["ok"][:M]]
    if len(sel) == 0: return Sig
    V, sd = b["V"], b["sd"]
    return Sig + V[:, sel] @ np.diag(vp - sd[sel] ** 2) @ V[:, sel].T
