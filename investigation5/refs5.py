"""
References for every test system, in Jacobi-scaled whitened coordinates.

Two changes from investigation 4's reference builder, both forced by the new systems. The MAP
comes from the preconditioned solver, because HIV's stock MAP is not stationary. And the whitening
metric is built in Jacobi-scaled coordinates: H itself has cond 4e16 on HIV and is numerically
singular, so H^(-1/2) cannot be formed, whereas D H D has cond 3e7 and inverts cleanly. The
composite map x = x* + (D W) y is the same transformation in exact arithmetic and a usable one in
floating point.

Sampling in a metric never changes the stationary distribution, so this remains an unbiased
reference regardless of how good or bad the metric is.
"""
import numpy as np, jax, jax.numpy as jnp, os, sys, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map
import blackjax


def metric(m, floor=1e-10):
    """(x*, M) with x = x* + M y whitening the Jacobi-scaled Hessian."""
    x = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.diag(H), np.finfo(float).tiny))
    Dm = 1.0 / d
    Hs = H * np.outer(Dm, Dm)
    Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs)
    w = np.maximum(w, floor * max(w.max(), 1.0))
    W = (V / np.sqrt(w)) @ V.T
    return x, Dm[:, None] * W, H, d


def rhat_ess(P):
    c, ns, _ = P.shape
    S = P.reshape(2 * c, ns // 2, -1)
    Wv = S.var(1, ddof=1).mean(0); B = S.mean(1).var(0, ddof=1)
    V = (ns // 2 - 1) / (ns // 2) * Wv + B
    return (np.sqrt(np.maximum(V / np.maximum(Wv, 1e-300), 0)),
            np.minimum(2 * c * (ns // 2) * V / np.maximum(B, 1e-300), 2 * c * (ns // 2)))


def build_ref(name, K=32, NW=1500, NS=3000, seed=5, target=0.9):
    out = f"ref5_{name}.npz"
    if os.path.exists(out):
        print(f"{name}: cached", flush=True); return
    m, ds = build(name)
    scaled_map(m, tol=1e-10, max_iter=400)
    x, M, H, dscale = metric(m)
    xj, Mj = jnp.asarray(x), jnp.asarray(M)
    logp = lambda y: m.magi_logdensity(xj + Mj @ y)
    dim = x.shape[0]

    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, logp, target_acceptance_rate=target)
        (st, par), _ = wu.run(wk, position=jnp.zeros(dim), num_steps=NW)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(logp, **par), initial_state=st, num_steps=NS)
        return states.position, info.is_divergent.sum(), info.num_integration_steps.sum()

    t0 = time.time()
    Y, nd, nl = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(seed), K))
    Y.block_until_ready(); dt = time.time() - t0
    P = np.asarray(xj[None, None, :] + Y @ Mj.T, np.float64)
    rh, ess = rhat_ess(P)
    F = P.reshape(-1, dim)
    np.savez(out, mean=F.mean(0), cov=np.cov(F, rowvar=False), rhat=rh, ess=ess,
             div=int(jnp.sum(nd)), sec=dt, ndraw=len(F), x_map=x, H=H,
             sub=F[np.random.default_rng(0).choice(len(F), min(4000, len(F)), replace=False)],
             half_mean=np.stack([P[:K//2].reshape(-1,dim).mean(0), P[K//2:].reshape(-1,dim).mean(0)]),
             half_cov=np.stack([np.cov(P[:K//2].reshape(-1,dim), rowvar=False),
                                np.cov(P[K//2:].reshape(-1,dim), rowvar=False)]))
    print(f'{name:>8}: {len(F)} draws in {dt:.0f}s | max Rhat {rh.max():.4f} | '
          f'min ESS {ess.min():.0f} | div {int(jnp.sum(nd))} | '
          f'leapfrog/chain {float(jnp.mean(nl)):.0f}', flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(SYSTEMS)):
        try:
            build_ref(nm)
        except Exception as e:
            print(f'{nm:>8}: FAILED {type(e).__name__}: {str(e)[:100]}', flush=True)
