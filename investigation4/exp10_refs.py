"""
Exp 10: build a NUTS reference for every setting, and validate the construction at baseline.

No gold chain exists for the sparse and noisy settings, so the pipeline cannot be checked there
without one. Preconditioned NUTS supplies it: the Laplace metric changes only the efficiency of
the sampler, never its stationary distribution, so using the very object under test as the
preconditioner is not circular. The step size is tightened to target_accept = 0.95 because the
baseline run at 0.8 reported 193 divergences, and divergences bias exactly the tail geometry this
is meant to measure.

Credibility is established by rebuilding the BASELINE reference this way and comparing it to the
independent 8-chain gold standard. If the two agree to within gold's own half-vs-half floor, the
same construction is trustworthy in the settings where no gold exists.
"""
import numpy as np, jax, jax.numpy as jnp, time, sys, os
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache, SETTINGS
import blackjax

K, NW, NS = 400, 300, 150
d = H.DIM

def rhat_ess(P):
    """split-Rhat and bulk ESS across chains, per coordinate; P is (chains, draws, dim)."""
    c, n, _ = P.shape
    S = P.reshape(2 * c, n // 2, -1)
    W = S.var(1, ddof=1).mean(0); Bn = S.mean(1).var(0, ddof=1)
    V = (n // 2 - 1) / (n // 2) * W + Bn
    rh = np.sqrt(np.maximum(V / np.maximum(W, 1e-300), 0))
    ess = 2 * c * (n // 2) * np.minimum(V / np.maximum(Bn, 1e-300), 2 * c)
    return rh, np.minimum(ess, 2 * c * (n // 2))

for name in SETTINGS:
    out = f"ref4_{name}.npz"
    if os.path.exists(out):
        print(f"{name}: cached"); continue
    m, x_map, Hs, Sig, L = cache(name)
    Lj, xm = jnp.asarray(L), jnp.asarray(x_map)
    logp_y = lambda y: m.logdensity(xm + Lj @ y, m.data)
    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, logp_y, target_acceptance_rate=0.95)
        (st, par), _ = wu.run(wk, position=jnp.zeros(d), num_steps=NW)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(logp_y, **par), initial_state=st, num_steps=NS)
        return states.position, info.num_integration_steps.sum(), info.is_divergent.sum()
    t0 = time.time()
    Y, nl, nd = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(7), K))
    Y.block_until_ready(); dt = time.time() - t0
    P = np.asarray(xm[None, None, :] + Y @ Lj.T, np.float64)
    rh, ess = rhat_ess(P)
    F = P.reshape(-1, d)
    sub = F[np.random.default_rng(0).choice(len(F), 4000, replace=False)]
    np.savez(out, mean=F.mean(0), cov=np.cov(F, rowvar=False), sub=sub,
             half_mean=np.stack([P[:K//2].reshape(-1,d).mean(0), P[K//2:].reshape(-1,d).mean(0)]),
             half_cov=np.stack([np.cov(P[:K//2].reshape(-1,d), rowvar=False),
                                np.cov(P[K//2:].reshape(-1,d), rowvar=False)]),
             rhat=rh, ess=ess, div=int(jnp.sum(nd)), sec=dt, ndraw=len(F))
    print(f'{name:>9}: {len(F)} draws in {dt:.0f}s, max Rhat {rh.max():.4f}, '
          f'min ESS {ess.min():.0f}, divergences {int(jnp.sum(nd))}')
