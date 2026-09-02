"""
Exp 21: longer references for the sparse settings, where the 400x150 configuration did not mix.

Max R-hat was 1.03 at baseline but 1.16 at half and 1.77 at quarter: with only 150 draws after
warmup, chains that start together at the mode have not decorrelated, so the between-chain
variance that R-hat measures is dominated by their common starting point rather than by the
posterior. The fix is the opposite trade -- far fewer chains, far longer -- which costs about the
same total gradient evaluations but actually mixes. R-hat is still well determined at 64 chains.
"""
import numpy as np, jax, jax.numpy as jnp, time, os, sys
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
import blackjax

K, NW, NS = 64, 1500, 2000
d = H.DIM

def rhat_ess(P):
    c, n, _ = P.shape
    S = P.reshape(2 * c, n // 2, -1)
    W = S.var(1, ddof=1).mean(0); Bn = S.mean(1).var(0, ddof=1)
    V = (n // 2 - 1) / (n // 2) * W + Bn
    return np.sqrt(np.maximum(V / np.maximum(W, 1e-300), 0)), \
           np.minimum(2 * c * (n // 2) * V / np.maximum(Bn, 1e-300), 2 * c * (n // 2))

for name in ["quarter", "half", "noisy"]:
    out = f"ref5_{name}.npz"
    if os.path.exists(out):
        print(f"{name}: cached"); continue
    m, x_map, Hs, Sig, L = cache(name)
    Lj, xm = jnp.asarray(L), jnp.asarray(x_map)
    logp_y = lambda y: m.logdensity(xm + Lj @ y, m.data)
    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, logp_y, target_acceptance_rate=0.9)
        (st, par), _ = wu.run(wk, position=jnp.zeros(d), num_steps=NW)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(logp_y, **par), initial_state=st, num_steps=NS)
        return states.position, info.num_integration_steps.sum(), info.is_divergent.sum()
    t0 = time.time()
    Y, nl, nd = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(11), K))
    Y.block_until_ready(); dt = time.time() - t0
    P = np.asarray(xm[None, None, :] + Y @ Lj.T, np.float64)
    rh, ess = rhat_ess(P)
    F = P.reshape(-1, d)
    np.savez(out, mean=F.mean(0), cov=np.cov(F, rowvar=False),
             sub=F[np.random.default_rng(0).choice(len(F), 4000, replace=False)],
             half_mean=np.stack([P[:K//2].reshape(-1,d).mean(0), P[K//2:].reshape(-1,d).mean(0)]),
             half_cov=np.stack([np.cov(P[:K//2].reshape(-1,d), rowvar=False),
                                np.cov(P[K//2:].reshape(-1,d), rowvar=False)]),
             rhat=rh, ess=ess, div=int(jnp.sum(nd)), sec=dt, ndraw=len(F))
    print(f'{name:>9}: {len(F)} draws ({K}x{NS}) in {dt:.0f}s, max Rhat {rh.max():.4f}, '
          f'min ESS {ess.min():.0f}, divergences {int(jnp.sum(nd))}', flush=True)
