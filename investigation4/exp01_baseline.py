"""
Exp 1: re-baseline every candidate on a correct MAP, scored and timed on equal footing.

The headline candidate is preconditioned NUTS. The 125 s reference cost in earlier
investigations was NUTS with an adapted DIAGONAL mass matrix on the untransformed target; given
the exact Hessian as a metric the geometry is near-isotropic, which is NUTS's best case. If it
reaches the floor in seconds it is the strongest answer available on safety and justifiability,
since it is exact, standard, and ships with R-hat, ESS and divergence diagnostics.
"""
import numpy as np, jax, jax.numpy as jnp, time
from functools import partial
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
import blackjax

G = H.Gold()
m, x_map, Hs, Sig, L = cache("baseline")
Lj, xm = jnp.asarray(L), jnp.asarray(x_map)
K = 400

def score(X, tag, secs, cost=None):
    r = H.evaluate(jnp.asarray(X), m, tag=tag)
    P = np.asarray(X, np.float64)
    pr = ((P - G.mean) @ G.evecs).var(0) / G.evals
    r["varwtd"] = float(np.sum(pr * G.evals) / np.sum(G.evals))
    r["t"] = secs; r["cost"] = cost
    return r

logp_y = lambda y: m.logdensity(xm + Lj @ y, m.data)
out = []
print(H.HDR); print("-" * len(H.HDR))

# ---------------------------------------------------------------- deterministic references
rng = np.random.default_rng(0)
Sh = np.linalg.cholesky(Sig + 1e-14 * np.eye(H.DIM))
t0 = time.time(); s = x_map[None, :] + rng.standard_normal((K, H.DIM)) @ Sh.T
r = score(s, "N(MAP, H^-1)", time.time() - t0); out.append(r); H.show(r)

# ---------------------------------------------------------------- preconditioned NUTS
def run_nuts(n_warm, n_samp, seed=0):
    keys = jax.random.split(jax.random.key(seed), K)
    inv_mass = jnp.ones(H.DIM)                       # identity: the whitening IS the mass matrix
    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, logp_y, target_acceptance_rate=0.8,
                                        is_mass_matrix_diagonal=True)
        (st, par), _ = wu.run(wk, position=jnp.zeros(H.DIM), num_steps=n_warm)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(logp_y, **par), initial_state=st, num_steps=n_samp)
        return states.position[-1], info.num_integration_steps.sum(), info.is_divergent.sum()
    return jax.jit(jax.vmap(one))(keys)

for n_warm, n_samp in [(50, 5), (100, 20), (200, 50)]:
    t0 = time.time()
    y, nleap, ndiv = run_nuts(n_warm, n_samp); y.block_until_ready()
    dt = time.time() - t0
    r = score(xm + y @ Lj.T, f"precond NUTS warm={n_warm} n={n_samp}", dt,
              int(jnp.mean(nleap)) + n_warm)
    r["div"] = int(jnp.sum(ndiv)); out.append(r); H.show(r)
    print(f'{"":>26}   leapfrog/chain (sampling) = {float(jnp.mean(nleap)):.0f}, '
          f'divergences = {int(jnp.sum(ndiv))}, {dt:.1f}s')

H.show(H.gold_row())
print(f'\n{"method":>34} {"energy":>8} {"varwtd":>7} {"dev":>6} {"grad evals":>11} {"sec":>7}')
for r in out:
    print(f'{r["tag"]:>34} {r["energy"]:>8.4f} {r["varwtd"]:>7.3f} {r["width_dev"]:>6.1f} '
          f'{str(r["cost"]):>11} {r["t"]:>7.2f}')
H.save(out, "exp01_baseline_results")
