"""
Exp 17: Gaussian VI on the mean, done properly (sample-average approximation).

exp07 showed the stochastic natural-gradient iteration mu <- mu + Sigma*E_q[grad log p] diverges
here even damped: fresh samples each step make E[grad log p] a noisy estimate, and Sigma has
condition number 1.9e4, so the iterate random-walks. The standard cure is to FIX the base
samples and optimize the resulting deterministic objective (sample-average approximation /
common random numbers):

    maximize over mu:   L(mu) = (1/n) sum_i log p(mu + H^(-1/2) z_i),   z_i fixed

This is the KL-optimal Gaussian mean for fixed covariance, and it needs only gradients -- no
SVGD run at all. If it works it is by far the cheapest route to the one quantity that matters.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time
import harness as H

z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
evc = np.maximum(ev, 1e-8 * ev.max())
Sig_h = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float64)
G = H.Gold()
jax.config.update("jax_enable_x64", True)
m = H.build_magi(dtype=jnp.float64)

def bias_of(mu):
    return float(np.sqrt((G.whiten(np.asarray(mu)[None, :]) ** 2).mean()))

print(f'MAP bias {bias_of(x_map):.4f}   SVGD-mean bias ~0.068   floor 0.034')
print(f'{"n":>6} {"opt":>10} {"iters":>6} {"bias":>8} {"sec":>6}')
out = []
for n in [256, 1024]:
    Z = jax.random.normal(jax.random.key(0), (n, H.DIM), dtype=jnp.float64)
    def L(mu):
        return -jnp.mean(jax.vmap(lambda zz: m.logdensity(mu + Sig_h @ zz, m.data))(Z))
    g = jax.jit(jax.value_and_grad(L))
    for oname, opt in [("adam1e-3", optax.adam(1e-3)), ("adam1e-2", optax.adam(1e-2))]:
        mu = jnp.asarray(x_map); st = opt.init(mu); t0 = time.time()
        for it in range(1, 601):
            v, gr = g(mu)
            if not jnp.all(jnp.isfinite(gr)):
                print(f'{n:>6} {oname:>10} {it:>6}   diverged'); mu = None; break
            up, st = opt.update(gr, st, mu); mu = optax.apply_updates(mu, up)
        if mu is None: continue
        print(f'{n:>6} {oname:>10} {600:>6} {bias_of(mu):>8.4f} {time.time()-t0:>6.1f}')
        out.append({"n": n, "opt": oname, "bias": bias_of(mu), "mu": np.asarray(mu).tolist()})
        np.save(f"sav_mean_n{n}_{oname}.npy", np.asarray(mu))

print(); print(H.HDR); print("-" * len(H.HDR))
rng = np.random.default_rng(0)
rows = [("N(MAP, H^-1)", x_map)]
best = min(out, key=lambda r: r["bias"]) if out else None
if best is not None:
    rows.append((f'N(SAV-mean n={best["n"]}, H^-1)', np.array(best["mu"])))
res = []
for tag, mu in rows:
    s = mu[None, :] + rng.standard_normal((800, H.DIM)) @ np.asarray(Sig_h).T
    r = H.evaluate(jnp.asarray(s), m, tag=tag); res.append(r); H.show(r)
r = H.gold_row(); res.append(r); H.show(r)
H.save({"sweep": out, "scored": res}, "exp17_sav_results")
