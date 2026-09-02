"""
Exp 16: verify preconditioned ULA across seeds and step sizes.

exp06 got whitened ULA to energy 0.049 (floor 0.048) in a single run. If that holds it
outranks the SVGD recipe on every metric while being far simpler -- no kernel, no bandwidth, no
mitosis, no stopping rule -- so the recommendation depends on it and one run is not enough.

ULA is biased by discretization at any finite step size, so sweep the step size too: the bias
should grow with lr, and the sweep shows where it starts to matter.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H
from msvgd import MSVGD

z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
evc = np.maximum(ev, 1e-8 * ev.max())
Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float32)
xm = jnp.asarray(x_map, jnp.float32)
m = H.build_magi()
s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)

from functools import partial
@partial(jax.jit, static_argnums=3)
def chain(y, key, lr, n):
    def body(c, _):
        y, k = c
        k, sk = jax.random.split(k)
        y = y + lr * s.gradient(y, m.data) + jnp.sqrt(2 * lr) * jax.random.normal(sk, y.shape, y.dtype)
        return (y, k), None
    (y, _), _ = jax.lax.scan(body, (y, key), None, length=n)
    return y

print(H.HDR); print("-" * len(H.HDR))
out = []
for lr in [3e-3, 1e-2, 3e-2]:
    rs = []
    for seed in [0, 1, 2]:
        y = jax.random.normal(jax.random.key(seed), (800, H.DIM), dtype=jnp.float32)
        y = chain(y, jax.random.key(1000 + seed), lr, 2000)
        r = H.evaluate(xm + y @ Lw.T, m, tag=f"whitened ULA lr={lr:g} s{seed}")
        rs.append(r); out.append(r)
    if all(not r.get("failed") for r in rs):
        w = np.mean([r["width_pct"] for r in rs], 0)
        print(f'{f"  MEAN lr={lr:g} (3 seeds)":>26} {str(np.round(w,1)):>21} '
              f'{np.mean([r["width_dev"] for r in rs]):5.1f} '
              f'{np.mean([r["R_global"] for r in rs]):7.3f} {"":>6} '
              f'{np.mean([r["energy"] for r in rs]):7.3f} '
              f'{np.mean([r["bias"] for r in rs]):6.3f} '
              f'{np.mean([r["sd_ratio_med"] for r in rs]):6.3f}')
    else:
        for r in rs: H.show(r)
r = H.gold_row(); out.append(r); H.show(r)
H.save(out, "exp16_ula_results")
