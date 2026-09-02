"""
Exp 14: the R stopping threshold as a knob, and the theta-width tension.

exp09's recipe reaches energy 0.053 (floor 0.048) but its theta 95% intervals are 87-93% of
NUTS -- i.e. excellent in full-dimensional distributional terms while ~10% narrow on exactly the
three quantities a MAGI user reports. Since the flow contracts monotonically, stopping earlier
trades one against the other. Sweep the threshold to quantify the trade rather than assert it.
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

@jax.jit
def step(y):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1. + L2sq / h) ** -.5; Kg = (1. + L2sq / h) ** -1.5
    dx = (Kg.sum(1, keepdims=True) * y - Kg @ y) * (1. / h)
    return y + 1e-2 * ((Kx @ s.gradient(y, m.data) + dx) / y.shape[0])

THRESH = [1.30, 1.20, 1.10, 1.05, 1.00]
res = {t: [] for t in THRESH}
for seed in [0, 1, 2]:
    y = jax.random.normal(jax.random.key(seed), (800, H.DIM), dtype=jnp.float32)
    pend = sorted(THRESH, reverse=True)
    for it in range(1, 3001):
        y = step(y)
        if it % 50: continue
        X = xm + y @ Lw.T
        Rg = float(-jnp.sum((X - X.mean(0)) * m.gradient(X, m.data)) / X.size)
        while pend and Rg <= pend[0]:
            r = H.evaluate(X, m, tag=""); r["iter"] = it; r["R_at_stop"] = Rg
            res[pend.pop(0)].append(r)
        if not pend: break

print(f'{"stop at R <=":>13} {"iter":>6} {"energy":>8} {"R":>7} {"theta width % of NUTS":>24} '
      f'{"dev":>6} {"sdrat":>6}')
for t in THRESH:
    rs = res[t]
    if not rs: continue
    w = np.mean([r["width_pct"] for r in rs], axis=0)
    print(f'{t:>13.2f} {np.mean([r["iter"] for r in rs]):>6.0f} '
          f'{np.mean([r["energy"] for r in rs]):>8.4f} {np.mean([r["R_global"] for r in rs]):>7.3f} '
          f'{str(np.round(w,1)):>24} {np.mean([r["width_dev"] for r in rs]):>6.1f} '
          f'{np.mean([r["sd_ratio_med"] for r in rs]):>6.3f}')
g = H.gold_row()
print(f'{"NUTS floor":>13} {"":>6} {g["energy"]:>8.4f} {g["R_global"]:>7.3f} '
      f'{str(np.round(g["width_pct"],1)):>24} {g["width_dev"]:>6.1f} {g["sd_ratio_med"]:>6.3f}')
H.save({str(k): v for k, v in res.items()}, "exp14_threshold_results")
