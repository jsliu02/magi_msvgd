"""
Exp 13: why does IMQ rescue SVGD in 325 dimensions, and RBF not?

In high dimension pairwise squared distances concentrate near their mean, so with the median
heuristic h = median(L2)/log(k) essentially every off-diagonal entry sees L2/h ~ log(k):

    RBF:  exp(-L2/h)          ->  exp(-log k)      = 1/k
    IMQ:  (1 + L2/h)^(-1/2)   ->  (1 + log k)^(-1/2)

At k=800 that is 0.00125 versus 0.35 -- a factor of ~280. The RBF Gram matrix collapses toward
the identity, every particle interacts only with itself, the repulsion vanishes and the drift
drives each particle independently to the MAP. IMQ's polynomial tail keeps the Gram matrix
dense. Measured here on real ensembles rather than argued.
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

print(f'{"ensemble":>34} {"k":>5} {"L2/h mean":>10} {"L2/h cv":>8} '
      f'{"RBF offdiag":>12} {"IMQ offdiag":>12} {"ratio":>7} {"RBF eff.nbrs":>13}')
for tag, k in [("Laplace samples, whitened", 200), ("Laplace samples, whitened", 800),
               ("Laplace samples, whitened", 3200)]:
    y = jax.random.normal(jax.random.key(0), (k, H.DIM), dtype=jnp.float32)
    L2, h = s.pairwise_distance(y, -1)
    r = np.asarray(L2 / h)
    off = ~np.eye(k, dtype=bool)
    rbf = np.exp(-r[off]); imq = (1 + r[off]) ** -0.5
    # effective neighbours: how many particles carry the kernel mass in a row
    Krbf = np.exp(-r); np.fill_diagonal(Krbf, 1.0)
    eff = float(np.mean(Krbf.sum(1) ** 2 / (Krbf ** 2).sum(1)))
    print(f'{tag:>34} {k:>5} {r[off].mean():>10.3f} {r[off].std()/r[off].mean():>8.4f} '
          f'{rbf.mean():>12.3e} {imq.mean():>12.3e} {imq.mean()/rbf.mean():>7.1f} {eff:>13.2f}')
    if k == 800:
        print(f'{"":>34}       predicted: L2/h -> log k = {np.log(k):.3f}, '
              f'RBF -> 1/k = {1/k:.3e}, IMQ -> {(1+np.log(k))**-0.5:.3e}')

# same in x-space, where the anisotropy makes distances even more concentrated
sx = MSVGD(lambda y, db: m.logdensity(y, db), data=m.data)
y = xm + jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32) @ Lw.T
L2, h = sx.pairwise_distance(y, -1)
r = np.asarray(L2 / h); off = ~np.eye(800, dtype=bool)
print(f'{"same ensemble, x-space":>34} {800:>5} {r[off].mean():>10.3f} '
      f'{r[off].std()/r[off].mean():>8.4f} {np.exp(-r[off]).mean():>12.3e} '
      f'{((1+r[off])**-0.5).mean():>12.3e}')
