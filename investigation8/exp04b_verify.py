"""
exp04b: verify the library change reproduces exp04's `plain_median` row, through MSVGD.solve.

exp04 measured the two bandwidth conventions with my own driver. The change was then made in
msvgd.MSVGD.pairwise_distance, so this re-runs the same configuration through the SHIPPED
solve() and checks it lands on exp04's h = Med number rather than its h = Med/lnK number.

Also checks the one place other than the kernel that reads the bandwidth: _mitotic_split
calibrates its jitter to h/2, so dropping the log makes the split jitter ln(K) ~ 6x wider. That is
consistent -- the jitter is meant to match the kernel's own implicit Gaussian variance, and the
kernel is what changed -- but it is a behaviour change and is measured here rather than assumed.
"""
import numpy as np, jax.numpy as jnp, optax, sys, time
import harness8 as H
from msvgd import MSVGD

name = sys.argv[1] if len(sys.argv) > 1 else "fn"
m, ds = H.build(name)
S = H.Scorer(name)
post = m.fit(verbose=False)
K = 400
X0 = jnp.asarray(post.sample(K, unpack=False), m.mu.dtype)
flk = S.energy_floor_k(K)
s = MSVGD(m.logdensity, data=m.data)
t0 = time.time()
P = np.asarray(s.solve(x0=X0, max_iter=2000, atol=0.0, rtol=0.0,
                       optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                       monitor_convergence=-1, random_seed=8), np.float64)
print(f'{name}: MSVGD.solve after the change -> energy {S.energy(P):.4f} '
      f'({S.energy(P)/flk:.2f}x floor), Stein R {H.stein_R(m, P):.4f}   ({time.time()-t0:.1f}s)',
      flush=True)
print(f'   exp04 measured: h=Med/lnK 6.8362 (83.32x, R 0.0193) | h=Med 2.8131 (34.29x, R 0.1778)'
      if name == "fn" else "", flush=True)

# mitosis path still runs, and the wider jitter is recorded
x1 = m.particles_init + 0.2 * np.random.default_rng(0).standard_normal(
    (50, S.mean.shape[0])).astype(np.float64)
s2 = MSVGD(m.logdensity, data=m.data)
P2 = np.asarray(s2.solve(x0=jnp.asarray(x1, m.mu.dtype), k_schedule=[100, 200], max_iter=500,
                         atol=0.0, rtol=0.0, optimizer=optax.contrib.prodigy,
                         optimizer_kwargs={}, monitor_convergence=-1, random_seed=8), np.float64)
print(f'{name}: mitosis 50->100->200 still runs, final shape {P2.shape}, '
      f'energy {S.energy(P2):.4f} ({S.energy(P2)/S.energy_floor_k(200):.2f}x its floor)',
      flush=True)
