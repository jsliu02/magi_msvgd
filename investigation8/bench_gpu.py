"""Wall-clock per 200 SVGD iterations on whatever device JAX picks, K=400, fn."""
import os, time, numpy as np, jax, jax.numpy as jnp, optax
import harness8 as H, msvgd8 as M7
print("devices:", jax.devices(), flush=True)
m, ds = H.build("fn")
S = H.Scorer("fn")
post = m.fit(verbose=False)
X0 = np.asarray(post.sample(400, unpack=False), np.float64)
h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
for rep in range(3):
    t0 = time.time()
    P, _, _ = M7.run_svgd(m, X0, 200, kernel="standard", bandwidth=100 * h0,
                          optimizer=optax.contrib.prodigy, optimizer_kwargs={})
    print(f"  rep {rep}: 200 iters in {time.time()-t0:.2f}s   energy {S.energy(P):.4f}", flush=True)
