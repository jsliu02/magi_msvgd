"""Exp 25: end-to-end smoke test of the rewritten public API, including the NUTS fallback."""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup4 import build, SETTINGS

m = build(*SETTINGS["baseline"], dtype=jnp.float64)
print("--- map_solve ---");  Xs, th, sg = m.map_solve(tol=1e-8, max_iter=200)
print(f"    shapes {Xs.shape} {th.shape} {sg.shape}, theta = {np.round(np.asarray(th[0]), 5)}")
print("--- condition_A ---"); print(f"    dA = {m.condition_A():.4f}")
print("--- fit ---");        post = m.fit()
print("--- sample ---");     Xs, th, sg = post.sample(k=200, seed=1)
print(f"    {Xs.shape} {th.shape} {sg.shape}; theta mean = {np.round(np.asarray(th).mean(0), 5)}")
print(f"    m.sample() shortcut: {m.sample(k=50)[1].shape}")
print("--- nuts (short, preconditioned) ---")
Xs, th, sg = m.nuts(warmup_steps=200, sampling_steps=200, n_chains=4)
print(f"    pooled draws {Xs.shape}; theta mean = {np.round(np.asarray(th).mean(0), 5)}")
print(f"    fit mean    = {np.round(post.mean[:3], 5)}")
print("--- put() invalidates caches ---")
m.put(dtype=jnp.float32)
print(f"    posterior cleared: {m.posterior is None}; refit ->")
p2 = m.fit(verbose=False)
print(f"    theta {np.round(p2.mean[:3], 5)}, ratio {p2.certificates['ratio']:.3f}, "
      f"applied {p2.applied}")
