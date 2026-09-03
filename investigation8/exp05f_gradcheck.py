"""exp05f: verify the preconditioned gradient against the analytic chain rule."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import harness8 as H, msvgd8 as M7

m, ds = H.build("fn")
S = H.Scorer("fn")
m.fit(verbose=False)
x_map, L = M7.laplace_metric(m)
d = S.mean.shape[0]
rng = np.random.default_rng(0)
Y = rng.standard_normal((4, d))
Lj, x0j = jnp.asarray(L, m.mu.dtype), jnp.asarray(x_map, m.mu.dtype)

g_code = jax.vmap(lambda y, dd: jax.grad(lambda z: m.logdensity(x0j + Lj @ z, dd))(y),
                  in_axes=(0, None))(jnp.asarray(Y, m.mu.dtype), m.data)
X = x_map + Y @ L.T
g_x = np.asarray(m.gradient(jnp.asarray(X, m.mu.dtype), m.data), np.float64)
g_analytic = g_x @ L                                   # rows: L^T grad_x
g_code = np.asarray(g_code, np.float64)
rel = np.abs(g_code - g_analytic).max() / np.abs(g_analytic).max()
print(f"fixed driver vs analytic L^T grad_x : max rel err {rel:.3e}")
g_old = g_analytic @ L                                 # what the buggy line computed: L^T L^T grad
print(f"buggy form (L^T L^T grad) differs by a factor of "
      f"{np.abs(g_old).max()/np.abs(g_analytic).max():.3e} in max magnitude")
