"""Exp 19: wall-clock cost of each stage, for the recommendation."""
import time, numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD
from functools import partial

def t(f, n=1):
    f(); jax.block_until_ready(0)                # warm/compile
    t0 = time.time()
    for _ in range(n): r = f()
    jax.block_until_ready(r)
    return (time.time() - t0) / n, r

jax.config.update("jax_enable_x64", True)
m64 = H.build_magi(dtype=jnp.float64)
t0 = time.time()
m64.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=30000, atol=1e-7, rtol=0.0,
          random_seed=0, monitor_convergence=-1, optimizer=optax.contrib.prodigy,
          optimizer_kwargs={})
t_map = time.time() - t0
x_map = np.asarray(m64.particles[0], np.float64)
t0 = time.time()
Hm = -np.asarray(jax.hessian(m64.magi_logdensity)(jnp.asarray(x_map)), np.float64)
t_hess = time.time() - t0
t0 = time.time(); ev, V = np.linalg.eigh(.5*(Hm+Hm.T)); t_eig = time.time() - t0
evc = np.maximum(ev, 1e-8*ev.max())

jax.config.update("jax_enable_x64", False)
m = H.build_magi()
Lw = jnp.asarray((V/np.sqrt(evc))@V.T, jnp.float32); xm = jnp.asarray(x_map, jnp.float32)
s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)

@jax.jit
def imq(y):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1.+L2sq/h)**-.5; Kg = (1.+L2sq/h)**-1.5
    dx = (Kg.sum(1, keepdims=True)*y - Kg@y)*(1./h)
    return y + 1e-2*((Kx@s.gradient(y, m.data) + dx)/y.shape[0])

@partial(jax.jit, static_argnums=2)
def ula(y, key, n):
    def b(c, _):
        y, k = c; k, sk = jax.random.split(k)
        return (y + 1e-2*s.gradient(y, m.data) + jnp.sqrt(2e-2)*jax.random.normal(sk, y.shape, y.dtype), k), None
    (y, _), _ = jax.lax.scan(b, (y, key), None, length=n)
    return y

y0 = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
dt_imq, _ = t(lambda: imq(y0), 20)
dt_ula, _ = t(lambda: ula(y0, jax.random.key(1), 100), 3)

print(f'{"stage":>36} {"seconds":>9}')
print(f'{"MAP solve (fp64)":>36} {t_map:>9.1f}')
print(f'{"dense Hessian 325x325 (fp64)":>36} {t_hess:>9.1f}')
print(f'{"eigendecomposition":>36} {t_eig:>9.1f}')
print(f'{"  setup subtotal":>36} {t_map+t_hess+t_eig:>9.1f}')
print(f'{"whitened IMQ, per iteration":>36} {dt_imq:>9.4f}')
print(f'{"whitened IMQ, ~600 it (R<=1.05)":>36} {600*dt_imq:>9.1f}')
print(f'{"whitened IMQ, ~1850 it (R<=1.0)":>36} {1850*dt_imq:>9.1f}')
print(f'{"whitened ULA, per iteration":>36} {dt_ula/100:>9.4f}')
print(f'{"whitened ULA, 2000 steps":>36} {2000*dt_ula/100:>9.1f}')
print(f'\n{"TOTAL: setup + ULA 2000":>36} {t_map+t_hess+t_eig+2000*dt_ula/100:>9.1f}')
print(f'{"TOTAL: setup + IMQ @R<=1.05":>36} {t_map+t_hess+t_eig+600*dt_imq:>9.1f}')
print(f'{"reference: NUTS (b30 study)":>36} {125.0:>9.1f}')
