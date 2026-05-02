import jax
import jax.numpy as jnp
from functools import partial

'''
Bessel functions. Note that the naming scheme kv_xp5 indicates parameter x.5.
'''

def _kv_prefactor(z):
    return jnp.sqrt(jnp.pi / (2 * z)) * jnp.exp(-z)

def kv_0p5(z):
    return _kv_prefactor(z)

def kv_1p5(z):
    return _kv_prefactor(z) * (1 + 1/z)

def kv_2p5(z):
    return _kv_prefactor(z) * (1 + 3/z + 3/z**2)

def kv_3p5(z):
    return _kv_prefactor(z) * (1 + 6/z + 15/z**2 + 15/z**3)

def kv_4p5(z):
    return _kv_prefactor(z) * (1 + 10/z + 45/z**2 + 105/z**3 + 105/z**4)


@partial(jax.jit, static_argnames=["n"])
def kvp_2p5(z, n):
    """
    Analogue of scipy.special.kvp for v=2.5 degrees of freedom. Only need n=0,1,2
    """
    z = jnp.where(z == 0, 1e-10, z)
    if n == 0:
        return kv_2p5(z)
    elif n == 1:
        return -0.5 * (kv_1p5(z) + kv_3p5(z))
    elif n == 2:
        return 0.25 * (kv_0p5(z) + 2 * kv_2p5(z) + kv_4p5(z))
    else:
        raise ValueError(f"n={n} not supported, use 0, 1, or 2")

def matern_2p5(x, y, phi1, phi2):
    """
    Closed form Matern kernel for v=2.5 degrees of freedom.
    """
    sq_dists = jnp.sum(
        (x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :])**2, axis=-1
    )
    r = jnp.sqrt(jnp.clip(sq_dists, min=1e-10))
    s = jnp.sqrt(5) * r / phi2
    return phi1 * (1 + s + s**2 / 3) * jnp.exp(-s)