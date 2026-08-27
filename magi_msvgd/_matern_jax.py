import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sp_special
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


##########################################################
# v=2.01 degrees of freedom: this is MAGI's actual smoothness parameter (the smallest
# value giving a twice mean-square differentiable process while staying as flexible as
# possible -- see the MAGI paper). Since 2.01 is not a half-integer, there's no closed
# form, and jax.scipy.special has no general-order modified Bessel K function at all.
# Raw values come from scipy via a host callback; only kv_2p01 needs a gradient rule
# (used inside fit_phisigma's gradient-based BFGS fit) -- its neighbors are only ever
# used as plain values inside that rule, never differentiated further, since BFGS only
# needs first-order gradients.
##########################################################

def _kv_host(v):
    '''Forward-only evaluation of scipy.special.kv(v, z) for a fixed v, via a host callback.'''
    def call(z):
        def _np_kv(z_np):
            return sp_special.kv(v, np.asarray(z_np)).astype(z.dtype)
        return jax.pure_callback(_np_kv, jax.ShapeDtypeStruct(z.shape, z.dtype), z)
    return call

_kv_0p01_raw = _kv_host(0.01)
_kv_1p01_raw = _kv_host(1.01)
_kv_3p01_raw = _kv_host(3.01)
_kv_4p01_raw = _kv_host(4.01)

@jax.custom_jvp
def kv_2p01(z):
    return _kv_host(2.01)(z)

@kv_2p01.defjvp
def _kv_2p01_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    primal_out = kv_2p01(z)
    # standard Bessel derivative recurrence: d/dz K_v(z) = -0.5*(K_{v-1}(z) + K_{v+1}(z))
    dkdz = -0.5 * (_kv_1p01_raw(z) + _kv_3p01_raw(z))
    return primal_out, dkdz * z_dot

def kvp_2p01(z, n):
    """
    Analogue of scipy.special.kvp for v=2.01 degrees of freedom. Only need n=0,1,2.
    Forward-value only for n=1,2 (used solely in build_matrices, which is never
    differentiated); n=0 stays differentiable (used in fit_phisigma) via kv_2p01's jvp.
    """
    z = jnp.where(z == 0, 1e-10, z)
    if n == 0:
        return kv_2p01(z)
    elif n == 1:
        return -0.5 * (_kv_1p01_raw(z) + _kv_3p01_raw(z))
    elif n == 2:
        return 0.25 * (_kv_0p01_raw(z) + 2 * kv_2p01(z) + _kv_4p01_raw(z))
    else:
        raise ValueError(f"n={n} not supported, use 0, 1, or 2")

def matern_v01(x, y, phi1, phi2):
    """
    General Matern kernel for v=2.01 degrees of freedom (see build_matrices_d in
    _helpers.py for the same formula/diagonal-handling pattern applied per-dimension).
    """
    v = 2.01
    sq_dists = jnp.sum(
        (x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :])**2, axis=-1
    )
    l = jnp.sqrt(jnp.clip(sq_dists, min=1e-10))
    u = jnp.sqrt(2*v) * l / phi2
    # avoid 0*inf cancellation as u->0 (K_v(u) diverges, u**v->0); patch the true r=0
    # limit (phi1) back in afterward. Unlike build_matrices_d's diagonal-NaN trick (never
    # differentiated), this function IS differentiated (fit_phisigma's BFGS), so the
    # masked-out branch must stay finite -- NaN here would poison d(cov)/d(phi2) through
    # jnp.where's backward pass (0 * NaN = NaN), even though the forward value is masked.
    u = jnp.where(sq_dists == 0, 1.0, u)
    coef = (phi1 / jax.scipy.special.gamma(v)) * (2 ** (1 - v/2)) * ((jnp.sqrt(v) / phi2) ** v)
    cov = coef * kv_2p01(u) * (l ** v)
    return jnp.where(sq_dists == 0, phi1, cov)