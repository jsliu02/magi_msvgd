"""
investigation4 setup: correct MAP + exact Hessian for every problem setting.

Everything in investigations 2 and 3 that used a metric was built on a MAP with gradient norm
~1e3. This rebuilds the cache with the Gauss-Newton solver so the whole investigation stands on
a genuine stationary point, and records the Hessian spectrum alongside.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from magi import MAGI
from gauss_newton import GaussNewtonMAP

SETTINGS = {"baseline": (1, 0.2), "half": (2, 0.2), "quarter": (4, 0.2), "noisy": (1, 0.5)}

def build(stride, sigma, dtype=jnp.float64):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")[::stride]
    g = np.arange(0, 20.001, 0.125)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], d[:, 0])] = d
    m = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    m.put(dtype=dtype, device=jax.devices()[0])
    return m

def cache(name, force=False):
    """(magi, x_map, H, Sigma, L=H^-1/2) with the Gauss-Newton MAP."""
    f = f"map4_{name}.npz"
    m = build(*SETTINGS[name])
    if os.path.exists(f) and not force:
        z = np.load(f)
        return m, z["x_map"], z["H"], z["Sig"], z["L"]
    gn = GaussNewtonMAP(m)
    gn.solve(verbose=False, tol=1e-8, max_iter=200)
    x = np.asarray(gn.map_particle, np.float64)
    Hs = np.asarray(jax.hessian(lambda u: -m.logdensity(u, m.data))(jnp.asarray(x)), np.float64)
    Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs)
    evc = np.maximum(ev, 1e-10 * ev.max())
    Sig = (V / evc) @ V.T
    L = (V / np.sqrt(evc)) @ V.T                                  # symmetric H^-1/2
    np.savez(f, x_map=x, H=Hs, Sig=Sig, L=L, evals=ev)
    return m, x, Hs, Sig, L

if __name__ == "__main__":
    print(f'{"setting":>10} {"||grad||":>10} {"log p":>12} {"min eig":>10} {"cond":>10} {"tr(Sig)":>10}')
    for name in SETTINGS:
        m, x, Hs, Sig, L = cache(name, force=True)
        g = float(jnp.linalg.norm(m.gradient(jnp.asarray(x)[None, :], m.data)))
        ev = np.linalg.eigvalsh(Hs)
        print(f'{name:>10} {g:>10.2e} {float(m.logdensity(jnp.asarray(x), m.data)):>12.4f} '
              f'{ev.min():>10.4f} {ev.max()/ev.min():>10.2e} {np.trace(Sig):>10.4f}')
