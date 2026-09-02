"""
Exp 32: an adaptive, guarded profile, tested where the Laplace covariance actually fails.

Exp 31's profile returned NaN on the one parameter it was needed for. The cause is the grid: it
spans +-4.5 LAPLACE standard deviations, and at the noisy setting the Laplace sd for theta_b is
1.7x too large, so the grid reaches +-7.6 true standard deviations -- far enough into the tail that
the Hessian restricted to the complement stops being positive definite and the Cholesky fails.

Two fixes, both cheap. Bracket the grid on the profile's OWN scale by solving once on a narrow
grid and re-gridding; and replace the Cholesky of H_perp with an eigendecomposition that reports a
node as unusable rather than returning NaN for the whole direction.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d, P = H.DIM, 3

def profile_sd(pr, m, v, x0, sd_init, half_width=4.0, nz=19, rounds=2):
    """Marginal sd along unit direction v, with the grid rebracketed on the profile's own scale."""
    v = jnp.asarray(v / np.linalg.norm(v))
    sd = sd_init
    for _ in range(rounds):
        zs = np.linspace(-half_width * sd, half_width * sd, nz)
        U, ld, ok = [], [], []
        warm = np.asarray(x0)
        for zt in zs:
            x = jnp.asarray(warm)
            c0 = float(v @ jnp.asarray(x0)) + zt
            for _ in range(6):
                x = pr._step(x, v, c0, 1e-8)
            if not bool(jnp.all(jnp.isfinite(x))):
                U.append(np.nan); ld.append(np.nan); ok.append(False); continue
            Hm = np.asarray(pr._hess(x), np.float64); Hm = 0.5 * (Hm + Hm.T)
            w, Q = np.linalg.eigh(Hm)
            if w.min() <= 0:                       # not a valid Laplace point: drop this node
                U.append(np.nan); ld.append(np.nan); ok.append(False); continue
            vv = np.asarray(v, np.float64)
            U.append(-float(m.logdensity(x, m.data)))
            ld.append(0.5 * (np.sum(np.log(w)) + np.log(vv @ ((Q / w) @ Q.T) @ vv)))
            ok.append(True)
            warm = np.asarray(x)
        U, ld, ok = np.array(U), np.array(ld), np.array(ok)
        if ok.sum() < 5:
            return np.nan, int((~ok).sum())
        zz, lg = zs[ok], (-U - ld)[ok]
        w_ = np.exp(lg - lg.max())
        Z = np.trapezoid(w_, zz); mu = np.trapezoid(w_ * zz, zz) / Z
        var = np.trapezoid(w_ * (zz - mu) ** 2, zz) / Z
        if not np.isfinite(var) or var <= 0:
            return np.nan, int((~ok).sum())
        sd = np.sqrt(var)
    return sd, int((~ok).sum())

for name, rf in [("noisy", "ref5_noisy.npz"), ("baseline", "ref4_baseline.npz"),
                 ("half", "ref5_half.npz"), ("quarter", "ref5_quarter.npz")]:
    z = np.load(rf); rs = np.sqrt(np.diag(z["cov"])); hc = z["half_cov"]
    m = build(*SETTINGS[name], dtype=jnp.float64)
    m.map_solve(verbose=False, tol=1e-9, max_iter=200)
    x0 = np.asarray(m.map_particle, np.float64)
    Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T
    pr = Profiler(GaussNewtonMAP(m), m)
    print(f'--- {name} ---')
    print(f'{"param":>8} {"Lap/ref":>8} {"prof/ref":>9} {"ref half":>9} {"dropped":>8} {"sec":>6}')
    for i, c in enumerate("abc"):
        e = np.zeros(d); e[i] = 1.0
        sdl = np.sqrt(Sig[i, i])
        t0 = time.time()
        sdp, ndrop = profile_sd(pr, m, e, x0, sdl)
        dt = time.time() - t0
        h = np.sqrt(hc[0][i, i] / hc[1][i, i])
        pf = f'{sdp/rs[i]:>9.4f}' if np.isfinite(sdp) else f'{"failed":>9}'
        print(f'{"theta_" + c:>8} {sdl/rs[i]:>8.4f} {pf} {h:>9.4f} {ndrop:>8} {dt:>6.2f}')
    print()
