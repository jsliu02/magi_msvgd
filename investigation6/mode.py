"""
Mode quality: is the point the Gauss-Newton solver returns actually a posterior mode?

Everything downstream of the MAP -- the Laplace covariance, the third-order correction, the
whitening metric, the profiled posterior's inner solve -- assumes the solver returned a strict
local maximum. Investigation 5 found that on HIV it does not: the exact Hessian there has a
smallest eigenvalue of -1.5e-10 against a largest of 6.5e5, which is machine epsilon relative, so
the point is a numerically FLAT stationary point rather than a maximum. A Gaussian centred there
is not merely inaccurate, it does not exist.

Three checks, all of which are affordable precisely because the MAP solve is fast:

    curvature      the exact Hessian's spectrum, and whether any direction is flat or negative
    escape         if a genuinely negative direction exists, follow it and re-solve; a mode that
                   survives this is at least a local maximum, and one that does not was never a
                   mode at all
    globality      re-solve from dispersed starts and from a homotopy in the prior tempering,
                   and report whether they agree

The homotopy is worth more than random restarts on this class of problem. MAGI's posterior
tempers its GP and ODE terms by beta^-1; at small beta^-1 the objective is dominated by the
observations and is close to convex, and following the solution from there to the target tempering
is a principled way to land in the right basin rather than a nearby wrong one.
"""
import numpy as np, jax, jax.numpy as jnp


def spectrum(m, x=None, tol_rel=1e-12):
    """Exact-Hessian spectrum at x, classified relative to the largest eigenvalue."""
    if x is None:
        x = m.map_particle
    H = np.asarray(m.hessian(np.asarray(x, np.float64)), np.float64)
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    scale = max(abs(w).max(), 1e-300)
    return dict(w=w, V=V, H=H, scale=scale,
                n_neg=int((w < -tol_rel * scale).sum()),
                n_flat=int((np.abs(w) <= tol_rel * scale).sum()),
                min_rel=float(w.min() / scale), cond=float(scale / max(abs(w).min(), 1e-300)))


def escape(m, x=None, tol_rel=1e-12, steps=(1e-3, 1e-2, 1e-1, 1.0), verbose=False):
    """
    If the Hessian has a negative direction, step along it and re-solve.

    Returns (particle, log p, moved) with moved=False when no negative direction exists or when
    no step along it improves the log-density -- the latter being the signature of a flat
    direction rather than a true saddle.
    """
    x = np.asarray(m.map_particle if x is None else x, np.float64)
    s = spectrum(m, x, tol_rel)
    lp0 = float(m.logdensity(jnp.asarray(x, m.mu.dtype), m.data))
    if s["n_neg"] == 0:
        return x, lp0, False
    v = s["V"][:, 0]
    sd = 1.0 / np.sqrt(max(abs(s["w"][0]), 1e-300))
    best = (lp0, x, False)
    for a in steps:
        for sign in (+1, -1):
            xt = x + sign * a * sd * v
            m.map_solve(x0=jnp.asarray(xt, m.mu.dtype), verbose=False, tol=1e-9,
                        max_iter=300, check=False)
            xn = np.asarray(m.map_particle, np.float64)
            lp = float(m.logdensity(m.map_particle, m.data))
            if np.isfinite(lp) and lp > best[0] + 1e-8:
                best = (lp, xn, True)
    if verbose:
        print(f'    escape: log p {lp0:.4f} -> {best[0]:.4f} (moved={best[2]})')
    return best[1], best[0], best[2]


def multistart(m, n=8, spread=0.25, seed=0, verbose=False):
    """Re-solve from dispersed starts; returns the distinct optima found, best first."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(m.particles_init, np.float64)
    p = m.p
    found = []
    for i in range(n):
        xt = x0.copy()
        if i:                       # perturb theta multiplicatively, states additively
            xt[:p] = x0[:p] * np.exp(spread * rng.standard_normal(p))
            xt[p:] = x0[p:] + spread * np.std(x0[p:]) * rng.standard_normal(len(x0) - p)
        try:
            m.map_solve(x0=jnp.asarray(xt, m.mu.dtype), verbose=False, tol=1e-9,
                        max_iter=300, check=False)
            xn = np.asarray(m.map_particle, np.float64)
            lp = float(m.logdensity(m.map_particle, m.data))
            if np.isfinite(lp):
                found.append((lp, xn))
        except Exception:
            pass
    found.sort(key=lambda z: -z[0])
    uniq = []
    for lp, xn in found:
        if all(abs(lp - u[0]) > 1e-4 for u in uniq):
            uniq.append((lp, xn))
    if verbose:
        print(f'    multistart: {len(found)}/{n} solved, {len(uniq)} distinct, '
              f'log p range [{found[-1][0]:.4f}, {found[0][0]:.4f}]' if found else
              '    multistart: none solved')
    return uniq


def homotopy(m, ladder=(0.02, 0.05, 0.15, 0.4, 1.0), verbose=False):
    """
    Continuation in the prior tempering.

    MAGI tempers its GP and ODE terms by beta_inv. Solving first at a small fraction of it leaves
    an objective dominated by the observation term -- nearly a smooth data fit, with a benign
    landscape -- and following the solution up to the target tempering keeps the iterate in that
    basin. Costs one MAP solve per rung, which the fast solver makes negligible.
    """
    beta = float(m.beta_inv)
    x = None
    for frac in ladder:
        m.beta_inv = beta * frac
        m._invalidate()
        m.map_solve(x0=None if x is None else jnp.asarray(x, m.mu.dtype),
                    verbose=False, tol=1e-9, max_iter=300, check=False)
        x = np.asarray(m.map_particle, np.float64)
    m.beta_inv = beta
    m._invalidate()
    m.map_solve(x0=jnp.asarray(x, m.mu.dtype), verbose=False, tol=1e-9, max_iter=300, check=False)
    x = np.asarray(m.map_particle, np.float64)
    lp = float(m.logdensity(m.map_particle, m.data))
    if verbose:
        print(f'    homotopy: log p {lp:.4f}')
    return x, lp


def properness(m, profiler, x=None, reach=(1.0, 1e2, 1e4), drop=3.0, verbose=False):
    """
    Is the posterior proper in each parameter direction?

    A null direction of the quadratic model is not by itself an improper posterior -- the density
    can still decay at higher order. The question matters twice over: an improper direction has no
    posterior mean or variance to estimate, and no MCMC reference can converge along it, so a
    chain left running on one is not slow but futile.

    Walks each parameter axis outward with the states RE-PROFILED at every step, so the walk
    follows the ridge instead of cutting across it, and reports the fall in the profiled
    log-density. A direction that has not fallen by `drop` nats by the furthest reach is reported
    as improper. Measured on HIV: lam falls 0.05 nats out to 1e5 (improper), rho falls 1608.
    """
    x = np.asarray(m.map_particle if x is None else x, np.float64)
    p = m.p
    th0 = x[:p]
    lp0 = profiler.logp(th0[None, :])[0][0]
    out = {}
    for j in range(p):
        scale = max(abs(th0[j]), 1.0)
        ts = np.array([r * scale for r in reach])
        TH = np.repeat(th0[None, :], 2 * len(ts), axis=0)
        TH[:len(ts), j] += ts
        TH[len(ts):, j] -= ts
        lp, _, ok = profiler.logp(TH)
        lp = np.where(ok, lp, -np.inf)
        fall = float(np.nanmin([lp0 - lp[len(ts) - 1], lp0 - lp[-1]]))
        out[j] = dict(fall=fall, proper=bool(fall > drop))
        if verbose:
            print(f'    theta[{j}]: log p_hat falls {fall:>12.3f} nats by {ts[-1]:.1e}'
                  f'  -> {"proper" if out[j]["proper"] else "IMPROPER"}')
    return out
