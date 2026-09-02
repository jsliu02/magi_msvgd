"""
Exp 14: an exact condition on the ODE under which the Laplace approximation is exact, and a test
that the error scales the way the theory says.

With sigma fixed, MAGI's negative log posterior is EXACTLY a sum of squares, -2 log p = ||R(x)||^2,
and only the ODE block of R is nonlinear -- through f alone. Differentiating twice,

    hess R_a = sqrt(b) (Lk^T)_a hess f            (pointwise in the grid index),

so the residual's whole departure from linearity is the ODE's second derivative. Writing
delta = x - x_MAP and E(delta) = R(x_MAP + delta) - R_MAP - J delta, stationarity gives
R_MAP^T J = 0, hence the EXACT identity (no truncation anywhere)

    log q(x) - log p(x) = (R_MAP + J delta)^T E(delta) + 0.5 ||E(delta)||^2 + const,
    q = N(x_MAP, (J^T J)^-1).

Two consequences, both tested here.

  CONDITION (L). If f is affine in (X, theta) jointly then hess f = 0, so E = 0 identically and
  the posterior is EXACTLY Gaussian -- Laplace is not an approximation, and no correction of any
  kind is needed. This is checkable by autodiff before any sampling is attempted.

  SCALING. Away from that case every error term is proportional to hess f through
  Lambda = sqrt(b) max_i ||Lk_i|| L2, so scaling the ODE's nonlinearity by alpha should scale
  ||E|| linearly and the leading mean error linearly in alpha. FitzHugh-Nagumo has exactly one
  nonlinear term, c V^3/3, so alpha multiplies it and interpolates the linear case (alpha = 0) to
  the real problem (alpha = 1), giving a falsifiable prediction rather than a loose bound.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from magi import MAGI
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d = H.DIM

def make(alpha):
    def ode(X, theta, t=None):
        V, R = X.T; a, b, c = theta
        return jnp.stack([c * (V - alpha * V ** 3 / 3 + R), -1 / c * (V - a + b * R)])
    dd = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")
    g = np.arange(0, 20.001, 0.125)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], dd[:, 0])] = dd
    m = MAGI(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    m.put(dtype=jnp.float64, device=jax.devices()[0])
    return m

print(f'{"alpha":>7} {"L2 = sup||hess f||":>19} {"Lambda":>9} {"||E|| at 1 sd":>14} '
      f'{"bias(MAP)":>10} {"bias(3rd)":>10} {"ratio to alpha":>15}')
base = None
for alpha in [0.0, 0.25, 0.5, 1.0]:
    m = make(alpha)
    gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
    gn.solve(verbose=False, tol=1e-9, max_iter=200)
    x = np.asarray(gn.map_particle, np.float64)
    Hs = np.asarray(pr._hess(jnp.asarray(x)), np.float64); Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs); Sig = (V / np.maximum(w, 1e-10 * w.max())) @ V.T

    # L2: sup over the posterior bulk of the pointwise ODE Hessian
    D, p, n = gn.D, gn.p, gn.n
    def f_local(z, t): return m.ode(z[:D][None, :], z[D:], t[None])[0]
    hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
    Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
    S = x[None, :] + np.random.default_rng(0).standard_normal((256, d)) @ Ch.T
    L2 = 0.0
    for s in S:
        Z = jnp.concatenate([s[p:].reshape(n, D), jnp.broadcast_to(s[:p], (n, p))], axis=1)
        L2 = max(L2, float(jnp.sqrt(jnp.sum(hl(Z, gn.I) ** 2, axis=(1, 2, 3))).max()))
    Lam = float(gn.b) * float(jnp.linalg.norm(gn.Lk, axis=(1, 2)).max()) * L2

    # exact ||E|| at one posterior sd, measured not bounded
    dl = Ch @ np.random.default_rng(1).standard_normal(d)
    dl = dl / np.sqrt(dl @ Hs @ dl / d)
    r0 = np.asarray(gn.residual(jnp.asarray(x[:p]), jnp.asarray(x[p:].reshape(n, D)), m.sigmas))
    J = np.asarray(gn.jacobian(jnp.asarray(x[:p]), jnp.asarray(x[p:].reshape(n, D)), m.sigmas))
    xx = x + dl
    r1 = np.asarray(gn.residual(jnp.asarray(xx[:p]), jnp.asarray(xx[p:].reshape(n, D)), m.sigmas))
    nE = float(np.linalg.norm(r1 - r0 - J @ dl))

    Sj = jnp.asarray(Sig)
    mu3 = x - 0.5 * np.asarray(Sj @ jax.grad(lambda z: jnp.sum(Sj * pr._hess(z)))(jnp.asarray(x)))
    # "truth" proxy: the low-rank-VI mean is unavailable here, so report the SIZE of the correction
    corr = float(np.sqrt((mu3 - x) @ Hs @ (mu3 - x) / d))
    if alpha == 1.0: base = (L2, Lam, nE, corr)
    ref = corr / alpha if alpha > 0 else float("nan")
    print(f'{alpha:>7.2f} {L2:>19.4f} {Lam:>9.4f} {nE:>14.3e} {"":>10} {corr:>10.5f} {ref:>15.5f}')
print("\n(bias(3rd) column is the SIZE of the third-order mean correction in posterior sd per dim;")
print(" 'ratio to alpha' is that divided by alpha -- constant means the predicted linear scaling holds)")
