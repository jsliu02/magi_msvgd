"""
Exp 16: does the condition-(A) diagnostic predict how badly the Laplace approximation will fail?

Exp 15 gave a two-Hessian, reference-free test of whether p(X|theta) is exactly Gaussian:
    dA = ||H_XX(theta, X1) - H_XX(theta, X2)|| / ||H_XX||,
zero exactly when f is affine in the state. Exp 13 validated a second certificate, `disagree` --
the gap between the third-order and VI means -- as a predictor of true bias (0.034 vs 0.027 at
baseline, 0.087 vs 0.093 at half). Chaining them asks whether dA, computable before any sampling
or correction, forecasts the size of the correction that will be needed. The cubic coefficient
alpha sweeps the ODE continuously from condition (A) to FitzHugh-Nagumo.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
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

print(f'{"alpha":>7} {"dA (cond A)":>12} {"kappa_S":>9} {"|3rd corr|":>11} {"disagree":>9}')
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    m = make(alpha); gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
    gn.solve(verbose=False, tol=1e-9, max_iter=200)
    x = np.asarray(gn.map_particle, np.float64)
    p, n, D = gn.p, gn.n, gn.D
    Hs = np.asarray(pr._hess(jnp.asarray(x))); Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs); Sig = (V / np.maximum(w, 1e-10 * w.max())) @ V.T
    xp = x.copy(); xp[p:] += 0.7 * np.random.default_rng(0).standard_normal(gn.nD)
    A1, A2 = np.asarray(pr._hess(jnp.asarray(x)))[p:, p:], np.asarray(pr._hess(jnp.asarray(xp)))[p:, p:]
    dA = float(np.linalg.norm(A1 - A2) / np.linalg.norm(A1))

    Sj = jnp.asarray(Sig)
    mu3 = x - 0.5 * np.asarray(Sj @ jax.grad(lambda z: jnp.sum(Sj * pr._hess(z)))(jnp.asarray(x)))
    tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))

    logp = lambda z: m.logdensity(z, m.data)
    grad = jax.jit(lambda P: m.gradient(P, m.data))
    lp = jax.jit(lambda P: jax.vmap(logp)(P))
    hvpb = jax.jit(lambda P, Vs: jax.vmap(lambda z: jax.vmap(
        lambda u: Vs.T @ (-jax.jvp(jax.grad(logp), (z,), (u,))[1]))(Vs.T))(P))
    sd = 1.0 / np.sqrt(w); lp0 = float(lp(jnp.asarray(x[None, :]))[0])
    P4 = np.concatenate([x[None, :] + s * (sd[:, None] * V.T) for s in (-2, -1, 1, 2)])
    Uq = -(np.asarray(lp(jnp.asarray(P4))) - lp0).reshape(4, d)
    qs = np.abs(Uq / np.array([2.0, .5, .5, 2.0])[:, None] - 1).mean(0)
    S_idx = np.argsort(-qs)[:12]; Vs = jnp.asarray(V[:, S_idx]); lamS = w[S_idx]
    Z = np.random.default_rng(0).standard_normal((1024, d))
    Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
    mu = x.copy()
    for _ in range(6):
        off = Z @ Ch.T
        Pm = jnp.asarray(np.concatenate([mu + off, mu - off]))
        g = np.asarray(grad(Pm)).mean(0)
        Bk = np.asarray(hvpb(Pm[:192], Vs)).mean(0); Bk = 0.5 * (Bk + Bk.T)
        Ai = Sig + V[:, S_idx] @ (np.linalg.inv(Bk) - np.diag(1.0 / lamS)) @ V[:, S_idx].T
        mu = mu + Ai @ g
    print(f'{alpha:>7.2f} {dA:>12.2e} {float((np.diag(Bk)/lamS).max()):>9.2f} '
          f'{tau(mu3-x):>11.5f} {tau(mu3-mu):>9.4f}')
