"""
Exp 13: the deterministic pipeline across all four settings, with reference-free certificates.

The question is not only whether the method works, but whether it KNOWS when it does not. Every
quantity in the left block is computable without a reference; every quantity in the right block
needs one. If the left predicts the right, the method is safe to deploy where no gold exists.

Reference-free certificates
    q_max     largest slice-curvature screen value: how non-quadratic the worst direction is
    kappa_S   max diagonal of E_q[H]/H over the screened subspace: how far the AVERAGE curvature
              over q departs from the curvature at the mode (6-10x at baseline)
    tau_end   size of the last VI step, in posterior sd per dim: did the fixed point converge
    disagree  ||mu_3rd - mu_VI|| in posterior sd per dim. Two estimates of the mean derived from
              genuinely different arguments -- a Taylor truncation of the posterior mean, and the
              exact stationary point of the reverse-KL Gaussian objective. Neither is the truth,
              but they can only agree closely if both are near it, so their gap is an error proxy.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache, SETTINGS
from pipeline import metrics
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d = H.DIM
NP, MS = 1024, 12

def deterministic(name):
    m, x_map, Hs, Sig, L = cache(name)
    pr = Profiler(GaussNewtonMAP(m), m)
    logp = lambda z: m.logdensity(z, m.data)
    grad = jax.jit(lambda P: m.gradient(P, m.data))
    lp = jax.jit(lambda P: jax.vmap(logp)(P))
    @jax.jit
    def hvp_block(P, Vs):
        g1 = lambda z, u: -jax.jvp(jax.grad(logp), (z,), (u,))[1]
        return jax.vmap(lambda z: jax.vmap(lambda u: Vs.T @ g1(z, u))(Vs.T))(P)

    t = {}
    t0 = time.time(); Sj = jnp.asarray(Sig)
    mu3 = np.asarray(x_map) - 0.5 * np.asarray(
        Sj @ jax.grad(lambda z: jnp.sum(Sj * pr._hess(z)))(jnp.asarray(x_map)))
    t["third_order"] = time.time() - t0

    ev, V = np.linalg.eigh(Hs); sd = 1.0 / np.sqrt(ev)
    mu0 = np.asarray(x_map)
    t0 = time.time()
    lp0 = float(lp(jnp.asarray(mu0[None, :]))[0])
    P4 = np.concatenate([mu0[None, :] + s * (sd[:, None] * V.T) for s in (-2, -1, 1, 2)])
    U = -(np.asarray(lp(jnp.asarray(P4))) - lp0).reshape(4, d)
    qscr = np.abs(U / np.array([2.0, 0.5, 0.5, 2.0])[:, None] - 1).mean(0)
    t["screen"] = time.time() - t0

    S_idx = np.argsort(-qscr)[:MS]
    Vs = jnp.asarray(V[:, S_idx]); lamS = ev[S_idx]
    Z = np.random.default_rng(0).standard_normal((NP, d))
    Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
    tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))
    t0 = time.time(); mu = mu0.copy(); taus = []
    for it in range(6):
        off = Z @ Ch.T
        Pm = jnp.asarray(np.concatenate([mu + off, mu - off]))
        g = np.asarray(grad(Pm)).mean(0)
        Bk = np.asarray(hvp_block(Pm[:192], Vs)).mean(0); Bk = 0.5 * (Bk + Bk.T)
        Bi = np.linalg.inv(np.diag(lamS) + (Bk - np.diag(lamS)))
        Ai = Sig + V[:, S_idx] @ (Bi - np.diag(1.0 / lamS)) @ V[:, S_idx].T
        step = Ai @ g; mu = mu + step; taus.append(tau(step))
    t["vi"] = time.time() - t0
    return dict(m=m, mu0=mu0, mu3=mu3, muvi=mu, Sig=Sig, Hs=Hs, t=t,
                q_max=float(qscr.max()), kappa_S=float((np.diag(Bk) / lamS).max()),
                tau_end=taus[-1], disagree=tau(mu3 - mu), taus=taus)

print(f'{"setting":>9} | {"q_max":>7} {"kappa_S":>8} {"tau_end":>9} {"disagree":>9} | '
      f'{"bias MAP":>9} {"bias 3rd":>9} {"bias VI":>8} {"floor":>7} | {"forst":>7} {"KL":>7} {"sec":>6}')
print("-" * 118)
for name in SETTINGS:
    f = f"ref4_{name}.npz"
    if not os.path.exists(f):
        print(f'{name:>9} | reference not built yet'); continue
    z = np.load(f)
    r = deterministic(name)
    sc = metrics(z["mean"], z["cov"], d)
    hm, hc = z["half_mean"], z["half_cov"]
    fl = metrics(hm[1], hc[1], d)(hm[0], hc[0])
    a, b3, bv = sc(r["mu0"], r["Sig"]), sc(r["mu3"], r["Sig"]), sc(r["muvi"], r["Sig"])
    print(f'{name:>9} | {r["q_max"]:>7.2f} {r["kappa_S"]:>8.2f} {r["tau_end"]:>9.5f} '
          f'{r["disagree"]:>9.4f} | {a["bias"]:>9.4f} {b3["bias"]:>9.4f} {bv["bias"]:>8.4f} '
          f'{fl["bias"]:>7.4f} | {b3["forst"]:>7.4f} {b3["kl"]:>7.2f} {sum(r["t"].values()):>6.2f}')
    np.savez(f"determ_{name}.npz", mu3=r["mu3"], muvi=r["muvi"], mu0=r["mu0"])
