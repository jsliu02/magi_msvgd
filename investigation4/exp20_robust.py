"""
Exp 20: is the recommended pipeline sensitive to its two free choices?

The low-rank VI solve has two knobs picked by judgement rather than derivation: the subspace size
MS (12) and the number of antithetic pairs NP (1024). If either materially moves the VI mean or the
`disagree` certificate, the certificate is measuring the knobs rather than the posterior, and the
recommendation needs a tuning caveat. Also varied: the random seed for the base samples, since the
"deterministic given the seed" claim is only useful if the seed does not matter.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
from pipeline import metrics
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d = H.DIM
z = np.load("ref4_baseline.npz"); Dz = np.load("determ_baseline.npz")
sc = metrics(z["mean"], z["cov"], d); I = np.eye(d)
m, x_map, Hs, Sig, L = cache("baseline")
pr = Profiler(GaussNewtonMAP(m), m)
logp = lambda u: m.logdensity(u, m.data)
grad = jax.jit(lambda P: m.gradient(P, m.data))
lp = jax.jit(lambda P: jax.vmap(logp)(P))
hvpb = jax.jit(lambda P, Vs: jax.vmap(lambda u0: jax.vmap(
    lambda u: Vs.T @ (-jax.jvp(jax.grad(logp), (u0,), (u,))[1]))(Vs.T))(P))

ev, V = np.linalg.eigh(Hs); sd = 1.0 / np.sqrt(ev)
x0 = np.asarray(x_map); lp0 = float(lp(jnp.asarray(x0[None, :]))[0])
P4 = np.concatenate([x0[None, :] + s * (sd[:, None] * V.T) for s in (-2, -1, 1, 2)])
Uq = -(np.asarray(lp(jnp.asarray(P4))) - lp0).reshape(4, d)
qs = np.abs(Uq / np.array([2.0, .5, .5, 2.0])[:, None] - 1).mean(0)
Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))
mu3 = Dz["mu3"]

def vi(MS, NP, seed, nh=192):
    S_idx = np.argsort(-qs)[:MS]; Vs = jnp.asarray(V[:, S_idx]); lamS = ev[S_idx]
    Z = np.random.default_rng(seed).standard_normal((NP, d))
    mu = x0.copy()
    for _ in range(6):
        off = Z @ Ch.T
        Pm = jnp.asarray(np.concatenate([mu + off, mu - off]))
        g = np.asarray(grad(Pm)).mean(0)
        Bk = np.asarray(hvpb(Pm[:min(nh, 2*NP)], Vs)).mean(0); Bk = 0.5 * (Bk + Bk.T)
        Ai = Sig + V[:, S_idx] @ (np.linalg.inv(Bk) - np.diag(1.0/lamS)) @ V[:, S_idx].T
        mu = mu + Ai @ g
    return mu

print(f'{"MS":>4} {"NP":>6} {"seed":>5} {"bias(VI)":>9} {"disagree":>9} {"bias(midpoint)":>15}')
for MS, NP, seed in [(12,1024,0),(6,1024,0),(24,1024,0),(12,256,0),(12,2048,0),
                     (12,1024,1),(12,1024,2)]:
    mu = vi(MS, NP, seed)
    print(f'{MS:>4} {NP:>6} {seed:>5} {sc(mu,I)["bias"]:>9.4f} {tau(mu3-mu):>9.4f} '
          f'{sc(0.5*(mu3+mu),I)["bias"]:>15.4f}')
print(f'\n{"third-order alone":>26} bias {sc(mu3,I)["bias"]:.4f}   floor 0.0062')
