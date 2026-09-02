"""
Exp 11: Gaussian variational inference by exact-Hessian fixed point, with a built-in certificate.

The third-order correction is a Taylor truncation of a quantity we can evaluate exactly. Minimising
KL(q || p) over Gaussians has stationarity conditions (Bonnet and Price)

    E_q[grad log p] = 0        and        Sigma^-1 = E_q[-hess log p],

so the Laplace approximation is precisely the ZEROTH iterate of a fixed point whose steps we can
already compute: gradients in batch, and the exact Hessian from the Gauss-Newton machinery. Two
things make the iteration practical here.

  * ANTITHETIC PAIRS. Exp 2's naive estimator of E_q[grad log p] diverged; exp 3 showed why -- the
    O(delta) term dominates the variance and is odd, so evaluating at mu +- delta cancels it
    exactly and leaves the cubic signal. The same pairing is used for the Hessian.
  * FIXED BASE SAMPLES. Reusing one set of standard normals across iterations makes this a sample
    average approximation: it converges to a genuine fixed point rather than wandering, and the
    result is deterministic given the seed.

The certificate is the size of the remaining step, tau = ||H^(1/2) Sigma E_q[grad log p]|| / sqrt(d),
in posterior sd per dimension. It requires no reference. Its resolution is bounded by Monte Carlo
error, estimated here by splitting the sample in half, so the report is "converged to within X".
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler
from pipeline import metrics

G = H.Gold(); d = H.DIM
m, x_map, Hs, Sig, L = cache("baseline")
gold = np.asarray(G.pos, np.float64)
sc = metrics(gold.mean(0), np.cov(gold, rowvar=False), d)
pr = Profiler(GaussNewtonMAP(m), m)
grad = jax.jit(lambda P: m.gradient(P, m.data))
hess = jax.jit(jax.vmap(pr._hess))

err = float(jnp.linalg.norm(pr._hess(jnp.asarray(x_map)) - jnp.asarray(Hs)) / np.linalg.norm(Hs))
print(f"exact-Hessian check against jax.hessian at the MAP: rel err {err:.2e}\n")

NP, NH = 1024, 96                       # antithetic pairs for the gradient / for the Hessian
Z = np.random.default_rng(0).standard_normal((NP, d))
mu = np.asarray(x_map).copy(); S = Sig.copy()
tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))

print(f'{"iter":>5} {"update":>16} {"bias":>7} {"trace":>7} {"forstner":>9} {"KL":>7} '
      f'{"tau (cert)":>11} {"+-MC":>7} {"sec":>6}')
r = sc(mu, S)
print(f'{0:>5} {"Laplace":>16} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} '
      f'{r["kl"]:>7.2f} {"":>11} {"":>7}')

for it in range(1, 7):
    t0 = time.time()
    Ch = np.linalg.cholesky(S + 1e-14 * np.trace(S) / d * np.eye(d))
    off = Z @ Ch.T
    Pm = np.concatenate([mu + off, mu - off])
    g = np.asarray(grad(jnp.asarray(Pm)))
    step = S @ g.mean(0)
    mc = float(np.std([tau(S @ g[i::2].mean(0)) for i in (0, 1)]) / np.sqrt(2))
    mu = mu + step
    if it % 2 == 0:                                   # covariance step, the costlier half
        o2 = Z[:NH] @ Ch.T
        Hm = np.asarray(hess(jnp.asarray(np.concatenate([mu + o2, mu - o2])))).mean(0)
        Hm = 0.5 * (Hm + Hm.T)
        w, Vv = np.linalg.eigh(Hm)
        S = (Vv / np.maximum(w, 1e-8 * w.max())) @ Vv.T
    r = sc(mu, S); dt = time.time() - t0
    print(f'{it:>5} {("mean + cov" if it%2==0 else "mean"):>16} {r["bias"]:>7.4f} '
          f'{r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>7.2f} {tau(step):>11.4f} '
          f'{mc:>7.4f} {dt:>6.2f}')

mu3 = np.load("build_baseline.npz")["mu3"]
r3 = sc(mu3, Sig)
print(f'\n{"reference points":>22} {"bias":>7} {"trace":>7} {"forstner":>9} {"KL":>7}')
print(f'{"third-order + Laplace":>22} {r3["bias"]:>7.4f} {r3["trace"]:>7.4f} {r3["forst"]:>9.4f} {r3["kl"]:>7.2f}')
print(f'{"floor":>22} {0.0042:>7.4f} {1.0001:>7.4f} {0.0767:>9.4f} {0.48:>7.2f}')
np.savez("gvi_baseline.npz", mu=mu, S=S)
