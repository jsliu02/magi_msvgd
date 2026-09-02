"""
Exp 12: low-rank Gaussian VI -- the VI fixed point solved only where it is identifiable.

Exp 11's plain fixed point diverged for two reasons with one cause. The mean step Sigma E_q[grad]
overshot by ~1.6x because the correct Newton preconditioner is E_q[H], not H(x_MAP), and the two
differ exactly in the soft directions where exp 3 measured the potential to be far steeper than
quadratic. The covariance step then exploded because a 325x325 matrix average over 192 samples has
O(sqrt(d/n)) eigenvalue noise that inversion amplifies.

Both are fixed by refusing to estimate what the samples cannot support. The screen from exp 4
supplies an m-dimensional subspace S carrying the non-quadraticity; restricted there, E_q[H] is an
m x m block estimated from the same 192 samples -- an easy problem instead of an impossible one.
Since S is spanned by eigenvectors of H, the corrected curvature is block diagonal in H's
eigenbasis and inverts in closed form:

    A = H + V_S Delta V_S^T,   A^-1 = sum_{i not in S} v_i v_i^T / lambda_i
                                      + V_S (diag(lambda_S) + Delta)^-1 V_S^T.

Only V_S^T H(x) V_S is ever needed, so m Hessian-VECTOR products per sample replace a full Hessian
-- 10 HVPs instead of a 325x325 assembly. A is used as the mean step's preconditioner and, tested
separately, as the covariance itself: a Laplace approximation VI-corrected in the subspace where
the correction is both needed and identifiable.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
from pipeline import metrics

G = H.Gold(); d = H.DIM
m, x_map, Hs, Sig, L = cache("baseline")
gold = np.asarray(G.pos, np.float64)
sc = metrics(gold.mean(0), np.cov(gold, rowvar=False), d)
b = dict(np.load("build_baseline.npz"))
ev, V, order, mu3 = b["ev"], b["V"], b["order"], b["mu3"]

logp = lambda z: m.logdensity(z, m.data)
grad = jax.jit(lambda P: m.gradient(P, m.data))
@jax.jit
def hvp_block(P, Vs):
    """V_S^T H(x) V_S for each x in P, by m Hessian-vector products (H = -hess log p)."""
    g1 = lambda z, u: -jax.jvp(jax.grad(logp), (z,), (u,))[1]
    return jax.vmap(lambda z: jax.vmap(lambda u: Vs.T @ g1(z, u))(Vs.T))(P)

NP, MS = 1024, 12
S_idx = order[:MS]
Vs = jnp.asarray(V[:, S_idx])
lamS = ev[S_idx]
Z = np.random.default_rng(0).standard_normal((NP, d))
Ch = np.linalg.cholesky(Sig + 1e-14 * np.trace(Sig) / d * np.eye(d))
tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))

def Ainv_of(Delta):
    """(H + V_S Delta V_S^T)^-1 in closed form; Delta is the S-block correction."""
    Bi = np.linalg.inv(np.diag(lamS) + Delta)
    return Sig + V[:, S_idx] @ (Bi - np.diag(1.0 / lamS)) @ V[:, S_idx].T

print(f'{"iter":>5} {"bias":>7} {"trace":>7} {"forstner":>9} {"KL":>7} {"tau(step)":>10} '
      f'{"lam_S ratio E_q[H]/H":>22} {"sec":>6}')
mu = np.asarray(x_map).copy()
r = sc(mu, Sig)
print(f'{0:>5} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>7.2f}')
for it in range(1, 6):
    t0 = time.time()
    off = Z @ Ch.T                                       # q's shape stays the Laplace one
    P = jnp.asarray(np.concatenate([mu + off, mu - off]))
    g = np.asarray(grad(P)).mean(0)
    Bk = np.asarray(hvp_block(P[:2 * min(96, NP):1][:192], Vs)).mean(0)
    Bk = 0.5 * (Bk + Bk.T)
    Delta = Bk - np.diag(lamS)
    Ai = Ainv_of(Delta)
    step = Ai @ g
    mu = mu + step
    r = sc(mu, Sig); dt = time.time() - t0
    rat = np.diag(Bk) / lamS
    print(f'{it:>5} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>7.2f} '
          f'{tau(step):>10.4f} {str(np.round(rat[:4], 2)):>22} {dt:>6.2f}')

print(f'\ncovariance from the same corrected curvature (mean held at the converged VI mean):')
r = sc(mu, Ainv_of(Delta))
print(f'{"low-rank VI cov":>22} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>7.2f}')
for lbl, mm, SS in [("third-order + Laplace", mu3, Sig), ("Laplace", np.asarray(x_map), Sig)]:
    r = sc(mm, SS)
    print(f'{lbl:>22} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>7.2f}')
print(f'{"floor":>22} {0.0042:>7.4f} {1.0001:>7.4f} {0.0767:>9.4f} {0.48:>7.2f}')
print(f'\ntau at the third-order point (is mu3 already the VI fixed point?):')
off = Z @ Ch.T
g3 = np.asarray(grad(jnp.asarray(np.concatenate([mu3 + off, mu3 - off])))).mean(0)
print(f'  tau(A^-1 E_q[grad log p] at mu3) = {tau(Ainv_of(Delta) @ g3):.4f}   '
      f'at the VI mean = {tau(Ainv_of(Delta) @ np.asarray(grad(jnp.asarray(np.concatenate([mu+off, mu-off])))).mean(0)):.4f}')
np.savez("gvi_lowrank_baseline.npz", mu=mu, Delta=Delta, S_idx=S_idx)
