"""
Exp 9: does the banded structure give an ASYMPTOTIC win, and what does the fast pipeline cost
end to end?

A Matern GP has a fixed correlation length in time, so refining the grid should widen the
precision's bandwidth in grid points proportionally -- in which case a banded solve is a constant
factor, not a better complexity. Worth checking rather than assuming.
"""
import numpy as np, jax, jax.numpy as jnp, time, os, scipy.linalg as sla
jax.config.update("jax_enable_x64", True)
import harness as H
from magi import MAGI
from lsq import LSQ
from jac import AnalyticJac

def build(step):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")
    g = np.arange(0, 20.0 + step / 2, step)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(np.round(full[:, 0], 9), np.round(d[:, 0], 9))] = d
    mm = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    mm.put(dtype=jnp.float64, device=jax.devices()[0]); return mm

def bandwidth(A, tol=1e-8):
    sc = np.abs(A).max()
    return next((k for k in range(A.shape[0]) if np.abs(np.diag(A, k)).max() / sc < tol), A.shape[0])

print("(a) does the precision bandwidth grow with the grid?")
print(f'{"step":>8} {"n":>6} {"bw(C^-1)":>10} {"bw/n":>7} {"bw(K^-1)":>10} {"bw/n":>7}')
for step in [0.25, 0.125, 0.0625]:
    mm = build(step); n = mm.n
    bc = bandwidth(np.asarray(mm.C_invs)[0]); bk = bandwidth(np.asarray(mm.K_invs)[0])
    print(f'{step:>8} {n:>6} {bc:>10} {bc/n:>7.3f} {bk:>10} {bk/n:>7.3f}')

print("\n(b) end-to-end deterministic pipeline, fast solver")
m = H.build_magi(dtype=jnp.float64)
l = LSQ(m); aj = AnalyticJac(l)
n, D, P = l.n, l.D, l.P; nD = n * D; b = np.sqrt(l.b)
def f_local(z, t): return m.ode(z[:D][None, :], z[D:], t[None])[0]
hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
IDX = jnp.asarray(np.concatenate([(P + np.arange(n)[:, None]*D + np.arange(D)[None, :]),
                  np.broadcast_to(np.arange(P)[None, :], (n, P))], axis=1))
def hess_U(x):
    J = aj(x); Hu = J.T @ J
    c = b * jnp.einsum('nd,dmn->md', l.residual(x)[2*nD:].reshape(n, D), aj.Lk)
    Z = jnp.concatenate([x[P:P+nD].reshape(n, D), jnp.broadcast_to(x[:P], (n, P))], axis=1)
    S = jnp.einsum('md,mdij->mij', c, hl(Z, aj.I))
    return Hu + jnp.zeros_like(Hu).at[IDX[:, :, None], IDX[:, None, :]].add(S)

@jax.jit
def gn_step(x):
    Jx = aj(x); A = Jx.T @ Jx
    c = jax.scipy.linalg.cho_factor(A + 1e-12*jnp.trace(A)/A.shape[0]*jnp.eye(A.shape[0]))
    return x - jax.scipy.linalg.cho_solve(c, Jx.T @ l.residual(x))

x = jnp.asarray(np.asarray(m.particles_init, np.float64))
gn_step(x).block_until_ready(); hess_U(x).block_until_ready()      # warm the compiles
x = jnp.asarray(np.asarray(m.particles_init, np.float64))
t0 = time.time()
for _ in range(30): x = gn_step(x)
x.block_until_ready(); t_map = time.time() - t0
t0 = time.time(); Hn = np.asarray(hess_U(x)); Hn = .5*(Hn+Hn.T); t_hess = time.time() - t0
t0 = time.time(); Sig = jnp.asarray(np.linalg.inv(Hn))
v = jax.grad(lambda z: jnp.sum(Sig * hess_U(z)))(x); v.block_until_ready(); t_corr = time.time()-t0
xc = np.asarray(x) - 0.5 * np.asarray(Sig @ v)
G = H.Gold()
bias = lambda mu: float(np.sqrt((G.whiten(np.asarray(mu)[None, :]) ** 2).mean()))
print(f"    MAP (30 Gauss-Newton steps)  {t_map:6.2f}s   ||grad||="
      f"{float(jnp.linalg.norm(m.gradient(x[None,:], m.data))):.1e}")
print(f"    exact Hessian                {t_hess:6.2f}s")
print(f"    third-order mean correction  {t_corr:6.2f}s")
print(f"    TOTAL                        {t_map+t_hess+t_corr:6.2f}s   "
      f"(was 21.2s)   bias {bias(x):.4f} -> {bias(xc):.4f}")
rng = np.random.default_rng(0)
Sh = np.linalg.cholesky(np.linalg.inv(Hn) + 1e-14*np.eye(H.DIM))
print(); print(H.HDR); print("-"*len(H.HDR))
for tag, mu in [("N(MAP, H^-1) fast", np.asarray(x)), ("N(MAP+corr, H^-1) fast", xc)]:
    H.show(H.evaluate(jnp.asarray(mu[None,:] + rng.standard_normal((800, H.DIM)) @ Sh.T), m, tag=tag))
H.show(H.gold_row())
