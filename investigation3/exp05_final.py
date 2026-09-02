"""
Exp 5: (a) why RTO's weights fail, (b) the deterministic pipeline across settings.
"""
import numpy as np, jax, jax.numpy as jnp, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from magi import MAGI
from lsq import LSQ
from jac import AnalyticJac
from rto2 import RTO2

G = H.Gold()

# ---------------------------------------------------------------- (a) weight decomposition
m = H.build_magi(dtype=jnp.float64)
r = RTO2(m, np.load("laplace_cache.npz")["x_map"])
xi = np.random.default_rng(0).standard_normal((200, H.DIM))
X, lw, kap, _ = r.run(xi, n_it=10, chunk=200)
Xj = jnp.asarray(X)
A = jax.vmap(r._QtJ)(Xj); _, ld = jnp.linalg.slogdet(A)
Rall = r.Rv(Xj)
r_perp2 = np.asarray(jnp.sum(Rall ** 2, 1) - jnp.sum((Rall @ r.Q) ** 2, 1))
ld = np.asarray(ld)
print("(a) RTO log-weight = -log|det(Qb^T J)| - ||R_perp||^2/2 + const")
print(f"    sd of -log|det| term        : {np.std(-ld):.4f}")
print(f"    sd of -||R_perp||^2/2 term  : {np.std(-0.5*r_perp2):.4f}")
print(f"    sd of the total             : {np.std(lw):.4f}")
print(f"    correlation between them    : {np.corrcoef(-ld, -0.5*r_perp2)[0,1]:+.3f}")
print(f"    R_perp lives in N - d = {r.l.N - H.DIM} dimensions; each contributes a little")
print(f"    variance, so over-determination is what limits RTO here.")

# ---------------------------------------------------------------- (b) deterministic pipeline
def build(stride, sigma):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")[::stride]
    g = np.arange(0, 20.001, 0.125)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], d[:, 0])] = d
    mm = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    mm.put(dtype=jnp.float64, device=jax.devices()[0]); return mm

def pipeline(mm):
    """Gauss-Newton MAP -> exact Hessian -> deterministic third-order mean correction."""
    l = LSQ(mm); aj = AnalyticJac(l)
    n, D, P = l.n, l.D, l.P; nD = n * D; b = np.sqrt(l.b)
    def f_local(z, t): return mm.ode(z[:D][None, :], z[D:], t[None])[0]
    hess_local = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
    idx = np.concatenate([(P + np.arange(n)[:, None]*D + np.arange(D)[None, :]),
                          np.broadcast_to(np.arange(P)[None, :], (n, P))], axis=1)
    IDX = jnp.asarray(idx)
    def hess_U(x):
        J = aj(x); Hu = J.T @ J
        R_ode = l.residual(x)[2*nD:].reshape(n, D)
        c = b * jnp.einsum('nd,dmn->md', R_ode, aj.Lk)
        th = x[:P]; Xs = x[P:P+nD].reshape(n, D)
        Z = jnp.concatenate([Xs, jnp.broadcast_to(th, (n, P))], axis=1)
        Sloc = jnp.einsum('md,mdij->mij', c, hess_local(Z, aj.I))
        return Hu + jnp.zeros_like(Hu).at[IDX[:, :, None], IDX[:, None, :]].add(Sloc)
    mm.particles = None
    import optax
    mm.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=20000, atol=1e-7, rtol=0.0,
             random_seed=0, monitor_convergence=-1, optimizer=optax.contrib.prodigy,
             optimizer_kwargs={})
    x = jnp.asarray(np.asarray(mm.particles[0], np.float64))
    for _ in range(60):
        x = x - jnp.linalg.lstsq(aj(x), l.residual(x), rcond=None)[0]
    Hn = np.asarray(hess_U(x)); Hn = .5*(Hn + Hn.T)
    Sig = jnp.asarray(np.linalg.inv(Hn))
    v = jax.grad(lambda z: jnp.sum(Sig * hess_U(z)))(x)
    return np.asarray(x), np.asarray(x - 0.5 * (Sig @ v)), np.linalg.inv(Hn)

class Ref:
    def __init__(s, pos):
        s.pos = pos; s.mean = pos.mean(0); s.sd = pos.std(0)
        ev, V = np.linalg.eigh(np.cov(pos, rowvar=False))
        s.evals = np.maximum(ev, 1e-14); s.evecs = V
        s.theta_w = np.quantile(pos[:, :3], .975, 0) - np.quantile(pos[:, :3], .025, 0)
        s.ref = pos[np.random.default_rng(0).choice(len(pos), 2000, False)]
    def whiten(s, X): return (np.asarray(X, np.float64) - s.mean) @ s.evecs / np.sqrt(s.evals)

print("\n(b) fully deterministic pipeline: GN MAP + exact Hessian + third-order correction")
print(f'{"setting":>16} {"bias(MAP)":>10} {"bias(corr)":>11} {"energy MAP":>11} '
      f'{"energy corr":>12} {"floor":>7} {"sec":>6}')
rng = np.random.default_rng(0)
cases = [("baseline", 1, 0.2, Ref(G.pos))]
p = "../investigation2/exp10_ref_noisy.npz"
if os.path.exists(p):
    cases.append(("noisy s=0.5", 1, 0.5, Ref(np.load(p)["pos"])))
for name, stride, sig, R_ in cases:
    mm = build(stride, sig)
    t0 = time.time(); xm, xc, Sg = pipeline(mm); dt = time.time() - t0
    Sh = np.linalg.cholesky(0.5*(Sg+Sg.T) + 1e-14*np.eye(H.DIM))
    def en(mu):
        s = mu[None, :] + rng.standard_normal((800, H.DIM)) @ Sh.T
        return H._energy_distance(R_.whiten(s), R_.whiten(R_.ref), rng)
    fl = H._energy_distance(R_.whiten(R_.pos[rng.choice(len(R_.pos), 800, False)]),
                            R_.whiten(R_.ref), rng)
    bi = lambda mu: float(np.sqrt((R_.whiten(mu[None, :])**2).mean()))
    print(f'{name:>16} {bi(xm):>10.4f} {bi(xc):>11.4f} {en(xm):>11.4f} {en(xc):>12.4f} '
          f'{fl:>7.4f} {dt:>6.1f}')
