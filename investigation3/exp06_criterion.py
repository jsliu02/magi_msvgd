"""
Exp 6: an a-priori trust criterion for the deterministic correction, and the small-sample
benefit of Rao-Blackwellization.

The third-order correction is the leading term of an asymptotic expansion, so it helps only
while that term is small. Its size RELATIVE TO THE POSTERIOR SCALE is computable without any
reference: measure the correction in the metric of the Hessian itself,

    tau = || H^{1/2} delta || / sqrt(d),

the correction expressed in posterior standard deviations per dimension. tau << 1 means the
cubic term is a perturbation; tau ~ 1 means it is not and the expansion should not be trusted.
"""
import numpy as np, jax, jax.numpy as jnp, os, optax
jax.config.update("jax_enable_x64", True)
from scipy import stats as sps
import harness as H
from magi import MAGI
from lsq import LSQ
from jac import AnalyticJac

G = H.Gold()

def build(stride, sigma):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")[::stride]
    g = np.arange(0, 20.001, 0.125); full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], d[:, 0])] = d
    mm = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    mm.put(dtype=jnp.float64, device=jax.devices()[0]); return mm

def pipeline(mm):
    l = LSQ(mm); aj = AnalyticJac(l)
    n, D, P = l.n, l.D, l.P; nD = n * D; b = np.sqrt(l.b)
    def f_local(z, t): return mm.ode(z[:D][None, :], z[D:], t[None])[0]
    hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
    IDX = jnp.asarray(np.concatenate([(P + np.arange(n)[:, None]*D + np.arange(D)[None, :]),
                      np.broadcast_to(np.arange(P)[None, :], (n, P))], axis=1))
    def hess_U(x):
        J = aj(x); Hu = J.T @ J
        c = b * jnp.einsum('nd,dmn->md', l.residual(x)[2*nD:].reshape(n, D), aj.Lk)
        Z = jnp.concatenate([x[P:P+nD].reshape(n, D),
                             jnp.broadcast_to(x[:P], (n, P))], axis=1)
        S = jnp.einsum('md,mdij->mij', c, hl(Z, aj.I))
        return Hu + jnp.zeros_like(Hu).at[IDX[:, :, None], IDX[:, None, :]].add(S)
    mm.particles = None
    mm.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=20000, atol=1e-7, rtol=0.0,
             random_seed=0, monitor_convergence=-1, optimizer=optax.contrib.prodigy,
             optimizer_kwargs={})
    x = jnp.asarray(np.asarray(mm.particles[0], np.float64))
    for _ in range(60): x = x - jnp.linalg.lstsq(aj(x), l.residual(x), rcond=None)[0]
    Hn = np.asarray(hess_U(x)); Hn = .5*(Hn+Hn.T)
    Sig = jnp.asarray(np.linalg.inv(Hn))
    delta = -0.5 * (Sig @ jax.grad(lambda z: jnp.sum(Sig * hess_U(z)))(x))
    w = np.linalg.eigvalsh(Hn)
    tau = float(np.sqrt(np.asarray(delta) @ Hn @ np.asarray(delta) / H.DIM))
    return np.asarray(x), np.asarray(x + delta), tau, w

print(f'{"setting":>16} {"tau":>8}  verdict          (tau = correction in posterior sd per dim)")')
for name, st, sg in [("baseline", 1, 0.2), ("half-obs", 2, 0.2),
                     ("quarter-obs", 4, 0.2), ("noisy s=0.5", 1, 0.5)]:
    _, _, tau, w = pipeline(build(st, sg))
    verdict = "trust" if tau < 0.1 else ("borderline" if tau < 0.25 else "DO NOT TRUST")
    print(f'{name:>16} {tau:>8.4f}  {verdict}')

# --------------------------------------------------- Rao-Blackwellization at small sample size
print("\nRao-Blackwellization of (a,b) given (X,c): benefit at small particle counts")
m = H.build_magi(dtype=jnp.float64)
n, D, P = m.n, m.D, m.p
K2inv = np.asarray(m.K_invs, np.float64)[1]
mu = np.asarray(m.mu, np.float64); mudot = np.asarray(m.mu_dot, np.float64)
ms = np.asarray(m.ms, np.float64); binv = float(m.beta_inv)
def ab_cond(Xall):
    th = Xall[:, :P]; Xs = Xall[:, P:P+n*D].reshape(-1, n, D)
    V, R = Xs[:, :, 0], Xs[:, :, 1]; c = th[:, 2]
    A0 = -(V/c[:, None]) - mudot[None, :, 1] - np.einsum('nm,km->kn', ms[1], Xs[:, :, 1]-mu[None, :, 1])
    M = np.stack([np.ones_like(V)/c[:, None], -R/c[:, None]], axis=2)
    KM = np.einsum('nm,kmj->knj', K2inv, M)
    Pm = np.einsum('kni,knj->kij', M, KM); q = np.einsum('kni,kn->ki', KM, A0)
    return -np.einsum('kij,kj->ki', np.linalg.inv(Pm), q), np.linalg.inv(binv*Pm)
def mix_ci(mn, sd):
    xs = np.linspace((mn-8*sd).min(), (mn+8*sd).max(), 20001)
    cdf = sps.norm.cdf((xs[:, None]-mn[None, :])/sd[None, :]).mean(1)
    return np.interp([0.025, 0.975], cdf, xs)
gw = G.theta_w
print(f'{"k":>6}   ' + "   ".join(f'{nm}: {"RB":>6} {"raw":>6}' for nm in ["a", "b"]))
for k in [10, 40, 200, 1000]:
    rows = []
    for rep in range(20):
        sub = G.pos[np.random.default_rng(rep).choice(len(G.pos), k, replace=False)]
        mn, Sg = ab_cond(sub)
        rr = []
        for j in range(2):
            lo, hi = mix_ci(mn[:, j], np.sqrt(Sg[:, j, j]))
            e = np.quantile(sub[:, j], [0.025, 0.975])
            rr += [100*(hi-lo)/gw[j], 100*(e[1]-e[0])/gw[j]]
        rows.append(rr)
    a = np.array(rows)
    print(f'{k:>6}   ' + "   ".join(
        f'{nm}: {a[:,2*j].mean():5.1f}% {a[:,2*j+1].mean():5.1f}%' for j, nm in enumerate(["a","b"]))
        + f'   (sd over 20 subsamples: RB {a[:,0].std():.1f} vs raw {a[:,1].std():.1f})')
