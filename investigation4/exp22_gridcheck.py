"""
Exp 22: cross-check the exact theta-marginal route with a structurally different quadrature rule.

Exp 15 showed Gauss-Hermite stable from 7 to 13 nodes, but all four are the same family of rules
against the same Gaussian weight, so their agreement partly reflects a shared blind spot. A uniform
tensor grid over a wide box shares nothing with them except the integrand, so agreement between the
two is real evidence that the theta moments are converged rather than jointly mis-weighted.

Both consume the same closed-form marginal, which is exact under condition (A):
    log p(theta) = -U(theta, X*(theta)) - 0.5 log det(A(theta)^T A(theta)) + const.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time, itertools
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from magi import MAGI
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

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

m = make(0.0); gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
gn.solve(verbose=False, tol=1e-9, max_iter=200)
x0 = np.asarray(gn.map_particle, np.float64)
n, D, p = gn.n, gn.D, gn.p

@jax.jit
def lpt(th):
    X = jnp.zeros((n, D), x0.dtype)
    A = gn.jacobian(th, X, m.sigmas)[:, p:]
    r0 = gn.residual(th, X, m.sigmas)
    cf = jax.scipy.linalg.cho_factor(A.T @ A)
    Xs = jax.scipy.linalg.cho_solve(cf, -(A.T @ r0))
    return -0.5 * jnp.sum((r0 + A @ Xs) ** 2) - jnp.sum(jnp.log(jnp.diag(cf[0])))
lptv = jax.jit(jax.vmap(lpt))

Hs = np.asarray(pr._hess(jnp.asarray(x0))); Hs = 0.5 * (Hs + Hs.T)
Sth = np.linalg.inv(Hs)[:p, :p]; Lth = np.linalg.cholesky(Sth)

def moments(TH, logw):
    lw = logw - logw.max(); W = np.exp(lw); W /= W.sum()
    mu = W @ TH
    return mu, np.sqrt(W @ (TH - mu) ** 2), float(W.max())

print(f'{"rule":>28} {"theta means":>32} {"theta sds":>32} {"max wt":>8} {"sec":>6}')
for nq in (9, 13):
    xg, wg = np.polynomial.hermite_e.hermegauss(nq)
    grid = np.array(list(itertools.product(xg, repeat=p)))
    lw = np.log(np.prod(np.array(list(itertools.product(wg, repeat=p))), 1)) + 0.5*np.sum(grid**2, 1)
    TH = x0[:p] + grid @ Lth.T
    t0 = time.time(); lv = np.asarray(lptv(jnp.asarray(TH)))
    mu, sd, mw = moments(TH, lv + lw)
    print(f'{f"Gauss-Hermite {nq}^3":>28} {str(np.round(mu,6)):>32} {str(np.round(sd,6)):>32} '
          f'{mw:>8.4f} {time.time()-t0:>6.1f}')
for ng, span in ((21, 5.0), (31, 6.0)):
    g1 = np.linspace(-span, span, ng)
    grid = np.array(list(itertools.product(g1, repeat=p)))
    TH = x0[:p] + grid @ Lth.T
    t0 = time.time(); lv = np.asarray(lptv(jnp.asarray(TH)))
    mu, sd, mw = moments(TH, lv)                      # uniform weights
    print(f'{f"uniform grid {ng}^3 +-{span:.0f}sd":>28} {str(np.round(mu,6)):>32} '
          f'{str(np.round(sd,6)):>32} {mw:>8.4f} {time.time()-t0:>6.1f}')
print(f'{"Laplace (for contrast)":>28} {str(np.round(x0[:p],6)):>32} '
      f'{str(np.round(np.sqrt(np.diag(Sth)),6)):>32}')
