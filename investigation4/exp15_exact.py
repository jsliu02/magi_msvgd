"""
Exp 15: an ODE class for which MAGI's posterior needs no approximation at all.

CONDITION (A): f(., theta) is affine in the state X for every theta. Linear compartment models,
linear pharmacokinetics and every constant-coefficient system satisfy it; FitzHugh-Nagumo with its
cubic term switched off is the test case here. Note this is much weaker than requiring f affine in
(X, theta) JOINTLY -- theta may still multiply states, as it does here through c(V+R), which is
why exp 14 measured L2 = 6.76 rather than 0 at alpha = 0.

Under (A) the residual is affine in X at fixed theta, R = R_0(theta) + A(theta) X, so

    U(theta, X) = U(theta, X*(theta)) + 0.5 (X - X*)^T A^T A (X - X*),   X* = -(A^T A)^-1 A^T R_0,

which is EXACTLY quadratic in X. Hence p(X | theta) is exactly Gaussian with covariance
(A^T A)^-1, and integrating X out is exact rather than approximate:

    log p(theta) = -U(theta, X*(theta)) - 0.5 log det(A(theta)^T A(theta)) + const.

So the 325-dimensional problem is exactly a 3-dimensional one. It is solved here by Gauss-Hermite
quadrature in Laplace-whitened theta coordinates -- no MCMC, no Gaussian assumption on theta, and
the only error is quadrature error, which is controlled by adding nodes and watching the answer
stop moving. The claim is falsifiable in two ways, both checked: the X-Hessian must be
independent of X to machine precision, and the resulting theta moments must match NUTS.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time, itertools
jax.config.update("jax_enable_x64", True)
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

m = make(0.0)
gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
gn.solve(verbose=False, tol=1e-9, max_iter=200)
x0 = np.asarray(gn.map_particle, np.float64)
n, D, p, nD = gn.n, gn.D, gn.p, gn.nD

# ------------------------------------------------- falsification 1: is the X-Hessian X-independent?
H1 = np.asarray(pr._hess(jnp.asarray(x0)))[p:, p:]
xp = x0.copy(); xp[p:] += 0.7 * np.random.default_rng(0).standard_normal(nD)
H2 = np.asarray(pr._hess(jnp.asarray(xp)))[p:, p:]
print(f'condition (A) check: ||H_XX(X1) - H_XX(X2)|| / ||H_XX|| = '
      f'{np.linalg.norm(H1-H2)/np.linalg.norm(H1):.2e}   (0 => p(X|theta) exactly Gaussian)')
mF = make(1.0); gnF = GaussNewtonMAP(mF); prF = Profiler(gnF, mF)
gnF.solve(verbose=False, tol=1e-9, max_iter=200)
xF = np.asarray(gnF.map_particle, np.float64)
G1 = np.asarray(prF._hess(jnp.asarray(xF)))[p:, p:]
xq = xF.copy(); xq[p:] += 0.7 * np.random.default_rng(0).standard_normal(nD)
G2 = np.asarray(prF._hess(jnp.asarray(xq)))[p:, p:]
print(f'   same check for the real cubic ODE (alpha=1):        '
      f'{np.linalg.norm(G1-G2)/np.linalg.norm(G1):.2e}   (nonzero => only approximate)\n')

# ------------------------------------------------- exact theta marginal by quadrature
@jax.jit
def logp_theta(th):
    """-U(theta, X*(theta)) - 0.5 logdet(A^T A); exact under condition (A)."""
    X = jnp.zeros((n, D), x0.dtype)
    A = gn.jacobian(th, X, m.sigmas)[:, p:]
    r0 = gn.residual(th, X, m.sigmas)
    AtA = A.T @ A
    cf = jax.scipy.linalg.cho_factor(AtA)
    Xs = jax.scipy.linalg.cho_solve(cf, -(A.T @ r0))
    U = 0.5 * jnp.sum((r0 + A @ Xs) ** 2)
    return -U - jnp.sum(jnp.log(jnp.diag(cf[0]))), Xs

Hs = np.asarray(pr._hess(jnp.asarray(x0))); Hs = 0.5 * (Hs + Hs.T)
Sig = np.linalg.inv(Hs)
Sth = Sig[:p, :p]                                    # Laplace theta covariance (marginal)
Lth = np.linalg.cholesky(Sth)
print(f'{"GH nodes":>9} {"theta means":>34} {"theta sds":>34} {"logZ":>10} {"sec":>6}')
prev = None
for nq in (7, 9, 11, 13):
    xg, wg = np.polynomial.hermite_e.hermegauss(nq)
    wg = wg / np.sqrt(2 * np.pi) ** 0 / wg.sum() * wg.sum()
    t0 = time.time()
    grid = np.array(list(itertools.product(xg, repeat=p)))
    wts = np.prod(np.array(list(itertools.product(wg, repeat=p))), axis=1)
    TH = x0[:p][None, :] + grid @ Lth.T
    lv = np.array([float(logp_theta(jnp.asarray(t))[0]) for t in TH])
    lw = lv + np.log(wts) + 0.5 * np.sum(grid ** 2, axis=1)     # undo the GH Gaussian weight
    lw -= lw.max(); W = np.exp(lw); W /= W.sum()
    mu_t = W @ TH
    sd_t = np.sqrt(W @ (TH - mu_t) ** 2)
    dt = time.time() - t0
    print(f'{nq:>9} {str(np.round(mu_t, 6)):>34} {str(np.round(sd_t, 6)):>34} '
          f'{np.log(np.sum(np.exp(lw))):>10.4f} {dt:>6.1f}')
    prev = (mu_t, sd_t)
np.savez("exact_alpha0.npz", mu_t=prev[0], sd_t=prev[1], x0=x0, Sth=Sth)
print(f'\nLaplace theta: means {np.round(x0[:p],6)} sds {np.round(np.sqrt(np.diag(Sth)),6)}')
