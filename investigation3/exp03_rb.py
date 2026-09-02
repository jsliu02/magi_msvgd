"""
Exp 3: exact Rao-Blackwellization of the parameters, under a mild condition on the ODE.

CONDITION (A): the vector field f is affine in the parameters,
                   f(X, theta) = g0(X) + sum_q theta_q g_q(X).
Then the ODE residual is affine in theta, the ODE term is a quadratic form in theta, and because
MAGI puts a flat prior on theta the conditional theta | X is EXACTLY Gaussian, in closed form.
No approximation, no sampling error in the theta directions.

Condition (A) holds outright for mass-action kinetics, linear compartment models and
Lotka-Volterra. FitzHugh-Nagumo satisfies it for (a, b) given c:

    r_2 = -(1/c)(V - a + b R) - mudot_2 - (m . diff)_2  =  A0 + M [a, b],
    M = [ (1/c) 1 , -(1/c) R ],

so with P = M^T K2^-1 M and q = M^T K2^-1 A0,

    (a, b) | X, c  ~  N( -P^-1 q , (beta_inv * P)^-1 ).

This turns the reported theta marginals from a particle histogram into an exact mixture of
Gaussians, removing all Monte Carlo error in exactly the quantities a MAGI user reports. It is
also a diagnostic: applied to gold-standard draws it must reproduce the reference marginals.
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy import stats as sps
import harness as H

G = H.Gold()
m = H.build_magi(dtype=jnp.float64)
n, D, P = m.n, m.D, m.p
K2inv = np.asarray(m.K_invs, np.float64)[1]
mu = np.asarray(m.mu, np.float64); mudot = np.asarray(m.mu_dot, np.float64)
ms = np.asarray(m.ms, np.float64); binv = float(m.beta_inv)

def ab_conditional(Xall):
    """(k, dim) -> per-particle exact conditional mean (k,2) and covariance (k,2,2) of (a,b)."""
    th = Xall[:, :P]; Xs = Xall[:, P:P + n * D].reshape(-1, n, D)
    V, R = Xs[:, :, 0], Xs[:, :, 1]; c = th[:, 2]
    diff = Xs - mu[None]
    mdiff2 = np.einsum('nm,kmd->kn', ms[1], diff[:, :, [1]] * 0 + diff[:, :, 1][:, :, None])[:, :]
    mdiff2 = np.einsum('nm,km->kn', ms[1], diff[:, :, 1])
    A0 = -(V / c[:, None]) - mudot[None, :, 1] - mdiff2
    Mmat = np.stack([np.ones_like(V) / c[:, None], -R / c[:, None]], axis=2)   # (k,n,2)
    KM = np.einsum('nm,kmj->knj', K2inv, Mmat)
    Pm = np.einsum('kni,knj->kij', Mmat, KM)
    q = np.einsum('kni,kn->ki', KM, A0)
    Sig = np.linalg.inv(binv * Pm)
    mean = -np.einsum('kij,kj->ki', np.linalg.inv(Pm), q)
    return mean, Sig

def mixture_ci(mean, sd, lo=0.025, hi=0.975, grid=20001):
    """quantiles of an equally-weighted Gaussian mixture, by inverting its CDF on a grid"""
    a = (mean - 8 * sd).min(); b = (mean + 8 * sd).max()
    xs = np.linspace(a, b, grid)
    cdf = sps.norm.cdf((xs[:, None] - mean[None, :]) / sd[None, :]).mean(1)
    return np.interp([lo, 0.5, hi], cdf, xs)

rng = np.random.default_rng(0)
gold = G.pos[rng.choice(len(G.pos), 4000, replace=False)]
print("validation: apply the exact conditional to gold-standard (X, c) draws")
mean, Sig = ab_conditional(gold)
names = ["a", "b"]
print(f'{"":>34} {"2.5%":>9} {"50%":>9} {"97.5%":>9} {"width":>9}')
for j in range(2):
    lo, md, hi = mixture_ci(mean[:, j], np.sqrt(Sig[:, j, j]))
    gl, gm, gh = np.quantile(G.pos[:, j], [0.025, 0.5, 0.975])
    print(f'{f"  {names[j]}: Rao-Blackwellized":>34} {lo:>9.4f} {md:>9.4f} {hi:>9.4f} {hi-lo:>9.4f}')
    print(f'{f"  {names[j]}: NUTS reference":>34} {gl:>9.4f} {gm:>9.4f} {gh:>9.4f} {gh-gl:>9.4f}')

print("\napplied to approximate ensembles (only their X and c are used):")
z = np.load("laplace_cache.npz")
evc = np.maximum(z["evals"], 1e-8 * z["evals"].max())
Sh = (z["evecs"] / np.sqrt(evc)) @ z["evecs"].T
ens = {"N(MAP, H^-1)": z["x_map"][None, :] + rng.standard_normal((4000, H.DIM)) @ Sh.T}
try:
    mc = np.load("mean_correction.npy")
    ens["N(MAP+corr, H^-1)"] = mc[None, :] + rng.standard_normal((4000, H.DIM)) @ Sh.T
except FileNotFoundError:
    pass
gw = G.theta_w
for tag, E in ens.items():
    mean, Sig = ab_conditional(E)
    row = []
    for j in range(2):
        lo, md, hi = mixture_ci(mean[:, j], np.sqrt(Sig[:, j, j]))
        emp = np.quantile(E[:, j], [0.025, 0.975])
        row.append(f"{names[j]}: RB {100*(hi-lo)/gw[j]:5.1f}%  raw {100*(emp[1]-emp[0])/gw[j]:5.1f}%")
    print(f'  {tag:>22}   ' + "   ".join(row) + "   (100% = NUTS width)")
