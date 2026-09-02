"""
Exp 24: does whitened_ula assume the posterior is Gaussian?

The metric is a linear change of variables and MALA's accept/reject is exact, so in principle
the answer is no -- only efficiency should suffer. Two direct tests of that claim.

(a) A target with strongly non-Gaussian SHAPE but Gaussian TAILS, so the transience of the
    previous experiment cannot confound the result: an independent skew-normal, density
    2*phi(x)*Phi(alpha*x). Its mode and Hessian are perfectly well behaved and the Laplace
    approximation is a symmetric Gaussian, so it has zero skew by construction. If the sampler
    recovers the true skewness and variance, it plainly is not assuming Gaussianity.

(b) The real MAGI posterior, whose theta_b marginal has a measured skew of -0.219 (23 standard
    errors from zero). A Gaussian-returning method cannot reproduce that; a sampler should.
    Skewness needs many draws to estimate, so k is raised and several seeds averaged.
"""
import numpy as np, jax, jax.numpy as jnp, sys
from scipy import stats as sps
sys.path.insert(0, "/home/jamie/storage-1/github-repos/msvgd/msvgd")
from msvgd import MSVGD
import harness as H

# ------------------------------------------------------------------ (a) skew-normal
ALPHA, D = 5.0, 16
delta = ALPHA / np.sqrt(1 + ALPHA ** 2)
true_mean = delta * np.sqrt(2 / np.pi)
true_var = 1 - 2 * delta ** 2 / np.pi
true_skew = (4 - np.pi) / 2 * true_mean ** 3 / true_var ** 1.5
def skewnormal(x):
    return jnp.sum(jax.scipy.stats.norm.logpdf(x) + jax.scipy.stats.norm.logcdf(ALPHA * x))

print(f"(a) skew-normal, alpha={ALPHA}, d={D}:  true mean {true_mean:.4f}, var {true_var:.4f}, "
      f"skew {true_skew:.4f}")
print(f'{"sampler":>34} {"mean":>9} {"var":>9} {"skew":>9} {"K-S vs exact":>13}')
s = MSVGD(skewnormal)
xm = None
for tag, kw in [("whitened MALA", dict(metropolis=True)),
                ("whitened ULA", dict(metropolis=False, step_size=0.02))]:
    P = np.asarray(s.whitened_ula(np.zeros(D), k=20000, n_steps=4000, random_seed=0,
                                  monitor_convergence=-1, x_map=xm, **kw)).ravel()
    xm = s.x_map
    ks = sps.kstest(P[:40000], lambda q: sps.skewnorm.cdf(q, ALPHA)).statistic
    print(f'{tag:>34} {P.mean():>9.4f} {P.var():>9.4f} {sps.skew(P):>9.4f} {ks:>13.4f}')
lap = np.asarray(s.x_map)[0] + np.random.default_rng(0).standard_normal(40000) * \
      float(np.sqrt(1 / s.laplace_evals[0]))
print(f'{"Laplace approximation (control)":>34} {lap.mean():>9.4f} {lap.var():>9.4f} '
      f'{sps.skew(lap):>9.4f} {sps.kstest(lap, lambda q: sps.skewnorm.cdf(q, ALPHA)).statistic:>13.4f}')

# ------------------------------------------------------------------ (b) real MAGI posterior
print(f"\n(b) MAGI posterior: does the sampler reproduce the true theta skew?")
G = H.Gold()
true_sk = [float(sps.skew(G.pos[:, j])) for j in range(3)]
m = H.build_magi(); xm = None; draws = []
for seed in range(4):
    P = m.whitened_ula(m.particles_init, k=4000, n_steps=2000, random_seed=seed,
                       monitor_convergence=-1, metropolis=True, x_map=xm)
    xm = m.x_map; draws.append(np.asarray(P, np.float64))
allP = np.concatenate(draws)
se = np.sqrt(6 / len(allP))
print(f'{"":>34} {"skew a":>9} {"skew b":>9} {"skew c":>9}')
print(f'{"NUTS gold standard (64k draws)":>34} ' + " ".join(f'{v:>9.3f}' for v in true_sk))
print(f'{f"whitened MALA ({len(allP)} draws)":>34} ' +
      " ".join(f'{float(sps.skew(allP[:, j])):>9.3f}' for j in range(3)))
rng = np.random.default_rng(0)
gh = allP.mean(0)[None, :] + rng.standard_normal((len(allP), H.DIM)) @ \
     ((np.load("laplace_cache.npz")["evecs"] /
       np.sqrt(np.maximum(np.load("laplace_cache.npz")["evals"], 1e-8))) @
      np.load("laplace_cache.npz")["evecs"].T).T
print(f'{"gauss-hybrid (Gaussian by design)":>34} ' +
      " ".join(f'{float(sps.skew(gh[:, j])):>9.3f}' for j in range(3)))
print(f'{"Monte Carlo std error":>34} ' + " ".join(f'{se:>9.3f}' for _ in range(3)))
