"""
Exp 4: is the Laplace error concentrated in a few identifiable directions?

Exp 3 found the potential is exactly quadratic along stiff directions and violently non-quadratic
along the softest one. If the non-Gaussianity lives in a low-dimensional subspace that can be
FOUND WITHOUT A REFERENCE, the fix is a low-rank non-Gaussian enrichment of the Laplace
approximation: exact Gaussian on the complement, something better on the few bad directions.

Screening statistic (reference-free): along eigenvector v_j of H, with unit = 1 posterior sd,

    q_j = mean over t in {-2,-1,1,2} of  [ U(mu + t sd_j v_j) / (t^2/2) - 1 ]

which is 0 iff the slice is exactly quadratic. Compared here against the truth: the gold marginal
variance along v_j relative to the Laplace prediction 1/eig_j, and the gold marginal's skew and
excess kurtosis. If q_j ranks the directions the way the gold errors do, the screen works.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from setup4 import cache
from scipy.stats import skew, kurtosis

G = H.Gold()
m, x_map, Hs, Sig, L = cache("baseline")
d = H.DIM
lp = jax.jit(lambda P: jax.vmap(lambda z: m.logdensity(z, m.data))(P))
mu = np.asarray(x_map); lp0 = float(lp(jnp.asarray(mu[None, :]))[0])
ev, V = np.linalg.eigh(Hs)                       # ascending: index 0 = softest
gold = np.asarray(G.pos, np.float64)

# ---------------------------------------------------------------- screen every direction at once
sd = 1.0 / np.sqrt(ev)
P = np.concatenate([mu[None, :] + t * (sd[:, None] * V.T) for t in (-2, -1, 1, 2)])
U = -(np.asarray(lp(jnp.asarray(P))) - lp0).reshape(4, d)
qscr = np.abs(U / np.array([2.0, 0.5, 0.5, 2.0])[:, None] - 1).mean(0)

proj = (gold - gold.mean(0)) @ V
vratio = proj.var(0) * ev                        # gold variance / Laplace variance
share = (1.0 / ev) / np.sum(1.0 / ev)            # Laplace's share of total variance

print(f'{"rank by screen":>15} {"eig":>9} {"var share":>10} {"screen q":>9} '
      f'{"gold var/Laplace":>17} {"skew":>7} {"exkurt":>8}')
for r, j in enumerate(np.argsort(-qscr)[:8]):
    print(f'{r:>15} {ev[j]:>9.3f} {share[j]:>9.2%} {qscr[j]:>9.3f} {vratio[j]:>17.3f} '
          f'{skew(proj[:, j]):>7.2f} {kurtosis(proj[:, j]):>8.2f}')
print(f'{"--- median over all 325 ---":>27} {np.median(qscr):>21.3f} {np.median(vratio):>17.3f} '
      f'{np.median(np.abs(skew(proj))):>7.2f} {np.median(kurtosis(proj)):>8.2f}')

k = np.argsort(-qscr)
cum = np.cumsum(share[k] * np.abs(vratio[k] - 1))
tot = np.sum(share * np.abs(vratio - 1))
print(f'\nvariance-weighted Laplace error captured by the top-m screened directions:')
print("   " + "  ".join(f'm={mm}: {cum[mm-1]/tot:.1%}' for mm in (1, 2, 3, 5, 10, 20)))
print(f'\nspearman(screen q, |gold var/Laplace - 1|) over all {d} directions = '
      f'{__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(qscr, np.abs(vratio-1))[0]:.3f}')
np.savez("dirs_baseline.npz", ev=ev, V=V, qscr=qscr, vratio=vratio)
