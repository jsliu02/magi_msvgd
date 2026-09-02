"""
Exp 29: do the subspace size m and the antithetic pair count N have to be retuned per problem?

Both were fixed at m=12, N=1024 for every result reported so far, and the sensitivity sweep that
justified them was run at the baseline setting only. If the right m tracks the problem, that is a
tuning burden and should be said; if it saturates at a value set by the geometry rather than by
the severity of the non-Gaussianity, it is a constant.

Two things are measured. How many directions the SCREEN says are non-quadratic, per setting --
this is the intrinsic quantity m has to cover, and it is computable without any reference. And how
the gate statistic and the theta error respond to m and N at each setting.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
d, P = H.DIM, 3
REF = {"baseline": "ref4_baseline.npz", "half": "ref5_half.npz",
       "noisy": "ref5_noisy.npz", "quarter": "ref5_quarter.npz"}

# ---------------------------------------------------------------- how many directions are bad?
print("intrinsic width of the non-quadratic subspace, from the screen alone (no reference)")
print(f'{"setting":>9} {"q_max":>9} ' + " ".join(f'{f"q>{t}":>7}' for t in (1.0, 0.1, 0.01)) +
      f' {"#dirs for 90% of sum(q)":>24}')
models, screens = {}, {}
for name in SETTINGS:
    m = build(*SETTINGS[name], dtype=jnp.float64)
    post = m.fit(n_pairs=256, verbose=False, tol=1e-8, max_iter=200)
    models[name] = m
    # recompute the screen (fit does not retain it)
    Hs = np.asarray(m.hessian(post.mu_map), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); sd = 1.0 / np.sqrt(ev)
    lp0 = m._logp_many(post.mu_map[None, :])[0]
    P4 = np.concatenate([post.mu_map[None, :] + s * (sd[:, None] * V.T) for s in (-2, -1, 1, 2)])
    q = np.abs(-(m._logp_many(P4) - lp0).reshape(4, d) /
               np.array([2.0, .5, .5, 2.0])[:, None] - 1).mean(0)
    screens[name] = q
    cs = np.cumsum(np.sort(q)[::-1]) / q.sum()
    print(f'{name:>9} {q.max():>9.2f} ' + " ".join(f'{int((q > t).sum()):>7}' for t in (1.0, 0.1, 0.01))
          + f' {int(np.searchsorted(cs, 0.9)) + 1:>24}')

# ---------------------------------------------------------------- response to m and N
print(f'\nresponse of the gate statistic (rho) and largest |theta error| to m and N')
for name in ["baseline", "half", "noisy"]:
    z = np.load(REF[name]); rm, rs = z["mean"], np.sqrt(np.diag(z["cov"]))
    the = lambda v: float(np.abs((v[:P] - rm[:P]) / rs[:P]).max())
    m = models[name]
    print(f'  {name}:')
    print(f'{"":>6} {"m":>4} {"N":>6} {"rho":>8} {"|theta| mu3":>12} {"|theta| mid":>12}')
    for mm in (4, 8, 12, 16, 24, 32):
        post = m.fit(subspace=mm, n_pairs=1024, verbose=False, tol=1e-8, max_iter=200)
        print(f'{"":>6} {mm:>4} {1024:>6} {post.certificates["ratio"]:>8.4f} '
              f'{the(post.mu3):>12.4f} {the(post.mu_mid):>12.4f}')
    for nn in (64, 128, 256, 512, 1024, 2048):
        post = m.fit(subspace=12, n_pairs=nn, verbose=False, tol=1e-8, max_iter=200)
        print(f'{"":>6} {12:>4} {nn:>6} {post.certificates["ratio"]:>8.4f} '
              f'{the(post.mu3):>12.4f} {the(post.mu_mid):>12.4f}')
    print()
