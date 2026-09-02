"""
Exp 3: how far out does the Gaussian approximation stay valid?

Exp 2's VI refinement exploded. Two candidate causes: Monte Carlo noise in E_q[grad log p], or a
genuine breakdown -- q putting real mass where the potential's higher-order terms dominate. They
are distinguished by scaling the radius. Write

    h(s) = Sigma * E_{delta ~ N(0, s^2 Sigma)} [ grad log p(mu + delta) ]   (antithetic pairs)

Third-order theory predicts h(s) = s^2 * Delta_3 exactly, so h(s)/s^2 is FLAT in s while the cubic
expansion holds and blows up where it does not. Antithetic pairing cancels the O(delta) noise term
exactly, so what is left is signal, not variance. The value of s at which flatness fails is a
reference-free measure of the Gaussian's radius of validity, in units of posterior sd.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from setup4 import cache

m, x_map, Hs, Sig, L = cache("baseline")
d = H.DIM
Sh = np.linalg.cholesky(Sig + 1e-14 * np.eye(d))
grad = jax.jit(lambda P: m.gradient(P, m.data))
tau = lambda v: float(np.sqrt(np.abs(np.asarray(v) @ Hs @ np.asarray(v)) / d))

npair = 512
base = np.random.default_rng(0).standard_normal((npair, d)) @ Sh.T
mu = np.asarray(x_map)

print(f'{"s":>6} {"tau(h(s)/s^2)":>14} {"max ||delta||_H/sqrt(d)":>24} {"max |log p| dev":>16}')
lp = jax.jit(lambda P: jax.vmap(lambda z: m.logdensity(z, m.data))(P))
lp0 = float(lp(jnp.asarray(mu[None, :]))[0])
for s in [0.0625, 0.125, 0.25, 0.5, 1.0, 1.5]:
    off = s * base
    P = np.concatenate([mu + off, mu - off])
    g = np.asarray(grad(jnp.asarray(P))).mean(0)
    rad = float(np.sqrt(np.einsum('ij,jk,ik->i', off, Hs, off)).max() / np.sqrt(d))
    dv = float(np.abs(np.asarray(lp(jnp.asarray(P))) - lp0).max())
    print(f'{s:>6.4f} {tau(Sig @ g) / s**2:>14.4f} {rad:>24.3f} {dv:>16.1f}')

# ---- where does the potential stop being quadratic? one-dimensional profiles along eigenvectors
ev, V = np.linalg.eigh(Hs)
print(f'\nquadratic-fit relative error of -log p along single eigendirections, at +/- t sd:')
print(f'{"direction":>22} ' + " ".join(f'{f"t={t}":>9}' for t in [1, 2, 4, 8, 16]))
for lbl, j in [("stiffest (max eig)", d - 1), ("median", d // 2), ("softest (min eig)", 0)]:
    v, row = V[:, j], []
    for t in [1, 2, 4, 8, 16]:
        z = t / np.sqrt(ev[j])
        u = -(float(lp(jnp.asarray((mu + z * v)[None, :]))[0]) - lp0)
        row.append(f'{u / (0.5 * ev[j] * z**2) - 1:>9.3f}')
    print(f'{lbl:>22} ' + " ".join(row))
print("\n(0 = exactly quadratic; the Gaussian is the quadratic model, so this is its local error)")
