"""
Exp 18: can the two mean estimates be combined, and does iterating the correction help?

The third-order mean and the VI mean are both O(Lambda)-accurate with DIFFERENT O(Lambda^2) errors,
so a linear combination mu_3 + lam (mu_3 - mu_VI) could cancel the leading residual the way
Richardson extrapolation does. The coefficient is not known a priori; the test is whether one
value works across settings, which would make it usable, or whether the optimum wanders, which
would make it curve fitting. Also tested: recomputing the Hessian and the correction AT mu_3 -- a
self-consistent Laplace step -- which costs one more Hessian and needs no tuning at all.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache, SETTINGS
from pipeline import metrics
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d = H.DIM
lams = np.linspace(-1.5, 1.5, 31)
print(f'{"setting":>9} {"bias(3rd)":>10} {"bias(VI)":>9} {"best lam":>9} {"bias@best":>10} '
      f'{"bias@lam=0.5":>13} {"self-consist":>13} {"floor":>7}')
for name in ["baseline", "half", "quarter", "noisy"]:
    f = f"ref4_{name}.npz"
    if not os.path.exists(f) or not os.path.exists(f"determ_{name}.npz"):
        print(f'{name:>9}  (pending)'); continue
    z = np.load(f); D = np.load(f"determ_{name}.npz")
    sc = metrics(z["mean"], z["cov"], d)
    mu3, muvi, mu0 = D["mu3"], D["muvi"], D["mu0"]
    b = [sc(mu3 + l * (mu3 - muvi), np.eye(d))["bias"] for l in lams]
    i = int(np.argmin(b))

    m, x_map, Hs, Sig, L = cache(name)
    pr = Profiler(GaussNewtonMAP(m), m)
    H2 = np.asarray(pr._hess(jnp.asarray(mu3)), np.float64); H2 = 0.5 * (H2 + H2.T)
    w, V = np.linalg.eigh(H2); S2 = (V / np.maximum(w, 1e-10 * w.max())) @ V.T
    S2j = jnp.asarray(S2)
    mu_sc = mu3 - 0.5 * np.asarray(
        S2j @ jax.grad(lambda u: jnp.sum(S2j * pr._hess(u)))(jnp.asarray(mu3)))
    hm, hc = z["half_mean"], z["half_cov"]
    fl = metrics(hm[1], hc[1], d)(hm[0], hc[0])["bias"]
    print(f'{name:>9} {sc(mu3, np.eye(d))["bias"]:>10.4f} {sc(muvi, np.eye(d))["bias"]:>9.4f} '
          f'{lams[i]:>9.2f} {b[i]:>10.4f} '
          f'{sc(mu3 + 0.5*(mu3-muvi), np.eye(d))["bias"]:>13.4f} '
          f'{sc(mu_sc, np.eye(d))["bias"]:>13.4f} {fl:>7.4f}')
