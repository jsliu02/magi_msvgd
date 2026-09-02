"""
Exp 6: the screen-then-profile approximation, scored end to end.

Pipeline, fully deterministic apart from the final draw:
    1. Gauss-Newton MAP                                       (0.05 s)
    2. exact Hessian H, eigendecomposition                    (~0.3 s)
    3. slice-curvature screen q_j over all d directions       (4 log-density evals, batched)
    4. Laplace marginal by bordered GN profile on the top m   (~0.3 s each)
    5. rebuild: mean from the profiles, variance rescaled along the screened directions

Three mean estimates are compared, since steps 4 and 5 offer two independent ones:
    MAP            no correction
    third-order    -0.5 Sigma grad tr(Sigma H)            (a global d-dimensional formula)
    profile        per-direction mean of the Laplace marginal   (m one-dimensional integrals)
and two shapes: Gaussian with corrected scale, and the profile densities' actual shape sampled by
inverse CDF. Gold reports near-zero skew and kurtosis along these directions, so the shape is
expected to be a no-op; it is included to confirm the scale is the whole story.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from setup4 import cache
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler, moments

G = H.Gold()
m, x_map, Hs, Sig, L = cache("baseline")
d, M = H.DIM, 10
gold = np.asarray(G.pos, np.float64)
lp = jax.jit(lambda P: jax.vmap(lambda z: m.logdensity(z, m.data))(P))
bias = lambda mu: float(np.sqrt((G.whiten(np.asarray(mu)[None, :]) ** 2).mean()))

t0 = time.time()
ev, V = np.linalg.eigh(Hs)
sd = 1.0 / np.sqrt(ev)
mu0 = np.asarray(x_map); lp0 = float(lp(jnp.asarray(mu0[None, :]))[0])
P = np.concatenate([mu0[None, :] + t * (sd[:, None] * V.T) for t in (-2, -1, 1, 2)])
U = -(np.asarray(lp(jnp.asarray(P))) - lp0).reshape(4, d)
qscr = np.abs(U / np.array([2.0, 0.5, 0.5, 2.0])[:, None] - 1).mean(0)
t_screen = time.time() - t0

pr = Profiler(GaussNewtonMAP(m), m)
top = np.argsort(-qscr)[:M]
t0 = time.time()
mprof, vprof, grids = np.zeros(M), np.zeros(M), []
for i, j in enumerate(top):
    zs = np.linspace(-4.5 * sd[j], 4.5 * sd[j], 17)
    Up, ldt, _ = pr.profile(V[:, j], zs, x_map)
    mprof[i], vprof[i] = moments(zs, -Up - ldt)
    grids.append((zs, -Up - ldt))
t_prof = time.time() - t0

# ---------------------------------------------------------------- three means, two shapes
mu_map = mu0.copy()
mu_pro = mu0 + V[:, top] @ mprof
mu_3rd = np.load("mu3_baseline.npy") if os.path.exists("mu3_baseline.npy") else None
true_disp = (gold.mean(0) - mu0) @ V[:, top]
print(f'{"dir":>5} {"profile mean":>13} {"true mean disp":>15} {"1 sd":>8}')
for i, j in enumerate(top[:6]):
    print(f'{j:>5} {mprof[i]:>+13.4f} {true_disp[i]:>+15.4f} {sd[j]:>8.4f}')
print(f'\nbias:  MAP {bias(mu_map):.4f}   profile-mean {bias(mu_pro):.4f}'
      + (f'   third-order {bias(mu_3rd):.4f}' if mu_3rd is not None else ''))

Sh = np.linalg.cholesky(Sig + 1e-14 * np.eye(d))
rng = np.random.default_rng(0)
K = 800
def draw(mu, shape):
    Z = rng.standard_normal((K, d))
    X = mu[None, :] + Z @ Sh.T
    for i, j in enumerate(top):                       # replace the screened coordinates
        cur = (X - mu[None, :]) @ V[:, j]
        if shape == "gauss":
            new = cur * np.sqrt(vprof[i] / sd[j] ** 2)
        else:
            zs, lg = grids[i]
            w = np.exp(lg - lg.max()); cdf = np.concatenate([[0], np.cumsum((w[1:]+w[:-1])/2*np.diff(zs))])
            cdf /= cdf[-1]
            from scipy.stats import norm
            new = np.interp(norm.cdf(cur / sd[j]), cdf, zs) - mprof[i]
        X = X + np.outer(new - cur, V[:, j])
    return X

print(); print(H.HDR); print("-" * len(H.HDR))
rows = []
for tag, mu, shp in [("N(MAP, H^-1)", mu_map, None), ("profile mean, H^-1", mu_pro, None),
                     ("profile mean + scale", mu_pro, "gauss"),
                     ("profile mean + full shape", mu_pro, "cdf")]:
    X = mu[None, :] + rng.standard_normal((K, d)) @ Sh.T if shp is None else draw(mu, shp)
    r = H.evaluate(jnp.asarray(X), m, tag=tag)
    prv = ((X - G.mean) @ G.evecs).var(0) / G.evals
    r["varwtd"] = float(np.sum(prv * G.evals) / np.sum(G.evals))
    rows.append(r); H.show(r)
H.show(H.gold_row())
print(f'\n{"method":>28} {"energy":>8} {"varwtd":>8}')
for r in rows: print(f'{r["tag"]:>28} {r["energy"]:>8.4f} {r["varwtd"]:>8.4f}')
print(f'\ncost: screen {t_screen:.2f}s ({4} batched log-density evals over all {d} directions), '
      f'profiles {t_prof:.2f}s for m={M}')
np.savez("corrected_baseline.npz", top=top, mprof=mprof, vprof=vprof, qscr=qscr, ev=ev, V=V)
