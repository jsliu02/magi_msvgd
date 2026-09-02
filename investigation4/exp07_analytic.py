"""
Exp 7: score the deterministic approximation ANALYTICALLY, and find the right screening depth.

Exp 6's verdict was blurred by Monte Carlo noise: at K=800 the sampled varwtd has a standard error
comparable to the 3-5% effect being measured, so it reported the scale correction as a slight LOSS
while exp 5 showed each corrected direction moving decisively toward truth. The approximation is a
Gaussian in closed form, so compare its (mean, covariance) to gold's directly and drop the noise.

    mean       bias  = ||Sigma_g^(-1/2) (mu_q - mu_g)|| / sqrt(d)          (posterior sd per dim)
    spread     trace ratio tr(Sigma_q)/tr(Sigma_g)                          (the varwtd analogue)
    shape      Forstner  ||log(Sigma_g^(-1/2) Sigma_q Sigma_g^(-1/2))||_F / sqrt(d)  (affine invariant)
    combined   KL(q || gold Gaussian) in nats

The mean and the covariance are corrected by different mechanisms and are tested separately: the
third-order Laplace term is a single global d-dimensional formula, while the profile correction is
inherently per-direction and only pays for itself where the screen fires.
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
d, MMAX = H.DIM, 40
gold = np.asarray(G.pos, np.float64)
mu_g, Sig_g = gold.mean(0), np.cov(gold, rowvar=False)
eg, Vg = np.linalg.eigh(Sig_g)
Wg = Vg / np.sqrt(eg)                                     # Sigma_g^(-1/2) = Wg Vg^T
half = lambda A: Wg.T @ A @ Wg
sgn, ldg = np.linalg.slogdet(Sig_g)

def scorecov(mu, S):
    M = half(S); w = np.linalg.eigvalsh(M)
    dl = Wg.T @ (mu - mu_g)
    return dict(bias=float(np.linalg.norm(dl) / np.sqrt(d)),
                trace=float(np.trace(S) / np.trace(Sig_g)),
                forst=float(np.linalg.norm(np.log(np.maximum(w, 1e-12))) / np.sqrt(d)),
                kl=float(0.5 * (w.sum() - d + dl @ dl - np.linalg.slogdet(S)[1] + ldg)))

# ---------------------------------------------------------------------- third-order mean
gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
Sj = jnp.asarray(Sig)
t0 = time.time()
mu3 = np.asarray(x_map) - 0.5 * np.asarray(
    Sj @ jax.grad(lambda z: jnp.sum(Sj * pr._hess(z)))(jnp.asarray(x_map)))
t_3rd = time.time() - t0
np.save("mu3_baseline.npy", mu3)

# ---------------------------------------------------------------------- screen + profiles
lp = jax.jit(lambda P: jax.vmap(lambda z: m.logdensity(z, m.data))(P))
ev, V = np.linalg.eigh(Hs); sd = 1.0 / np.sqrt(ev)
mu0 = np.asarray(x_map); lp0 = float(lp(jnp.asarray(mu0[None, :]))[0])
t0 = time.time()
P = np.concatenate([mu0[None, :] + t * (sd[:, None] * V.T) for t in (-2, -1, 1, 2)])
U = -(np.asarray(lp(jnp.asarray(P))) - lp0).reshape(4, d)
qscr = np.abs(U / np.array([2.0, 0.5, 0.5, 2.0])[:, None] - 1).mean(0)
t_screen = time.time() - t0

order = np.argsort(-qscr)[:MMAX]
t0 = time.time(); vprof = np.zeros(MMAX); mprof = np.zeros(MMAX)
for i, j in enumerate(order):
    zs = np.linspace(-4.5 * sd[j], 4.5 * sd[j], 17)
    Up, ldt, _ = pr.profile(V[:, j], zs, x_map)
    mprof[i], vprof[i] = moments(zs, -Up - ldt)
t_prof = (time.time() - t0) / MMAX

print(f'{"mean":>22} {"covariance":>26} {"bias":>7} {"trace":>7} {"forstner":>9} {"KL":>9} {"sec":>7}')
def row(lbl_m, lbl_c, mu, S, sec):
    r = scorecov(mu, S)
    print(f'{lbl_m:>22} {lbl_c:>26} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} '
          f'{r["kl"]:>9.1f} {sec:>7.2f}')
    return r

t_base = 0.05 + 0.30
rows = {}
for mlab, mu, tm in [("MAP", mu0, 0.0), ("+ third order", mu3, t_3rd)]:
    rows[(mlab, 0)] = row(mlab, "H^-1 (Laplace)", mu, Sig, t_base + tm)
for M in (3, 5, 10, 20, 40):
    S = Sig + V[:, order[:M]] @ np.diag(vprof[:M] - sd[order[:M]] ** 2) @ V[:, order[:M]].T
    rows[("+ third order", M)] = row("+ third order", f"+ profile scale m={M}", mu3, S,
                                     t_base + t_3rd + t_screen + M * t_prof)
print(f'{"gold (self)":>22} {"":>26} {0.0:>7.4f} {1.0:>7.4f} {0.0:>9.4f} {0.0:>9.1f}')

Sfull = Sig + V[:, order] @ np.diag(vprof - sd[order] ** 2) @ V[:, order].T
row("+ profile mean", "+ profile scale m=40", mu0 + V[:, order] @ mprof, Sfull,
    t_base + t_screen + MMAX * t_prof)
row("+ 3rd & profile mean", "+ profile scale m=40",
    mu3 + V[:, order] @ ((mu0 + V[:, order] @ mprof - mu3) @ V[:, order]), Sfull, 0)
print(f'\nper-profile cost {t_prof:.2f}s; screen {t_screen:.2f}s for all {d}; third order {t_3rd:.2f}s')
np.savez("analytic_baseline.npz", mu3=mu3, order=order, vprof=vprof, mprof=mprof, qscr=qscr, ev=ev, V=V)
