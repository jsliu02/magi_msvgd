"""
Exp 33: what the profile correction actually replaces, shown on theta_b at the noisy setting.

Four curves for the 1-d marginal of z = theta_b - theta_b*, each a candidate for -log p(z):
  quadratic  0.5 z^2 / Sigma_bb            what N(x*, H^-1) implies
  slice      U(x* + z e_b) - U(x*)         move theta_b, hold all 324 other coordinates FIXED
  profile    U_prof(z) - U_prof(0)         move theta_b, let the other 324 RELAX to their optimum
  +logdet    profile + 0.5 log det H_perp  also account for the complement's width changing with z
The last is the Tierney-Kadane Laplace marginal and is what the correction uses.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler

d = H.DIM
z = np.load("ref5_noisy.npz")
m = build(*SETTINGS["noisy"], dtype=jnp.float64)
m.map_solve(verbose=False, tol=1e-9, max_iter=200)
x0 = np.asarray(m.map_particle, np.float64)
Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T
pr = Profiler(GaussNewtonMAP(m), m)

i = 1                                            # theta_b
e = np.zeros(d); e[i] = 1.0
sd_lap, sd_ref = np.sqrt(Sig[i, i]), np.sqrt(z["cov"][i, i])
zs = np.linspace(-2.2 * sd_ref, 2.2 * sd_ref, 13)
lp0 = float(m.logdensity(jnp.asarray(x0), m.data))

U_prof, LD, xs = pr.profile(e, zs, x0)
U_prof = U_prof - float(np.interp(0.0, zs, U_prof))
LDc = LD - float(np.interp(0.0, zs, LD))
sl = np.array([-(float(m.logdensity(jnp.asarray(x0 + t * e), m.data)) - lp0) for t in zs])

def sd_of(zz, negU):
    w = np.exp(-(negU - negU.min())); Z = np.trapezoid(w, zz)
    mu = np.trapezoid(w * zz, zz) / Z
    return np.sqrt(np.trapezoid(w * (zz - mu) ** 2, zz) / Z)

quad = 0.5 * zs ** 2 / Sig[i, i]
print(f'theta_b at the noisy setting.  1 reference sd = {sd_ref:.5f}\n')
print(f'{"z / ref sd":>11} {"quadratic":>10} {"slice":>10} {"profile":>10} {"+logdet":>10}')
for k, t in enumerate(zs):
    if k % 2 == 0:
        print(f'{t/sd_ref:>11.2f} {quad[k]:>10.3f} {sl[k]:>10.3f} {U_prof[k]:>10.3f} '
              f'{U_prof[k]+LDc[k]:>10.3f}')
print(f'\n{"curve":>28} {"implied sd":>11} {"/ reference":>12}')
for lbl, c in [("quadratic  (= Laplace H^-1)", quad), ("slice (others held fixed)", sl),
               ("profile (others relax)", U_prof), ("profile + logdet (used)", U_prof + LDc)]:
    s = sd_of(zs, c)
    print(f'{lbl:>28} {s:>11.5f} {s/sd_ref:>12.4f}')
print(f'{"reference (NUTS)":>28} {sd_ref:>11.5f} {1.0:>12.4f}')
print(f'\nhow far the other 324 coordinates move when theta_b is displaced by 1 reference sd:')
j = int(np.argmin(np.abs(zs - sd_ref)))
dx = xs[j] - x0
print(f'  ||shift||_H / sqrt(d) = {np.sqrt(dx @ Hs @ dx / d):.4f} posterior sd per coordinate')
print(f'  of which theta_b itself accounts for {abs(dx[i])/np.sqrt(dx@dx):.1%} of the euclidean norm')
