"""
Exp 31: the profile correction was dismissed at baseline. Does it earn its cost where the
covariance is actually wrong?

Exp 30 found the Laplace standard deviation for theta_b is 1.70x the reference at the noisy
setting -- a credible interval 70% too wide -- while at baseline every theta sd is within 7%. The
Tierney-Kadane profile is precisely a marginal-variance corrector, and it was rejected earlier for
moving the marginal variances 4.68% -> 1.66% at baseline, which was not worth 13 s. That is a
statement about baseline, not about the method.

Here it is applied directly along the theta coordinate axes, where the answer is checkable, at the
setting where the Laplace covariance fails.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler, moments

d, P = H.DIM, 3
for name, rf in [("noisy", "ref5_noisy.npz"), ("baseline", "ref4_baseline.npz")]:
    z = np.load(rf); rs = np.sqrt(np.diag(z["cov"])); hc = z["half_cov"]
    m = build(*SETTINGS[name], dtype=jnp.float64)
    m.map_solve(verbose=False, tol=1e-9, max_iter=200)
    x0 = np.asarray(m.map_particle, np.float64)
    Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T
    pr = Profiler(GaussNewtonMAP(m), m)
    print(f'--- {name} ---')
    print(f'{"param":>8} {"Laplace sd":>11} {"profile sd":>11} {"reference sd":>13} '
          f'{"Lap/ref":>8} {"prof/ref":>9} {"sec":>6}')
    for i, c in enumerate("abc"):
        e = np.zeros(d); e[i] = 1.0
        sdl = np.sqrt(Sig[i, i])
        zs = np.linspace(-4.5 * sdl, 4.5 * sdl, 19)
        t0 = time.time()
        U, ld, _ = pr.profile(e, zs, x0)
        _, vP = moments(zs, -U - ld)
        dt = time.time() - t0
        sdp = np.sqrt(max(vP, 0))
        print(f'{"theta_" + c:>8} {sdl:>11.5f} {sdp:>11.5f} {rs[i]:>13.5f} '
              f'{sdl/rs[i]:>8.4f} {sdp/rs[i]:>9.4f} {dt:>6.2f}')
    print()
