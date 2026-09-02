"""
Exp 15: is the float32 failure the finite-difference stencil rather than the arithmetic?

Exp 14 showed log p_hat carries only ~0.007 nats of float32 noise, which degrades the importance
weights by a factor of 0.997 -- nothing. But the profile-mode Newton estimates curvature by central
differences, which divide by h^2, so noise of size s in log p_hat becomes noise of size ~4s/h^2 in
the Hessian. At the default h = 0.02 sd that is an amplification of 2500 and the curvature estimate
is pure noise, which would send the proposal somewhere arbitrary and collapse the effective sample
size exactly as observed. If that is the mechanism the cure is a larger stencil, not more precision.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build
from profiled import ProfiledPosterior

z = np.load("../investigation5/ref5_fn.npz"); rm = z["mean"]; rs = np.sqrt(np.diag(z["cov"]))
hm = z["half_mean"]
floor = np.abs((hm[0][:3] - hm[1][:3]) / rs[:3]).max()
print(f'{"dtype":>9} {"fd_rel":>7} {"ESS":>7} {"khat":>7} {"reliable":>9} {"max|err|":>10} {"sec":>6}')
for dt in (jnp.float64, jnp.float32):
    for fd in (0.02, 0.1, 0.3, 0.5):
        m, ds = build("fn", dtype=dt)
        m.map_solve(verbose=False, tol=1e-9, max_iter=300)
        p = m.p
        t0 = time.time()
        pp = ProfiledPosterior(m, n_nodes=512, seed=0, fd_rel=fd).build(verbose=False)
        el = time.time() - t0
        rel = (pp.ess / pp.n_nodes >= 0.10 and np.isfinite(pp.khat) and pp.khat < 0.7
               and getattr(pp, "mode_ok", True))
        e = np.abs((pp.theta_mean - rm[:p]) / rs[:p]).max()
        print(f'{dt.__name__:>9} {fd:>7} {pp.ess/pp.n_nodes:>7.1%} {pp.khat:>7.2f} '
              f'{str(rel):>9} {e:>10.4f} {el:>6.1f}')
    print()
print(f'reference half-vs-half floor: {floor:.4f}')
