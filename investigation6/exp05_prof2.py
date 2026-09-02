"""Exp 5: profiled posterior built on the profile's own mode and curvature."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build, SYSTEMS
from profiled import ProfiledPosterior
from profiled2 import ProfiledPosterior2

REF = lambda n: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "investigation5", f"ref5_{n}.npz")
for name in SYSTEMS:
    if not os.path.exists(REF(name)):
        print(f'--- {name}: reference pending ---\n'); continue
    z = np.load(REF(name)); rm = z["mean"]; rs = np.sqrt(np.maximum(np.diag(z["cov"]), 0))
    hm, hc = z["half_mean"], z["half_cov"]
    m, ds = build(name); m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    p = m.p; x0 = np.asarray(m.map_particle, np.float64)
    print(f'--- {name} (p={p}) ---')
    t0 = time.time(); v1 = ProfiledPosterior(m, n_nodes=512, seed=0).adapt(rounds=6, verbose=False)
    t1 = time.time() - t0
    t0 = time.time(); v2 = ProfiledPosterior2(m, n_nodes=512, seed=0).build(verbose=False)
    t2 = time.time() - t0
    err = lambda mu: np.abs((np.asarray(mu)[:p] - rm[:p]) / np.maximum(rs[:p], 1e-300))
    sde = lambda sd: np.abs(np.asarray(sd) / np.maximum(rs[:p], 1e-300) - 1)
    fm = np.abs((hm[0][:p] - hm[1][:p]) / np.maximum(rs[:p], 1e-300)).max()
    fs = np.abs(np.sqrt(np.diag(hc[0])[:p] / np.maximum(np.diag(hc[1])[:p], 1e-300)) - 1).max()
    print(f'    {"method":>26} {"max|err|":>10} {"max|sd err|":>12} {"ESS":>7} {"khat":>7} {"sec":>7}')
    print(f'    {"MAP":>26} {err(x0).max():>10.4f} {"-":>12} {"-":>7} {"-":>7} {"-":>7}')
    print(f'    {"v1 joint-Laplace proposal":>26} {err(v1.theta_mean).max():>10.4f} '
          f'{sde(np.sqrt(np.maximum(np.diag(v1.theta_cov),0))).max():>12.2%} '
          f'{v1.ess/v1.n_nodes:>7.1%} {v1.khat:>7.2f} {t1:>7.1f}')
    print(f'    {"v2 profile-mode proposal":>26} {err(v2.theta_mean).max():>10.4f} '
          f'{sde(np.sqrt(np.maximum(np.diag(v2.theta_cov),0))).max():>12.2%} '
          f'{v2.ess/v2.n_nodes:>7.1%} {v2.khat:>7.2f} {t2:>7.1f}')
    print(f'    {"reference half-vs-half":>26} {fm:>10.4f} {fs:>12.2%}')
    print(f'    profile mode vs joint MAP theta, in Laplace sd: '
          f'{np.round((v2.theta_hat - x0[:p]) / np.sqrt(np.maximum(np.diag(np.linalg.pinv(np.asarray(m.hessian(),np.float64)))[:p],1e-300)), 3)}')
    print()
