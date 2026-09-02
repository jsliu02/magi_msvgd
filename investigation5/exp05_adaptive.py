"""
Exp 5: the profiled posterior with an adapted proposal, plus a correctness check that needs no
reference -- at theta = theta_MAP the inner profile must reproduce the MAP trajectory exactly.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map
from profiled import ProfiledPosterior

N = {"fn": 512, "hes1": 512, "lorenz": 512, "hiv": 128}
for name in ["fn", "hes1", "lorenz", "hiv"]:
    m, ds = build(name)
    scaled_map(m, tol=1e-10, max_iter=400)
    x0 = np.asarray(m.map_particle, np.float64)
    p = m.p
    pp = ProfiledPosterior(m, n_nodes=N[name], seed=0)
    err, lw, ok = pp.check()
    print(f'--- {name} ---   inner solve at theta_MAP reproduces X_MAP to {err:.2e} '
          f'(finite: {ok})')
    t0 = time.time(); pp.adapt(rounds=4, verbose=True); dt = time.time() - t0
    tm, tc = pp.theta_mean, pp.theta_cov
    sd = np.sqrt(np.maximum(np.diag(tc), 0))
    tru = np.asarray(ds.hyperparams["theta"], np.float64)
    k = min(p, 6)
    print(f'    {"":>14} ' + " ".join(f'{f"th{j}":>11}' for j in range(k)))
    for lbl, v in [("true", tru), ("MAP", x0[:p]), ("profiled mean", tm), ("profiled sd", sd)]:
        print(f'    {lbl:>14} ' + " ".join(f'{v[j]:>11.5f}' for j in range(k)))
    print(f'    total {dt:.1f}s\n')
