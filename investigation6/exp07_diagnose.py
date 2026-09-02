"""
Exp 7: the whole pre-inference diagnosis, on every system, as one report.

None of this needs a reference chain or a sampler. It is what the fast MAP makes affordable, and
it answers the questions that decide whether any posterior approximation is meaningful before one
is attempted.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build, SYSTEMS
from profiled2 import ProfiledPosterior2
import mode

NAMES = {"fn": ["a", "b", "c"], "hes1": list("abcdefg"),
         "hiv": ["lam", "rho", "delta", "N", "c"], "lorenz": ["beta", "rho", "sigma"]}
for name in SYSTEMS:
    t0 = time.time()
    m, ds = build(name)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    x = np.asarray(m.map_particle, np.float64); p = m.p
    s = mode.spectrum(m, x)
    H = s["H"]; d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    w, V = np.linalg.eigh(H / np.outer(d, d)); sc = max(abs(w).max(), 1e-300)
    keep = w > 1e-10 * sc
    Sig = ((V[:, keep] / w[keep]) @ V[:, keep].T) / np.outer(d, d)
    sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
    pp = ProfiledPosterior2(m, n_nodes=8, seed=0)
    pr = mode.properness(m, pp, x)
    dt = time.time() - t0
    print(f'--- {name}   (dim {len(x)}, p {p}, diagnosis in {dt:.1f}s) ---')
    print(f'    mode: log p {float(m.logdensity(m.map_particle, m.data)):.4f}, '
          f'{s["n_neg"]} negative and {int((~keep).sum())} null directions, cond(DHD) '
          f'{sc/max(abs(w).min(),1e-300):.1e}')
    print(f'    {"param":>7} {"MAP":>13} {"post sd":>12} {"sd/|MAP|":>10} '
          f'{"log p fall":>12} {"verdict":>16}')
    for j in range(p):
        r = sd[j] / max(abs(x[j]), 1e-300)
        v = ("IMPROPER" if not pr[j]["proper"] else
             "identified" if r < 0.5 else "weak" if r < 5 else "diffuse")
        print(f'    {NAMES[name][j]:>7} {x[j]:>13.5g} {sd[j]:>12.5g} {r:>10.3g} '
              f'{pr[j]["fall"]:>12.4g} {v:>16}')
    print()
