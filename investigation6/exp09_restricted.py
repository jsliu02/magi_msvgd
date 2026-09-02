"""
Exp 9: does excluding the improper directions rescue inference on HIV?

The claim being tested is specific. HIV's importance sampling collapses to ESS = 1 not because the
target is hard but because two of the five parameter directions have no finite width, so a
proposal that must cover them wastes every node. Profiling those directions out instead of
integrating them should restore a healthy effective sample size on the three the data determines,
without changing what is being asked about them.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build
from profiled2 import ProfiledPosterior2
from restricted import RestrictedProfile

NM = ["lam", "rho", "delta", "N", "c"]
m, ds = build("hiv")
m.map_solve(verbose=False, tol=1e-9, max_iter=300)
x0 = np.asarray(m.map_particle, np.float64); p = m.p
H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
w, V = np.linalg.eigh(H / np.outer(d, d))
keep = w > 1e-10 * max(abs(w).max(), 1e-300)
Sig = ((V[:, keep] / w[keep]) @ V[:, keep].T) / np.outer(d, d)
sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
print(f'Laplace theta sds: ' + "  ".join(f'{NM[j]} {sd[j]:.4g}' for j in range(p)))

print(f'\nfull 5-dimensional theta integration:')
t0 = time.time()
pp = ProfiledPosterior2(m, n_nodes=256, seed=0).build(verbose=False)
print(f'    ESS {pp.ess/pp.n_nodes:>6.1%}  khat {pp.khat:>6.2f}  reliable={pp.reliable}  '
      f'{time.time()-t0:.1f}s')

for tag, cols in [("identified axes (delta, N, c)", [2, 3, 4]),
                  ("identified + rho", [1, 2, 3, 4])]:
    B = np.zeros((p, len(cols)))
    for i, c in enumerate(cols):
        B[c, i] = 1.0
    t0 = time.time()
    rp = RestrictedProfile(m, B, n_nodes=256, seed=0).build(verbose=False)
    tm = rp.theta_mean
    print(f'\nrestricted to {tag}:')
    print(f'    ESS {rp.ess/rp.n_nodes:>6.1%}  failed {int((~rp.ok).sum())}  {time.time()-t0:.1f}s')
    print(f'    {"param":>7} {"MAP":>13} {"restricted mean":>17} {"true":>10} {"role":>12}')
    tru = np.asarray(ds.hyperparams["theta"], np.float64)
    for j in range(p):
        role = "integrated" if j in cols else "profiled out"
        print(f'    {NM[j]:>7} {x0[j]:>13.5g} {tm[j]:>17.5g} {tru[j]:>10.5g} {role:>12}')
