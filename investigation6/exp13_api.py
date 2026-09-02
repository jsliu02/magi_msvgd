"""Exp 13: the renamed API, matrix precision, and fit() on the profiled method."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
from magi import MAGI
import tests as T

REF = lambda n: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "investigation5", f"ref5_{n}.npz")
print("=== fit() on all four systems, flat prior (theta_prec = 0) ===")
for name in SYSTEMS:
    m, ds = build(name)
    t0 = time.time(); post = m.fit(verbose=False, tol=1e-9, max_iter=300); dt = time.time() - t0
    p = m.p
    line = f'{name:>8} reliable={str(post.reliable):>5} ESS {post.diagnostics["ess"]/post.diagnostics["n_nodes"]:>5.1%} '
    line += f'khat {post.diagnostics["khat"]:>6.2f} null {post.diagnostics["n_null"]:>3} {dt:>6.1f}s'
    if os.path.exists(REF(name)):
        z = np.load(REF(name)); rs = np.sqrt(np.maximum(np.diag(z["cov"]), 0))
        e = np.abs((post.mean[:p] - z["mean"][:p]) / np.maximum(rs[:p], 1e-300)).max()
        hm = z["half_mean"]
        fl = np.abs((hm[0][:p] - hm[1][:p]) / np.maximum(rs[:p], 1e-300)).max()
        line += f'  max|err| {e:>8.4f} (floor {fl:.4f})'
    print(line)

print("\n=== theta_prec as scalar / vector / matrix must agree when equivalent ===")
ds = T.FitzHughNagumo
data = ds.reset().dataset(seed=0, step=1e-3)
g = np.array([1.0, 1.0, 1.0])
outs = {}
for lbl, prec in [("scalar 4.0", 4.0), ("vector [4,4,4]", np.array([4.0, 4.0, 4.0])),
                  ("matrix 4*I", 4.0 * np.eye(3))]:
    m = MAGI(ds.ode, data, g, prec, sigmas=[0.2, 0.2]); m.put(jnp.float64)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    outs[lbl] = np.asarray(m.map_particle, np.float64)[:3]
    print(f'  {lbl:>16}: theta = {np.round(outs[lbl], 8)}')
k = list(outs)
print(f'  max disagreement: {max(np.abs(outs[k[0]]-outs[x]).max() for x in k[1:]):.2e}')

print("\n=== a correlated (non-diagonal) precision is accepted and does something different ===")
Pc = np.array([[4.0, 3.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
m = MAGI(ds.ode, data, g, Pc, sigmas=[0.2, 0.2]); m.put(jnp.float64)
m.map_solve(verbose=False, tol=1e-9, max_iter=300)
print(f'  correlated prec: theta = {np.round(np.asarray(m.map_particle)[:3], 6)}')
print(f'  jacobian check with a full precision matrix: {m._gn.check_jacobian():.2e}')

print("\n=== errors are raised, not silently broadcast ===")
for lbl, prec in [("wrong length", np.ones(4)), ("wrong shape", np.ones((2, 2))),
                  ("not PSD", np.array([[1.0, 5.0, 0], [5.0, 1.0, 0], [0, 0, 1.0]]))]:
    try:
        MAGI(ds.ode, data, g, prec, sigmas=[0.2, 0.2])
        print(f'  {lbl:>14}: NO ERROR (bad)')
    except ValueError as e:
        print(f'  {lbl:>14}: ValueError - {str(e)[:64]}')
