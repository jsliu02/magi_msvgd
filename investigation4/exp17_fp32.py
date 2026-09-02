"""
Exp 17: does the recommended pipeline survive in float32, the production default?

The third-order correction is a gradient of a trace of a Hessian -- three derivative levels on a
quantity that is already a difference of large terms -- and the exact Hessian is assembled from
J^T J plus a residual-weighted second-derivative scatter. Both are plausible places for single
precision to fail, and MAGI's production default is float32. If it does fail the recommendation in
investigation4.md needs a precision caveat attached to it, so this is checked rather than assumed.

Compared against the float64 answer: the MAP, the Hessian, the correction vector itself, and the
resulting bias against the reference.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import build, cache
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler
from pipeline import metrics

d = H.DIM
z = np.load("ref4_baseline.npz")
sc = metrics(z["mean"], z["cov"], d)

def run(dtype):
    m = build(1, 0.2, dtype=dtype)
    gn = GaussNewtonMAP(m); pr = Profiler(gn, m)
    t0 = time.time(); gn.solve(verbose=False, tol=1e-6 if dtype == jnp.float64 else 1e-3, max_iter=200)
    x = np.asarray(gn.map_particle, np.float64); t_map = time.time() - t0
    t0 = time.time()
    Hj = pr._hess(jnp.asarray(x, dtype)); Hs = np.asarray(Hj, np.float64); Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs); Sig = (V / np.maximum(w, 1e-10 * w.max())) @ V.T
    t_h = time.time() - t0
    t0 = time.time()
    Sj = jnp.asarray(Sig, dtype)
    corr = np.asarray(0.5 * Sj @ jax.grad(lambda u: jnp.sum(Sj * pr._hess(u)))(jnp.asarray(x, dtype)),
                      np.float64)
    t_c = time.time() - t0
    g = float(jnp.linalg.norm(m.gradient(jnp.asarray(x, dtype)[None, :], m.data)))
    return dict(x=x, Hs=Hs, Sig=Sig, corr=corr, mu3=x - corr, gnorm=g,
                t=(t_map, t_h, t_c), minev=w.min())

r64, r32 = run(jnp.float64), run(jnp.float32)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
Hs = r64["Hs"]
tau = lambda v: float(np.sqrt(np.abs(v @ Hs @ v) / d))

print(f'{"":>26} {"float64":>14} {"float32":>14} {"rel diff":>11}')
print(f'{"MAP ||grad||":>26} {r64["gnorm"]:>14.2e} {r32["gnorm"]:>14.2e} '
      f'{rel(r32["x"], r64["x"]):>11.2e}')
print(f'{"Hessian":>26} {"":>14} {"":>14} {rel(r32["Hs"], r64["Hs"]):>11.2e}')
print(f'{"min eig(H)":>26} {r64["minev"]:>14.4f} {r32["minev"]:>14.4f} '
      f'{abs(r32["minev"]/r64["minev"]-1):>11.2e}')
print(f'{"third-order correction":>26} {"":>14} {"":>14} {rel(r32["corr"], r64["corr"]):>11.2e}')
print(f'{"  its size (tau)":>26} {tau(r64["corr"]):>14.5f} {tau(r32["corr"]):>14.5f}')
print(f'{"  fp32-vs-fp64 gap (tau)":>26} {"":>14} {"":>14} {tau(r32["corr"]-r64["corr"]):>11.5f}')
print()
print(f'{"":>26} {"bias":>9} {"trace":>8} {"forstner":>9} {"KL":>8}')
for lbl, r in [("fp64 MAP", (r64["x"], r64["Sig"])), ("fp64 + third order", (r64["mu3"], r64["Sig"])),
               ("fp32 MAP", (r32["x"], r32["Sig"])), ("fp32 + third order", (r32["mu3"], r32["Sig"]))]:
    s = sc(*r)
    print(f'{lbl:>26} {s["bias"]:>9.4f} {s["trace"]:>8.4f} {s["forst"]:>9.4f} {s["kl"]:>8.2f}')
print(f'\ntimings (MAP, Hessian, correction):  fp64 {tuple(round(v,2) for v in r64["t"])}  '
      f'fp32 {tuple(round(v,2) for v in r32["t"])}')
