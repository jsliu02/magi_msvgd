"""
Exp 8: how fast can the least-squares MAP solve be?

Five independent axes, measured separately so they can be combined:
  (1) is the first-order warm start needed at all?
  (2) which linear solve -- lstsq (SVD), QR, or normal equations + Cholesky?
  (3) fp32 with fp64 iterative refinement, on hardware where fp64 is 1/64 rate
  (4) CPU vs GPU at these matrix sizes
  (5) is the GP precision banded? if so the whole solve is O(n b^2), not O(n^3)
"""
import numpy as np, jax, jax.numpy as jnp, time, sys
jax.config.update("jax_enable_x64", True)
import harness as H
from lsq import LSQ
from jac import AnalyticJac

m = H.build_magi(dtype=jnp.float64)
l = LSQ(m); aj = AnalyticJac(l)
x_init = jnp.asarray(np.asarray(m.particles_init, np.float64))
gnorm = lambda x: float(jnp.linalg.norm(m.gradient(jnp.asarray(x)[None, :], m.data)))

# ---------------------------------------------------------------- (5) bandedness of the GP precision
print("(5) structure of the GP/ODE precision matrices")
for nm, A in [("C^-1", np.asarray(m.C_invs)[0]), ("K^-1", np.asarray(m.K_invs)[0])]:
    n = A.shape[0]; sc = np.abs(A).max()
    dec = [np.abs(np.diag(A, k)).max() / sc for k in (1, 3, 5, 10, 20, 40)]
    bw = next((k for k in range(n) if np.abs(np.diag(A, k)).max() / sc < 1e-8), n)
    print(f"    {nm}: |offdiag|/max at lag 1,3,5,10,20,40 = "
          + ", ".join(f"{v:.1e}" for v in dec) + f"   numerical bandwidth (1e-8) = {bw}/{n}")

# ---------------------------------------------------------------- (2)(3) linear solve variants
J = np.asarray(aj(x_init), np.float64); r = np.asarray(l.residual(x_init), np.float64)
Jj, rj = jnp.asarray(J), jnp.asarray(r)
J32, r32 = jnp.asarray(J, jnp.float32), jnp.asarray(r, jnp.float32)
print(f"\n(2,3) one linear solve, J is {J.shape}, cond(J) = {np.linalg.cond(J):.1f}")
ref = np.linalg.lstsq(J, r, rcond=None)[0]
def bench(fn, name, n=20):
    fn().block_until_ready(); t0 = time.time()
    for _ in range(n): out = fn()
    out.block_until_ready()
    err = float(np.linalg.norm(np.asarray(out, np.float64) - ref) / np.linalg.norm(ref))
    print(f"    {name:>44} {(time.time()-t0)/n*1000:8.2f} ms   rel err {err:.2e}")
bench(jax.jit(lambda: jnp.linalg.lstsq(Jj, rj, rcond=None)[0]), "lstsq / SVD (fp64)")
def qr64():
    Q, Rb = jnp.linalg.qr(Jj); return jax.scipy.linalg.solve_triangular(Rb, Q.T @ rj, lower=False)
bench(jax.jit(qr64), "QR + triangular solve (fp64)")
def chol64():
    A = Jj.T @ Jj; c = jax.scipy.linalg.cho_factor(A); return jax.scipy.linalg.cho_solve(c, Jj.T @ rj)
bench(jax.jit(chol64), "normal equations + Cholesky (fp64)")
def chol32():
    A = J32.T @ J32; c = jax.scipy.linalg.cho_factor(A); return jax.scipy.linalg.cho_solve(c, J32.T @ r32)
bench(jax.jit(chol32), "normal equations + Cholesky (fp32)")
def chol32_ir():
    A = J32.T @ J32; c = jax.scipy.linalg.cho_factor(A)
    d = jax.scipy.linalg.cho_solve(c, J32.T @ r32).astype(jnp.float64)
    res = Jj.T @ (rj - Jj @ d)                                   # fp64 residual
    return d + jax.scipy.linalg.cho_solve((c[0].astype(jnp.float64), c[1]), res)
bench(jax.jit(chol32_ir), "fp32 Cholesky + one fp64 refinement step")

# ---------------------------------------------------------------- (1) is the warm start needed?
print("\n(1) Gauss-Newton straight from the MAGI initialization (no first-order phase)")
@jax.jit
def gn_step(x):
    Jx = aj(x); A = Jx.T @ Jx
    c = jax.scipy.linalg.cho_factor(A + 1e-12 * jnp.trace(A) / A.shape[0] * jnp.eye(A.shape[0]))
    return x - jax.scipy.linalg.cho_solve(c, Jx.T @ l.residual(x))
x = x_init; gn_step(x).block_until_ready()
t0 = time.time(); hist = []
for i in range(30):
    x = gn_step(x)
    if i in (0, 1, 2, 4, 9, 19, 29): hist.append((i + 1, gnorm(x)))
x.block_until_ready(); dt = time.time() - t0
print("    ||grad|| after k steps: " + "  ".join(f"k={k}:{g:.1e}" for k, g in hist))
print(f"    30 Gauss-Newton steps from the raw initialization: {dt:.2f}s")
print(f"    (investigation3's pipeline used 20000 Prodigy iterations first, then 60 GN steps)")
np.save("x_map_fast.npy", np.asarray(x))
