"""
Exp 14: where does the profiled estimate lose precision in float32, and can a matmul guard fix it?

The importance weights depend on log p_hat(theta_i) only through their DIFFERENCES, so a constant
offset is harmless and what matters is the spread of the error across nodes. If that spread is s
nats then the effective sample size is degraded by roughly exp(-s^2), which is why an error of one
nat is fatal and one of 0.01 is not.

Three quantities are separated. The inner solve's accuracy, by evaluating the log-determinant in
float64 at the float32 solution. The log-determinant's own arithmetic. And the log-density term.
The first is limited by float32 itself; only the second and third can be helped by forcing
full-precision matmuls, and only on hardware where reduced-precision matmuls are the default --
on CPU float32 matmuls are already exact to float32 and the guard is a no-op, which is why the
earlier CPU measurement cannot have been a TF32 effect.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build

NAME = sys.argv[1] if len(sys.argv) > 1 else "fn"
NN = 96

def pieces(m, TH, X0):
    """Per-node (U, logdet H_XX, X*) with the model in whatever dtype it was put in."""
    gn = m._gn_solver(); p, n, D, nD = m.p, m.n, m.D, gn.nD
    dt = m.mu.dtype; sig = m.sigmas; hess = m._hessian_fn()
    eyeX = jnp.eye(nD, dtype=dt)
    def one(theta, Xs):
        def body(X, _):
            A, g, _ = gn._normal_equations(theta, X, sig)
            Axx = A[p:, p:] + 1e-10 * jnp.trace(A[p:, p:]) / nD * eyeX
            dg = jnp.diag(Axx); dg = jnp.where(dg > jnp.finfo(dg.dtype).tiny, dg, jnp.ones_like(dg))
            Di = jax.lax.rsqrt(dg)
            dX = Di * jax.scipy.linalg.cho_solve(
                jax.scipy.linalg.cho_factor(Axx * Di[:, None] * Di[None, :]), -(g[p:] * Di))
            return X + dX.reshape(n, D), None
        X, _ = jax.lax.scan(body, Xs, None, length=3)
        x = jnp.concatenate([theta, X.ravel()])
        Hxx = hess(x)[p:, p:]
        Hxx = 0.5 * (Hxx + Hxx.T)
        c = jax.scipy.linalg.cho_factor(Hxx)
        return (-m.logdensity(x, m.data), 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(c[0])))), X)
    f = jax.jit(jax.vmap(one, in_axes=(0, 0)))
    a, b, c = f(jnp.asarray(TH, dt), jnp.broadcast_to(jnp.asarray(X0, dt), (len(TH), n, D)))
    return np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(c, np.float64)

m64, ds = build(NAME, dtype=jnp.float64)
m64.map_solve(verbose=False, tol=1e-9, max_iter=300)
x0 = np.asarray(m64.map_particle, np.float64); p = m64.p
H = np.asarray(m64.hessian(), np.float64); H = 0.5 * (H + H.T)
d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
w, V = np.linalg.eigh(H / np.outer(d, d)); k = w > 1e-10 * abs(w).max()
Sig = ((V[:, k] / w[k]) @ V[:, k].T) / np.outer(d, d)
Sth = 0.5 * (Sig[:p, :p] + Sig[:p, :p].T)
wt, Vt = np.linalg.eigh(Sth); wt = np.maximum(wt, 1e-14 * wt.max())
L = (Vt * np.sqrt(wt)) @ Vt.T
rng = np.random.default_rng(0)
TH = x0[None, :p] + rng.standard_normal((NN, p)) @ L.T
X0 = x0[p:].reshape(m64.n, m64.D)

U64, LD64, Xs64 = pieces(m64, TH, X0)
m32, _ = build(NAME, dtype=jnp.float32)
m32.map_solve(verbose=False, tol=1e-9, max_iter=300)
U32, LD32, Xs32 = pieces(m32, TH, X0)

lw64, lw32 = -(U64 + 0.5 * LD64), -(U32 + 0.5 * LD32)
spread = lambda a: float(np.nanstd(a[np.isfinite(a)]))
print(f'--- {NAME} on {jax.devices()[0].platform}, {NN} nodes ---')
print(f'{"quantity":>34} {"spread of fp32-fp64 error":>26} {"implied ESS factor":>20}')
for lbl, e in [("log p_hat (end to end)", lw32 - lw64),
               ("  its U term", -(U32 - U64)),
               ("  its 0.5 logdet term", -0.5 * (LD32 - LD64))]:
    s = spread(e)
    print(f'{lbl:>34} {s:>26.4g} {np.exp(-s**2):>20.3g}')
print(f'{"inner solve ||X*_32 - X*_64|| / ||X*||":>34} '
      f'{float(np.linalg.norm(Xs32-Xs64)/max(np.linalg.norm(Xs64),1e-300)):>26.3e}')
