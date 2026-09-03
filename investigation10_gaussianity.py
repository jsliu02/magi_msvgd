"""
Reproduces every number in posterior_gaussianity.tex.  python investigation10_gaussianity.py
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "magi_msvgd"))
import tests as ode
from magi import MAGI


def affine_ode(X, theta, t):
    """Affine in X; theta enters through exp, sin, a square and a reciprocal. Eq. (7)."""
    a, b, c = theta
    A = jnp.array([[-jnp.exp(a), jnp.sin(3 * b)], [c ** 2, -1.0 / (1.0 + a ** 2)]])
    return A @ X + jnp.array([jnp.cos(2 * b), jnp.exp(-c)])


AFFINE = ode.DynamicalSystem(affine_ode, {
    "theta": np.array([0.3, 0.7, 0.5]), "x0": np.array([1.0, -0.5]),
    "sigma": np.array([0.05, 0.05]), "tau": [np.linspace(0, 4, 21)] * 2,
    "I": np.linspace(0, 4, 81)})

CASES = [("affine, eq. (7)", AFFINE, affine_ode, np.ones(3) * 1e-2)] + [
    (n, ode.SYSTEMS[n], ode.SYSTEMS[n].ode,
     np.zeros(len(ode.SYSTEMS[n].hyperparams["theta"]))) for n in
    ("HIV", "Lorenz", "FitzHughNagumo", "Hes1")]


def fit(mdl, odef, prec):
    mdl.reset()
    m = MAGI(odef, mdl.dataset(seed=0), mdl.hyperparams["theta"], theta_prec=prec,
             sigmas=mdl.hyperparams["sigma"])
    m.put(jnp.float64, jax.devices()[0])
    m.map_solve(verbose=False)
    return m


print("Table 2 -- departure from Condition 1, absolute vs relative perturbation")
print(f'{"system":<18} {"|X| rms":>10} {"absolute 0.7":>14} {"relative 10%":>14}')
models = {}
for nm, mdl, odef, prec in CASES:
    m = models[nm] = fit(mdl, odef, prec)
    rms = float(np.sqrt(np.mean(np.asarray(m.map_particle, np.float64)[m.p:] ** 2)))
    print(f"{nm:<18} {rms:>10.3g} {m.condition_A(scale=0.7):>14.3e} "
          f"{m.condition_A(rel=0.1):>14.3e}", flush=True)

print("\nTable 1 -- relative error of the quadratic model of U in x")
print(f'{"|delta|":>10} {"affine, eq. (7)":>18} {"FitzHughNagumo":>18}')
rows = {}
for nm in ("affine, eq. (7)", "FitzHughNagumo"):
    m = models[nm]
    p, nD = m.p, m.n * m.D
    x0 = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(x0), np.float64)[p:p + nD, p:p + nD]
    g = np.asarray(m.gradient(jnp.asarray(x0)[None, :], m.data), np.float64)[0][p:p + nD]
    U0 = -float(m.logdensity(jnp.asarray(x0), m.data))
    v = np.random.default_rng(1).standard_normal(nD); v /= np.linalg.norm(v)
    out = []
    for s in (1.0, 10.0, 100.0):
        z = x0.copy(); dx = s * v; z[p:p + nD] += dx
        U = -float(m.logdensity(jnp.asarray(z), m.data))
        out.append(abs(U - (U0 - g @ dx + 0.5 * dx @ H @ dx)) / max(abs(U - U0), 1e-300))
    rows[nm] = out
for i, s in enumerate((1.0, 10.0, 100.0)):
    print(f'{s:>10g} {rows["affine, eq. (7)"][i]:>18.3e} {rows["FitzHughNagumo"][i]:>18.3e}')

print("\nSection 4 -- third derivative along a line in x, and along a line in theta")
m = models["affine, eq. (7)"]
p, nD = m.p, m.n * m.D
x0 = np.asarray(m.map_particle, np.float64)
U = lambda z: -float(m.logdensity(jnp.asarray(z), m.data))
rng = np.random.default_rng(0)
h = 1e-3
for label, k, sc in (("x", nD, 50.0), ("theta", p, 0.5)):
    u = rng.standard_normal(k); u /= np.linalg.norm(u)
    def g(s):
        z = x0.copy()
        sl = slice(p, p + nD) if label == "x" else slice(0, p)
        z[sl] += s * u * sc
        return U(z)
    d2 = (g(h) - 2 * g(0) + g(-h)) / h ** 2
    d3 = (g(2 * h) - 2 * g(h) + 2 * g(-h) - g(-2 * h)) / (2 * h ** 3)
    print(f"  along {label:<6} U'' = {d2:>13.6e}   U''' = {d3:>11.3e}   "
          f"ratio {abs(d3) / abs(d2):.2e}")
