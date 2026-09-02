"""
Exp 2: why is HIV unsolvable, and is it the problem or the parameterisation?

cond(H) = 4.3e16 sits at float64's limit and the Gauss-Newton solve stalls at ||grad|| = 1.7e-2
where it reaches 1e-8 elsewhere. Two candidate causes with very different remedies. If the
conditioning is intrinsic -- genuine non-identifiability -- no amount of arithmetic helps and the
posterior simply has near-flat directions. If it is an artefact of the units, in which HIV's states
span 30 to 1e5 and its parameters 0.108 to 1000, then a diagonal rescaling fixes it and the fix is
general: the normal equations SQUARE the condition number, so a Jacobian conditioned at 1e8 -- easy
to reach just by mixing units -- destroys the Cholesky that the fast solver depends on.

Reported per system: the conditioning before and after Jacobi scaling, and where the small
eigenvalues actually live (parameters, observed states, unobserved states).
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS

print(f'{"system":>8} {"cond(H)":>10} {"cond(DHD)":>10} {"gain":>8} | '
      f'{"softest direction: mass on":>28} | {"scale spread of diag(H)":>24}')
print("-" * 110)
for name in SYSTEMS:
    m, ds = build(name)
    m.map_solve(verbose=False, tol=1e-10, max_iter=400)
    x = np.asarray(m.map_particle, np.float64)
    Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
    p, n, D = m.p, m.n, m.D
    dg = np.sqrt(np.maximum(np.diag(Hs), 1e-300))
    Dm = 1.0 / dg
    Hd = Hs * np.outer(Dm, Dm)                      # Jacobi-scaled: unit diagonal
    c0 = np.linalg.cond(Hs); c1 = np.linalg.cond(Hd)
    w, V = np.linalg.eigh(Hs)
    v = V[:, 0] ** 2
    obs = np.asarray(m.Ns) > 2
    idx_states = p + np.arange(n * D)
    comp = idx_states.reshape(n, D)
    mass_th = v[:p].sum()
    mass_obs = sum(v[comp[:, j]].sum() for j in range(D) if obs[j])
    mass_un = sum(v[comp[:, j]].sum() for j in range(D) if not obs[j])
    print(f'{name:>8} {c0:>10.2e} {c1:>10.2e} {c0/max(c1,1e-300):>8.1e} | '
          f'theta {mass_th:>6.1%}  obs X {mass_obs:>6.1%}  unobs X {mass_un:>6.1%} | '
          f'{dg.max()/dg.min():>24.2e}')

print(f'\nHIV detail: diagonal scale of H by block')
m, ds = build("hiv")
m.map_solve(verbose=False, tol=1e-10, max_iter=400)
Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
dg = np.sqrt(np.abs(np.diag(Hs)))
p, n, D = m.p, m.n, m.D
print(f'  theta      : {np.array2string(dg[:p], precision=2, max_line_width=110)}')
for j, nm in enumerate(["T_U", "T_I", "V"]):
    b = dg[p + np.arange(n) * D + j]
    print(f'  X[{nm:>3}]    : median {np.median(b):.3e}   min {b.min():.3e}   max {b.max():.3e}')
tr = np.asarray(ds.hyperparams["theta"], np.float64)
print(f'  true theta : {tr}')
print(f'  state scale: {np.abs(np.asarray(ds.solution)).max(axis=0)}')
