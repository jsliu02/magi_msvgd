"""
Exp 1: what does each system's posterior actually look like, and does the investigation-4 pipeline
survive contact with it?

Everything in investigation 4 was measured on FitzHugh-Nagumo. This runs the shipped pipeline
unchanged on all four systems and records the geometry alongside, so that later results can be
read against how hard each problem is rather than in isolation.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS

print(f'{"system":>8} {"dim":>5} {"MAP sec":>8} {"||grad||":>10} {"min eig":>10} {"cond":>9} '
      f'{"tr(Sig)":>9} | {"q_max":>9} {"kappa_S":>10} {"ratio":>9} {"gate":>8} {"fit s":>6}')
print("-" * 128)
for name in SYSTEMS:
    try:
        t0 = time.time(); m, ds = build(name); t_b = time.time() - t0
        t0 = time.time(); m.map_solve(verbose=False, tol=1e-8, max_iter=300); t_m = time.time() - t0
        g = float(jnp.linalg.norm(m.gradient(jnp.asarray(m.map_particle)[None, :], m.data)))
        Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
        ev = np.linalg.eigvalsh(Hs)
        pd = ev.min() > 0
        tr = np.trace(np.linalg.inv(Hs)) if pd else np.nan
        t0 = time.time()
        post = m.fit(verbose=False, tol=1e-8, max_iter=300) if pd else None
        t_f = time.time() - t0
        c = post.certificates if post else {}
        print(f'{name:>8} {m.p + m.n * m.D:>5} {t_m:>8.2f} {g:>10.2e} {ev.min():>10.3e} '
              f'{ev.max()/max(ev.min(),1e-300):>9.2e} {tr:>9.3f} | '
              + (f'{c["q_max"]:>9.2f} {c["kappa_S"]:>10.2e} {c["ratio"]:>9.3f} '
                 f'{("apply" if post.applied else "WARN"):>8} {t_f:>6.2f}'
                 if post else f'{"HESSIAN NOT PD -- no Gaussian exists here":>45}'))
    except Exception as e:
        import traceback
        print(f'{name:>8}  FAILED {type(e).__name__}: {str(e)[:80]}')
