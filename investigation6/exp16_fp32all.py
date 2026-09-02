"""Exp 16: with the corrected stencil, does float32 match float64 on every system?"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "investigation5", f"ref5_{n}.npz")
print(f'{"system":>8} {"dtype":>9} {"reliable":>9} {"ESS":>7} {"khat":>7} {"max|err|":>10} '
      f'{"floor":>8} {"sec":>6}')
for name in SYSTEMS:
    for dt in (jnp.float64, jnp.float32):
        m, ds = build(name, dtype=dt)
        t0 = time.time(); post = m.fit(verbose=False, tol=1e-9, max_iter=300); el = time.time()-t0
        p = m.p; g = post.diagnostics
        e = fl = np.nan
        if os.path.exists(R(name)):
            z = np.load(R(name)); rs = np.sqrt(np.maximum(np.diag(z["cov"]), 0)); hm = z["half_mean"]
            e = np.abs((post.mean[:p]-z["mean"][:p])/np.maximum(rs[:p],1e-300)).max()
            fl = np.abs((hm[0][:p]-hm[1][:p])/np.maximum(rs[:p],1e-300)).max()
        print(f'{name:>8} {dt.__name__:>9} {str(post.reliable):>9} '
              f'{g["ess"]/g["n_nodes"]:>7.1%} {g["khat"]:>7.2f} {e:>10.4f} {fl:>8.4f} {el:>6.1f}')
    print()
