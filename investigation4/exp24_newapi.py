"""Exp 24: validate the rewritten magi.py against the numbers investigation4 established."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from pipeline import metrics

d = H.DIM; I = np.eye(d)
print(f'{"setting":>9} {"dtype":>8} {"pairs":>6} | {"q_max":>7} {"kappa_S":>10} {"tau_end":>9} {"ratio":>9} {"gate":>9} | '
      f'{"bias MAP":>9} {"bias fit":>9} {"expect":>7} {"sec":>5}')
EXPECT = {("baseline", 1024): 0.0234, ("half", 1024): 0.0503,
          ("noisy", 1024): 0.1699, ("quarter", 1024): 0.1972}
for name in ["baseline", "half", "noisy", "quarter"]:
    z = np.load(f"ref4_{name}.npz"); sc = metrics(z["mean"], z["cov"], d)
    for dtype, pairs in [(jnp.float64, 1024), (jnp.float64, 256), (jnp.float32, 256)]:
        m = build(*SETTINGS[name], dtype=dtype)
        t0 = time.time()
        post = m.fit(n_pairs=pairs, verbose=False, tol=1e-8, max_iter=200)
        dt = time.time() - t0
        c = post.certificates
        exp = EXPECT.get((name, pairs))
        print(f'{name:>9} {str(dtype.__name__):>8} {pairs:>6} | {c["q_max"]:>7.2f} '
              f'{c["kappa_S"]:>10.2e} {c["tau_end"]:>9.2e} {c["ratio"]:>9.3f} '
              f'{("apply" if post.applied else "SUPPRESS"):>9} | '
              f'{sc(np.asarray(m.map_particle, np.float64), I)["bias"]:>9.4f} '
              f'{sc(post.mean, I)["bias"]:>9.4f} '
              f'{(f"{exp:.4f}" if exp else "-"):>7} {dt:>5.2f}')
    print()

m = build(*SETTINGS["baseline"], dtype=jnp.float64)
post = m.fit(verbose=False)
Xs, th, sg = post.sample(k=500, seed=0)
print(f'sample(): Xs {Xs.shape}  thetas {th.shape}  sigmas {sg.shape}')
print(f'condition_A (FitzHugh-Nagumo, expect ~0.13): {m.condition_A():.4f}')
print()
print(repr(post))
