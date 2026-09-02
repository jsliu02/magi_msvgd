"""
Exp 27: re-score the pipeline against the longer references, before committing to conclusions.

The 400-chain x 150-draw references had Rhat up to 1.77. These are 64 chains x 2000 draws, which
brings noisy to 1.03 and quarter to 1.11. The theta-versus-aggregate finding and the gate
threshold were both read off the short references, so both are re-checked here.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from pipeline import metrics

d, P, I = H.DIM, 3, np.eye(H.DIM)
REF = {"baseline": "ref4_baseline.npz", "half": "ref5_half.npz",
       "noisy": "ref5_noisy.npz", "quarter": "ref5_quarter.npz"}

print(f'{"setting":>9} {"Rhat":>6} {"div%":>6} | {"estimate":>12} ' +
      " ".join(f'{f"th_{c}":>8}' for c in "abc") + f' {"max|th|":>8} {"aggregate":>10}')
print("-" * 104)
rows = {}
for name in ["baseline", "half", "noisy", "quarter"]:
    z = np.load(REF[name]); rm, rs = z["mean"], np.sqrt(np.diag(z["cov"]))
    sc = metrics(rm, z["cov"], d)
    m = build(*SETTINGS[name], dtype=jnp.float64)
    post = m.fit(n_pairs=1024, verbose=False, tol=1e-8, max_iter=200)
    c = post.certificates
    rows[name] = (post, c)
    dv = 100 * int(z["div"]) / int(z["ndraw"])
    first = True
    for lbl, v in [("MAP", post.mu_map), ("third order", post.mu3),
                   ("midpoint", post.mu_mid), ("VI", post.mu_vi)]:
        e = (v[:P] - rm[:P]) / rs[:P]
        agg = sc(v, I)["bias"] if np.all(np.isfinite(v)) else np.inf
        tag = f'{name:>9} {float(z["rhat"].max()):>6.3f} {dv:>6.2f} | ' if first else f'{"":>9} {"":>6} {"":>6} | '
        print(tag + f'{lbl:>12} ' + " ".join(f'{x:>+8.4f}' for x in e) +
              f' {np.abs(e).max():>8.4f} {agg:>10.4f}')
        first = False
    print(f'{"":>9} {"":>6} {"":>6} | {"GATE":>12} ratio {c["ratio"]:>10.3f}  |d3| {c["d3"]:.3f}  '
          f'-> {"apply" if post.applied else "SUPPRESS"}')
    print()

print("does the gate call match what theta says (correction better than MAP)?")
for name, (post, c) in rows.items():
    z = np.load(REF[name]); rm, rs = z["mean"], np.sqrt(np.diag(z["cov"]))
    e = lambda v: float(np.abs((v[:P] - rm[:P]) / rs[:P]).max())
    helps = e(post.mu3) < e(post.mu_map)
    print(f'  {name:>9}: gate={"apply" if post.applied else "suppress":>8}  '
          f'third order helps theta: {str(helps):>5}  '
          f'({e(post.mu_map):.3f} -> {e(post.mu3):.3f})   '
          f'{"AGREE" if post.applied == helps else "DISAGREE"}')
