"""
Exp 26: does the midpoint help on the ODE parameters, or only in aggregate?

The aggregate bias averages over 325 coordinates, 322 of which are trajectory states. A MAGI user
reads the p ODE parameters. A mean that wins on the average while losing on theta would be the
wrong recommendation, so theta is scored on its own here, against both the NUTS reference and the
independent 8-chain gold standard at baseline.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from setup4 import build, SETTINGS
from pipeline import metrics

d, P = H.DIM, 3
G = H.Gold()
gold = np.asarray(G.pos, np.float64)

for name in ["baseline", "half", "noisy"]:
    z = np.load(f"ref4_{name}.npz")
    ref_mu, ref_sd = z["mean"], np.sqrt(np.diag(z["cov"]))
    m = build(*SETTINGS[name], dtype=jnp.float64)
    post = m.fit(n_pairs=1024, verbose=False, tol=1e-8, max_iter=200)
    x0 = np.asarray(m.map_particle, np.float64)
    # recover the two component means from the certificates' construction
    mid = post.mean if post.applied else None
    sc = metrics(ref_mu, z["cov"], d); I = np.eye(d)
    print(f'--- {name} (theta errors in reference sd; aggregate bias over all {d} dims) ---')
    hdr = f'{"estimate":>14} ' + " ".join(f'{f"theta_{c}":>9}' for c in "abc") + \
          f' {"max|theta|":>11} {"aggregate":>10}'
    print(hdr)
    cands = [("MAP", x0)]
    if mid is not None:
        cands.append(("midpoint", mid))
    # rebuild mu3 alone for comparison
    Hs = np.asarray(m.hessian(x0), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, V = np.linalg.eigh(Hs); Sig = (V / ev) @ V.T
    Sj = jnp.asarray(Sig, m.mu.dtype)
    hfn = m._hessian_fn()
    mu3 = x0 - 0.5 * np.asarray(
        Sj @ jax.grad(lambda u: jnp.sum(Sj * hfn(u)))(jnp.asarray(x0, m.mu.dtype)), np.float64)
    cands.insert(1, ("third order", mu3))
    if mid is not None:
        cands.append(("VI (implied)", 2 * mid - mu3))
    for lbl, v in cands:
        e = (v[:P] - ref_mu[:P]) / ref_sd[:P]
        print(f'{lbl:>14} ' + " ".join(f'{x:>+9.4f}' for x in e) +
              f' {np.abs(e).max():>11.4f} {sc(v, I)["bias"]:>10.4f}')
    if name == "baseline":
        gm, gs = gold[:, :P].mean(0), gold[:, :P].std(0)
        for lbl, v in cands:
            e = (v[:P] - gm) / gs
            print(f'{lbl + " vs gold":>14} ' + " ".join(f'{x:>+9.4f}' for x in e) +
                  f' {np.abs(e).max():>11.4f}')
    print()
