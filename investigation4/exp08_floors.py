"""Exp 8 (rerun): calibrated floors, then the deterministic pipeline scored against them."""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
from pipeline import metrics, floors, build, cov_of

G = H.Gold(); d = H.DIM
m, x_map, Hs, Sig, L = cache("baseline")
gold = np.asarray(G.pos, np.float64)
obs, n_eff, fl = floors(gold, d)
print(f'gold half-vs-half forstner = {obs:.4f}  ->  inferred n_eff = {n_eff} '
      f'({100*n_eff/len(gold):.1f}% of {len(gold)} draws)')
print(f'FLOOR for a noiseless approximation vs the full chain: '
      f'bias {fl["bias"]:.4f}  trace {fl["trace"]:.4f}  forstner {fl["forst"]:.4f}  KL {fl["kl"]:.2f}\n')

sc = metrics(gold.mean(0), np.cov(gold, rowvar=False), d)
b = build(m, x_map, Hs, Sig, M=40)
np.savez("build_baseline.npz", **{k: v for k, v in b.items() if k != "t"})
print(f'{"approximation":>34} {"bias":>7} {"trace":>7} {"forstner":>9} {"KL":>9}')
def show(l, r): print(f'{l:>34} {r["bias"]:>7.4f} {r["trace"]:>7.4f} {r["forst"]:>9.4f} {r["kl"]:>9.2f}')
show("N(MAP, H^-1)", sc(b["mu0"], Sig))
show("N(MAP + third order, H^-1)", sc(b["mu3"], Sig))
for M in (5, 10, 20, 40):
    show(f"  + profile scale m={M}", sc(b["mu3"], cov_of(b, Sig, M)))
show("FLOOR (noiseless vs full chain)", fl)
print(f'\nrejected profiles: {int((~b["ok"]).sum())}/40   cost {b["t"]}')
