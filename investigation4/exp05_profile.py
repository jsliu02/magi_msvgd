"""
Exp 5: does the Laplace marginal recover the true variance along the screened directions?

The screen (exp 4) flags directions; this tests whether the profile construction FIXES them. Truth
is the gold chain's marginal variance along each direction; the baseline to beat is the Laplace
value 1/eig, which was 8% and 6% too large on the two worst directions.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import harness as H
from setup4 import cache
from gauss_newton import GaussNewtonMAP
from profile_marg import Profiler, moments

G = H.Gold()
m, x_map, Hs, Sig, L = cache("baseline")
D = np.load("dirs_baseline.npz")
ev, V, qscr = D["ev"], D["V"], D["qscr"]
gold = np.asarray(G.pos, np.float64)
proj = (gold - gold.mean(0)) @ V

pr = Profiler(GaussNewtonMAP(m), m)
top = np.argsort(-qscr)[:6]
print(f'{"dir":>4} {"eig":>8} {"screen q":>9} | {"var: Laplace":>12} {"profile":>9} {"gold":>9} | '
      f'{"err Lap":>8} {"err prof":>9} | {"mean: prof":>11} {"gold":>8} | {"sec":>6}')
res = []
for j in top:
    sd = 1.0 / np.sqrt(ev[j])
    zs = np.linspace(-4 * sd, 4 * sd, 17)
    t0 = time.time()
    U, ld, _ = pr.profile(V[:, j], zs, x_map)
    _, vP = moments(zs, -U - ld)
    mP, _ = moments(zs, -U - ld)
    dt = time.time() - t0
    vL, vG, mG = sd ** 2, proj[:, j].var(), proj[:, j].mean()
    res.append((j, vL, vP, vG))
    print(f'{j:>4} {ev[j]:>8.3f} {qscr[j]:>9.3f} | {vL:>12.5f} {vP:>9.5f} {vG:>9.5f} | '
          f'{vL/vG-1:>+8.1%} {vP/vG-1:>+9.1%} | {mP:>+11.4f} {mG:>+8.4f} | {dt:>6.2f}')
print(f'\nmean |variance error| over these 6: Laplace {np.mean([abs(a/c-1) for _,a,_,c in res]):.2%}'
      f'  ->  profile {np.mean([abs(b/c-1) for _,_,b,c in res]):.2%}')
np.savez("profile_baseline.npz", top=top, res=np.array(res))
