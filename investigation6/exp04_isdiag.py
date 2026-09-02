"""
Exp 4: why does importance sampling collapse on Hes1, and is it the proposal or the target?

Hes1's parameters are all weakly determined -- posterior standard deviations of order 1 around a
mode near zero -- so the Laplace theta-marginal is wide and the profiled marginal may be far from
Gaussian. Three things are separable here. Whether more adaptation rounds recover it (proposal
LOCATION). Whether heavier proposal tails recover it (proposal SHAPE). And whether the profiled
log-density is simply rough or multimodal in theta, which no Gaussian proposal will fix and which
argues for replacing importance sampling with a sampler.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build, SYSTEMS
from profiled import ProfiledPosterior

for name in ("hes1", "fn"):
    m, ds = build(name)
    m.map_solve(verbose=False, tol=1e-9, max_iter=300)
    print(f'--- {name} ---')
    for rounds in (3, 6, 10):
        pp = ProfiledPosterior(m, n_nodes=512, seed=0).adapt(rounds=rounds, verbose=False)
        print(f'    rounds {rounds:>3}: ESS {pp.ess/pp.n_nodes:>5.1%}  khat {pp.khat:>6.2f}  '
              f'failed {int((~pp.ok).sum()):>4}')
    for sc in (0.3, 0.5, 2.0):
        pp = ProfiledPosterior(m, n_nodes=512, seed=0, scale=sc).adapt(rounds=6, verbose=False)
        print(f'    scale {sc:>4}: ESS {pp.ess/pp.n_nodes:>5.1%}  khat {pp.khat:>6.2f}')
    # shape of the profiled surface along the Laplace principal axes
    pp = ProfiledPosterior(m, n_nodes=512, seed=0).adapt(rounds=6, verbose=False)
    lr = pp.log_ratio[np.isfinite(pp.log_ratio)]
    print(f'    log-weight spread: sd {lr.std():.2f}, range {lr.max()-lr.min():.1f} '
          f'(ESS ~ exp(-var) rule: {np.exp(-lr.var()):.3f})')
    inner = pp.inner_grad[pp.ok]
    print(f'    inner solve residual: median {np.median(inner):.2e}, max {inner.max():.2e}')
    print()
