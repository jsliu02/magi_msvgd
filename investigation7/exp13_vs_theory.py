"""
exp13: check the measurement apparatus against the published theory.

Ba, Erdogdu, Ghassemi, Sun, Suzuki, Wu & Zhang, "Understanding the Variance Collapse of SVGD in
High Dimensions", ICLR 2022 (OpenReview Qycd9j5Qp9J; no arXiv version) prove, for an isotropic
Gaussian target in the proportional limit n, d -> infinity with gamma = d/n > 1, and for the
Gaussian RBF kernel under the PLAIN median heuristic sigma = sqrt(Med{||x_i-x_j||^2}/2)
(equivalently h = 2 sigma^2 = Med, with NO 1/log n factor):

    Corollary 4:   v_SVGD  =  (e - 1)^{-1} gamma^{-1}  =  0.5820 * n / d.

`msvgd.MSVGD.pairwise_distance` implements the Liu & Wang (2016) variant with the extra
1/ln K -- h = Med / ln K -- and under THAT rule exp07/exp08 measure v_SVGD = ln(K)/d. The two
bandwidth conventions are different, so the two laws should be different; but if my harness is
sound, running it with Ba et al.'s convention must reproduce Ba et al.'s constant.

This runs both conventions side by side at gamma = d/n of 2 and 4, where their proportional-limit
assumption holds.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import msvgd7 as M7

MAXIT = int(os.environ.get("MAXIT", 4000))
CASES = [(400, 800), (400, 1600), (200, 400), (200, 800), (100, 400)]


class Iso:
    def __init__(self):
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * jnp.sum(x ** 2)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -x, in_axes=(0, None)))


def _pairwise_plain(particles, h=-1):
    """Ba et al.'s convention: h = median(||x-y||^2), no 1/ln K."""
    k = particles.shape[0]
    sq = jnp.sum(particles ** 2, axis=1)
    with jax.default_matmul_precision("highest"):
        L2sq = sq[:, None] + sq[None, :] - 2 * particles @ particles.T
    iu = np.triu_indices(k, k=1)
    med = jnp.median(jnp.clip(L2sq[iu], min=jnp.array(1e-6, dtype=particles.dtype)))
    return L2sq, jnp.where(h <= 0, med, h)


def _std_plain(particles, raw_grad, logp, h=-1):
    L2sq, h = _pairwise_plain(particles, h)
    return M7._combine(particles, raw_grad, jnp.exp(-L2sq / h), h, drift=1.0)


M7.KERNELS["plain_median"] = _std_plain

g = Iso()
res = {}
print(f'{"convention":>14} {"K":>6} {"d":>6} {"gamma=d/K":>10} {"v measured":>11} '
      f'{"Ba et al.":>10} {"ln K / d":>10} {"sec":>6}', flush=True)
for K, d in CASES:
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((K, d))
    gam = d / K
    for lab, kern in (("Ba (h=Med)", "plain_median"), ("msvgd (/lnK)", "standard")):
        t0 = time.time()
        P, _, _ = M7.run_svgd(g, X0, MAXIT, kernel=kern,
                              optimizer=optax.contrib.prodigy, optimizer_kwargs={})
        v = float(np.mean(P.var(0)))
        print(f'{lab:>14} {K:>6} {d:>6} {gam:>10.2f} {v:>11.5f} '
              f'{1/((np.e-1)*gam):>10.5f} {np.log(K)/d:>10.5f} {time.time()-t0:>6.1f}',
              flush=True)
        res[f"{lab}|{K}|{d}"] = dict(v=v, ba=1 / ((np.e - 1) * gam), mine=np.log(K) / d)
        json.dump(res, open("exp13_results.json", "w"), indent=1)
