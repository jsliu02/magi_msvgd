"""
exp05c: is the missing attractor caused by anisotropy or by non-Gaussianity?

Section 2 explained the absence of a stable fixed point on the MAGI posteriors by anisotropy: one
scalar bandwidth cannot hold directions whose scales differ by orders of magnitude, whereas on
N(0, I) it can (exp02d). exp05 was built to confirm that by removing the anisotropy with an
exact-Hessian preconditioner, and it diverged. exp05b then showed the preconditioner is not at
fault -- whitening takes cond(Sigma) from 7.9e3 to 47 on fn and from 5.6e10 to 1.6 on hiv, with
the 5-95% eigenvalue range at 0.82-1.22 -- so the whitened target really is isotropic in its
second moment and the divergence has some other cause.

That leaves the explanation for section 2 unconfirmed. This settles it with a two-cell experiment
that changes exactly one thing at a time, on exact Gaussians where there is no non-Gaussianity to
confound anything:

  N(0, I)                  isotropic   -- exp02d says the attractor is stable
  N(0, Sigma_ref)          anisotropic, cond 7.9e3 (fn) / 5.6e10 (hiv), same dimension

Same fixed bandwidths, same starts, same length. If the anisotropic Gaussian's attractor decays
like the MAGI posteriors', anisotropy is the cause. If it is stable like the isotropic one's,
anisotropy is ruled out and the cause is non-Gaussianity.

Bandwidth is set from the target's own median heuristic so the two cells are comparable:
h* = median E||x-y||^2 / ln K, computed analytically as 2*tr(Sigma)/ln K.
"""
import numpy as np, jax, jax.numpy as jnp, optax, os, json, time
jax.config.update("jax_enable_x64", True)
import harness8 as H
import msvgd8 as M7

NAME = os.environ.get("NAME", "fn")
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 200000))
MULTS = [float(x) for x in os.environ.get("MULTS", "3,10,30").split(",")]
CHECK = [int(x) for x in os.environ.get("CHECK", "2000,20000,100000,200000").split(",")]


class Gauss:
    def __init__(self, cov):
        d = cov.shape[0]
        self.cov = cov
        self.P = jnp.asarray(np.linalg.inv(cov + 1e-12 * np.trace(cov) / d * np.eye(d)))
        self.mu = jnp.zeros((1,))
        self.data = None
        self.logdensity = lambda x, data: -0.5 * x @ (self.P @ x)
        self.gradient = jax.jit(jax.vmap(lambda x, data: -(self.P @ x), in_axes=(0, None)))


S = H.Scorer(NAME)
d = S.mean.shape[0]
Sig = 0.5 * (S.cov + S.cov.T)
w = np.linalg.eigvalsh(Sig)
print(f'{NAME}: d={d} K={K}  cond(Sigma_ref)={w.max()/max(w.min(),1e-300):.3e}  '
      f'checkpoints {CHECK}', flush=True)
print(f'{"target":>22} {"h/h*":>6} {"start":>9}   mean variance ratio over the checkpoints',
      flush=True)

for lab, C in (("N(0, I)", np.eye(d)), ("N(0, Sigma_ref)", Sig)):
    g = Gauss(C)
    Ch = np.linalg.cholesky(C + 1e-12 * np.trace(C) / d * np.eye(d))
    Ci = np.linalg.inv(Ch)
    hstar = 2.0 * np.trace(C) / np.log(K)
    for mult in MULTS:
        for slab, sc in (("correct", 1.0), ("narrow4x", 0.25)):
            rng = np.random.default_rng(0)
            X0 = sc * (rng.standard_normal((K, d)) @ Ch.T)
            t0 = time.time()
            P, _, hist = M7.run_svgd(g, X0, MAXIT, kernel="standard", bandwidth=mult * hstar,
                                     optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                     record_every=min(CHECK))
            hd = dict(hist)
            vals = [float(np.mean(((hd[c]) @ Ci.T).var(0))) for c in CHECK if c in hd]
            print(f'{lab:>22} {mult:>6.0f} {slab:>9}   '
                  + " ".join(f'{v:>8.4f}' for v in vals) + f'   ({time.time()-t0:.0f}s)',
                  flush=True)
