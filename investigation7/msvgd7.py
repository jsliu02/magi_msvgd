"""
Kernel variants for SVGD, kept OUTSIDE the msvgd package (which is settled and must not be
edited). Three kernels:

  standard    the shipped RBF joint kernel, MSVGD._svgd_update. Reimplemented here rather than
              called, so that all four variants share one driver and one convergence rule;
              exp01 checks it reproduces MSVGD.solve to round-off.
  reweighted  the density-reweighted kernel of Huang, Dong & Fang (2023), Eq. 24,
              k(x,y) = p(x)^-1/2 k_rbf(x,y) p(y)^-1/2. Recovered verbatim from
              `git show HEAD:msvgd/msvgd.py` in the msvgd repo, where it was removed in the
              recent refactor. Historically the better performer, so it has to be in the
              comparison.
  matrix      Wang, Tang, Bajaj & Liu (NeurIPS 2019) Eq. 12-15 with Q the diagonal empirical
              Fisher normalised to mean 1: K_Q(x,y) = Q^-1 exp(-||x-y||^2_Q / h). Q cancels in
              the repulsion and survives as an elementwise factor on the drift.
  precond    SVGD run in coordinates whitened by a FIXED metric supplied by the caller
              (the exact Hessian at the MAP). y = L^-1 (x - x0) with Sigma = L L^T; this is
              exactly the constant matrix-valued kernel K(x,y) = Sigma k_Sigma(x,y), and it is
              the strongest form of the anisotropy fix, since the metric is the true local
              posterior covariance rather than a diagonal estimate.

The driver runs a FIXED iteration count (no tolerance test) because a shared absolute tolerance
reports false convergence after ~1 iteration once a kernel rescales the gradient.
"""
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax
from functools import partial

RIDGE = 1e-6


def _pairwise(particles, h=-1):
    k = particles.shape[0]
    sq = jnp.sum(particles ** 2, axis=1)
    with jax.default_matmul_precision("highest"):
        L2sq = sq[:, None] + sq[None, :] - 2 * particles @ particles.T
    iu = np.triu_indices(k, k=1)
    med = jnp.median(jnp.clip(L2sq[iu], min=jnp.array(1e-6, dtype=particles.dtype)))
    return L2sq, jnp.where(h <= 0, med / jnp.log(jnp.array(k, dtype=particles.dtype)), h)


def _combine(particles, raw_grad, K, h, drift):
    k = particles.shape[0]
    dxkxy = (K.sum(axis=1, keepdims=True) * particles - K @ particles) * (2.0 / h)
    return (drift * (K @ raw_grad) - dxkxy) / k


def _std_update(particles, raw_grad, logp, h=-1):
    L2sq, h = _pairwise(particles, h)
    return _combine(particles, raw_grad, jnp.exp(-L2sq / h), h, drift=1.0)


def _rw_update(particles, raw_grad, logp, h=-1, clip_exponent=20.0):
    L2sq, h = _pairwise(particles, h)
    ld = logp - jnp.max(logp)
    reweight = jnp.exp(jnp.clip(-0.5 * (ld[:, None] + ld[None, :]), max=clip_exponent))
    return _combine(particles, raw_grad, reweight * jnp.exp(-L2sq / h), h, drift=0.5)


def _mat_update(particles, raw_grad, logp, h=-1):
    Qd = jnp.mean(raw_grad ** 2, axis=0)
    Qd = Qd + RIDGE * jnp.mean(Qd)
    Qd = Qd / jnp.mean(Qd)
    L2sq, hh = _pairwise(particles * jnp.sqrt(Qd), h)
    return _combine(particles, raw_grad, jnp.exp(-L2sq / hh), hh, drift=1.0 / Qd)


KERNELS = {"standard": _std_update, "reweighted": _rw_update, "matrix": _mat_update}


def run_svgd(m, X0, iters, kernel="standard", optimizer=optax.contrib.prodigy,
             optimizer_kwargs=None, bandwidth=-1.0, precond=None, is_MAP=False,
             record_every=0, seed=0):
    """
    Run `iters` SVGD steps from X0.

    precond : None, or (x0, L) with Sigma = L L^T. When given, the whole run happens in
              y = L^-1 (x - x0) and the returned particles are mapped back. Note the KERNEL is
              then also computed in y, which is the point.
    record_every : if > 0, also return the particle history every this many iterations.
    """
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    dt = m.mu.dtype
    kfn = KERNELS[kernel]
    data = m.data

    if precond is None:
        logp_v = jax.vmap(lambda x: m.logdensity(x, data))
        grad_v = m.gradient
        Y0 = jnp.asarray(X0, dt)
        fwd = lambda Y: Y
    else:
        x0, L = precond
        x0j = jnp.asarray(x0, dt); Lj = jnp.asarray(L, dt)
        Linv = jnp.asarray(np.linalg.inv(np.asarray(L, np.float64)), dt)
        logp_v = jax.vmap(lambda y: m.logdensity(x0j + Lj @ y, data))
        grad_v = jax.jit(jax.vmap(
            lambda y, d: Lj.T @ jax.grad(lambda z: m.logdensity(x0j + Lj @ z, d))(y),
            in_axes=(0, None)))
        Y0 = ((jnp.asarray(X0, dt) - x0j) @ Linv.T)
        fwd = lambda Y: x0j + Y @ Lj.T

    opt = optimizer(**optimizer_kwargs)

    @jax.jit
    def step(carry, _):
        Y, state = carry
        g = grad_v(Y, data)
        raw = -g
        R = -jnp.sum((Y - Y.mean(0)) * g) / Y.size
        if is_MAP:
            upd_in = raw
        else:
            upd_in = kfn(Y, raw, logp_v(Y), bandwidth)
        updates, state = jax.vmap(opt.update)(upd_in, state, Y)
        Y = optax.apply_updates(Y, updates)
        return (Y, state), R

    state = jax.vmap(opt.init)(Y0)
    Y = Y0
    hist = []
    if record_every > 0:
        Rs = []
        done = 0
        while done < iters:
            n = min(record_every, iters - done)
            (Y, state), R = jax.lax.scan(step, (Y, state), None, length=n)
            Rs.append(np.asarray(R, np.float64))
            done += n
            hist.append((done, np.asarray(fwd(Y), np.float64)))
        return np.asarray(fwd(Y), np.float64), np.concatenate(Rs), hist
    (Y, state), R = jax.lax.scan(step, (Y, state), None, length=iters)
    return np.asarray(fwd(Y), np.float64), np.asarray(R, np.float64), hist


def laplace_metric(m, floor=1e-10):
    """(x_map, L) with L L^T = H^-1, Jacobi-stabilised. Same construction as refs5.metric."""
    x = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
    dg = np.sqrt(np.maximum(np.abs(np.diag(H)), np.finfo(float).tiny))
    Hs = H / np.outer(dg, dg); Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs)
    w = np.maximum(w, floor * max(w.max(), 1.0))
    W = (V / np.sqrt(w)) @ V.T
    return x, W / dg[:, None]
