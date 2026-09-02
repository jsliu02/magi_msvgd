"""
Exp 1: preconditioning by the full dense Laplace metric.

investigation.md diagnosed anisotropic collapse (NUTS cov condition number 2e4) and found a
DIAGONAL empirical-Fisher preconditioner only partly helps. At d=325 the full dense Hessian is
cheap, so we can precondition exactly instead of approximately.

Rather than build a matrix-valued kernel, reparameterize: x = x_MAP + L y with H^-1 = L L^T,
H = -grad^2 log p at the MAP. In y-coordinates the target is approximately N(0, I), so a
scalar-bandwidth RBF kernel with the median heuristic is appropriate -- the assumption SVGD
actually needs. No change to msvgd.py: just a transformed logdensity handed to plain MSVGD.

Also scores the Laplace approximation itself, which is the natural control: if the posterior is
near-Gaussian, N(x_MAP, H^-1) is a strong baseline that any SVGD variant should beat.
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD

CACHE = os.path.join(H.HERE, "laplace_cache.npz")


def get_laplace(force=False):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE)
        return z["x_map"], z["evals"], z["evecs"]
    jax.config.update("jax_enable_x64", True)
    m = H.build_magi(dtype=jnp.float64)
    t0 = time.time()
    m.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=30000, atol=1e-7, rtol=0.0,
            random_seed=0, monitor_convergence=-1,
            optimizer=optax.contrib.prodigy, optimizer_kwargs={})
    x_map = np.asarray(m.particles[0], dtype=np.float64)
    print(f"  MAP found in {time.time()-t0:.1f}s   logp={float(m.magi_logdensity(m.particles[0])):.3f}"
          f"   theta={np.round(x_map[:3],4)}")
    t0 = time.time()
    Hess = -np.asarray(jax.hessian(m.magi_logdensity)(jnp.asarray(x_map)), dtype=np.float64)
    Hess = 0.5 * (Hess + Hess.T)
    ev, V = np.linalg.eigh(Hess)
    print(f"  Hessian {Hess.shape} in {time.time()-t0:.1f}s   eig range [{ev.min():.4e}, {ev.max():.4e}]"
          f"   n_negative={int((ev<=0).sum())}   cond={ev.max()/max(ev.min(),1e-30):.3e}")
    np.savez(CACHE, x_map=x_map, evals=ev, evecs=V)
    return x_map, ev, V


def main():
    H.patch_split()
    x_map, ev, V = get_laplace()
    G = H.Gold()
    ridge = 1e-8 * ev.max()
    evc = np.maximum(ev, ridge)
    L = (V / np.sqrt(evc)) @ V.T                     # symmetric sqrt of H^-1
    Linv = (V * np.sqrt(evc)) @ V.T
    print(f"  Laplace vs NUTS: |mean diff| (whitened) = "
          f"{np.sqrt((G.whiten(x_map[None,:])**2).mean()):.3f}")

    results = []
    rng = np.random.default_rng(0)

    # --- control: the Laplace approximation itself
    for k in [800]:
        samp = x_map[None, :] + rng.standard_normal((k, H.DIM)) @ L
        jax.config.update("jax_enable_x64", True)
        m64 = H.build_magi(dtype=jnp.float64)
        r = H.evaluate(jnp.asarray(samp), m64, tag=f"Laplace N(MAP,H^-1) k={k}")
        results.append(r); H.show(r)

    # --- SVGD in whitened coordinates
    jax.config.update("jax_enable_x64", False)
    m = H.build_magi(dtype=jnp.float32)
    L32, Linv32 = jnp.asarray(L, jnp.float32), jnp.asarray(Linv, jnp.float32)
    xmap32 = jnp.asarray(x_map, jnp.float32)

    def ld_y(y, data_batch):
        return m.logdensity(xmap32 + L32 @ y, data_batch)

    y_center = (jnp.asarray(m.particles_init, jnp.float32) - xmap32) @ Linv32

    for rw in [False, True]:
        for seed in [0, 1, 2]:
            s = MSVGD(ld_y, data=m.data)
            y0 = y_center + jax.random.normal(jax.random.key(seed), (200, H.DIM),
                                              dtype=jnp.float32) * 0.01
            t0 = time.time()
            y = s.solve(y0, k_schedule=800, random_seed=seed, monitor_convergence=-1,
                        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                        atol=0.0, rtol=0.0, max_iter=1000, reweighted_kernel=rw)
            x = xmap32 + y @ L32.T
            r = H.evaluate(x, m, tag=f"whitened SVGD rw={rw} s{seed}")
            r["elapsed"] = time.time() - t0
            results.append(r); H.show(r)

    H.save(results, "exp01_whiten_results")


if __name__ == "__main__":
    print(H.HDR); print("-" * len(H.HDR))
    main()
