"""
Exp 4: does SVGD DESTROY a correct ensemble?

Every previous run initialized SVGD far from the posterior, so "SVGD is underdispersed" could
mean either (a) its fixed point is underdispersed, or (b) it simply never gets there. This
distinguishes them: initialize at N(0,I) in Laplace-whitened coordinates -- i.e. exact Laplace
samples, which exp02/03 showed are statistically indistinguishable from NUTS -- and watch what
the dynamics do to a starting ensemble that is already correct.

Whitened coordinates also give SVGD its best possible shot: the target is near-isotropic there,
which is the regime a scalar-bandwidth RBF kernel is actually designed for.

If R decays from 1, the fixed point is the problem and no initialization or kernel tweak saves
it. If R holds near 1, the problem was optimization/initialization and is fixable.
"""
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD


def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    L = (V / np.sqrt(evc)) @ V.T
    m = H.build_magi()
    L32, xm = jnp.asarray(L, jnp.float32), jnp.asarray(x_map, jnp.float32)

    def ld_y(y, db):
        return m.logdensity(xm + L32 @ y, db)

    s = MSVGD(ld_y, data=m.data)
    K = 800
    out = []
    print(H.HDR); print("-" * len(H.HDR))

    for opt_name, opt in [("adam1e-2", optax.adam(1e-2)), ("adam1e-3", optax.adam(1e-3)),
                          ("sgd1e-2", optax.sgd(1e-2))]:
        for rw in [False, True]:
            y = jax.random.normal(jax.random.key(0), (K, H.DIM), dtype=jnp.float32)  # exact Laplace
            st = jax.vmap(opt.init)(y)
            tag0 = f"{opt_name} rw={rw}"
            for it in range(0, 1001):
                if it in (0, 50, 200, 500, 1000):
                    r = H.evaluate(xm + y @ L32.T, m, tag=f"{tag0} it={it}")
                    r["iter"] = it; out.append(r); H.show(r)
                    if r.get("failed"): break
                raw = -s.gradient(y, m.data)
                upd = (s._reweighted_svgd_update(y, raw, m.data, -1)[0] if rw
                       else s._svgd_update(y, raw, -1))
                du, st = jax.vmap(opt.update)(upd, st, y)
                y = optax.apply_updates(y, du)
            print()
    H.save(out, "exp04_from_laplace_results")


if __name__ == "__main__":
    main()
