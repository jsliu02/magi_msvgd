"""
Exp 12: blockwise-kernel SVGD, and particle-count scaling.

The high-dimensional failure of the RBF kernel is a concentration effect: in 325 dimensions
pairwise distances concentrate, the Gram matrix degenerates toward the identity, and particles
stop interacting. exp08 attacked that with heavier tails (IMQ). The other standard escape is to
never apply a kernel in 325 dimensions at all -- apply it to low-dimensional coordinate blocks
and update each block with its own kernel (the message-passing / graphical SVGD idea, Zhuo et
al. 2018). MAGI is a natural fit: the ODE couples adjacent time points locally, and the GP
prior, though dense, is a Gauss-Markov field.

Sweeps block size B over the divisors of 325 (1, 5, 13, 25, 65, 325). B=325 recovers ordinary
SVGD, so the sweep contains its own control. Run in both x-space and whitened coordinates to
separate the block effect from the anisotropy effect.

Also answers a question left open in investigation.md: does SVGD's collapse simply go away with
more particles?
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H
from msvgd import MSVGD


def blocked_flow(y, score, B):
    k, d = y.shape
    nb = d // B
    Yb = y.reshape(k, nb, B).transpose(1, 0, 2)
    Sb = score.reshape(k, nb, B).transpose(1, 0, 2)

    def per_block(yb, sb):
        sq = (yb ** 2).sum(-1)
        L2 = jnp.maximum(sq[:, None] + sq[None, :] - 2 * yb @ yb.T, 0.0)
        h = jnp.maximum(jnp.median(L2) / jnp.log(k), 1e-10)
        K = jnp.exp(-L2 / h)
        dx = (K.sum(1, keepdims=True) * yb - K @ yb) * (2.0 / h)
        return (K @ sb + dx) / k
    return jax.vmap(per_block)(Yb, Sb).transpose(1, 0, 2).reshape(k, d)


def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float32)
    xm = jnp.asarray(x_map, jnp.float32)
    m = H.build_magi()
    out = []
    print(H.HDR); print("-" * len(H.HDR))

    for coord in ["whitened", "x-space"]:
        Amat = Lw if coord == "whitened" else jnp.eye(H.DIM, dtype=jnp.float32)
        lr = 1e-2 if coord == "whitened" else 1e-4
        s = MSVGD(lambda y, db: m.logdensity(xm + Amat @ y, db), data=m.data)
        step = jax.jit(lambda y, B: y + lr * blocked_flow(y, s.gradient(y, m.data), B),
                       static_argnums=1)
        for B in [1, 5, 13, 25, 65, 325]:
            y = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
            ok = True
            for it in range(1, 2001):
                y = step(y, B)
                if it % 250 == 0 and not bool(jnp.all(jnp.isfinite(y))):
                    ok = False; break
            if not ok:
                print(f'{coord} block B={B:<4}  DIVERGED'); continue
            r = H.evaluate(xm + y @ Amat.T, m, tag=f"{coord} block B={B}")
            out.append(r); H.show(r)
        print()

    # ---- particle-count scaling of ordinary x-space SVGD
    print("particle-count scaling, ordinary x-space SVGD (2000 iters, lr 1e-4)")
    s = MSVGD(lambda y, db: m.logdensity(y, db), data=m.data)
    for k in [200, 800, 3200]:
        y = xm + jax.random.normal(jax.random.key(0), (k, H.DIM), dtype=jnp.float32) @ Lw.T
        for it in range(2000):
            y = y - 1e-4 * s._svgd_update(y, -s.gradient(y, m.data), -1)
        r = H.evaluate(y, m, tag=f"x-space SVGD k={k}")
        out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp12_blocked_results")


if __name__ == "__main__":
    main()
