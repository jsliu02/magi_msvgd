"""
Exp 6: can SVGD's FIXED POINT be made correct, instead of corrected post hoc?

exp04 showed the standard fixed point is collapsed (R->0.03) and the reweighted one sits at
R~0.78. Both are finite-particle biases of a deterministic flow. The textbook cure for a
deterministic flow with the wrong stationary measure is to add the diffusion term that makes
the measure exact: Langevin. So sweep

    x <- x + lr*(SVGD drift/repulsion) + noise_mult*sqrt(2*lr)*z

noise_mult=0 is plain SVGD; noise_mult=1 with the kernel disabled is ULA, whose stationary
distribution IS the target (up to discretization) and which therefore brackets what is
achievable. Run in whitened coordinates, where the target is near-isotropic so a single scalar
step size is appropriate for every direction.

Also sweeps two cheap literature knobs on the same footing:
  IMQ kernel (Gorham & Mackey 2017) -- heavier tails than RBF, the recommended KSD default
  bandwidth multiplier -- the median heuristic is known to be poor in high dimension
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
    s = MSVGD(lambda y, db: m.logdensity(xm + L32 @ y, db), data=m.data)
    K, LR, ITERS = 800, 1e-2, 1500
    out = []

    def combine(y, raw, kind, bw_mult):
        L2sq, h = s.pairwise_distance(y, -1)
        h = h * bw_mult
        if kind == "none":
            return raw
        Kx = jnp.exp(-L2sq / h) if kind == "rbf" else (1.0 + L2sq / h) ** -0.5   # IMQ
        if kind == "imq":
            # grad_x IMQ = -(x-y)/h * (1+d2/h)^-1.5 ; _combine assumes the RBF -2(x-y)/h factor,
            # so pass the IMQ weight matrix scaled to reproduce the correct repulsion
            Kg = (1.0 + L2sq / h) ** -1.5
            dxkxy = (Kg.sum(axis=1, keepdims=True) * y - Kg @ y) * (1.0 / h)
            return (Kx @ raw - dxkxy) / y.shape[0]
        return s._combine(y, raw, Kx, h, 1.0)

    print(H.HDR); print("-" * len(H.HDR))
    grid = ([("none", 0.0, 1.0), ("none", 1.0, 1.0)] +
            [("rbf", nm, 1.0) for nm in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)] +
            [("rbf", 1.0, bw) for bw in (0.1, 10.0, 100.0)] +
            [("imq", nm, 1.0) for nm in (0.0, 1.0)])
    for kind, nm, bw in grid:
        y = jax.random.normal(jax.random.key(0), (K, H.DIM), dtype=jnp.float32)
        key = jax.random.key(100)
        for it in range(ITERS):
            key, sk = jax.random.split(key)
            raw = s.gradient(y, m.data)                      # +score
            drift = combine(y, -raw, kind, bw)               # descent direction
            y = y - LR * drift + nm * np.sqrt(2 * LR) * jax.random.normal(sk, y.shape, y.dtype)
            if not bool(jnp.all(jnp.isfinite(y))):
                break
        tag = f"{kind:>4} noise={nm:<4} bw={bw:g}"
        r = H.evaluate(xm + y @ L32.T, m, tag=tag); out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp06_noise_results")


if __name__ == "__main__":
    main()
