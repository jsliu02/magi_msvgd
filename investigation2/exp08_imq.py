"""
Exp 8: (a) calibrate the energy metric, (b) ablate the exp-6 IMQ result.

exp06 found IMQ at energy 0.051 vs RBF at 1.359 under identical settings, which is a large
enough claim to need checking. Two confounds to rule out: RBF may simply be SLOWER rather than
worse, and the win may come from the whitening rather than the kernel. So cross kernel x
coordinate system x iteration budget x seed.

The calibration matters for reading every table in this investigation: energy distance is only
meaningful relative to what it can resolve, so measure it against Gaussians deliberately
mis-scaled and mis-centred by known amounts.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H
from msvgd import MSVGD


def make_flow(s, kind):
    def flow(y, raw_neg):                       # raw_neg = -score
        L2sq, h = s.pairwise_distance(y, -1)
        k = y.shape[0]
        if kind == "rbf":
            return s._combine(y, raw_neg, jnp.exp(-L2sq / h), h, 1.0)
        Kx = (1.0 + L2sq / h) ** -0.5
        Kg = (1.0 + L2sq / h) ** -1.5
        dxkxy = (Kg.sum(axis=1, keepdims=True) * y - Kg @ y) * (1.0 / h)
        return (Kx @ raw_neg - dxkxy) / k
    return flow


def main():
    G = H.Gold()
    rng = np.random.default_rng(0)
    jax.config.update("jax_enable_x64", True)
    m64 = H.build_magi(dtype=jnp.float64)
    Cn = np.linalg.cholesky(G.cov + 1e-12 * np.eye(H.DIM))

    print("energy-metric calibration (800 samples, exact Gaussian at NUTS moments)")
    print(f'{"perturbation":>34} {"energy":>8} {"R_glob":>8} {"dev":>6}')
    for tag, sc, sh in [("none (= floor)", 1.00, 0.0), ("sd x0.97", 0.97, 0.0),
                        ("sd x0.94", 0.94, 0.0), ("sd x0.90", 0.90, 0.0),
                        ("sd x0.80", 0.80, 0.0), ("mean shift 0.10 sd", 1.00, 0.10),
                        ("mean shift 0.33 sd", 1.00, 0.33)]:
        s = G.mean[None, :] + sh * np.sqrt(G.evals).mean() * 0 + \
            sc * (rng.standard_normal((800, H.DIM)) @ Cn.T)
        if sh:
            s = s + sh * np.sqrt(np.diag(G.cov))[None, :]
        r = H.evaluate(jnp.asarray(s), m64, tag="cal")
        print(f'{tag:>34} {r["energy"]:>8.4f} {r["R_global"]:>8.3f} {r["width_dev"]:>6.1f}')

    # ---------------- kernel x coordinates x budget
    jax.config.update("jax_enable_x64", False)
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float32)
    xm = jnp.asarray(x_map, jnp.float32)
    m = H.build_magi()
    out = []
    print(); print(H.HDR); print("-" * len(H.HDR))

    for coord in ["whitened", "x-space"]:
        if coord == "whitened":
            s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)
            back = lambda y: xm + y @ Lw.T
            init = lambda key: jax.random.normal(key, (800, H.DIM), dtype=jnp.float32)
            lr = 1e-2
        else:
            s = MSVGD(lambda y, db: m.logdensity(y, db), data=m.data)
            back = lambda y: y
            init = lambda key: (xm + jax.random.normal(key, (800, H.DIM), dtype=jnp.float32)
                                @ Lw.T)          # same distribution, x coordinates
            lr = 1e-4
        for kind in ["rbf", "imq"]:
            flow = make_flow(s, kind)
            for seed in [0, 1]:
                y = init(jax.random.key(seed))
                for it in range(1, 4001):
                    y = y - lr * flow(y, -s.gradient(y, m.data))
                    if it in (500, 1500, 4000):
                        if not bool(jnp.all(jnp.isfinite(y))):
                            print(f'{coord} {kind} s{seed} it={it}: NON-FINITE'); break
                        if seed == 0 or it == 4000:
                            r = H.evaluate(back(y), m, tag=f"{coord} {kind} s{seed} it={it}")
                            out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp08_imq_results")


if __name__ == "__main__":
    main()
