"""
Exp 3: exploit the exp-2 decomposition.

exp02 showed the whole error of N(x_MAP, H^-1) is the MAP-vs-mean displacement, and that the
Hessian covariance is otherwise exact. Meanwhile SVGD's covariance is ruined but its MEAN is
good (bias 0.068 standard, vs 0.335 for the MAP). The two failure modes are complementary, so
combine them. Everything here is NUTS-free: H comes from the model at the MAP, the mean from
SVGD.

  gauss-hybrid   resample N(mean_svgd, H^-1); discards SVGD's shape entirely
  affine-recal   keep the SVGD particles, apply the linear map T with T A T' = H^-1
                 (A = ensemble covariance), preserving whatever non-Gaussian shape SVGD found
  shrink-recal   as above but with eigenvalue flooring, since directions SVGD collapsed to
                 ~0 variance would otherwise have numerical noise amplified by 1/sqrt(eps)

Also re-tests the Gaussianity claim at higher sample resolution, since energy distance at
n=1500 may simply be too blunt to see residual non-Gaussianity.
"""
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H


def sym_pow(A, q, floor_rel=1e-10):
    w, U = np.linalg.eigh(0.5 * (A + A.T))
    w = np.maximum(w, floor_rel * max(w.max(), 1e-300))
    return (U * w ** q) @ U.T


def main():
    H.patch_split()
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    G = H.Gold()
    evc = np.maximum(ev, 1e-8 * ev.max())
    Sig = (V / evc) @ V.T                       # H^-1, the Laplace covariance
    Sig_h = (V / np.sqrt(evc)) @ V.T            # H^-1/2
    rng = np.random.default_rng(0)

    # ---------- resolution check on the Gaussianity claim
    jax.config.update("jax_enable_x64", True)
    m64 = H.build_magi(dtype=jnp.float64)
    print("resolution check -- energy distance vs NUTS at increasing sample size")
    for n in [1500, 4000]:
        e_floor = H._energy_distance(G.whiten(G.pos[rng.choice(len(G.pos), 6000, False)]),
                                     G.whiten(G.ref), rng, n=n)
        s = G.mean[None, :] + rng.standard_normal((6000, H.DIM)) @ Sig_h.T
        e_lap = H._energy_distance(G.whiten(s), G.whiten(G.ref), rng, n=n)
        print(f"    n={n:>5}   NUTS-vs-NUTS floor {e_floor:.4f}    N(mu,H^-1)-vs-NUTS {e_lap:.4f}")

    # ---------- run SVGD variants, then correct them
    jax.config.update("jax_enable_x64", False)
    out = []
    print(); print(H.HDR); print("-" * len(H.HDR))
    for rw in [False, True]:
        for seed in [0, 1, 2]:
            m = H.build_magi()
            m.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
                    optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000,
                    random_seed=seed, monitor_convergence=-1, reweighted_kernel=rw)
            P = np.asarray(m.particles, dtype=np.float64)
            mu = P.mean(axis=0)
            base = f"rw={rw} s{seed}"
            if seed == 0:
                r = H.evaluate(m.particles, m, tag=f"raw SVGD {base}"); out.append(r); H.show(r)

            g = mu[None, :] + rng.standard_normal((800, H.DIM)) @ Sig_h.T
            r = H.evaluate(jnp.asarray(g, jnp.float32), m, tag=f"gauss-hybrid {base}")
            out.append(r); H.show(r)

            A = np.cov(P, rowvar=False)
            for tag, fl in [("affine-recal", 1e-10), ("shrink-recal", 1e-4)]:
                T = Sig_h @ sym_pow(A, -0.5, floor_rel=fl)
                Q = mu[None, :] + (P - mu) @ T.T
                r = H.evaluate(jnp.asarray(Q, jnp.float32), m, tag=f"{tag} {base}")
                out.append(r); H.show(r)
    H.save(out, "exp03_hybrid_results")


if __name__ == "__main__":
    main()
