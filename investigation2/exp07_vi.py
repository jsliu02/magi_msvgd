"""
Exp 7: fix the mean directly with Gaussian variational inference.

exp02 established that the ONLY defect of N(x_MAP, H^-1) is that the MAP is not the posterior
mean, and exp05 got the residual error down to bias 0.068 by borrowing SVGD's mean. But SVGD is
an expensive way to obtain one vector. The KL-optimal Gaussian mean satisfies E_q[grad log p]=0,
which is a cheap fixed point needing only gradients:

    mu <- mu + Sigma * mean_i grad log p(x_i),     x_i ~ N(mu, Sigma)

This is natural-gradient ascent on the ELBO in mu (the natural gradient for the mean of a
Gaussian with fixed covariance is exactly Sigma * E[grad log p]). No NUTS, no SVGD, no Hessian
beyond the one already computed.

Also refines the covariance by averaging Hessians over the current q, as a check on exp02's
claim that the Hessian AT THE MAP is already the right covariance -- if averaging changes
nothing, the posterior really is Gaussian over its bulk.
"""
import numpy as np, jax, jax.numpy as jnp, time
import harness as H


def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    Sig = (V / evc) @ V.T
    Sig_h = (V / np.sqrt(evc)) @ V.T
    G = H.Gold()
    jax.config.update("jax_enable_x64", True)
    m = H.build_magi(dtype=jnp.float64)
    grad = jax.jit(jax.vmap(lambda x: m.logdensity(x, m.data), out_axes=0))
    gfn = jax.jit(lambda X: m.gradient(X, m.data))
    rng = np.random.default_rng(0)
    out = []

    def bias_of(mu):
        return float(np.sqrt((G.whiten(mu[None, :]) ** 2).mean()))

    # ---------------- mean-only natural-gradient VI
    print("Gaussian VI on the mean (Sigma fixed at H^-1)")
    print(f'{"iter":>5} {"n_samp":>7} {"|E[grad]| (natural)":>20} {"bias":>8}   (MAP bias '
          f'{bias_of(x_map):.4f}, SVGD ~0.068, floor 0.034)')
    for n_samp in [500, 4000]:
        mu = x_map.copy()
        damp = 0.25                       # undamped Newton overshoots into the cubic ODE term
        for it in range(1, 201):
            S = mu[None, :] + rng.standard_normal((n_samp, H.DIM)) @ Sig_h.T
            g = np.asarray(gfn(jnp.asarray(S))).mean(axis=0)
            if not np.all(np.isfinite(g)):
                print(f'{it:>5} {n_samp:>7}   non-finite gradient, stopping'); break
            step = damp * (Sig @ g)
            mu = mu + step
            if it in (1, 5, 20, 50, 100, 200):
                print(f'{it:>5} {n_samp:>7} {np.linalg.norm(step):>20.5f} {bias_of(mu):>8.4f}')
        np.save(f"vi_mean_n{n_samp}.npy", mu)
        print()

    # ---------------- does averaging the Hessian over q change the covariance?
    print("covariance check: Hessian at MAP vs averaged over q")
    hess = jax.jit(jax.hessian(lambda x: m.logdensity(x, m.data)))
    mu = np.load("vi_mean_n4000.npy")
    S = mu[None, :] + rng.standard_normal((12, H.DIM)) @ Sig_h.T
    t0 = time.time()
    Hbar = -np.mean([np.asarray(hess(jnp.asarray(x))) for x in S], axis=0)
    Hbar = 0.5 * (Hbar + Hbar.T)
    Hmap = (V * ev) @ V.T
    rel = np.linalg.norm(Hbar - Hmap) / np.linalg.norm(Hmap)
    print(f"  12 Hessians in {time.time()-t0:.1f}s;  ||Hbar - H_MAP||/||H_MAP|| = {rel:.4f}")
    ew, EV = np.linalg.eigh(Hbar); ew = np.maximum(ew, 1e-8 * ew.max())
    Sbar_h = (EV / np.sqrt(ew)) @ EV.T

    # ---------------- score the resulting Gaussians
    print(); print(H.HDR); print("-" * len(H.HDR))
    for tag, mu_, A in [("N(MAP, H^-1)", x_map, Sig_h),
                        ("N(VI-mean n=500,  H^-1)", np.load("vi_mean_n500.npy"), Sig_h),
                        ("N(VI-mean n=4000, H^-1)", np.load("vi_mean_n4000.npy"), Sig_h),
                        ("N(VI-mean, Hbar^-1)", np.load("vi_mean_n4000.npy"), Sbar_h)]:
        s = mu_[None, :] + rng.standard_normal((800, H.DIM)) @ A.T
        r = H.evaluate(jnp.asarray(s), m, tag=tag); out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp07_vi_results")


if __name__ == "__main__":
    main()
