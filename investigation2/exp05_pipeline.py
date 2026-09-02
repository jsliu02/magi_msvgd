"""
Exp 5: build and test corrected pipelines.

Three ingredients discovered so far:
  * H^-1 (Hessian at MAP) is the exact posterior covariance (exp02).
  * SVGD's mean is good even when its covariance is destroyed (exp03).
  * In WHITENED coordinates the reweighted kernel has a stable fixed point at R~0.78 with a
    FLAT spectral profile (exp04) -- uniform deficiency, which a scalar inflation corrects
    exactly. In x-space the profile is not flat, so scalar inflation would be wrong there;
    included anyway to demonstrate that.

Corrections tested:
  stein-inflate   x -> mu + (x-mu)/sqrt(R). Makes R=1 by construction; only VALID when the
                  profile is flat, otherwise it overshoots soft directions to fix stiff ones.
  IS              self-normalized importance sampling against the exact target using
                  N(mu, H^-1) as proposal. Asymptotically exact, and reports its own ESS, so
                  unlike everything else here it is self-certifying.
"""
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD


def load_lap():
    z = np.load("laplace_cache.npz")
    x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    return x_map, evc, V, (V / np.sqrt(evc)) @ V.T, (V * np.sqrt(evc)) @ V.T


def inflate(P, R):
    mu = P.mean(axis=0)
    return mu + (P - mu) / np.sqrt(max(R, 1e-8))


def run_whitened(m, xm, L32, rw, seed, iters=400, k=800):
    s = MSVGD(lambda y, db: m.logdensity(xm + L32 @ y, db), data=m.data)
    y = jax.random.normal(jax.random.key(seed), (k, H.DIM), dtype=jnp.float32)
    opt = optax.adam(1e-2); st = jax.vmap(opt.init)(y)
    for _ in range(iters):
        raw = -s.gradient(y, m.data)
        upd = (s._reweighted_svgd_update(y, raw, m.data, -1)[0] if rw else s._svgd_update(y, raw, -1))
        du, st = jax.vmap(opt.update)(upd, st, y)
        y = optax.apply_updates(y, du)
    return xm + y @ L32.T


def main():
    H.patch_split()
    x_map, evc, V, Sig_h, _ = load_lap()
    logdet_q = -float(np.sum(np.log(evc)))          # log|H^-1|
    m = H.build_magi()
    L32, xm = jnp.asarray(Sig_h, jnp.float32), jnp.asarray(x_map, jnp.float32)
    rng = np.random.default_rng(0)
    out = []
    print(H.HDR); print("-" * len(H.HDR))

    # ---------- x-space standard SVGD: best mean source
    means_x = []
    for seed in [0, 1, 2]:
        mm = H.build_magi()
        mm.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
                 optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=seed,
                 monitor_convergence=-1, reweighted_kernel=False)
        means_x.append(np.asarray(mm.particles, np.float64).mean(axis=0))

    # ---------- whitened reweighted SVGD, raw / inflated / mean-swapped
    for seed in [0, 1, 2]:
        X = np.asarray(run_whitened(m, xm, L32, True, seed), np.float64)
        r = H.evaluate(jnp.asarray(X, jnp.float32), m, tag=f"whitened rw s{seed}")
        out.append(r); H.show(r)
        Xi = inflate(X, r["R_global"])
        r2 = H.evaluate(jnp.asarray(Xi, jnp.float32), m, tag=f"  +stein-inflate s{seed}")
        out.append(r2); H.show(r2)
        Xm = Xi - X.mean(axis=0) + means_x[seed]
        r3 = H.evaluate(jnp.asarray(Xm, jnp.float32), m, tag=f"  +inflate+SVGDmean s{seed}")
        out.append(r3); H.show(r3)

    # ---------- counter-demonstration: scalar inflation on the non-flat x-space ensemble
    mm = H.build_magi()
    mm.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
             optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=0,
             monitor_convergence=-1, reweighted_kernel=True)
    Px = np.asarray(mm.particles, np.float64)
    r = H.evaluate(mm.particles, mm, tag="x-space rw (non-flat)")
    out.append(r); H.show(r)
    r = H.evaluate(jnp.asarray(inflate(Px, r["R_global"]), jnp.float32), mm,
                   tag="  +stein-inflate (wrong)")
    out.append(r); H.show(r)

    # ---------- gauss-hybrid + importance sampling correction
    jax.config.update("jax_enable_x64", True)
    m64 = H.build_magi(dtype=jnp.float64)
    ldf = jax.jit(jax.vmap(lambda x: m64.logdensity(x, m64.data)))
    for seed, tag in [(0, "gauss-hybrid s0"), (1, "gauss-hybrid s1"), (2, "gauss-hybrid s2")]:
        mu = means_x[seed]
        for n_prop, label in [(800, ""), (20000, " (IS n=20k)")]:
            S = mu[None, :] + rng.standard_normal((n_prop, H.DIM)) @ Sig_h.T
            if n_prop == 800:
                r = H.evaluate(jnp.asarray(S), m64, tag=tag); out.append(r); H.show(r)
                continue
            d = S - mu
            logq = -0.5 * (np.einsum("ij,jk,ik->i", d, (V * evc) @ V.T, d) + logdet_q)
            logw = np.asarray(ldf(jnp.asarray(S))) - logq
            logw -= logw.max()
            w = np.exp(logw); w /= w.sum()
            ess = 1.0 / np.sum(w ** 2)
            idx = rng.choice(n_prop, 800, replace=True, p=w)
            r = H.evaluate(jnp.asarray(S[idx]), m64, tag=f"{tag}{label}")
            r["ESS"] = float(ess); out.append(r); H.show(r)
            print(f'{"":>26}   ESS = {ess:.1f} / {n_prop}  ({100*ess/n_prop:.1f}%)   '
                  f'IS mean bias = {np.sqrt((H.Gold().whiten((w[:,None]*S).sum(0)[None,:])**2).mean()):.4f}')
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp05_pipeline_results")


if __name__ == "__main__":
    main()
