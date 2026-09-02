"""
Exp 25: safer and/or faster alternatives to whitened MALA.

MALA is a random walk with a first-order integrator. Two structurally different alternatives:

  HMC   -- L leapfrog steps per proposal. In whitened coordinates the target is near-isotropic,
           which is exactly HMC's best case: for an exact Gaussian with identity mass matrix a
           trajectory of length eps*L ~ pi/2 returns a near-independent draw. Optimal scaling in
           dimension is also better (d^-1/4 vs MALA's d^-1/3). Cost is L gradients per proposal,
           so the comparison must be per gradient evaluation, not per proposal.
  pCN   -- proposal y' = sqrt(1-rho^2) y + rho*xi preserves N(0,I) exactly, so it is
           UNCONDITIONALLY stable: no step-size restriction, no transience, ever. It also needs
           no gradients. Acceptance is governed by the spread of r = log p - log N, measured at
           sd(r) ~ 14-20 nats here, so rho must be small and mixing will be slow. Safety at the
           cost of speed.

Everything is scored on the same footing: energy/varwtd against the gold standard, and cost in
gradient (or density) evaluations per chain.
"""
import numpy as np, jax, jax.numpy as jnp, time
from functools import partial
import harness as H

G = H.Gold()
z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
evc = np.maximum(ev, 1e-8 * ev.max())
Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float32)
xm = jnp.asarray(x_map, jnp.float32)
m = H.build_magi()

def score(X, tag, cost, secs):
    r = H.evaluate(X, m, tag=tag)
    pr = ((np.asarray(X, np.float64) - G.mean) @ G.evecs).var(0) / G.evals
    r["varwtd"] = float(np.sum(pr * G.evals) / np.sum(G.evals)); r["cost"] = cost; r["t"] = secs
    return r

logp_and_grad = lambda y: m.value_and_gradient(xm + y @ Lw, m.data)

@partial(jax.jit, static_argnums=(3, 4))
def hmc(y, key, eps, L, n_iter):
    def one(carry, _):
        y, lp, key = carry
        key, kp, ku = jax.random.split(key, 3)
        p0 = jax.random.normal(kp, y.shape, y.dtype)
        _, g = logp_and_grad(y)
        p = p0 + 0.5 * eps * (g @ Lw)
        yy = y
        def leap(c, _):
            yy, p = c
            yy = yy + eps * p
            _, g = logp_and_grad(yy)
            return (yy, p + eps * (g @ Lw)), None
        (yy, p), _ = jax.lax.scan(leap, (yy, p), None, length=L - 1)
        yy = yy + eps * p
        lp2, g = logp_and_grad(yy)
        p = p + 0.5 * eps * (g @ Lw)      # same sign as the opening half-kick; leapfrog must
                                          # be reversible or every proposal is rejected
        dH = (lp2 - 0.5 * jnp.sum(p ** 2, 1)) - (lp - 0.5 * jnp.sum(p0 ** 2, 1))
        acc = jnp.log(jax.random.uniform(ku, (y.shape[0],), y.dtype)) < dH
        return (jnp.where(acc[:, None], yy, y), jnp.where(acc, lp2, lp), key), jnp.mean(acc)
    lp0, _ = logp_and_grad(y)
    (y, _, _), a = jax.lax.scan(one, (y, lp0, key), None, length=n_iter)
    return y, jnp.mean(a)

@partial(jax.jit, static_argnums=(3,))
def pcn(y, key, rho, n_iter):
    def one(carry, _):
        y, lp, key = carry
        key, kx, ku = jax.random.split(key, 3)
        xi = jax.random.normal(kx, y.shape, y.dtype)
        yy = jnp.sqrt(1 - rho ** 2) * y + rho * xi
        lp2 = jax.vmap(lambda a: m.logdensity(xm + a @ Lw, m.data))(yy)
        # r = log p - log N(0,I); the Gaussian part cancels in the pCN ratio
        dr = (lp2 + 0.5 * jnp.sum(yy ** 2, 1)) - (lp + 0.5 * jnp.sum(y ** 2, 1))
        acc = jnp.log(jax.random.uniform(ku, (y.shape[0],), y.dtype)) < dr
        return (jnp.where(acc[:, None], yy, y), jnp.where(acc, lp2, lp), key), jnp.mean(acc)
    lp0 = jax.vmap(lambda a: m.logdensity(xm + a @ Lw, m.data))(y)
    (y, _, _), a = jax.lax.scan(one, (y, lp0, key), None, length=n_iter)
    return y, jnp.mean(a)

def main():
    out = []
    print(H.HDR); print("-" * len(H.HDR))
    y0 = lambda s: jax.random.normal(jax.random.key(s), (800, H.DIM), dtype=jnp.float32)

    # reference: MALA through the shipped method
    for seed in [0]:
        t0 = time.time()
        P = m.whitened_ula(m.particles_init, k=800, n_steps=2000, random_seed=seed,
                           monitor_convergence=-1, metropolis=True, x_map=x_map)
        r = score(P, "MALA 2500 grad", 2500, time.time() - t0); H.show(r); out.append(r)

    for eps, L, n in [(1.2, 2, 50), (0.8, 3, 50), (0.5, 5, 40), (0.3, 10, 30),
                      (0.5, 5, 200), (0.3, 10, 100)]:
        t0 = time.time()
        y, a = hmc(y0(0), jax.random.key(1), eps, L, n)
        y.block_until_ready()
        r = score(xm + y @ Lw.T, f"HMC eps={eps} L={L} n={n}", n * L, time.time() - t0)
        r["acc"] = float(a); H.show(r); out.append(r)

    for rho, n in [(0.05, 2000), (0.05, 20000), (0.15, 20000)]:
        t0 = time.time()
        y, a = pcn(y0(0), jax.random.key(1), rho, n)
        y.block_until_ready()
        r = score(xm + y @ Lw.T, f"pCN rho={rho} n={n}", n, time.time() - t0)
        r["acc"] = float(a); H.show(r); out.append(r)

    H.show(H.gold_row())
    print(f'\n{"method":>26} {"evals/chain":>12} {"accept":>7} {"energy":>7} {"varwtd":>7} {"sec":>6}')
    for r in out:
        print(f'{r["tag"]:>26} {r["cost"]:>12} {r.get("acc", float("nan")):>7.3f} '
              f'{r["energy"]:>7.3f} {r["varwtd"]:>7.3f} {r["t"]:>6.1f}')
    H.save(out, "exp25_faster_results")

if __name__ == "__main__":
    main()
