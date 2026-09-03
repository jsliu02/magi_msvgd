"""
Exp 06b: does dropping the 1/log(K) change the particle requirement's RATE, not just its constant?

exp06 measured K_crit(p) under h = Med/ln K, the bandwidth msvgd used to ship, and found
ln K_crit = 0.862p + 1.449 -- exponential in the dimension. That rule has since been replaced by
the plain median h = Med, for which Ba et al. (ICLR 2022) Cor. 4 gives Var/Var_target = 0.582 K/d.
Setting that to 1 predicts K_crit ~ 1.72 d: LINEAR. The prediction sits outside the regime their
theorem covers (it assumes d/K > 1), so it is an extrapolation and needs checking. Same sweep as
exp06, same thresholds, only the bandwidth differs.
"""
import numpy as np, jax, jax.numpy as jnp, optax, sys, os

jax.config.update("jax_enable_x64", True)
KS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
PS = [2, 3, 4, 5, 6, 7, 8]
ITERS, THRESH = 4000, (0.8, 0.9)


def pairwise(x, use_log):
    sq = jnp.sum(x ** 2, axis=1)
    with jax.default_matmul_precision("highest"):
        L2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
    k = x.shape[0]
    med = jnp.median(jnp.clip(L2[np.triu_indices(k, 1)], min=1e-6))
    h = med / jnp.log(jnp.asarray(k, x.dtype)) if use_log else med
    return L2, h


def equilibrium_ratio(p, k, use_log, seed=0):
    """Isotropic N(0, I_p), started at exact draws so there is no burn-in to mistake for one."""
    x0 = jnp.asarray(np.random.default_rng(seed).standard_normal((k, p)))
    opt = optax.adam(learning_rate=0.05)

    def step(carry, _):
        x, st = carry
        raw = x                                     # -grad log p for N(0, I)
        L2, h = pairwise(x, use_log)
        K = jnp.exp(-L2 / h)
        rep = (K.sum(axis=1, keepdims=True) * x - K @ x) * (2.0 / h)
        upd, st = jax.vmap(opt.update)((K @ raw - rep) / k, st, x)
        return (optax.apply_updates(x, upd), st), None

    (xf, _), _ = jax.lax.scan(jax.jit(step), (x0, jax.vmap(opt.init)(x0)), None, length=ITERS)
    return float(jnp.mean(jnp.var(xf, axis=0)))


print(f'{"p":>3} ' + " ".join(f'{k:>6}' for k in KS))
crit = {t: {} for t in THRESH}
for p in PS:
    row = []
    for k in KS:
        r = equilibrium_ratio(p, k, use_log=False)
        row.append(r)
        for t in THRESH:
            if r >= t and p not in crit[t]:
                crit[t][p] = k
    print(f'{p:>3} ' + " ".join(f'{v:>6.3f}' for v in row), flush=True)

print()
for t in THRESH:
    ks = [crit[t].get(p) for p in PS]
    print(f'K_crit at ratio >= {t}: ' + " ".join(f'{k if k else "--":>6}' for k in ks))
    have = [(p, k) for p, k in zip(PS, ks) if k]
    if len(have) >= 3:
        pp = np.array([p for p, _ in have], float); kk = np.array([k for _, k in have], float)
        a_lin = np.polyfit(pp, kk, 1); a_log = np.polyfit(pp, np.log(kk), 1)
        r_lin = 1 - np.sum((kk - np.polyval(a_lin, pp)) ** 2) / np.sum((kk - kk.mean()) ** 2)
        r_log = 1 - np.sum((np.log(kk) - np.polyval(a_log, pp)) ** 2) / np.sum((np.log(kk) - np.log(kk).mean()) ** 2)
        print(f'   linear fit  K_crit = {a_lin[0]:.2f} p + {a_lin[1]:.2f}     R^2 = {r_lin:.4f}')
        print(f'   log fit  ln K_crit = {a_log[0]:.3f} p + {a_log[1]:.3f}     R^2 = {r_log:.4f}')
        print(f'   -> {"LINEAR" if r_lin > r_log else "EXPONENTIAL"} is the better description')
