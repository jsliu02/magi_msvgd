"""
Exp 10: does any of this generalize, or is it an artifact of one dataset?

The central claim so far -- that the posterior is Gaussian with covariance H^-1 at the MAP, so
the whole problem reduces to locating the mean -- was established on a single FHN dataset at
one noise level and observation density. Both of those control how prior-dominated and how
nonlinear the posterior is, so vary them and re-test on independently generated references.

Settings: baseline; half the observations; quarter of the observations; assumed noise 0.5.
Sparser data and larger noise both push the posterior toward the (nonlinear) ODE prior, which
is where Gaussianity should fail first if it is going to.
"""
import os, sys, time, json
import numpy as np, jax, jax.numpy as jnp, optax
import harness as H
from msvgd import MSVGD

N_CHAINS, WARMUP, DRAWS = 4, 1000, 3000
MAX_DOUBLINGS = 10


class Ref:
    def __init__(self, pos):
        self.pos = pos
        self.mean = pos.mean(0); self.sd = pos.std(0)
        self.cov = np.cov(pos, rowvar=False)
        ev, V = np.linalg.eigh(self.cov)
        self.evals = np.maximum(ev, 1e-14); self.evecs = V
        self.theta_w = (np.quantile(pos[:, :3], .975, 0) - np.quantile(pos[:, :3], .025, 0))
        rng = np.random.default_rng(0)
        self.ref = pos[rng.choice(len(pos), min(2000, len(pos)), replace=False)]

    def whiten(self, X):
        return (np.asarray(X, np.float64) - self.mean) @ self.evecs / np.sqrt(self.evals)

    def score(self, P, tag):
        P = np.asarray(P, np.float64)
        rng = np.random.default_rng(1)
        lo, hi = np.quantile(P[:, :3], .025, 0), np.quantile(P[:, :3], .975, 0)
        w = 100 * (hi - lo) / self.theta_w
        prof = ((P - self.mean) @ self.evecs).var(0) / self.evals
        order = np.argsort(self.evals)[::-1]
        return {"tag": tag, "energy": H._energy_distance(self.whiten(P), self.whiten(self.ref), rng),
                "bias": float(np.sqrt((self.whiten(P.mean(0)[None, :]) ** 2).mean())),
                "width_dev": float(np.abs(w - 100).mean()),
                "profile": [float(np.median(prof[b])) for b in np.array_split(order, 5)],
                "sd_ratio": float(np.median(P.std(0) / self.sd))}


def show(r):
    print(f'{r["tag"]:>30} energy={r["energy"]:8.4f}  bias={r["bias"]:6.3f}  '
          f'dev={r["width_dev"]:5.1f}  sdrat={r["sd_ratio"]:5.3f}  '
          f'profile={np.round(r["profile"],3)}')


def build(obs_stride, sigma):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")
    d = d[::obs_stride]
    grid = np.arange(0, 20.001, 0.125)
    full = np.full((grid.shape[0], 3), np.nan); full[:, 0] = grid
    full[np.isin(full[:, 0], d[:, 0])] = d
    from magi import MAGI
    m = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    return m


def run_nuts(m, seed=0):
    import blackjax
    from blackjax.diagnostics import potential_scale_reduction
    keys = jax.random.split(jax.random.key(seed), N_CHAINS)
    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, m.magi_logdensity, target_acceptance_rate=0.9)
        (st, par), _ = wu.run(wk, position=m.particles_init, num_steps=WARMUP)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(m.magi_logdensity, **par, max_num_doublings=MAX_DOUBLINGS),
            initial_state=st, num_steps=DRAWS)
        return states.position, info.is_divergent
    pos, div = jax.vmap(one)(keys)
    rhat = potential_scale_reduction(pos)
    return np.asarray(pos).reshape(-1, H.DIM), float(np.max(rhat)), int(np.sum(div))


def main():
    H.patch_split()
    import os as _os
    ALL = [("baseline  s=0.2 stride1", 1, 0.2), ("half-obs  s=0.2 stride2", 2, 0.2),
           ("quarter   s=0.2 stride4", 4, 0.2), ("noisy     s=0.5 stride1", 1, 0.5)]
    pick = _os.environ.get("SETTINGS", "0,1,2,3").split(",")
    settings = [ALL[int(i)] for i in pick]
    allres = {}
    for name, stride, sig in settings:
        print(f"\n{'='*100}\n{name}")
        jax.config.update("jax_enable_x64", True)
        m = build(stride, sig); m.put(dtype=jnp.float64, device=jax.devices()[0])
        t0 = time.time()
        pos, rhat, ndiv = run_nuts(m)
        print(f"  NUTS {N_CHAINS}x{DRAWS} on CPU in {time.time()-t0:.0f}s   max Rhat={rhat:.4f}  divergences={ndiv}")
        R = Ref(pos)

        m.particles = None
        m.solve(k=1, sigma_init=0.0, is_MAP=True, max_iter=30000, atol=1e-7, rtol=0.0,
                random_seed=0, monitor_convergence=-1, optimizer=optax.contrib.prodigy,
                optimizer_kwargs={})
        x_map = np.asarray(m.particles[0], np.float64)
        Hs = -np.asarray(jax.hessian(m.magi_logdensity)(jnp.asarray(x_map)), np.float64)
        Hs = .5 * (Hs + Hs.T)
        ev, V = np.linalg.eigh(Hs); evc = np.maximum(ev, 1e-8 * ev.max())
        Sig_h = (V / np.sqrt(evc)) @ V.T
        rng = np.random.default_rng(0)
        rows = []
        rows.append(R.score(x_map[None, :] + rng.standard_normal((800, H.DIM)) @ Sig_h.T,
                            "N(MAP, H^-1)"))
        rows.append(R.score(R.mean[None, :] + rng.standard_normal((800, H.DIM)) @ Sig_h.T,
                            "N(REFmean, H^-1)  <- Gaussianity"))
        rows.append(R.score(pos[rng.choice(len(pos), 800, replace=False)], "NUTS k=800 (floor)"))

        # production baseline + the whitened-IMQ recipe
        jax.config.update("jax_enable_x64", False)
        m32 = build(stride, sig); m32.put(dtype=jnp.float32, device=jax.devices()[0])
        m32.particles = None
        m32.solve(k=200, sigma_init=0.01, k_schedule=800, optimizer=optax.contrib.prodigy,
                  optimizer_kwargs={}, atol=0.0, rtol=0.0, max_iter=1000, random_seed=0,
                  monitor_convergence=-1, reweighted_kernel=True)
        rows.append(R.score(m32.particles, "x-space reweighted (current)"))

        Lw = jnp.asarray(Sig_h, jnp.float32); xm = jnp.asarray(x_map, jnp.float32)
        s = MSVGD(lambda y, db: m32.logdensity(xm + Lw @ y, db), data=m32.data)
        for lr in [1e-2, 1e-3, 1e-4]:
            y = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
            best, div = None, False
            for it in range(1, 8001):
                L2sq, h = s.pairwise_distance(y, -1)
                Kx = (1. + L2sq / h) ** -.5; Kg = (1. + L2sq / h) ** -1.5
                dx = (Kg.sum(1, keepdims=True) * y - Kg @ y) * (1. / h)
                y = y + lr * ((Kx @ s.gradient(y, m32.data) + dx) / y.shape[0])
                if it % 100 == 0:
                    if not bool(jnp.all(jnp.isfinite(y))):
                        div = True; break
                    X = np.asarray(xm + y @ Lw.T, np.float64)
                    sc = float(-np.sum((X - X.mean(0)) * np.asarray(m32.gradient(
                        jnp.asarray(X, jnp.float32), m32.data), np.float64)) / X.size)
                    best = X
                    if sc <= 1.05:
                        break
            if div or best is None:
                print(f'  whitened IMQ lr={lr:g}: DIVERGED'); continue
            rows.append(R.score(best, f"whitened IMQ @R<=1.05 lr={lr:g}"))
            break
        np.savez_compressed(f"exp10_ref_{name.split()[0]}.npz", pos=pos, x_map=x_map, Hess=Hs)
        for r in rows: show(r)
        allres[name] = {"rhat": rhat, "ndiv": ndiv, "rows": rows}
        with open(_os.environ.get("OUTJSON", "exp10_generalize_results.json"), "w") as f:
            json.dump(allres, f, indent=2)
    print("\nsaved")


if __name__ == "__main__":
    main()
