"""
Exp 9: verify whitened-IMQ, and test R as a STOPPING RULE.

exp08 gave whitened IMQ energy 0.051 at it=1500 (floor 0.048) but 0.062 by it=4000 -- the flow
slowly over-contracts, so there is an optimal stopping time. R tracks it: 1.010 at 1500, 0.950
at 4000. That suggests stopping when the Stein diagnostic first reaches 1, which is a criterion
with a derived target rather than a tuned patience.

Tests: seed robustness, particle count, sensitivity to initialization (does it need Laplace
samples to start, or will any init do?), and whether "stop at R=1" lands near the energy optimum.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H
from msvgd import MSVGD


def imq_flow(s, y, raw_neg):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1.0 + L2sq / h) ** -0.5
    Kg = (1.0 + L2sq / h) ** -1.5
    dxkxy = (Kg.sum(axis=1, keepdims=True) * y - Kg @ y) * (1.0 / h)
    return (Kx @ raw_neg - dxkxy) / y.shape[0]


def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T, jnp.float32)
    Lwi = jnp.asarray((V * np.sqrt(evc)) @ V.T, jnp.float32)
    xm = jnp.asarray(x_map, jnp.float32)
    m = H.build_magi()
    s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)
    out = []

    def run(k, seed, init_kind, iters=3000, lr=1e-2, every=100):
        key = jax.random.key(seed)
        if init_kind == "laplace":
            y = jax.random.normal(key, (k, H.DIM), dtype=jnp.float32)
        elif init_kind == "tight":
            y = jax.random.normal(key, (k, H.DIM), dtype=jnp.float32) * 0.05
        else:                                    # the MAGI initialization, mapped to y
            y = ((jnp.asarray(m.particles_init, jnp.float32) - xm) @ Lwi
                 + jax.random.normal(key, (k, H.DIM), dtype=jnp.float32) * 0.01)
        traj = []
        for it in range(1, iters + 1):
            y = y - lr * imq_flow(s, y, -s.gradient(y, m.data))
            if it % every == 0:
                if not bool(jnp.all(jnp.isfinite(y))):
                    traj.append((it, np.nan, np.nan)); break
                X = xm + y @ Lw.T
                r = H.evaluate(X, m, tag="")
                traj.append((it, r["R_global"], r["energy"]))
        return y, traj

    print(f'{"config":>34} {"stop@R=1":>9} {"E@stop":>7} {"bestE":>7} {"it*":>5} {"E@3000":>7}')
    for k, seed, ik in [(800, 0, "laplace"), (800, 1, "laplace"), (800, 2, "laplace"),
                        (800, 3, "laplace"), (800, 0, "tight"), (800, 0, "magi-init"),
                        (200, 0, "laplace"), (1600, 0, "laplace")]:
        y, traj = run(k, seed, ik)
        T = np.array([t for t in traj if np.isfinite(t[1])])
        if len(T) == 0:
            print(f'{f"k={k} s{seed} {ik}":>34}   DIVERGED"'); continue
        below = np.where(T[:, 1] <= 1.0)[0]
        istop = int(T[below[0], 0]) if len(below) else int(T[-1, 0])
        estop = float(T[below[0], 2]) if len(below) else float(T[-1, 2])
        j = int(np.argmin(T[:, 2]))
        print(f'{f"k={k} s{seed} {ik}":>34} {istop:>9} {estop:>7.4f} '
              f'{T[j,2]:>7.4f} {int(T[j,0]):>5} {T[-1,2]:>7.4f}')
        out.append({"k": k, "seed": seed, "init": ik, "stop_iter": istop,
                    "energy_at_stop": estop, "best_energy": float(T[j, 2]),
                    "best_iter": int(T[j, 0]), "energy_3000": float(T[-1, 2]),
                    "traj": T.tolist()})

    # final scored ensembles at the R=1 stopping point
    print(); print(H.HDR); print("-" * len(H.HDR))
    finals = []
    for seed in [0, 1, 2, 3]:
        rec = [o for o in out if o["k"] == 800 and o["seed"] == seed and o["init"] == "laplace"][0]
        y, _ = run(800, seed, "laplace", iters=rec["stop_iter"], every=10 ** 9)
        r = H.evaluate(xm + y @ Lw.T, m, tag=f"whitened IMQ @R=1 s{seed}")
        finals.append(r); H.show(r)
    A = {kk: float(np.mean([f[kk] for f in finals])) for kk in
         ["R_global", "energy", "bias", "width_dev", "sd_ratio_med"]}
    print(f'{"MEAN of 4 seeds":>26} {"":>21} {A["width_dev"]:5.1f} {A["R_global"]:7.3f} '
          f'{"":>6} {A["energy"]:7.3f} {A["bias"]:6.3f} {A["sd_ratio_med"]:6.3f}')
    r = H.gold_row(); H.show(r)
    H.save({"sweep": out, "finals": finals}, "exp09_verify_results")


if __name__ == "__main__":
    main()
