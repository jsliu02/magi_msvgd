'''
B=100 simulated-dataset study comparing four approaches to sampling MAGI's FHN posterior:

  1. standard    -- MAGI (joint-kernel MSVGD)
  2. reweighted  -- ReweightedMAGI (density-reweighted SVGD kernel, reweighted_svgd.pdf)
  3. mk          -- MKMAGI (Multiple Kernel SVGD, mksvgd.pdf; safe bandwidth range
                    n_kernels=5, ratio=2.0 -- see msvgd/test_msvgd_mk.py for why the naive
                    wide range collapses)
  4. nuts        -- short single-chain NUTS reference (fp32, warmup=200, sampling=500,
                    target_acceptance_rate=0.8). NOT a rigorous gold standard like
                    nuts_gold_standard.npz (that was 8 chains x 10000 steps, fp64,
                    diagnostics-verified, ~32 min) -- this is a much cheaper per-dataset
                    reference, acceptable per instruction for this larger study. Each dataset
                    gets a fresh MAGI instance so NUTS's logdensity closure differs every
                    time -- this means (unlike the MSVGD paths, which pass data as an
                    explicit jit argument specifically to enable persistent-cache reuse
                    across datasets, see magi.py's own docstring) every dataset's NUTS run
                    pays a full retrace+recompile; measured ~15-20s/dataset at these settings,
                    which is what makes B=100 tractable at all here.

Since we know the TRUE theta used to simulate every dataset (unlike the single real y.csv
dataset), coverage is evaluated directly against ground truth for all four methods,
including NUTS -- this also serves as a sanity check on whether the short NUTS run itself is
behaving reasonably (nominal coverage should be near 95% if warmup/sampling were adequate).

Checkpoints results to b100_mk_study_results.json every 10 datasets (this is a long-running
job; don't want to lose 100 datasets of compute to an interruption).

Run: python b100_mk_study.py
'''

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import blackjax
from scipy.integrate import solve_ivp

from magi import MAGI
from magi_reweighted import ReweightedMAGI
from magi_mk import MKMAGI

B = 100
TRUE_THETA = np.array([0.2, 0.2, 3.0])
TRUE_X0 = np.array([-1.0, 1.0])
TRUE_SIGMA = np.array([0.2, 0.2])
OBS_TIMES = np.arange(0, 20.001, 0.5)
DISC_TIMES = np.arange(0, 20.001, 0.125)
CHECKPOINT_PATH = 'b100_mk_study_results.json'
DATASET_SEED_OFFSET = 5000  # distinct from other seeds used elsewhere this session


def fhn_rhs(t, X, a, b, c):
    V, R = X
    return [c * (V - V**3 / 3 + R), -1 / c * (V - a + b * R)]


def ode(X, theta, t=None):
    V, R = X.T
    a, b, c = theta
    return jnp.stack([c * (V - V**3 / 3 + R),
                       -1 / c * (V - a + b * R)])


def simulate_dataset(seed):
    sol = solve_ivp(fhn_rhs, [0, 20], TRUE_X0, args=tuple(TRUE_THETA),
                     t_eval=OBS_TIMES, rtol=1e-10, atol=1e-10)
    rng = np.random.default_rng(seed)
    noisy = sol.y.T + rng.normal(0, TRUE_SIGMA, size=sol.y.T.shape)
    data = np.column_stack([OBS_TIMES, noisy])
    full = np.full((DISC_TIMES.shape[0], 3), np.nan)
    full[:, 0] = DISC_TIMES
    full[np.isin(full[:, 0], data[:, 0])] = data
    return full


def build_solver(cls, full, dtype=jnp.float32):
    solver = cls(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    solver.put(dtype=dtype, device=jax.devices()[0])
    return solver


def run_msvgd(solver, random_seed=0, k_final=800, max_iter=1000):
    _, thetas, _ = solver.solve(
        k=200, sigma_init=0.01, k_final=k_final,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=max_iter, random_seed=random_seed,
        monitor_convergence=-1,
    )
    return thetas


def run_short_nuts(solver, warmup_steps=200, sampling_steps=500, seed=0, target_accept=0.8):
    key = jr.key(seed)
    warmup_key, sample_key = jr.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, solver.magi_logdensity,
                                         target_acceptance_rate=target_accept)
    (state, parameters), _ = warmup.run(warmup_key, position=solver.particles_init, num_steps=warmup_steps)
    kernel = blackjax.nuts(solver.magi_logdensity, **parameters)
    _, (states, info) = blackjax.util.run_inference_algorithm(
        sample_key, kernel, initial_state=state, num_steps=sampling_steps)
    thetas = states.position[:, :3]
    return thetas, int(info.is_divergent.sum())


def summarize(thetas):
    mean = np.array(thetas.mean(axis=0))
    std = np.array(thetas.std(axis=0))
    lo = np.array(jnp.quantile(thetas, 0.025, axis=0))
    hi = np.array(jnp.quantile(thetas, 0.975, axis=0))
    covered = ((lo <= TRUE_THETA) & (TRUE_THETA <= hi)).tolist()
    return dict(mean=mean.tolist(), std=std.tolist(), lo=lo.tolist(), hi=hi.tolist(), covered=covered)


if __name__ == '__main__':
    results = {m: [] for m in ['standard', 'reweighted', 'mk', 'nuts']}
    t_start = time.time()

    for b in range(B):
        full = simulate_dataset(DATASET_SEED_OFFSET + b)
        row = {}

        for cls, label in [(MAGI, 'standard'), (ReweightedMAGI, 'reweighted'), (MKMAGI, 'mk')]:
            t0 = time.time()
            solver = build_solver(cls, full, dtype=jnp.float32)
            thetas = run_msvgd(solver, random_seed=0)
            jax.block_until_ready(thetas)
            row[label] = summarize(thetas)
            row[label]['elapsed'] = time.time() - t0

        t0 = time.time()
        solver64 = build_solver(MAGI, full, dtype=jnp.float64)
        thetas_nuts, n_div = run_short_nuts(solver64, seed=b)
        jax.block_until_ready(thetas_nuts)
        row['nuts'] = summarize(thetas_nuts)
        row['nuts']['elapsed'] = time.time() - t0
        row['nuts']['n_divergent'] = n_div

        for label in results:
            results[label].append(row[label])

        elapsed = time.time() - t_start
        covered_str = {m: row[m]['covered'] for m in results}
        print(f"dataset {b + 1}/{B} done, total_elapsed={elapsed:.1f}s, covered={covered_str}", flush=True)

        if (b + 1) % 10 == 0 or b == B - 1:
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(results, f)
            print(f"  checkpointed to {CHECKPOINT_PATH}", flush=True)

    print(f"\nB={B} study complete in {time.time() - t_start:.1f}s")

    print("\n=== Coverage rate (per theta [a,b,c]) ===")
    for label, rows in results.items():
        covered = np.array([r['covered'] for r in rows])
        print(f"  {label:11s}: {covered.mean(axis=0)}")

    print("\n=== Mean CI width (per theta [a,b,c]) ===")
    for label, rows in results.items():
        widths = np.array([np.array(r['hi']) - np.array(r['lo']) for r in rows])
        print(f"  {label:11s}: {widths.mean(axis=0)}")

    print("\n=== Mean |mean - true theta| (per theta [a,b,c]) ===")
    for label, rows in results.items():
        errs = np.array([np.abs(np.array(r['mean']) - TRUE_THETA) for r in rows])
        print(f"  {label:11s}: {errs.mean(axis=0)}")
