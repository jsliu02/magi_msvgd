'''
Robustness check for the reweighted-kernel result (test_reweighted_kernel.py): is
"reweighted kernel beats standard MAGI, beta=1.0, on FHN" a lucky draw from one seed on one
dataset?

Part 1: 5 different MSVGD random seeds (particle init + mitotic split + optimizer
    randomness) on the SAME y.csv dataset, compared to the existing NUTS gold standard
    (nuts_gold_standard.npz) -- tests whether the single-seed result was itself a fluke of
    MSVGD's own stochasticity.

Part 2: 4 newly-simulated datasets (fresh noise realizations of the same true generating
    process: theta=[0.2,0.2,3.0], X0=[-1,1], sigma=[0.2,0.2], same observation grid as
    y.csv), one MSVGD seed each -- tests whether the result generalizes across datasets.
    NOTE: no fresh NUTS gold standard per dataset here (each one is a ~30 min, 8-chain run;
    not tractable for "a few" datasets) -- Part 2 instead reports coverage of the KNOWN true
    theta (we generated the data, so we know it) and the standard-vs-reweighted CI width
    ratio directly, which doesn't need an external gold standard to be informative.

Run: python test_reweighted_kernel_robustness.py
fp32, matching prior experiments.
'''

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.integrate import solve_ivp

from magi import MAGI
from magi_reweighted import ReweightedMAGI

TRUE_THETA = np.array([0.2, 0.2, 3.0])
TRUE_X0 = np.array([-1.0, 1.0])
TRUE_SIGMA = np.array([0.2, 0.2])
OBS_TIMES = np.arange(0, 20.001, 0.5)      # 41 points, matches y.csv
DISC_TIMES = np.arange(0, 20.001, 0.125)   # 161 points, matches MAGI discretization


def fhn_rhs(t, X, a, b, c):
    V, R = X
    return [c * (V - V**3 / 3 + R), -1 / c * (V - a + b * R)]


def ode(X, theta, t=None):
    V, R = X.T
    a, b, c = theta
    return jnp.stack([c * (V - V**3 / 3 + R),
                       -1 / c * (V - a + b * R)])


def load_original_data():
    data = np.loadtxt('y.csv', delimiter=',')
    full = np.full((DISC_TIMES.shape[0], data.shape[1]), np.nan)
    full[:, 0] = DISC_TIMES
    full[np.isin(full[:, 0], data[:, 0])] = data
    return full


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


def build_solver(cls, full):
    solver = cls(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    solver.put(dtype=jnp.float32, device=jax.devices()[0])
    return solver


def run_msvgd(solver, random_seed=0, k_final=800, max_iter=1000):
    start = time.time()
    _, thetas, _ = solver.solve(
        k=200, sigma_init=0.01, k_final=k_final,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=max_iter, random_seed=random_seed,
        monitor_convergence=-1,
    )
    jax.block_until_ready(thetas)
    return thetas, time.time() - start


def summarize(thetas):
    mean = np.array(thetas.mean(axis=0))
    std = np.array(thetas.std(axis=0))
    lo = np.array(jnp.quantile(thetas, 0.025, axis=0))
    hi = np.array(jnp.quantile(thetas, 0.975, axis=0))
    return mean, std, lo, hi


if __name__ == '__main__':
    gold = np.load('nuts_gold_standard.npz')
    nuts_mean, nuts_std = gold['theta_mean'], gold['theta_std']
    nuts_width = gold['theta_ci_hi'] - gold['theta_ci_lo']
    print(f"Gold standard width = {nuts_width}\n")

    full_orig = load_original_data()

    print("=== Part 1: 5 MSVGD seeds on y.csv (vs NUTS gold standard) ===")
    part1 = {}
    for seed in range(5):
        for cls, label in [(MAGI, 'standard'), (ReweightedMAGI, 'reweighted')]:
            solver = build_solver(cls, full_orig)
            thetas, elapsed = run_msvgd(solver, random_seed=seed)
            mean, std, lo, hi = summarize(thetas)
            width_pct = 100 * (hi - lo) / nuts_width
            covered = (lo <= TRUE_THETA) & (TRUE_THETA <= hi)
            mean_err = np.abs(mean - nuts_mean)
            part1.setdefault(label, []).append(width_pct)
            print(f"  seed={seed} {label:11s} ({elapsed:.1f}s): width%NUTS={width_pct}, "
                  f"|mean-NUTS|={mean_err}, covered={covered}")

    print("\n  --- Part 1 summary (mean +/- std over 5 seeds, width% of NUTS) ---")
    for label, widths in part1.items():
        widths = np.stack(widths)
        print(f"  {label:11s}: mean={widths.mean(axis=0)}, std={widths.std(axis=0)}")

    print("\n=== Part 2: 4 new simulated datasets, 1 MSVGD seed each (no fresh NUTS) ===")
    part2 = {}
    for dseed in range(100, 104):
        full = simulate_dataset(dseed)
        row = {}
        for cls, label in [(MAGI, 'standard'), (ReweightedMAGI, 'reweighted')]:
            solver = build_solver(cls, full)
            thetas, elapsed = run_msvgd(solver, random_seed=0)
            mean, std, lo, hi = summarize(thetas)
            width = hi - lo
            covered = (lo <= TRUE_THETA) & (TRUE_THETA <= hi)
            row[label] = dict(mean=mean, std=std, width=width, covered=covered)
            part2.setdefault(label, []).append(dict(width=width, covered=covered, mean=mean))
            print(f"  dataset={dseed} {label:11s} ({elapsed:.1f}s): mean={mean}, "
                  f"width={width}, covered={covered}")
        ratio = row['reweighted']['width'] / row['standard']['width']
        print(f"    -> reweighted/standard width ratio = {ratio}")

    print("\n  --- Part 2 summary over 4 datasets ---")
    for label, rows in part2.items():
        widths = np.stack([r['width'] for r in rows])
        covered = np.stack([r['covered'] for r in rows])
        print(f"  {label:11s}: mean width={widths.mean(axis=0)}, "
              f"coverage rate (per-theta, over {len(rows)} datasets)={covered.mean(axis=0)}")
