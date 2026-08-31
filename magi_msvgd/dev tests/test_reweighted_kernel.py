'''
Test the reweighted-kernel SVGD (magi_reweighted.ReweightedMAGI, from
"Local KL Convergence Rate for SVGD with Reweighted Kernel", reweighted_svgd.pdf) against
standard MAGI on the FHN benchmark, compared to the diagnostics-verified NUTS gold standard
(nuts_gold_standard.npz from test_nuts_gold_standard.py).

Also tries combining it with the best lever found so far (moderate tempering, beta~0.6-0.7),
since the reweighted kernel and tempering address different (but related) aspects of
SVGD's underdispersion.

Run: python test_reweighted_kernel.py
fp32, matching prior MSVGD/tempering experiments.
'''

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from magi import MAGI
from magi_reweighted import ReweightedMAGI

TRUE_THETA = np.array([0.2, 0.2, 3.0])


def ode(X, theta, t=None):
    V, R = X.T
    a, b, c = theta
    return jnp.stack([c * (V - V**3 / 3 + R),
                       -1 / c * (V - a + b * R)])


def load_data():
    data = np.loadtxt('y.csv', delimiter=',')
    I = np.arange(0, 20.001, 0.125)
    full = np.full((I.shape[0], data.shape[1]), np.nan)
    full[:, 0] = I
    full[np.isin(full[:, 0], data[:, 0])] = data
    return full


def build_solver(cls):
    full = load_data()
    solver = cls(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    solver.put(dtype=jnp.float32, device=jax.devices()[0])
    return solver


def run_msvgd(solver, label, beta=1.0, k=200, k_final=800, max_iter=1000, random_seed=0):
    if beta != 1.0:
        original_logdensity = solver.logdensity
        solver.logdensity = lambda x, d: beta * original_logdensity(x, d)

    start = time.time()
    _, thetas, _ = solver.solve(
        k=k, sigma_init=0.01, k_final=k_final,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=max_iter, random_seed=random_seed,
        monitor_convergence=-1,
    )
    jax.block_until_ready(thetas)
    elapsed = time.time() - start
    print(f"  {label}: solve done in {elapsed:.1f}s")
    return thetas


def summarize(theta_samples):
    mean = np.array(theta_samples.mean(axis=0))
    std = np.array(theta_samples.std(axis=0))
    lo = np.array(jnp.quantile(theta_samples, 0.025, axis=0))
    hi = np.array(jnp.quantile(theta_samples, 0.975, axis=0))
    return mean, std, lo, hi


def report(name, mean, std, lo, hi, nuts_mean, nuts_std, nuts_width):
    width = hi - lo
    mean_err = np.abs(mean - nuts_mean)
    std_err = np.abs(std - nuts_std)
    width_pct = 100 * width / nuts_width
    covered = (lo <= TRUE_THETA) & (TRUE_THETA <= hi)
    print(f"  {name:45s}: |mean-NUTS|={mean_err}, |std-NUTS|={std_err}, "
          f"width%NUTS={width_pct}, covered={covered}")
    return dict(mean=mean, std=std, lo=lo, hi=hi, width_pct=width_pct)


if __name__ == '__main__':
    gold = np.load('nuts_gold_standard.npz')
    nuts_mean, nuts_std = gold['theta_mean'], gold['theta_std']
    nuts_lo, nuts_hi = gold['theta_ci_lo'], gold['theta_ci_hi']
    nuts_width = nuts_hi - nuts_lo
    print(f"Gold standard: mean={nuts_mean}, std={nuts_std}, width={nuts_width}\n")

    results = {}

    print("=== Standard MAGI (joint kernel) ===")
    for beta in [1.0, 0.7]:
        solver = build_solver(MAGI)
        thetas = run_msvgd(solver, f"standard, beta={beta}", beta=beta)
        mean, std, lo, hi = summarize(thetas)
        results[f"standard_beta={beta}"] = report(
            f"standard, beta={beta}", mean, std, lo, hi, nuts_mean, nuts_std, nuts_width)

    print("\n=== ReweightedMAGI (density-reweighted kernel) ===")
    for beta in [1.0, 0.7]:
        solver = build_solver(ReweightedMAGI)
        thetas = run_msvgd(solver, f"reweighted, beta={beta}", beta=beta)
        mean, std, lo, hi = summarize(thetas)
        results[f"reweighted_beta={beta}"] = report(
            f"reweighted, beta={beta}", mean, std, lo, hi, nuts_mean, nuts_std, nuts_width)

    print("\n=== Summary (CI width as % of NUTS width) ===")
    for name, r in results.items():
        print(f"{name:28s}: {r['width_pct']}")
