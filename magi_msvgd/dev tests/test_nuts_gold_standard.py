'''
Build a robust, diagnostics-verified NUTS gold standard for the FHN benchmark (y.csv), to
replace the single-chain, short (1000 warmup + 4000 sampling) NUTS run used as a baseline in
test_stein_reweight.py. That single chain had no way to check convergence -- no R-hat (needs
>=2 chains), no ESS-vs-independent-chains cross-check, no divergence count reported, and a
single starting point can't rule out missed modes.

This script instead runs several independent NUTS chains (vmapped, so wall-clock cost is
close to a single chain despite full multi-chain diagnostics), started from dispersed points
drawn from the existing MSVGD posterior approximation (so if MSVGD found some spread, NUTS
gets a fair chance to confirm or reject it), with:
  - float64 throughout (not fp32) -- reduces leapfrog integration error, the standard fix for
    spurious NUTS divergences in models with tricky curvature like MAGI's GP-coupled ODE
    likelihood.
  - a higher target acceptance rate (0.95, vs blackjax's 0.8 default) during window
    adaptation, which drives the adapted step size down -- fewer divergences, at the cost of
    more leapfrog steps per iteration.
  - long warmup + sampling, since "a much longer run is acceptable" for a gold standard.

Diagnostics reported: per-chain divergence counts, rank-normalized split-R-hat and ESS
(blackjax.diagnostics, matching arviz's default rhat(method="rank")) for theta specifically
and for the full 325-dim position vector (worst case across ALL parameters, not just the 3
we care about -- a chain can look converged on theta while still not having mixed over X).
Only trust the pooled-chain theta summary if these look good (R-hat ~1.00, no meaningful
divergences, ESS in the thousands after pooling).

Run: python test_nuts_gold_standard.py
'''

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import blackjax
import blackjax.diagnostics as diagnostics

from magi import MAGI

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


def run_gold_standard(n_chains=8, warmup_steps=3000, sampling_steps=15000,
                       target_accept=0.95, base_seed=100,
                       save_path='nuts_gold_standard.npz'):
    full = load_data()
    solver = MAGI(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    solver.put(dtype=jnp.float64, device=jax.devices()[0])

    print("Getting dispersed starting points from a quick MSVGD run...")
    solver.solve(
        k=200, sigma_init=0.01, k_final=800,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=1000, random_seed=0,
        monitor_convergence=-1,
    )
    idx = jr.choice(jr.PRNGKey(base_seed), solver.particles.shape[0],
                     shape=(n_chains,), replace=False)
    starts = solver.particles[idx]

    logdensity_fn = solver.magi_logdensity

    def run_chain(seed, init_position):
        key = jr.key(seed)
        warmup_key, sample_key = jr.split(key)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, target_acceptance_rate=target_accept)
        (state, parameters), _ = warmup.run(warmup_key, position=init_position, num_steps=warmup_steps)
        kernel = blackjax.nuts(logdensity_fn, **parameters)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sample_key, kernel, initial_state=state, num_steps=sampling_steps)
        return states.position, info.is_divergent, parameters['step_size']

    seeds = base_seed + jnp.arange(n_chains)
    print(f"Running {n_chains} chains x ({warmup_steps} warmup + {sampling_steps} sampling) "
          f"NUTS steps, vmapped, fp64, target_accept={target_accept}...")
    start = time.time()
    positions, divergent, step_sizes = jax.jit(jax.vmap(run_chain))(seeds, starts)
    jax.block_until_ready(positions)
    print(f"  done in {time.time() - start:.1f}s")

    div_counts = divergent.sum(axis=1)
    print(f"\nDivergences per chain: {np.array(div_counts)} / {sampling_steps} "
          f"({100 * np.array(div_counts) / sampling_steps} %)")
    print(f"Adapted step sizes per chain: {np.array(step_sizes)}")

    theta_chains = positions[:, :, :3]  # (n_chains, sampling_steps, 3)
    rhat_theta = diagnostics.rhat(theta_chains, chain_axis=0, sample_axis=1)
    ess_theta = diagnostics.effective_sample_size(theta_chains, chain_axis=0, sample_axis=1)
    print(f"\ntheta R-hat (rank-normalized split, want ~1.00): {np.array(rhat_theta)}")
    print(f"theta ESS (pooled across chains): {np.array(ess_theta)}")

    rhat_all = diagnostics.rhat(positions, chain_axis=0, sample_axis=1)
    ess_all = diagnostics.effective_sample_size(positions, chain_axis=0, sample_axis=1)
    print(f"\nmax R-hat over all {positions.shape[-1]} dims (theta+X): {float(jnp.max(rhat_all)):.4f}")
    print(f"# dims with R-hat > 1.01: {int(jnp.sum(rhat_all > 1.01))} / {positions.shape[-1]}")
    print(f"min ESS over all dims: {float(jnp.min(ess_all)):.1f}")

    pooled_theta = theta_chains.reshape(-1, 3)
    mean = pooled_theta.mean(axis=0)
    std = pooled_theta.std(axis=0)
    lo = jnp.quantile(pooled_theta, 0.025, axis=0)
    hi = jnp.quantile(pooled_theta, 0.975, axis=0)
    covered = (lo <= TRUE_THETA) & (TRUE_THETA <= hi)

    print(f"\n=== NUTS gold standard: pooled {n_chains} chains x {sampling_steps} "
          f"samples = {pooled_theta.shape[0]} total ===")
    print(f"true theta = {TRUE_THETA}")
    print(f"mean       = {np.array(mean)}")
    print(f"std        = {np.array(std)}")
    print(f"95% CI lo  = {np.array(lo)}")
    print(f"95% CI hi  = {np.array(hi)}")
    print(f"CI width   = {np.array(hi - lo)}")
    print(f"covered    = {np.array(covered)}")

    if save_path is not None:
        print(f"\nSaving full chain (positions, divergences, diagnostics) to {save_path} ...")
        np.savez(
            save_path,
            positions=np.asarray(positions),           # (n_chains, sampling_steps, dim), fp64
            divergent=np.asarray(divergent),            # (n_chains, sampling_steps)
            step_sizes=np.asarray(step_sizes),           # (n_chains,)
            rhat_theta=np.asarray(rhat_theta),
            ess_theta=np.asarray(ess_theta),
            rhat_all=np.asarray(rhat_all),
            ess_all=np.asarray(ess_all),
            theta_mean=np.asarray(mean), theta_std=np.asarray(std),
            theta_ci_lo=np.asarray(lo), theta_ci_hi=np.asarray(hi),
            true_theta=TRUE_THETA,
            n_chains=n_chains, warmup_steps=warmup_steps, sampling_steps=sampling_steps,
            target_accept=target_accept, base_seed=base_seed,
        )
        size_mb = np.asarray(positions).nbytes / 1e6
        print(f"  saved ({size_mb:.1f} MB for positions array alone)")

    return dict(
        positions=positions, divergent=divergent, step_sizes=step_sizes,
        rhat_theta=rhat_theta, ess_theta=ess_theta, rhat_all=rhat_all, ess_all=ess_all,
        mean=mean, std=std, lo=lo, hi=hi,
    )


if __name__ == '__main__':
    run_gold_standard()
