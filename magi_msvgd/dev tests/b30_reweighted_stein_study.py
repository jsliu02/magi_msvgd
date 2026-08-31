'''
B=30 simulated-dataset study comparing:

  1. reweighted       -- ReweightedMAGI (density-reweighted SVGD kernel) alone, uniform weights
  2. reweighted_stein -- the same ReweightedMAGI particles, then post-hoc Stein importance
                          reweighting on top (test_stein_reweight.py's technique, using the
                          TRUE score -- beta=1 here, no tempering, so solver.logdensity already
                          is the true target)
  3. nuts             -- a properly multi-chain, diagnostics-checked NUTS reference this time:
                          4 chains (vmapped), 800 warmup + 3000 sampling steps, fp32,
                          target_acceptance_rate=0.9. Verified on a single dataset before this
                          study: R-hat <= 1.007, ESS 896-2818, 0 divergences -- much more
                          trustworthy than the B=100 study's short single-chain NUTS (which had
                          52-55% coverage on b/c, clearly under-mixed). Per-dataset R-hat/ESS/
                          divergences are still recorded and reported here so this study's own
                          NUTS quality isn't just taken on faith either.

Each dataset gets a fresh MAGI instance, so NUTS's logdensity closure differs every time and
pays a full retrace+recompile per dataset (see magi.py's own docstring on why the MSVGD path
avoids this by passing data as an explicit jit argument, and NUTS's 1-arg blackjax API
doesn't allow the same trick) -- this is why NUTS is the expensive part here (~200s/dataset)
despite being "shorter" than the B=100-study's rigorous 8-chain/10000-step gold standard
(~1900s/dataset).

Device placement: NUTS runs on CPU, the two SVGD variants run on GPU (explicit device_put
throughout run_multichain_nuts/build_solver) -- NUTS's single/few-chain sequential leapfrog
work doesn't parallelize well on GPU anyway, so this keeps the GPU free and avoids the two
workloads contending for it.

Checkpoints to b30_reweighted_stein_results.json every 5 datasets.

Run: python b30_reweighted_stein_study.py
'''

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import blackjax
import blackjax.diagnostics as diagnostics
from scipy.integrate import solve_ivp

from magi_reweighted import ReweightedMAGI

B = 30
TRUE_THETA = np.array([0.2, 0.2, 3.0])
TRUE_X0 = np.array([-1.0, 1.0])
TRUE_SIGMA = np.array([0.2, 0.2])
OBS_TIMES = np.arange(0, 20.001, 0.5)
DISC_TIMES = np.arange(0, 20.001, 0.125)
CHECKPOINT_PATH = 'b30_reweighted_stein_results.json'
DATASET_SEED_OFFSET = 8000  # distinct from other seeds used elsewhere this session

N_CHAINS = 4
NUTS_WARMUP = 800
NUTS_SAMPLING = 3000
NUTS_TARGET_ACCEPT = 0.9


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


def build_solver(cls, full, dtype=jnp.float32, device=None):
    if device is None:
        device = jax.devices()[0]  # GPU by default
    solver = cls(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2], init_device=device)
    solver.put(dtype=dtype, device=device)
    return solver


def run_msvgd(solver, random_seed=0, k_final=800, max_iter=1000):
    _, thetas, _ = solver.solve(
        k=200, sigma_init=0.01, k_final=k_final,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=max_iter, random_seed=random_seed,
        monitor_convergence=-1,
    )
    return thetas


def stein_kernel_matrix(solver, particles, score):
    '''Standard (RBF) KSD Gram matrix -- see test_stein_reweight.py for derivation.'''
    k, dim = particles.shape
    L2sq, h = solver.pairwise_distance(particles, -1)
    S, X = score, particles
    SS = S @ S.T
    A = S @ X.T
    diag_sx = jnp.sum(S * X, axis=1)
    cross = diag_sx[:, None] + diag_sx[None, :] - A - A.T
    K = jnp.exp(-L2sq / h)
    return K * (SS + (2.0 / h) * cross + 2.0 * dim / h - (4.0 / h**2) * L2sq)


def optimize_stein_weights(M, n_steps=3000, lr=0.05):
    k = M.shape[0]
    logits = jnp.zeros(k)
    opt = optax.adam(lr)
    opt_state = opt.init(logits)

    def loss_fn(logits):
        w = jax.nn.softmax(logits)
        return w @ M @ w

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(n_steps):
        loss, grad = loss_and_grad(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
    w = jax.nn.softmax(logits)
    ess = float(1.0 / jnp.sum(w**2))
    return w, ess


def weighted_quantile(values, weights, q):
    order = jnp.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = jnp.cumsum(w_sorted) - 0.5 * w_sorted
    cdf = cdf / jnp.sum(w_sorted)
    return jnp.interp(q, cdf, v_sorted)


def summarize(thetas, weights=None):
    if weights is None:
        mean = np.array(thetas.mean(axis=0))
        std = np.array(thetas.std(axis=0))
        lo = np.array(jnp.quantile(thetas, 0.025, axis=0))
        hi = np.array(jnp.quantile(thetas, 0.975, axis=0))
    else:
        mean_j = jnp.sum(thetas * weights[:, None], axis=0)
        var_j = jnp.sum(weights[:, None] * (thetas - mean_j) ** 2, axis=0)
        mean = np.array(mean_j)
        std = np.array(jnp.sqrt(var_j))
        lo = np.array(jnp.stack([weighted_quantile(thetas[:, d], weights, 0.025) for d in range(3)]))
        hi = np.array(jnp.stack([weighted_quantile(thetas[:, d], weights, 0.975) for d in range(3)]))
    covered = ((lo <= TRUE_THETA) & (TRUE_THETA <= hi)).tolist()
    return dict(mean=mean.tolist(), std=std.tolist(), lo=lo.tolist(), hi=hi.tolist(), covered=covered)


def run_multichain_nuts(full, seed):
    # NUTS runs entirely on CPU (single/few-chain sequential leapfrog doesn't parallelize well
    # on GPU anyway) -- build_solver's init_device/put(device=...) keep the whole MAGI instance
    # (precomputed matrices, particles_init, magi_logdensity's closed-over data) on CPU, and the
    # vmapped NUTS computation itself is placed on CPU explicitly below.
    from magi import MAGI
    cpu = jax.devices('cpu')[0]
    solver = build_solver(MAGI, full, dtype=jnp.float32, device=cpu)
    logdensity_fn = solver.magi_logdensity

    def run_chain(chain_seed, init_position):
        key = jr.key(chain_seed)
        warmup_key, sample_key = jr.split(key)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn,
                                             target_acceptance_rate=NUTS_TARGET_ACCEPT)
        (state, parameters), _ = warmup.run(warmup_key, position=init_position, num_steps=NUTS_WARMUP)
        kernel = blackjax.nuts(logdensity_fn, **parameters)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sample_key, kernel, initial_state=state, num_steps=NUTS_SAMPLING)
        return states.position, info.is_divergent

    with jax.default_device(cpu):
        starts = solver.particles_init[None, :] + jr.normal(
            jr.fold_in(jr.PRNGKey(seed), 999), (N_CHAINS, solver.particles_init.shape[0])) * 0.05
        seeds = seed * N_CHAINS + jnp.arange(N_CHAINS)
        starts = jax.device_put(starts, cpu)
        seeds = jax.device_put(seeds, cpu)

    positions, divergent = jax.jit(jax.vmap(run_chain), backend='cpu')(seeds, starts)
    theta_chains = positions[:, :, :3]
    rhat = np.array(diagnostics.rhat(theta_chains, chain_axis=0, sample_axis=1))
    ess = np.array(diagnostics.effective_sample_size(theta_chains, chain_axis=0, sample_axis=1))
    n_div = int(divergent.sum())

    pooled = theta_chains.reshape(-1, 3)
    return pooled, rhat, ess, n_div


if __name__ == '__main__':
    results = {m: [] for m in ['reweighted', 'reweighted_stein', 'nuts']}
    t_start = time.time()

    for b in range(B):
        full = simulate_dataset(DATASET_SEED_OFFSET + b)
        row = {}

        t0 = time.time()
        solver = build_solver(ReweightedMAGI, full, dtype=jnp.float32)
        thetas = run_msvgd(solver, random_seed=0)
        jax.block_until_ready(thetas)
        row['reweighted'] = summarize(thetas)
        row['reweighted']['elapsed'] = time.time() - t0

        t0 = time.time()
        score = solver.gradient(solver.particles, solver.data)  # beta=1 (untempered) -> true score
        M = stein_kernel_matrix(solver, solver.particles, score)
        weights, ess_stein = optimize_stein_weights(M)
        row['reweighted_stein'] = summarize(thetas, weights)
        row['reweighted_stein']['elapsed'] = time.time() - t0
        row['reweighted_stein']['ess_stein'] = ess_stein

        t0 = time.time()
        pooled, rhat, ess, n_div = run_multichain_nuts(full, seed=b)
        jax.block_until_ready(pooled)
        row['nuts'] = summarize(jnp.array(pooled))
        row['nuts']['elapsed'] = time.time() - t0
        row['nuts']['rhat'] = rhat.tolist()
        row['nuts']['ess'] = ess.tolist()
        row['nuts']['n_divergent'] = n_div

        for label in results:
            results[label].append(row[label])

        elapsed = time.time() - t_start
        print(f"dataset {b + 1}/{B} done, total_elapsed={elapsed:.1f}s, "
              f"covered={ {m: row[m]['covered'] for m in results} }, "
              f"nuts_rhat={np.round(rhat, 4)}, nuts_ess={np.round(ess, 0)}, nuts_div={n_div}, "
              f"stein_ess={ess_stein:.1f}", flush=True)

        if (b + 1) % 5 == 0 or b == B - 1:
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(results, f)
            print(f"  checkpointed to {CHECKPOINT_PATH}", flush=True)

    print(f"\nB={B} study complete in {time.time() - t_start:.1f}s")

    print("\n=== Coverage rate (per theta [a,b,c]) ===")
    for label, rows in results.items():
        covered = np.array([r['covered'] for r in rows])
        print(f"  {label:18s}: {covered.mean(axis=0)}")

    print("\n=== Mean CI width (per theta [a,b,c]) ===")
    for label, rows in results.items():
        widths = np.array([np.array(r['hi']) - np.array(r['lo']) for r in rows])
        print(f"  {label:18s}: {widths.mean(axis=0)}")

    print("\n=== Mean |mean - true theta| (per theta [a,b,c]) ===")
    for label, rows in results.items():
        errs = np.array([np.abs(np.array(r['mean']) - TRUE_THETA) for r in rows])
        print(f"  {label:18s}: {errs.mean(axis=0)}")

    print("\n=== NUTS diagnostics summary over 30 datasets ===")
    rhats = np.array([r['rhat'] for r in results['nuts']])
    esss = np.array([r['ess'] for r in results['nuts']])
    n_divs = np.array([r['n_divergent'] for r in results['nuts']])
    print(f"  mean R-hat per theta: {rhats.mean(axis=0)}, max R-hat: {rhats.max(axis=0)}")
    print(f"  mean ESS per theta: {esss.mean(axis=0)}, min ESS: {esss.min(axis=0)}")
    print(f"  total divergences: {n_divs.sum()} / {B * N_CHAINS * NUTS_SAMPLING}")

    print("\n=== Stein-reweighting ESS summary ===")
    stein_esss = np.array([r['ess_stein'] for r in results['reweighted_stein']])
    print(f"  mean ESS: {stein_esss.mean():.1f} / 800, min: {stein_esss.min():.1f}")
