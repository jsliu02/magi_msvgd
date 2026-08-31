'''
Prototype: post-hoc Stein importance reweighting (Liu & Lee, 2017, "Black-Box Importance
Sampling") applied to MAGI's standard (joint-kernel) MSVGD particles on the FHN benchmark
(y.csv, true theta = [0.2, 0.2, 3.0]).

Idea: MSVGD particles {x_i} are only an approximate sample from the posterior p. Instead of
treating them as equally-weighted, solve for weights w on the simplex that minimize the
kernelized Stein discrepancy (KSD) of the weighted empirical measure:

    min_w  w^T M w   s.t.  w >= 0, sum(w) = 1

where M[i,j] = k_p(x_i, x_j) is the (joint, full-dimensional) Stein kernel Gram matrix built
from the same score function and RBF kernel/bandwidth MSVGD already uses. M is PSD, so this
is a convex QP; solved here via softmax-reparameterized gradient descent (Adam) rather than
pulling in a QP solver dependency, since we don't need an exact optimum, just a good one.

Compares theta posteriors on the same dataset:
  1. NUTS (gold standard, via blackjax)
  2. MSVGD, uniform weights (baseline)
  3. MSVGD, Stein-reweighted
  4. Tempered MSVGD (solve against p(x)^beta, beta<1, to flatten the target and encourage
     broader exploration), uniform weights
  5. Tempered MSVGD, Stein-reweighted against the TRUE (beta=1) target's score -- this is
     the "fixed-beta-then-reweight" idea: use tempering only to get particles to spread out
     more, then correct back to the real posterior via the same KSD-minimizing weights as
     #3, but computed with the untempered score. Note this reweights against the true
     target regardless of what generated the particles, so no importance-ratio p^(1-beta)
     or intractable p^beta normalizer is needed.

Run: python test_stein_reweight.py
fp32 throughout.
'''

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

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


def true_score(solver, particles):
    '''
    grad log p(x) under the TRUE (beta=1) target, via solver.magi_logdensity -- a plain 1-arg
    closure over solver.data set once at MAGI construction time, so it's unaffected by any
    later override of solver.logdensity used to temper the MSVGD solve itself.
    '''
    return jax.vmap(jax.grad(solver.magi_logdensity))(particles)


def stein_kernel_matrix(solver, particles, score):
    '''
    Full-dimensional (not sliced) Stein kernel Gram matrix M (k, k), M[i,j] = k_p(x_i, x_j),
    for the RBF base kernel with solver's own median-heuristic bandwidth (via
    solver.pairwise_distance, which uses the sum-of-squares expansion instead of a naive
    (k,k,dim) broadcast -- avoids an ~830MB intermediate at k=800, dim=325). Standard KSD
    kernel (e.g. Liu, Lee & Jordan 2016 Eq. 5):

        k_p(x,y) = K(x,y) * [ s(x).s(y) + (2/h)(s(x)-s(y)).(x-y) + 2D/h - (4/h^2)||x-y||^2 ]

    where s = grad log p (the TRUE target's score, regardless of what generated `particles`),
    K(x,y) = exp(-||x-y||^2/h). Passing in `score` explicitly (rather than recomputing it
    from `particles` via the solver) is what makes this valid for reweighting particles that
    came from a tempered solve: `score` must always be the TRUE target's score for the
    resulting weights to minimize KSD against the real posterior.
    '''
    k, dim = particles.shape

    L2sq, h = solver.pairwise_distance(particles, -1)
    K = jnp.exp(-L2sq / h)

    S = score
    X = particles
    SS = S @ S.T                      # (k,k) s_i . s_j
    A = S @ X.T                       # (k,k) s_i . x_j
    diag_sx = jnp.sum(S * X, axis=1)  # (k,) s_i . x_i

    cross = diag_sx[:, None] + diag_sx[None, :] - A - A.T  # (s_i - s_j).(x_i - x_j)

    M = K * (SS + (2.0 / h) * cross + 2.0 * dim / h - (4.0 / h**2) * L2sq)
    return M


def optimize_stein_weights(M, n_steps=3000, lr=0.05, seed=0):
    k = M.shape[0]
    logits = jnp.zeros(k)
    opt = optax.adam(lr)
    opt_state = opt.init(logits)

    def loss_fn(logits):
        w = jax.nn.softmax(logits)
        return w @ M @ w

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for step in range(n_steps):
        loss, grad = loss_and_grad(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
        if step % 500 == 0:
            print(f"  [stein-weights] step {step}, KSD^2 = {loss:.6e}")

    w = jax.nn.softmax(logits)
    print(f"  [stein-weights] final KSD^2 = {loss_fn(logits):.6e}, "
          f"ESS = {1.0 / jnp.sum(w**2):.1f} / {k}")
    return w


def weighted_quantile(values, weights, q):
    '''values: (k,), weights: (k,) summing to 1, q: scalar in [0,1]. Per-dimension caller.'''
    order = jnp.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = jnp.cumsum(w_sorted) - 0.5 * w_sorted  # midpoint rule
    cdf = cdf / jnp.sum(w_sorted)
    return jnp.interp(q, cdf, v_sorted)


def summarize(name, theta_samples, weights=None):
    if weights is None:
        weights = jnp.full(theta_samples.shape[0], 1.0 / theta_samples.shape[0])
    mean = jnp.sum(theta_samples * weights[:, None], axis=0)
    var = jnp.sum(weights[:, None] * (theta_samples - mean) ** 2, axis=0)
    std = jnp.sqrt(var)
    lo = jnp.stack([weighted_quantile(theta_samples[:, d], weights, 0.025) for d in range(3)])
    hi = jnp.stack([weighted_quantile(theta_samples[:, d], weights, 0.975) for d in range(3)])
    covered = (lo <= TRUE_THETA) & (TRUE_THETA <= hi)
    print(f"\n=== {name} ===")
    print(f"mean = {np.array(mean)}")
    print(f"std  = {np.array(std)}")
    print(f"95% CI lo = {np.array(lo)}")
    print(f"95% CI hi = {np.array(hi)}")
    print(f"covered   = {np.array(covered)}")
    return mean, std, lo, hi


def build_solver():
    full = load_data()
    solver = MAGI(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    solver.put(dtype=jnp.float32, device=jax.devices()[0])
    return solver


def run_msvgd(solver, label, beta=1.0, k=200, k_final=800, max_iter=1000, random_seed=0):
    '''
    beta < 1 tempers the target solver.solve() actually optimizes against (flattens p(x) to
    p(x)^beta by scaling the log-density), by overriding solver.logdensity right before the
    first solve() call on this instance -- solver.gradient was jitted once in __init__ and
    reads self.logdensity dynamically at (first) trace time, so this works as long as it's
    set before any solve()/gradient() call. solver.magi_logdensity (used by true_score and
    nuts()) is a separate closure fixed at construction time and is unaffected by this.
    '''
    if beta != 1.0:
        original_logdensity = solver.logdensity
        solver.logdensity = lambda x, d: beta * original_logdensity(x, d)

    print(f"\nRunning {label}...")
    start = time.time()
    Xs, thetas, sigmas = solver.solve(
        k=k, sigma_init=0.01, k_final=k_final,
        optimizer=optax.contrib.prodigy, optimizer_kwargs={},
        atol=0.1, rtol=0, max_iter=max_iter, random_seed=random_seed,
        monitor_convergence=-1,
    )
    jax.block_until_ready(thetas)
    print(f"  done in {time.time() - start:.1f}s")
    return thetas


def stein_reweight(solver, particles):
    print("Computing Stein kernel matrix (against TRUE target) and optimizing weights...")
    start = time.time()
    score = true_score(solver, particles)
    M = stein_kernel_matrix(solver, particles, score)
    weights = optimize_stein_weights(M)
    print(f"  done in {time.time() - start:.1f}s")
    return weights


if __name__ == '__main__':
    results = {}

    solver = build_solver()
    thetas = run_msvgd(solver, "MSVGD (joint kernel, beta=1)")
    results["uniform"] = summarize("MSVGD (uniform weights)", thetas)
    weights = stein_reweight(solver, solver.particles)
    results["stein"] = summarize("MSVGD (Stein-reweighted)", thetas, weights)

    for beta in (0.5, 0.3):
        solver_t = build_solver()
        thetas_t = run_msvgd(solver_t, f"Tempered MSVGD (beta={beta})", beta=beta)
        results[f"tempered_beta={beta}_uniform"] = summarize(
            f"Tempered MSVGD beta={beta} (uniform weights)", thetas_t)
        weights_t = stein_reweight(solver_t, solver_t.particles)
        results[f"tempered_beta={beta}_stein"] = summarize(
            f"Tempered MSVGD beta={beta} (Stein-reweighted)", thetas_t, weights_t)

    print("\nRunning NUTS (gold standard)...")
    start = time.time()
    nuts_Xs, nuts_thetas, nuts_sigmas = solver.nuts(random_seed=0, warmup_steps=1000, sampling_steps=4000)
    jax.block_until_ready(nuts_thetas)
    print(f"  done in {time.time() - start:.1f}s")

    nuts_mean, nuts_std, nuts_lo, nuts_hi = summarize("NUTS (gold standard)", nuts_thetas)
    nuts_width = nuts_hi - nuts_lo

    print("\n=== Comparison to NUTS gold standard ===")
    for name, (mean, std, lo, hi) in results.items():
        mean_err = jnp.abs(mean - nuts_mean)
        std_err = jnp.abs(std - nuts_std)
        width_pct = 100 * (hi - lo) / nuts_width
        print(f"{name:32s}: |mean-NUTS|={np.array(mean_err)}, |std-NUTS|={np.array(std_err)}, "
              f"CI width % of NUTS={np.array(width_pct)}")
