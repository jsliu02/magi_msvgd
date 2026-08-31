'''
Follow-up to test_stein_reweight.py: investigate three fixes for the over-dispersion seen
when stacking beta=0.5 tempering with Stein-reweighting (CI widths 110-118% of the NUTS gold
standard, worse than tempering alone at 86-106%):

  1. More moderate tempering: sweep beta over {1.0, 0.9, 0.8, 0.7, 0.6, 0.5} instead of just
     {1.0, 0.5, 0.3}, to find where dispersion actually crosses 100% of NUTS width per
     parameter (uniform weights only, no reweighting, to isolate tempering's own effect).
  2. More SVGD particles: repeat a couple of the above at k_final=1600 (vs. 800) to see
     whether a bigger, less redundant particle set makes the Stein QP better-conditioned
     (higher ESS after reweighting, less risk of concentrating on a few tail particles).
  3. Different kernel for reweighting: IMQ (inverse multiquadric, Gorham & Mackey 2017)
     instead of RBF for the Stein kernel Gram matrix used in reweighting. IMQ is the
     kernel the KSD literature recommends for detecting/avoiding the kind of degenerate
     "satisfy the discrepancy via a few outlying particles" failure mode discussed for RBF.

Compares everything against the diagnostics-verified NUTS gold standard saved by
test_nuts_gold_standard.py (nuts_gold_standard.npz) -- no NUTS re-run needed.

Run: python test_stein_reweight_v2.py
fp32 for MSVGD (matches prior tempering experiment); gold standard itself was computed fp64.
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
    return jax.vmap(jax.grad(solver.magi_logdensity))(particles)


def stein_kernel_matrix(solver, particles, score, kernel='rbf', beta_imq=-0.5):
    '''
    General Stein kernel Gram matrix M (k,k), M[i,j] = k_p(x_i,x_j), for a radial base kernel
    k(x,y) = f(||x-y||^2):

        k_p(x,y) = f(s)*s(x).s(y) - 2f'(s)*(s_i-s_j).(x_i-x_j) - 4*s*f''(s) - 2*D*f'(s)
        (s = ||x-y||^2, s(.) = score = grad log p)

    which reduces to the standard RBF-KSD formula (Liu, Lee & Jordan 2016) for f(s)=exp(-s/h),
    and to the IMQ-KSD formula (Gorham & Mackey 2017) for f(s)=(c^2+s)^beta_imq,
    beta_imq in (-1,0) (here -0.5, the standard choice). IMQ is recommended over RBF in the
    KSD literature specifically because RBF-based KSD can fail to detect (or can be gamed by)
    mass escaping to the tails, whereas IMQ with beta_imq in (-1,0) provably controls this --
    directly relevant to the over-dispersion failure mode from reweighting particles drawn
    from an already tail-heavy tempered proposal.
    '''
    k, dim = particles.shape
    L2sq, h = solver.pairwise_distance(particles, -1)  # h is RBF's own median-heuristic bandwidth
    L2sq = jnp.clip(L2sq, min=0.0)  # guard against tiny fp-cancellation negatives (matters for IMQ's fractional power)

    S = score
    X = particles
    SS = S @ S.T
    A = S @ X.T
    diag_sx = jnp.sum(S * X, axis=1)
    cross = diag_sx[:, None] + diag_sx[None, :] - A - A.T  # (s_i-s_j).(x_i-x_j)

    if kernel == 'rbf':
        f = jnp.exp(-L2sq / h)
        fprime = -f / h
        fpp = f / h**2
    elif kernel == 'imq':
        upper = L2sq[jnp.triu_indices(k, k=1)]
        c2 = jnp.median(jnp.clip(upper, min=1e-6))  # median-heuristic analog for IMQ's c^2
        base = c2 + L2sq
        f = base ** beta_imq
        fprime = beta_imq * base ** (beta_imq - 1)
        fpp = beta_imq * (beta_imq - 1) * base ** (beta_imq - 2)
    else:
        raise ValueError(kernel)

    trace_term = -4.0 * L2sq * fpp - 2.0 * dim * fprime
    M = f * SS - 2.0 * fprime * cross + trace_term
    return M


def optimize_stein_weights(M, n_steps=3000, lr=0.05, verbose=True):
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

    w = jax.nn.softmax(logits)
    ess = float(1.0 / jnp.sum(w**2))
    if verbose:
        print(f"    final KSD^2 = {loss_fn(logits):.4e}, ESS = {ess:.1f} / {k}")
    return w, ess


def weighted_quantile(values, weights, q):
    order = jnp.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = jnp.cumsum(w_sorted) - 0.5 * w_sorted
    cdf = cdf / jnp.sum(w_sorted)
    return jnp.interp(q, cdf, v_sorted)


def summarize(theta_samples, weights=None):
    if weights is None:
        weights = jnp.full(theta_samples.shape[0], 1.0 / theta_samples.shape[0])
    mean = jnp.sum(theta_samples * weights[:, None], axis=0)
    var = jnp.sum(weights[:, None] * (theta_samples - mean) ** 2, axis=0)
    std = jnp.sqrt(var)
    lo = jnp.stack([weighted_quantile(theta_samples[:, d], weights, 0.025) for d in range(3)])
    hi = jnp.stack([weighted_quantile(theta_samples[:, d], weights, 0.975) for d in range(3)])
    return np.array(mean), np.array(std), np.array(lo), np.array(hi)


def build_solver():
    full = load_data()
    solver = MAGI(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
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
    print(f"  {label}: solve done in {elapsed:.1f}s ({solver.particles.shape[0]} particles)")
    return thetas


def report(name, mean, std, lo, hi, nuts_mean, nuts_std, nuts_width, ess=None):
    width = hi - lo
    mean_err = np.abs(mean - nuts_mean)
    std_err = np.abs(std - nuts_std)
    width_pct = 100 * width / nuts_width
    ess_str = f", ESS={ess:.0f}" if ess is not None else ""
    print(f"  {name:40s}: |mean-NUTS|={mean_err}, |std-NUTS|={std_err}, "
          f"width%NUTS={width_pct}{ess_str}")
    return dict(mean=mean, std=std, lo=lo, hi=hi, width_pct=width_pct,
                mean_err=mean_err, std_err=std_err, ess=ess)


if __name__ == '__main__':
    gold = np.load('nuts_gold_standard.npz')
    nuts_mean, nuts_std = gold['theta_mean'], gold['theta_std']
    nuts_lo, nuts_hi = gold['theta_ci_lo'], gold['theta_ci_hi']
    nuts_width = nuts_hi - nuts_lo
    print(f"Gold standard: mean={nuts_mean}, std={nuts_std}, width={nuts_width}\n")

    results = {}

    # ------------------------------------------------------------------
    # 1. Moderate tempering sweep, k_final=800, uniform weights only
    # ------------------------------------------------------------------
    print("=== 1. Moderate tempering sweep (uniform weights, k_final=800) ===")
    for beta in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        solver = build_solver()
        thetas = run_msvgd(solver, f"beta={beta}", beta=beta, k_final=800)
        mean, std, lo, hi = summarize(thetas)
        results[f"beta={beta}_k800_uniform"] = report(
            f"beta={beta}, k=800, uniform", mean, std, lo, hi, nuts_mean, nuts_std, nuts_width)

    # ------------------------------------------------------------------
    # 2. More particles: k_final=1600 at beta=1.0 and beta=0.7 (moderate pick)
    # ------------------------------------------------------------------
    print("\n=== 2. More particles (k_final=1600, uniform weights) ===")
    for beta in [1.0, 0.7]:
        solver = build_solver()
        thetas = run_msvgd(solver, f"beta={beta}, k=1600", beta=beta, k_final=1600)
        mean, std, lo, hi = summarize(thetas)
        results[f"beta={beta}_k1600_uniform"] = report(
            f"beta={beta}, k=1600, uniform", mean, std, lo, hi, nuts_mean, nuts_std, nuts_width)

        print("  Stein-reweighting (RBF)...")
        score = true_score(solver, solver.particles)
        M_rbf = stein_kernel_matrix(solver, solver.particles, score, kernel='rbf')
        w_rbf, ess_rbf = optimize_stein_weights(M_rbf)
        mean, std, lo, hi = summarize(thetas, w_rbf)
        results[f"beta={beta}_k1600_rbf"] = report(
            f"beta={beta}, k=1600, RBF-reweighted", mean, std, lo, hi, nuts_mean, nuts_std, nuts_width, ess_rbf)

    # ------------------------------------------------------------------
    # 3. Kernel comparison for reweighting: RBF vs IMQ, at beta=1.0 and beta=0.7, k_final=800
    # ------------------------------------------------------------------
    print("\n=== 3. Kernel comparison for reweighting (k_final=800) ===")
    for beta in [1.0, 0.7]:
        solver = build_solver()
        thetas = run_msvgd(solver, f"beta={beta}, k=800 (for kernel comparison)", beta=beta, k_final=800)

        score = true_score(solver, solver.particles)
        for kernel in ['rbf', 'imq']:
            print(f"  Stein-reweighting ({kernel.upper()})...")
            M = stein_kernel_matrix(solver, solver.particles, score, kernel=kernel)
            w, ess = optimize_stein_weights(M)
            mean, std, lo, hi = summarize(thetas, w)
            results[f"beta={beta}_k800_{kernel}"] = report(
                f"beta={beta}, k=800, {kernel.upper()}-reweighted", mean, std, lo, hi,
                nuts_mean, nuts_std, nuts_width, ess)

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:28s}: width%NUTS={r['width_pct']}")
