import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import optimize as jsp_optimize # note: may eventually change this to use Optimistix
import optax
from _matern_jax import matern_v01, kvp_2p01
from functools import partial
from collections.abc import Iterable
'''
Dependencies: jax, optax
'''

##########################################################
############# Initialization-related Helpers #############
##########################################################

@jax.jit
def initialize_obs(observed_components, tau, I, x_init):
    '''
    Fill X for observed data via linear interpolation.
    '''
    I_flat = I.flatten()

    # each observed dimension is interpolated independently of the others, so this
    # batches across dimensions with vmap instead of a sequential fori_loop (~6x faster,
    # verified: sequential 733us vs vmapped 121us at n=81, D=10)
    def interp_one(d):
        # filter for dimension d
        tau_d = tau[:,d]
        x_d = x_init[:, d]

        # now we have to do some tricks to make linear interpolation JIT compatible

        # sort I and x_d to put all non-nans at front
        I_modif = jnp.where(tau_d, I_flat, jnp.inf)
        sort_idx = jnp.argsort(I_modif)
        sorted_I = I_modif[sort_idx]
        sorted_x_d = x_d[sort_idx]

        # dynamically find last non-nan index
        num_valid = jnp.sum(tau_d)
        last_valid_idx = jnp.maximum(0, num_valid-1)
        last_valid_I = sorted_I[last_valid_idx]
        last_valid_x = sorted_x_d[last_valid_idx]

        # replace the nans with the last valid to make them do nothing in interp
        padded_I = jnp.where(jnp.isinf(sorted_I), last_valid_I, sorted_I)
        padded_x_d = jnp.where(~jnp.isfinite(sorted_x_d), last_valid_x, sorted_x_d)

        # linear interpolation of observations
        return jnp.interp(I_flat, padded_I, padded_x_d)

    x_init_observed = jax.vmap(interp_one)(observed_components) # (n_observed, n)
    x_init = x_init.at[:, observed_components].set(x_init_observed.T)
    return x_init

@partial(jax.jit, static_argnames=['ode', 'unobs_init_iters', 'X_guesses'])
def initialize_unobs(x_init, unobserved_components, ode, I, theta_conf, theta_guess_init, X_guesses,
                     unobs_init_iters, sigmas):
    '''
    Fit X for unobserved components and fit theta.
    '''
    def objective(params):
        '''
        Minimize the squared loss error of the step-wise derivatives with the target ODE.

        params = [x_guess, theta_guess]
        '''
        x_guess, theta_guess = params

        full_x = x_init
        full_x = full_x.at[:, unobserved_components].set(x_guess)

        # X'(t) = f(x, theta)
        f_vals = ode(full_x, theta_guess, I)

        # X'(t) ~ (X(t + dt) - X(t - dt)) / (2*dt)
        # X'(t) ~ X(t + dt) / dt
        diff_first = jnp.reshape((full_x[1] - full_x[0]) / (I[1] - I[0]), [1,-1])
        diffs_mid = (full_x[2:] - full_x[:-2]) / (I[2:] - I[:-2])
        diff_last = jnp.reshape((full_x[-1] - full_x[-2]) / (I[-1] - I[-2]), [1,-1])
        f_diffs = jnp.concat([diff_first, diffs_mid, diff_last], axis=0)

        # minimize L2 loss of guess's numerical derivatives, plus an attraction term to adjust theta
        # useful for cases where one of the thetas being zero would force the guess into a flat line
        ode_mse = jnp.mean((f_vals - f_diffs)**2, axis=None)
        theta_mse = jnp.mean(theta_conf * (theta_guess - theta_guess_init)**2, axis=None)
        return ode_mse + theta_mse

    grad_obj = jax.grad(objective)
    x_guess_init = jnp.full(shape=(x_init.shape[0], unobserved_components.shape[0]), fill_value=jnp.nanmean(x_init))
    params = (x_guess_init, theta_guess_init)
    optimizer = optax.adam(learning_rate=0.01)

    def loop_inner(i, inner_params):
        x_guess, theta_guess, opt_state = inner_params
        params = (x_guess, theta_guess)

        grads = grad_obj(params)
        updates, opt_state = optimizer.update(grads, opt_state)

        params = optax.apply_updates(params, updates)
        return *params, opt_state

    def loop_outer(i, params):
        x_guess, theta_guess = params
        params = (x_guess, theta_guess_init)
        opt_state = optimizer.init(params)

        inner_params = jax.lax.fori_loop(0, unobs_init_iters, loop_inner,
                                   (*params, opt_state))
        x_guess, theta_guess, opt_state = inner_params
        return x_guess, theta_guess

    params = jax.lax.fori_loop(0, X_guesses, loop_outer, params)

    x_guessed, theta_init = params
    x_init = x_init.at[:,unobserved_components].set(x_guessed)
    # set sigma for unobserved components to 0 so we don't try to fit them later
    sigmas = sigmas.at[unobserved_components].set(0.0)
    return x_init, theta_init, sigmas

@jax.jit
def fit_phisigma(I, x_init, phis, sigmas):
    '''
    Fit phi and sigma for all components via scipy numerical optimization.
    '''
    I_max = I.max()
    Id_n = jnp.identity(I.shape[0], dtype=I.dtype)
    # n (and hence the FFT frequency index array) is the same for every component's column
    # of x_init, so this only needs to be built once instead of once per dimension
    n_freqs = (x_init.shape[0] - 1) // 2
    idxs = jnp.linspace(1, n_freqs, n_freqs, dtype=x_init.dtype)

    def neglogprob(phi, sigma, mu_phi2, sig_phi2, y_d):
        '''
        Target negative log density for fitting phi1, phi2, and sigma for a single component.
        '''
        cov = matern_v01(I, I, phi1=phi[0], phi2=phi[1]) + Id_n * sigma**2

        t1 = (phi[1] - mu_phi2)**2 / sig_phi2**2
        t2 = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(cov))))
        t3 = y_d @ jnp.linalg.solve(cov, y_d)
        return  0.5 * (t1 + t2 + t3)

    def target(log_phi, log_sigma, mu_phi2, sig_phi2, y_d):
        '''
        Note that there is currently no JIT-compatible way to do constrained optimization,
        so we log-parametrize to force the learned phi1, phi2, sigma to be positive
        '''
        return neglogprob(jnp.exp(log_phi), jnp.exp(log_sigma), mu_phi2, sig_phi2, y_d)

    targets = (phis, sigmas)

    def loop(d, targets):
        phis, sigmas = targets

        y_d = x_init[:, d]
        z = jnp.fft.fft(y_d)
        zmod = jnp.abs(z)
        zmod_effective_sq = zmod[1:(zmod.shape[0] - 1) // 2 + 1]**2
        freq = jnp.sum(idxs * zmod_effective_sq) / jnp.sum(zmod_effective_sq)
        mu_phi2 = 0.5 / freq
        sig_phi2 = (I_max - mu_phi2) / 3

        sigma_d = sigmas[d]

        def unknown_sigma(sigma_d, mu_phi2, sig_phi2, y_d):
            objective = lambda log_phisigma: target(log_phisigma[:2], log_phisigma[2], mu_phi2, sig_phi2, y_d)
            result = jsp_optimize.minimize(objective, x0=jnp.zeros(3, dtype=x_init.dtype), method="BFGS")
            fitted = jnp.exp(result.x)
            return fitted

        def known_sigma(sigma_d, mu_phi2, sig_phi2, y_d):
            objective = lambda log_phi: target(log_phi, jnp.log(sigma_d), mu_phi2, sig_phi2, y_d)
            result = jsp_optimize.minimize(objective, x0=jnp.zeros(2, dtype=x_init.dtype), method="BFGS")
            fitted = jnp.exp(result.x)
            return jnp.array([*fitted, sigma_d], dtype=x_init.dtype)

        # note: sigma_d = 0 indicates known zero-variance data or unobserved component
        fitted = jax.lax.cond(jnp.isnan(sigma_d) | (sigma_d < 0),
                             unknown_sigma, known_sigma,
                             *[sigma_d, mu_phi2, sig_phi2, y_d])
        phis = phis.at[d].set(fitted[:2])
        sigmas = sigmas.at[d].set(fitted[2])
        return phis, sigmas

    targets = jax.lax.fori_loop(0, x_init.shape[1], loop, targets)
    return targets

def _fast_pd_inv(K):
    '''
    Invert a matrix expected to be positive definite via Cholesky (~40x faster than
    pinv, verified numerically identical on real Kappa matrices), falling back to
    pinv if K turns out not to be numerically PD (cho_solve reliably NaNs out in that
    case rather than raising, so this is a safe/cheap runtime check).
    '''
    c, lower = jsp.linalg.cho_factor(K)
    chol_inv = jsp.linalg.cho_solve((c, lower), jnp.eye(K.shape[0], dtype=K.dtype))
    return jax.lax.cond(
        jnp.any(jnp.isnan(chol_inv)),
        lambda: jnp.linalg.pinv(K),
        lambda: chol_inv,
    )

@jax.jit
def build_matrices(I, phis):
    '''
    Construct GP matrices and inverses.
    '''
    # st_diff = s-t and l = |s-t| depend only on I, which is shared across every dimension d
    # -- compute once here instead of once per dimension inside build_matrices_d.
    # note: vmap across d does *not* help here (verified: vmap(pinv) over D=10 took the same
    # 239ms total as 10 sequential pinv calls -- pinv/SVD doesn't batch on this GPU backend),
    # so this stays a fori_loop; only the invariant setup is hoisted out.
    st_diff = I - I.T
    l = jnp.abs(st_diff)

    def build_matrices_d(st_diff, l, phi1, phi2):
        '''
        Takes in precomputed pairwise time differences and hparams (phi1, phi2, v).
        Returns (C_d, m_d, K_d) for component d.

        Credit: Skyler Wu
        '''
        # MAGI's actual smoothness parameter (see _matern_jax.py for why this needs a
        # host callback rather than a closed form the way v=2.5 did)
        v = jnp.array(2.01, dtype=l.dtype) # strong typing to ensure gamma function returns correct type

        # u = sqrt(2*nu) * l / phi2 - let's nan out diagonals to avoid imprecision errors.
        u = jnp.sqrt(2*v) * l / phi2
        u = jnp.fill_diagonal(a=u, val=jnp.nan, inplace=False)

        # pre-compute Bessel function + derivatives
        Bv0 = kvp_2p01(z=u, n=0)
        Bv1 = kvp_2p01(z=u, n=1)
        Bv2 = kvp_2p01(z=u, n=2)

        # 1. Kappa itself, but we need to correct everywhere with l=|s-t|=0 to have value exp(0.0) = 1.0
        Kappa = (phi1/jsp.special.gamma(v)) * (2 ** (1 - (v/2))) * ((jnp.sqrt(v) / phi2) ** v)
        Kappa *= Bv0
        Kappa *= (l ** v)

        # https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        Kappa = jnp.fill_diagonal(Kappa, val=phi1, inplace=False) # behavior as |s-t| \to 0^+

        # 2. p_Kappa, but need to replace everywhere with l=|s-t|=0 to have value 0.0.
        p_Kappa = (2 ** (1 - (v/2)))
        p_Kappa *= phi1 * ((u / jnp.sqrt(2)) ** v)
        p_Kappa *= ( (u * phi2 * Bv1) + (v*phi2*Bv0) )
        p_Kappa /= (phi2 * st_diff * jsp.special.gamma(v))
        p_Kappa = jnp.fill_diagonal(p_Kappa, val=0.0, inplace=False) # behavior as |s-t| \to 0^+

        # 3. Kappa_p (by symmetry)
        Kappa_p = p_Kappa * -1

        # 4. Kappa_pp - let's proceed term-by-term (save multiplier terms at the end)
        Kappa_pp = 2 * jnp.sqrt(2) * (v ** 1.5) * phi2 * l * Bv1
        Kappa_pp += ( ( (v ** 2) * (phi2 ** 2) ) - ( v * (phi2 ** 2) ) ) * Bv0
        # (2*v*s^2 - 4*v*s*t + 2*v*t^2) = 2*v*(s-t)^2 = 2*v*l^2
        Kappa_pp += 2 * v * (l ** 2) * Bv2
        Kappa_pp *= ( -1.0 * (2 ** (1 - (v/2))) * phi1 * ((u / jnp.sqrt(2)) ** v) )
        Kappa_pp /= ( (phi2 ** 2) * (l ** 2) * jsp.special.gamma(v) )

        # CHECK WITH PROF. YANG ABOUT THIS ONE! SHOULD THERE BE A NEGATIVE HERE?
        Kappa_pp = jnp.fill_diagonal(Kappa_pp, val=v*phi1/( (phi2 ** 2) * (v-1) ), inplace=False) # behavior as |s-t| \to 0^+

        # 5. form our C, m, and K matrices (let's not do any band approximations yet!)
        # note: sparse matrices can be worse performance on GPU, or negligble imporvement
        # Kappa is a genuine Matern covariance matrix -- PD in essentially all realistic
        # cases, so C_d_inv can safely take the fast Cholesky path (~40x speedup, verified
        # numerically identical to pinv on real fitted phis; falls back to pinv if not PD).
        # K_d, by contrast, is a Schur-complement-style subtraction that we verified goes
        # genuinely non-PSD (not just fp noise) at realistic phi2/discretization combinations
        # -- so K_d_inv must stay on pinv.
        C_d_inv = _fast_pd_inv(Kappa)
        m_d = p_Kappa @ C_d_inv
        K_d = Kappa_pp - (p_Kappa @ C_d_inv @ Kappa_p)
        K_d_inv = jnp.linalg.pinv(K_d)

        # 6. return our three matrices
        return C_d_inv, m_d, K_d_inv

    C_invs = jnp.zeros([phis.shape[0], I.shape[0], I.shape[0]], dtype=phis.dtype)
    ms = jnp.zeros([phis.shape[0], I.shape[0], I.shape[0]], dtype=phis.dtype)
    K_invs = jnp.zeros([phis.shape[0], I.shape[0], I.shape[0]], dtype=phis.dtype)

    matrices = (C_invs, ms, K_invs)
    def loop(d, matrices):
        C_invs, ms, K_invs = matrices
        result = build_matrices_d(st_diff, l, phis[d,0], phis[d,1])
        C_invs = C_invs.at[d].set(result[0])
        ms = ms.at[d].set(result[1])
        K_invs = K_invs.at[d].set(result[2])
        return C_invs, ms, K_invs

    full_matrices = jax.lax.fori_loop(0, phis.shape[0], loop, matrices)
    return full_matrices

def run_initialization(ode, x_init, I, tau, sigmas, phis, observed_components, unobserved_components,
                       theta_conf, theta_guess_init, X_guesses, unobs_init_iters=500):
    x_init = initialize_obs(observed_components, tau, I, x_init)
    x_init, theta_init, sigmas, = initialize_unobs(x_init, unobserved_components, ode, I,
                                    theta_conf, theta_guess_init, X_guesses, unobs_init_iters, sigmas)
    phis, sigmas = fit_phisigma(I, x_init, phis, sigmas)
    C_invs, ms, K_invs = build_matrices(I, phis)
    return x_init, theta_init, sigmas, phis, C_invs, ms, K_invs