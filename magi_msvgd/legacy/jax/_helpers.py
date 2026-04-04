import jax
import jax.numpy as jnp
import jax.scipy as jsp
# from jaxopt import ScipyBoundedMinimize
import numpy as np
import scipy as sp
import optax
from sklearn.gaussian_process import kernels as skl_kernels
from tqdm.notebook import trange
from collections.abc import Iterable
'''
Dependencies: jax, optax, numpy, scipy, sklearn, tqdm
'''

def initialize_obs(solver):
    '''
    Fill X for observed data via linear interpolation.
    Modifies in place.
    '''
    # iterate over only observed components
    for d in solver.observed_components:
        # filter for dimension d
        tau_d = solver.tau[d]
        I_d = solver.I[tau_d].flatten()
        y_d = solver.x_init[tau_d, d].flatten()

        # linear interpolation of observations
        x_init_d = jnp.interp(solver.I.flatten(), I_d, y_d)
        solver.x_init = solver.x_init.at[:,d].set(x_init_d)

def initialize_unobs(solver):
    '''
    Fit X for unobserved components and fit theta.
    Modifies in place.

    Partially JIT compiled (gradient of objective).
    '''
    def objective(params):
        '''
        Minimize the squared loss error of the step-wise derivatives with the target ODE.

        params = [x_guess, theta_guess]
        '''
        x_guess, theta_guess = params

        full_x = solver.x_init.copy()
        full_x = full_x.at[:,solver.unobserved_components].set(x_guess)

        # X'(t) = f(x, theta)
        f_vals = solver.ode(full_x, theta_guess, solver.I)

        # X'(t) ~ (X(t + dt) - X(t - dt)) / (2*dt)
        # X'(t) ~ X(t + dt) / dt
        diff_first = jnp.reshape((full_x[1] - full_x[0]) / (solver.I[1] - solver.I[0]), [1,-1])
        diffs_mid = (full_x[2:] - full_x[:-2]) / (solver.I[2:] - solver.I[:-2])
        diff_last = jnp.reshape((full_x[-1] - full_x[-2]) / (solver.I[-1] - solver.I[-2]), [1,-1])
        f_diffs = jnp.concat([diff_first, diffs_mid, diff_last], axis=0)

        # minimize L2 loss of guess's numerical derivatives, plus an attraction term to adjust theta
        # useful for cases where one of the thetas being zero would force the guess into a flat line
        ode_mse = jnp.mean((f_vals - f_diffs)**2, axis=None)
        theta_mse = jnp.mean(solver.theta_conf * (theta_guess - solver.theta_guess)**2, axis=None)
        return ode_mse + theta_mse

    grad_obj = jax.jit(jax.value_and_grad(objective))
    x_guess_init = jnp.full(shape=(solver.n, len(solver.unobserved_components)), fill_value=jnp.nanmean(solver.x_init),
                           dtype=solver.init_dtype, device=solver.init_device)
    for i in range(solver.X_guess):
        x_guess0 = x_guess_init
        theta_guess0 = solver.theta_guess.copy()

        params = [x_guess0, theta_guess0]
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(params)
        last_loss = 0
        for j in trange(10_000, desc="Computing X_unobs and theta initialization"):
            loss, grads = grad_obj(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # set a stopping condition if loss decreases by <= 0.1
            if j % 200 == 0:
                if jnp.abs(last_loss - loss) <= 0.1:
                    break
                else:
                    last_loss = loss
        x_guess_init = params[0]

    # store the solved starting state guesses
    solver.x_init = solver.x_init.at[:,solver.unobserved_components].set(params[0])
    solver.theta_init = params[1]

    # set sigma for unobserved components to -1 so we don't try to fit them later
    solver.sigmas = solver.sigmas.at[solver.unobserved_components].set(-1)

def pairwise_sq_distances(x, y):
    return jnp.sum((x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :])**2, axis=-1)

# def Matern(x, y, phi1, phi2, v):
#     def _matern(r, phi2):
#         def _scipy_matern(r, phi2):
#             r = np.where(r == 0, 1e-10, r)  # avoid 0 in bessel
#             s = np.sqrt(2 * v) * r / phi2
#             return (2 ** (1 - v) / sp.special.gamma(v)
#                     * s ** v * sp.special.kv(v, s))
#         return jax.pure_callback(
#             lambda r: _scipy_matern(r).astype(r.dtype),
#             jax.ShapeDtypeStruct(r.shape, r.dtype), r, phi2)
#     r = jnp.sqrt(pairwise_sq_distances(x, y))
#     return phi1 * _matern(r, phi2)

class MaternKernel():
    '''
    Struggling to write Matern Kernel purely in JAX. The above implementation (commented out)
    works, but does not play well with jaxopt.ScipyBoundedMinimize, which is used to solve for
    phi and sigma. Reverted to sklearn + scipy implementation.
    '''
    def __init__(self, phi1, phi2, v):
        self.skl_kernel = phi1 * skl_kernels.Matern(length_scale=phi2, nu=v)
    def eval(self, x, y=None):
        if y is None:
            y = x
        return self.skl_kernel(x, y)

def fit_phisigma(solver, v=2.01):
    '''
    Fit phi and sigma for all components via scipy numerical optimization.
    Modifies in place.

    Not JIT compiled. Trying to do so was an enormous mess.
    '''
    I = np.array(solver.I)
    I_max = I.max()
    Id_n = np.identity(I.shape[0])

    def neglogprob(phi, sigma_d, mu_phi2, sig_phi2, y_d):
        '''
        phi[0] : phi1
        phi[1] : phi2
        sigma : (optional) sigma

        Target negative log density for fitting phi1, phi2, and sigma
        '''
        Kappa_phi = MaternKernel(phi1=phi[0], phi2=phi[1], v=v)
        cov = Kappa_phi.eval(I) + Id_n * sigma_d**2

        t1 = (phi[1] - mu_phi2)**2 / sig_phi2**2
        t2 = 2 * np.sum(jnp.log(np.diag(np.linalg.cholesky(cov))))
        t3 = y_d @ np.linalg.solve(cov, y_d)
        return  0.5 * (t1 + t2 + t3)

    solver.x_init = np.array(solver.x_init)
    for d in range(solver.D):
        y_d = solver.x_init[:, d]

        # set phi_2 prior
        # note jax fft only works on CPU
        z = sp.fft.fft(y_d)
        zmod = np.abs(z)
        zmod_effective_sq = zmod[1:(len(zmod) - 1) // 2 + 1]**2
        idxs = np.linspace(1, len(zmod_effective_sq), len(zmod_effective_sq))
        freq = np.sum(idxs * zmod_effective_sq) / jnp.sum(zmod_effective_sq)
        mu_phi2 = 0.5 / freq; sig_phi2 = (I_max - mu_phi2) / 3

        # use scipy.optimize to fit phi and sigma
        # fit based on interpolated points, rather than only observed points
        # method = 'Nelder-Mead'
        method = 'Nelder-Mead'
        sigma_d = solver.sigmas[d]

        if (not sigma_d) or (sigma_d != sigma_d) or (sigma_d < 0):
            # fit sigma if it is not specified
            target = lambda phisigma: neglogprob(phisigma[:2], phisigma[2], mu_phi2, sig_phi2, y_d)
            fitted = sp.optimize.minimize(target, x0=np.ones(3), bounds=[(1e-10, np.inf)]*3, method=method).x
            solver.phis[d] = fitted[:2]
            solver.sigmas = solver.sigmas.at[d].set(fitted[2])
        else:
            # fit just phi, holding sigma constant
            target = lambda phi: neglogprob(phi, sigma_d, mu_phi2, sig_phi2, y_d)
            fitted = sp.optimize.minimize(target, x0=np.ones(2),
                                          bounds=[(1e-10, np.inf)]*2, method=method).x
            solver.phis[d] = fitted

def kvp(v, z, n):
    result_shape = jax.ShapeDtypeStruct(z.shape, z.dtype)
    return jax.pure_callback(lambda v, z, n: sp.special.kvp(v=v, z=z, n=n).astype(z.dtype), result_shape, v, z, n)

def build_matrices(solver, v=2.01):
    '''
    Construct GP matrices and inverses.
    '''
    @jax.jit
    def build_matrices_d(I, phi1, phi2, v=v):
        '''
        Takes in discretized timesteps I and hparams (phi1, phi2, v). Returns (C_d, m_d, K_d) for component d.
        - I is an jnp.array of discretized timesteps, phi1 & phi2 are floats.

        Credit: Skyler Wu
        '''
        # tile appropriately to facilitate vectorization
        s = jnp.tile(A=I, reps=I.shape[0]); t = s.T

        # l = |s-t|, u = sqrt(2*nu) * l / phi2 - let's nan out diagonals to avoid imprecision errors.
        l = jnp.abs(s - t); u = jnp.sqrt(2*v) * l / phi2
        u = jnp.fill_diagonal(a=u, val=jnp.nan, inplace=False)

        # pre-compute Bessel function + derivatives
        Bv0 = kvp(v=v, z=u, n=0)
        Bv1 = kvp(v=v, z=u, n=1)
        Bv2 = kvp(v=v, z=u, n=2)

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
        p_Kappa /= (phi2 * (s-t) * jsp.special.gamma(v))
        p_Kappa = jnp.fill_diagonal(p_Kappa, val=0.0, inplace=False) # behavior as |s-t| \to 0^+

        # 3. Kappa_p (by symmetry)
        Kappa_p = p_Kappa * -1

        # 4. Kappa_pp - let's proceed term-by-term (save multiplier terms at the end)
        Kappa_pp = 2 * jnp.sqrt(2) * (v ** 1.5) * phi2 * l * Bv1
        Kappa_pp += ( ( (v ** 2) * (phi2 ** 2) ) - ( v * (phi2 ** 2) ) ) * Bv0
        Kappa_pp += ( (2 * v * (s ** 2)) - (4 * v * s * t) + (2 * v * (t ** 2)) ) * Bv2
        Kappa_pp *= ( -1.0 * (2 ** (1 - (v/2))) * phi1 * ((u / jnp.sqrt(2)) ** v) )
        Kappa_pp /= ( (phi2 ** 2) * (l ** 2) * jsp.special.gamma(v) )

        # CHECK WITH PROF. YANG ABOUT THIS ONE! SHOULD THERE BE A NEGATIVE HERE?
        Kappa_pp = jnp.fill_diagonal(Kappa_pp, val=v*phi1/( (phi2 ** 2) * (v-1) ), inplace=False) # behavior as |s-t| \to 0^+

        # 5. form our C, m, and K matrices (let's not do any band approximations yet!)
        C_d_inv = jnp.linalg.pinv(Kappa)
        m_d = p_Kappa @ C_d_inv
        K_d = Kappa_pp - (p_Kappa @ C_d_inv @ Kappa_p)
        K_d_inv = jnp.linalg.inv(K_d)

        # 6. return our three matrices
        return C_d_inv, m_d, K_d_inv

    # Compute and save matrices for all components
    solver.C_invs, solver.ms, solver.K_invs = [jnp.array(mats, dtype=solver.init_dtype, device=solver.init_device) for mats in \
                zip(*[build_matrices_d(solver.I, phi[0], phi[1], v=2.01) for phi in solver.phis])]

def listify(val, length):
    '''
    Prepare a numerical/iterable argument for mitosis splits.
    '''
    if isinstance(val, Iterable) and type(val) is not dict:
        if len(val) == length: return val
        else: raise ValueError(f"Incorrect gradient descent hyperparameter argument length, got {len(val)}, expecting {length}.")
    else:
        return [val] * length

def jnp_pad(array, axis=-1):
    return jnp.expand_dims(array, axis=axis)