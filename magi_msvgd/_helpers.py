import numpy as np
import scipy as sp
from sklearn.gaussian_process import kernels as skl_kernels
from tqdm.notebook import trange
from collections.abc import Iterable
'''
Dependencies: numpy, scipy, sklearn, tqdm
'''

class MaternKernel():
    '''
    Different implementations of the Matern kernel.
    tf_kernel is MUCH slower, especially for initialize_obs. This may be because tensorflow converts the
    numpy arrays back and forth from tensors and transfers back and forth from the GPU.
    '''
    def __init__(self, phi1, phi2, v):
        self.skl_kernel = phi1 * skl_kernels.Matern(length_scale=phi2, nu=v)
    def eval(self, x, y=None):
        if y is None:
            y = x
        return self.skl_kernel(x, y)

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
        interpolator = sp.interpolate.interp1d(I_d, y_d, fill_value='extrapolate')
        x_init_d = interpolator(solver.I.flatten())
        solver.x_init[:,d] = x_init_d

def initialize_unobs(solver):
    '''
    Fit X for unobserved components and fit theta.
    Modifies in place.
    '''
    # convert numpy arrays to tensors
    x_tensor = solver.to_tensor(solver.x_init, dtype="float32")
    I_tensor = solver.to_tensor(solver.I, dtype="float32")
    theta_set_guess = solver.to_tensor(solver.theta_guess, dtype="float32")
    theta_conf = solver.to_tensor(solver.theta_conf, dtype="float32")

    def objective(x_guess, theta_guess):
        '''
        Minimize the squared loss error of the step-wise derivatives with the target ODE.
        '''
        full_x = solver.clone(x_tensor)
        full_x = solver.update_tensor(full_x, solver.unobserved_components, x_guess)
        
        # X'(t) = f(x, theta)
        f_vals = solver.ode(full_x, theta_guess, I_tensor)
        
        # X'(t) ~ (X(t + dt) - X(t - dt)) / (2*dt)
        # X'(t) ~ X(t + dt) / dt
        diff_first = solver.reshape((full_x[[1]] - full_x[[0]]) / (I_tensor[1] - I_tensor[0]), [1,-1])
        diffs_mid = (full_x[2:] - full_x[:-2]) / (I_tensor[2:] - I_tensor[:-2])
        diff_last = solver.reshape((full_x[[-1]] - full_x[[-2]]) / (I_tensor[-1] - I_tensor[-2]), [1,-1])
        f_diffs = solver.concat([diff_first, diffs_mid, diff_last], axis=0)
        
        # minimize L2 loss of guess's numerical derivatives, plus an attraction term to adjust theta
        # useful for cases where one of the thetas being zero would force the guess into a flat line
        ode_mse = solver.tensor_mean((f_vals - f_diffs)**2, axis=None)
        theta_mse = solver.tensor_mean(theta_conf * (theta_guess - theta_set_guess)**2, axis=None)
        return ode_mse + theta_mse

    x_guess_init = np.ones(shape=(solver.n, len(solver.unobserved_components))) * np.nanmean(solver.x_init)
    for i in range(solver.X_guess):
        # set up tensors for optimization
        x_guess0 = solver.to_tensor(x_guess_init, dtype="float32", requires_grad=True)
        theta_guess0 = solver.to_tensor(solver.theta_guess, dtype="float32", requires_grad=True)
        
        opt = solver.Adam([x_guess0, theta_guess0], lr=0.01)
        last_loss = 0
        for j in trange(10_000, desc="Computing X_unobs and theta initialization"):
            loss = solver.autograd(objective, [x_guess0, theta_guess0], opt)
            # set a stopping condition if loss decreases by <= 0.1
            if j % 200 == 0:
                if np.abs(last_loss - loss) <= 0.1:
                    break
                else:
                    last_loss = loss
        x_guess_init = solver.to_arr(x_guess0)
                
    # store the solved starting state guesses
    solver.x_init[:,solver.unobserved_components] = solver.to_arr(x_guess0).astype(float)
    solver.theta_init = solver.to_arr(theta_guess0).astype(float)

    # set sigma for unobserved components to -1 so we don't try to fit them later
    solver.sigmas[solver.unobserved_components] = -1.

def fit_phisigma(solver, v=2.01):
    '''
    Fit phi and sigma for all components via scipy numerical optimization.
    Modifies in place.
    '''
    def neglogprob(phi, sigma_d, mu_phi2, sig_phi2, y_d, I_d):
        '''
        phi[0] : phi1
        phi[1] : phi2
        sigma : (optional) sigma

        Target negative log density for fitting phi1, phi2, and sigma
        '''
        Kappa_phi = MaternKernel(phi1=phi[0], phi2=phi[1], v=v)
    
        cov = Kappa_phi.eval(I_d) + np.identity(I_d.shape[0])*sigma_d**2

        t1 = 1/sig_phi2**2 * (phi[1] - mu_phi2)**2
        t2 = np.linalg.slogdet(cov).logabsdet
        t3 = y_d.T @ np.linalg.pinv(cov) @ y_d
        return  0.5 * (t1 + t2 + t3)

    solver.x_init = np.array(solver.x_init, dtype=np.float64)
    for d in range(solver.D):
        # tau_d = solver.tau[d].flatten()
        # I_d = solver.I[tau_d].reshape(-1, 1)
        # y_d = solver.x_init[tau_d, d]
        
        I_d = solver.I
        y_d = solver.x_init[:, d]
        
        # set phi_2 prior
        z = sp.fft.fft(y_d)
        zmod = np.abs(z)
        zmod_effective_sq = zmod[1:(len(zmod) - 1) // 2 + 1]**2
        idxs = np.linspace(1, len(zmod_effective_sq), len(zmod_effective_sq))
        freq = np.sum(idxs * zmod_effective_sq) / np.sum(zmod_effective_sq)
        mu_phi2 = 0.5 / freq; sig_phi2 = (solver.I.max() - mu_phi2) / 3
    
        # use scipy.optimize to fit phi and sigma
        # fit based on interpolated points, rather than only observed points
        method = 'Nelder-Mead'
        
        if (not solver.sigmas[d]) or (np.isnan(solver.sigmas[d])):
            # fit sigma if it is not specified
            target = lambda phisigma: neglogprob(phisigma[:2], phisigma[2], mu_phi2, sig_phi2, y_d, I_d)
            fitted = sp.optimize.minimize(target, x0=np.ones(3),
                                          bounds=[(1e-10, np.inf)]*3, method=method).x
            solver.phis[d] = fitted[:2]; solver.sigmas[d] = fitted[2]
        else:
            # fit just phi, holding sigma constant
            sigma_d = max(0, solver.sigmas[d])
            target = lambda phi: neglogprob(phi, sigma_d, mu_phi2, sig_phi2, y_d, I_d)
            fitted = sp.optimize.minimize(target, x0=np.ones(2),
                                          bounds=[(1e-10, np.inf)]*2, method=method).x
            solver.phis[d] = fitted

def build_matrices(solver, v=2.01):
    '''
    Construct GP matrices and inverses.
    '''
    def build_matrices_d(I, phi1, phi2, v=v):
        '''
        Takes in discretized timesteps I and hparams (phi1, phi2, v). Returns (C_d, m_d, K_d) for component d.
        - I is an np.array of discretized timesteps, phi1 & phi2 are floats.
    
        Credit: Skyler Wu
        '''
        # tile appropriately to facilitate vectorization
        s = np.tile(A=I, reps=I.shape[0]); t = s.T
    
        # l = |s-t|, u = sqrt(2*nu) * l / phi2 - let's nan out diagonals to avoid imprecision errors.
        l = np.abs(s - t); u = np.sqrt(2*v) * l / phi2; np.fill_diagonal(a=u, val=np.nan)
    
        # pre-compute Bessel function + derivatives
        Bv0 = sp.special.kvp(v=v, z=u, n=0)
        Bv1 = sp.special.kvp(v=v, z=u, n=1)
        Bv2 = sp.special.kvp(v=v, z=u, n=2)
    
        # 1. Kappa itself, but we need to correct everywhere with l=|s-t|=0 to have value exp(0.0) = 1.0
        Kappa = (phi1/sp.special.gamma(v)) * (2 ** (1 - (v/2))) * ((np.sqrt(v) / phi2) ** v)
        Kappa *= Bv0
        Kappa *= (l ** v)
        
        # https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        np.fill_diagonal(Kappa, val=phi1) # behavior as |s-t| \to 0^+
    
        # 2. p_Kappa, but need to replace everywhere with l=|s-t|=0 to have value 0.0.
        p_Kappa = (2 ** (1 - (v/2)))
        p_Kappa *= phi1 * ((u / np.sqrt(2)) ** v)
        p_Kappa *= ( (u * phi2 * Bv1) + (v*phi2*Bv0) )
        p_Kappa /= (phi2 * (s-t) * sp.special.gamma(v))
        np.fill_diagonal(p_Kappa, val=0.0) # behavior as |s-t| \to 0^+
    
        # 3. Kappa_p (by symmetry)
        Kappa_p = p_Kappa * -1
    
        # 4. Kappa_pp - let's proceed term-by-term (save multiplier terms at the end)
        Kappa_pp = 2 * np.sqrt(2) * (v ** 1.5) * phi2 * l * Bv1
        Kappa_pp += ( ( (v ** 2) * (phi2 ** 2) ) - ( v * (phi2 ** 2) ) ) * Bv0
        Kappa_pp += ( (2 * v * (s ** 2)) - (4 * v * s * t) + (2 * v * (t ** 2)) ) * Bv2
        Kappa_pp *= ( -1.0 * (2 ** (1 - (v/2))) * phi1 * ((u / np.sqrt(2)) ** v) )
        Kappa_pp /= ( (phi2 ** 2) * (l ** 2) * sp.special.gamma(v) )
        
        # CHECK WITH PROF. YANG ABOUT THIS ONE! SHOULD THERE BE A NEGATIVE HERE?
        np.fill_diagonal(Kappa_pp, val=v*phi1/( (phi2 ** 2) * (v-1) )) # behavior as |s-t| \to 0^+
    
        # 5. form our C, m, and K matrices (let's not do any band approximations yet!)
        C_d_inv = np.linalg.pinv(Kappa)
        m_d = p_Kappa @ C_d_inv
        K_d = Kappa_pp - (p_Kappa @ C_d_inv @ Kappa_p)
        K_d_inv = np.linalg.inv(K_d)
        
        # 6. return our three matrices
        return C_d_inv, m_d, K_d_inv

    # Compute and save matrices for all components
    solver.C_invs, solver.ms, solver.K_invs = [np.array(mats) for mats in \
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
