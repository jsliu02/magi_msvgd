import jax
import jax.numpy as jnp
import jax.random as jr
from msvgd import MSVGD
from functools import partial

from _helpers import run_initialization

'''
Dependencies: jax, optax, numpy, tqdm
Additional helpers dependencies: jaxopt, scipy, sklearn
'''
class MAGI(MSVGD):
    def __init__(self, ode, data, theta_guess, sigmas=None,
                 theta_conf=0, X_guesses=1, unobs_init_iters=500,
                 mu=None, mu_dot=None, prior_temperature='default',
                 init_device=jax.devices('cpu')[0], init_dtype='float32'):
        '''
        Initializing theta and unobserved components is done using acceleration library via autograd.

        Xs : n x D
        thetas : p
        t : n or None

        NOTE: ode should be written for a single observation at a single time point.

        ARGUMENTS:
        ode (function, (Xs, thetas, t) -> n x D) : ODE system
        data (array, n x (D+1)) : observed data, column 0 is the discretization index I, record NaN for unobserved points
        theta_guess (array, p) : initial guess for theta

        OPTIONAL:
        sigmas (array or None) : observation noise standard deviation (if known); individual entries can be set to nan
        theta_conf (float or array) : confidence in initial guess for theta, larger theta_conf will pull theta initialization toward guess
        X_guesses (int) : number of times to run X initialization procedure, can give more stable results
        unobs_init_iters (int) : number of Adam steps when solving for initialization of theta and unobserved components
        mu (array, n x D) : prior mean function evaluated at discretization index I
        mu_dot (array, n x D) : derivative of prior mean function with respect to time, evaluated at I

        temper_prior (float) : prior tempering factor, default: beta = Dn/N
        init_dtype (str or dtype) : data type to be used for initialization, default: float32
        '''
        # validate dtype
        # self.init_dtype = init_dtype
        # self.init_device = init_device
        # NOTE: we do initialization on the CPU since
        ## initializations are relatively non-parallel and seem to be faster on CPU
        ## TODO: test on beefier hardware (if GPU outperforms, add user-facing toggle switch)

        # save ode function and its gradients, as well as map versions that apply over dim 0
        # use mapped version to apply to the entire batch of particles

        # ode: ((n, D), (p,), (n,)) -> (D,)
        self.ode = jax.vmap(ode, in_axes=(0, None, 0))

        # I: n x 1
        self.I = jnp.array(data[:,0], dtype=init_dtype, device=init_device).reshape(-1, 1)

        # x_init: n x D
        # contains NaNs where unobserved, will later be filled
        # we do not need to store raw y, since we use boolean mask tau and x_init
        self.x_init = jnp.array(data[:,1:], dtype=init_dtype, device=init_device)

        # number of discretization points
        self.n = self.I.shape[0]
        # number of dimensions in the ODE
        self.D = self.x_init.shape[1]

        # theta guess for initialization
        self.theta_guess = jnp.array(theta_guess, dtype=init_dtype, device=init_device)
        # confidence level:
            # positive to force theta toward guess
            # negative to force theta away from guess
        self.theta_conf = jnp.array(theta_conf, dtype=init_dtype, device=init_device)
        # number of parameters in theta
        self.p = len(theta_guess)

        self.X_guesses = X_guesses
        self.unobs_init_iters = unobs_init_iters

        # boolean mask for observed data
        tau = jnp.isfinite(self.x_init)

        # number of data observations, shape = (D,)
        self.Ns = tau.sum(axis=0)
        self.N = self.Ns.sum().item()

        # dimension indices of observed components
        # consider > 2 observations to be observed, else can't fit matern kernel
        self.observed_components = jnp.where(self.Ns > 2)[0]
        self.unobserved_components = jnp.where(self.Ns <= 2)[0]

        # tau : n x D
        self.tau = tau

        self.phis = jnp.zeros([self.D, 2], dtype=init_dtype, device=init_device)
        if sigmas is None:
            self.sigmas = jnp.full(self.D, -1.0, dtype=init_dtype, device=init_device)
            self.unknown_sigmas = jnp.full(self.D, True, device=init_device)
        else:
            self.sigmas = jnp.array(sigmas, dtype=init_dtype, device=init_device)
            self.unknown_sigmas = jnp.where((self.sigmas >= 0) & (self.Ns > 2), False, True)
            if len(self.unknown_sigmas) == 0:
                self.unknown_sigmas = None

        # run_initialization is fully JIT-compiled
        initializations = run_initialization(self.ode, self.x_init, self.I, self.tau,
                            self.sigmas, self.phis, self.observed_components, self.unobserved_components,
                            self.theta_conf, self.theta_guess, self.X_guesses, self.unobs_init_iters)
        self.x_init = initializations[0] # (n, d)
        self.theta_init = initializations[1] # (p,)
        self.sigmas = initializations[2] # (n_unknown,)
        self.phis = initializations[3] # (d,2)
        self.C_invs = initializations[4] # (d, n, n)
        self.ms = initializations[5] # (d, n, n)
        self.K_invs = initializations[6] # (d, n, n)

        self.particles_init = jnp.concatenate([self.theta_init, self.x_init.flatten(), self.sigmas[self.unknown_sigmas]])

        # set GP mean priors
        # mu, mu_dot: n x D
        if mu is not None:
            self.mu = jnp.array(mu, dtype=init_dtype, device=init_device)
            self.mu_dot = jnp.array(mu_dot, dtype=init_dtype, device=init_device)
        else:
            self.mu = jnp.zeros([self.n, self.D], dtype=init_dtype, device=init_device)
            self.mu_dot = jnp.zeros([self.n, self.D], dtype=init_dtype, device=init_device)

        # set prior tempering
        if prior_temperature.lower() == 'default':
            self.beta_inv = self.N / (self.D * self.n)
        else:
            self.beta_inv = prior_temperature

        def magi_logdensity(particle):
            '''
            Full MAGI log-density. (n*d + p + n_unknown_sigmas:,) -> scalar
            '''
            # unpack particle
            theta = particle[:self.p] # (p,)
            X = particle[self.p:self.p+self.n*self.D].reshape(self.n, self.D) # (n, d)
            sigmas = self.sigmas.at[self.unknown_sigmas].set(jnp.clip(particle[self.p+self.n*self.D:], min=1e-5)) # (d,)

            ode_eval = self.ode(X, theta, self.I.flatten())                         # (n, D)
        
            log_p = 0.0
            for d in range(self.D):
                x_d       = X[:, d]                                                 # (n,)
                mu_d      = self.mu[:, d]                                           # (n,)
                mu_dot_d  = self.mu_dot[:, d]                                       # (n,)
                tau_d     = self.tau[:, d]                                          # (n,) bool
                y_d       = self.x_init[:, d]                                       # (n,)
                sigma_d   = sigmas[d]                                               # scalar
                N_d       = self.Ns[d]                                              # scalar
        
                diff_d    = x_d - mu_d                                              # (n,)
                resid_d   = jnp.where(tau_d, x_d - y_d, 0.0)                       # (n,)
                ode_resid_d = ode_eval[:, d] - mu_dot_d - self.ms[d] @ diff_d      # (n,)
        
                gp_term_d   = diff_d    @ self.C_invs[d] @ diff_d                   # scalar
                ode_term_d  = ode_resid_d @ self.K_invs[d] @ ode_resid_d           # scalar
                obs_term_d  = jnp.sum(resid_d**2) / sigma_d**2                     # scalar
                log_norm_d  = N_d * jnp.log(2 * jnp.pi * sigma_d**2)              # scalar
        
                log_p += (1/self.beta_inv) * gp_term_d \
                       + log_norm_d \
                       + obs_term_d \
                       + (1/self.beta_inv) * ode_term_d
        
            return -0.5 * log_p

        super().__init__(magi_logdensity)


    def device_put(self, device=jax.devices('gpu')[0]):
            '''
            Move everything to new device.
            '''
            for attr, val in self.__dict__.items():
                if isinstance(val, jax.Array):
                    setattr(self, attr, jax.device_put(val, device))
    # def solve(
    #     self,
    #     k=200,
    #     sigma_0=0.2,
    #     mitosis_splits=0,
    #     random_seed=8,
    #     optimizer=None,
    #     optimizer_kwargs={"learning_rate": 1e-2},
    #     max_iter=10_000,
    #     atol=1e-2,
    #     rtol=1e-8,
    #     bandwidth=-1,
    #     monitor_convergence=0,
    # ):
    #     '''
    #     Solve mSVGD optimization for MAGI.

    #     Arguments
    #     ----------
    #     k                   : int, number of initial particles
    #     sigma_init          : float, standard deviation for sampling initial state
    #     mitosis_splits      : number of particle-doubling steps
    #     key                 : a jax.random key to sample mitosis jitters

    #     Note: The following arguments may each be passed as a single value to be used globally
    #         or as a list of length `mitosis_splits+1`, containing (different) values for each mitosis phase.
    #     optimizer           : an optax optimizer constructor, or list thereof
    #     optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
    #     max_iter            : int or list of ints (one per phase)
    #     atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
    #     bandwidth           : RBF bandwidths (-1 = median heuristic)

    #     monitor_convergence : int — print max grad every N iterations
    #         (0 = print status after each mitosis split, < 0 = fully silence)
    #     '''