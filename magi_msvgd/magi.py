import os
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from msvgd import MSVGD
from functools import partial

from _helpers import run_initialization

'''
Dependencies: jax, optax, msvgd
Additional helpers dependencies: numpy, scipy
'''

# Persistent on-disk compilation cache, CWD-based.
jax.config.update("jax_compilation_cache_dir", os.path.join(os.getcwd(), ".jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

class MAGI(MSVGD):
    def __init__(self, ode, data, theta_guess, sigmas=None,
                 theta_conf=0, X_guesses=1, unobs_init_iters=500,
                 mu=None, mu_dot=None, prior_temperature='default',
                 init_dtype='float64', init_device=None):
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
        init_dtype (str or dtype) : data type to be used for initialization, default: float64 (unstable at lower precision)
        init_device : jax.device used for initialization. Use .put() to move later for mSVGD
            (default: GPU if available, else CPU -- resolved lazily so this class can still be
            imported and used on machines with no GPU)
        '''
        if init_device is None:
            init_device = jax.devices()[0]

        # NOTE: we may want to do initialization on the CPU because fp64
        # is much more stable for constructing precomputed matrices,
        # which is sometimes faster on CPU
        if jnp.dtype(init_dtype) == jnp.float64:
            jax.config.update("jax_enable_x64", True)

        # ode: ((1, D), (p,), (1,)) -> (D,)
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
        self.Ns = tau.sum(axis=0, dtype=jnp.int32)
        self.N = self.Ns.sum().item()

        # dimension indices of observed components
        # consider > 2 observations to be observed, else can't fit matern kernel
        self.observed_components = jnp.where(self.Ns > 2)[0].astype(jnp.int32)
        self.unobserved_components = jnp.where(self.Ns <= 2)[0].astype(jnp.int32)

        # tau : n x D
        self.tau = tau

        self.phis = jnp.zeros([self.D, 2], dtype=init_dtype, device=init_device)
        if sigmas is None:
            self.sigmas = jnp.full(self.D, -1.0, dtype=init_dtype, device=init_device)
            self.unknown_sigmas = jnp.full(self.D, True, device=init_device)
        else:
            self.sigmas = jnp.array(sigmas, dtype=init_dtype, device=init_device)
            # unknown (needs Bayesian fitting) iff sigma wasn't given AND the component
            # is observed enough to fit it
            self.unknown_sigmas = jnp.logical_and(~(self.sigmas >= 0), self.Ns > 2)

        # run_initialization is fully JIT-compiled
        initializations = run_initialization(self.ode, self.x_init, self.I, self.tau,
                            self.sigmas, self.phis, self.observed_components, self.unobserved_components,
                            self.theta_conf, self.theta_guess, self.X_guesses, self.unobs_init_iters)
        # force this to actually execute now to avoid dtype casting race conditions
        jax.block_until_ready(initializations)
        self.x_init = initializations[0] # (n, d)
        self.theta_init = initializations[1] # (p,)
        self.sigmas = initializations[2] # (n_unknown,)
        self.phis = initializations[3] # (d,2)
        self.C_invs = initializations[4] # (d, n, n)
        self.ms = initializations[5] # (d, n, n)
        self.K_invs = initializations[6] # (d, n, n)

        self.particles_init = jnp.concatenate([self.theta_init, self.x_init.flatten(), self.sigmas[self.unknown_sigmas]])
        self.particles = None
        # tracks whether the user has explicitly chosen a dtype/device via put() -- if not,
        # solve() will default to float32 for speed (fp64 is ~3x+ slower for the einsum-heavy
        # mSVGD gradient, and much worse than that on non-datacenter GPUs)
        self._put_called = False

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
            with jax.default_matmul_precision("highest"):
                theta = particle[:self.p] # (p,)
                X = particle[self.p:self.p+self.n*self.D].reshape(self.n, self.D) # (n, d)
                sigmas = self.sigmas.at[self.unknown_sigmas].set(jnp.clip(particle[self.p+self.n*self.D:], min=1e-5)) # (d,)
                # fully-unobserved (Ns=0) dimensions carry a placeholder sigma (0.0) that's never fit
                safe_sigmas = jnp.where(self.Ns > 0, sigmas, 1.0) # (d,)

                diff_X    = X - self.mu # (n, D)
                resid_obs = jnp.where(self.tau, X - self.x_init, 0.0) # (n, D)
                ode_resid = (self.ode(X, theta, self.I)
                             - self.mu_dot
                             - jnp.einsum('dnm,md->nd', self.ms, diff_X)) # (n, D)

                Cinv_x    = jnp.einsum('dnm,md->nd', self.C_invs, diff_X) # (n, D)
                Kinv_r    = jnp.einsum('dnm,md->nd', self.K_invs, ode_resid) # (n, D)

                gp_term   = jnp.sum(diff_X * Cinv_x) # scalar
                log_norm  = jnp.sum(self.Ns * jnp.log(2 * jnp.pi * safe_sigmas**2)) # scalar
                obs_term  = jnp.sum(resid_obs**2 / safe_sigmas**2) # scalar
                ode_term  = jnp.sum(ode_resid * Kinv_r) # scalar

            return -0.5 * (self.beta_inv * gp_term + log_norm + obs_term + self.beta_inv * ode_term)
        super().__init__(magi_logdensity)
        self.logdensity = magi_logdensity


    def put(self, dtype=jnp.float32, device=None):
            '''
            Move everything to new device.
            device : default GPU if available, else CPU (resolved lazily).
            '''
            if device is None:
                device = jax.devices()[0]
            if jnp.dtype(dtype) == jnp.float64:
                jax.config.update("jax_enable_x64", True)
            else:
                jax.config.update("jax_enable_x64", False)
            for attr, val in self.__dict__.items():
                if isinstance(val, jax.Array):
                    if jnp.issubdtype(val.dtype, jnp.floating):
                        val = jnp.astype(val, dtype)
                    setattr(self, attr, jax.device_put(val, device))
            self._put_called = True


    def unpack_particles(self, particles):
        thetas = particles[:,:self.p]
        Xs = particles[:,self.p:self.p+self.n*self.D].reshape(particles.shape[0], self.n, self.D)
        sigmas = particles[:,self.p+self.n*self.D:]

        return Xs, thetas, sigmas
    
        
    def solve(
        self,
        k=200,
        sigma_init=0.2,
        mitosis_splits=0,
        random_seed=8,
        optimizer=optax.adam,
        optimizer_kwargs={"learning_rate": 0.1},
        batch_size=None,
        is_MAP=False,
        max_iter=10_000,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        grad_clip=None,
        monitor_convergence=0,
    ):
        '''
        Solve mSVGD optimization for MAGI.

        Arguments
        ----------
        k                   : int, number of initial particles
        sigma_init          : float, standard deviation for sampling initial state
        Note: If self.particles is not None, solve() will use previous results by default. Set to None to reset.
        
        mitosis_splits      : number of particle-doubling steps
        random_seed         : a jax.random key to sample mitosis jitters

        Note: The following arguments may each be passed as a single value to be used globally
            or as a list of length `mitosis_splits+1`, containing (possibly different) values for each mitosis phase.
        optimizer           : an optax optimizer constructor, or list thereof, configured for descent
        optimizer_kwargs    : dict of kwargs passed to the optimizer, or list thereof
            Warning : It is necessary in some case for optimizer kwargs to have the same dtype as x0,
                e.g. {"learning_rate" : jnp.array(0.1, dtype=x0.dtype)}
        batch_size          : int or list of ints (one per phase) for batched optimization, None for full dataset
        is_MAP              : bool or list of bools for whether to mode-find using on the gradient of only the logdensity
        max_iter            : int or list of ints (one per phase)
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidths (-1 = median heuristic)
        grad_clip           : float or list of floats (one per phase), max global norm for the particle
            gradient before the optimizer step, None to disable. Useful to guard against exploding
            updates in batched/stochastic optimization.

        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each mitosis split, < 0 = fully silence)

        Note: if put() has not already been called, solve() defaults to float32 (call
        put(dtype=jnp.float64, ...) beforehand if you want float64 sampling instead).
        '''
        if not self._put_called:
            self.put(dtype=jnp.float32)

        init_key, msvgd_key = jr.split(jr.key(random_seed))
        if self.particles is None:
            particles = self.particles_init + jr.normal(init_key, shape=(k, self.particles_init.shape[0])) * sigma_init
            if self.unknown_sigmas.sum() > 0: # ensure non-negative sigma initializations
                sigma_slice = slice(self.p + self.n * self.D, None)
                particles = particles.at[:, sigma_slice].set(jnp.abs(particles[:, sigma_slice]))
            self.particles = particles

        self.particles = super().solve(x0=self.particles,
            mitosis_splits=mitosis_splits,
            random_seed=msvgd_key,
            data=None,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=None,
            is_MAP=is_MAP,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
            bandwidth=bandwidth,
            grad_clip=grad_clip,
            monitor_convergence=monitor_convergence)
        
        return self.unpack_particles(self.particles)

    def nuts(self, random_seed=8, warmup_steps=1000, sampling_steps=9000):
        import blackjax
        rng_key, warmup_key, sample_key = jax.random.split(jr.key(random_seed), 3)

        warmup = blackjax.window_adaptation(blackjax.nuts, self.logdensity)
        (state, parameters), _ = warmup.run(warmup_key, position=self.particles_init, num_steps=warmup_steps)
        
        kernel = blackjax.nuts(self.logdensity, **parameters)
        self.nuts_final_state, self.nuts_history = blackjax.util.run_inference_algorithm(sample_key, kernel, initial_state=state, num_steps=sampling_steps)

        return self.unpack_particles(self.nuts_history[0].position)