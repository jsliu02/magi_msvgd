import os
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


import sys
sys.path.append("../../msvgd/msvgd/")

from msvgd import MSVGD
from functools import partial
from _initializer import run_initialization

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

        def magi_logdensity(particle, data_batch):
            '''
            Full MAGI log-density. (n*d + p + n_unknown_sigmas:,) -> scalar

            data_batch : dict bundling mu, mu_dot, C_invs, ms, K_invs, tau, x_init, I, sigmas,
                Ns -- passed as an explicit (non-batched, shared-across-particles) MSVGD `data`
                argument instead of closed over from `self`. A closed-over jnp array gets
                embedded in the compiled program as a literal HLO constant, which (a) makes
                compile time scale with its size and (b) makes the compiled executable
                un-shareable across different MAGI instances (different datasets/fits) even at
                identical n/D/p/dtype/particle-count shapes, since the embedded literal
                differs -- every new instance pays a full fresh compile. This matters a lot for
                simulation studies that create many MAGI instances over different simulated
                datasets: measured >60s/replication with data closed over (mostly compile
                time) at a shape that otherwise runs in ~4s once compiled once. Passing data as
                an explicit jit argument makes the compiled executable purely a function of
                shape/dtype, so the (already-enabled) persistent compile cache can skip the
                dominant backend-codegen cost for every replication after the first.
                self.p/self.n/self.D/self.beta_inv/self.ode stay closed over (Python
                scalars/a function, not array data). self.unknown_sigmas also stays closed
                over rather than moving into data_batch: it's used below as a *boolean index*
                (`.at[unknown_sigmas].set(...)`), which JAX requires to be concrete at trace
                time -- a traced/dynamic boolean index raises NonConcreteBooleanIndexError.
                It's also tiny (D-length), so leaving it closed over costs nothing toward the
                goal above.
            '''
            mu, mu_dot, C_invs, ms, K_invs, tau, x_init, I, sigmas0, Ns = (
                data_batch['mu'], data_batch['mu_dot'], data_batch['C_invs'], data_batch['ms'],
                data_batch['K_invs'], data_batch['tau'], data_batch['x_init'], data_batch['I'],
                data_batch['sigmas'], data_batch['Ns'])

            # Wraps the WHOLE body, not just the matmul-looking parts. The einsums below are
            # batched matrix-vector contractions, which are unaffected by this setting in the
            # forward pass -- but their VJPs are matrix-matrix products, which on tensor-core
            # hardware are computed in reduced precision by default. Since the gradient is what
            # every sampler actually consumes, dropping this costs ~4 significant digits there:
            # measured on FitzHugh-Nagumo in float32, the gradient's relative error against a
            # float64 reference goes from 5.1e-7 to 3.9e-3. The forward log-density value alone
            # is bit-identical either way, so the value is not the thing to test.
            with jax.default_matmul_precision("highest"):
                theta = particle[:self.p] # (p,)
                X = particle[self.p:self.p+self.n*self.D].reshape(self.n, self.D) # (n, d)
                sigmas = sigmas0.at[self.unknown_sigmas].set(jnp.clip(particle[self.p+self.n*self.D:], min=1e-5)) # (d,)
                # fully-unobserved (Ns=0) dimensions carry a placeholder sigma (0.0) that's never fit
                safe_sigmas = jnp.where(Ns > 0, sigmas, 1.0) # (d,)

                diff_X    = X - mu # (n, D)
                resid_obs = jnp.where(tau, X - x_init, 0.0) # (n, D)
                ode_resid = (self.ode(X, theta, I)
                             - mu_dot
                             - jnp.einsum('dnm,md->nd', ms, diff_X)) # (n, D)

                Cinv_x    = jnp.einsum('dnm,md->nd', C_invs, diff_X) # (n, D)
                Kinv_r    = jnp.einsum('dnm,md->nd', K_invs, ode_resid) # (n, D)

                gp_term   = jnp.sum(diff_X * Cinv_x) # scalar
                log_norm  = jnp.sum(Ns * jnp.log(2 * jnp.pi * safe_sigmas**2)) # scalar
                obs_term  = jnp.sum(resid_obs**2 / safe_sigmas**2) # scalar
                ode_term  = jnp.sum(ode_resid * Kinv_r) # scalar

            return -0.5 * (self.beta_inv * gp_term + log_norm + obs_term + self.beta_inv * ode_term)
        self._sync_data()
        super().__init__(magi_logdensity, data=self.data)
        # nuts() needs a plain 1-arg logdensity_fn (blackjax's API has no data-argument slot);
        # this wrapper wires the current self.data through by closure, so it doesn't get the
        # cross-instance compile-sharing benefit above, but nuts() is typically run standalone
        # rather than across many simulated-dataset replications.
        self.magi_logdensity = lambda particle: magi_logdensity(particle, self.data)


    def _sync_data(self):
        '''
        Rebuild the MSVGD `data` bundle (passed as an explicit jit argument to
        magi_logdensity, see its docstring) from the current top-level GP-matrix/observation
        attributes. `data` holds separate references, not aliases, to those attributes, so
        it must be refreshed whenever they change -- currently only put() does that (casting
        dtype / moving device), so put() calls this at the end.
        '''
        self.data = {
            'mu': self.mu, 'mu_dot': self.mu_dot,
            'C_invs': self.C_invs, 'ms': self.ms, 'K_invs': self.K_invs,
            'tau': self.tau, 'x_init': self.x_init, 'I': self.I,
            'sigmas': self.sigmas, 'Ns': self.Ns,
        }

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
                if attr == 'data':
                    continue  # rebuilt from the (about-to-be-updated) top-level attrs below
                if isinstance(val, jax.Array):
                    if jnp.issubdtype(val.dtype, jnp.floating):
                        val = jnp.astype(val, dtype)
                    setattr(self, attr, jax.device_put(val, device))
            self._sync_data()
            self._put_called = True


    def unpack_particles(self, particles):
        thetas = particles[:,:self.p]
        Xs = particles[:,self.p:self.p+self.n*self.D].reshape(particles.shape[0], self.n, self.D)
        sigmas = particles[:,self.p+self.n*self.D:]

        return Xs, thetas, sigmas
    
        
    def solve(
        self,
        k=200,
        sigma_init=0.01,
        k_schedule=None,
        random_seed=8,
        monitor_convergence=0,
        optimizer=optax.contrib.prodigy,
        optimizer_kwargs=dict(),
        batch_size=None,
        is_MAP=False,
        max_iter=10_000,
        atol=1e-2,
        rtol=1e-8,
        bandwidth=-1,
        grad_clip=None,
        reweighted_kernel=True,
    ):
        '''
        Solve mSVGD optimization for MAGI.

        Arguments
        ----------
        k                   : int, number of initial particles
        sigma_init          : float, standard deviation for sampling initial state
        Note: If self.particles is not None, solve() will use previous results by default. Set to None to reset.

        k_schedule          : int, list of ints, or None (default). None runs one phase at k
            particles with no growth. Otherwise each entry is the particle count after one
            covariance-matched split, giving len(k_schedule) splits and len(k_schedule)+1
            phases; entries must strictly increase, the first exceeding k. An int is shorthand
            for a single split.
        random_seed         : a jax.random key to sample the mitotic splits
        monitor_convergence : int — print max grad every N iterations
            (0 = print status after each phase, < 0 = fully silence)

        ----------
        Note: each argument below takes either one value used for every phase, or a list of
            n_phases values (one per phase). A list of the wrong length is an error.

        optimizer           : an optax optimizer constructor, configured for descent
        optimizer_kwargs    : dict of kwargs passed to the optimizer
            Warning : some optimizer kwargs must share the particle dtype,
                e.g. {"learning_rate" : jnp.array(0.1, dtype=jnp.float32)}
        batch_size          : int for batched optimization, None for the full dataset
        is_MAP              : bool, mode-find on the logdensity gradient alone (no SVGD kernel).
            Prefer map_solve() if the mode itself is what you want: this path is first-order and
            stalls well short of it on a posterior this ill-conditioned.
        max_iter            : int, iteration cap for the phase
        atol, rtol          : convergence tolerances,  all(grad <= atol + rtol * particles)
        bandwidth           : RBF bandwidth (-1 = median heuristic)
        grad_clip           : float, max global norm for the particle gradient before the
            optimizer step, None to disable. Guards against exploding updates in
            batched/stochastic optimization.
        reweighted_kernel   : bool, default True here. Uses the density-reweighted kernel
            (Huang, Dong, Fang [2023], see MSVGD._reweighted_svgd_update) rather than the
            standard joint RBF kernel. Amplifies repulsion in low-density regions, countering
            SVGD's variance-collapse; gave the best credible-interval calibration of the
            corrective techniques tried on this problem. No effect when is_MAP is True.

        ----------
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
            k_schedule=k_schedule,
            random_seed=msvgd_key,
            data=None,
            monitor_convergence=monitor_convergence,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=None,
            is_MAP=is_MAP,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
            bandwidth=bandwidth,
            grad_clip=grad_clip,
            reweighted_kernel=reweighted_kernel)

        return self.unpack_particles(self.particles)

        
    def nuts(self, random_seed=8, warmup_steps=1000, sampling_steps=9000):
        import blackjax
        rng_key, warmup_key, sample_key = jax.random.split(jr.key(random_seed), 3)

        warmup = blackjax.window_adaptation(blackjax.nuts, self.magi_logdensity)
        (state, parameters), _ = warmup.run(warmup_key, position=self.particles_init, num_steps=warmup_steps)
        
        kernel = blackjax.nuts(self.magi_logdensity, **parameters)
        self.nuts_final_state, self.nuts_history = blackjax.util.run_inference_algorithm(sample_key, kernel, initial_state=state, num_steps=sampling_steps)

        return self.unpack_particles(self.nuts_history[0].position)


    def map_solve(self, x0=None, **kwargs):
        '''
        Find the MAP by Gauss-Newton on the least-squares form of the log-density.

        Thin wrapper over gauss_newton.GaussNewtonMAP, which is where the method and its
        rationale are documented. Prefer it over solve(is_MAP=True): that path is first-order and
        stalls well short of the mode on a posterior conditioned around 1e4, returning a point
        that is not stationary. The solver instance is cached on self, so repeated calls reuse
        one compilation; construct GaussNewtonMAP directly for finer control.

        Returns the unpacked (Xs, thetas, sigmas) at the mode and leaves the particle on
        self.map_particle. Does not touch self.particles.
        '''
        from gauss_newton import GaussNewtonMAP
        if getattr(self, "_gn", None) is None:
            self._gn = GaussNewtonMAP(self)
        out = self._gn.solve(x0=x0, **kwargs)
        self.map_particle = self._gn.map_particle
        return out
