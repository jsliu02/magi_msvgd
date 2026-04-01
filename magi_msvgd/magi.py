import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import optax

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.notebook import trange

from . import _helpers as helpers
'''
Dependencies: jax, optax, numpy, scipy, sklearn, tqdm
'''
class baseMAGISolver():
    def __init__(self, ode, data, theta_guess, dfdx=None, dfdtheta=None, sigmas=None,
                 theta_conf=0, X_guess=1, mu=None, mu_dot=None, pos_X=False, pos_theta=False,
                 prior_temperature='default', bayesian_sigma=True, init_dtype='float64'):
        '''
        Initialization is all mostly in Numpy, and class variables are all stored as Numpy arrays.
        This a polymorphic base class that can be used to build an SVGD module on many libraries.

        Fitting theta and unobserved components is done using acceleration library via autograd.

        Xs : n x D
        thetas : p
        t : n or None

        NOTE: ode, dfdx, dfdtheta should be written for a single observation at a single time point.

        ARGUMENTS:
        ode (function, (Xs, thetas, t) -> n x D) : ODE system
        data (array, n x (D+1)) : observed data, column 0 is the discretization index I, record NaN for unobserved points
        theta_guess (array, p) : initial guess for theta

        OPTIONAL:
        dfdx (function, (Xs, thetas, t) -> n x D x D) : gradient of ODE with respect to X (autograd if not provided)
        dfdtheta (function, (Xs, thetas, t) -> n x p x D) : gradient of ODE with respect to theta (autograd if not provided)
        sigmas (array or None) : observation noise standard deviation, if known; individual entries can be set to None
        theta_conf (float or array) : confidence in initial guess for theta, larger theta_conf will pull theta initialization toward guess
        X_guess (int) : number of times to run X initialization procedure, can give more stable results
        mu (array, n x D) : prior mean function evaluated at discretization index I
        mu_dot (array, n x D) : derivative of prior mean function with respect to time, evaluated at I
        pos_X (bool) : whether to restrict X to strictly positive values (PyTorch only)
        pos_theta (bool) : whether to restrict theta to strictly positive values (PyTorch only)
        temper_prior (float) : prior tempering factor, default: beta = Dn/N
        bayesian_sigma (bool) : whether to give Bayesian treatment to sigma or fix at initial value
        init_dtype (str or dtype) : data type to be used for initialization, default: float64
        '''
        # validate dtype
        _ = jnp.dtype(init_dtype)

        # save ode function and its gradients, as well as map versions that apply over dim 0
        # use mapped version to apply to the entire batch of particles
        # (Xs: k x n x D, thetas: k x p)

        # ode: -> 1 x D
        # mapode: -> k x n x D
        self.mapode = jax.vmap(jax.vmap(ode, in_axes=(0, None, 0), in_axes=(0, 0, None))

        # dfdx: -> n x D x D
        # mapdfdx: -> k x n x D x D
        if dfdx is None:
            mapdfdx = jax.vmap(jax.vmap(jax.jacobian(ode, argnums=0), in_axes=(0, None, 0)), in_axes=(0, 0, None))
            self.mapdfdx = lambda X, theta, t: jnp.permute_dims(mapdfdx(X, theta, t), [0, 1, 3, 2])
        else:
            self.mapdfdx = jax.vmap(jax.vmap(dfdx, in_axes=(0, None, 0), in_axes=(0, 0, None))

        # dfdtheta: -> n x p x D
        # mapdfdtheta: -> k x n x p x D
        if dfdtheta is None:
            mapdfdtheta = jax.vmap(jax.vmap(jax.jacobian(ode, argnums=1), in_axes=(0, None, 0)), in_axes=(0, 0, None))
            self.mapdfdtheta = lambda X, theta, t: jnp.permute_dims(mapdfdtheta(X, theta, t), [0, 1, 3, 2])
        else:
            self.mapdfdtheta = jax.vmap(jax.vmap(dfdtheta, in_axes=(0, None, 0), in_axes=(0, 0, None))

        # I: n x 1
        self.I = jnp.array(data[:,0], dtype=init_dtype).reshape(-1, 1)

        # x_init: n x D
        # contains NaNs where unobserved, will later be filled
        # we do not need to store raw y, since we use boolean mask tau and x_init
        # to be replicated to k x D x n x 1 later
        self.x_init = jnp.array(data[:,1:], dtype=init_dtype)

        # number of discretization points
        self.n = self.I.shape[0]
        # number of dimensions in the ODE
        self.D = self.x_init.shape[1]

        # theta guess for initialization
        self.theta_guess = jnp.array(theta_guess, dtype=init_dtype)
        # confidence level:
            # positive to force theta toward guess
            # negative to force theta away from guess
        self.theta_conf = jnp.array(theta_conf, dtype=init_dtype)
        # number of parameters in theta
        self.p = len(theta_guess)

        self.X_guess = X_guess

        # boolean mask for observed data
        tau = np.isfinite(self.x_init)

        # number of data observations, shape = (D,)
        self.Ns = tau.sum(axis=0)
        self.N = self.Ns.sum().item()

        # dimension indices of observed components
        # consider > 2 observations to be observed, else can't fit matern kernel
        self.observed_components = jnp.where(self.Ns > 2)[0]
        self.unobserved_components = jnp.where(self.Ns <= 2)[0]

        # tau : D x n -> to be replicated to k x D x n x 1 later
        self.tau = tau.T

        self.phis = [None] * self.D
        if sigmas is None:
            self.sigmas = jnp.zeros(self.D)
            self.unknown_sigmas = jnp.arange(self.D, dtype=init_dtype)
        else:
            self.sigmas = jnp.array(sigmas, dtype=float)
            self.unknown_sigmas = jnp.where((1 - (self.sigmas > 0)) * (self.Ns > 2))[0]
            if len(self.unknown_sigmas) == 0:
                self.unknown_sigmas = None
        if not bayesian_sigma:
            self.unknown_sigmas = None

        # interpolate data for observed components
        helpers.initialize_obs(self)

        # fit derivatives on unobserved components, fit theta
        helpers.initialize_unobs(self)

        # fit phi on all components, sigma on observed components
        helpers.fit_phisigma(self, v=2.01)

        # phis: D x 2
        self.phis = jnp.array(self.phis, dtype=init_dtype)
        # sigmas: D x 1 -> to be replicated to k x D x n x 1 later
        self.sigmas = self.sigmas.reshape(-1, 1)

        # C_invs, ms, K_invs : D x n x n -> to be replicated to k x D x n x n
        helpers.build_matrices(self, v=2.01)

        # set GP mean priors
        # mu, mu_dot: n x D -> to be replicated to k x D x n x 1 later
        if mu is not None:
            self.mu = jnp.array(mu, dtype=init_dtype)
            self.mu_dot = jnp.array(mu_dot, dtype=init_dtype)
        else:
            self.mu = jnp.zeros([self.n, self.D])
            self.mu_dot = jnp.zeros([self.n, self.D])

        self.pos_X = pos_X
        self.pos_theta = pos_theta

        # set prior tempering
        if prior_temperature.lower() == 'default':
            self.beta_inv = self.N / (self.D * self.n)
        else:
            self.beta_inv = prior_temperature


    def initialize_particles(self, k_0, init_sd=0.2, dtype='float32', device=None, random_seed=None, mitosis=False):
        '''
        Initialization of particles for mSVGD.

        ARGUMENTS:
        k (int) : number of starting particles

        OPTIONAL:
        init_sd (float) : initial distribution SD for SVGD initialization
        dtype (type) : data type to be used for computation, default: float32
        device (device) : device used to perform compuation
        random_seed (int) : random seed to use for sampling particles

        NOT TO BE USED:
        mitosis: used to set the particles after a mitotis split -- user should always set this to False
        '''
        self.k = k_0
        self.MAP = (k_0 == 1)
        self.logk = np.log(k_0)
        self.dtype = dtype
        self.device = device

        # Is: k x n
        self.Is = jnp.tile(self.I.reshape(1, -1), [self.k, 1]).astype(self.dtype).to_device(self.device)

        # kNs: k x D
        self.kNs = jnp.tile(self.Ns.reshape(1, -1), [self.k, 1]).astype(self.dtype).to_device(self.device)

        # ktau: k x D x n x 1
        self.ktau = helpers.jnp_pad(jnp.tile(self.tau, [self.k, 1, 1])).astype(self.dtype).to_device(self.device)

        # ksigmas: k x D x n x 1
        self.ksigmasq_inv = helpers.jnp_pad(jnp.tile(self.sigmas**-2, [self.k, 1, self.n])).astype(self.dtype).to_device(self.device)

        # kC_invs: k x D x n x n
        self.kC_invs = jnp.tile(self.C_invs, [self.k, 1, 1, 1]).astype(self.dtype).to_device(self.device)

        # kms: k x D x n x n
        self.kms = jnp.tile(self.ms, [self.k, 1, 1, 1]).astype(self.dtype).to_device(self.device)

        # kK_invs: k x D x n x n
        self.kK_invs = jnp.tile(self.K_invs, [self.k, 1, 1, 1]).astype(self.dtype).to_device(self.device)

        # kmu: K x D x n x 1
        self.kmu = helpers.jnp_pad(jnp.tile(self.mu.T, [self.k, 1, 1])).astype(self.dtype).to_device(self.device)

        # kmu_dot: K x D x n x 1
        self.kmu_dot = helpers.jnp_pad(jnp.tile(self.mu_dot.T, [self.k, 1, 1])).astype(self.dtype).to_device(self.device)

        # kx_init, x0: k x D x n x 1
        self.kx_init = helpers.jnp_pad(jnp.tile(self.x_init.T, [self.k, 1, 1])).astype(self.dtype).to_device(self.device)

        # ktheta_init, theta0: k x p
        self.ktheta_init = jnp.tile(self.theta_init.reshape(1, -1), [self.k, 1]).astype(self.dtype).to_device(self.device)

        if mitosis is False:
            np.random.seed(random_seed)
            x0 = jnp.array(np.random.normal(loc=self.kx_init, scale=init_sd), dtype=self.dtype, device=self.device)
            if self.pos_X:
                x0 = jnp.abs(x0)

            theta0 = jnp.array(np.random.normal(loc=self.ktheta_init, scale=init_sd), dtype=self.dtype, device=self.device)
            if self.pos_theta:
                theta0 = self.tensor_abs(theta0)

            if self.unknown_sigmas is not None:
                sigma0 = jnp.array(np.random.normal(loc=jnp.tile(self.sigmas[self.unknown_sigmas].T, [self.k, 1]),
                                                    scale=init_sd), dtype=self.dtype, device=self.device)
                sigma0 = self.tensor_abs(sigma0)
                self.particles0 = jnp.concat([theta0, x0.reshape([self.k, self.D*self.n]), sigma0], axis=1)
            else:
                self.particles0 = jnp.concat([theta0, x0.reshape([self.k, self.D*self.n])], axis=1)
            self.particles = self.particles0.copy()
        else:
            self.particles = mitosis.copy()


    def from_svgd_vector(self, particles):
        '''
        *** HELPER METHOD: USER SAFE. ***

        Extract matrix forms of parameters from the vector used for SVGD.
        '''
        if self.pos_X:
            particles = particles.at[:,self.p:self.p+self.D*self.n].set(jnp.clip(particles[:,self.p:self.p+self.D*self.n], a_min=0))
        if self.pos_theta:
            particles = particles.at[:,:self.p].set(jnp.clip(particles[:,:self.p], a_min=0))
        if self.unknown_sigmas is not None:
            particles = particles.at[:,self.p+self.D*self.n:].set(jnp.clip(particles[:,self.p+self.D*self.n:], a_min=0))

        # Xs: k x n x D
        Xs = jnp.permute_dims(jnp.reshape(
                particles[:,self.p:self.p+self.D*self.n],
                [self.k, self.D, self.n]), [0, 2, 1])
        # thetas: k x p
        thetas = particles[:,:self.p]
        # sigmas: k x n_unknown
        sigmas = particles[:,self.p+self.D*self.n:]

        return Xs, thetas, sigmas


    def gradient(self, particles):
        '''
        *** HELPER METHOD: USER SAFE BUT UNLIKELY TO BE USED. ***

        Compute MAGI posterior gradient.
        '''
        Xs, thetas, sigmas = self.from_svgd_vector(particles)
        if self.unknown_sigmas is not None:
            self.ksigmasq_inv = self.ksigmasq_inv.at[self.unknown_sigmas].set(jnp.permute_dims(
                    jnp.tile(sigmas**-2, [self.n, 1, 1, 1]), [2, 3, 0, 1]))

        # f: k x n x D -> k x D x n x 1
        f = helpers.jnp_pad(jnp.permute_dims(self.mapode(Xs, thetas, self.Is), [0, 2, 1]))
        # df_dx: k x n x D x D -> k x D x D x n
        df_dx = jnp.permute_dims(self.mapdfdx(Xs, thetas, self.Is), [0, 3, 2, 1])
        # df_dtheta: k x n x p x D -> k x D x p x n
        df_dtheta = jnp.permute_dims(self.mapdfdtheta(Xs, thetas, self.Is), [0, 3, 2, 1])

        # Xs: k x D x n x 1
        Xs = helpers.jnp_pad(jnp.permute_dims(Xs, [0, 2, 1]))

        # fmx: k x D x n x 1
        fmx = f - self.kmu_dot - self.kms @ (Xs - self.kmu)
        # kfmx: k x D x n x 1
        kfmx = self.kK_invs @ fmx

        # grad_theta: k x p
        grad_theta = - self.beta_inv * jnp.sum(df_dtheta @ kfmx, axis=1)
        grad_theta = jnp.reshape(grad_theta, [self.k, self.p])

        # dfdxdiag: k x D x D x n x n
        dfdxdiag = jnp.diag(df_dx)
        # stack_kfmx: k x D x D x n x 1
        stack_kfmx = jnp.tile(helpers.jnp_pad(kfmx, 0), [self.D, 1, 1, 1, 1])
        stack_kfmx = jnp.permute_dims(stack_kfmx, [1, 2, 0, 3, 4])

        # grad_x: k x D x n x 1
        # term 1
        grad_x = - self.beta_inv * self.kC_invs @ (Xs - self.kmu)
        # term 2
        grad_x += - self.ksigmasq_inv * (Xs - self.kx_init) * self.ktau
        # term 3
        grad_x += - self.beta_inv * (jnp.sum(dfdxdiag @ stack_kfmx, axis=1) -
                          jnp.permute_dims(self.kms, [0, 1, 3, 2]) @ kfmx)
        grad_x = jnp.reshape(grad_x, [self.k, self.D*self.n])

        # grad_sigma: k x n_unknown
        if self.unknown_sigmas is not None:
            grad_sigma = - self.kNs[:,self.unknown_sigmas] / sigmas
            unobs_disc = ((Xs - self.kx_init) * self.ktau)[:,self.unknown_sigmas][:,:,:,0]
            grad_sigma += sigmas**-3 * jnp.diagonal(unobs_disc @ jnp.permute_dims(unobs_disc, [0, 2, 1]), axis1=1, axis2=2)

            return jnp.concat([grad_theta, grad_x, grad_sigma], axis=1)
        else:
            return jnp.concat([grad_theta, grad_x], axis=1)


    def svgd_kernel(self, particles, h=-1):
        '''
        *** HELPER METHOD: USER SAFE BUT UNLIKELY TO BE USED. ***

        Compute SVGD kernel.
        '''
        L2sq = jnp.array(cdist(particles), dtype=self.dtype, device=self.device)
        if h <= 0:
            h = jnp.median(L2sq) / self.logk

        Kxy = jnp.exp(-L2sq / h)
        dxkxy = - Kxy @ particles
        sumkxy = jnp.reshape(jnp.sum(Kxy, axis=1), [-1, 1])
        dxkxy += particles * jnp.tile(sumkxy, [1, particles.shape[1]])
        dxkxy *= 2/h

        return Kxy, dxkxy


    def mitotic_split(self, opt, grad_particles):
        '''
        *** HELPER METHOD: USER SAFE BUT SHOULD NOT BE CALLED. ***
        
        Perform mitotic split for mSVGD.
        '''
        old_particles = self.particles.copy()
        self.gradient_step(opt, grad_particles, self.particles)
        new_particles = self.concat([old_particles, self.particles], axis=0)
        self.initialize_particles(2*self.k, self.dtype, self.device, init_sd=None, mitosis=new_particles)

        
    def solve(self, optimizer, optimizer_kwargs=dict(), max_iter=10_000, mitosis_splits=0,
              atol=1e-2, rtol=1e-8, bandwidth=-1, monitor_convergence=False):
        '''
        This is a descent problem, so optimizers should be configured to minimize.

        ARGUMENTS:
        optimizer (optimizer) : optimizer object used to solve descent
        optimizer_kwargs (dict) : keyword arguments for the optimizer

        OPTIONAL:
        max_iter (int) : maximum number of descent iterations per mitosis split
        mitosis_splits (int) : number of mitosis splits to make
        atol (float) : stopping criterion -- absolute tolerance for gradient elements
        rtol (float) : stopping criterion -- relative tolerance for gradient elements
        bandwidth (float) : bandwidth for SVGD's RBF kernel, set to -1 for adaptive bandwidth
        monitor_convergence (int) : interval of descent steps at which to record particle state, set to 0 for no monitoring
        '''
        optimizer = helpers.listify(optimizer, mitosis_splits+1)
        optimizer_kwargs = helpers.listify(optimizer_kwargs, mitosis_splits+1)
        max_iter = helpers.listify(max_iter, mitosis_splits+1)
        atol = helpers.listify(atol, mitosis_splits+1)
        rtol = helpers.listify(rtol, mitosis_splits+1)
        bandwidth = helpers.listify(bandwidth, mitosis_splits+1)
        
        if monitor_convergence:
            trajectories = []
        
        for i in range(mitosis_splits+1):
            bandwidth_i = bandwidth[i]
            atol_i = atol[i]
            rtol_i = rtol[i]
            
            if optimizer_kwargs[i].pop('params', None):
                opt = optimizer[i](params=[self.particles], **optimizer_kwargs[i])
                optimizer_kwargs[i]['params'] = True
            else:
                opt = optimizer[i](**optimizer_kwargs[i])

            with trange(max_iter[i]) as pbar:
                for iteration in range(max_iter[i]):
                    grad_particles = -self.gradient(self.particles)
                    if not self.MAP:
                        kxy, dxkxy = self.svgd_kernel(self.particles, h=bandwidth_i)
                        grad_particles = (kxy @ grad_particles - dxkxy) / self.k
        
                    if monitor_convergence and iteration % monitor_convergence == 0:
                        m = self.tensor_max(self.tensor_abs(grad_particles))
                        print(f'Iteration {iteration}, Max Grad = {m:.5f}')
                        trajectories.append(self.clone(self.particles[:,:self.p]))
                            
                    if self.tensor_allsmall(grad_particles, self.particles, atol_i, rtol_i):
                        pbar.update()
                        break
                    else:
                        self.gradient_step(opt, grad_particles, self.particles)
                        pbar.update()
    
                m = self.tensor_max(self.tensor_abs(grad_particles))
                pbar.set_description(f'Split {i} finished with max grad = {m:.5f}')

            if i < mitosis_splits:
                self.mitotic_split(opt, grad_particles)
            
        # Xs: k x n x D
        # thetas: k x p
        # sigmas: k x n_unknown
        Xs, thetas, sigmas = self.from_svgd_vector(self.particles)
        
        if monitor_convergence:
            return Xs, thetas, sigmas, trajectories
        else:
            return Xs, thetas, sigmas
