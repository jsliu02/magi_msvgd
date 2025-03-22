import numpy as np
from . import _helpers as helpers
from tqdm.notebook import trange
'''
Dependencies: numpy, scipy, sklearn, tqdm
'''
class baseMAGISolver():
    def __init__(self, ode, dfdx, dfdtheta, data, theta_guess, theta_conf=0,
                 sigmas=None, mu=None, mu_dot=None, pos_X=False, pos_theta=False,
                 prior_temperature=None, bayesian_sigma=True):
        '''
        Initialization is all mostly in Numpy, and class variables are all stored as Numpy arrays.
        This a polymorphic base class that can be used to build an SVGD module on many libraries.

        Fitting theta and unobserved components is done using acceleration library via autograd.

        Xs : n x D
        thetas : p
        t : n or None

        ARGUMENTS:        
        ode (function, (Xs, thetas, t) -> n x D) : ODE system
        dfdx (function, (Xs, thetas, t) -> n x D x D) : gradient of ODE with respect to X
        dfdtheta (function, (Xs, thetas, t) -> n x p x D) : gradient of ODE with respect to theta
        data (array, (n + 1) x D) : observed data, column 0 is the discretization index I, record NaN for unobserved points
        theta_guess (array, p) : initial guess for theta

        OPTIONAL:
        theta_conf (float or array) : confidence in initial guess for theta, larger theta_conf will pull theta initialization toward guess
        sigmas (array or None) : observation noise standard deviation, if known; individual entries can be set to None
        mu (array, n x D) : prior mean function evaluated at discretization index I
        mu_dot (array, n x D) : derivative of prior mean function with respect to time, evaluated at I
        temper_prior (bool) : whether to use beta = Dn/N for prior tempering
        bayesian_sigma (bool) : whether to give Bayesian treatment to sigma or fix at initial value
        '''
        # save ode function and its gradients, as well as map versions that apply over dim 0
        # use mapped version to apply to the entire batch of particles
        # (Xs: k x n x D, thetas: k x p) -> 

        # ode: -> n x D
        # mapode: -> k x n x D
        self.ode = ode
        self.mapode = self.vmap(self.ode)
        
        # dfdx: -> n x D x D
        # mapdfdx: -> k x n x D x D
        self.dfdx = dfdx
        self.mapdfdx = self.vmap(self.dfdx)
        
        # dfdtheta: -> n x p x D
        # mapdfdtheta: -> k x n x p x D
        self.dfdtheta = dfdtheta
        self.mapdfdtheta = self.vmap(self.dfdtheta)

        # I: n x 1
        self.I = np.array(data[:,0], dtype=float).reshape(-1, 1)
        
        # x_init: n x D
        # contains NaNs where unobserved, will later be filled
        # we do not need to store raw y, since we use boolean mask tau and x_init
        # to be replicated to k x D x n x 1 later
        self.x_init = np.array(data[:,1:], dtype=float)

        # number of discretization points
        self.n = self.I.shape[0]
        # number of dimensions in the ODE
        self.D = self.x_init.shape[1]

        # theta guess for initialization
        self.theta_guess = np.array(theta_guess, dtype=float)
        # confidence level:
            # positive to force theta toward guess
            # negative to force theta away from guess
        self.theta_conf = np.array(theta_conf, dtype=float)
        # number of parameters in theta
        self.p = len(theta_guess)

        # boolean mask for observed data
        tau = np.isfinite(self.x_init)
        
        # number of data observations, shape = (D,)
        self.Ns = tau.sum(axis=0)
        self.N = self.Ns.sum().item()
        
        # dimension indices of observed components
        # consider > 2 observations to be observed, else can't fit matern kernel
        self.observed_components = np.where(self.Ns > 2)[0]
        self.unobserved_components = np.where(self.Ns <= 2)[0]
        
        # tau : D x n -> to be replicated to k x D x n x 1 later
        self.tau = tau.transpose()

        self.phis = [None] * self.D
        if sigmas is None:
            self.sigmas = np.zeros(self.D)
            self.unknown_sigmas = np.arange(self.D, dtype=np.int64)
        else:
            self.sigmas = np.array(sigmas, dtype=float)
            self.unknown_sigmas = np.where((1 - (self.sigmas > 0)) * (self.Ns > 2))[0]
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
        self.phis = np.array(self.phis, dtype=float)
        # sigmas: D x 1 -> to be replicated to k x D x n x 1 later
        self.sigmas = self.sigmas.reshape(-1, 1)

        # C_invs, ms, K_invs : D x n x n -> to be replicated to k x D x n x n
        self.C_invs, self.ms, self.K_invs = [np.array(mats) for mats in \
                zip(*[helpers.build_matrices(self.I, phi[0], phi[1], v=2.01) for phi in self.phis])]

        # set GP mean priors
        # mu, mu_dot: n x D -> to be replicated to k x D x n x 1 later
        if mu is not None:
            self.mu = np.array(mu, dtype=float)
            self.mu_dot = np.array(mu_dot, dtype=float)
        else:
            self.mu = np.zeros([self.n, self.D])
            self.mu_dot = np.zeros([self.n, self.D])

        self.pos_X = pos_X
        self.pos_theta = pos_theta

        # set prior tempering
        if temper_prior is None:
            self.beta_inv = self.N / (self.D * self.n)
        else:
            self.beta_inv = 1 / prior_temperature


    def _configure_polymorphism(self, to_tensor, to_arr, is_tensor, vmap, replicate, pad_tensor, normal, concat, reshape,
            tensor_sum, tensor_mean, tensor_exp, tensor_log, batch_diag, embed_diagonal, permute, square_distances, tensor_median, tile,
            prepare_particles, tensor_abs, tensor_allsmall, tensor_max, clone, gradient_step, Adam, autograd, stack, update_tensor, clip, slice_2):
        '''
        *** HELPER METHOD: NOT USER SAFE. DO NOT CALL. ***
        
        Set up front-end polymorphism.
        
        POLYMORPHIC EXPRESSIONS:
            - to_tensor(arr, dtype, device, requires_grad) : cast array to tensor
            - to_arr(tensor) : cast tensor to array
            - is_tensor(arg) : returns bool describing whether arg is a tensor
            - vmap(func) : map a function along dimension 0 of a tensor
            - replicate(arraylike, dims, dtype) : replicate tensor along dims, CAST TO TYPE, DEVCIE
            - normal(mean, sd, dtype, device) : random normal tensor, SPECIFY TYPE, DEVCIE
            - pad_tensor(tensor, axis) : insert dimension of size one at position
            - concat([tensors], axis) : concatenate tensors along axis
            - reshape(tenosr, shape) : reshape tensor to shape
            - tensor_sum(tensor, axis) : add along axis of tensor
            - tensor_mean(tensor, axis) : compute mean along axis of tensor
            - tensor_exp(tensor) : exponentiate tensor
            - tensor_log(tensor) : log tensor
            - batch_diag(tensor) : extract the diagonal of a batch of tensors, batched along dim 0
            - embed_diagonal(tensor) : embed tensor into a diagonal tensor
            - permute(tensor, permutation) : permute the axes of a tensor
            - square_distances(tensor) : pairwise L2 squared distance vector
            - tensor_median(tensor) : median of a tensor
            - tile(tensor, shape) : tile a tensor according to shape
            - prepare_particles(tensor) : ceate tensor ready for gradient descent
            - tensor_abs(tensor) : absolute value of tensor
            - tensor_allsmall(tensor, ref, atol, rtol) : all entries of tensor are small relative to ref
            - tensor_max(tensor) : maximum value of tensor as a float
            - clone(tensor) : return a new copy of the same tensor
            - gradient_step(tensor) : specify how the optimizer makes the gradient step
            - Adam(params, lr) : return Adam optimizer for params with specified learning rate
            - autograd(loss_fn, args, opt) : calculate loss_fn(*args), backpropogate, take step on optimizer
            - stack([tensors], axis) : stack tensors along axis
            - update_tensor(tensor, col_indices, updates) : plug updates into tensor at specified columns
            - slice_2(tensor, indices) : grabs slice along 2nd dim at list of 
            - clip(tensor) : return tensor clipped at zero
        '''
        self.to_tensor = to_tensor
        self.to_arr = to_arr
        self.is_tensor = is_tensor
        self.vmap = vmap
        self.replicate = replicate
        self.normal = normal
        self.pad_tensor = pad_tensor
        self.concat = concat
        self.reshape = reshape
        self.tensor_sum = tensor_sum
        self.tensor_mean = tensor_mean
        self.tensor_exp = tensor_exp
        self.tensor_log = tensor_log
        self.batch_diag = batch_diag
        self.embed_diagonal = embed_diagonal
        self.permute = permute
        self.square_distances = square_distances
        self.tensor_median = tensor_median
        self.tile = tile
        self.prepare_particles = prepare_particles
        self.tensor_abs = tensor_abs
        self.tensor_allsmall = tensor_allsmall
        self.tensor_max = tensor_max
        self.clone = clone
        self.gradient_step = gradient_step
        self.Adam = Adam
        self.autograd = autograd
        self.stack = stack
        self.update_tensor = update_tensor
        self.slice_2 = slice_2
        self.clip = clip
        self.device = None


    def initialize_particles(self, k_0, dtype, device=None, init_sd=0.2, random_seed=None, mitosis=False):
        '''
        Initialization of particles for mSVGD.

        ARGUMENTS:
        k (int) : number of starting particles
        dtype (type) : data type to be used for computation

        OPTIONAL:
        device (device) : device used to perform compuation
        init_sd (float) : initial distribution SD for SVGD initialization
        log_X (bool) : if X is log-transformed, sample particles from lognormal
        pos_X (bool) : if X is positive, take absolute value of starting particles
        pos_theta (bool) : if theta is positive, take absolute value of starting particles
        random_seed (int) : random seed to use for sampling particles

        NOT TO BE USED:
        mitosis: used to set the particles after a mitotis split -- user should always set this to False
        '''
        self.k = k_0
        self.MAP = (k_0 == 1)
        self.logk = np.log(k_0)
        self.dtype = dtype
        self.device = device

        def np_pad_back(array):
            return np.expand_dims(array, axis=-1)

        # Is: k x n
        self.Is = self.replicate(self.I.reshape(1, -1), [self.k, 1], dtype=self.dtype, device=self.device)

        # kNs: k x D
        self.kNs = self.replicate(self.Ns.reshape(1, -1), [self.k, 1], dtype=self.dtype, device=self.device) 

        # ktau: k x D x n x 1
        self.ktau = self.replicate(np_pad_back(self.tau), [self.k, 1, 1, 1],
                                    dtype=self.dtype, device=self.device)
        # ksigmas: k x D x n x 1
        self.ksigmasq_inv = self.replicate(np_pad_back(self.sigmas**-2),
                                [self.k, 1, self.n, 1], dtype=self.dtype, device=self.device)
        # kC_invs: k x D x n x n
        self.kC_invs = self.replicate(self.C_invs, [self.k, 1, 1, 1],
                                       dtype=self.dtype, device=self.device)
        # kms: k x D x n x n
        self.kms = self.replicate(self.ms, [self.k, 1, 1, 1],
                                   dtype=self.dtype, device=self.device)
        # kK_invs: k x D x n x n
        self.kK_invs = self.replicate(self.K_invs, [self.k, 1, 1, 1],
                                       dtype=self.dtype, device=self.device)

        # kmu: K x D x n x 1
        self.kmu = self.replicate(np_pad_back(self.mu.T), [self.k, 1, 1, 1],
                                   dtype=self.dtype, device=self.device)
        # kmu_dot: K x D x n x 1
        self.kmu_dot = self.replicate(np_pad_back(self.mu_dot.T), [self.k, 1, 1, 1],
                                       dtype=self.dtype, device=self.device)

        # kx_init, x0: k x D x n x 1
        self.kx_init = self.replicate(np_pad_back(self.x_init.T), [self.k, 1, 1, 1],
                                    dtype=self.dtype, device=self.device)

        # ktheta_init, theta0: k x p
        self.ktheta_init = self.replicate(self.theta_init, [self.k, 1],
                                        dtype=self.dtype, device=self.device)
        
        if mitosis is False:
            x0 = self.normal(mean=self.kx_init, sd=init_sd, random_seed=random_seed,
                            dtype=self.dtype, device=self.device)
            if self.pos_X:
                x0 = self.tensor_abs(x0)
    
            theta0 = self.normal(mean=self.ktheta_init, sd=init_sd, random_seed=random_seed,
                                    dtype=self.dtype, device=self.device)
            if self.pos_theta:
                theta0 = self.tensor_abs(theta0)

            if self.unknown_sigmas is not None:
                sigma0 = self.normal(mean=self.replicate(self.sigmas[self.unknown_sigmas].T,
                                [self.k, 1], dtype=self.dtype, device=self.device),
                                sd=init_sd, random_seed=random_seed, dtype=self.dtype, device=self.device)
                sigma0 = self.tensor_abs(sigma0)
                self.particles0 = self.concat([theta0, self.reshape(x0, [self.k, self.D*self.n]), sigma0], axis=1)
            else:
                self.particles0 = self.concat([theta0, self.reshape(x0, [self.k, self.D*self.n])], axis=1)
            self.particles = self.prepare_particles(self.particles0)
        else:
            if not self.is_tensor(mitosis):
                raise TypeError("mitosis: argument should be of type tensor")
            if not (mitosis.shape[0] == self.k) and (mitosis.shape[1] == self.p + self.D*self.n):
                raise ValueError("mitosis: incorrect dimensionality of particles")

            self.particles = self.prepare_particles(mitosis)
            
    
    def from_msvgd_vector(self, particles):
        '''
        *** HELPER METHOD: USER SAFE. ***
        
        Extract matrix forms of parameters from the vector used for SVGD.
        '''
        if self.pos_X:
            self.update_tensor(particles, np.arange(self.p, self.p+self.D*self.n),
                    particles[:,self.p:self.p+self.D*self.n] * self.to_tensor(particles[:,self.p:self.p+self.D*self.n] > 0, dtype=self.dtype))
        if self.pos_theta:
            self.update_tensor(particles, np.arange(self.p),
                    particles[:,:self.p] *self.to_tensor(particles[:,:self.p] > 0, dtype=self.dtype))
            
        # Xs: k x n x D
        Xs = self.permute(self.reshape(
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
        Xs, thetas, sigmas = self.from_msvgd_vector(particles)
        if self.unknown_sigmas is not None:
            self.ksigmasq_inv = self.update_tensor(self.ksigmasq_inv, self.unknown_sigmas, self.permute(
                    self.tile(sigmas**-2, [self.n, 1, 1, 1]), [2, 3, 0, 1]))

        # f: k x n x D -> k x D x n x 1
        f = self.pad_tensor(
            self.permute(self.mapode(Xs, thetas, self.Is), [0, 2, 1]) , -1)
        # df_dx: k x n x D x D -> k x D x D x n
        df_dx = self.permute(self.mapdfdx(Xs, thetas, self.Is), [0, 3, 2, 1])
        # df_dtheta: k x n x p x D -> k x D x p x n
        df_dtheta = self.permute(self.mapdfdtheta(Xs, thetas, self.Is), [0, 3, 2, 1])

        # Xs: k x D x n x 1
        Xs = self.pad_tensor(self.permute(Xs, [0, 2, 1]), -1)

        # fmx: k x D x n x 1
        fmx = f - self.kmu_dot - self.kms @ (Xs - self.kmu)
        # kfmx: k x D x n x 1
        kfmx = self.kK_invs @ fmx

        # grad_theta: k x p
        grad_theta = - self.beta_inv * self.tensor_sum(df_dtheta @ kfmx, axis=1)
        grad_theta = self.reshape(grad_theta, [self.k, self.p])
            
        # dfdxdiag: k x D x D x n x n
        dfdxdiag = self.embed_diagonal(df_dx)
        # stack_kfmx: k x D x D x n x 1
        stack_kfmx = self.tile(self.pad_tensor(kfmx, 0), [self.D, 1, 1, 1, 1])
        stack_kfmx = self.permute(stack_kfmx, [1, 2, 0, 3, 4])

        # grad_x: k x D x n x 1
        # term 1
        grad_x = - self.beta_inv * self.kC_invs @ (Xs - self.kmu)
        # term 2
        grad_x += - self.ksigmasq_inv * (Xs - self.kx_init) * self.ktau
        # term 3
        grad_x += - self.beta_inv * (self.tensor_sum(dfdxdiag @ stack_kfmx, axis=1) - 
                          self.permute(self.kms, [0, 1, 3, 2]) @ kfmx)
        grad_x = self.reshape(grad_x, [self.k, self.D*self.n])

        # grad_sigma: k x n_unknown
        if self.unknown_sigmas is not None:
            grad_sigma = - self.slice_2(self.kNs, indices=self.unknown_sigmas) / sigmas
            unobs_disc = self.slice_2((Xs - self.kx_init) * self.ktau, indices=self.unknown_sigmas)[:,:,:,0]
            grad_sigma += sigmas**-3 * self.batch_diag(unobs_disc @ self.permute(unobs_disc, [0, 2, 1]))

            return self.concat([grad_theta, grad_x, grad_sigma], axis=1)
        else:
            return self.concat([grad_theta, grad_x], axis=1)

    
    def svgd_kernel(self, particles, h=-1):
        '''
        *** HELPER METHOD: USER SAFE BUT UNLIKELY TO BE USED. ***
        
        Compute SVGD kernel.
        '''
        L2sq = self.square_distances(particles)
        if h <= 0:
            h = self.tensor_median(L2sq) / self.logk
            
        Kxy = self.tensor_exp(-L2sq / h)
        dxkxy = - Kxy @ particles
        sumkxy = self.reshape(self.tensor_sum(Kxy, axis=1), [-1, 1])
        dxkxy += particles * self.tile(sumkxy, [1, particles.shape[1]])
        dxkxy *= 2/h

        return Kxy, dxkxy

    
    def mitotic_split(self, opt, grad_particles):
        '''
        *** HELPER METHOD: USER SAFE BUT SHOULD NOT BE CALLED. ***
        
        Perform mitotic split for mSVGD.
        '''
        old_particles = self.clone(self.particles)
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
        if monitor_convergence:
            trajectories = []
        
        for i in range(mitosis_splits+1):          
            if optimizer_kwargs.pop('params', None):
                opt = optimizer(params=[self.particles], **optimizer_kwargs)
                optimizer_kwargs['params'] = True
            else:
                opt = optimizer(**optimizer_kwargs)

            with trange(max_iter) as pbar:
                for iteration in range(max_iter):
                    grad_particles = -self.gradient(self.particles)
                    if not self.MAP:
                        kxy, dxkxy = self.svgd_kernel(self.particles, h=bandwidth)
                        grad_particles = (kxy @ grad_particles - dxkxy) / self.k
        
                    if monitor_convergence and iteration % monitor_convergence == 0:
                        m = self.tensor_max(self.tensor_abs(grad_particles))
                        if monitor_convergence:
                            print(f'Iteration {iteration}, Max Grad = {m:.5f}')
                            trajectories.append(self.clone(self.particles[:,:self.p]))
                            
                    if self.tensor_allsmall(grad_particles, self.particles, atol, rtol):
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
        Xs, thetas, sigmas = self.from_msvgd_vector(self.particles)
        
        if monitor_convergence:
            return Xs, thetas, sigmas, trajectories
        else:
            return Xs, thetas, sigmas
