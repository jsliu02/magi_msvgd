from .. import _magisolver_base as _base

import numpy as np
import tensorflow as tf
from tqdm.notebook import trange

'''
Polymorphic expressions.
'''

def to_tensor(arr, dtype, device=None, requires_grad=False):
    if not requires_grad:
        return tf.cast(arr, dtype=dtype)
    else:
        return tf.Variable(arr, dtype=dtype)

def to_arr(tensor):
    return tensor.numpy()

def is_tensor(arg):
    return tf.is_tensor(arg)

def vmap(func):
    '''
    Requires testing. Also, can use tf.map_fn for slower but much less memory.
    '''
    def packed_func(args):
        Xs, thetas, t = args
        return func(Xs, thetas, t)
        
    return lambda *tensors: tf.vectorized_map(packed_func, tensors)
    
def replicate(tensor, dims, dtype, device):
    '''
    Replicate existing matrices, convert to tensors.
    '''
    if len(dims) > len(tensor.shape):
        return tf.cast(tf.tile(tf.expand_dims(tensor, axis=0), dims), dtype=dtype)
    else:
        return tf.cast(tf.tile(tensor, dims), dtype=dtype)

def normal(mean, sd, random_seed, dtype, device):
    if random_seed is not None:
        tf.random.set_seed(random_seed)
    return tf.random.normal(shape=mean.shape, mean=mean, stddev=sd, dtype=dtype)

def pad_tensor(tensor, axis):
    return tf.expand_dims(tensor, axis=axis)

def concat(tensors, axis):
    return tf.concat(tensors, axis=axis)

def reshape(tensor, shape):
    return tf.reshape(tensor, shape)

def tensor_sum(tensor, axis):
    return tf.reduce_sum(tensor, axis=axis)

def tensor_mean(tensor, axis):
    return tf.reduce_mean(tensor, axis=axis)

def tensor_exp(tensor):
    return tf.exp(tensor)

def tensor_log(tensor):
    return tf.log(tensor)

def batch_diag(tensor):
    return tf.linalg.diag_part(tensor)

def embed_diagonal(tensor):
    return tf.linalg.diag(tensor)

def permute(tensor, permutation):
    return tf.transpose(tensor, permutation)

def square_distances(tensor):
    temp = tf.expand_dims(tensor, 0)
    return tf.reduce_sum((temp - tf.transpose(temp, [1, 0, 2]))**2, axis=2)

def tensor_median(tensor):
    # median is buggy with tf.function type inference
    # return tf.keras.ops.median(tensor)
    return tf.reduce_mean(tensor)

def tile(tensor, shape):
    for _ in range(len(shape) - len(tensor.shape)):
        tensor = tf.expand_dims(tensor, axis=0)
    return tf.tile(tensor, shape)

def prepare_particles(tensor):
    return tf.Variable(initial_value=tensor)

def tensor_abs(tensor):
    return tf.abs(tensor)

def tensor_allsmall(tensor, ref, atol, rtol):
    return tf.reduce_all(
        tf.less_equal(tf.abs(tensor), atol + rtol * tf.abs(ref))
    ).numpy()

def tensor_max(tensor):
    return tf.reduce_max(tensor).numpy()

def clone(tensor):
    return tf.identity(tensor)

def gradient_step(optimizer, gradient, tensor):
    optimizer.apply_gradients([(gradient, tensor)])

def Adam(params, lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def autograd(loss_fn, args, opt):
    '''
    Requires testing.
    '''
    with tf.GradientTape() as tape:
        loss = loss_fn(*args)
    grads = tape.gradient(loss, args)
    opt.apply_gradients(zip(grads, args))
    return loss
    
def stack(tensors, axis):
    return tf.stack(tensors, axis=axis)

def update_tensor(tensor, col_indices, updates):
    order = np.arange(len(tensor.shape), dtype="int32")
    order[0] = 1; order[1] = 0
    return tf.transpose(
        tf.tensor_scatter_nd_update(
            tf.transpose(tensor, perm=order),
            tf.reshape(col_indices, [-1, 1]),
            tf.transpose(updates, perm=order)
        ), perm=order
    )

def slice_2(tensor, indices):
    return tf.gather(tensor, indices, axis=1)

def clip(tensor):
    return tf.clip_by_value(tensor, 0, np.inf)
    
#########################################################################################################

class MAGISolver(_base.baseMAGISolver):
    def __init__(self, ode, dfdx, dfdtheta, data, theta_guess, theta_conf=0, X_guess=None,
                 sigmas=None, mu=None, mu_dot=None, pos_X=False, pos_theta=False,
                 prior_temperature=None, bayesian_sigma=True,
                 debugging=False):
        
        super()._configure_polymorphism(
            to_tensor=to_tensor,
            to_arr=to_arr,
            is_tensor=is_tensor,
            vmap=vmap,
            replicate=replicate,
            normal=normal,
            pad_tensor=pad_tensor,
            concat=concat,
            reshape=reshape,
            tensor_sum=tensor_sum,
            tensor_mean=tensor_mean,
            tensor_exp=tensor_exp,
            tensor_log=tensor_log,
            batch_diag=batch_diag,
            embed_diagonal=embed_diagonal,
            permute=permute,
            square_distances=square_distances,
            tensor_median=tensor_median,
            tile=tile,
            prepare_particles=prepare_particles,
            tensor_abs=tensor_abs,
            tensor_allsmall=tensor_allsmall,
            tensor_max=tensor_max,
            clone=clone,
            gradient_step=gradient_step,
            Adam=Adam,
            autograd=autograd,
            stack=stack,
            update_tensor=update_tensor,
            slice_2=slice_2,
            clip=clip
        )
        super().__init__(ode=ode, dfdx=dfdx, dfdtheta=dfdtheta, data=data, theta_guess=theta_guess,
                         theta_conf=theta_conf, X_guess=X_guess, sigmas=sigmas, mu=mu, mu_dot=mu_dot, pos_X=pos_X, pos_theta=pos_theta,
                         prior_temperature=prior_temperature, bayesian_sigma=bayesian_sigma)
        if debugging:
            tf.debugging.set_log_device_placement(True)
        
    @tf.function
    def gradient(self, particles):
        return super().gradient(particles)

    @tf.function
    def svgd_kernel(self, particles, h=-1):
        return super().svgd_kernel(particles, h)

    def solve(self, optimizer, optimizer_kwargs=dict(), max_iter=10_000, mitosis_splits=0,
              atol=1e-5, rtol=1e-8, bandwidth=-1, monitor_convergence=False):
        optimizer_kwargs['params'] = False
        results = super().solve(optimizer, optimizer_kwargs, max_iter, mitosis_splits, 
                             atol, rtol, bandwidth, monitor_convergence)
        return results        
