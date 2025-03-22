from .. import _magisolver_base as _base

import numpy as np
import torch
from tqdm.notebook import trange

'''
Polymorphic expressions.
'''

def to_tensor(arr, dtype, device="cuda", requires_grad=False):
    if type(dtype) is str:
        dtype = eval("torch." + dtype)
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    return torch.tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def to_arr(tensor):
    return tensor.detach().cpu().numpy()

def is_tensor(arg):
    return torch.is_tensor(arg)
    
def vmap(func):
    return torch.func.vmap(func, in_dims=0)

def replicate(tensor, dims, dtype, device):
    '''
    Replicate existing matrices, convert to tensors.
    '''
    return torch.tensor(tensor, dtype=dtype, device=device).repeat(*dims)

def normal(mean, sd, random_seed, dtype, device):
    if random_seed is not None:
        torch.random.manual_seed(random_seed)
    return torch.normal(mean=mean, std=sd)

def pad_tensor(tensor, axis):
    return torch.unsqueeze(tensor, axis=axis)

def concat(tensors, axis):
    return torch.concat(tensors, axis=axis)

def reshape(tensor, shape):
    return torch.reshape(tensor, shape)

def tensor_sum(tensor, axis):
    return torch.sum(tensor, axis=axis)

def tensor_mean(tensor, axis):
    return torch.mean(tensor, axis=axis)

def tensor_exp(tensor):
    return torch.exp(tensor)

def tensor_log(tensor):
    return torch.log(tensor)

def batch_diag(tensor):
    return torch.diagonal(tensor, dim1=1, dim2=2)

def embed_diagonal(tensor):
    return torch.diag_embed(tensor)

def permute(tensor, permutation):
    return torch.permute(tensor, permutation)

def square_distances(tensor):
    return torch.cdist(tensor, tensor)**2

def tensor_median(tensor):
    return torch.median(tensor)

def tile(tensor, shape):
    return torch.tile(tensor, shape)

def prepare_particles(tensor):
    return torch.clone(tensor)

def tensor_abs(tensor):
    return torch.abs(tensor)

def tensor_allsmall(tensor, ref, atol, rtol):
    return torch.all(torch.abs(tensor) <= atol + rtol * torch.abs(ref))

def tensor_max(tensor):
    return torch.max(tensor).item()

def clone(tensor):
    return torch.clone(tensor)

def gradient_step(optimizer, gradient, tensor):
    tensor.grad = gradient
    optimizer.step()

def Adam(params, lr=0.001):
    return torch.optim.Adam(params, lr=lr)

def autograd(loss_fn, args, opt):
    opt.zero_grad()
    loss = loss_fn(*args)
    loss.backward()
    opt.step()
    return loss.item()

def stack(tensors, axis):
    return torch.stack(tensors, axis=axis)

def update_tensor(tensor, col_indices, updates):
    tensor[:,col_indices] = updates
    return tensor

def slice_2(tensor, indices):
    return tensor[:,indices]

def clip(tensor):
    return torch.clamp(tensor, min=0)
    
#########################################################################################################

class MAGISolver(_base.baseMAGISolver):
    def __init__(self, ode, dfdx, dfdtheta, data, theta_guess, theta_conf=0,
                 sigmas=None, mu=None, mu_dot=None, pos_X=False, pos_theta=False,
                 prior_temperature=None, bayesian_sigma=True):
        
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
                         theta_conf=theta_conf, sigmas=sigmas, mu=mu, mu_dot=mu_dot, pos_X=pos_X, pos_theta=pos_theta,
                         prior_temperature=prior_temperature, bayesian_sigma=bayesian_sigma)

    def solve(self, optimizer, optimizer_kwargs=dict(), max_iter=10_000, mitosis_splits=0,
              atol=1e-5, rtol=1e-8, bandwidth=-1, monitor_convergence=False):
        optimizer_kwargs['params'] = True
        return super().solve(optimizer, optimizer_kwargs, max_iter, mitosis_splits,
                             atol, rtol, bandwidth, monitor_convergence)
