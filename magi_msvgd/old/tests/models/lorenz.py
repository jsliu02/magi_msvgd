# lorenz model

import numpy as np
import torch
from .. import test_helpers

tau = [np.linspace(0, 2.5, 26),
       np.linspace(0, 2.5, 26),
       np.linspace(0, 2.5, 26)]

hyperparameters = {
    "theta" : np.array([8/3, 28.0, 10.0]),
    "X0" : np.array([2.0, 2.0, 2.0]),
    "sigma" : np.array([2.96546738, 3.78528167, 4.52163049]),
    "tau" : tau,
    "obs_times" : test_helpers.obs_matrix(tau),
    "I" : np.linspace(0, 2.5, 101)
}

def ode(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D
    '''
    n = X.shape[0]
    x, y, z = X.T
    beta, rho, sigma = theta.repeat([n, 1]).T

    return torch.stack([sigma * (y - x),
                        x * (rho - z) - y,
                        x * y - beta * z], axis=1)


def dfdx(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D x D
    '''
    n = X.shape[0]
    x, y, z = X.T
    beta, rho, sigma = theta.repeat([n, 1]).T

    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    one = torch.ones(n, device=theta.device, dtype=theta.dtype)

    return torch.stack([torch.stack([-sigma, sigma, zero], axis=1),
                        torch.stack([rho - z, -one, -x], axis=1),
                        torch.stack([y, x, -beta], axis=1)], axis=2)
    

def dfdtheta(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x p x D
    '''
    n = X.shape[0]
    x, y, z = X.T
    beta, rho, sigma = theta.repeat([n, 1]).T
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    one = torch.ones(n, device=theta.device, dtype=theta.dtype)
    
    return torch.stack([torch.stack([zero, zero, y-x], axis=1),
                        torch.stack([zero, x, zero], axis=1),
                       torch.stack([-z, zero, zero], axis=1)], axis=2)