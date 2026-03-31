# log-transformed hes1 model

import numpy as np
import torch
from .. import test_helpers

tau = [np.linspace(0, 20, 101),
       np.linspace(0, 20, 101),
       np.linspace(0, 20, 101)]

hyperparameters = {
    "theta" : np.array([36, 0.108, 0.5, 1000, 3]),
    "X0" : np.array([600, 30, 1e5]),
    "sigma" : np.array([10**0.5, 10**0.5, 10]),
    "tau" : tau,
    "obs_times" : test_helpers.obs_matrix(tau),
    "I" : np.linspace(0, 20, 201)
}

def ode(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D
    '''
    n = X.shape[0]
    T_U, T_I, V = X.T
    lam, rho, delta, N, c = theta.repeat([n, 1]).T
    t = t.flatten()
    eta = 9e-5 * (1 - 0.9*torch.cos(torch.pi*t/1000))

    return torch.stack([lam - rho*T_U - eta*T_U*V,
                        eta*T_U*V - delta*T_I,
                       N*delta*T_I - c*V], axis=1)


def dfdx(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D x D
    '''
    n = X.shape[0]
    T_U, T_I, V = X.T
    lam, rho, delta, N, c = theta.repeat([n, 1]).T
    t = t.flatten()
    eta = 9e-5 * (1 - 0.9*torch.cos(torch.pi*t/1000))
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)

    return torch.stack([torch.stack([-rho - eta*V, zero, -eta*T_U], axis=1),
                        torch.stack([eta*V, -delta, eta*T_U], axis=1),
                        torch.stack([zero, N*delta, -c], axis=1)], axis=2)
    

def dfdtheta(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x p x D
    '''
    n = X.shape[0]
    T_U, T_I, V = X.T
    lam, rho, delta, N, c = theta.repeat([n, 1]).T
    t = t.flatten()
    eta = 9e-5 * (1 - 0.9*torch.cos(torch.pi*t/1000))
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    one = torch.ones(n, device=theta.device, dtype=theta.dtype)
    
    return torch.stack([torch.stack([one, -T_U, zero, zero, zero], axis=1),
                        torch.stack([zero, zero, -T_I, zero, zero], axis=1),
                       torch.stack([zero, zero, N*T_I, delta*T_I, -V], axis=1)], axis=2)