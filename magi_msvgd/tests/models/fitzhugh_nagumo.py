# FitzHugh-Nagumo model

import numpy as np
import torch
from .. import test_helpers
    
tau = [np.linspace(0, 20, 41),
       np.linspace(0, 20, 41)]

hyperparameters = {
    "theta" : np.array([0.2, 0.2, 3.0]),
    "X0" : np.array([-1.0, 1.0]),
    "sigma" : np.array([0.2, 0.2]),
    "tau" : tau,
    "obs_times" : test_helpers.obs_matrix(tau),
    "I" : np.linspace(0, 20, int(160 +1))
}

def ode(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D
    '''
    n = X.shape[0]
    V, R = X.T
    a, b, c = theta.repeat([n, 1]).T

    return torch.stack([c * (V - V**3/3 + R),
                        -1/c * (V - a + b*R)], axis=1)


def dfdx(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D x D
    '''
    n = X.shape[0]
    V, R = X.T
    a, b, c = theta.repeat([n, 1]).T

    return torch.stack([torch.stack([c * (1 - V**2), c], axis=1),
                        torch.stack([-1/c, -b/c], axis=1)], axis=2)


def dfdtheta(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x p x D
    '''
    n = X.shape[0]
    V, R = X.T
    a, b, c = theta.repeat([n, 1]).T
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    
    return torch.stack([torch.stack([zero, zero, V - V**3/3 + R], axis=1),
                        torch.stack([1/c, -R/c, (V - a + b*R)/c**2], axis=1)], axis=2)