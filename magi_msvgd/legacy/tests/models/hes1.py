# log-transformed hes1 model

import numpy as np
import torch
from .. import test_helpers

tau = [np.linspace(0, 240, int(240/15 +1)),
       np.linspace(7.5, 232.5, int((232.5-7.5)/15 +1)),
       np.array([])]

hyperparameters = {
    "theta" : np.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3]),
    "X0" : np.log([1.438575, 2.037488, 17.90385]),
    "sigma" : np.array([0.15, 0.15, np.nan]),
    "tau" : tau,
    "obs_times" : test_helpers.obs_matrix(tau),
    "I" : np.linspace(0, 240, int(240/7.5 +1))
}

def ode(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D
    '''
    n = X.shape[0]
    P, M, H = torch.exp(X.T)
    a, b, c, d, e, f, g = theta.repeat([n, 1]).T

    return torch.stack([-a*H + b*M/P - c,
                        -d + e/(1+P**2)/M,
                       -a*P + f/(1+P**2)/H - g], axis=1)


def dfdx(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x D x D
    '''
    n = X.shape[0]
    logP, logM, logH = X.T
    a, b, c, d, e, f, g = theta.repeat([n, 1]).T
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    dP = -(1 + torch.exp(2*logP))**-2 * torch.exp(2*logP) * 2

    return torch.stack([torch.stack([-b*torch.exp(logM - logP), b*torch.exp(logM - logP), -a*torch.exp(logH)], axis=1),
                        torch.stack([e*torch.exp(-logM)*dP, -e*torch.exp(-logM)/(1+torch.exp(2*logP)), zero], axis=1),
                        torch.stack([-a*torch.exp(logP) + f*torch.exp(-logH)*dP, zero, -f*torch.exp(-logH)/(1+torch.exp(2*logP))], axis=1)], axis=2)
    

def dfdtheta(X, theta, t=None):
    '''
    X : n x D
    theta : p

    return : n x p x D
    '''
    n = X.shape[0]
    logP, logM, logH = X.T
    a, b, c, d, e, f, g = theta.repeat([n, 1]).T
    
    zero = torch.zeros(n, device=theta.device, dtype=theta.dtype)
    one = torch.ones(n, device=theta.device, dtype=theta.dtype)
    
    return torch.stack([torch.stack([-torch.exp(logH), torch.exp(logM - logP), -one, zero, zero, zero, zero], axis=1),
                        torch.stack([zero, zero, zero, -one, torch.exp(-logM)/(1+torch.exp(2*logP)), zero, zero], axis=1),
                       torch.stack([-torch.exp(logP), zero, zero, zero, zero, torch.exp(-logH)/(1+torch.exp(2*logP)), -one], axis=1)], axis=2)