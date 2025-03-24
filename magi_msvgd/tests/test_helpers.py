import numpy as np
import torch
from tqdm.notebook import trange

def obs_matrix(tau, rounded=3):
    '''
    Convert list of observed times to matrix of observed times, column 0 the joined times,
    remaining columns contain truthy/falsy indicating whether observed.
    Rounds times in I to `rounded` decimal places to mitigate floating point errors.
    '''
    obs_t = np.unique(np.concatenate(tau))
    obs_t = np.round(obs_t, rounded)
    
    obs_times = np.zeros([obs_t.shape[0], len(tau)+1])
    obs_times[:,0] = obs_t
    for d, tau_d in enumerate(tau):
        obs_times[np.where(np.isin(obs_times[:,0], np.round(tau_d, rounded))),d+1] = 1.0
    return obs_times


def discretize_data(sample, I, rounded=3):
    '''
    Discretize a dataset at time set I. Rounds times in I to `rounded` decimal places to mitigate floating point errors.
    '''
    I = np.round(I, rounded)
    data_disc = np.full(shape=(len(I), sample.shape[1]), fill_value=np.nan)
    data_disc[:,0] = I
    data_disc[np.isin(data_disc[:,0], sample[:,0])] = sample
    return data_disc


def check_gradients(ode, dfdx, dfdtheta, n, D, p, trials=100, atol=1e-8, rtol=1e-5):
    '''
    Check manaul gradients against autograd. Returns scores in [0, 1] and the average maximum
    asbolute distance between non-close gradients. May not give full score due to float imprecision.
    '''
    x_score = 0
    t_score = 0
    avg_max_x_diff = 0
    avg_max_t_diff = 0
    for _ in trange(trials):
        X = torch.normal(0, 1, size=[n, D], requires_grad=True)
        t = torch.normal(0, 1, size=[p], requires_grad=True)
        time = torch.normal(0, 1, size=[1])
        X.grad = None
        t.grad = None
        result = ode(X, t, time)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i,j].backward(retain_graph=True)
        auto_x = X.grad
        auto_t = t.grad
    
        x_grad = dfdx(X, t, time).sum(axis=2)
        t_grad = dfdtheta(X, t, time).sum(axis=[0,2])
    
        if torch.allclose(auto_x, x_grad, atol=atol, rtol=rtol):
            x_score += 1
        else:
            avg_max_x_diff += torch.max(torch.abs(auto_x - x_grad))
            
        if torch.allclose(auto_t, t_grad, atol=atol, rtol=rtol):
            t_score += 1
        else:
            avg_max_t_diff += torch.max(torch.abs(auto_t - t_grad))

    print(f"Average max X difference: {avg_max_x_diff / max(trials - x_score, 1)}")
    print(f"Average max theta difference: {avg_max_t_diff / max(trials - t_score, 1)}")
    
    return x_score / trials, t_score / trials
