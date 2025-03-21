import numpy as np
import torch
from scipy.integrate import solve_ivp

class ODEmodel():
    def __init__(self, model, hyperparameters=None):
        self.model = model
        self.ode = model.ode
        if hyperparameters is None:
            self.hyperparameters = model.hyperparameters
        else:
            self.hyperparameters = hyperparameters
        self.theta = self.hyperparameters["theta"]
        self.X0 = self.hyperparameters["X0"]
        self.sigma = self.hyperparameters["sigma"]
        self.obs_times = self.hyperparameters["obs_times"]
        self.I = self.hyperparameters["I"]
        
    def np_ode(self, t, x):
        '''
        Convert PyTorch version of ODE to take and return NumPy arrays for SciPy compatibility.
        '''
        x_torch = torch.tensor(x).reshape(1, -1)
        theta_torch = torch.tensor(self.theta)
        de = self.ode(x_torch, theta_torch, torch.tensor([[t]]))
        return de.numpy().flatten()
        
    def get_ode_solution(self, X0=None, T=20, step=1e-4):
        '''
        Solve ODE via numerical integration using scipy.integrate.odeint.
        Some floating point errors for step < 1e-4 makes filtering for data generation unstable.
        '''
        if X0 is None:
            X0 = self.X0
            
        times = np.linspace(0, T, int(np.round(T/step))+1)
        times = np.round(times, int(-np.log10(step)))
        ode_solution = solve_ivp(fun=self.np_ode, t_span=(times[0], times[-1]), y0=X0, t_eval=times).y.T
        self.solution = np.concatenate([times.reshape(-1, 1), ode_solution], axis=1)

    def generate_sample(self, obs_times=None, sigma=None, random_seed=None):
        '''
        Generate random sample. 0th column of obs_times should be observed times, other columns
        are truthy/falsy values depending on whether the component is observed at that time.
        '''
        if obs_times is None:
            obs_times = self.obs_times

        if sigma is None:
            sigma = self.sigma
            
        if type(random_seed) is int:
            np.random.seed(random_seed)
            
        t = obs_times[:,0]
        obs_mask = obs_times[:,1:].copy()
        obs_ind = obs_mask.astype(bool)
        obs_mask[~obs_ind] = np.nan
        
        ground_truth = self.solution[np.where(np.isin(self.solution[:,0], t))]
        sample = ground_truth.copy()
        sample[:,1:] = np.random.normal(ground_truth[:,1:] * obs_mask, sigma)

        return sample