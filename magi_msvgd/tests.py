'''
Test ODE systems for MAGI.

Each system bundles a vector field with the settings needed to generate a synthetic dataset from
it: true parameters, initial condition, observation noise, per-component observation times, and
the discretization set MAGI will infer on. The usual sequence is

    model = FitzHughNagumo
    model.ground_truth()                       # integrate once
    t, sample = model.sample(seed=0)           # noisy observations
    data = model.discretize(t, sample)         # (len(I), D+1) with NaN where unobserved

or simply `data = model.dataset(seed=0)`, which does all three.

CONVENTIONS

The vector field takes a single state at a single time, `f(x, theta, t) -> (D,)` with `x` of shape
`(D,)` and `t` a SCALAR, which is what MAGI's vmap supplies. This matters for time-dependent
fields: given a length-1 `t` instead, any component built from `t` inherits its trailing axis
while `t`-independent components stay scalar, and `jnp.array([...])` over the mixture raises.

The initial condition is `x0`. `X0` is accepted as a synonym on input.

INTEGRATION

Classical RK4 by default, with the solution retained only at the output times rather than at every
substep. Both changes are substantial:

  * Storing every substep costs `(t_max / step) * D * 8` bytes, which for Hes1's horizon of 240 at
    a 1e-6 step is 5.8 GB. Retaining only the output grid makes it a few kilobytes, independent of
    the step, so accuracy no longer trades against memory.
  * RK4 converges at fourth order against Euler's first, so it reaches a given accuracy at a far
    coarser step. On HIV, Euler needs a step below 1e-6 to bring the integration error under the
    observation noise -- and 1e-6 already costs 480 MB under the old scheme -- while RK4 is well
    converged at 1e-3, roughly a thousand times less work.

`method='euler'` is kept for comparison.
'''

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _rk4(f, x, t, h):
    k1 = f(x, t)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(x + h * k3, t + h)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler(f, x, t, h):
    return x + h * f(x, t)


_STEPPERS = {"rk4": _rk4, "euler": _euler}


class DynamicalSystem:
    def __init__(self, ode, hyperparams):
        self.ode = ode
        self.hyperparams = self._normalize(hyperparams)
        self.T = None
        self.solution = None
        self._key = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _normalize(hp):
        hp = dict(hp)
        if "x0" not in hp and "X0" in hp:
            hp["x0"] = hp.pop("X0")
        hp["x0"] = np.asarray(hp["x0"], np.float64)
        hp["theta"] = np.asarray(hp["theta"], np.float64)
        hp["sigma"] = np.asarray(hp["sigma"], np.float64)
        hp["I"] = np.asarray(hp["I"], np.float64)
        hp["tau"] = [np.asarray(td, np.float64).ravel() for td in hp["tau"]]
        if len(hp["tau"]) != len(hp["x0"]):
            raise ValueError(f'tau lists {len(hp["tau"])} components, x0 has {len(hp["x0"])}')
        if len(hp["sigma"]) != len(hp["x0"]):
            raise ValueError(f'sigma has {len(hp["sigma"])} entries, x0 has {len(hp["x0"])}')
        return hp

    @property
    def obs_times(self):
        """Sorted union of the per-component observation times."""
        return np.unique(np.concatenate(self.hyperparams["tau"] + [np.empty(0)]))

    def _locate(self, times, grid, what="time"):
        """Index of the nearest grid point, erroring if the gap is larger than rounding."""
        times, grid = np.asarray(times, np.float64), np.asarray(grid, np.float64)
        j = np.clip(np.searchsorted(grid, times), 1, len(grid) - 1)
        j = np.where(np.abs(grid[j - 1] - times) <= np.abs(grid[j] - times), j - 1, j)
        gap = np.abs(grid[j] - times)
        tol = 1e-9 * np.maximum(1.0, np.abs(times))
        if np.any(gap > tol):
            bad = times[np.argmax(gap)]
            raise ValueError(f"{what} {bad!r} is not on the grid (nearest is "
                             f"{grid[j[np.argmax(gap)]]!r})")
        return j

    # ------------------------------------------------------------------ integration
    def ground_truth(self, step=1e-3, out_times=None, method="rk4", t_min=None, t_max=None):
        """
        Integrate the system and retain the solution at `out_times`.

        out_times defaults to the union of the discretization set I and every observation time, so
        `sample` and `discretize` can index it exactly rather than searching a float grid. `step`
        is an upper bound on the substep: each output interval is divided into a whole number of
        equal substeps, so output times are hit exactly rather than approached.
        """
        hp = self.hyperparams
        if out_times is None:
            out_times = np.unique(np.concatenate([hp["I"], self.obs_times]))
        out_times = np.unique(np.asarray(out_times, np.float64))
        if t_min is not None:
            out_times = out_times[out_times >= t_min]
        if t_max is not None:
            out_times = out_times[out_times <= t_max]

        x0 = jnp.asarray(hp["x0"])
        theta = jnp.asarray(hp["theta"], x0.dtype)
        f = lambda x, t: jnp.asarray(self.ode(x, theta, t)).reshape(x0.shape)
        advance = _STEPPERS[method]

        t0 = float(out_times[0])
        dts = np.diff(out_times)
        if len(dts) == 0:
            self.T, self.solution = out_times, np.asarray(x0)[None, :]
            return self.T, self.solution
        # one static substep count for the whole scan; the substep size varies per interval so
        # that every output time is reached exactly
        nsub = max(1, int(np.ceil(float(dts.max()) / float(step))))

        def interval(carry, dt):
            x, t = carry
            h = dt / nsub
            def sub(c, _):
                xx, tt = c
                return (advance(f, xx, tt, h), tt + h), None
            (x, t), _ = jax.lax.scan(sub, (x, t), None, length=nsub)
            return (x, t), x

        (_, _), traj = jax.lax.scan(interval, (x0, jnp.asarray(t0, x0.dtype)),
                                    jnp.asarray(dts, x0.dtype))
        self.T = out_times
        self.solution = np.asarray(jnp.concatenate([x0[None, :], traj], axis=0), np.float64)
        self._integration = dict(step=step, method=method, nsub=nsub,
                                 evals=int(nsub * len(dts) * (4 if method == "rk4" else 1)))
        return self.T, self.solution

    def truth_at(self, times):
        """The stored solution at `times`."""
        if self.solution is None:
            raise RuntimeError("run ground_truth() first")
        return self.solution[self._locate(times, self.T)]

    # ------------------------------------------------------------------ data
    def sample(self, seed, tau=None):
        """
        Noisy observations at the union of the observation times.

        Returns `(t, y)` with `y` of shape `(len(t), D)` and NaN wherever component d is not
        observed at that time. Noise is drawn once for the whole array, so a given seed gives the
        same draw regardless of the observation pattern.
        """
        if self.solution is None:
            raise RuntimeError("run ground_truth() first")
        hp = self.hyperparams
        tau = [np.asarray(td, np.float64).ravel() for td in (tau if tau is not None else hp["tau"])]
        t = np.unique(np.concatenate(tau + [np.empty(0)]))
        truth = self.truth_at(t)
        mask = np.stack([np.isin(self._locate(t, self.T),
                                 self._locate(td, self.T)) if len(td) else np.zeros(len(t), bool)
                         for td in tau], axis=1)
        noise = np.asarray(jr.normal(jr.key(seed), shape=truth.shape), np.float64)
        y = truth + noise * hp["sigma"][None, :]
        return t, np.where(mask, y, np.nan)

    def discretize(self, t, y, I=None):
        """Place observations onto the discretization set, NaN elsewhere."""
        I = np.asarray(self.hyperparams["I"] if I is None else I, np.float64)
        y = np.asarray(y, np.float64)
        out = np.full((len(I), y.shape[1] + 1), np.nan)
        out[:, 0] = I
        out[self._locate(t, I, what="observation time"), 1:] = y
        return out

    def dataset(self, seed=0, **kwargs):
        """ground_truth (if needed) -> sample -> discretize, in one call."""
        if self.solution is None:
            self.ground_truth(**kwargs)
        return self.discretize(*self.sample(seed))

    def reset(self):
        self.T = self.solution = None
        return self

    def __repr__(self):
        hp = self.hyperparams
        D, p = len(hp["x0"]), len(hp["theta"])
        nobs = [len(td) for td in hp["tau"]]
        return (f'DynamicalSystem(D={D}, p={p}, |I|={len(hp["I"])}, '
                f'obs per component={nobs}, horizon=[{hp["I"].min():g}, {hp["I"].max():g}])')


####################################################################################################

def fn_ode(X, theta, t=None):
    V, R = X
    a, b, c = theta

    return jnp.array([c * (V - V**3/3 + R), -1/c * (V - a + b*R)])

FitzHughNagumo = DynamicalSystem(fn_ode,
    {
        "theta" : jnp.array([0.2, 0.2, 3.0]),
        "x0" : jnp.array([-1.0, 1.0]),
        "sigma" : jnp.array([0.2, 0.2]),
        "tau" : [jnp.linspace(0, 20, 41),
                   jnp.linspace(0, 20, 41)],
        "I" : jnp.linspace(0, 20, int(160 +1))
    }
)

####################################################################################################

def hes1_ode(X, theta, t=None):
    '''Note: This system is on a log scale'''
    P, M, H = jnp.exp(X)
    a, b, c, d, e, f, g = theta

    return jnp.array([-a*H + b*M/P - c, -d + e/(1+P**2)/M, -a*P + f/(1+P**2)/H - g])

Hes1 = DynamicalSystem(hes1_ode,
    {
        "theta" : jnp.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3]),
        "x0" : np.log([1.438575, 2.037488, 17.90385]),
        "sigma" : jnp.array([0.15, 0.15, jnp.nan]),
        "tau" : [jnp.linspace(0, 240, int(240/15 +1)),
           jnp.linspace(7.5, 232.5, int((232.5-7.5)/15 +1)),
           jnp.array([])],
        "I" : jnp.linspace(0, 240, int(240/7.5 +1))
    }
)

####################################################################################################

def hiv_ode(X, theta, t):
    T_U, T_I, V = X
    lam, rho, delta, N, c = theta
    eta = 9e-5 * (1 - 0.9*jnp.cos(jnp.pi*t/1000))

    return jnp.array([lam - rho*T_U - eta*T_U*V, eta*T_U*V - delta*T_I, N*delta*T_I - c*V])

HIV = DynamicalSystem(hiv_ode,
    {
        "theta" : np.array([36, 0.108, 0.5, 1000, 3]),
        "x0" : np.array([600, 30, 1e5]),
        "sigma" : np.array([10**0.5, 10**0.5, 10]),
        "tau" : [np.linspace(0, 20, 101),
       np.linspace(0, 20, 101),
       np.linspace(0, 20, 101)],
        "I" : np.linspace(0, 20, 201)
    }
)

####################################################################################################

def lorenz_ode(X, theta, t=None):
    x, y, z = X
    beta, rho, sigma = theta

    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

Lorenz = DynamicalSystem(lorenz_ode,
    {
        "theta" : np.array([8/3, 28.0, 10.0]),
        "x0" : np.array([2.0, 2.0, 2.0]),
        "sigma" : np.array([2.96546738, 3.78528167, 4.52163049]),
        "tau" : [np.linspace(0, 2.5, 26),
       np.linspace(0, 2.5, 26),
       np.linspace(0, 2.5, 26)],
        "I" : np.linspace(0, 2.5, 101)
    }
)

SYSTEMS = {"FitzHughNagumo": FitzHughNagumo, "Hes1": Hes1, "HIV": HIV, "Lorenz": Lorenz}
