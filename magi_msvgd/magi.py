import os
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _initializer import run_initialization
from gauss_newton import GaussNewtonMAP

'''
MAGI posterior inference: an exact mode, a profiled posterior, and a diagnosis of both.

Dependencies: jax. Helpers also use numpy and scipy; nuts() additionally needs blackjax.

The posterior over (theta, X) is dominated numerically by the states -- 300 to 600 of them
against 3 to 7 parameters -- and the states are the part the data and the GP prior pin down.
So they are not approximated. fit() integrates them out by Laplace at each theta,

    p(theta) ~ exp(-U(theta, X*(theta))) det H_XX(theta, X*(theta))^(-1/2),   X* = argmin_X U,

and does the remaining p-dimensional integral directly, by importance sampling on a scrambled
Sobol set. Nothing Gaussian is assumed about theta. The result is a mixture -- one Gaussian in X
per theta node -- so every moment follows in closed form, including Cov(theta, X).

Against long NUTS references, the largest parameter error is at or below the level at which two
halves of the reference agree with each other on every system that has a usable one: 0.0126
against a floor of 0.0100 on FitzHugh-Nagumo, 0.0093 against 0.0081 on HIV, 0.0119 against 0.0405
on the chaotic Lorenz. The mode alone is 1.03, 0.15 and 1.80 out. See magi_msvgd.profiled.

The pipeline
------------
    map_solve()   the mode, by Gauss-Newton on the exact least-squares form (gauss_newton.py).
                  Convergence is measured by the Newton decrement, which is unit-free.
    diagnose()    is the question well posed? Mode validity, uniqueness, identifiability,
                  properness, whether the GP constrains the states between observations, and how
                  far the curvature moves over the posterior. Seconds, no sampling.
    fit()         the profiled posterior, with an effective-sample-size and Pareto k-hat gate.
                  Falls back to the Laplace approximation when the gate declines.
    sample()      draws from whichever was reported.
    nuts()        the exact fallback, preconditioned by the mode's Hessian.

Precision and device
--------------------
float32 is the default and is the right choice: it matches float64 on FitzHugh-Nagumo and Lorenz
and beats it on HIV, in each case by selecting a larger finite-difference step to stay clear of
its own round-off floor. Hes1 is the exception -- single precision takes its effective sample
size from 21% to 2%, and the gate catches it. The factorisations that define the residual are
always computed in float64 regardless (see gauss_newton), because they define the model rather
than approximate it.

On a GPU the per-node work is ~200x faster than on CPU but each distinct array shape costs a
fresh XLA compile, so the profile dispatches are padded to one shape there and left exact on CPU;
see profiled.ProfiledPosterior._pad_rows. The persistent on-disk compilation cache below makes
the compile a once-per-machine cost rather than a once-per-process one.

An exact route exists for a useful subclass of problems. If the ODE is affine in the state X at
fixed theta -- linear compartment models, linear pharmacokinetics, any constant-coefficient
system, and much weaker than requiring f affine in (X, theta) jointly -- then p(X | theta) is
EXACTLY Gaussian and the profiling above carries no approximation error at all. condition_A()
tests for this in two Hessian evaluations.
'''

# Persistent on-disk compilation cache, CWD-based. The path is recorded rather than recomputed on
# demand, because it is fixed at import time and the working directory may move afterwards --
# clearing a cache the process is not using would look like it had worked.
CACHE_DIR = os.path.join(os.getcwd(), ".jax_cache")
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def jax_cache_info(path=None):
    """Entries and bytes in the persistent compilation cache. Returns (path, n_files, n_bytes)."""
    path = os.path.abspath(path or jax.config.jax_compilation_cache_dir or CACHE_DIR)
    n = b = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                b += os.path.getsize(os.path.join(root, f)); n += 1
            except OSError:
                pass
    return path, n, b


def clear_jax_cache(path=None, force=False, dry_run=False, verbose=True):
    """
    Delete the persistent compilation cache.

    Worth doing after changing anything the cache cannot see a change in -- the residual layout,
    the vector field, a jax or driver upgrade -- since a stale entry is reused silently and the
    symptom is a wrong answer rather than an error.

    Defaults to the directory configured at import, not to one recomputed from the current working
    directory, since those differ as soon as anything calls chdir. Refuses to remove a directory
    that is not named `.jax_cache` unless `force`, so that a mistaken argument deletes nothing:
    this is an unguarded recursive delete otherwise.

    Returns (path, n_files, n_bytes) describing what was removed, or would be with `dry_run`.
    """
    import shutil
    path, n, b = jax_cache_info(path)
    if not os.path.isdir(path):
        if verbose:
            print(f'no cache at {path}')
        return path, 0, 0
    if os.path.basename(path.rstrip(os.sep)) != ".jax_cache" and not force:
        raise ValueError(
            f'refusing to delete {path!r}: expected a directory named ".jax_cache". This is a '
            f'recursive delete, so pass force=True only if that path is certainly a cache.')
    if os.path.dirname(path.rstrip(os.sep)) in ("", os.sep) or path.rstrip(os.sep) in (
            os.sep, os.path.expanduser("~").rstrip(os.sep)):
        raise ValueError(f'refusing to delete {path!r}')
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)     # JAX writes here again without re-configuring
    if verbose:
        print(f'{"would clear" if dry_run else "cleared"} {n} entries, {b / 1e6:.1f} MB '
              f'from {path}')
    return path, n, b


def _solve_upper(U, B):
    """Solve U y = B for upper-triangular U. scipy when present, LU otherwise."""
    try:
        from scipy.linalg import solve_triangular
        return solve_triangular(U, B, lower=False)
    except ImportError:
        return np.linalg.solve(U, B)


class _Laplace:
    """
    Everything read off the exact Hessian at the MAP, computed once and shared.

    x       the mode              Sig     Laplace covariance, pseudo-inverted over the kept span
    H       the exact Hessian     sd      marginal posterior sd of theta
    whiten  L with L L^T = Sig    S       dX*/dtheta by the implicit function theorem
    n_neg, n_null, cond           counts and conditioning of the unit-diagonal scaled Hessian
    """
    __slots__ = ("x", "H", "Sig", "sd", "whiten", "n_neg", "n_null", "cond", "S")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _status(ok, warn=False):
    """OK when the check passes; WARN for a soft failure, FAIL for a hard one."""
    if ok:
        return "OK"
    return "WARN" if warn else "FAIL"


def _row(name, value, threshold, status, meaning):
    return f'  {name:<22} {value:>14} {threshold:>12}   {status:<5} {meaning}'


class MAGIPosterior:
    """
    The fitted posterior, either a profiled mixture or a Laplace fallback.

    reliable      whether the profiled estimate passed its diagnostics. When False the profiled
                  estimate is discarded and `mean`/`cov` are the Laplace approximation, which on a
                  posterior where nothing is identified is the better answer -- there the mode is
                  already at the reference mean and correcting it makes matters worse.
    mean          full particle mean
    theta_cov     (p, p) parameter covariance
    cov           full covariance, Laplace fallback only; the profiled posterior is a mixture and
                  does not carry a single d x d covariance
    diagnostics   ESS, Pareto k-hat and the profile-mode shift, none of which need a reference
    """

    def __init__(self, magi, mean, reliable, diagnostics, timings,
                 theta_cov=None, cov=None, profiled=None):
        self._magi, self.mean, self.reliable = magi, mean, reliable
        self.diagnostics, self.timings = diagnostics, timings
        self.theta_cov, self.cov, self.profiled = theta_cov, cov, profiled

    def sample(self, k=1000, seed=0, unpack=True, state_noise=True):
        """
        Draw k particles.

        Profiled: pick a theta node with probability w_i, then draw X from its conditional
        Gaussian N(X*(theta_i), H_XX(theta_i)^-1). Each distinct node drawn costs one
        factorisation of the state block, so `state_noise=False` returns the conditional means
        only and is much cheaper when the trajectory spread is not needed.
        Laplace: an ordinary Gaussian draw.
        """
        m = self._magi
        rng = np.random.default_rng(seed)
        d = self.mean.shape[0]
        if self.profiled is None or not self.reliable:
            C = np.linalg.cholesky(self.cov + 1e-14 * np.trace(self.cov) / d * np.eye(d))
            X = self.mean[None, :] + rng.standard_normal((k, d)) @ C.T
            return m.unpack_particles(jnp.asarray(X)) if unpack else X
        pp, p = self.profiled, m.p
        idx = rng.choice(len(pp.w), size=k, p=pp.w)
        out = np.empty((k, d))
        uniq = np.unique(idx)
        for j in uniq:
            sel = idx == j
            out[sel, :p] = pp.TH[j]
            out[sel, p:] = pp.Xstar[j].ravel()
        if not state_noise:
            return m.unpack_particles(jnp.asarray(out)) if unpack else out
        # One batched Hessian assembly for every distinct node drawn, then a Cholesky each.
        # Sequential assembly plus an eigh of the nD x nD state block per node was the expensive
        # part of drawing from the mixture: at nD = 603 an eigh is 30x a Cholesky, and H_XX is
        # positive definite at a profile solution -- the inner solve already checked, which is
        # what pp.ok records -- so the eigh is only a fallback for the node where it is not.
        XP = np.stack([np.concatenate([pp.TH[j], pp.Xstar[j].ravel()]) for j in uniq])
        Hs = np.asarray(m._hessian_batch()(jnp.asarray(XP, m.mu.dtype)), np.float64)[:, p:, p:]
        for j, Hj in zip(uniq, Hs):
            sel = idx == j
            Hj = 0.5 * (Hj + Hj.T)
            z = rng.standard_normal((int(sel.sum()), Hj.shape[0]))
            try:
                # H = C C^T, so C^-T z has covariance H^-1.
                C = np.linalg.cholesky(Hj)
                noise = _solve_upper(C.T, z.T).T
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(Hj)
                L = (V / np.sqrt(np.maximum(w, 1e-12 * max(w.max(), 1.0)))) @ V.T
                noise = z @ L.T
            out[sel, p:] += noise
        return m.unpack_particles(jnp.asarray(out)) if unpack else out

    def report(self):
        """A line per diagnostic: what it is, what it must satisfy, and whether it does."""
        g = self.diagnostics
        n, ess = g["n_nodes"], g["ess"]
        ok_ess = ess / n >= g["ess_min"]
        ok_k = np.isfinite(g["khat"]) and g["khat"] < g["khat_max"]
        ok_fail = g["failed"] == 0
        ok_null = g["n_null"] == 0
        kind = ("profiled mixture: states integrated out, parameters integrated over "
                f'{n} nodes' if self.reliable else
                "Laplace approximation at the mode (the profiled estimate was rejected)")
        L = [f'MAGI posterior  --  {kind}', '',
             f'  {"diagnostic":<22} {"value":>14} {"required":>12}   {"":<5} what it measures',
             '  ' + '-' * 104]
        L.append(_row("effective sample size", f'{ess:.0f} ({ess/n:.1%})',
                      f'>= {g["ess_min"]:.0%}', _status(ok_ess),
                      "concentration of the importance weights; below this the"))
        L.append(_row("", "", "", "", "  estimate is dominated by a handful of nodes"))
        L.append(_row("Pareto k-hat", f'{g["khat"]:.2f}', f'< {g["khat_max"]:.2f}',
                      _status(ok_k), "tail heaviness of the importance ratios; above this"))
        L.append(_row("", "", "", "", "  the weighted variance is not reliable"))
        L.append(_row("failed profile solves", f'{g["failed"]} / {n}', "0",
                      _status(ok_fail, warn=g["failed"] < 0.1 * n),
                      "nodes whose inner solve diverged or gave an"))
        L.append(_row("", "", "", "", "  indefinite state Hessian; these are dropped"))
        L.append(_row("null directions", f'{g["n_null"]}', "0", _status(ok_null),
                      "flat directions of the Hessian at the mode. The"))
        L.append(_row("", "", "", "", "  posterior is IMPROPER along them unless"))
        L.append(_row("", "", "", "", "  theta_prec constrains them, and no MCMC"))
        L.append(_row("", "", "", "", "  reference can converge along them either"))
        L.append(_row("profile-mode shift", f'{g["mode_shift"]:.2f} sd', "--", "info",
                      "distance from the joint MAP in Laplace sd. Large"))
        L.append(_row("", "", "", "", "  values mean the mode is a poor summary; this"))
        L.append(_row("", "", "", "", "  is the bias the profiling removes"))
        L.append('  ' + '-' * 104)
        overall = self.reliable and ok_null
        L.append(f'  STATUS: {"OK" if overall else "FAIL"}')
        if not ok_null:
            L.append('    The posterior is improper. Set a proper theta_prec on the flat '
                     'directions; until then no')
            L.append('    estimate of them exists and the identified parameters are biased '
                     'by wherever they settle.')
        if not self.reliable:
            L.append('    The profiled estimate was rejected, so the Laplace approximation is '
                     'reported instead.')
            L.append('    That is the correct answer when the parameters are weakly identified '
                     '-- run diagnose()')
            L.append('    to see whether they are, rather than treating this as a mere fallback.')
        if overall:
            L.append('    Reporting the profiled posterior.')
        t = self.timings
        L.append('')
        L.append(f'  cost {sum(t.values()):.2f} s   ' +
                 " | ".join(f'{k} {v:.2f}' for k, v in t.items()))
        return "\n".join(L)

    def __repr__(self):
        return self.report()


class MAGI:
    def __init__(self, ode, data, theta_mean, theta_prec, sigmas=None,
                 X_guesses=1, unobs_init_iters=500,
                 mu=None, mu_dot=None, prior_temperature='default',
                 init_dtype='float64', init_device=None):
        '''
        Initializing theta and unobserved components is done using acceleration library via autograd.

        Xs : n x D
        thetas : p
        t : n or None

        NOTE: ode should be written for a single observation at a single time point.

        ARGUMENTS:
        ode (function, (Xs, thetas, t) -> n x D) : ODE system
        data (array, n x (D+1)) : observed data, column 0 is the discretization index I, record NaN for unobserved points
        theta_mean (array, p) : prior mean for theta, also the starting guess
        theta_prec (scalar, array (p,), or array (p,p)) : prior precision for theta. A scalar or
            vector is read as a diagonal precision, a matrix as a full one; pi(theta) is
            proportional to exp(-0.5 (theta - theta_mean)^T theta_prec (theta - theta_mean)),
            which is the paper's pi_Theta(theta) term.

            Required, with no default, deliberately. Zero precision is a flat improper prior,
            which leaves the posterior improper along any direction the data does not determine.
            Where that happens it is not only a missing variance for the unidentified parameter:
            the estimates of the parameters that ARE identified are then set by wherever the
            unidentified ones come to rest, and no curvature-based diagnostic reports it. Every
            system in tests.py turns out to be identified once the GP hyperparameters are fitted
            properly, so we have no worked example of this on the suite -- an earlier one, on HIV,
            was measuring the hyperparameter bug and has been withdrawn. The argument stands on
            its own; the term is in the paper's derivation regardless. Passing zero is allowed and
            reproduces the older behaviour bit for bit, but it should be a choice rather than a
            default.

        OPTIONAL:
        sigmas (array or None) : observation noise standard deviation (if known); individual entries can be set to nan
        X_guesses (int) : number of times to run X initialization procedure, can give more stable results
        unobs_init_iters (int) : number of Adam steps when solving for initialization of theta and unobserved components
        mu (array, n x D) : prior mean function evaluated at discretization index I
        mu_dot (array, n x D) : derivative of prior mean function with respect to time, evaluated at I

        temper_prior (float) : prior tempering factor, default: beta = Dn/N
        init_dtype (str or dtype) : data type to be used for initialization, default: float64 (unstable at lower precision)
        init_device : jax.device used for initialization. Use .put() to move later
            (default: GPU if available, else CPU -- resolved lazily so this class can still be
            imported and used on machines with no GPU)
        '''
        if init_device is None:
            init_device = jax.devices()[0]

        # The precomputed GP matrices are built in float64: C^-1 reaches condition 1e9 once the
        # hyperparameters are fitted properly, which is well inside float64's range and outside
        # float32's. That needs jax's global x64 flag, so constructing a MAGI turns it ON and
        # leaves it on. The flag is never turned OFF here -- doing so downgrades unrelated float64
        # work elsewhere in the process, and the symptom surfaces somewhere else entirely.
        if jnp.dtype(init_dtype) == jnp.float64:
            jax.config.update("jax_enable_x64", True)

        # ode: ((D,), (p,), ()) -> (D,), vmapped over the grid to (n, D), (p,), (n, 1) -> (n, D).
        # t is handed to the user's field as a SCALAR, matching the documented contract that the
        # ode is written for a single observation at a single time point. Passing the length-1
        # slice of I instead is silently destructive for time-dependent fields: any component
        # built from t inherits its trailing axis while t-independent components stay scalar, so
        # jnp.array([...]) over a partly time-dependent field is inhomogeneous and raises. The
        # reshape is a no-op for the (n,) case, so callers may pass I either way.
        def _pointwise_ode(x_i, theta, t_i):
            return ode(x_i, theta, jnp.reshape(t_i, ()))
        self.ode = jax.vmap(_pointwise_ode, in_axes=(0, None, 0))

        # I: n x 1
        self.I = jnp.array(data[:,0], dtype=init_dtype, device=init_device).reshape(-1, 1)

        # x_init: n x D
        # contains NaNs where unobserved, will later be filled
        # we do not need to store raw y, since we use boolean mask tau and x_init
        self.x_init = jnp.array(data[:,1:], dtype=init_dtype, device=init_device)

        # number of discretization points
        self.n = self.I.shape[0]
        # number of dimensions in the ODE
        self.D = self.x_init.shape[1]

        # prior mean for theta, also the initialization guess
        self.theta_mean = jnp.array(theta_mean, dtype=init_dtype, device=init_device)
        self.p = len(theta_mean)
        # prior precision, normalised to a (p, p) matrix so scalar, vector and full forms are
        # one code path everywhere downstream
        P = jnp.asarray(theta_prec, dtype=init_dtype)
        if P.ndim == 0:
            P = P * jnp.eye(self.p, dtype=init_dtype)
        elif P.ndim == 1:
            if P.shape[0] != self.p:
                raise ValueError(f'theta_prec has {P.shape[0]} entries, theta_mean has {self.p}')
            P = jnp.diag(P)
        elif P.shape != (self.p, self.p):
            raise ValueError(f'theta_prec has shape {P.shape}, expected (), ({self.p},) or '
                             f'({self.p}, {self.p})')
        P = 0.5 * (P + P.T)
        if bool(jnp.any(jnp.linalg.eigvalsh(P) < -1e-10 * max(float(jnp.max(jnp.abs(P))), 1.0))):
            raise ValueError('theta_prec must be positive semidefinite')
        self.theta_prec = jax.device_put(P, init_device)

        self.X_guesses = X_guesses
        self.unobs_init_iters = unobs_init_iters

        # boolean mask for observed data
        tau = jnp.isfinite(self.x_init)

        # number of data observations, shape = (D,)
        self.Ns = tau.sum(axis=0, dtype=jnp.int32)
        self.N = self.Ns.sum().item()

        # dimension indices of observed components
        # consider > 2 observations to be observed, else can't fit matern kernel
        self.observed_components = jnp.where(self.Ns > 2)[0].astype(jnp.int32)
        self.unobserved_components = jnp.where(self.Ns <= 2)[0].astype(jnp.int32)

        # tau : n x D
        self.tau = tau

        self.phis = jnp.zeros([self.D, 2], dtype=init_dtype, device=init_device)
        if sigmas is None:
            self.sigmas = jnp.full(self.D, -1.0, dtype=init_dtype, device=init_device)
            self.unknown_sigmas = jnp.full(self.D, True, device=init_device)
        else:
            self.sigmas = jnp.array(sigmas, dtype=init_dtype, device=init_device)
            # unknown (needs Bayesian fitting) iff sigma wasn't given AND the component
            # is observed enough to fit it
            self.unknown_sigmas = jnp.logical_and(~(self.sigmas >= 0), self.Ns > 2)

        # run_initialization is fully JIT-compiled
        initializations = run_initialization(self.ode, self.x_init, self.I, self.tau,
                            self.sigmas, self.phis, self.observed_components, self.unobserved_components,
                            jnp.diag(self.theta_prec), self.theta_mean,
                            self.X_guesses, self.unobs_init_iters)
        # force this to actually execute now to avoid dtype casting race conditions
        jax.block_until_ready(initializations)
        self.x_init = initializations[0] # (n, d)
        self.theta_init = initializations[1] # (p,)
        self.sigmas = initializations[2] # (n_unknown,)
        self.phis = initializations[3] # (d,2)
        self.C_invs = initializations[4] # (d, n, n)
        self.ms = initializations[5] # (d, n, n)
        self.K_invs = initializations[6] # (d, n, n)

        # Float64 snapshots of the two GP precision matrices, taken before any put() downcasts
        # them. The Gauss-Newton residual is defined through their Cholesky factors, so those
        # factors have to be computed accurately even when the iteration itself runs in float32 --
        # otherwise the least-squares problem being solved is not the one magi_logdensity scores,
        # and the mode it finds fails the gradient check by standard deviations. C^-1 in particular
        # is badly conditioned once the hyperparameters are fitted properly (1.3e9 on HIV), which
        # is well inside float64's range and well outside float32's. These are numpy, not jax
        # arrays, so put() leaves them alone; they cost (D, n, n) doubles, under a megabyte here.
        self._C_invs64 = np.asarray(self.C_invs, np.float64)
        self._K_invs64 = np.asarray(self.K_invs, np.float64)

        self.particles_init = jnp.concatenate([self.theta_init, self.x_init.flatten(), self.sigmas[self.unknown_sigmas]])
        # Tracks whether the user has explicitly chosen a dtype/device via put(); if not, the
        # pipeline defaults to float32, which is measured as the better choice on three of the
        # four test systems and caught by the gate on the fourth.
        self._put_called = False
        self._invalidate()

        # set GP mean priors
        # mu, mu_dot: n x D
        if mu is not None:
            self.mu = jnp.array(mu, dtype=init_dtype, device=init_device)
            self.mu_dot = jnp.array(mu_dot, dtype=init_dtype, device=init_device)
        else:
            self.mu = jnp.zeros([self.n, self.D], dtype=init_dtype, device=init_device)
            self.mu_dot = jnp.zeros([self.n, self.D], dtype=init_dtype, device=init_device)

        # set prior tempering
        if prior_temperature.lower() == 'default':
            self.beta_inv = self.N / (self.D * self.n)
        else:
            self.beta_inv = prior_temperature

        def magi_logdensity(particle, data_batch):
            '''
            Full MAGI log-density. (n*d + p + n_unknown_sigmas:,) -> scalar

            data_batch : dict bundling mu, mu_dot, C_invs, ms, K_invs, tau, x_init, I, sigmas,
                Ns -- passed as an explicit (non-batched, shared-across-particles) argument
                instead of closed over from `self`. A closed-over jnp array gets
                embedded in the compiled program as a literal HLO constant, which (a) makes
                compile time scale with its size and (b) makes the compiled executable
                un-shareable across different MAGI instances (different datasets/fits) even at
                identical n/D/p/dtype shapes, since the embedded literal
                differs -- every new instance pays a full fresh compile. This matters a lot for
                simulation studies that create many MAGI instances over different simulated
                datasets: measured >60s/replication with data closed over (mostly compile
                time) at a shape that otherwise runs in ~4s once compiled once. Passing data as
                an explicit jit argument makes the compiled executable purely a function of
                shape/dtype, so the (already-enabled) persistent compile cache can skip the
                dominant backend-codegen cost for every replication after the first.
                self.p/self.n/self.D/self.beta_inv/self.ode stay closed over (Python
                scalars/a function, not array data). self.unknown_sigmas also stays closed
                over rather than moving into data_batch: it's used below as a *boolean index*
                (`.at[unknown_sigmas].set(...)`), which JAX requires to be concrete at trace
                time -- a traced/dynamic boolean index raises NonConcreteBooleanIndexError.
                It's also tiny (D-length), so leaving it closed over costs nothing toward the
                goal above.
            '''
            mu, mu_dot, C_invs, ms, K_invs, tau, x_init, I, sigmas0, Ns = (
                data_batch['mu'], data_batch['mu_dot'], data_batch['C_invs'], data_batch['ms'],
                data_batch['K_invs'], data_batch['tau'], data_batch['x_init'], data_batch['I'],
                data_batch['sigmas'], data_batch['Ns'])

            # Wraps the WHOLE body, not just the matmul-looking parts. The einsums below are
            # batched matrix-vector contractions, which are unaffected by this setting in the
            # forward pass -- but their VJPs are matrix-matrix products, which on tensor-core
            # hardware are computed in reduced precision by default. Since the gradient is what
            # every sampler actually consumes, dropping this costs ~4 significant digits there:
            # measured on FitzHugh-Nagumo in float32, the gradient's relative error against a
            # float64 reference goes from 5.1e-7 to 3.9e-3. The forward log-density value alone
            # is bit-identical either way, so the value is not the thing to test.
            with jax.default_matmul_precision("highest"):
                theta = particle[:self.p] # (p,)
                X = particle[self.p:self.p+self.n*self.D].reshape(self.n, self.D) # (n, d)
                sigmas = sigmas0.at[self.unknown_sigmas].set(jnp.clip(particle[self.p+self.n*self.D:], min=1e-5)) # (d,)
                # fully-unobserved (Ns=0) dimensions carry a placeholder sigma (0.0) that's never fit
                safe_sigmas = jnp.where(Ns > 0, sigmas, 1.0) # (d,)

                diff_X    = X - mu # (n, D)
                resid_obs = jnp.where(tau, X - x_init, 0.0) # (n, D)
                ode_resid = (self.ode(X, theta, I)
                             - mu_dot
                             - jnp.einsum('dnm,md->nd', ms, diff_X)) # (n, D)

                Cinv_x    = jnp.einsum('dnm,md->nd', C_invs, diff_X) # (n, D)
                Kinv_r    = jnp.einsum('dnm,md->nd', K_invs, ode_resid) # (n, D)

                gp_term   = jnp.sum(diff_X * Cinv_x) # scalar
                log_norm  = jnp.sum(Ns * jnp.log(2 * jnp.pi * safe_sigmas**2)) # scalar
                obs_term  = jnp.sum(resid_obs**2 / safe_sigmas**2) # scalar
                ode_term  = jnp.sum(ode_resid * Kinv_r) # scalar

            # pi_Theta(theta): the paper's posterior is proportional to pi_Theta(theta) times the
            # three terms above, and omitting it leaves a flat improper prior on theta. Wherever a
            # parameter is genuinely unidentified that prior leaves the posterior improper along
            # its direction, so no posterior mean or variance exists there and no MCMC reference
            # can converge along it.
            # A Gaussian prior is another sum of squares, so it appends one residual block and
            # leaves the Gauss-Newton structure of the mode problem intact while also lifting the
            # theta block of J^T J away from singularity. A zero precision reproduces the flat
            # prior bit for bit, so this is inert unless asked for.
            dth = theta - self.theta_mean
            log_prior = -0.5 * (dth @ (self.theta_prec @ dth))

            return -0.5 * (self.beta_inv * gp_term + log_norm + obs_term
                           + self.beta_inv * ode_term) + log_prior

        self._sync_data()
        self.logdensity = magi_logdensity
        self.gradient = jax.jit(jax.vmap(
            lambda x, data_batch: jax.grad(lambda z: magi_logdensity(z, data_batch))(x),
            in_axes=(0, None)))
        # nuts() and the screen need a plain 1-arg logdensity (blackjax's API has no
        # data-argument slot); this wrapper wires the current self.data through by closure, so
        # it doesn't get the cross-instance compile-sharing benefit above.
        self.magi_logdensity = lambda particle: magi_logdensity(particle, self.data)


    def _invalidate(self):
        '''Drop everything cached against a particular dtype/device. Called by put().'''
        self._gn = None
        self._hess_jit = None
        self._hess_batch = None
        self._logp_batch = None
        self._kernels = {}
        self._lap = None
        self.posterior = None


    def _sync_data(self):
        '''
        Rebuild the `data` bundle (passed as an explicit jit argument to magi_logdensity, see
        its docstring) from the current top-level GP-matrix/observation attributes. `data` holds
        separate references, not aliases, to those attributes, so it must be refreshed whenever
        they change -- currently only put() does that (casting dtype / moving device), so put()
        calls this at the end.
        '''
        self.data = {
            'mu': self.mu, 'mu_dot': self.mu_dot,
            'C_invs': self.C_invs, 'ms': self.ms, 'K_invs': self.K_invs,
            'tau': self.tau, 'x_init': self.x_init, 'I': self.I,
            'sigmas': self.sigmas, 'Ns': self.Ns,
        }

    def put(self, dtype=jnp.float32, device=None):
        """
        Cast every array on the model to `dtype` and move it to `device`.

        device defaults to jax.devices()[0], the GPU where there is one.

        On jax's global x64 flag: float64 arrays cannot exist unless it is on, so asking for
        float64 turns it on. Asking for float32 does NOT turn it off, which is what this used to
        do -- a float32 MAGI then silently downgraded unrelated float64 work elsewhere in the same
        process, and the symptom surfaced in whatever ran next rather than here. Leaving it on
        costs this model nothing: every array is cast explicitly below and every kernel names its
        dtype, so a float32 model stays float32 either way.
        """
        if device is None:
            device = jax.devices()[0]
        if jnp.dtype(dtype) == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        for attr, val in list(self.__dict__.items()):    # setattr below mutates the dict
            if attr == 'data':
                continue                # rebuilt from the top-level attrs by _sync_data()
            if isinstance(val, jax.Array):
                if jnp.issubdtype(val.dtype, jnp.floating):
                    val = jnp.astype(val, dtype)
                setattr(self, attr, jax.device_put(val, device))
        self._sync_data()
        self._put_called = True
        self._invalidate()


    def unpack_particles(self, particles):
        thetas = particles[:,:self.p]
        Xs = particles[:,self.p:self.p+self.n*self.D].reshape(particles.shape[0], self.n, self.D)
        sigmas = particles[:,self.p+self.n*self.D:]

        return Xs, thetas, sigmas


    # ------------------------------------------------------------------ pipeline internals

    def _gn_solver(self):
        if self._gn is None:
            self._gn = GaussNewtonMAP(self)
        return self._gn

    def _hessian_fn(self):
        '''
        Exact Hessian of U = -log p, jitted and cached.

        Assembled structurally rather than by jax.hessian. Only the ODE block of the residual is
        nonlinear, and only through f, so hess R_a = sqrt(b) (Lk^T)_a hess f pointwise: the
        second-derivative term is n small (D+p)x(D+p) blocks from the user's ODE, scattered into
        place, on top of the J^T J that the Gauss-Newton solver already forms. That is one
        Jacobian assembly instead of the dim forward passes jax.hessian would take, and it
        agrees with jax.hessian to 2.5e-9.

        Unknown sigmas enter the log-density through a clip and a log rather than the residual,
        so the structural form does not cover them and a generic Hessian is used instead. A
        Gaussian approximation in sigma is in any case weaker than one in (theta, X), since
        sigma is positivity-constrained.
        '''
        if self._hess_jit is not None:
            return self._hess_jit
        gn = self._gn_solver()
        n, D, p, nD = self.n, self.D, self.p, gn.nD
        if int(jnp.sum(self.unknown_sigmas)) > 0:
            self._hess_jit = jax.jit(jax.hessian(lambda z: -self.logdensity(z, self.data)))
            return self._hess_jit

        def f_local(z, t):                                           # R^(D+p) -> R^D
            return self.ode(z[:D][None, :], z[D:], t[None])[0]
        hl = jax.vmap(jax.jacfwd(jax.jacfwd(f_local)), in_axes=(0, 0))
        # grid point i contributes a (D+p)x(D+p) block over particle indices
        # [p + i*D .. p + i*D + D) for its own state, plus [0 .. p) for the shared theta
        idx = jnp.asarray(np.concatenate(
            [p + np.arange(n)[:, None] * D + np.arange(D)[None, :],
             np.broadcast_to(np.arange(p)[None, :], (n, p))], axis=1))

        def hess(x):
            th, X = x[:p], x[p:p + nD].reshape(n, D)
            # J^T J comes from the solver's structured assembly, which never materialises J. The
            # dense route -- form the (3nD+p) x dim Jacobian, including an nD x nD diagonal block
            # and an nD x nD GP block, then multiply it out -- computes the same matrix for about
            # 2.5x the work, and the exact Hessian is evaluated once per profile node.
            # _normal_equations keeps its own default_matmul_precision("highest") guard, which
            # matters here: J^T J is a genuine matrix-matrix product, so on tensor-core hardware
            # it would otherwise run in TF32, and on FitzHugh-Nagumo in float32 that widens the
            # spread of the error in log det H_XX across nodes from 1.1e-4 to 5.5e-2 nats, landing
            # straight on the importance weights.
            A, _g, _f, r_ode = gn._normal_equations_r(th, X, self.sigmas)
            c = gn.b * jnp.einsum('nd,dmn->md', r_ode.reshape(n, D), gn.Lk)
            Z = jnp.concatenate([X, jnp.broadcast_to(th, (n, p))], axis=1)
            S = jnp.einsum('md,mdij->mij', c, hl(Z, gn.I))
            # The theta prior is linear, so it contributes to J^T J and nothing second order.
            return A.at[idx[:, :, None], idx[:, None, :]].add(S)
        self._hess_jit = jax.jit(hess)
        return self._hess_jit

    def _hessian_batch(self):
        """vmapped exact Hessian. One dispatch for a set of points instead of one each."""
        if self._hess_batch is None:
            self._hess_batch = jax.jit(jax.vmap(self._hessian_fn()))
        return self._hess_batch

    def _laplace(self):
        """
        The exact Hessian at the MAP and everything read off it, computed once and cached.

        fit(), diagnose() and the profiled posterior each want some of the Laplace covariance,
        the marginal theta scale and dX*/dtheta, and each used to assemble the Hessian and
        eigendecompose it for itself -- two assemblies and three dim x dim eigendecompositions
        per fit. Invalidated by map_solve(), since it is tied to a particular mode.

        float64 numpy throughout, deliberately. put(float32) turns jax's x64 off, and an eigh in
        single precision of a matrix conditioned at 1e9 says nothing about its small eigenvalues
        -- which are exactly what the null-direction and identifiability checks read.
        """
        if self._lap is not None:
            return self._lap
        if getattr(self, "map_particle", None) is None:
            self.map_solve(verbose=False)
        p = self.p
        x = np.asarray(self.map_particle, np.float64)
        H = np.asarray(self.hessian(x), np.float64); H = 0.5 * (H + H.T)
        # Scale before decomposing: raw Hessian eigenvalues carry units, so "small" would not be
        # a statement about the posterior. D H D has a unit diagonal and a comparable spectrum.
        d = np.sqrt(np.maximum(np.abs(np.diag(H)), np.finfo(float).tiny))
        w, V = np.linalg.eigh(H / np.outer(d, d))
        sc = max(abs(w).max(), 1e-300)
        keep = w > 1e-10 * sc
        Vk = V[:, keep] / d[:, None]
        Sig = (Vk / w[keep]) @ Vk.T
        # dX*/dtheta by the implicit function theorem: H_XX dX*/dtheta + H_Xtheta = 0. A solve,
        # not an eigendecomposition -- H_XX is positive definite at a mode, and eigh of it costs
        # an order more. The eigh path is kept for the case where it is not.
        Hxx, Hxt = H[p:, p:], H[p:, :p]
        try:
            S = -np.linalg.solve(Hxx, Hxt)
            if not np.all(np.isfinite(S)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            wx, Vx = np.linalg.eigh(0.5 * (Hxx + Hxx.T))
            S = -((Vx / np.maximum(wx, 1e-12 * max(wx.max(), 1e-300))) @ (Vx.T @ Hxt))
        self._lap = _Laplace(
            x=x, H=H, Sig=Sig, S=S,
            sd=np.sqrt(np.maximum(np.diag(Sig)[:p], 0)),
            whiten=((V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T) / d[:, None],
            n_neg=int((w < -1e-10 * sc).sum()), n_null=int((~keep).sum()),
            cond=float(sc / max(abs(w).min(), 1e-300)))
        return self._lap

    def hessian(self, particle=None):
        '''Exact Hessian of -log p at `particle` (default: the MAP). Returns a jax array.'''
        if particle is None:
            if getattr(self, 'map_particle', None) is None:
                self.map_solve(verbose=False)
            particle = self.map_particle
        return self._hessian_fn()(jnp.asarray(particle, self.mu.dtype))

    def _logp_many(self, P):
        if self._logp_batch is None:
            self._logp_batch = jax.jit(jax.vmap(lambda z: self.logdensity(z, self.data)))
        return np.asarray(self._logp_batch(jnp.asarray(P, self.mu.dtype)), np.float64)

    # ------------------------------------------------------------------ public pipeline

    def map_solve(self, x0=None, **kwargs):
        '''
        Find the MAP by Gauss-Newton on the least-squares form of the log-density.

        Thin wrapper over gauss_newton.GaussNewtonMAP, which is where the method and its
        rationale are documented. The solver instance is cached on self, so repeated calls reuse
        one compilation; construct GaussNewtonMAP directly for finer control.

        Returns the unpacked (Xs, thetas, sigmas) at the mode and leaves the particle on
        self.map_particle.
        '''
        gn = self._gn_solver()
        out = gn.solve(x0=x0, **kwargs)
        self.map_particle = gn.map_particle
        self._lap = None                 # the cached Hessian belongs to the previous mode
        return out

    def diagnose(self, n_starts=4, spread=0.25, reach=1e4, drop=3.0, mode_tol=0.05, n_curv=8,
                 seed=0, verbose=True, **map_kwargs):
        """
        Check that the question is well posed, before answering it.

        Everything the pipeline reports rests on assumptions that are cheap to test once the mode
        is cheap: that the point returned is a maximum, that it is the only one, that the data
        determines the parameters, and that the posterior is proper. On the benchmark suite two of
        four systems fail one of these, and in both cases it changes what should be reported.

        The parameter table's last column is the one that cannot be obtained from curvature. A
        Hessian eigenvalue says how sharply the density falls off LOCALLY; it cannot distinguish a
        parameter that is merely uncertain from one that is unbounded. So each parameter axis is
        walked outward with the states re-profiled at every step -- following the ridge rather than
        cutting across it -- and the fall in the profiled log-density is recorded. A direction that
        has not fallen by `drop` nats by `reach` times its own scale is improper: no posterior mean
        or variance exists for it, and no MCMC reference will converge along it either.

        n_starts  dispersed restarts for the globality check, in addition to the mode already
                  found; 0 skips it, which is the expensive part at roughly one mode solve each
        mode_tol  two solves count as the same optimum when they are closer than this, measured in
                  posterior standard deviations per dimension
        n_curv    Hessian evaluations away from the mode used to measure how much the curvature
                  varies over the posterior; 0 skips it

        Returns a dict; prints the report when verbose.
        """
        import time
        from profiled import ProfiledPosterior
        if not self._put_called:
            self.put(dtype=jnp.float32)
        t0 = time.time()
        if getattr(self, "map_particle", None) is None or map_kwargs:
            self.map_solve(verbose=False, **map_kwargs)
        x = np.asarray(self.map_particle, np.float64)
        p = self.p
        lp0 = float(self.logdensity(self.map_particle, self.data))
        g = np.asarray(self.gradient(jnp.asarray(x, self.mu.dtype)[None, :], self.data),
                       np.float64)[0]
        gnorm = float(np.linalg.norm(g))

        lap = self._laplace()
        H, Sig, sd = lap.H, lap.Sig, lap.sd
        n_neg, n_null, cond = lap.n_neg, lap.n_null, lap.cond
        # Distance from here to the true mode, in posterior standard deviations. The raw gradient
        # norm cannot be thresholded: it carries the units of the log-density and of theta, so
        # what counts as small depends on the problem and on the working precision. This does not
        # -- to second order it is the Mahalanobis distance to the stationary point, so 0.01 means
        # the mode is located to within a hundredth of a posterior standard deviation whatever the
        # scales involved. In single precision the raw norm on FitzHugh-Nagumo sits at 2.9e-2,
        # which looks alarming against any absolute tolerance and corresponds to 6e-3 sd.
        mode_dist = float(np.sqrt(max(g @ Sig @ g, 0.0)))

        # How far the curvature departs from the mode's. The Laplace metric whitens the target
        # exactly AT the mode, so cond(M) = 1 there by construction and what matters is a standard
        # deviation away. This predicts how hard the posterior is to sample before any chain runs,
        # and the condition number at the mode does not: on Hes1 that is 1.5e5 while cond(M)
        # reaches 1.8e6, and Hes1 is the one system whose reference fails (R-hat 1.76).
        cond_M = np.nan
        if n_curv:
            Lw = lap.whiten
            rr = np.random.default_rng(seed)
            XI = x[None, :] + rr.standard_normal((n_curv, len(x))) @ Lw.T
            Hs = np.asarray(self._hessian_batch()(jnp.asarray(XI, self.mu.dtype)), np.float64)
            cs = []
            for Hi in Hs:
                Hi = 0.5 * (Hi + Hi.T)
                ei = np.linalg.eigvalsh(Lw.T @ Hi @ Lw)
                cs.append(float(np.abs(ei).max() / max(np.abs(ei).min(), 1e-300)))
            cond_M = float(np.median(cs))

        # Globality. Two solves found the same optimum when they land at the same POINT, in the
        # Hessian metric and per dimension. Comparing log-densities instead fails: in float32 the
        # mode is located to ~5e-3 posterior sd, so two solves of the same optimum differ by far
        # more than any tight absolute tolerance, and rounding counted every restart as new.
        best_alt, n_distinct, n_solved = lp0, 1, 1
        if n_starts:
            rng = np.random.default_rng(seed)
            opts = [(lp0, x.copy())]
            for _ in range(n_starts):
                xt = x.copy()
                xt[:p] = x[:p] * np.exp(spread * rng.standard_normal(p))
                xt[p:] = x[p:] + spread * np.std(x[p:]) * rng.standard_normal(len(x) - p)
                try:
                    self.map_solve(x0=jnp.asarray(xt, self.mu.dtype), verbose=False,
                                   check=False, max_iter=300)
                    xn = np.asarray(self.map_particle, np.float64)
                    v = float(self.logdensity(self.map_particle, self.data))
                    if not np.isfinite(v):
                        continue
                    n_solved += 1
                    dist = lambda a, b: float(np.sqrt(max((a - b) @ H @ (a - b), 0.0)
                                                      / len(a)))
                    if all(dist(xn, o[1]) > mode_tol for o in opts):
                        opts.append((v, xn))
                except Exception:
                    pass
            # Restore the original mode. The restarts each invalidated the cached Laplace, but
            # this solve starts at x and returns to it, so the cached decomposition is still the
            # decomposition at this point and is reinstated rather than recomputed -- a dim x dim
            # Hessian assembly and eigendecomposition saved.
            self.map_solve(x0=jnp.asarray(x, self.mu.dtype), verbose=False, check=False)
            self._lap = lap
            best_alt = max(o[0] for o in opts)
            n_distinct = len(opts)

        pp = ProfiledPosterior(self, n_nodes=8, seed=seed)
        # Several radii, not one. A probe whose inner solve fails also returns -inf, which would
        # be indistinguishable from a density that has genuinely decayed, so the reported fall is
        # taken at the furthest radius that actually resolved, and the radius is reported with it.
        radii = np.array([r for r in (1.0, reach ** 0.5, reach) if r > 0])
        nr = len(radii)
        falls, at_r = np.full(p, np.nan), np.zeros(p)
        # Every parameter at every radius in one dispatch. Walking them one at a time cost p
        # separate launches of a kernel whose per-call overhead dwarfs six extra rows.
        TH = np.repeat(x[None, :p], 1 + 2 * nr * p, axis=0)     # row 0 is the reference value
        for j in range(p):
            scale = max(abs(x[j]), 1.0)
            for i, r in enumerate(radii):
                TH[1 + (j * nr + i) * 2, j] += r * scale
                TH[2 + (j * nr + i) * 2, j] -= r * scale
        lp, _, ok = pp.logp(TH)
        base = lp[0]
        for j in range(p):
            for i, r in enumerate(radii):                       # furthest resolved radius
                sl = slice(1 + (j * nr + i) * 2, 3 + (j * nr + i) * 2)
                if np.all(ok[sl]) and np.all(np.isfinite(lp[sl])):
                    falls[j] = float(np.min(base - lp[sl]))
                    at_r[j] = r
            if not np.isfinite(falls[j]):                       # nothing resolved even at r = 1
                falls[j] = np.inf
        secs = time.time() - t0

        # GP smoothness. The states between observations are held by the GP prior alone, so a
        # lengthscale short compared with the grid spacing leaves them unconstrained -- the
        # trajectory then interpolates the data at observed points and is free everywhere else.
        # This is a property of the hyperparameter fit, not of the posterior approximation, and
        # nothing else in the pipeline notices it.
        Ig = np.asarray(self.I).ravel()
        dt = float(np.median(np.diff(Ig))) if len(Ig) > 1 else np.nan
        ell = np.asarray(self.phis, np.float64)[:, 1]
        tau_np = np.asarray(self.tau)
        gaps = np.array([float(np.median(np.diff(Ig[tau_np[:, j]]))) if tau_np[:, j].sum() > 1
                         else np.nan for j in range(self.D)])
        rel = sd / np.maximum(np.abs(x[:p]), 1e-300)
        verdict = ["improper" if f <= drop else
                   "identified" if r < 0.5 else "weak" if r < 5 else "diffuse"
                   for f, r in zip(falls, rel)]
        out = dict(log_p=lp0, grad_norm=gnorm, mode_dist=mode_dist, n_neg=n_neg,
                   n_null=n_null, cond=cond,
                   theta=x[:p], theta_sd=sd, rel_sd=rel, fall=falls, verdict=verdict,
                   at_radius=at_r, n_distinct=n_distinct, best_alt=best_alt,
                   n_solved=n_solved, ell=ell, grid_dt=dt, obs_gap=gaps, cond_M=cond_M,
                   secs=secs)
        if verbose:
            print(self._diagnosis_report(out, n_starts, drop, reach))
        return out

    def _diagnosis_report(self, o, n_starts, drop, reach):
        p = self.p
        ok_grad = o["mode_dist"] < 0.01
        ok_neg, ok_null = o["n_neg"] == 0, o["n_null"] == 0
        ok_glob = o["n_distinct"] == 1 or o["best_alt"] <= o["log_p"] + 1e-4
        L = [f'MAGI diagnosis  --  d = {self.p + self.n * self.D}, p = {p}, '
             f'{o["secs"]:.1f} s, no sampling', '',
             f'  {"check":<22} {"value":>14} {"required":>12}   {"":<5} what it measures',
             '  ' + '-' * 104]
        L.append(_row("distance to the mode", f'{o["mode_dist"]:.1e}', "< 1e-02",
                      _status(ok_grad, warn=o["mode_dist"] < 0.1),
                      "sqrt(g' H^-1 g): how far this point is from the"))
        L.append(_row("", "", "", "", "  true mode, in posterior sd. Unitless, so it"))
        L.append(_row("", "", "", "", "  means the same across problems and dtypes"))
        L.append(_row("gradient norm", f'{o["grad_norm"]:.1e}', "--", "info",
                      "the raw ||grad log p||, for reference. It carries"))
        L.append(_row("", "", "", "", "  units and its floor is set by the working"))
        L.append(_row("", "", "", "", "  precision, so it is not thresholded here"))
        L.append(_row("negative curvature", f'{o["n_neg"]}', "0", _status(ok_neg),
                      "directions in which the mode is a saddle, not a"))
        L.append(_row("", "", "", "", "  maximum; a Gaussian there does not exist"))
        L.append(_row("null directions", f'{o["n_null"]}', "0", _status(ok_null),
                      "flat directions of the SCALED Hessian, so this is"))
        L.append(_row("", "", "", "", "  unit-free; see the parameter table for which"))
        L.append(_row("condition number", f'{o["cond"]:.1e}', "< 1e+12",
                      _status(o["cond"] < 1e12), "of the scaled Hessian; beyond this the "
                      "Laplace"))
        L.append(_row("", "", "", "", "  covariance is not numerically meaningful"))
        if np.isfinite(o["cond_M"]):
            cm = o["cond_M"]
            L.append(_row("curvature variation", f'{cm:.1e}', "< 1e+03",
                          _status(cm < 1e3, warn=cm < 1e5),
                          "cond of the Hessian in the mode's own metric, a"))
            L.append(_row("", "", "", "", "  standard deviation away. 1 means the posterior is"))
            L.append(_row("", "", "", "", "  Gaussian in that metric; large means the curvature"))
            L.append(_row("", "", "", "", "  varies and a fixed mass matrix cannot track it, so"))
            L.append(_row("", "", "", "", "  HMC will struggle however long it is run"))
        if n_starts:
            L.append(_row("distinct optima", f'{o["n_distinct"]}', "1", _status(ok_glob),
                          f'over {o["n_solved"]} solves that converged, from the mode'))
            L.append(_row("", "", "", "", f'  plus {n_starts} dispersed restarts. More than one'))
            L.append(_row("", "", "", "", "  means the mode found depends on the start"))
            if o["best_alt"] > o["log_p"] + 1e-6:
                L.append(_row("better optimum found", f'{o["best_alt"] - o["log_p"]:+.3g}', "<= 0",
                              "FAIL", "nats above the mode being reported; a restart"))
                L.append(_row("", "", "", "", "  found a higher point, so this is not the mode"))
        L.append('  ' + '-' * 104)
        L.append('  The mode check is "distance to the mode", NOT the gradient norm. '
                 '||grad log p|| carries the')
        L.append('  units of the log-density and of theta, and its floor is set by the working '
                 'precision -- in')
        L.append('  float32 it bottoms out near 1e-2 on these problems and no absolute tolerance '
                 'can be met.')
        L.append('  sqrt(g\' H^-1 g) rescales it to posterior standard deviations, where < 0.01 '
                 'means the mode is')
        L.append('  located to within a hundredth of one and nothing downstream is affected.')
        L.append('')
        L.append(f'  {"parameter":<10} {"MAP":>13} {"post sd":>12} {"sd/|MAP|":>10} '
                 f'{"log p fall":>12} {"at":>8}   {"":<5} verdict')
        L.append('  ' + '-' * 104)
        for j in range(p):
            v = o["verdict"][j]
            st = "FAIL" if v == "improper" else ("WARN" if v in ("diffuse", "weak") else "OK")
            rr = o["at_radius"][j]
            L.append(f'  theta[{j}]{"":<3} {o["theta"][j]:>13.5g} {o["theta_sd"][j]:>12.4g} '
                     f'{o["rel_sd"][j]:>10.3g} {o["fall"][j]:>12.4g} '
                     f'{(f"{rr:.0e}x" if rr else "--"):>8}   {st:<5} {v}')
        L.append('  ' + '-' * 104)
        L.append('')
        L.append(f'  {"state":<10} {"GP lengthscale":>15} {"grid dt":>10} {"ell/dt":>9} '
                 f'{"obs gap/dt":>11}   {"":<5} what it means')
        L.append('  ' + '-' * 104)
        for j in range(self.D):
            r = o["ell"][j] / o["grid_dt"] if o["grid_dt"] else np.nan
            st = _status(r >= 3.0, warn=r >= 1.0)
            note = ("smooth enough to constrain between observations" if r >= 3 else
                    "marginal" if r >= 1 else
                    "states between observations are UNCONSTRAINED")
            gp = o["obs_gap"][j] / o["grid_dt"] if o["grid_dt"] else np.nan
            L.append(f'  X[{j}]{"":<6} {o["ell"][j]:>15.4g} {o["grid_dt"]:>10.4g} {r:>9.2f} '
                     f'{gp:>11.1f}   {st:<5} {note}')
        L.append('  ' + '-' * 104)
        L.append('  The GP prior is all that holds the trajectory between observations, so a '
                 'lengthscale short')
        L.append('  against the grid spacing lets it interpolate the data and do anything in '
                 'between. This is')
        L.append('  set by the hyperparameter fit, before any inference, and is not something '
                 'the posterior')
        L.append('  approximation can repair.')
        L.append('')
        L.append(f'  "log p fall" is the drop in the profiled log-density on walking that '
                 f'parameter out to "at" times its')
        L.append(f'  own scale, with the states re-profiled at each step so the walk follows the '
                 f'ridge. Below')
        L.append(f'  {drop:g} nats the posterior is improper there. "at" is the furthest radius '
                 f'that resolved.')
        L.append('')
        n_bad = sum(1 for v in o["verdict"] if v == "improper")
        n_rough = int(np.sum(o["ell"] / o["grid_dt"] < 1.0)) if o["grid_dt"] else 0
        hard = np.isfinite(o["cond_M"]) and o["cond_M"] >= 1e5
        # A mode located to between 1% and 10% of a posterior standard deviation is reported as a
        # soft failure, matching the row above, and not as a hard one: nothing downstream can tell
        # the difference, and it is the ordinary float32 floor on the larger systems. HIV in
        # single precision reads 5.5e-3 on CPU and 1.2e-2 on GPU, where the reduction order in the
        # linear solve differs -- the same fit, either side of an absolute threshold.
        soft = (not ok_grad) and o["mode_dist"] < 0.1
        clean = ok_neg and ok_null and ok_glob and n_bad == 0 and n_rough == 0 and not hard
        overall = "OK" if (clean and ok_grad) else ("WARN" if (clean and soft) else "FAIL")
        L.append(f'  STATUS: {overall}')
        if clean and soft:
            L.append(f'    Everything passes except the distance to the mode, at '
                     f'{o["mode_dist"]:.1e} posterior sd against a target of 1e-02. That is the '
                     f'precision')
            L.append('    floor rather than a failure to converge; use float64 if an exact '
                     'stationary point is needed.')
        if hard:
            L.append(f'    The curvature varies by {o["cond_M"]:.0e} over the posterior, so no '
                     f'fixed metric fits it and')
            L.append('    HMC or NUTS will mix badly however long they are run. Methods that '
                     'integrate the')
            L.append('    states out rather than traversing the joint geometry are not affected '
                     'in the same way.')
        if n_rough:
            L.append(f'    {n_rough} state component(s) have a GP lengthscale below the grid '
                     f'spacing, so the trajectory')
            L.append('    between observations is not constrained by anything. Expect the fitted '
                     'states to')
            L.append('    interpolate the data and be arbitrary elsewhere, whatever the '
                     'parameters do.')
        if n_bad:
            L.append(f'    {n_bad} parameter(s) improper: the posterior does not integrate, so '
                     f'they have no mean or')
            L.append('    variance, and in a partially identified model they also bias the '
                     'parameters that ARE')
            L.append('    identified by wherever they come to rest. Set a proper theta_prec.')
        elif overall == "FAIL":
            L.append('    The mode itself is not trustworthy; anything built on it inherits that.')
        elif overall == "OK" and not n_rough:
            L.append('    The mode is a unique maximum, every parameter is proper, and the GP '
                     'constrains the states.')
        return "\n".join(L)

    def fit(self, n_nodes=512, seed=0, inner_iters=3, ess_min=0.10, khat_max=0.7,
            inflate=1.3, x0=None, verbose=True, **map_kwargs):
        """
        Fit the posterior by profiling the states out and integrating the parameters.

        The states are integrated out by Laplace at each theta and the p-dimensional parameter
        integral is done by importance sampling; see magi_msvgd.profiled for why this beats a
        Gaussian centred at the mode, and by how much. On every test system with a usable
        reference the largest parameter error lands at or below the level at which two halves of
        that reference agree with each other: 0.0126 against a floor of 0.0100 on FitzHugh-Nagumo,
        0.0093 against 0.0081 on HIV, 0.0119 against 0.0405 on the chaotic Lorenz system. The
        mode alone is 1.03, 0.15 and 1.80 out respectively.

        Falls back to the Laplace approximation when the profiled estimate fails its diagnostics
        (effective sample size below `ess_min`, Pareto k-hat above `khat_max`, or no
        positive-definite profile curvature). That case is not hypothetical: where no parameter is
        identified the mode is already at the reference mean and profiling is worse than doing
        nothing, so the fallback is the correct answer rather than a safety net.

        n_nodes     importance-sampling nodes, on a scrambled Sobol set, so the result is
                    deterministic given `seed`
        inner_iters Gauss-Newton iterations for the inner profile. Three suffices because every
                    node is warm-started from the implicit-function prediction
                    X_MAP + (dX*/dtheta)(theta - theta_MAP) rather than from X_MAP. Measured on
                    FitzHugh-Nagumo the predictor reaches the reference floor at two iterations
                    where starting from X_MAP needs four, so the default carries one iteration of
                    margin over what the measurement requires.
        map_kwargs  forwarded to the Gauss-Newton mode solve (tol, max_iter, check, ...)

        Single precision is fine. The importance weights are barely affected by it -- the spread
        of the float32 error in log p_hat is about 0.007 nats, degrading the effective sample size
        by a factor of 0.997 -- and what once made this method appear to need float64 was the
        finite-difference stencil in the profile-mode step, which divides by h^2 and so amplified
        that noise 2500-fold. See profiled.profile_mode; the default stencil now accounts for it.

        Leaves the result on self.posterior as well as returning it.
        """
        import time
        from profiled import ProfiledPosterior
        if not self._put_called:
            self.put(dtype=jnp.float32)
        t = {}
        t0 = time.time()
        self.map_solve(x0=x0, verbose=False, **map_kwargs)
        x_map = np.asarray(self.map_particle, np.float64)
        t["map"] = time.time() - t0

        # Laplace, always computed: it is the fallback, and its theta scale sets the profile
        # Newton's finite-difference stencil. Jacobi-stabilised because forming the inverse of a
        # Hessian whose coordinates span many orders of magnitude is otherwise unreliable.
        t0 = time.time()
        lap = self._laplace()
        Sig, n_null = lap.Sig, lap.n_null
        t["hessian"] = time.time() - t0

        pp = ProfiledPosterior(self, n_nodes=n_nodes, seed=seed, inner_iters=inner_iters,
                               inflate=inflate)
        pp.build(verbose=False)
        t.update(pp.t)
        p = self.p
        shift = float(np.max(np.abs(pp.theta_hat - x_map[:p]) / np.maximum(lap.sd, 1e-300)))
        reliable = bool(pp.ess / pp.n_nodes >= ess_min
                        and np.isfinite(pp.khat) and pp.khat < khat_max
                        and getattr(pp, "mode_ok", True))
        diag = dict(ess=pp.ess, n_nodes=pp.n_nodes, khat=pp.khat,
                    failed=int((~pp.ok).sum()), mode_shift=shift, n_null=n_null,
                    ess_min=ess_min, khat_max=khat_max)

        if reliable:
            mean = np.concatenate([pp.theta_mean, np.einsum('i,ijk->jk', pp.w, pp.Xstar).ravel()])
            post = MAGIPosterior(self, mean, True, diag, t,
                                 theta_cov=pp.theta_cov, profiled=pp)
        else:
            post = MAGIPosterior(self, x_map, False, diag, t,
                                 theta_cov=Sig[:p, :p], cov=Sig, profiled=pp)
        self.posterior = post
        if verbose:
            print(post)          # MAGIPosterior.report() covers the null directions and the gate
        return post

    def sample(self, k=1000, seed=0, unpack=True, **fit_kwargs):
        """Draw k particles from the fitted posterior, fitting first if needed."""
        if getattr(self, "posterior", None) is None:
            self.fit(**fit_kwargs)
        return self.posterior.sample(k=k, seed=seed, unpack=unpack)

    def condition_A(self, particle=None, scale=0.7, seed=0):
        '''
        Test whether the ODE is affine in the state at fixed theta, which makes p(X | theta)
        exactly Gaussian and opens an exact route that needs no Gaussian approximation at all.

        Returns ||H_XX(X1) - H_XX(X2)|| / ||H_XX|| at two states sharing one theta, which is
        exactly 0 under the condition and grows with the departure from it. Measured 0.0 for
        FitzHugh-Nagumo with its cubic term removed and 0.13 for the real one -- and the size
        of the mean correction the pipeline then needs tracks it monotonically (0.005 at 0.0,
        0.174 at 0.13).
        '''
        if particle is None:
            if getattr(self, 'map_particle', None) is None:
                self.map_solve(verbose=False)
            particle = self.map_particle
        p = self.p
        x1 = np.asarray(particle, np.float64)
        x2 = x1.copy()
        x2[p:p + self.n * self.D] += scale * np.random.default_rng(seed).standard_normal(
            self.n * self.D)
        hfn = self._hessian_fn()
        dt = self.mu.dtype
        A1 = np.asarray(hfn(jnp.asarray(x1, dt)), np.float64)[p:, p:]
        A2 = np.asarray(hfn(jnp.asarray(x2, dt)), np.float64)[p:, p:]
        return float(np.linalg.norm(A1 - A2) / np.linalg.norm(A1))

    def nuts(self, random_seed=8, warmup_steps=1000, sampling_steps=9000, n_chains=4,
             preconditioned=True, target_acceptance_rate=0.9, verbose=True):
        '''
        Exact fallback. Use when fit() reports applied=False, or whenever an exactness
        guarantee is wanted.

        preconditioned : sample in coordinates whitened by the exact Hessian, y = H^(1/2)
            (x - x_MAP), with an identity mass matrix. The metric only affects efficiency, never
            the stationary distribution, so this is not circular -- and the geometry it produces
            is near-isotropic, which is NUTS's best case.

        Runs n_chains independently under vmap and reports split-Rhat, minimum bulk ESS and
        divergences. READ THEM. At the baseline data density a run like this beats an
        independent 8-chain gold standard's own half-vs-half agreement, but on sparser data the
        same configuration did NOT converge (Rhat up to 1.77): being the exact method does not
        make a particular run correct.

        Returns the unpacked (Xs, thetas, sigmas) over all chains pooled.
        '''
        import blackjax
        if not self._put_called:
            self.put(dtype=jnp.float32)
        dt = self.mu.dtype

        if preconditioned:
            if getattr(self, 'map_particle', None) is None:
                self.map_solve(verbose=False)
            xm = jnp.asarray(self.map_particle, dt)
            Hs = np.asarray(self.hessian(np.asarray(xm, np.float64)), np.float64)
            Hs = 0.5 * (Hs + Hs.T)
            w, V = np.linalg.eigh(Hs)
            L = jnp.asarray((V / np.sqrt(np.maximum(w, 1e-10 * w.max()))) @ V.T, dt)
            logp = lambda y: self.magi_logdensity(xm + L @ y)
            start = jnp.zeros(xm.shape[0], dt)
        else:
            logp = self.magi_logdensity
            start = jnp.asarray(self.particles_init, dt)

        def one(key):
            wk, sk = jr.split(key)
            wu = blackjax.window_adaptation(blackjax.nuts, logp,
                                            target_acceptance_rate=target_acceptance_rate)
            (state, params), _ = wu.run(wk, position=start, num_steps=warmup_steps)
            _, (states, info) = blackjax.util.run_inference_algorithm(
                sk, blackjax.nuts(logp, **params), initial_state=state,
                num_steps=sampling_steps)
            return states.position, info.is_divergent.sum()

        Y, ndiv = jax.jit(jax.vmap(one))(jr.split(jr.key(random_seed), n_chains))
        P = xm[None, None, :] + Y @ L.T if preconditioned else Y
        P = np.asarray(P, np.float64)

        c, ns = P.shape[0], P.shape[1]
        S = P.reshape(2 * c, ns // 2, -1)
        W = S.var(1, ddof=1).mean(0)
        B = S.mean(1).var(0, ddof=1)
        Vh = (ns // 2 - 1) / (ns // 2) * W + B
        self.nuts_rhat = np.sqrt(np.maximum(Vh / np.maximum(W, 1e-300), 0))
        self.nuts_ess = np.minimum(2 * c * (ns // 2) * Vh / np.maximum(B, 1e-300),
                                   2 * c * (ns // 2))
        self.nuts_divergences = int(jnp.sum(ndiv))
        if verbose:
            print(f'NUTS: {c} chains x {ns} draws | max Rhat = {self.nuts_rhat.max():.4f} | '
                  f'min ESS = {self.nuts_ess.min():.0f} | divergences = {self.nuts_divergences}')
            if self.nuts_rhat.max() > 1.01:
                print('  max Rhat > 1.01: these draws are NOT a converged sample. Lengthen the '
                      'chains.')
        self.nuts_draws = P.reshape(-1, P.shape[-1])
        return self.unpack_particles(jnp.asarray(self.nuts_draws))
