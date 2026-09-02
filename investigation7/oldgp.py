"""
A read-only reconstruction of the PRE-FIX GP hyperparameter fit, for isolating how much of
investigation 4's mSVGD result was caused by that bug.

`magi_msvgd/_initializer.py` is settled and must not be edited, so this file carries its own copy
of `fit_phisigma` with exactly the two post-fix changes reverted:

  * the optimisation starts at `log phi1 = log phi2 = log sigma = 0`, i.e. phi1 = phi2 = sigma = 1
    whatever the data's units are, instead of at the data's own marginal variance and the FFT
    lengthscale estimate;
  * the fitted lengthscale is not confined to [2 dt, span/4].

Everything else -- the objective, the prior on phi2, the BFGS call -- is verbatim. Installed by
`install()`, which rebinds the name inside the `_initializer` module so that
`run_initialization`'s global lookup picks it up; `restore()` puts the real one back.
"""
import os, sys
import jax, jax.numpy as jnp
from jax.scipy import optimize as jsp_optimize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "magi_msvgd"))
import _initializer as _I

_REAL = _I.fit_phisigma


def old_fit_phisigma(I, x_init, phis, sigmas):
    I_max = I.max()
    Id_n = jnp.identity(I.shape[0], dtype=I.dtype)
    n_freqs = (x_init.shape[0] - 1) // 2
    idxs = jnp.linspace(1, n_freqs, n_freqs, dtype=x_init.dtype)

    def neglogprob(phi, sigma, mu_phi2, sig_phi2, y_d):
        cov = _I.matern_v01(I, I, phi1=phi[0], phi2=phi[1]) + Id_n * sigma ** 2
        t1 = (phi[1] - mu_phi2) ** 2 / sig_phi2 ** 2
        t2 = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(cov))))
        with jax.default_matmul_precision("highest"):
            t3 = y_d @ jnp.linalg.solve(cov, y_d)
        return 0.5 * (t1 + t2 + t3)

    def target(log_phi, log_sigma, mu_phi2, sig_phi2, y_d):
        return neglogprob(jnp.exp(log_phi), jnp.exp(log_sigma), mu_phi2, sig_phi2, y_d)

    targets = (phis, sigmas)

    def loop(d, targets):
        phis, sigmas = targets
        y_d = x_init[:, d]
        z = jnp.fft.fft(y_d)
        zmod = jnp.abs(z)
        zmod_effective_sq = zmod[1:(zmod.shape[0] - 1) // 2 + 1] ** 2
        freq = jnp.sum(idxs * zmod_effective_sq) / jnp.sum(zmod_effective_sq)
        mu_phi2 = 0.5 / freq
        sig_phi2 = (I_max - mu_phi2) / 3
        sigma_d = sigmas[d]

        # THE BUG: start at 1 in every variable, regardless of the data's units, and do not
        # confine the resulting lengthscale.
        def unknown_sigma(sigma_d, mu_phi2, sig_phi2, y_d):
            objective = lambda u: target(u[:2], u[2], mu_phi2, sig_phi2, y_d)
            r = jsp_optimize.minimize(objective, x0=jnp.zeros(3, dtype=x_init.dtype),
                                      method="BFGS")
            return jnp.exp(r.x)

        def known_sigma(sigma_d, mu_phi2, sig_phi2, y_d):
            objective = lambda u: target(u, jnp.log(sigma_d), mu_phi2, sig_phi2, y_d)
            r = jsp_optimize.minimize(objective, x0=jnp.zeros(2, dtype=x_init.dtype),
                                      method="BFGS")
            f = jnp.exp(r.x)
            return jnp.array([f[0], f[1], sigma_d], dtype=x_init.dtype)

        fitted = jax.lax.cond(jnp.isnan(sigma_d) | (sigma_d < 0),
                              unknown_sigma, known_sigma,
                              *[sigma_d, mu_phi2, sig_phi2, y_d])
        phis = phis.at[d].set(fitted[:2])
        sigmas = sigmas.at[d].set(fitted[2])
        return phis, sigmas

    return jax.lax.fori_loop(0, x_init.shape[1], loop, targets)


def install():
    _I.fit_phisigma = old_fit_phisigma


def restore():
    _I.fit_phisigma = _REAL
