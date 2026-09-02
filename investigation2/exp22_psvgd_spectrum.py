"""
Exp 22: is the pSVGD premise true for MAGI?

pSVGD (Chen & Ghattas 2020) assumes the posterior differs from the prior only in a
low-dimensional, data-informed subspace. It builds the gradient information matrix of the
LIKELIHOOD, H = E[grad log f (grad log f)^T], solves the generalized eigenproblem H psi =
lambda Gamma psi against the prior covariance Gamma, keeps the top r eigenvectors and freezes
the complement at prior samples. Their criterion for r is lambda_r < 1e-2.

That premise is an empirical property of the problem, so test it before implementing anything.
For MAGI the natural Gaussian prior is the GP on the latent trajectory, so

    log p_0(x) = -0.5 * beta_inv * sum_d (X_d - mu_d)^T C_d^{-1} (X_d - mu_d),
    Gamma      = blockdiag(C_d / beta_inv)  for the trajectory, broad for theta,
    grad log f = grad log p - grad log p_0.

Note this puts the ODE term on the likelihood side: it is not Gaussian, so it cannot go into
Gamma. That is a real difference from the paper's examples, where the likelihood is a handful of
observations and the informed subspace is genuinely tiny.

H is estimated at gold-standard posterior draws -- the best case, since the paper must bootstrap
it adaptively from the prior.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H

G = H.Gold()
jax.config.update("jax_enable_x64", True)
m = H.build_magi(dtype=jnp.float64)
n, D, P, DIM = m.n, m.D, m.p, H.DIM
beta = float(m.beta_inv)
Cinv = np.asarray(m.C_invs, np.float64)                    # (D, n, n)
mu = np.asarray(m.mu, np.float64)                          # (n, D)

# prior precision and covariance in the particle layout
Prior_prec = np.zeros((DIM, DIM))
idx_of = lambda j: P + np.arange(n) * D + j
for j in range(D):
    Prior_prec[np.ix_(idx_of(j), idx_of(j))] = beta * Cinv[j]
THETA_SD = 10.0                                            # broad but proper, so Gamma is PD
Prior_prec[:P, :P] = np.eye(P) / THETA_SD ** 2
Gamma = np.linalg.inv(Prior_prec)

def grad_log_prior(X):
    g = np.zeros_like(X)
    for j in range(D):
        c = idx_of(j)
        g[:, c] = -beta * (X[:, c] - mu[None, :, j]) @ Cinv[j].T
    g[:, :P] = -X[:, :P] / THETA_SD ** 2
    return g

rng = np.random.default_rng(0)
sub = G.pos[rng.choice(len(G.pos), 2000, replace=False)]
gp_full = np.asarray(m.gradient(jnp.asarray(sub), m.data), np.float64)
gf = gp_full - grad_log_prior(sub)                          # likelihood gradient
Hgim = gf.T @ gf / len(sub)

# generalized eigenproblem H psi = lambda Gamma psi, via the symmetric form
Ls = np.linalg.cholesky(Gamma + 1e-14 * np.eye(DIM))
lam = np.linalg.eigvalsh(Ls.T @ Hgim @ Ls)[::-1]
lam = np.maximum(lam, 0)

print(f"generalized eigenvalues of (H_likelihood, Gamma_prior), d = {DIM}")
print(f"  lambda_1 = {lam[0]:.4e}   lambda_10 = {lam[9]:.4e}   lambda_100 = {lam[99]:.4e}"
      f"   lambda_325 = {lam[-1]:.4e}")
for thr in [1e-2, 1e-1, 1.0]:
    r = int(np.sum(lam > thr))
    print(f"  r with lambda_r > {thr:<5g}: {r:>4}  ({100*r/DIM:.0f}% of d)"
          + ("   <- the paper's criterion" if thr == 1e-2 else ""))
print(f"\n  paper's error bound: KL <= (gamma/2) * sum_{{i>r}} lambda_i")
tot = lam.sum()
for r in [10, 25, 50, 100, 200, 300]:
    print(f"    r={r:>4}: truncated mass sum_{{i>r}} lambda_i = {lam[r:].sum():.4e}"
          f"  ({100*lam[r:].sum()/tot:5.1f}% of total)")
np.save("psvgd_eigvals.npy", lam)
