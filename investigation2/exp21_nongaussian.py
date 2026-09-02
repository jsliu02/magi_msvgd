"""
Exp 21: does whitened ULA break under non-Gaussianity?

Three distinct things could go wrong, and they have different severities:
  (a) BIAS      -- returns a confidently wrong answer. ULA targets the exact posterior whatever
                   its shape, so in principle this should NOT happen; the metric is only a
                   preconditioner. Test on a banana, where the Laplace approximation is badly
                   wrong but the target is smooth and log-concave-ish.
  (b) MIXING    -- correct but slow, if H at the mode misrepresents the global geometry.
  (c) TRANSIENCE-- ULA's explicit Euler step is unstable for super-quadratic potentials
                   (Roberts & Tweedie 1996): for U ~ |x|^p with p > 2 the drift -grad U ~ |x|^(p-1)
                   OVERSHOOTS, so a chain that wanders past ~eps^(-1/(p-2)) is flung outward and
                   never returns. This is the adversarial case for ODEs: a degree-q polynomial
                   vector field gives an ODE residual ~ x^q and hence a potential ~ x^(2q).
                   FitzHugh-Nagumo is q=3, so its potential is degree SIX.
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import sys
sys.path.insert(0, "/home/jamie/storage-1/github-repos/msvgd/msvgd")
from msvgd import MSVGD

def energy(X, Y, rng, n=1200):
    X = X[rng.choice(len(X), min(n, len(X)), False)]; Y = Y[rng.choice(len(Y), min(n, len(Y)), False)]
    md = lambda A, B: np.sqrt(np.maximum(((A[:,None,:]-B[None,:,:])**2).sum(-1), 0)).mean()
    return float(2*md(X,Y) - md(X,X) - md(Y,Y))

# ---------------------------------------------------------------- (a) banana: strongly non-Gaussian
print("(a) BANANA -- Laplace approximation is badly wrong, target still smooth")
D, B, S = 12, 0.05, 10.0
def banana(x):
    return -(x[0]**2)/(2*S**2) - 0.5*(x[1] - B*(x[0]**2 - S**2))**2 - 0.5*jnp.sum(x[2:]**2)
rng = np.random.default_rng(0)
ex = rng.standard_normal((20000, D)); ex[:,0] *= S
ex[:,1] = B*(ex[:,0]**2 - S**2) + rng.standard_normal(20000)
s = MSVGD(banana)
sk = lambda v: float(((v-v.mean())**3).mean()/v.std()**3)
print(f'{"eps":>8} {"steps":>7} {"energy":>8} {"sd x0":>7} {"sd x1":>7} {"skew x1":>9}  (exact skew '
      f'{sk(ex[:,1]):+.2f})')
for eps in [0.05, 0.01, 2e-3, 5e-4]:
    for nst in [6000, 40000]:
        try:
            P = np.asarray(s.whitened_ula(np.zeros(D), k=4000, n_steps=nst, step_size=eps,
                                          random_seed=0, monitor_convergence=-1))
        except FloatingPointError:
            print(f'{eps:>8.4f} {nst:>7} {"DIVERGED":>8}'); continue
        Z = np.diag(1/ex.std(0))
        print(f'{eps:>8.4f} {nst:>7} {energy((P-ex.mean(0))@Z,(ex-ex.mean(0))@Z,rng):>8.4f} '
              f'{P[:,0].std()/ex[:,0].std():>7.3f} {P[:,1].std()/ex[:,1].std():>7.3f} {sk(P[:,1]):>9.2f}')
print(f"   for reference, the Laplace approximation itself has sd ratio "
      f"{S/ex[:,0].std():.3f}, {1/ex[:,1].std():.3f} -- the metric is badly wrong here")

# ------------------------------------------------- (c) super-quadratic tails: the ODE-adversarial case
print("\n(c) SUPER-QUADRATIC TAILS  U(x) = |x|^2/2 + a|x|^p/p   (Gaussian at the mode, so H is PD)")
print(f'{"p":>3} {"a":>6} {"eps":>7} {"outcome":>34}')
for p, a in [(2, 0.0), (4, 1e-3), (4, 1.0), (6, 1e-3), (6, 1.0)]:
    for eps in [0.05, 0.2]:
        f = (lambda x, p=p, a=a: -0.5*jnp.sum(x**2) - (a/p)*jnp.sum(jnp.abs(x)**p))
        sp = MSVGD(f)
        try:
            P = sp.whitened_ula(np.zeros(8), k=2000, n_steps=4000, step_size=eps,
                                random_seed=0, monitor_convergence=-1)
            print(f'{p:>3} {a:>6.0e} {eps:>7.2f} {f"finite, max|x| = {float(jnp.abs(P).max()):.2f}":>34}')
        except Exception as e:
            print(f'{p:>3} {a:>6.0e} {eps:>7.2f} {type(e).__name__ + " (diverged)":>34}')
