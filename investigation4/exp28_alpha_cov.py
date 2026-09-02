"""
Exp 28: does the covariance error stay second order as the ODE gets more nonlinear?

The claim that the covariance is not worth chasing rests so far on one vector field. There is a
theoretical reason to expect it to generalise. From the exact identity of the paper, the leading
mean shift is driven by the CUBIC term of the potential and is O(Lambda); a correction to the
covariance needs either the quartic term or the square of the cubic, both O(Lambda^2). So as a
problem becomes more non-Gaussian the mean error should grow linearly while the covariance error
grows quadratically from a much smaller base -- meaning the mean dominates over the whole range
where a Gaussian is worth fitting at all.

That is a falsifiable scaling law, and it is tested here by sweeping the FitzHugh-Nagumo cubic
coefficient alpha from 0 (affine in the state, Laplace exact for p(X|theta)) to 1, building a
preconditioned-NUTS reference at each, and measuring mean and covariance error separately against
a floor obtained from each reference's own half-vs-half agreement.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from magi import MAGI
from pipeline import metrics
import blackjax

d, P = H.DIM, 3
K, NW, NS = 64, 1000, 1500
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def make(alpha):
    def ode(X, theta, t=None):
        V, R = X.T; a, b, c = theta
        return jnp.stack([c * (V - alpha * V ** 3 / 3 + R), -1 / c * (V - a + b * R)])
    dd = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")
    g = np.arange(0, 20.001, 0.125)
    full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], dd[:, 0])] = dd
    m = MAGI(ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[0.2, 0.2])
    m.put(dtype=jnp.float64, device=jax.devices()[0])
    return m

for alpha in ALPHAS:
    out = f"refA_{alpha:.2f}.npz"
    if os.path.exists(out):
        print(f"alpha={alpha}: cached", flush=True); continue
    m = make(alpha)
    m.map_solve(verbose=False, tol=1e-9, max_iter=200)
    xm = jnp.asarray(m.map_particle)
    Hs = np.asarray(m.hessian(), np.float64); Hs = 0.5 * (Hs + Hs.T)
    w, V = np.linalg.eigh(Hs)
    L = jnp.asarray((V / np.sqrt(np.maximum(w, 1e-10 * w.max()))) @ V.T)
    logp = lambda y: m.magi_logdensity(xm + L @ y)
    def one(key):
        wk, sk = jax.random.split(key)
        wu = blackjax.window_adaptation(blackjax.nuts, logp, target_acceptance_rate=0.9)
        (st, par), _ = wu.run(wk, position=jnp.zeros(d), num_steps=NW)
        _, (states, info) = blackjax.util.run_inference_algorithm(
            sk, blackjax.nuts(logp, **par), initial_state=st, num_steps=NS)
        return states.position, info.is_divergent.sum()
    t0 = time.time()
    Y, nd = jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(3), K))
    Y.block_until_ready(); dt = time.time() - t0
    Pp = np.asarray(xm[None, None, :] + Y @ L.T, np.float64)
    c, ns = K, NS
    S = Pp.reshape(2 * c, ns // 2, -1)
    W = S.var(1, ddof=1).mean(0); B = S.mean(1).var(0, ddof=1)
    Vh = (ns // 2 - 1) / (ns // 2) * W + B
    rh = np.sqrt(np.maximum(Vh / np.maximum(W, 1e-300), 0))
    F = Pp.reshape(-1, d)
    np.savez(out, mean=F.mean(0), cov=np.cov(F, rowvar=False), rhat=rh, div=int(jnp.sum(nd)),
             half_mean=np.stack([Pp[:K//2].reshape(-1,d).mean(0), Pp[K//2:].reshape(-1,d).mean(0)]),
             half_cov=np.stack([np.cov(Pp[:K//2].reshape(-1,d), rowvar=False),
                                np.cov(Pp[K//2:].reshape(-1,d), rowvar=False)]), sec=dt)
    print(f'alpha={alpha}: {len(F)} draws in {dt:.0f}s, max Rhat {rh.max():.4f}, '
          f'div {int(jnp.sum(nd))}', flush=True)

print()
print(f'{"alpha":>6} {"|D3|":>7} | {"MEAN: MAP":>10} {"mu3":>8} {"floor":>8} | '
      f'{"COV: forst":>11} {"floor":>8} {"excess":>8} | {"med|var-1|":>11} {"floor":>8}')
print("-" * 108)
for alpha in ALPHAS:
    z = np.load(f"refA_{alpha:.2f}.npz")
    sc = metrics(z["mean"], z["cov"], d); I = np.eye(d)
    hm, hc = z["half_mean"], z["half_cov"]
    fl = metrics(hm[1], hc[1], d)(hm[0], hc[0])
    m = make(alpha)
    post = m.fit(n_pairs=1024, verbose=False, tol=1e-9, max_iter=200)
    Hs = np.asarray(m.hessian(post.mu_map), np.float64); Hs = 0.5 * (Hs + Hs.T)
    ev, Vv = np.linalg.eigh(Hs); Sig = (Vv / ev) @ Vv.T
    a_map, a_mu3 = sc(post.mu_map, I)["bias"], sc(post.mu3, I)["bias"]
    cf = sc(post.mu_map, Sig)["forst"]
    # marginal-variance error, the quantity a user reads off a credible interval
    mv = np.median(np.abs(np.sqrt(np.diag(Sig) / np.diag(z["cov"])) ** 2 - 1))
    mvf = np.median(np.abs(np.diag(hc[0]) / np.diag(hc[1]) - 1))
    ex = np.sqrt(max(cf ** 2 - fl["forst"] ** 2, 0))
    print(f'{alpha:>6.2f} {post.certificates["d3"]:>7.4f} | {a_map:>10.4f} {a_mu3:>8.4f} '
          f'{fl["bias"]:>8.4f} | {cf:>11.4f} {fl["forst"]:>8.4f} {ex:>8.4f} | '
          f'{mv:>11.4f} {mvf:>8.4f}')
