"""
Exp 23: pSVGD (Chen & Ghattas 2020) on the FHN/MAGI posterior.

Formulated in prior-whitened coordinates, which is equivalent to the paper's generalized
eigenproblem and makes the coefficient prior exactly standard normal:

    Gamma = prior covariance,  Ls = chol(Gamma),  u = Ls^-1 x   so  u ~ N(0, I) under the prior
    S     = Ls^T H Ls          the likelihood GIM in u-coordinates (dimensionless spectrum)
    V_r   = top-r eigenvectors of S
    u     = V_r w + u_perp,    w in R^r sampled,  u_perp frozen at a per-particle prior draw
    grad_w log pi(w) = V_r^T Ls^T grad_x log p(x)     [exactly the paper's Eq 28/30: the prior
                                                       term -w falls out automatically]

SVGD then runs in R^r with an RBF kernel and the median heuristic. Algorithm 2 (adaptive) also
re-estimates S from the current samples and rebuilds V_r periodically.

Two initializations are compared, because the paper's prescription (start from prior draws) is
severe here: MAGI's theta prior is improper, so any Gamma for it is our choice, and prior draws
put theta far from the posterior. The second variant starts the coefficients at the MAGI
initialization so the comparison isolates the projection from the starting point.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time
from functools import partial
import harness as H

G = H.Gold()
THETA_SD = 10.0

def build_prior(m):
    n, D, P = m.n, m.D, m.p
    beta = float(m.beta_inv)
    Cinv = np.asarray(m.C_invs, np.float64)
    prec = np.zeros((H.DIM, H.DIM))
    for j in range(D):
        c = P + np.arange(n) * D + j
        prec[np.ix_(c, c)] = beta * Cinv[j]
    prec[:P, :P] = np.eye(P) / THETA_SD ** 2
    Gamma = np.linalg.inv(prec)
    return np.linalg.cholesky(0.5 * (Gamma + Gamma.T) + 1e-14 * np.eye(H.DIM))

def basis(Ls, X, m, r):
    """top-r eigenvectors of the likelihood GIM in prior-whitened coordinates"""
    gp = np.asarray(m.gradient(jnp.asarray(X, jnp.float32), m.data), np.float64)
    gpri = -np.linalg.solve(Ls @ Ls.T, X.T).T          # grad log N(0, Gamma)
    gf = gp - gpri
    S = Ls.T @ (gf.T @ gf / len(X)) @ Ls
    lam, V = np.linalg.eigh(0.5 * (S + S.T))
    return V[:, ::-1][:, :r], lam[::-1]

@jax.jit
def svgd_phi(w, gw):
    """SVGD ascent direction in coefficient space, RBF kernel + median heuristic."""
    kp = w.shape[0]
    sq = jnp.sum(w ** 2, axis=1)
    L2 = jnp.maximum(sq[:, None] + sq[None, :] - 2 * w @ w.T, 0.0)
    h = jnp.maximum(jnp.median(L2) / jnp.log(kp), 1e-10)
    K = jnp.exp(-L2 / h)
    rep = (K.sum(1, keepdims=True) * w - K @ w) * (2.0 / h)
    return (K @ gw + rep) / kp

def run_psvgd(m, Ls, r, k=800, n_iter=1000, lr=1e-2, adapt_every=100, init="prior",
              seed=0, opt_name="adam"):
    rng = np.random.default_rng(seed)
    Ls32 = jnp.asarray(Ls, jnp.float32)
    u0 = rng.standard_normal((k, H.DIM))                       # prior draws in u-coords
    X0 = u0 @ Ls.T
    Vr, lam = basis(Ls, X0, m, r)
    if init == "magi":                                          # same complement, informed start
        u_init = np.linalg.solve(Ls, np.asarray(m.particles_init, np.float64))
        u0 = u_init[None, :] + 0.01 * rng.standard_normal((k, H.DIM))
    w = u0 @ Vr
    u_perp = u0 - w @ Vr.T
    opt = {"adam": optax.adam(lr), "adagrad": optax.adagrad(lr),
           "prodigy": optax.contrib.prodigy()}[opt_name]
    wj = jnp.asarray(w, jnp.float32)
    state = jax.vmap(opt.init)(wj)
    g0 = None
    for it in range(1, n_iter + 1):
        Vr32 = jnp.asarray(Vr, jnp.float32); up32 = jnp.asarray(u_perp, jnp.float32)
        X = (jnp.asarray(w, jnp.float32) @ Vr32.T + up32) @ Ls32.T
        gx = m.gradient(X, m.data)
        gw = (gx @ Ls32) @ Vr32
        phi = svgd_phi(wj, gw)
        if g0 is None:
            g0 = float(jnp.abs(gw).max())
        upd, state = jax.vmap(opt.update)(-phi, state, wj)
        wj = optax.apply_updates(wj, upd)
        w = np.asarray(wj, np.float64)
        if not np.all(np.isfinite(w)):
            return None, lam, it
        if adapt_every and it % adapt_every == 0 and it < n_iter:
            Xc = np.asarray(X, np.float64)
            Vr, lam = basis(Ls, Xc, m, r)
            u_cur = np.linalg.solve(Ls, Xc.T).T
            w = u_cur @ Vr
            u_perp = u_cur - w @ Vr.T
            wj = jnp.asarray(w, jnp.float32)
            state = jax.vmap(opt.init)(wj)
    X = (wj @ jnp.asarray(Vr, jnp.float32).T
         + jnp.asarray(u_perp, jnp.float32)) @ Ls32.T
    return X, lam, n_iter

def main():
    m = H.build_magi()
    Ls = build_prior(m)
    print(H.HDR); print("-" * len(H.HDR))
    out = []
    for init, r, opt_name, lr in [(i, r, o, l) for i in ["prior", "magi"]
                                  for r in [50] for o in ["adam", "adagrad", "prodigy"]
                                  for l in ([1e-2, 1e-4] if o != "prodigy" else [1.0])]:
            t0 = time.time()
            X, lam, it = run_psvgd(m, Ls, r, init=init, opt_name=opt_name, lr=lr)
            tag = f"r={r} {init} {opt_name} {lr:g}"
            if X is None:
                print(f'{tag:>26}   DIVERGED at iter {it}'); continue
            rr = H.evaluate(X, m, tag=tag)
            pr = ((np.asarray(X, np.float64) - G.mean) @ G.evecs).var(0) / G.evals
            rr["varwtd"] = float(np.sum(pr * G.evals) / np.sum(G.evals))
            rr["r"] = r; rr["init"] = init; rr["t"] = time.time() - t0
            H.show(rr); out.append(rr)
    H.show(H.gold_row())
    print(f'\n{"variant":>26} {"varwtd":>7} {"sec":>6}')
    for rr in out:
        print(f'{rr["tag"]:>26} {rr["varwtd"]:>7.3f} {rr["t"]:>6.1f}')
    H.save(out, "exp23_psvgd_results")

if __name__ == "__main__":
    main()
