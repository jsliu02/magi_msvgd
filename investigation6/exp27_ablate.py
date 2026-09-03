"""
Exp 27: re-run the two remaining pre-fix ablations quoted in the writeup.

(a) The implicit-function warm start (tab:predictor): largest parameter error and wall clock on
    FitzHugh-Nagumo, starting each profile node from X_MAP versus from the linear predictor
    X_MAP + (dX*/dtheta)(theta - theta_MAP), across inner iteration counts.
(b) The proposal: drawing theta from the joint Laplace marginal N(theta_MAP, (H^-1)_thth) instead
    of from the profiled mode and curvature. The writeup quotes Hes1 log-weights spanning 8,400
    nats at ESS 0.6%, measured before the hyperparameter fit.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
from profiled import ProfiledPosterior, _pareto_k
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")


def laplace(m):
    """(theta_MAP, Sigma_thth) from the exact Hessian, Jacobi-stabilised."""
    x0 = np.asarray(m.map_particle, np.float64)
    H = np.asarray(m.hessian(x0), np.float64); H = 0.5 * (H + H.T)
    d = np.sqrt(np.maximum(np.abs(np.diag(H)), 1e-300))
    w, V = np.linalg.eigh(H / np.outer(d, d)); k = w > 1e-10 * max(abs(w).max(), 1e-300)
    Sig = ((V[:, k] / w[k]) @ V[:, k].T) / np.outer(d, d)
    return x0, Sig


print("=== (a) warm start, FitzHugh-Nagumo ===")
z = np.load(R("fn")); rm, rs = z["mean"], np.sqrt(np.maximum(np.diag(z["cov"]), 0))
hm = z["half_mean"]
print(f'{"start":>14} ' + " ".join(f'{f"{k} inner iters":>18}' for k in (2, 4, 8)))
for lbl, pred in (("X_MAP", False), ("predictor", True)):
    row = []
    for k in (2, 4, 8):
        m, ds = build("fn"); p = m.p
        t0 = time.time()
        m.map_solve(verbose=False)
        pp = ProfiledPosterior(m, n_nodes=512, seed=0, inner_iters=k)
        if not pred:
            orig = pp.logp
            pp.logp = lambda TH, X0=None, predict=True, _o=orig: _o(TH, X0, False)
        pp.build(verbose=False)
        el = time.time() - t0
        e = np.abs((pp.theta_mean - rm[:p]) / np.maximum(rs[:p], 1e-300)).max()
        row.append(f'{e:.4f} / {el:.1f} s')
    print(f'{lbl:>14} ' + " ".join(f'{r:>18}' for r in row), flush=True)
print(f'chain-to-chain floor {np.abs((hm[0][:3] - hm[1][:3]) / np.maximum(rs[:3], 1e-300)).max():.4f}\n')

print("=== (b) proposal: joint Laplace marginal vs profiled ===")
from scipy.stats import qmc, norm
print(f'{"system":>8} {"proposal":>16} {"ESS":>8} {"khat":>7} {"logw span":>11} {"failed":>7}')
for name in SYSTEMS:
    m, ds = build(name); p = m.p
    m.map_solve(verbose=False)
    pp = ProfiledPosterior(m, n_nodes=512, seed=0)
    pp.build(verbose=False)
    lr = pp.log_ratio[np.isfinite(pp.log_ratio)]
    print(f'{name:>8} {"profiled":>16} {pp.ess/pp.n_nodes:>7.1%} {pp.khat:>7.2f} '
          f'{lr.max()-lr.min():>11.1f} {int((~pp.ok).sum()):>7}', flush=True)

    # same node budget, but proposed from the joint Laplace marginal at the joint mode
    x0, Sig = laplace(m)
    S = Sig[:p, :p]
    wq, Vq = np.linalg.eigh(0.5 * (S + S.T))
    L = (Vq * np.sqrt(np.maximum(wq, 1e-300))) @ Vq.T * pp.inflate
    Z = norm.ppf(np.clip(qmc.Sobol(d=p, scramble=True, seed=0).random(pp.n_nodes), 1e-12, 1-1e-12))
    TH = x0[None, :p] + Z @ L.T
    lp, Xs, ok = pp.logp(TH)
    lq = -0.5 * np.sum(Z ** 2, axis=1)
    lr = np.where(np.isfinite(lp) & ok, lp - lq, -np.inf); lr -= lr.max()
    w = np.exp(lr); w /= w.sum()
    fin = lr[np.isfinite(lr)]
    print(f'{"":>8} {"joint Laplace":>16} {1/np.sum(w**2)/pp.n_nodes:>7.1%} '
          f'{_pareto_k(np.exp(fin)):>7.2f} {fin.max()-fin.min():>11.1f} '
          f'{int((~ok).sum()):>7}', flush=True)
