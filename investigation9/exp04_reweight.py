"""
exp04: can importance reweighting recover the parameters from the equilibrium ensemble?

exp02 shows the theta displacement is a finite-K bias, but a slowly decaying one, and the decay
rate is system dependent:

    fn      |t| = 0.394 (K=100)  0.186 (K=400)  0.051 (K=1600)   ~ K^-0.74
    lorenz  |t| = 0.299          0.221          0.110            ~ K^-0.36

and SVGD's theta error stays 1.4-3.1x fit()'s on fn and 4.1-5.7x on lorenz at every K tested. So
buying it off needs ~1.5e4 particles on fn and ~1.2e5 on lorenz. The remaining hope is a
post-hoc correction, and the ensemble is well placed for one: after the 1/sqrt(R) step its SPREAD
matches the reference (whsd^2 = 1.02-1.06, stiff-var 1.065), so only its location is wrong, and a
mislocated proposal with the right scale is exactly what importance sampling exists to fix.

SVGD returns particles, not a density, so the proposal has to be supplied. The honest choice is a
Gaussian fitted to the ensemble's own theta block, q = N(mean, cov) of the rescaled ensemble; then

    log w_i = log p_hat(theta_i) - log q(theta_i)

with log p_hat the profiled log marginal from profiled.ProfiledPosterior.logp -- the same quantity
fit() integrates, so this is fit() with its Sobol proposal replaced by the SVGD ensemble.
Reads the equilibrium ensembles saved by exp01 (eq_*.npz, regenerate by rerunning exp01).
Reported with ESS and Pareto k-hat, which are the diagnostics that decide whether the weights are
usable, and against fit() at the same K.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, json, time
jax.config.update("jax_enable_x64", True)
import harness9 as H
from profiled import ProfiledPosterior, _pareto_k

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
out = {}

for name in SYS:
    z = np.load(f"eq_{name}.npz")
    P, Pc, X0 = z["P"], z["Pc"], z["X0"]
    m, ds = H.build(name)
    S = H.Scorer(name)
    p = m.p
    post = m.fit(verbose=False)
    K = P.shape[0]
    pp = ProfiledPosterior(m, n_nodes=8, inner_iters=3)
    sd = S.sd[:p]
    b = S.mean[:p]

    def err(TH):
        return float(np.max(np.abs(TH.mean(0) - b) / sd))

    def werr(TH, w):
        mu = w @ TH
        return float(np.max(np.abs(mu - b) / sd))

    print(f"\n===== {name} p={p} K={K} =====", flush=True)
    print(f'{"estimate":>36} {"max|theta err|":>15} {"x floor":>8} {"ESS":>10} {"khat":>7}',
          flush=True)
    fl = S.theta_err_floor(p)
    rec = {"floor": fl, "rows": {}}

    def show(lab, e, ess=np.nan, kh=np.nan):
        print(f'{lab:>36} {e:>15.4f} {e/fl:>8.2f} '
              f'{("%.0f (%.0f%%)" % (ess, 100*ess/K)) if np.isfinite(ess) else "":>10} '
              f'{kh if np.isfinite(kh) else float("nan"):>7.2f}', flush=True)
        return dict(err=e, ess=float(ess), khat=float(kh))

    rec["rows"]["fit"] = show("fit() sample at same K", err(np.asarray(
        post.sample(K, unpack=False), np.float64)[:, :p]))
    rec["rows"]["start"] = show("start: fit() draws", err(X0[:, :p]))
    rec["rows"]["equilibrium"] = show("precond SVGD equilibrium", err(P[:, :p]))
    rec["rows"]["rescaled"] = show("  + 1/sqrt(R) rescaling", err(Pc[:, :p]))

    TH = np.asarray(Pc[:, :p], np.float64)
    t0 = time.time()
    lp, Xs, ok = pp.logp(TH)
    mu_q, cov_q = TH.mean(0), np.cov(TH, rowvar=False) + 1e-12 * np.eye(p)
    sgn, ld = np.linalg.slogdet(cov_q)
    Ciq = np.linalg.inv(cov_q)
    dq = TH - mu_q
    lq = -0.5 * (np.einsum('ij,jk,ik->i', dq, Ciq, dq) + ld + p * np.log(2 * np.pi))
    lw = np.where(ok & np.isfinite(lp), lp - lq, -np.inf)
    lw -= lw.max()
    w = np.exp(lw); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    kh = _pareto_k(np.sort(np.exp(lw)))
    dt = time.time() - t0
    rec["rows"]["reweighted"] = show(f"  + IS reweight by log p_hat ({dt:.0f}s)",
                                     werr(TH, w), ess, kh)
    print(f'{"FLOOR: reference half-vs-half":>36} {fl:>15.4f} {1.0:>8.2f}', flush=True)
    print(f'   failed profile solves: {int((~ok).sum())}/{K}; '
          f'log-weight spread {float(lw.max()-lw[np.isfinite(lw)].min()):.1f} nats', flush=True)
    out[name] = rec
    json.dump(out, open(f"exp04_results_{name}.json", "w"), indent=1)
