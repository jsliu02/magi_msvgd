"""
exp07: if the preconditioned error is a pure scale error, rescaling by Stein R should remove it.

exp05e (after the exp05f bug fix) finds that preconditioned SVGD with a fixed bandwidth and a
fixed step has a genuine attractor -- reached from 4x-narrow, correct and 4x-wide starts alike --
whose defect is a UNIFORM variance deficit rather than the misallocation the unpreconditioned
dynamics produces. On fn at h = 10*h*, all three starts land on stiff-var 0.75 / whsd^2 0.72,
i.e. the same deficit in the stiffest directions as in the median ones. Compare unpreconditioned
at h = 100*h0: whsd^2 0.95 but stiff-var 0.048, a 20-fold mismatch.

A uniform deficit is a one-parameter error, and Stein R measures exactly that parameter without a
reference: for a Gaussian target R = tr(A Sigma^-1)/dim, so an ensemble that is uniformly a factor
c too narrow reads R = c. Rescaling the ensemble about its own mean by 1/sqrt(R) should therefore
correct it -- with no reference, no tuning, and no extra gradient evaluations.

Scored before and after the rescaling, on all three systems, with the non-trace metrics.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness8 as H
import msvgd8 as M7
import metrics8 as MM

SYS = sys.argv[1:] or ["fn", "lorenz", "hiv"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 100000))
MULT = float(os.environ.get("MULT", 10))
LR = float(os.environ.get("LR", 0.01))
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    mu0 = X0.mean(0)
    x_map, L = M7.laplace_metric(m)
    hstar = 2.0 * d / np.log(K)
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, _ = MM.floors(S, K)
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} d={d} K={K} precond h={MULT:g}h* lr={LR:g} it={MAXIT} | "
          f"floors energy {flk:.4f} stiffvar {sv_fl:.3f} KS {ks_fl:.4f} =====", flush=True)
    print(f'{"ensemble":>34} {"SteinR":>8} {"energy":>9} {"x flr":>7} {"stiffvar":>9} '
          f'{"whsd^2":>8} {"KS/flr":>7} {"thErr":>8}', flush=True)

    def row(lab, P):
        R = H.stein_R(m, P)
        sc = MM.score(S, P, ref)
        r = dict(steinR=R, energy=S.energy(P), whsd2=S.mahalanobis_sd(P) ** 2,
                 therr=S.theta_err(P, m.p), **sc)
        print(f'{lab:>34} {R:>8.4f} {r["energy"]:>9.4f} {r["energy"]/flk:>7.2f} '
              f'{r["stiff_var"]:>9.4f} {r["whsd2"]:>8.3f} {r["ks"]/ks_fl:>7.2f} '
              f'{r["therr"]:>8.4f}', flush=True)
        return r

    rec = {"floor": flk, "sv_floor": sv_fl, "ks_floor": ks_fl}
    rec["start"] = row("start: fit() draws", X0)
    t0 = time.time()
    P, _, _ = M7.run_svgd(m, X0, MAXIT, kernel="standard", precond=(x_map, L),
                          bandwidth=MULT * hstar, optimizer=optax.sgd,
                          optimizer_kwargs={"learning_rate": LR})
    dt = time.time() - t0
    rec["equilibrium"] = row(f"precond SVGD ({dt:.0f}s)", P)
    Rq = rec["equilibrium"]["steinR"]
    mu = P.mean(0)
    Pc = mu[None, :] + (P - mu) / np.sqrt(max(Rq, 1e-12))
    rec["corrected"] = row(f"  rescaled by 1/sqrt(R={Rq:.3f})", Pc)
    print(f'{"FLOOR: K exact draws":>34} {1.0:>8.4f} {flk:>9.4f} {1.0:>7.2f} '
          f'{sv_fl:>9.4f} {1.0:>8.3f} {1.0:>7.2f} {S.theta_err_floor(m.p):>8.4f}', flush=True)
    out[name] = rec
    json.dump(out, open(f"exp07_results_{name}.json", "w"), indent=1)
