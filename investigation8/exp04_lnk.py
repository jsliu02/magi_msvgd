"""
exp04: does deleting the 1/ln K from the median heuristic help on the REAL posteriors?

Investigation 7 sec. 10 measured a 39x gain on an isotropic Gaussian and validated both laws
against Ba et al. (ICLR 2022) Cor. 4:

    h = Med          (Ba et al.'s convention)   ->  v = (e-1)^-1 n/d = 0.582 n/d
    h = Med / ln K   (what msvgd ships)         ->  v = ln(K) / d

At K = 400 that is a factor of 0.582*K/lnK = 39. Nothing in that was measured on a MAGI posterior,
and a library change should not rest on a Gaussian. Both conventions here, same everything else,
on fn / hiv / lorenz, from two starts:

  fit()  the profiled-posterior ensemble (the polisher case)
  cold   particles_init + 0.2*N(0, I), exactly as the old MAGI.solve built it

Scored with energy against its K-particle floor AND with metrics8's non-trace scores, since
exp01b showed the energy distance cannot see a trace-preserving misallocation.

PREDICTION (recorded before running): it transfers but disappoints. 39x on the variance ratio
takes fn from 0.018 to 0.7 of the correct variance in the isotropic model, which would be a large
improvement -- but the MAGI posteriors are anisotropic over 4-6 orders of magnitude and a single
adaptive scalar h still cannot serve them, so I expect the energy distance to improve by well
under 39x and to stay far above the floor. I do NOT expect it to reach what the fixed large h of
sec. 1 achieves.
"""
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7
import metrics8 as MM


def _pairwise_plain(particles, h=-1):
    """h = median(||x-y||^2), i.e. Ba et al.'s convention: the shipped rule without the 1/ln K."""
    k = particles.shape[0]
    sq = jnp.sum(particles ** 2, axis=1)
    with jax.default_matmul_precision("highest"):
        L2sq = sq[:, None] + sq[None, :] - 2 * particles @ particles.T
    iu = np.triu_indices(k, k=1)
    med = jnp.median(jnp.clip(L2sq[iu], min=jnp.array(1e-6, dtype=particles.dtype)))
    return L2sq, jnp.where(h <= 0, med, h)


def _std_plain(particles, raw_grad, logp, h=-1):
    L2sq, hh = _pairwise_plain(particles, h)
    return M7._combine(particles, raw_grad, jnp.exp(-L2sq / hh), hh, drift=1.0)


M7.KERNELS["plain_median"] = _std_plain

SYS = sys.argv[1:] or list(H.USABLE)
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 2000))
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    d = S.mean.shape[0]
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    Xc = np.asarray(m.particles_init, np.float64)[None, :] + 0.2 * \
        np.asarray(jr.normal(jr.key(8), (K, d), dtype=m.mu.dtype), np.float64)
    flk = S.energy_floor_k(K)
    sv_fl, ks_fl, wb_fl = MM.floors(S, K)
    ref = MM.ref_split(S)[1]
    print(f"\n===== {name} d={d} K={K} it={MAXIT} | floors: energy {flk:.4f} "
          f"stiffvar {sv_fl:.3f} KS {ks_fl:.4f} =====", flush=True)
    print(f'{"start":>8} {"bandwidth rule":>22} {"SteinR":>8} {"energy":>9} {"x flr":>7} '
          f'{"stiffvar":>9} {"KS/flr":>7} {"thErr":>8} {"h final":>10} {"sec":>6}', flush=True)
    rec = {"floors": dict(energy=flk, stiff_var=sv_fl, ks=ks_fl), "rows": {}}

    for slab, Xs in (("fit()", X0), ("cold", Xc)):
        for klab, kern in (("h = Med / ln K (shipped)", "standard"),
                           ("h = Med (Ba et al.)", "plain_median")):
            t0 = time.time()
            P, _, _ = M7.run_svgd(m, Xs, MAXIT, kernel=kern,
                                  optimizer=optax.contrib.prodigy, optimizer_kwargs={})
            dt = time.time() - t0
            hf = float((_pairwise_plain if kern == "plain_median" else M7._pairwise)(
                jnp.asarray(P, m.mu.dtype), -1.0)[1])
            sc = MM.score(S, P, ref)
            r = dict(steinR=H.stein_R(m, P), energy=S.energy(P), therr=S.theta_err(P, m.p),
                     h_final=hf, sec=dt, **sc)
            print(f'{slab:>8} {klab:>22} {r["steinR"]:>8.4f} {r["energy"]:>9.4f} '
                  f'{r["energy"]/flk:>7.2f} {r["stiff_var"]:>9.4f} {r["ks"]/ks_fl:>7.2f} '
                  f'{r["therr"]:>8.4f} {hf:>10.4g} {dt:>6.1f}', flush=True)
            rec["rows"][f"{slab}|{kern}"] = r
    print(f'{"":>8} {"FLOOR: K exact draws":>22} {1.0:>8.4f} {flk:>9.4f} {1.0:>7.2f} '
          f'{sv_fl:>9.4f} {1.0:>7.2f} {S.theta_err_floor(m.p):>8.4f}', flush=True)
    out[name] = rec
    json.dump(out, open(f"exp04_results_{name}.json", "w"), indent=1)
