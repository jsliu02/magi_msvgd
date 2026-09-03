"""
exp02: at a large fixed bandwidth, is SVGD a sampler or only a polisher?

Investigation 7 sec. 9 saw convergence from a correctly-scaled start but not, within 2000
iterations, from one 4x too narrow, and sec. 6 suggested on an isotropic Gaussian that this was a
budget question (5000 iterations sufficed there). If it is a budget question the budget should be
stated; if the ensemble plateaus short of the target, the sec. 13 recommendation needs qualifying.

Four starts, all built from the fit() ensemble so the only thing that varies is where they are put:

  correct     the fit() ensemble itself (control -- must stay)
  narrow      shrunk 4x about its mean
  wide        stretched 4x about its mean
  displaced   correct spread, mean moved 5 reference sd along the reference covariance's
              LEAST-variable direction, which is the direction the posterior constrains hardest
              and therefore the hardest place to be wrong

20,000 iterations, checkpointed every 1000, so the trajectory is reported rather than the
endpoint. Bandwidth is the rule's pick from exp01.
"""
import numpy as np, jax.numpy as jnp, optax, time, sys, os, json
import harness8 as H
import msvgd8 as M7

SYS = sys.argv[1:] or ["fn", "lorenz"]
K = int(os.environ.get("K", 400))
MAXIT = int(os.environ.get("MAXIT", 20000))
EVERY = int(os.environ.get("EVERY", 1000))
MULT = float(os.environ.get("MULT", 1000))
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    T = S.theta_scorer(m.p)
    post = m.fit(verbose=False)
    X0 = np.asarray(post.sample(K, unpack=False), np.float64)
    h0 = float(M7._pairwise(jnp.asarray(X0, m.mu.dtype), -1.0)[1])
    flk = S.energy_floor_k(K)
    mu0 = X0.mean(0)

    w, V = np.linalg.eigh(0.5 * (S.cov + S.cov.T))
    stiff = V[:, int(np.argmin(w))]                       # least-variable reference direction
    shift = 5.0 * np.sqrt(max(w.min(), 0.0)) * stiff

    STARTS = {"correct":   X0,
              "narrow 4x": mu0 + 0.25 * (X0 - mu0),
              "wide 4x":   mu0 + 4.00 * (X0 - mu0),
              "displaced": X0 + shift[None, :]}

    print(f"\n===== {name} K={K} h={MULT:g}*h0 ({MULT*h0:.4g}) maxit={MAXIT} "
          f"floor={flk:.4f} =====", flush=True)
    rec = {"h0": h0, "mult": MULT, "floor": flk, "starts": {}}
    for slab, Xs in STARTS.items():
        print(f'-- start "{slab}": energy {S.energy(Xs):.4f} ({S.energy(Xs)/flk:.1f}x) '
              f'whsd {S.mahalanobis_sd(Xs):.3f} Stein R {H.stein_R(m, Xs):.4f}', flush=True)
        print(f'{"iter":>8} {"energy":>9} {"x floor":>8} {"whsd":>7} {"SteinR":>8} '
              f'{"thErr":>8} {"sec":>7}', flush=True)
        t0 = time.time()
        P, Rs, hist = M7.run_svgd(m, Xs, MAXIT, kernel="standard", bandwidth=MULT * h0,
                                  optimizer=optax.contrib.prodigy, optimizer_kwargs={},
                                  record_every=EVERY)
        dt = time.time() - t0
        traj = []
        for it, Q in hist:
            r = dict(it=it, energy=S.energy(Q), whsd=S.mahalanobis_sd(Q),
                     steinR=H.stein_R(m, Q), therr=S.theta_err(Q, m.p),
                     band=S.band_profile(Q).tolist())
            traj.append(r)
            print(f'{it:>8} {r["energy"]:>9.4f} {r["energy"]/flk:>8.2f} {r["whsd"]:>7.3f} '
                  f'{r["steinR"]:>8.4f} {r["therr"]:>8.4f} {dt*it/MAXIT:>7.1f}', flush=True)
        rec["starts"][slab] = dict(start_energy=S.energy(Xs), traj=traj, sec=dt)
        out[name] = rec
        json.dump(out, open(f"exp02_results_{name}.json", "w"), indent=1)
