"""
exp06: the fixed-point test with the optimizer removed.

exp01 runs an optimizer, so anything it finds can be argued away as a step-size artefact:
Prodigy adapts its own learning rate and could be amplifying a drift that a smaller step would
never express. This experiment removes that argument.

In the mean-field limit SVGD's velocity field vanishes identically when the ensemble law IS the
target -- phi_p(x) = E_{y~p}[k(y,x) s(y) + grad_y k(y,x)] = 0 by integration by parts. So at exact
draws from the target, phi is pure finite-K sampling noise plus an O(1/K) systematic bias, and
that bias is the entire mechanism of variance collapse. It can be measured directly:

  * take R independent K-particle subsamples of the REFERENCE (exact draws, by construction);
  * evaluate the SVGD velocity phi at each;
  * project into the reference covariance's eigenbasis and compute, band by band from the
    softest direction to the stiffest,

        rate_b = 2 * Cov(e.y, e.phi_y) / Var_ref(e.y)   averaged over the band,

    which is d/dt log(variance along e) under the flow, in units of 1/step;
  * average over the R replicates. The mean is the systematic part, the spread over replicates
    divided by sqrt(R) is the noise on it.

A correct sampler has rate_b = 0 in every band, within that error bar. A negative rate in the
stiff bands is anisotropic collapse, measured at the target itself with no optimizer, no step
size and no convergence criterion involved.

The same quantity is also reported for the ensemble MEAN, drift_b = mean(e.phi_y), which says
whether the flow also translates the ensemble off the target.
"""
import numpy as np, jax, jax.numpy as jnp, optax, sys, os, json, time
jax.config.update("jax_enable_x64", True)
import harness7 as H
import msvgd7 as M7

SYS = sys.argv[1:] or list(H.USABLE)
KS = [int(x) for x in os.environ.get("KS", "100,400,1600").split(",")]
R = int(os.environ.get("R", 8))
NB = 5
KERNELS = os.environ.get("KERNELS", "standard,reweighted,matrix").split(",")
out = {}

for name in SYS:
    m, ds = H.build(name)
    S = H.Scorer(name)
    d = S.mean.shape[0]
    w, V = np.linalg.eigh(0.5 * (S.cov + S.cov.T))
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-300), V[:, o]
    bands = np.array_split(np.arange(d), NB)
    print(f"\n===== {name}  dim={d}  R={R} replicates of exact reference draws =====", flush=True)
    print(f'{"kernel":>14} {"K":>6}   d/dt log var, by band (soft -> stiff), +/- 1 se over '
          f'replicates', flush=True)
    rec = {}

    for kern in KERNELS:
        kfn = M7.KERNELS[kern]

        @jax.jit
        def field(P):
            g = m.gradient(P, m.data)
            lp = jax.vmap(lambda x: m.logdensity(x, m.data))(P)
            return -kfn(P, -g, lp, -1.0)      # driver returns -phi; phi is the velocity

        for K in KS:
            rng = np.random.default_rng(3)
            rates, drifts, sizes = [], [], []
            for r in range(R):
                idx = rng.choice(len(S.sub), K, replace=False)
                X = S.sub[idx]
                phi = np.asarray(field(jnp.asarray(X, m.mu.dtype)), np.float64)
                Z = (X - S.mean) @ V          # (K, d) in reference eigen-coordinates
                Q = phi @ V                   # velocity in the same coordinates
                cov_zq = ((Z - Z.mean(0)) * (Q - Q.mean(0))).mean(0)
                rates.append(np.array([(2 * cov_zq[b] / w[b]).mean() for b in bands]))
                drifts.append(np.array([(Q.mean(0)[b] / np.sqrt(w[b])).mean() for b in bands]))
                sizes.append(float(np.sqrt((phi ** 2).sum(1).mean())))
            Rt = np.array(rates); Dr = np.array(drifts)
            mu_r, se_r = Rt.mean(0), Rt.std(0, ddof=1) / np.sqrt(R)
            mu_d, se_d = Dr.mean(0), Dr.std(0, ddof=1) / np.sqrt(R)
            print(f'{kern:>14} {K:>6}   ' +
                  "  ".join(f'{a:+9.2e}+-{b:7.1e}' for a, b in zip(mu_r, se_r)), flush=True)
            print(f'{"":>14} {"drift":>6}   ' +
                  "  ".join(f'{a:+9.2e}+-{b:7.1e}' for a, b in zip(mu_d, se_d)), flush=True)
            rec[f"{kern}_{K}"] = dict(rate=mu_r.tolist(), rate_se=se_r.tolist(),
                                      drift=mu_d.tolist(), drift_se=se_d.tolist(),
                                      phi_norm=float(np.mean(sizes)))
    out[name] = rec
    json.dump(out, open("exp06_results.json", "w"), indent=1)
