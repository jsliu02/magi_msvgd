"""
exp05: was investigation 4's mSVGD verdict an artefact of the GP bug, of the integrator change,
or of neither?

Two things changed under it: the GP hyperparameter fit (investigation 6 sec. 8) and the ODE
integrator (forward Euler -> RK4). This runs the fixed-point test on all four combinations on
FitzHugh-Nagumo, the one system investigation 4 measured.

Only the rk4 + fixed-GP cell has a reference, so the score here is deliberately REFERENCE-FREE
and identical in all four cells:

  * Stein R, whose target value of 1 is fixed by theory rather than by a reference;
  * the band profile taken against the LAPLACE covariance H^-1 at the MAP rather than against a
    reference covariance -- the same anisotropy readout, in a metric each cell supplies itself;
  * how far the ensemble moved from its start, in that same Laplace metric.

The start is `fit()` draws in every cell. In the rk4 + fixed-GP cell that start is known good
(exp00); in the others it is only "the best the pipeline can do", which is the honest analogue of
what investigation 4 had available.
"""
import numpy as np, jax, jax.numpy as jnp, optax, time, sys, os, json
jax.config.update("jax_enable_x64", True)
import harness7 as H
import msvgd7 as M7
import oldgp
import tests as T
from magi import MAGI

KERNELS = os.environ.get("KERNELS", "standard,reweighted").split(",")
MAXIT = int(os.environ.get("MAXIT", 1000))
K = int(os.environ.get("K", 400))
STEP = 1e-3


def build_fn(method, old_gp):
    jax.config.update("jax_enable_x64", True)
    if old_gp:
        oldgp.install()
    else:
        oldgp.restore()
    ds = T.FitzHughNagumo.reset()
    data = ds.dataset(seed=0, step=STEP, method=method)
    m = MAGI(ds.ode, data, [1, 1, 1], np.zeros(3),
             sigmas=np.asarray(ds.hyperparams["sigma"], np.float64))
    m.put(dtype=jnp.float64)
    oldgp.restore()
    return m, ds


def lap_band(Sig, mean, X, nbands=5):
    w, V = np.linalg.eigh(0.5 * (Sig + Sig.T))
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-300), V[:, o]
    Z = (np.asarray(X, np.float64) - mean) @ V
    r = Z.var(0) / w
    return np.array([r[b].mean() for b in np.array_split(np.arange(len(w)), nbands)])


out = {}
print(f'{"cell / variant":>34} {"l/dt":>14} {"SteinR":>8} {"moved":>8}   '
      f'band profile vs Laplace (soft -> stiff)', flush=True)
for method in ("rk4", "euler"):
    for old_gp in (False, True):
        tag = f'{method}/{"OLD gp" if old_gp else "fixed gp"}'
        try:
            m, ds = build_fn(method, old_gp)
            dt_grid = float(np.median(np.diff(np.asarray(m.I).ravel())))
            ell = np.asarray(m.phis, np.float64)[:, 1] / dt_grid
            post = m.fit(verbose=False)
            X0 = np.asarray(post.sample(K, unpack=False), np.float64)
            x_map, L = M7.laplace_metric(m)
            Sig = L @ L.T
            mu0 = X0.mean(0)
            Wi = np.linalg.inv(L)
            wh = lambda X: (np.asarray(X, np.float64) - x_map) @ Wi.T
            rec = {"ell_over_dt": ell.tolist(), "reliable": bool(post.reliable),
                   "start": dict(steinR=H.stein_R(m, X0),
                                 band=lap_band(Sig, x_map, X0).tolist()), "runs": {}}
            print(f'{tag + " START":>34} {str(np.round(ell, 2)):>14} '
                  f'{rec["start"]["steinR"]:>8.4f} {0.0:>8.4f}   '
                  + " ".join(f'{v:>6.3f}' for v in rec["start"]["band"]), flush=True)
            for kern in KERNELS:
                t0 = time.time()
                P, Rs, _ = M7.run_svgd(m, X0, MAXIT, kernel=kern,
                                       optimizer=optax.contrib.prodigy, optimizer_kwargs={})
                r = dict(steinR=H.stein_R(m, P), band=lap_band(Sig, x_map, P).tolist(),
                         moved=float(np.sqrt(((wh(P).mean(0) - wh(X0).mean(0)) ** 2).sum())),
                         sec=time.time() - t0)
                rec["runs"][kern] = r
                print(f'{f"  {kern}, {MAXIT} it":>34} {"":>14} {r["steinR"]:>8.4f} '
                      f'{r["moved"]:>8.4f}   '
                      + " ".join(f'{v:>6.3f}' for v in r["band"]), flush=True)
            out[tag] = rec
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[tag] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        json.dump(out, open("exp05_results.json", "w"), indent=1)
