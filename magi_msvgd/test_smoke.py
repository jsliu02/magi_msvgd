"""
Smoke tests for the MAGI pipeline. Run directly:

    python test_smoke.py                  # CPU, both precisions
    python test_smoke.py --dev gpu        # and on the accelerator
    python test_smoke.py --quick          # skip nuts() and the slower systems

Checks behaviour a caller depends on, not internals: that every public entry point runs on every
system in both precisions, that the outputs satisfy the invariants the docstrings claim, and that
the documented failure modes fail in the documented way. Exits nonzero on any failure.
"""
import argparse, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--x64", action="store_true", help="enable x64 before importing, as scripts do")
A = ap.parse_args()
if A.dev == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
if A.x64:
    jax.config.update("jax_enable_x64", True)

import tests as ode
from magi import MAGI, MAGIPosterior, clear_jax_cache, jax_cache_info

DEV = jax.devices(A.dev)[0]
N_PASS, N_FAIL, FAILS = 0, 0, []


def check(name, cond, detail=""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        N_FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}   {detail}")


def raises(name, fn, exc=Exception):
    try:
        fn()
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:50]}")
        return
    except Exception as e:
        check(name, False, f"wrong exception {type(e).__name__}")
        return
    check(name, False, "no exception raised")


def build(sysname, dtype):
    m_ode = ode.SYSTEMS[sysname]
    m_ode.reset()
    data = m_ode.dataset(seed=0)
    hp = m_ode.hyperparams
    m = MAGI(m_ode.ode, data, hp["theta"], theta_prec=np.zeros(len(hp["theta"])),
             sigmas=hp["sigma"], init_device=DEV)
    m.put(dtype, DEV)
    return m, m_ode


# =============================================================== tests.py, the ODE systems
print(f"\n### tests.py  (x64={jax.config.jax_enable_x64})")
for nm, mdl in ode.SYSTEMS.items():
    mdl.reset()
    T, sol = mdl.ground_truth()
    I = np.asarray(mdl.hyperparams["I"], np.float64)
    check(f"{nm}: output grid is the discretisation set", len(T) == len(I),
          f"|T|={len(T)} |I|={len(I)}")
    check(f"{nm}: ground truth is float64 and finite",
          sol.dtype == np.float64 and np.all(np.isfinite(sol)))
    check(f"{nm}: truth_at accepts a float32 grid",
          mdl.truth_at(np.asarray(I, np.float32).astype(np.float64)).shape == (len(I), mdl.hyperparams["x0"].shape[0]))
    t, y = mdl.sample(seed=0)
    d = mdl.discretize(t, y)
    check(f"{nm}: dataset places every observation", d.shape == (len(I), y.shape[1] + 1))
    nobs = np.isfinite(d[:, 1:]).sum(0)
    want = [len(td) for td in mdl.hyperparams["tau"]]
    check(f"{nm}: observation counts survive discretisation", list(nobs) == want,
          f"{list(nobs)} vs {want}")
    check(f"{nm}: sample is reproducible", np.allclose(mdl.sample(seed=0)[1], y, equal_nan=True))
    check(f"{nm}: a different seed differs", not np.allclose(mdl.sample(seed=1)[1], y, equal_nan=True))
    check(f"{nm}: repr works", "DynamicalSystem" in repr(mdl))
    mdl.reset()
    check(f"{nm}: reset clears the solution", mdl.solution is None)

fn = ode.SYSTEMS["FitzHughNagumo"]
fn.reset(); fn.ground_truth()
raises("truth_at rejects an off-grid time", lambda: fn.truth_at([0.06123]), ValueError)
fn.reset()
raises("truth_at before ground_truth errors", lambda: fn.truth_at([0.0]), RuntimeError)
fn.reset()
T4, s4 = fn.ground_truth(method="rk4", step=1e-3)
fn.reset()
Te, se = fn.ground_truth(method="euler", step=1e-3)
check("rk4 and euler differ at a 1e-3 step", not np.allclose(s4, se, atol=1e-6),
      f"max diff {np.abs(s4 - se).max():.2e}")
fn.reset()
T5, s5 = fn.ground_truth(method="rk4", step=1e-4)
fn.reset(); T6, s6 = fn.ground_truth(method="rk4", step=1e-3)
check("rk4 is converged at 1e-3", np.allclose(s5, s6, atol=1e-8),
      f"max diff vs 1e-4 step {np.abs(s5 - s6).max():.2e}")

# =============================================================== MAGI construction & validation
print("\n### MAGI: argument validation")
fnm = ode.SYSTEMS["FitzHughNagumo"]; fnm.reset(); data = fnm.dataset(seed=0)
hp = fnm.hyperparams
mk = lambda prec: MAGI(fnm.ode, data, hp["theta"], theta_prec=prec, sigmas=hp["sigma"],
                       init_device=DEV)
raises("theta_prec wrong length rejected", lambda: mk(np.ones(2)), ValueError)
raises("theta_prec wrong shape rejected", lambda: mk(np.ones((2, 2))), ValueError)
raises("theta_prec non-PSD rejected", lambda: mk(-np.eye(3)), ValueError)
for label, prec in (("scalar", 1e-3), ("vector", np.ones(3) * 1e-3), ("matrix", np.eye(3) * 1e-3)):
    m = mk(prec)
    check(f"theta_prec as a {label} normalises to (p, p)", m.theta_prec.shape == (3, 3))

# =============================================================== the pipeline, per system/dtype
systems = ["FitzHughNagumo", "Lorenz"] if A.quick else list(ode.SYSTEMS)
for sysname in systems:
    for dt in (jnp.float32, jnp.float64):
        tag = f"{sysname}/{dt.__name__}"
        print(f"\n### {tag} on {DEV}")
        t0 = time.time()
        m, mdl = build(sysname, dt)
        p, nD = m.p, m.n * m.D
        check(f"{tag}: put() sets the dtype", m.mu.dtype == dt)
        check(f"{tag}: float64 GP snapshots survive put()",
              m._C_invs64.dtype == np.float64 and m._K_invs64.dtype == np.float64)

        m.map_solve(verbose=False)
        gn = m._gn_solver()
        check(f"{tag}: analytic Jacobian matches jacfwd", gn.check_jacobian() < 1e-3,
              f"rel {gn.check_jacobian():.2e}")
        check(f"{tag}: mode is finite", bool(np.all(np.isfinite(np.asarray(m.map_particle)))))
        check(f"{tag}: Cholesky ridge is float64-scale",
              max(gn.chol_ridge.values()) < 1e-9, f"max {max(gn.chol_ridge.values()):.1e}")

        lap = m._laplace()
        check(f"{tag}: Laplace cache is reused", m._laplace() is lap)
        H = np.asarray(m.hessian(), np.float64)
        check(f"{tag}: Hessian is symmetric",
              np.allclose(H, H.T, rtol=1e-5, atol=1e-8 * np.abs(H).max()))
        check(f"{tag}: Sigma H is the identity on the kept span",
              np.abs(lap.Sig @ lap.H @ lap.Sig - lap.Sig).max() <
              1e-6 * np.abs(lap.Sig).max(), "Sig H Sig = Sig")
        check(f"{tag}: whitening factors the covariance",
              np.abs(lap.whiten @ lap.whiten.T - lap.Sig).max() < 1e-8 * np.abs(lap.Sig).max())
        m.map_solve(verbose=False)
        check(f"{tag}: map_solve invalidates the Laplace cache", m._lap is None)

        o = m.diagnose(n_starts=2, n_curv=3, verbose=False)
        for k in ("mode_dist", "cond", "cond_M", "n_null", "theta", "theta_sd", "verdict",
                  "ell", "grid_dt", "n_distinct", "fall"):
            check(f"{tag}: diagnose reports {k}", k in o)
        check(f"{tag}: mode located to < 0.1 sd", o["mode_dist"] < 0.1, f'{o["mode_dist"]:.2e}')
        check(f"{tag}: no negative curvature at the mode", o["n_neg"] == 0)
        check(f"{tag}: report renders", "STATUS" in m._diagnosis_report(o, 2, 3.0, 1e4))

        post = m.fit(verbose=False)
        check(f"{tag}: fit returns a MAGIPosterior", isinstance(post, MAGIPosterior))
        check(f"{tag}: posterior mean is finite and the right length",
              post.mean.shape[0] >= p + nD and np.all(np.isfinite(np.asarray(post.mean))))
        check(f"{tag}: theta_cov is symmetric PSD",
              np.all(np.linalg.eigvalsh(0.5 * (np.asarray(post.theta_cov, np.float64) +
                                               np.asarray(post.theta_cov, np.float64).T)) > -1e-12))
        check(f"{tag}: report renders", "STATUS" in post.report())
        check(f"{tag}: posterior is cached on the model", m.posterior is post)
        pp = post.profiled
        check(f"{tag}: stencil came from a plateau or said so", pp.fd_plateau in (True, False))
        check(f"{tag}: fd_pick is on the ladder", pp.fd_pick in pp.fd_ladder)
        if post.reliable:
            check(f"{tag}: gate passed => ESS >= 10%",
                  pp.ess / pp.n_nodes >= 0.10, f"{pp.ess/pp.n_nodes:.1%}")
            w = pp.w
            check(f"{tag}: mixture weights normalise", abs(float(w.sum()) - 1) < 1e-10)
            check(f"{tag}: weighted mean matches the property",
                  np.allclose(pp.theta_mean, w @ pp.TH))
        else:
            check(f"{tag}: gate declined => Laplace reported", post.cov is not None)

        Xs, th, sg = post.sample(k=200, seed=0)
        check(f"{tag}: sample shapes", th.shape == (200, p) and Xs.shape[0] == 200,
              f"{th.shape} {Xs.shape}")
        check(f"{tag}: sample is finite",
              bool(np.all(np.isfinite(np.asarray(th)))) and bool(np.all(np.isfinite(np.asarray(Xs)))))
        check(f"{tag}: sample is reproducible",
              np.allclose(np.asarray(post.sample(k=200, seed=0)[1]), np.asarray(th)))
        Xs0, th0, _ = post.sample(k=200, seed=0, state_noise=False)
        check(f"{tag}: state_noise=False removes state spread",
              float(np.asarray(Xs0).std(0).max()) <= float(np.asarray(Xs).std(0).max()))
        raw = post.sample(k=50, seed=1, unpack=False)
        check(f"{tag}: unpack=False returns a flat array", raw.ndim == 2 and raw.shape[0] == 50)
        smean = np.asarray(th).mean(0)
        sd = np.sqrt(np.maximum(np.diag(np.asarray(post.theta_cov, np.float64)), 1e-300))
        check(f"{tag}: sample mean matches the closed form",
              np.max(np.abs(smean - np.asarray(post.mean)[:p]) / sd) < 0.5,
              f'{np.max(np.abs(smean - np.asarray(post.mean)[:p]) / sd):.3f} sd')

        dA = m.condition_A()
        check(f"{tag}: condition_A returns a finite non-negative number",
              np.isfinite(dA) and dA >= 0, f"{dA:.4f}")
        print(f"  ({time.time() - t0:.1f}s)")

# =============================================================== float32 vs float64 agreement
print("\n### precision agreement (FitzHughNagumo)")
res = {}
for dt in (jnp.float32, jnp.float64):
    m, _ = build("FitzHughNagumo", dt)
    post = m.fit(verbose=False)
    res[dt.__name__] = (np.asarray(post.mean, np.float64)[:m.p],
                        np.sqrt(np.diag(np.asarray(post.theta_cov, np.float64))))
th32, sd32 = res["float32"]; th64, sd64 = res["float64"]
check("float32 agrees with float64 on theta", np.max(np.abs(th32 - th64) / sd64) < 0.1,
      f"{np.max(np.abs(th32 - th64) / sd64):.4f} sd")
check("float32 agrees with float64 on the spread", np.max(np.abs(sd32 / sd64 - 1)) < 0.1,
      f"{np.max(np.abs(sd32 / sd64 - 1)):.4f}")

# =============================================================== cache helpers
print("\n### compilation cache helpers")
path, n, b = jax_cache_info()
check("jax_cache_info returns a path and counts", isinstance(path, str) and n >= 0 and b >= 0,
      f"{n} entries, {b/1e6:.1f} MB")
_, n2, _ = clear_jax_cache(dry_run=True, verbose=False)
check("clear_jax_cache(dry_run) removes nothing", jax_cache_info()[1] == n, f"{n2} would go")
raises("clear_jax_cache refuses a non-cache directory",
       lambda: clear_jax_cache(path="/tmp", verbose=False), ValueError)
raises("clear_jax_cache refuses $HOME",
       lambda: clear_jax_cache(path=os.path.expanduser("~"), force=True, verbose=False), ValueError)

# =============================================================== nuts
if not A.quick:
    print("\n### nuts (short run)")
    try:
        m, _ = build("FitzHughNagumo", jnp.float64)
        out = m.nuts(warmup_steps=25, sampling_steps=25, n_chains=2, verbose=False)
        arr = out[0] if isinstance(out, tuple) else out
        check("nuts runs and returns finite draws", bool(np.all(np.isfinite(np.asarray(arr)))))
    except ImportError as e:
        print(f"  SKIP  nuts (blackjax unavailable: {e})")
    except Exception as e:
        check("nuts runs", False, f"{type(e).__name__}: {str(e)[:70]}")

print(f"\n{'=' * 70}\n{N_PASS} passed, {N_FAIL} failed"
      + (f"\nfailures: {FAILS}" if FAILS else ""))
sys.exit(1 if N_FAIL else 0)
