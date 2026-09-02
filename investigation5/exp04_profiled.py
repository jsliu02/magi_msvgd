"""
Exp 4: is the profiled posterior viable at all, judged without a reference?

Three internal checks, before any reference chain exists. Do the inner profile solves succeed at
every node. Is the importance sampling healthy -- effective sample size and Pareto k-hat, which
diagnose the proposal, not the target. And does the parameter mean it produces differ from the
mode in the same direction the third-order correction says it should, which is a weak but real
consistency test between two independent constructions.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup5 import build, SYSTEMS
from precond_gn import scaled_map
from profiled import ProfiledPosterior

N = {"fn": 512, "hes1": 512, "lorenz": 512, "hiv": 128}
for name in ["fn", "hes1", "lorenz", "hiv"]:
    try:
        m, ds = build(name)
        scaled_map(m, tol=1e-10, max_iter=400)
        x0 = np.asarray(m.map_particle, np.float64)
        pp = ProfiledPosterior(m, n_nodes=N[name], seed=0).build(verbose=False)
        p = m.p
        tm = pp.theta_mean
        sd = np.sqrt(np.maximum(np.diag(pp.theta_cov), 0))
        H = np.asarray(m.hessian(), np.float64); H = 0.5 * (H + H.T)
        dsc = np.sqrt(np.maximum(np.diag(H), np.finfo(float).tiny))
        Hs = H / np.outer(dsc, dsc)
        w_, V_ = np.linalg.eigh(0.5 * (Hs + Hs.T))
        Sig = ((V_ / np.maximum(w_, 1e-12 * w_.max())) @ V_.T) / np.outer(dsc, dsc)
        lap_sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
        tru = np.asarray(ds.hyperparams["theta"], np.float64)
        print(f'--- {name}  (p={p}, dim={x0.shape[0]}, nodes={N[name]}) ---')
        print(f'    {pp}')
        print(f'    {"":>12} ' + " ".join(f'{f"th{j}":>11}' for j in range(min(p, 6))))
        for lbl, v in [("true", tru), ("MAP", x0[:p]), ("profiled mean", tm)]:
            print(f'    {lbl:>12} ' + " ".join(f'{v[j]:>11.5f}' for j in range(min(p, 6))))
        print(f'    {"sd: Laplace":>12} ' + " ".join(f'{lap_sd[j]:>11.5f}' for j in range(min(p, 6))))
        print(f'    {"sd: profiled":>12} ' + " ".join(f'{sd[j]:>11.5f}' for j in range(min(p, 6))))
        print(f'    {"shift/lap sd":>12} ' +
              " ".join(f'{(tm[j]-x0[j])/max(lap_sd[j],1e-300):>11.3f}' for j in range(min(p, 6))))
        print()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'--- {name}: FAILED {type(e).__name__}: {str(e)[:120]}\n')
