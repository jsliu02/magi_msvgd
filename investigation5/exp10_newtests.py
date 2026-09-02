"""Exp 10: validate the rewritten tests.py -- accuracy, cost, and the old call sequence."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time, tracemalloc
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "magi_msvgd"))
import tests as T

print("=== accuracy: max |solution - reference| / sigma at the observation times ===")
print(f'{"system":>14} {"method":>7} {"step":>8} {"substeps":>9} {"f evals":>10} {"sec":>7} '
      f'{"err/sigma":>11}')
for name, ds in T.SYSTEMS.items():
    sg = ds.hyperparams["sigma"]
    ds.reset(); ds.ground_truth(step=1e-5, method="rk4")          # reference
    ref = ds.truth_at(ds.obs_times).copy()
    for method, steps in (("rk4", [1e-2, 1e-3]), ("euler", [1e-4, 1e-6])):
        for st in steps:
            ds.reset()
            t0 = time.time(); ds.ground_truth(step=st, method=method); dt = time.time() - t0
            cur = ds.truth_at(ds.obs_times)
            e = np.nanmax(np.abs(cur - ref) / np.where(np.isfinite(sg), sg, np.nan))
            i = ds._integration
            print(f'{name:>14} {method:>7} {st:>8.0e} {i["nsub"]:>9} {i["evals"]:>10,} '
                  f'{dt:>7.2f} {e:>11.3e}')
    print()

print("=== memory: stored solution ===")
for name, ds in T.SYSTEMS.items():
    ds.reset(); ds.ground_truth(step=1e-3)
    horizon = ds.hyperparams["I"].max()
    old = horizon / 1e-6 * len(ds.hyperparams["x0"]) * 8 / 1e9
    new = ds.solution.nbytes / 1e6
    print(f'{name:>14}: stored {ds.solution.shape} = {new:>8.3f} MB   '
          f'(old scheme at 1e-6 would be {old:>7.2f} GB)')

print("\n=== the original call sequence still works ===")
m = T.FitzHughNagumo
m.reset(); m.ground_truth(step=1e-3)
t, sample = m.sample(seed=0)
data = m.discretize(t, sample)
print(f'  ground_truth/sample/discretize -> data {data.shape}, '
      f'{int(np.isfinite(data[:,1:]).sum())} observations, NaN elsewhere')
print(f'  dataset() one-liner            -> {T.Lorenz.reset().dataset(seed=0).shape}')
print(f'  repr: {T.Hes1!r}')
print(f'  X0 synonym accepted for HIV: x0 = {T.HIV.hyperparams["x0"]}')
print(f'  Hes1 unobserved component: column 3 all NaN = '
      f'{bool(np.all(np.isnan(T.Hes1.reset().dataset(seed=0)[:,3])))}')
