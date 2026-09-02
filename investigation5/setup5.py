"""
investigation5 setup: every test system from tests.py, driven through MAGI identically.

The point of this file is to stop investigating one ODE. tests.py supplies four systems spanning
regimes that FitzHugh-Nagumo does not: Hes1 has seven parameters, a rational nonlinearity and a
state that is NEVER observed; HIV has time-dependent forcing and states spanning 30 to 1e5, so its
Hessian conditioning is a different problem entirely; Lorenz is chaotic.

Every workaround this file used to carry is gone: tests.py now normalises x0/X0, locates
observation times by tolerance rather than float equality, integrates with RK4, and retains the
solution only on the output grid. The single default step below is justified by exp10 -- RK4 at
1e-3 holds the integration error under 1e-8 observation standard deviations on all four systems,
against 1e-3 for the forward Euler scheme it replaced.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "magi_msvgd"))
from magi import MAGI
import tests as T

STEP = 1e-3

SYSTEMS = {
    "fn":     dict(sys=T.FitzHughNagumo, guess=[1, 1, 1]),
    "hes1":   dict(sys=T.Hes1,           guess=[0.02, 0.3, 0.03, 0.03, 0.5, 20, 0.3]),
    "hiv":    dict(sys=T.HIV,            guess=[30, 0.1, 0.5, 1000, 3]),
    "lorenz": dict(sys=T.Lorenz,         guess=[3.0, 25.0, 10.0]),
}


def build(name, seed=0, dtype=jnp.float64, device=None, step=STEP):
    # put(float32) turns x64 off process-wide, so a later build would generate its data in single
    # precision. Data generation is always float64 regardless of the model's final dtype.
    jax.config.update("jax_enable_x64", True)
    spec = SYSTEMS[name]
    ds = spec["sys"]
    data = ds.reset().dataset(seed=seed, step=step)
    m = MAGI(ds.ode, data, spec["guess"], np.zeros(len(spec["guess"])),
             sigmas=np.asarray(ds.hyperparams["sigma"], np.float64),
             init_device=device or jax.devices()[0])
    m.put(dtype=dtype, device=device or jax.devices()[0])
    return m, ds


if __name__ == "__main__":
    print(f'{"system":>8} {"p":>3} {"n":>5} {"D":>3} {"dim":>6} {"obs":>6} {"unobs D":>8} '
          f'{"theta true":>34}')
    for name in SYSTEMS:
        m, ds = build(name)
        th = np.asarray(ds.hyperparams["theta"], np.float64)
        print(f'{name:>8} {m.p:>3} {m.n:>5} {m.D:>3} {m.p + m.n * m.D:>6} {int(m.N):>6} '
              f'{int((np.asarray(m.Ns) <= 2).sum()):>8} {str(np.round(th, 4))[:34]:>34}')
