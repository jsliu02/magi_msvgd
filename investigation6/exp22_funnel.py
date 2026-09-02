"""
Exp 22: is Hes1's varying curvature a funnel, and what drives it?

Exp 21 showed the local curvature varies by six to eight orders of magnitude across Hes1's
posterior while HIV's barely moves. Hes1's states are on a log scale -- its vector field reads
P, M, H = exp(X) -- so the ODE residual and its derivatives are exponential in the coordinates, and
a curvature that scales exponentially with a coordinate is the textbook funnel that defeats a fixed
mass matrix.

Tested by regressing the log of the local curvature on the parameters and on the state magnitude
across reference draws. A funnel shows up as a strong, near-linear dependence on the coordinate
that sets the scale.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "magi_msvgd"))
from setup6 import build, SYSTEMS
R = lambda n: os.path.join("..", "investigation5", f"ref5_{n}.npz")
NM = {"fn": ["a", "b", "c"], "hes1": list("abcdefg"),
      "hiv": ["lam", "rho", "delta", "N", "c"], "lorenz": ["beta", "rho", "sigma"]}

for name in ("hes1", "hiv"):
    z = np.load(R(name)); sub = np.asarray(z["sub"], np.float64)
    m, ds = build(name); m.map_solve(verbose=False)
    p, n, D = m.p, m.n, m.D
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sub), 60, replace=False)
    logdet, maxeig, feats = [], [], []
    for i in idx:
        x = sub[i]
        H = np.asarray(m.hessian(x), np.float64); H = 0.5 * (H + H.T)
        w = np.linalg.eigvalsh(H)
        wp = w[w > 0]
        logdet.append(np.sum(np.log(wp)) if len(wp) else np.nan)
        maxeig.append(np.log(max(abs(w).max(), 1e-300)))
        X = x[p:].reshape(n, D)
        feats.append(list(x[:p]) + [X[:, j].max() for j in range(D)])
    logdet, maxeig = np.array(logdet), np.array(maxeig)
    F = np.array(feats)
    names = NM[name] + [f'max X[{j}]' for j in range(D)]
    print(f'--- {name}: spread of local curvature over the posterior ---')
    print(f'    log|det H| range {np.nanmin(logdet):.1f} .. {np.nanmax(logdet):.1f} '
          f'(spread {np.nanmax(logdet)-np.nanmin(logdet):.1f} nats)')
    print(f'    log(max eig)  range {maxeig.min():.2f} .. {maxeig.max():.2f} '
          f'(a factor of {np.exp(maxeig.max()-maxeig.min()):.3g})')
    print(f'    {"driver":>12} {"corr with log(max eig)":>24} {"slope per unit":>16}')
    for k in range(F.shape[1]):
        f = F[:, k]
        if np.std(f) < 1e-12:
            continue
        c = float(np.corrcoef(f, maxeig)[0, 1])
        sl = float(np.polyfit(f, maxeig, 1)[0])
        if abs(c) > 0.3:
            print(f'    {names[k]:>12} {c:>24.3f} {sl:>16.3f}')
    print()
