"""
Exp 7: re-run investigation 2's whitened samplers with a properly converged MAP.

Investigation 2 concluded that sigma=0.5 is a regime where the Laplace metric is unusable and
every sampler built on it fails. Section 3 showed the metric there was corrupted by an
unconverged MAP -- a spurious eigenvalue of 0.0078 instead of 1.07, which whitening turns into a
100x stretch of one direction. With a Gauss-Newton MAP that artifact is gone, so the failure
needs re-testing. Minimal standalone implementations, since the versions in msvgd.py were removed.
"""
import numpy as np, jax, jax.numpy as jnp, os, time
from functools import partial
jax.config.update("jax_enable_x64", True)
import harness as H
from magi import MAGI

def build(stride, sigma):
    d = np.loadtxt(os.path.join(H.REPO, "magi_msvgd", "y.csv"), delimiter=",")[::stride]
    g = np.arange(0, 20.001, 0.125); full = np.full((g.shape[0], 3), np.nan); full[:, 0] = g
    full[np.isin(full[:, 0], d[:, 0])] = d
    mm = MAGI(H.fn_ode, full, [1, 1, 1], theta_conf=[0, 0, 0], sigmas=[sigma, sigma])
    mm.put(dtype=jnp.float64, device=jax.devices()[0]); return mm

class Ref:
    def __init__(s, pos):
        s.pos = pos; s.mean = pos.mean(0); s.sd = pos.std(0)
        ev, V = np.linalg.eigh(np.cov(pos, rowvar=False))
        s.evals = np.maximum(ev, 1e-14); s.evecs = V
        s.theta_w = np.quantile(pos[:, :3], .975, 0) - np.quantile(pos[:, :3], .025, 0)
        s.ref = pos[np.random.default_rng(0).choice(len(pos), 2000, False)]
    def whiten(s, X): return (np.asarray(X, np.float64) - s.mean) @ s.evecs / np.sqrt(s.evals)
    def score(s, X, tag):
        rng = np.random.default_rng(1); P = np.asarray(X, np.float64)
        lo, hi = np.quantile(P[:, :3], .025, 0), np.quantile(P[:, :3], .975, 0)
        w = 100 * (hi - lo) / s.theta_w
        pr = ((P - s.mean) @ s.evecs).var(0) / s.evals
        return dict(tag=tag, energy=H._energy_distance(s.whiten(P), s.whiten(s.ref), rng),
                    dev=float(np.abs(w - 100).mean()), sdrat=float(np.median(P.std(0) / s.sd)),
                    varwtd=float(np.sum(pr * s.evals) / np.sum(s.evals)),
                    bias=float(np.sqrt((s.whiten(P.mean(0)[None, :]) ** 2).mean())))

def run(mm, xmap, Hs, R_, tag_prefix):
    ev, V = np.linalg.eigh(Hs); evc = np.maximum(ev, 1e-8 * ev.max())
    Lw = jnp.asarray((V / np.sqrt(evc)) @ V.T)
    xm = jnp.asarray(xmap)
    lp = jax.jit(jax.vmap(lambda y: mm.logdensity(xm + y @ Lw, mm.data)))
    gr = jax.jit(lambda y: mm.gradient(xm + y @ Lw, mm.data) @ Lw)
    out = []

    @partial(jax.jit, static_argnums=(2,))
    def mala(y, key, n, eps):
        def b(c, _):
            y, l, g, k = c
            k, kn, ku = jax.random.split(k, 3)
            z = jax.random.normal(kn, y.shape, y.dtype); st = eps * g + jnp.sqrt(2 * eps) * z
            l2 = lp(y + st); g2 = gr(y + st)
            lr = (l2 - l) + 0.5 * jnp.sum(z ** 2, 1) - jnp.sum((st + eps * g2) ** 2, 1) / (4 * eps)
            a = jnp.log(jax.random.uniform(ku, (y.shape[0],), y.dtype)) < lr
            return (jnp.where(a[:, None], y + st, y), jnp.where(a, l2, l),
                    jnp.where(a[:, None], g2, g), k), jnp.mean(a)
        (y, _, _, _), acc = jax.lax.scan(b, (y, lp(y), gr(y), key), None, length=n)
        return y, jnp.mean(acc)

    @partial(jax.jit, static_argnums=(2,))
    def pcn(y, key, n, rho):
        def b(c, _):
            y, l, k = c
            k, kx, ku = jax.random.split(k, 3)
            yy = jnp.sqrt(1 - rho ** 2) * y + rho * jax.random.normal(kx, y.shape, y.dtype)
            l2 = lp(yy)
            dr = (l2 + 0.5 * jnp.sum(yy ** 2, 1)) - (l + 0.5 * jnp.sum(y ** 2, 1))
            a = jnp.log(jax.random.uniform(ku, (y.shape[0],), y.dtype)) < dr
            return (jnp.where(a[:, None], yy, y), jnp.where(a, l2, l), k), jnp.mean(a)
        (y, _, _), acc = jax.lax.scan(b, (y, lp(y), key), None, length=n)
        return y, jnp.mean(acc)

    y0 = jax.random.normal(jax.random.key(0), (400, H.DIM), dtype=jnp.float64)
    for nm, fn, prm, nit in [("MALA", mala, 3e-2, 2000), ("pCN", pcn, 0.1, 6000)]:
        t0 = time.time()
        y, acc = fn(y0, jax.random.key(1), nit, prm)
        if not bool(jnp.all(jnp.isfinite(y))):
            print(f"  {tag_prefix} {nm}: DIVERGED"); continue
        r = R_.score(xm + y @ Lw.T, f"{tag_prefix} {nm}")
        print(f'  {r["tag"]:>26}  energy={r["energy"]:7.4f}  varwtd={r["varwtd"]:6.3f}  '
              f'dev={r["dev"]:5.1f}  sdrat={r["sdrat"]:5.3f}  bias={r["bias"]:.3f}  '
              f'acc={float(acc):.2f}  {time.time()-t0:.0f}s')
        out.append(r)
    return out

G = H.Gold()
print("re-run with a Gauss-Newton MAP and the Hessian evaluated there:")
for name, stride, sig, ref in [("baseline", 1, 0.2, Ref(G.pos)),
                               ("noisy s=0.5", 1, 0.5,
                                Ref(np.load("../investigation2/exp10_ref_noisy.npz")["pos"]))]:
    z = np.load(f"map_{name.split()[0]}.npz")
    mm = build(stride, sig)
    rng = np.random.default_rng(0)
    fl = H._energy_distance(ref.whiten(ref.pos[rng.choice(len(ref.pos), 400, False)]),
                            ref.whiten(ref.ref), rng)
    print(f"\n{name}  (sampling floor energy = {fl:.4f})")
    run(mm, z["x_map"], z["H"], ref, name)
