"""
Exp 11: how much preconditioner structure does the recipe actually need?

The whitened-IMQ recipe needs a metric. The full dense Hessian costs O(d) gradient evaluations,
O(d^2) memory and O(d^3) to factor -- fine at d=325, prohibitive later. So ask what weaker
structure still works:

  full H        the exact Laplace metric (reference)
  diag(H)       cheapest possible; investigation.md's diagonal empirical Fisher failed, but the
                diagonal of the TRUE Hessian is a different and better-motivated object
  GP-prior      beta_inv * C_inv, the MAGI GP prior precision. Analytic, already stored on the
                solver, needs NO Hessian at all -- and the stiff directions are the
                high-frequency GP modes, so the prior may already contain the anisotropy
  blocktime     2x2 blocks per time point plus the theta block: exploits the ODE's local
                coupling but ignores the GP's dense long-range coupling
  identity      x-space control

If GP-prior works, the recipe is Hessian-free and scales.
"""
import numpy as np, jax, jax.numpy as jnp, scipy.linalg as sla
import harness as H
from msvgd import MSVGD


def imq_step(s, y, lr):
    L2sq, h = s.pairwise_distance(y, -1)
    Kx = (1.0 + L2sq / h) ** -0.5
    Kg = (1.0 + L2sq / h) ** -1.5
    dx = (Kg.sum(axis=1, keepdims=True) * y - Kg @ y) * (1.0 / h)
    return y + lr * ((Kx @ s.gradient(y, None if s.data is None else s.data) + dx) / y.shape[0])


def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    evc = np.maximum(ev, 1e-8 * ev.max())
    Hfull = (V * ev) @ V.T
    m = H.build_magi()
    n, D, P = m.n, m.D, m.p

    # GP prior precision: beta_inv * C_inv per component, laid out to match particle ordering
    Cinv = np.asarray(m.C_invs, np.float64)                  # (D, n, n)
    Pgp = np.zeros((H.DIM, H.DIM))
    Pgp[:P, :P] = np.eye(P) * np.mean(np.diag(Hfull)[:P])
    for j in range(D):
        idx = P + np.arange(n) * D + j
        Pgp[np.ix_(idx, idx)] = float(m.beta_inv) * Cinv[j]
    # scale so its overall magnitude matches H (only anisotropy is being tested)
    Pgp *= np.trace(Hfull) / np.trace(Pgp)

    Pblk = np.zeros((H.DIM, H.DIM))
    Pblk[:P, :P] = Hfull[:P, :P]
    for i in range(n):
        idx = P + i * D + np.arange(D)
        Pblk[np.ix_(idx, idx)] = Hfull[np.ix_(idx, idx)]

    def sqrt_inv(A, tag):
        w, U = np.linalg.eigh(0.5 * (A + A.T))
        neg = int((w <= 0).sum())
        w = np.maximum(w, 1e-8 * w.max())
        print(f"    {tag:>10}: cond={w.max()/w.min():.3e}  n_nonpos={neg}")
        return (U / np.sqrt(w)) @ U.T

    print("preconditioners:")
    mats = {"full H": sqrt_inv(Hfull, "full H"),
            "diag(H)": sqrt_inv(np.diag(np.diag(Hfull)), "diag(H)"),
            "GP-prior": sqrt_inv(Pgp, "GP-prior"),
            "blocktime": sqrt_inv(Pblk, "blocktime"),
            "identity": np.eye(H.DIM)}

    out = []
    print(); print(H.HDR); print("-" * len(H.HDR))
    xm = jnp.asarray(x_map, jnp.float32)
    for name, A in mats.items():
        Lw = jnp.asarray(A, jnp.float32)
        s = MSVGD(lambda y, db: m.logdensity(xm + Lw @ y, db), data=m.data)
        y = jax.random.normal(jax.random.key(0), (800, H.DIM), dtype=jnp.float32)
        lr = 1e-2 if name != "identity" else 1e-4
        best, bestR = None, None
        for it in range(1, 2501):
            L2sq, h = s.pairwise_distance(y, -1)
            Kx = (1.0 + L2sq / h) ** -0.5; Kg = (1.0 + L2sq / h) ** -1.5
            dx = (Kg.sum(axis=1, keepdims=True) * y - Kg @ y) * (1.0 / h)
            y = y + lr * ((Kx @ s.gradient(y, m.data) + dx) / y.shape[0])
            if not bool(jnp.all(jnp.isfinite(y))):
                print(f'{name:>26}   DIVERGED at it={it}'); best = None; break
            if it % 100 == 0:
                X = xm + y @ Lw.T
                sc = float(-jnp.sum((X - X.mean(0)) * m.gradient(X, m.data)) / X.size)
                best = X
                if sc <= 1.0:
                    break
        if best is not None:
            r = H.evaluate(best, m, tag=f"whitened IMQ / {name}")
            out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)
    H.save(out, "exp11_precond_results")


if __name__ == "__main__":
    main()
