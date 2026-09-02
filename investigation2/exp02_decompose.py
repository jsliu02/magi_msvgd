"""
Exp 2: decompose the error of a Gaussian approximation, to find what is actually worth fixing.

Exp 1 found the Laplace approximation N(x_MAP, H^-1) beats every SVGD variant on energy
distance (1.45 vs 3.71). Before trying to improve SVGD, establish where the remaining error
lives: the mean, the covariance, or genuine non-Gaussianity. Each row swaps in one true
quantity, so the drop from row to row attributes the error.
"""
import numpy as np, jax, jax.numpy as jnp
import harness as H

def main():
    z = np.load("laplace_cache.npz"); x_map, ev, V = z["x_map"], z["evals"], z["evecs"]
    G = H.Gold()
    evc = np.maximum(ev, 1e-8 * ev.max())
    L_lap = (V / np.sqrt(evc)) @ V.T                      # chol-free sqrt of H^-1
    Cn = np.linalg.cholesky(G.cov + 1e-12 * np.eye(H.DIM))
    rng = np.random.default_rng(0)
    jax.config.update("jax_enable_x64", True)
    m = H.build_magi(dtype=jnp.float64)

    rows = [
        ("N(MAP,    H^-1)   [Laplace]", x_map,   L_lap),
        ("N(NUTSmu, H^-1)   [mean fixed]", G.mean, L_lap),
        ("N(MAP,    NUTScov)[cov fixed]",  x_map,  Cn),
        ("N(NUTSmu, NUTScov)[best Gauss]", G.mean, Cn),
    ]
    out = []
    for tag, mu, A in rows:
        s = mu[None, :] + rng.standard_normal((800, H.DIM)) @ A.T
        r = H.evaluate(jnp.asarray(s), m, tag=tag); out.append(r); H.show(r)
    r = H.gold_row(); out.append(r); H.show(r)

    # how big is the MAP->mean displacement, and where does it point?
    d = G.whiten((x_map - G.mean + G.mean)[None, :])[0]
    print(f'\n  MAP displacement from NUTS mean, in NUTS-whitened units: '
          f'norm={np.linalg.norm(d):.3f} over sqrt(325)={np.sqrt(325):.1f} '
          f'(rms per-dim {np.sqrt((d**2).mean()):.3f})')
    ordr = np.argsort(G.evals)[::-1]
    q = np.array_split(ordr, 5)
    print(f'  displacement rms by stiffness bin (soft->stiff): '
          f'{np.round([np.sqrt((d[b]**2).mean()) for b in q], 3)}')
    print(f'  theta-block displacement: {np.round(x_map[:3]-G.mean[:3], 4)} '
          f'(NUTS sd {np.round(G.sd[:3],4)})')
    H.save(out, "exp02_decompose_results")

if __name__ == "__main__":
    print(H.HDR); print("-" * len(H.HDR)); main()
