"""Exp 12: the profiled posterior on Lorenz -- well posed, mostly identified, and chaotic."""
import numpy as np, jax, jax.numpy as jnp, sys, os, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "investigation5"))
from setup6 import build
from profiled import ProfiledPosterior
from profiled2 import ProfiledPosterior2

NM = ["beta", "rho", "sigma"]
z = np.load("../investigation5/ref5_lorenz.npz")
rm = z["mean"]; rs = np.sqrt(np.maximum(np.diag(z["cov"]), 0))
hm, hc = z["half_mean"], z["half_cov"]
m, ds = build("lorenz"); m.map_solve(verbose=False, tol=1e-9, max_iter=300)
p = m.p; x0 = np.asarray(m.map_particle, np.float64)
post = m.fit(verbose=False, tol=1e-9, max_iter=300)
t0 = time.time(); v1 = ProfiledPosterior(m, n_nodes=512, seed=0).adapt(rounds=6, verbose=False); t1 = time.time()-t0
t0 = time.time(); v2 = ProfiledPosterior2(m, n_nodes=512, seed=0).build(verbose=False); t2 = time.time()-t0
err = lambda v: np.abs((np.asarray(v)[:p] - rm[:p]) / np.maximum(rs[:p], 1e-300))
H = np.asarray(m.hessian(), np.float64); H = 0.5*(H+H.T)
d = np.sqrt(np.abs(np.diag(H))); w, V = np.linalg.eigh(H/np.outer(d,d)); k = w > 1e-10*abs(w).max()
Sig = ((V[:,k]/w[k]) @ V[:,k].T)/np.outer(d,d)
lap_sd = np.sqrt(np.maximum(np.diag(Sig)[:p], 0))
sde = lambda sd: np.abs(np.asarray(sd)/np.maximum(rs[:p],1e-300) - 1)
print(f'reference: Rhat {float(z["rhat"].max()):.4f}, div {int(z["div"])}/{int(z["ndraw"])} '
      f'({100*int(z["div"])/int(z["ndraw"]):.1f}%)')
print(f'\n{"method":>26} {"max|err|":>10} {"max|sd err|":>12} {"ESS":>7} {"khat":>7} {"sec":>7}')
print(f'{"MAP":>26} {err(x0).max():>10.4f} {sde(lap_sd).max():>12.2%} {"-":>7} {"-":>7} {"-":>7}')
print(f'{"third order":>26} {err(post.mu3).max():>10.4f} {sde(lap_sd).max():>12.2%} '
      f'{"-":>7} {"-":>7} {sum(post.timings.values()):>7.1f}')
print(f'{"profiled v1":>26} {err(v1.theta_mean).max():>10.4f} '
      f'{sde(np.sqrt(np.maximum(np.diag(v1.theta_cov),0))).max():>12.2%} '
      f'{v1.ess/v1.n_nodes:>7.1%} {v1.khat:>7.2f} {t1:>7.1f}')
print(f'{"profiled v2":>26} {err(v2.theta_mean).max():>10.4f} '
      f'{sde(np.sqrt(np.maximum(np.diag(v2.theta_cov),0))).max():>12.2%} '
      f'{v2.ess/v2.n_nodes:>7.1%} {v2.khat:>7.2f} {t2:>7.1f}')
print(f'{"reference half-vs-half":>26} '
      f'{np.abs((hm[0][:p]-hm[1][:p])/np.maximum(rs[:p],1e-300)).max():>10.4f} '
      f'{np.abs(np.sqrt(np.diag(hc[0])[:p]/np.maximum(np.diag(hc[1])[:p],1e-300))-1).max():>12.2%}')
print(f'\n{"":>12} ' + " ".join(f'{n:>10}' for n in NM))
for lbl, v in [("reference", rm[:p]), ("MAP", x0[:p]), ("third order", post.mu3[:p]),
               ("profiled v2", v2.theta_mean), ("true", np.asarray(ds.hyperparams["theta"]))]:
    print(f'{lbl:>12} ' + " ".join(f'{np.asarray(v)[j]:>10.4f}' for j in range(p)))
