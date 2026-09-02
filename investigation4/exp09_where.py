"""
Exp 9: where does the residual covariance error live, is it real, and does it matter?

Three questions the aggregate Forstner number cannot answer.

REAL?  Forstner weights all 325 directions equally and the floor was calibrated with iid draws,
       but gold is MCMC: slow directions have far lower effective sample size than fast ones, so
       the floor is direction dependent. The honest test is consistency -- score the Laplace
       covariance against each half of gold separately. A real error is reproduced by both halves;
       noise is not. Reported as the correlation of the per-direction log-variance ratio between
       halves, against the same statistic for the Laplace covariance.

WHERE? The spectrum of Sigma_g^(-1/2) Sigma_L Sigma_g^(-1/2), sorted, against posterior variance
       share. An affine-invariant metric treats a stiff direction holding 0.001% of the variance
       the same as the softest one holding 20%.

MATTER? The quantities a MAGI user actually reads: the theta marginals and the pointwise
       trajectory bands.
"""
import numpy as np, jax, jax.numpy as jnp, sys, os
jax.config.update("jax_enable_x64", True)
import harness as H
from setup4 import cache
from pipeline import metrics, cov_of

G = H.Gold(); d = H.DIM
m, x_map, Hs, Sig, L = cache("baseline")
gold = np.asarray(G.pos, np.float64)
ch = gold.reshape(8, -1, d)
A, B = ch[:4].reshape(-1, d), ch[4:].reshape(-1, d)
b = dict(np.load("build_baseline.npz"))
Sc = cov_of(b, Sig, 40)
Sg = np.cov(gold, rowvar=False)

# ------------------------------------------------------------------- REAL? half-vs-half consistency
def logratio(S, ref):
    """per-direction log(variance ratio) in the reference's own eigenbasis."""
    eg, Vg = np.linalg.eigh(ref)
    return np.log(np.maximum(np.einsum('ij,jk,ki->i', Vg.T, S, Vg), 1e-14) / np.maximum(eg, 1e-14)), eg
SA, SB = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
rA, _ = logratio(Sig, SA); rB, _ = logratio(Sig, SB)
nA, _ = logratio(SB, SA)
print(f'consistency of the per-direction log-variance error across gold halves')
print(f'  corr(Laplace-vs-halfA, Laplace-vs-halfB) = {np.corrcoef(rA, rB)[0,1]:>6.3f}   '
      f'(1 = fully real, 0 = pure noise)')
print(f'  same statistic for halfB-vs-halfA (pure noise control) = '
      f'{np.corrcoef(nA, logratio(SA, SB)[0])[0,1]:>6.3f}')
print(f'  rms log-ratio: Laplace {np.sqrt((rA**2).mean()):.3f}, gold-half {np.sqrt((nA**2).mean()):.3f}')

# ------------------------------------------------------------------- WHERE? by variance share
lr, eg = logratio(Sig, Sg); lrc, _ = logratio(Sc, Sg)
share = eg / eg.sum()
o = np.argsort(-share)
print(f'\n{"variance-share bin":>22} {"share":>8} {"rms log-ratio Laplace":>22} {"+profile":>10}')
for lbl, sl in [("top 5 dirs", o[:5]), ("next 20", o[5:25]), ("next 100", o[25:125]),
                ("remaining 200", o[125:])]:
    print(f'{lbl:>22} {share[sl].sum():>7.1%} {np.sqrt((lr[sl]**2).mean()):>22.3f} '
          f'{np.sqrt((lrc[sl]**2).mean()):>10.3f}')

# ------------------------------------------------------------------- MATTER? user-facing quantities
print(f'\n{"quantity":>34} {"Laplace":>10} {"+3rd+prof":>11} {"gold halfA":>11}')
mu3 = b["mu3"]; mu0 = b["mu0"]
th_g, th_A = gold[:, :3], A[:, :3]
for i, nm in enumerate(["theta_a", "theta_b", "theta_c"]):
    sL, sC, sA = np.sqrt(Sig[i, i]), np.sqrt(Sc[i, i]), th_A[:, i].std()
    print(f'{nm+" sd / gold sd":>34} {sL/th_g[:,i].std():>10.4f} {sC/th_g[:,i].std():>11.4f} '
          f'{sA/th_g[:,i].std():>11.4f}')
for i, nm in enumerate(["theta_a", "theta_b", "theta_c"]):
    g = th_g[:, i].std()
    print(f'{nm+" mean err / gold sd":>34} {(mu0[i]-th_g[:,i].mean())/g:>+10.4f} '
          f'{(mu3[i]-th_g[:,i].mean())/g:>+11.4f} {(th_A[:,i].mean()-th_g[:,i].mean())/g:>+11.4f}')
dg = np.sqrt(np.diag(Sg))[3:]; dL = np.sqrt(np.diag(Sig))[3:]; dC = np.sqrt(np.diag(Sc))[3:]
dA = A[:, 3:].std(0)
print(f'{"trajectory sd ratio  median":>34} {np.median(dL/dg):>10.4f} {np.median(dC/dg):>11.4f} '
      f'{np.median(dA/dg):>11.4f}')
print(f'{"                     90th pct":>34} {np.percentile(dL/dg,90):>10.4f} '
      f'{np.percentile(dC/dg,90):>11.4f} {np.percentile(dA/dg,90):>11.4f}')
print(f'{"trajectory mean err / sd  max":>34} {np.abs((mu0[3:]-Sg.shape[0]*0-gold[:,3:].mean(0))/dg).max():>10.4f} '
      f'{np.abs((mu3[3:]-gold[:,3:].mean(0))/dg).max():>11.4f} '
      f'{np.abs((A[:,3:].mean(0)-gold[:,3:].mean(0))/dg).max():>11.4f}')
