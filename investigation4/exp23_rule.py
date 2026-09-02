"""Exp 23: the decision rule of section 4.2, evaluated on all four settings."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from pipeline import metrics
d = H.DIM; I = np.eye(d)
print(f'{"setting":>9} {"|D3|":>7} {"|Dvi|":>7} {"disagree":>9} {"dis/|D3|":>9} {"RULE":>7} | '
      f'{"bias MAP":>9} {"bias 3rd":>9} {"bias mid":>9} {"helped?":>8} {"rule ok?":>9}')
for n in ["baseline", "half", "noisy", "quarter"]:
    z = np.load(f"ref4_{n}.npz"); D = np.load(f"determ_{n}.npz"); mp = np.load(f"map4_{n}.npz")
    Hs = mp["H"]; tau = lambda v: float(np.sqrt(abs(v @ Hs @ v) / d))
    sc = metrics(z["mean"], z["cov"], d)
    c3 = tau(D["mu3"] - D["mu0"]); cv = tau(D["muvi"] - D["mu0"]); dis = tau(D["mu3"] - D["muvi"])
    apply = (dis / c3 < 0.5) and (c3 < 0.5)
    bM, b3 = sc(D["mu0"], I)["bias"], sc(D["mu3"], I)["bias"]
    bmid = sc(0.5 * (D["mu3"] + D["muvi"]), I)["bias"]
    helped = b3 < bM
    print(f'{n:>9} {c3:>7.3f} {cv:>7.3f} {dis:>9.4f} {dis/c3:>9.3f} '
          f'{("apply" if apply else "SUPPRESS"):>7} | {bM:>9.4f} {b3:>9.4f} {bmid:>9.4f} '
          f'{("yes" if helped else "NO"):>8} {("yes" if apply == helped else "NO"):>9}')
print('\nrule: apply the correction iff  disagree/|D3| < 0.5  and  |D3| < 0.5')
