import json, sys, numpy as np
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print(f'{"case":>12} {"fit b/a":>14} {"diag b/a":>14} {"dtheta/sd":>10} {"dESS":>8} {"dfd":>10} {"gate":>6}')
tf = td = tf2 = td2 = 0.0
for k in a:
    x, y = a[k], b.get(k)
    if y is None: print(f'{k:>12}  MISSING'); continue
    th0, th1 = np.array(x["theta"]), np.array(y["theta"])
    sd = np.maximum(np.array(x["theta_sd"]), 1e-300)
    dth = np.max(np.abs(th1 - th0) / sd)
    tf += x["t_fit"]; tf2 += y["t_fit"]; td += x["t_diag"]; td2 += y["t_diag"]
    g = "same" if x["reliable"] == y["reliable"] else f'{x["reliable"]}->{y["reliable"]}'
    print(f'{k:>12} {x["t_fit"]:6.1f}->{y["t_fit"]:6.1f} {x["t_diag"]:6.1f}->{y["t_diag"]:6.1f} '
          f'{dth:>10.2e} {y["ess"]-x["ess"]:>8.1f} {x["fd"]:g}->{y["fd"]:g} {g:>6}')
print(f'\nTOTAL fit {tf:.1f} -> {tf2:.1f}s  ({tf/max(tf2,1e-9):.2f}x)   '
      f'diag {td:.1f} -> {td2:.1f}s  ({td/max(td2,1e-9):.2f}x)')
