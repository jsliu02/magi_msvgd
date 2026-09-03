# Investigation 8 — settling what investigation 7 left open

[investigation7.md](investigation7.md) ended with a "what is not settled" list. This works through
it. Nothing here revises investigation 7; where a result changes a recommendation, the change is
stated against the section it changes.

Code in `investigation8/`; the harness, the SVGD driver and the systems are the investigation-7
ones, symlinked rather than copied, so every number below is scored the same way as every number
there. Scoring rules unchanged and worth restating once:

* **Mahalanobis energy distance** to the reference, always with the floor at the ensemble's own
  particle count K (400 exact draws score 0.082 / 0.108 / 0.078 on fn / hiv / lorenz — not 0).
* **Stein R** `= -(1/(k·dim)) Σ (x_i − x̄)·s(x_i)`, 1 under the target, and the only quantity here
  that a user could compute without a reference.
* **Marginal standard deviations are never a score.** They read 0.99 on ensembles 60× the floor.
* **Band profile**: variance ratio along the reference covariance's own fixed eigenvectors, five
  equal-count bands from softest to stiffest. Unbiased at any K, unlike the raw eigenvalue
  spectrum.
* float64 throughout, `atol = rtol = 0` so iteration counts are exact.

## 0. Precondition: are the cached references still valid?

`tests.py` changed under investigation 7 (commit `cbf5088`): the system definitions are numpy now,
so grids are float64 regardless of the x64 flag, and `_locate`'s tolerance scales with the grid
spacing. Every number in this report is scored against `investigation5/ref5_*.npz`, built before
that change, so this had to be checked rather than assumed. `investigation8/exp00_check.py`:

| system | model dim | reference dim | log p(ref mean) | ‖∇‖ at ref mean | Stein R on `sub` | `fit()` energy | floor |
|---|---|---|---|---|---|---|---|
| fn | 325 | 325 | −8.119 | 9.8e1 | 0.9996 | 0.1372 | 0.0820 |
| hiv | 608 | 608 | −894.03 | 5.6e0 | 1.0008 | 0.1114 | 0.1078 |
| lorenz | 306 | 306 | −231.93 | 3.2e1 | 0.9981 | 0.0972 | 0.0783 |

Dimensions match, Stein R on the reference subsample is 1 to three decimals, and the `fit()`
energies and floors are **bit-identical** to investigation 7's. Everything in x64 was unaffected
by the grid change, as expected — the 45-vs-33-point discrepancy was an x64-off effect and Hes1 is
excluded here anyway. The float32 MAP regression (`cbf5088`) likewise touches nothing: the float64
Cholesky ridge went from a fixed 1e-12 to 4096·eps = 9.1e-13.

## 1. A reference-free bandwidth rule

Everything in investigation 7 §9 rests on a bandwidth of `1000·h0` found by sweeping against a
reference. §13 proposed a reference-free substitute — *raise h until Stein R ≈ 1* — and never
tested it. `investigation8/exp01_bwrule.py` sweeps `h/h0` over five decades on fn, hiv and lorenz,
K = 400, 2000 iterations, recording at each rung both what the rule sees (R) and what the rule is
trying to minimise (energy distance).

### Prediction, written before the run

> **(a) From a `fit()` start the rule is degenerate and will fail.** `R(start)` is already ≈ 1
> there, and as `h → ∞` the SVGD update tends to a rigid translation that cannot change the
> ensemble's shape (investigation 7 §10). So `R → R(start) ≈ 1` *for free* at large h, and
> "raise h until R ≈ 1" selects `h = ∞` and returns the starting ensemble untouched — passing its
> own test while doing nothing. If this holds, the rule as stated in §13 is wrong.
>
> **(b) From a mis-scaled (0.25×) start the rule has content**, because `R(start) ≈ 0.06` and only
> a bandwidth that actually moves the ensemble can raise it. There I expect argmax R within a
> decade of argmin energy, since both are driven by the same expansion.

Both starts are therefore run: the `fit()` ensemble, and the same ensemble shrunk 4× about its
mean.

