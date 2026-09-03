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

### The prediction was half wrong

**(a) was wrong.** From a `fit()` start the rule is *not* degenerate. R rises monotonically with h
and crosses 1 at a finite bandwidth on all three systems; it does not saturate at `R(start)`. The
reason the freezing argument fails is Prodigy: as `h` grows the SVGD update shrinks like `1/h`, but
Prodigy estimates its own step size from the gradient it is given and scales up to compensate, so
the ensemble keeps moving. The rigid-translation limit of investigation 7 §10 is a statement about
the *flow*, and an adaptive optimizer does not integrate the flow at a fixed rate.

**(b) was right**, and stronger than predicted: from the mis-scaled start argmax-R and
argmin-energy are the *same* rung on all three systems.

### The sweep, `fit()` start, K = 400, 2000 iterations

fn, `h0 = 0.402`, floor 0.0820:

| h/h0 | 1 | 10 | 100 | 300 | 1000 | 3000 | 10⁴ | 3·10⁴ | 10⁵ |
|---|---|---|---|---|---|---|---|---|---|
| Stein R | 0.027 | 0.326 | 0.754 | 0.862 | **0.966** | 1.041 | 1.091 | 1.116 | 1.137 |
| energy | 6.322 | 1.450 | 0.193 | 0.105 | **0.069** | 0.062 | 0.060 | **0.058** | 0.059 |
| × floor | 77.1 | 17.7 | 2.35 | 1.27 | **0.83** | 0.75 | 0.73 | **0.71** | 0.72 |

lorenz, `h0 = 135.4`, floor 0.0783:

| h/h0 | 1 | 10 | 100 | 300 | 1000 | 3000 | 10⁴ | 3·10⁴ | 10⁵ |
|---|---|---|---|---|---|---|---|---|---|
| Stein R | 0.033 | 0.277 | 0.799 | 0.894 | 0.944 | 0.969 | 0.986 | 0.996 | **1.004** |
| × floor | 67.9 | 20.0 | 1.43 | 0.82 | 0.67 | 0.63 | **0.62** | 0.63 | **0.63** |

### Does the rule work?

| system | start | rule picks | oracle picks | h ratio | energy at rule | at oracle | **penalty** |
|---|---|---|---|---|---|---|---|
| fn | `fit()` | 1000·h0 | 3·10⁴·h0 | 1/30 | 0.0685 (0.83×) | 0.0579 (0.71×) | **1.18×** |
| hiv | `fit()` | 10⁵·h0 | 10⁴·h0 | 10 | 0.0677 (0.63×) | 0.0669 (0.62×) | **1.01×** |
| lorenz | `fit()` | 10⁵·h0 | 10⁴·h0 | 10 | 0.0491 (0.63×) | 0.0489 (0.62×) | **1.00×** |
| fn | 0.25× | 10·h0 | 10·h0 | 1 | 1.138 (13.9×) | same | 1.00× |
| lorenz | 0.25× | 10·h0 | 10·h0 | 1 | 1.739 (22.2×) | same | 1.00× |
| hiv | 0.25× | — | — | — | R never exceeds 0.063 at any h | | — |

**The rule is usable.** It costs 1.00–1.18× the oracle's energy distance on every system, well
inside the 2× bar, even though it picks a bandwidth up to 30× away from the oracle's — the energy
curve is flat over a decade or more at its bottom, so a loose rule is enough.

Two honest qualifications.

* **The rule optimises the joint, not θ.** On fn the θ error is minimised at `h = 100·h0` (0.049
  reference sd) and is 5× worse at the rule's pick (0.256). A user who cares only about parameters
  should not use this h.
* **On a badly-scaled start the rule reports failure rather than a wrong answer**, which is the
  desirable behaviour. On HIV from the 0.25× start R never exceeds 0.063 at any bandwidth in five
  decades, so "raise h until R ≈ 1" terminates without a candidate and tells the user the run has
  not converged. That is §2's budget question, not a defect in the rule.

### Is R flat near its maximum?

Not flat enough to matter, and — more usefully — **its flatness tracks the energy's.** On fn near
the crossing R moves 0.075 per half-decade of h (0.966 → 1.041 → 1.091) and the energy is still
changing; on lorenz R moves 0.01 per half-decade (0.986 → 0.996 → 1.004) and the energy is
*also* flat there (0.0489 → 0.0489 → 0.0491). Where the rule cannot resolve h, it does not need
to.

