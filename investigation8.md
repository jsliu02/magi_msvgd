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

> **Erratum affecting investigation 7.** §2.6 below identifies a bug in
> `investigation7/msvgd7.py`: the preconditioned gradient applied the metric twice. **Every
> `precond` row in investigation 7 §3 is invalid** — both the "worst of four kernels on fn" and the
> "best of four on lorenz" readings. The driver is fixed (verified to 1.9e-15 against the analytic
> chain rule) and the corrected result reverses the conclusion: preconditioning is the best
> configuration found in either investigation. investigation7.md is left unedited; this note is the
> correction.

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

### An unwelcome discovery about the instruments themselves

The rule above is validated by energy distance, so before trusting it: can R be satisfied by a
wrong ensemble? `investigation8/exp01b_Rsufficiency.py`. For a Gaussian target
`R = tr(A Σ⁻¹)/dim`, so *any* ensemble whose whitened covariance eigenvalues average 1 scores
R = 1 however they are distributed. Take K = 400 exact reference draws, express them in the
reference covariance's own eigenbasis, multiply the **softest half** of the directions by
`√(1+a)` and the **stiffest half** by `√(1−a)`, map back. The mean is untouched and R = 1 exactly
for every `a`. This is not a contrived perturbation — it is precisely SVGD's failure mode.

fn, K = 400 (floors: energy 0.0820, stiff-var 1.000, KS 1.00):

| a | Stein R | energy | × floor | stiff-var | KS / floor | sd ratio | band profile |
|---|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 0.0702 | 0.86 | 1.001 | 0.96 | 1.010 | 1.01 1.01 1.00 1.01 0.99 |
| 0.50 | 0.9985 | 0.0764 | 0.93 | 0.501 | 1.61 | 1.223 | 1.51 1.51 0.99 0.51 0.50 |
| **0.95** | **0.9971** | **0.0873** | **1.06** | **0.050** | 2.22 | 1.387 | 1.97 1.96 0.98 0.05 0.05 |

**At a = 0.95 the ensemble has 1.95× the correct variance in half the directions and 0.05× in the
other half — a 40-fold misallocation — and the Mahalanobis energy distance is 1.06× the floor.**
hiv and lorenz behave identically (1.00× and 1.08× at a = 0.95).

So investigation 7's two headline diagnostics are *both* trace statistics, and this distortion
preserves the trace. In d ≈ 300 the energy distance is dominated by the radial distribution, and
scaling half the directions up and half down leaves `‖y‖²` essentially unchanged. That is why
investigation 7 saw energies of 6–7 for the collapsed ensembles: those had lost 98% of their
*total* variance, which is a radial error, and energy distance is superb at radial errors and
nearly blind to angular ones.

**This qualifies investigation 7 §9.** "Energy 0.052 against a floor of 0.078, therefore
indistinguishable from 400 exact draws" is too strong. It rules out a scale error; it does not
rule out this one. Three scores without that blind spot are added in `investigation8/metrics8.py`
and used from here on:

* **stiff-var** — variance ratio over the stiffest 10% of reference directions (0.050 above);
* **worst-band** — 5th/95th percentile of the per-direction variance ratio;
* **KS** — mean per-coordinate two-sample Kolmogorov–Smirnov statistic against reference draws,
  no covariance in it at all (2.2× its floor above).

### Re-scored with instruments that can see it

`investigation8/exp01c_sharp.py` repeats the sweep with all of them. HIV
(`h0 = 2.04e6`; floors: energy 0.1078, stiff-var 0.998, KS 0.0469, worst-band [0.884, 1.125]):

| h/h0 | Stein R | energy ×flr | stiff-var | wb lo | wb hi | KS/flr | max\|θ err\| |
|---|---|---|---|---|---|---|---|
| **1 (the shipped default)** | 0.080 | 60.5 | **0.0013** | 0.001 | 0.528 | 8.34 | 0.104 |
| 10 | 0.249 | 26.6 | 0.0065 | 0.004 | 1.010 | 3.21 | 0.094 |
| 100 | 0.783 | 1.60 | 0.655 | 0.575 | 1.060 | 0.94 | 0.013 |
| 1000 | 0.878 | 0.87 | 0.824 | 0.727 | 1.073 | 0.85 | 0.013 |
| **10⁴** | 0.972 | **0.62** | **0.993** | 0.865 | 1.118 | **0.83** | 0.028 |
| 10⁵ | 0.995 | 0.63 | 1.058 | 0.881 | 1.148 | 0.83 | 0.019 |
| *400 exact draws* | *1.001* | *0.86* | *0.990* | *0.892* | *1.136* | *0.98* | *0.114* |

Lorenz at `h ≥ 10⁴·h0`: stiff-var 1.002, KS 1.00× floor, θ error 0.075. fn at `h ≥ 3000·h0`:
stiff-var 0.985–1.044, KS 1.12–1.40× floor.

**The answer to investigation 7's "right, or only quasi-uniform?" is: right.** On all three
systems the large-`h` ensembles match the reference on the stiffest directions to within 1–6%,
match its marginal quantile calibration to within 17% of the KS floor (better than it on hiv and
lorenz), and do so while scoring below the K-particle energy floor. Meanwhile the *shipped*
default at `h = h0` retains **0.13% of the variance in HIV's stiffest directions** and is 8.3× the
KS floor — the collapse is if anything worse than the energy distance made it look.

### Section 1's verdict, after §2 and §2.5 — read them together

§2 shows the 2000-iteration numbers above are **transients**: the flow does have a fixed point
(§2.5) but it is not the target, and these ensembles are caught passing through the target on the
way to it. The bandwidth rule still selects well *at a fixed iteration budget*, and the penalty
figures stand as measured, but "the rule finds a good bandwidth" has to be read as "the rule finds
a good place to stop". §2.6 then supersedes the whole approach with one that has a genuine fixed
point and needs no stopping rule.

## 2. Sampler or polisher? — a fixed bandwidth buys time, and misallocates variance

> **This section was written twice.** The first version concluded there is *no* fixed point at a
> fixed bandwidth. That is wrong, and §2.5 below retracts it: there is a perfectly good fixed
> point, reached from both starts and stable over half a million iterations — it simply is not the
> target, because it puts the variance in the wrong directions. The measurements are unchanged;
> the reading of them is corrected. I have left the original reasoning in place because the route
> to the correction runs through it.

Investigation 7 §9 saw a large fixed bandwidth converge from a well-scaled start but not from a 4×
narrow one in 2000 iterations, and guessed it was a budget question.
`investigation8/exp02_budget.py` runs four starts to 20,000 iterations at `h = 1000·h0`; all are
built from the same `fit()` ensemble so only the placement varies.

fn, `h = 1000·h0`, stiff-var (1 = correct) at 1k / 5k / 10k / 20k iterations:

| start | initial | 1k | 5k | 10k | 20k |
|---|---|---|---|---|---|
| correct | 1.06 | 0.968 | 0.826 | 0.741 | **0.654** ↓ |
| displaced 5 sd along the stiffest direction | 1.06 | 0.979 | 0.837 | 0.750 | **0.661** ↓ |
| wide 4× | 16.9 | 11.61 | — | — | **2.25** ↓ |
| narrow 4× | 0.065 | 0.0653 | 0.0667 | 0.0689 | **0.0729** ↑ |

**The correct start decays.** It is not a fixed point at all: energy goes 0.73× → 1.95× the floor
while stiff-var falls by a third and Stein R falls from 1.03 to 0.78. The displaced start is
indistinguishable from it after 1000 iterations (the mean is repaired fast — θ error 0.33 → 0.03 —
and then the spread decays identically), and the wide start descends toward the same place. All
four are converging on a common attractor, and it is not the target.

### The bandwidth is a clock, not a fix

`investigation8/exp02b_equilibrium.py` runs the correct and 4×-narrow starts at four bandwidths
spanning three decades, to 100,000 iterations. fn, stiff-var from the correct start:

| iterations | 1k | 2k | 5k | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|---|
| h = 10³·h0 | 0.968 | 0.918 | 0.826 | 0.741 | 0.654 | 0.581 | 0.607 |
| h = 10⁴·h0 | 1.030 | 1.018 | 0.994 | **0.963** | 0.918 | 0.829 | **0.746** |
| h = 10⁵·h0 | 1.049 | 1.044 | 1.036 | 1.028 | 1.017 | 0.993 | **0.963** |
| h = 10⁶·h0 | 1.057 | 1.056 | 1.053 | 1.049 | 1.043 | 1.035 | 1.028 |

Read it diagonally. `h = 10⁴·h0` at 100k (0.746) is `h = 10³·h0` at 10k (0.741).
`h = 10⁵·h0` at 100k (0.963) is `h = 10⁴·h0` at 10k (0.963). **Ten times the bandwidth buys
exactly ten times the delay, and nothing else.** Lorenz gives the same table (0.809/0.814 and
0.965/0.965 at the corresponding cells). The rescaling also holds downward: exp01c's `h = 300·h0`
at 2000 iterations (stiff-var 0.794) matches `h = 10³·h0` at ~6700 (0.79).

So investigation 7 §9's "the fix transfers" and §1's bandwidth rule above are both **early
stopping**. A large `h` does not give the flow a correct fixed point; it makes the flow slow
enough that 2000 iterations lands while the ensemble is still passing through the neighbourhood of
the target on its way down. That is a real and usable effect — the ensembles measured at that
point are genuinely correct by every score in §1 — but it must be described as what it is, and
"run it longer to be safe" is precisely the wrong advice.

### Where it ends up: the collapse, at every bandwidth

`investigation8/exp02c_longtime.py` exploits the rescaling — to see 100× further into the future,
divide `h` by 100 rather than multiplying the iteration count — and runs `h = 10·h0` and
`h = 100·h0` to 500,000 iterations from both a correct and a 4×-narrow start.

fn (floors: energy 0.0820, stiff-var 1.010):

| h | start | 5k | 20k | 50k | 100k | 200k | 500k |
|---|---|---|---|---|---|---|---|
| 10·h0 | correct | 0.006 | 0.004 | 0.004 | 0.005 | 0.005 | **0.004** |
| 10·h0 | narrow 4× | 0.009 | 0.012 | 0.005 | 0.007 | 0.004 | **0.007** |
| 100·h0 | correct | 0.498 | 0.408 | 0.412 | 0.360 | 0.202 | **0.048** ↓ |
| 100·h0 | narrow 4× | 0.082 | 0.165 | 0.315 | 0.264 | 0.094 | **0.017** ↓ |

At `h = 10·h0` the two starts **meet** — 0.004 and 0.007, energy 22.1× and 22.0× the floor, Stein
R 0.269 and 0.270 — and sit there unchanged from 20k to 500k iterations. That is a genuine fixed
point, and it is the collapse. At `h = 100·h0` the two starts also converge (0.36 and 0.26 at
100k) and then **fall together** toward the same place.

Lorenz at `h = 10·h0` is identical in character (0.010 / 0.011, energy 26.4× / 26.2× the floor,
R 0.204 / 0.206, flat from 50k on). At `h = 100·h0` it passes through a good window near 100k
iterations (energy 4.85× and 1.88× the floor from the two starts) and then leaves it, inflating to
stiff-var 1.59 and energy 20.9× by 500k.

**So the answer to "sampler or polisher" is: neither, exactly.** A fixed bandwidth does not give
the flow a correct fixed point at any value tested. What it does is open a **transient window**
during which the ensemble is genuinely correct, and move that window later in proportion to `h`.
Investigation 7 §9's result and §1's above are both measurements taken inside that window.

### Which makes §1's rule a stopping rule, and a good one

This is not as negative as it sounds, because the window is wide and Stein R locates it. On fn at
`h = 10⁶·h0`, R runs 1.166 → 1.085 and the energy stays at 0.68–0.71× the floor for the entire
100,000 iterations measured. On lorenz at `h = 10⁵·h0`, R runs 1.029 → 0.965 and energy stays at
0.61–0.64× the floor throughout. Where R is near 1 the ensemble is correct; where it has fallen
away, the ensemble has left the window. The reference-free procedure is therefore:

> Take `h` in the range 10⁵–10⁶ × the median heuristic evaluated on the starting ensemble, run,
> monitor Stein R, and **stop when R crosses 1**. Do not run to convergence — convergence is the
> collapse.

That is what §1 measured under a different description, and its 1.00–1.18× penalties stand. What
does *not* stand is any statement of the form "SVGD with a large fixed bandwidth samples this
posterior". It does not; it passes through it.

## 2.5 Correction: there *is* a fixed point. It is not the target.

§2 concluded "a fixed bandwidth has no correct fixed point ... what it does is open a transient
window". The second half is right and the first half is wrong, and the error was in which number I
read. §2 tracked **stiff-var**, the stiffest 10% of directions. Re-reading the same runs by the
**median** whitened variance ratio (`whsd²`, recorded all along in `exp02c_results_*.json`):

fn, `h = 10·h0`, over 5k → 500k iterations:

| | 5k | 20k | 50k | 100k | 200k | 500k |
|---|---|---|---|---|---|---|
| correct start, `whsd²` | 0.290 | 0.146 | 0.145 | 0.144 | 0.143 | **0.145** |
| narrow 4× start, `whsd²` | 0.400 | 0.157 | 0.145 | 0.148 | 0.144 | **0.146** |
| correct start, stiff-var | 0.006 | 0.004 | 0.004 | 0.005 | 0.005 | 0.004 |

**That is a fixed point.** Two starts a factor of 16 apart in variance meet at 0.145 by 50,000
iterations and hold it, to three digits, for the next 450,000. Lorenz likewise: 0.236 and 0.238.
And at `h = 100·h0` on fn the median direction converges to **0.950 / 0.962** from the two
starts — nearly correct — while stiff-var falls to 0.048 / 0.017.

So the flow is not failing to converge. It converges to a distribution whose *bulk* is close to
right and whose *stiffest directions* are 20–60× too narrow. What I recorded in §2 as "still
falling, no equilibrium" was the stiff directions draining toward that misallocated fixed point.

### The cause is anisotropy, and this is the clean test of it

`investigation8/exp05c_aniso_vs_nongauss.py`. Two exact Gaussians in d = 325, K = 400, identical
in everything but the covariance, at three fixed bandwidths, from a correct and a 4×-narrow start,
to 200,000 iterations. Mean whitened variance ratio:

| target | h/h* | correct: 2k → 200k | narrow 4×: 2k → 200k |
|---|---|---|---|
| N(0, I) | 3 | 0.5844 → **0.5845** | 0.5844 → **0.5845** |
| N(0, I) | 10 | 0.8106 → **0.8151** | 0.8106 → **0.8151** |
| N(0, I) | 30 | 0.9256 → **0.9286** | 0.9256 → **0.9286** |
| N(0, Σ_ref), cond 7.9e3 | 3 | 0.0933 → **0.0864** | 0.1042 → **0.0907** |
| N(0, Σ_ref) | 10 | 0.3481 → **0.3300** | 0.1450 → **0.3746** |
| N(0, Σ_ref) | 30 | 0.5894 → **0.5657** | 0.1300 → **0.5269** |

Both targets have a stable attractor; both are reached from both starts. **Anisotropy costs a
factor of 1.6–6.8 in the equilibrium variance ratio and does not destroy the fixed point.** Since
the only difference between the two rows is the covariance — same dimension, same K, same
bandwidth rule, exactly Gaussian in both cases — this isolates anisotropy as the cause with no
non-Gaussianity anywhere in the experiment.

The MAGI posteriors sit where this predicts: fn at `h = 10·h0 ≈ 10·h*` equilibrates at a median
ratio of 0.145 against the anisotropic Gaussian's mean of 0.33 at the same bandwidth, with the
extra loss attributable to fn's spread of scales being worse in the tails of its spectrum than the
5–95% band suggests.

### What survives from §2, and what does not

* **Retracted:** "a fixed bandwidth has no correct fixed point at any value tested". It has one.
* **Stands:** the 1/h time rescaling (ten times the bandwidth, ten times the delay — the diagonal
  reading of §2's table), the transient window, the fact that running to convergence is worse than
  stopping early, and the stopping rule. The window is now explicable rather than mysterious: the
  ensemble starts correct and *relaxes to the misallocated equilibrium*, passing through a long
  stretch during which the stiff directions have not yet drained.
* **Stands and is now confirmed:** the anisotropy explanation, by `exp05c` rather than by the
  preconditioning experiment that was supposed to confirm it.

## 2.6 The preconditioned route — a bug, then a genuine fixed point

§2.5 says the misallocation is caused by anisotropy. The obvious way to act on that is to remove
the anisotropy: run SVGD in coordinates whitened by `H⁻¹` at the MAP, with a *fixed* bandwidth.
Investigation 7 §3 tested preconditioning only at the adaptive bandwidth, where it was the worst
of four kernels, so this combination had never been run.

`investigation8/exp05_precond_bigh.py` ran it and it **diverged catastrophically** — energy 1192×
the floor on fn, Stein R 5×10⁵. Three follow-ups were needed to find out why.

**It was not the whitening** (`exp05b_whitening.py`). Whitening the *reference* covariance by the
Laplace factor gives:

| system | cond(Σ_ref) | cond(L⁻¹ Σ_ref L⁻ᵀ) | 5–95% eigenvalues after | floored eigenvalues of H |
|---|---|---|---|---|
| fn | 7.9e3 | **47** | 0.86 – 1.22 | 0 |
| hiv | **5.6e10** | **1.6** | 0.82 – 1.21 | 0 |
| lorenz | 1.7e5 | 192 | 0.82 – 1.53 | 0 |

The Laplace metric whitens these posteriors almost perfectly — HIV's condition number falls by ten
orders of magnitude.

**It was not anisotropy or non-Gaussianity** (`exp05d_precond_why.py`): preconditioned SVGD
diverged on an *exact* Gaussian too, and was perfectly stable on the *real posterior* with a small
fixed step. So it was the optimizer — or so it looked.

### It was a bug in my own driver

`investigation7/msvgd7.py`, used by every `precond` run in investigations 7 and 8:

```python
grad_v = jax.vmap(lambda y, d: Lj.T @ jax.grad(lambda z: m.logdensity(x0j + Lj @ z, d))(y))
```

`jax.grad` already differentiates through the `Lj @ z` inside, so it returns `Lᵀ ∇ₓ log p` on its
own; the explicit `Lj.T @` applied the metric a **second** time, giving `Lᵀ Lᵀ ∇ₓ log p`.
`exp05f_gradcheck.py` verifies the corrected line against the analytic chain rule to **1.9e-15**.

**Every `precond` row in investigation 7 §3 is invalid**, including the "worst of four kernels on
fn" and "best of four on lorenz" readings, and so are `exp05`, `exp05d` and the first `exp05e`
here. The driver is fixed with a comment recording this.

### With the bug fixed, it works — and the failure mode changes character

`investigation8/exp05e_precond_sgd.py`, fn, K = 400, whitened, `h = 10·h*`, SGD at 0.01, three
starts, 100,000 iterations. Variance ratio in the stiffest 10% of directions:

| start | 5k | 25k | 50k | 100k |
|---|---|---|---|---|
| correct | 0.959 | 0.787 | 0.738 | **0.748** |
| narrow 4× | 0.077 | 0.137 | 0.244 | **0.494** ↑ |
| wide 4× | 13.18 | 4.703 | 1.587 | **0.788** ↓ |

**A genuine attractor, approached from a factor of 4 below and a factor of 4 above.** And — this is
the point — the deficit is now **uniform**: stiff-var 0.748 against `whsd²` 0.719, a 1.04×
mismatch, where the unpreconditioned dynamics at `h = 100·h0` gives `whsd²` 0.95 against stiff-var
0.048, a **20× mismatch**. Preconditioning converts a *shape* error into a *scale* error, exactly
as §2.5 predicts, and the residual matches exp02d's isotropic equilibrium (0.815 at `h = 10·h*`).

### A scale error is one number, and Stein R measures it without a reference

For a Gaussian target `R = tr(A Σ⁻¹)/dim`, so an ensemble uniformly a factor `c` too narrow reads
`R = c`. Rescaling about its own mean by `1/√R` should therefore correct it — no reference, no
tuning, no extra gradient evaluations. `investigation8/exp07_inflate.py`:

| system | ensemble | Stein R | energy | × floor | stiff-var | whsd² | KS/floor | max\|θ err\| |
|---|---|---|---|---|---|---|---|---|
| fn | start: `fit()` draws | 1.274 | 0.1357 | 1.65 | 1.058 | 1.007 | 1.22 | 0.077 |
| fn | precond SVGD equilibrium | 0.702 | 0.1984 | 2.42 | 0.748 | 0.719 | 1.87 | 0.242 |
| fn | **rescaled by 1/√R** | 1.074 | **0.0554** | **0.68** | **1.065** | **1.024** | 1.89 | 0.242 |
| lorenz | start: `fit()` draws | 1.019 | 0.0974 | 1.24 | 1.037 | 1.007 | 1.16 | 0.071 |
| lorenz | precond SVGD equilibrium | 0.653 | 0.2290 | 2.92 | 0.696 | 0.694 | 1.64 | 0.401 |
| lorenz | **rescaled by 1/√R** | 1.074 | **0.0752** | **0.96** | **1.066** | **1.063** | 2.10 | 0.401 |
| hiv | start: `fit()` draws | 1.000 | 0.1111 | 1.03 | 1.074 | 1.017 | 1.01 | 0.096 |
| hiv | precond SVGD equilibrium | 0.538 | 0.7307 | 6.78 | 0.573 | 0.548 | 2.00 | 0.067 |
| hiv | **rescaled by 1/√R** | 1.000 | **0.0615** | **0.57** | **1.065** | **1.017** | **0.80** | **0.067** |

**A 2.4–6.8× ensemble becomes a 0.57–0.96× one, by a rescaling computed from the ensemble alone.**
On HIV — the best-conditioned system, with the zero-divergence reference — the result is complete:
energy 0.57× the floor, KS **0.80×** the floor (better than 400 exact draws), stiff-var 1.065, and
the parameter error *improves* from 0.096 to 0.067.

So the full reference-free recipe is:

> 1. Start from `fit()` draws. 2. Whiten by `H⁻¹` at the MAP. 3. Run SVGD at a fixed
> `h = 10·h*`, `h* = 2d/ln K`, with a small fixed step — **not** Prodigy, which is unstable in
> whitened coordinates. 4. Run to equilibrium (it is one; you may stop when it stops moving).
> 5. Measure Stein R and rescale the ensemble about its mean by `1/√R`.

Unlike §1's rule this needs no stopping rule, because the target of the dynamics is a fixed point
rather than a point passed through.

**What it does not fix, on two of three systems.** The parameter mean drifts on fn (max |θ error|
0.077 → 0.242 against a floor of 0.010) and lorenz (0.071 → 0.401 against 0.041), and the rescaling
cannot help because it leaves the mean alone; their KS statistics also sit at 1.9–2.1× the floor.
HIV does neither — θ error improves and KS lands below its floor. So the drift is not intrinsic to
the method, and what distinguishes the three is not established here. Until it is, `fit()` remains
the safer answer for parameters, and this recipe is for the joint posterior's shape.

## 3. Profiled SVGD, broadened

Investigation 7 §11's strongest positive result — SVGD on the p-dimensional profiled marginal,
below the K-particle floor and beating `fit()` — was two systems at one particle count, timed on a
contended CPU. `investigation8/exp03_profiled.py` adds HIV, sweeps K over {16, 64, 256}, and times
everything on one GPU with nothing else running, including each run's own JIT compile.

The target is `log p̂(θ) = log p(θ, X*(θ)) − ½ log det H_XX(θ)`, with `X*(θ)` from three
Gauss–Newton steps warm-started by the implicit-function predictor, differentiated end to end.
Cold start: Laplace θ draws at the joint MAP. 400 iterations.

**FitzHugh–Nagumo** (p = 3, nD = 322; `fit()` = 2.3 s on this device):

| K | method | θ energy | × floor | max\|θ err\| | sec | vs `fit()` |
|---|---|---|---|---|---|---|
| 16 | `fit()` | 0.2088 | 1.77 | 0.392 | 2.3 | — |
| 16 | **profiled SVGD** | **0.0417** | **0.35** | **0.0198** | 13.3 | 5.8× |
| 64 | `fit()` | 0.0758 | 2.52 | 0.239 | 2.3 | — |
| 64 | **profiled SVGD** | **0.0089** | **0.30** | **0.0119** | 26.9 | 11.8× |
| 256 | `fit()` | 0.0136 | 1.61 | 0.108 | 2.3 | — |
| 256 | **profiled SVGD** | **0.0027** | **0.32** | **0.0100** | 80.4 | 35.4× |
| | *reference θ-error floor* | | | *0.0100* | | |

**Lorenz** (p = 3, nD = 303; `fit()` = 2.0 s):

| K | method | θ energy | × floor | max\|θ err\| | sec | vs `fit()` |
|---|---|---|---|---|---|---|
| 16 | `fit()` | 0.1227 | 1.56 | 0.393 | 2.0 | — |
| 16 | **profiled SVGD** | **0.0410** | **0.52** | **0.0656** | 10.9 | 5.5× |
| 64 | `fit()` | 0.0416 | 1.55 | 0.175 | 2.0 | — |
| 64 | **profiled SVGD** | **0.0126** | **0.47** | **0.0710** | 21.7 | 10.9× |
| 256 | `fit()` | 0.0153 | 1.46 | 0.0839 | 2.0 | — |
| 256 | **profiled SVGD** | **0.0025** | **0.24** | **0.0142** | 69.1 | 34.8× |
| | *reference θ-error floor* | | | *0.0405* | | |

Three things this settles.

* **The advantage is stable in K, not an artefact of K = 64.** Profiled SVGD sits at 0.24–0.52× the
  K-particle floor at every K on both systems, while `fit()` sits at 1.46–2.52×. The ratio is
  4.4–8.5× throughout.
* **It survives at the K a user would pick.** The cheapest setting, K = 16 at 5.5–5.8× `fit()`'s
  wall clock, is already 4.5–5× better than `fit()` on the same statistic. Buying more particles
  costs linearly and does not improve the ratio, so K = 16–64 is the operating point, and the
  honest price is **6–12× `fit()`**, not the 20–80× investigation 7 reported from a contended CPU.
* **The reweighted kernel must not be used here.** Investigation 7 saw it fail on lorenz at K = 64;
  it fails at every K (23×, 29×, 48× the floor) while succeeding on fn (0.55×, 0.49×, 0.93×). Its
  density reweighting exists to fight a collapse that does not occur at p = 3, and what is left is
  a distortion — the sd ratio runs 1.3–1.8 on lorenz.

### HIV: where profiled SVGD stops working, and why the law predicts it

**HIV** (p = 5, nD = 603; `fit()` = 11.0 s):

| K | method | θ energy | × floor | max\|θ err\| | sd ratio | sec |
|---|---|---|---|---|---|---|
| 16 | `fit()` | 0.1193 | **0.50** | 0.119 | 0.961 | 11.0 |
| 16 | profiled SVGD | 0.2529 | 1.05 | 0.103 | **0.578** | 30.8 |
| 64 | `fit()` | 0.0514 | **1.25** | 0.123 | 0.931 | 11.0 |
| 64 | profiled SVGD | 0.0529 | 1.29 | 0.281 | **0.881** | 96.9 |
| 256 | `fit()` | 0.0145 | **0.76** | 0.103 | 0.986 | 11.0 |

On HIV profiled SVGD **loses** at K = 16 (1.05× the floor against `fit()`'s 0.50×) and ties at
K = 64. The sd ratios say why: 0.578 and 0.881, i.e. the θ ensemble is itself underdispersed.

This is investigation 7 §5's law reappearing at p = 5. `Var_SVGD/Var_target ≈ ln(K)/d` predicts
0.55 at (K = 16, d = 5) and 0.83 at (K = 64, d = 5); the measured variance ratios (sd ratio
squared) are 0.33 and 0.78. Against p = 3, where the law predicts 0.92 at K = 16 and >1 at K = 64,
and fn/lorenz duly succeed at every K. The law therefore doubles as a **design rule**:

> Profiled SVGD needs `ln K ≳ p`, i.e. **K ≳ eᵖ** — about 20 particles at p = 3 and 150 at p = 5.
> Below that the parameter ensemble collapses exactly as the joint one does, and `fit()` wins.

HIV at K = 256 (`ln 256 / 5 = 1.11`, so the rule says it should work) first ran out of memory on a
32 GB V100 in the reverse pass — 256 particles × 3 Cholesky factorisations of a 603 × 603 block,
16.7 GiB — and was rerun with the inner Newton steps rematerialised (`jax.checkpoint`) and the
particle axis mapped with `lax.map` rather than `vmap`. The rule is confirmed and the news is
mixed:

| K = 256, HIV | θ energy | × floor | max\|θ err\| | sd ratio | sec |
|---|---|---|---|---|---|
| `fit()` | 0.01451 | 0.76 | 0.1026 | 0.986 | 4.1 |
| profiled SVGD | 0.01401 | **0.73** | **0.0627** | 0.958 | 465.8 (**113×**) |

At K = 256 the θ ensemble is no longer underdispersed (sd ratio 0.958 against 0.578 at K = 16),
exactly as `K ≳ eᵖ` predicts, and profiled SVGD does edge `fit()` — by 1.04× on energy and 1.6× on
the parameter mean. But it costs 113× as much, because the memory-safe version has to
rematerialise. **The advantage that is real at p = 3 is gone at p = 5**, and the reason is
structural rather than incidental: the particle count SVGD needs grows like `eᵖ` while the cost per
particle grows with the state dimension, so the product runs away.

So the honest scope of investigation 7 §11's positive result is: **p ≤ 3, K = 16–64, 6–12×
`fit()`, for a 4.4–8.5× better parameter posterior.** At p = 5 it is a wash at 113× the price.

### `K ≳ eᵖ` tested, not just fitted

The rule above came from two distinct values of p. The *law* it rests on can be tested at as many
as one likes, because it is a statement about SVGD on a target of dimension p and says nothing
about where the target came from. `investigation8/exp06_Kcrit.py`: isotropic `N(0, I_p)` for
p = 2…10, K from 4 to 4096, 4000 iterations, recording for each p the smallest K whose equilibrium
variance ratio clears a threshold.

| p | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| K_crit at ratio ≥ 0.8 | 24 | 64 | 128 | 256 | 768 | 2048 | 4096 |
| `exp(0.8p)`, the law | 5 | 11 | 25 | 55 | 122 | 270 | 602 |
| K_crit at ratio ≥ 0.9 | 96 | 192 | 512 | 2048 | 4096 | — | — |
| `exp(0.9p)`, the law | 6 | 15 | 37 | 90 | 221 | 545 | 1339 |

Least squares on log K_crit:

    threshold 0.8:  ln K_crit = 0.862 p + 1.449   (law: slope 0.8, intercept 0)
    threshold 0.9:  ln K_crit = 0.987 p + 2.451   (law: slope 0.9, intercept 0)

**The exponential rate is confirmed to within 8–10% over seven values of p**, and there is a
constant factor the law omits: you need about **4× more particles at the 0.8 level and 12× at the
0.9 level** than `exp(fp)`. So the working rule is `K ≳ 4·e^{0.86p}` — 45 particles at p = 3, 320
at p = 5, 2400 at p = 7.

The transfer to the actual profiled marginals is good. HIV (p = 5) measured variance ratios of
0.334 at K = 16 and 0.918 at K = 256; the isotropic table at p = 5 gives **0.335** and 0.801 at the
same K. fn and lorenz (p = 3) come out somewhat better than the isotropic table (0.74 measured
against 0.50 at K = 16), which is the direction one would expect for a target that is neither
isotropic nor Gaussian, so the rule is conservative there.

## 4. Deleting the `1/ln K`: it transfers, at 2× rather than 39×

Investigation 7 §10 measured a 39× gain from the plain median heuristic `h = Med` over the shipped
`h = Med / ln K`, on an isotropic Gaussian, and validated both laws against Ba et al. Cor. 4.
`investigation8/exp04_lnk.py` puts the same comparison on the real posteriors: K = 400, 2000
iterations, two starts, everything else identical.

**Prediction, written before the run:** *it transfers but disappoints — the MAGI posteriors are
anisotropic over four to six orders of magnitude and a single adaptive scalar `h` still cannot
serve them, so I expect well under 39× and nothing near the floor.* That is what happened.

| system | start | `h = Med/lnK` (shipped) | `h = Med` (Ba et al.) | gain |
|---|---|---|---|---|
| fn | `fit()` | 6.836 (83.3× floor), R 0.019 | 2.813 (34.3×), R 0.178 | **2.43×** |
| fn | cold | 7.646 (93.2×), R 0.020 | 3.909 (47.7×), R 0.169 | 1.96× |
| lorenz | `fit()` | 5.942 (75.9×), R 0.021 | 2.812 (35.9×), R 0.144 | 2.11× |
| lorenz | cold | 6.075 (77.6×), R 0.023 | 3.393 (43.3×), R 0.144 | 1.79× |
| hiv | `fit()` | 8.554 (79.4×), R 0.145 | 3.799 (35.3×), R 0.187 | 2.25× |
| hiv | cold | 18.26 (169×), R 0.016 | 15.61 (145×), R 0.048 | 1.17× |

**1.8–2.4× in energy on five of six cells, 5–9× in Stein R, 1.2–1.8× in the KS statistic, and
never worse.** Not 39×: the isotropic law's prediction does not survive contact with an
anisotropic posterior, where the stiff directions collapse far below what either law says
(stiff-var 0.0005 → 0.0035 on fn, both a long way from 1).

### The change is made

`msvgd/msvgd.py`, `MSVGD.pairwise_distance`, one line, with the citation and the measured effect in
the comment:

```python
return L2sq, jnp.where(h <= 0, median, h)      # was: median / jnp.log(k)
```

`investigation8/exp04b_verify.py` runs the same configuration through the shipped `MSVGD.solve`
after the change and gets energy **2.8138** (34.30× floor, R 0.1777) against exp04's own
`h = Med` measurement of 2.8131 (34.29×, R 0.1778) — agreement to four digits.

Two things to be honest about.

* **It is not a fix.** Both conventions leave the ensemble 34–48× its Monte-Carlo floor. The
  change is worth making because it is free, strictly better in every cell tested, and replaces a
  rule with no analysis by one with a theorem attached — not because it rescues anything. The
  bandwidth that matters is a large fixed one (§2).
* **It changes mitosis too.** `_mitotic_split` calibrates its jitter to `h/2`, so offspring are now
  scattered `√(ln K) ≈ 2.4×` more widely. That is consistent — the jitter is meant to match the
  kernel's implicit Gaussian variance and the kernel is what changed — and `exp04b` confirms the
  50→100→200 schedule still runs, but it is a behaviour change and anyone with tuned mitosis
  settings should know.

### Why the isotropic Gaussian behaves differently — and investigation 7 §6 survives

If a fixed bandwidth has no correct fixed point on the MAGI posteriors, investigation 7 §6's
isotropic result (`N(0, I_325)`, K = 400, variance ratio 0.976 at `h = 100·h*`, converging from
starts spanning 0.05× to 2×) is immediately suspect: it was measured at 5000 iterations.
`investigation8/exp02d_iso_long.py` re-runs it to 200,000 iterations, which by the rescaling is
equivalent to `h = 100·h*` for ~10⁷:

| h/h* | var ratio at 3,000 | at 200,000 | from 0.25× start | from 1.0× start |
|---|---|---|---|---|
| 1 | 0.3462 | **0.3462** | 0.3462 | 0.3462 |
| 3 | 0.5845 | **0.5845** | 0.5845 | 0.5845 |
| 10 | 0.8111 | 0.8151 | 0.8151 | 0.8151 |
| 30 | 0.9256 | 0.9286 | 0.9286 | 0.9286 |

**Perfectly stable, and identical from both starts, over 67× more iterations** — and reproducing
investigation 7's `exp08` Part 2 values (0.34621, 0.58455) to four digits. **Investigation 7 §6
stands: on an isotropic target a fixed bandwidth really does have a genuine attractor, and it
rises toward 1 with h.**

So the difference is the **anisotropy**, and §2's own data show the mechanism. On fn at
`h = 100·h0` after 500,000 iterations the energy distance is only 2.67× the floor — the bulk of the
directions are fine — while stiff-var is 0.048. One scalar bandwidth can hold the directions whose
scale it matches and cannot hold the rest, so on a posterior spanning four to six orders of
magnitude in scale the stiff directions keep draining however long you wait, and there is no `h`
at which all of them are held at once. On `N(0, I)` there is only one scale, so one `h` suffices
and the attractor is real.

That also explains why the transient window exists at all: starting from a correct ensemble, the
stiff directions drain on a timescale set by `h`, and for a while the ensemble is still close
enough to right for every score to pass.

## 5. What "below the floor" means — settled in §1

Investigation 7 left open whether an energy distance below the K-particle floor meant the ensemble
was right or merely quasi-uniform. §1 above settles it in both directions, and the answer is not
the one either reading anticipated:

* **The ambiguity was real and worse than stated.** `exp01b` constructs an ensemble with a 40-fold
  misallocation of variance across directions that scores Stein R = 1.000 and energy 1.06× the
  floor. Energy distance in d ≈ 300 is essentially a radial statistic; it is superb at scale errors
  and nearly blind to angular ones, and Stein R is a trace statistic with the same blind spot.
* **But the specific ensembles at issue are genuinely right.** Re-scored with `stiff-var`,
  worst-band and KS (`exp01c`), the large-`h` ensembles match the reference on the stiffest 10% of
  directions to 1–6%, and their marginal quantile calibration is at or below the KS floor on hiv
  and lorenz. Read at 2000 iterations they are correct by every instrument available.

So the honest statement is: **investigation 7 §9's conclusion was right, its evidence was
insufficient, and its framing ("a fixed bandwidth is a fix") was wrong** — §2 shows those
ensembles are a transient.

## 6. Conclusions

### The five open questions

**1. A reference-free bandwidth rule.** "Raise h until Stein R ≈ 1" costs 1.00–1.18× the oracle's
energy distance on all three systems, inside the 2× bar, even when it picks a bandwidth 30× from
the oracle's — the energy curve is flat over a decade at its bottom, and where R cannot resolve h
neither can the energy. My prediction that the rule would be degenerate from a `fit()` start was
**wrong**: Prodigy rescales its step to compensate for the `1/h` shrinkage, so the ensemble keeps
moving. §2.5 then reframes what the rule is doing — it is a *stopping* rule, not a bandwidth rule.

**2. Sampler or polisher?** Neither, as posed — and §2's first answer was wrong and is retracted in
§2.5. There **is** a fixed point at a fixed bandwidth: two starts a factor of 16 apart in variance
meet at `whsd² = 0.145` on fn by 50,000 iterations and hold it to three digits for 450,000 more.
What it is not is the *target*: it puts the variance in the wrong directions (median direction
0.95, stiffest 10% 0.048 at `h = 100·h0`). The cause is anisotropy, established cleanly by
`exp05c` on two exact Gaussians differing only in covariance — the isotropic one equilibrates at
0.58/0.82/0.93 for `h = 3/10/30 h*`, the anisotropic one at 0.086/0.33/0.57, both stable, both
reached from either start. Investigation 7 §6's isotropic result is confirmed to 200,000
iterations (`exp02d`). Ten times the bandwidth still buys exactly ten times the delay.

**3. Profiled SVGD.** Stable in K on fn and lorenz — 0.24–0.52× the K-particle floor at K = 16, 64
and 256 against `fit()`'s 1.46–2.52× — and the cheap end works: K = 16 at 5.5–5.8× `fit()`'s wall
clock is 4.5–5× better. Honest price **6–12× `fit()`**, not investigation 7's 20–80× (that was a
contended CPU). It **fails at p = 5**: HIV ties `fit()` at K = 256 and costs 113×. `exp06` turns
`K ≳ eᵖ` into a test over seven values of p: the exponential rate is confirmed to within 8–10%
(`ln K_crit = 0.86p + 1.45` at the 0.8 level) with a constant factor of ~4 the law omits, and the
isotropic table predicts HIV's measured variance ratio at K = 16 to three digits (0.335 vs 0.334).

**4. Deleting the `1/ln K`.** Transfers at **1.8–2.4×, not 39×** — as predicted. Better in all six
cells, 5–9× on Stein R, never worse. Change made in `msvgd/msvgd.py` with Ba et al. Cor. 4 and the
measured effect in the comment, verified through the shipped `MSVGD.solve` to four digits. Not a
fix: both conventions leave the ensemble 34–48× its floor.

**5. "Below the floor" — right, or quasi-uniform?** Right, but the question exposed a defect in the
instruments: an ensemble with a 40-fold misallocation of variance scores **Stein R = 1.000 and
energy 1.06× the floor**, because both are trace statistics and the energy distance in d ≈ 300 is
effectively radial. Re-scored with `stiff-var`, worst-band and KS, the ensembles at issue are
genuinely correct — and the *shipped default* is worse than the energy distance suggested (0.13%
of the variance in HIV's stiffest directions).

### The headline: preconditioning works, and investigation 7 §3 was measuring a bug

`investigation7/msvgd7.py`'s preconditioned gradient applied the metric twice —
`jax.grad` through `x₀ + Lz` already returns `Lᵀ∇ₓ log p`, and the code multiplied by `Lᵀ` again.
**Every `precond` row in investigation 7 §3 is invalid.** Fixed and verified against the analytic
chain rule to 1.9e-15.

With the fix, whitened SVGD at a fixed bandwidth and a **fixed small step** (not Prodigy, which is
unstable in whitened coordinates) has a genuine attractor reached from 4×-narrow, correct and
4×-wide starts alike, and — the substantive point — **its error is a uniform scale deficit rather
than a misallocation**: stiff-var 0.748 against `whsd²` 0.719 on fn, a 1.04× mismatch, where the
unpreconditioned dynamics gives a 20× mismatch. Since a uniform deficit is one number and Stein R
measures exactly that number without a reference, rescaling by `1/√R` corrects it:

| | Stein R | energy × floor | stiff-var | whsd² | KS/floor | max\|θ err\| |
|---|---|---|---|---|---|---|
| fn, equilibrium | 0.702 | 2.42 | 0.748 | 0.719 | 1.87 | 0.242 |
| fn, **rescaled by 1/√R** | 1.074 | **0.68** | **1.065** | **1.024** | 1.89 | 0.242 |
| lorenz, equilibrium | 0.653 | 2.92 | 0.696 | 0.694 | 1.64 | 0.401 |
| lorenz, **rescaled by 1/√R** | 1.074 | **0.96** | **1.066** | **1.063** | 2.10 | 0.401 |
| hiv, equilibrium | 0.538 | 6.78 | 0.573 | 0.548 | 2.00 | 0.067 |
| hiv, **rescaled by 1/√R** | 1.000 | **0.57** | **1.065** | **1.017** | **0.80** | **0.067** |

This is a complete reference-free procedure with a genuine fixed point, which §1's stopping rule
is not, and it works on all three systems. On HIV it is unambiguous — energy 0.57× the floor, KS
**below** its floor, and the parameter error improves. On fn and lorenz the spread is fixed but the
parameter mean drifts (0.077 → 0.242 and 0.071 → 0.401), which the rescaling cannot touch; why two
systems drift and the third does not is the main thing left open.

### What I would now tell a user

1. **Do not run mSVGD on the joint MAGI posterior at the default bandwidth.** 34–93× the
   Monte-Carlo floor from any start, and the collapse is in the directions the data determines
   best.
2. **If you want the joint posterior's shape**, use the preconditioned recipe of §2.6: whiten by
   `H⁻¹`, fixed `h = 10·h*`, small fixed step, run to equilibrium, rescale by `1/√R`. At or below
   the K-particle energy floor, with a real fixed point and no stopping rule.
3. **If you want parameters, use `fit()`** — or profiled SVGD at `p ≤ 3` with `K ≈ 16–64`, which
   beats it 4–8× at 6–12× the cost. Never the reweighted kernel there.
4. **Score with something that is not a trace statistic** (`investigation8/metrics8.py`).

### What is still not settled

* **Why the parameter mean drifts on fn and lorenz but not HIV.** All three reach an equilibrium
  whose spread the rescaling fixes; two of them move θ away from the reference in the process and
  the third moves it closer. That is the single unexplained thing left in §2.6, and it decides
  whether the recipe is usable for MAGI's actual purpose.
* **The step size matters and was not swept properly.** At `h = 10·h*` Prodigy equilibrates at
  0.498 and SGD at 0.01 equilibrates at 0.748 — the discrete map's fixed point depends on the step,
  and the continuous-flow limit was not established.
* **Why the parameter mean drifts** under the preconditioned dynamics. The rescaling fixes the
  spread and leaves this untouched, and it is the single thing standing between this recipe and
  being useful for MAGI's actual purpose.
* **`exp02c` on HIV is now complete** (both starts converge to 40–42× the floor, stiff-var 0.002,
  R 0.155 at `h = 10·h0`), so that gap from the previous version is closed.
* **hes1** remains excluded — no usable reference.

### Provenance

float64 throughout, `atol = rtol = 0`, K = 400 unless stated. V100 (`jax-cuda12`) and RTX 3090
(`magi` env), agreeing to four digits on a shared benchmark. References re-validated against the
current `tests.py` in §0; the `gauss_newton` and `profiled` optimisations made during this
investigation leave float64 output unchanged and none of the numbers here are float32.
