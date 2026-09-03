# Investigation 7 — mitosis-SVGD, re-examined after the GP fix

Investigation 4 §3.6 called mSVGD "a decisive negative result": started *at* a correct posterior
sample on FitzHugh–Nagumo it moved away, degrading the energy distance from 0.085 to 3.80
(density-reweighted kernel) or 6.89 (standard RBF) and driving the Stein-identity ratio from 1 to
0.24 / 0.019. The verdict was that mSVGD's fixed point on this problem is not the posterior.

Two things that verdict rested on have since changed:

1. **The GP hyperparameter fit was broken** (investigation 6 §8). `fit_phisigma` started BFGS at
   `phi1 = phi2 = sigma = 1` regardless of the data's units and, on three of the four benchmark
   systems, settled on lengthscales far below the grid spacing, so the state was unconstrained
   between observations. Fixed: scale-free parameterisation, lengthscale confined to
   `[2dt, span/4]`. HIV's Hessian condition number fell 4.1e17 → 4.0e2.
2. **The benchmark integrator** moved from forward Euler to RK4, so the data itself is different.

FitzHugh–Nagumo's GP fit happened to be fine, so the old number may survive — but the posterior
geometry has changed and the other three systems were never tested. This investigation measures
what is actually true now.

Code in `investigation7/`. Systems and references are reused, not rebuilt:
`investigation5/setup5.py` (via the `setup7.py` symlink) and `investigation5/ref5_*.npz`,
96,000 NUTS draws each, rebuilt after the GP fix.

> **Reading order.** This was written as it happened, so two of its own conclusions are
> overturned later in the document. Sections 2-5 establish that the failure is real, is not about
> MAGI, and follows `Var/Var_target = ln(K)/d`; **section 6 then shows the cause is the
> median-heuristic bandwidth rather than the dimension**, section 9 tests that on the real
> posteriors, section 10 places it against the literature (the `1/d` scaling is a known theorem;
> the `ln K` is not), and section 11 gives the one setting in which mSVGD beats the incumbent.
> Section 13 is the summary. If you want the answer rather than the route, read 13 first.

## 0. Ground rules

**hes1 is excluded from every quantitative statement.** Its reference is R-hat 1.76 with 13%
divergences. The three usable references are fn (R-hat 1.006, 1.4% divergent), hiv (**1.0001, 0%**)
and lorenz (1.007, 1.1%).

**Marginal standard deviations are not a score.** That is the specific trap this problem sets: the
investigation-4 reweighted-kernel run held every marginal sd at 0.991 of the reference — visually
perfect credible intervals — while sitting 45× the energy-distance floor from the target, because
SVGD's collapse is anisotropic and no marginal sees it. The sd ratio is still reported below,
purely to show it stays uninformative.

What is scored (`investigation7/harness7.py`):

* **energy distance in Mahalanobis coordinates** — whiten ensemble and reference subsample by the
  reference covariance, then `2E|X−Y| − E|X−X'| − E|Y−Y'|`, always **with its floor**: the same
  statistic between two disjoint 2000-draw halves of the reference subsample. A second floor at
  the ensemble's own particle count K is reported too, since a K-particle ensemble cannot beat the
  Monte-Carlo error of K exact draws.
* **Stein-identity ratio** `R = −(1/(k·dim)) Σ (x_i − x̄)·s(x_i)`, 1 under the target.
* **max |θ error|** in reference sd, against the `half_mean` floor.
* **the whitened covariance spectrum** — all eigenvalues are 1 under the target. This is the
  direct read of anisotropic collapse, and it is new here: investigation 4 inferred the collapse
  from R alone.

## 1. Calibration: the diagnostics, and the "known-good" starting ensemble

`investigation7/exp00_smoke.py`. Before measuring anything, check the instruments.

**Stein R's sign convention is right and its target really is 1.** Evaluated on the reference
subsample itself — which is by construction distributed as the target — R comes out at
**0.9996 (fn), 1.0008 (hiv), 0.9981 (lorenz)**. The harness's own expression and
`MSVGD._stein_R(particles, -gradient)` agree to all printed digits, so the two conventions are
consistent and either may be quoted.

**`fit()` is a legitimate known-good start.** All three pass their own gate (ESS 63–76% of 512,
k̂ < 0.3, no null directions) and land at or near the reference floor:

| | dim | p | fit() s | energy | floor (2000 v 2000) | max\|θ err\| | θ floor | Stein R |
|---|---|---|---|---|---|---|---|---|
| fn | 325 | 3 | 12.2 | 0.1036 | 0.0366 | 0.0472 | 0.0100 | 1.263 |
| hiv | 608 | 5 | 28.6 | 0.0489 | 0.0446 | 0.0491 | 0.0081 | 1.000 |
| lorenz | 306 | 3 | 9.4 | 0.0479 | 0.0317 | 0.0220 | 0.0405 | 1.020 |

HIV is at 1.1× the floor and lorenz at 1.5×; fn at 2.8× is the worst of the three, and its
Stein R of 1.26 says the profiled mixture is mildly *over*dispersed there. Either way these
ensembles are 40–80× closer to the reference than anything mSVGD produced in investigation 4,
which is what makes the fixed-point test meaningful.

### Two floors that were previously wrong or missing

* The floor **depends on the ensemble size**. A K-particle ensemble cannot beat the Monte-Carlo
  error of K exact draws. On fn: 2000 exact draws score 0.037 against the reference, but **400
  exact draws score 0.082**. Every K = 400 result below is judged against 0.082.
* The **raw whitened eigenvalue spectrum is useless at small K.** For 4000 reference draws in
  d = 325 the whitened covariance's eigenvalues already run 0.52 to 1.64 — exactly the
  Marchenko–Pastur band (1 ± √(d/n))² — purely from sampling noise, and at K = 400 the band is
  0.013 to 3.66. So a collapsed ensemble and an exact one are indistinguishable by that spectrum.
  Replaced by a **band profile**: the variance ratio (ensemble / reference) along the reference
  covariance's *own* eigenvectors, averaged in five equal-count bands from the softest direction
  to the stiffest. Those directions are fixed rather than estimated, so the ratio is unbiased at
  any K — 400 exact reference draws give 0.99, 0.99, 1.00, 0.99, 1.01.

## 2. The control that should have been run first

`investigation7/exp04_gaussian.py`. Before asking whether MAGI's corrected posterior is the
problem, remove MAGI. Two targets, both exact multivariate Gaussians in fn's dimension d = 325:
`N(ref_mean, ref_cov)` built from the reference itself, and plain `N(0, I)`. The score and
gradient are analytic, the correct answer is known to machine precision, and the ensemble starts
at K = 400 **exact draws**. 1000 iterations, Prodigy, as in investigation 4.

| target | variant | energy | ×floor | Stein R | band profile (soft → stiff) |
|---|---|---|---|---|---|
| N(ref_mean, ref_cov) | exact draws (start) | 0.0758 | 0.91 | 1.0001 | 1.01 1.00 0.99 1.01 1.00 |
| | standard RBF | 7.410 | 88.9 | 0.0195 | 0.10 0.00 0.00 0.00 0.00 |
| | reweighted | 5.111 | 61.3 | 0.2301 | 0.52 0.09 0.05 0.12 0.37 |
| | matrix-valued (diag Fisher) | 7.367 | 88.4 | 0.0199 | 0.10 0.00 0.00 0.00 0.00 |
| **N(0, I)** | exact draws (start) | 0.0758 | 0.91 | 1.0001 | 1.00 0.99 1.01 1.00 1.01 |
| | **standard RBF** | **7.438** | **89.2** | **0.0183** | **0.018 0.018 0.018 0.018 0.018** |
| | reweighted | 5.496 | 65.9 | 0.0573 | 0.058 0.057 0.057 0.057 0.057 |
| | matrix-valued | 7.169 | 86.0 | 0.0228 | 0.021 0.018 0.022 0.034 0.020 |

**SVGD started at exact draws from a standard normal in 325 dimensions collapses to 1.8% of the
correct variance in every direction.** Nothing about MAGI is involved: no ODE, no GP, no
conditioning, no non-Gaussianity, no anisotropy. The isotropic target is what a *perfect*
preconditioner would deliver, and SVGD fails on it exactly as badly (89× the floor, Stein R 0.018)
as on the correlated one.

Two things follow immediately.

* **Investigation 4's numbers were right but its diagnosis was wrong.** Its energies (6.89
  standard / 3.80 reweighted) and Stein R (0.019 / 0.239) are reproduced here to within a factor
  of ~1.1–1.3 *on a target that has nothing to do with MAGI*. What it called "anisotropic
  collapse" is not anisotropic: on `N(0, I)` the band profile is a flat 0.018 across every
  band. The collapse is uniform, and it is a function of the dimension — and, as section 6
  eventually pins down, of the median-heuristic bandwidth rule, which is what makes the dimension
  bite.
* **Preconditioning cannot be the fix**, and does not need to be tested further to know that. The
  entire purpose of a preconditioner is to turn the target into `N(0, I)`, and `N(0, I)` is the
  row that fails hardest. (Section 3 tests it anyway, with the exact Hessian, and it is the worst
  of the four kernels on fn.)

The reweighted kernel is again the milder failure (0.057 rather than 0.018), reproducing its
historical ranking, but 0.057 is not a sampler either.

## 3. The fixed-point test, redone

`investigation7/exp01_fixedpoint.py`. K = 400 particles initialised at `fit()` draws, Prodigy,
`atol = rtol = 0` so the iteration count is exactly what is asked for. Four kernels. The whole
trajectory is scored, not just the endpoint, so "slow" and "wrong" can be told apart.

Columns: `energy` and `thEnergy` are Mahalanobis energy distances over all d coordinates and over
θ alone; `moved` is the energy distance from the *starting* ensemble; `thErr` is max |θ error| in
reference sd; `sdrat` is the median marginal sd ratio, present only to show it lies; `band` is the
variance ratio from softest to stiffest reference direction.

### FitzHugh–Nagumo (d = 325, p = 3)

| variant | energy | thEnergy | moved | Stein R | thErr | sdrat | band profile |
|---|---|---|---|---|---|---|---|
| **start: fit() draws** | 0.137 | 0.0087 | 0 | 1.250 | 0.077 | 1.018 | 0.98 0.98 1.03 1.01 1.03 |
| standard, 100 it | 6.110 | 0.178 | 6.25 | 0.030 | 0.614 | 0.804 | 0.22 0.010 0.003 0.002 0.001 |
| standard, 1000 it | 6.847 | 0.674 | 6.95 | 0.019 | 1.012 | 0.692 | 0.14 0.008 0.002 0.001 0.001 |
| reweighted, 100 it | 3.992 | 0.018 | 4.08 | 0.243 | 0.167 | 0.962 | 0.48 0.20 0.16 0.17 0.20 |
| **reweighted, 1000 it** | 4.652 | 0.053 | 4.81 | 0.162 | 0.370 | **0.994** | 0.58 0.14 0.06 0.02 0.10 |
| matrix, 1000 it | 6.981 | 0.553 | 7.07 | 0.019 | 0.913 | 0.684 | 0.13 0.007 0.002 0.002 0.001 |
| precond (exact Hessian), 1000 it | 11.379 | 0.826 | 11.71 | 4.402 | 1.847 | 2.396 | 6.3 5.0 1.9 3.6 15.5 |
| **FLOOR: 400 exact draws** | **0.082** | **0.0055** | | **1.000** | **0.010** | 1.000 | 0.99 0.99 1.00 0.99 1.01 |

Investigation 4's numbers, on Euler data with the broken GP fit, were energy 6.89 / Stein R 0.019
for the standard kernel and 3.80 / 0.239 / sd ratio 0.991 for the reweighted one. **They reproduce
to within about 20%.** The GP fix and the integrator change did not touch this result at all.

The sd-ratio trap reproduces exactly too: the reweighted kernel ends at a median marginal sd of
**0.994** of the reference — credible intervals that would look perfect on any plot — while its
energy distance is 57× the floor, its Stein R is 0.16, and its variance along the stiffest
directions is 2–10% of what it should be.

### Lorenz (d = 306, p = 3)

| variant | energy | thEnergy | moved | Stein R | thErr | sdrat | band profile |
|---|---|---|---|---|---|---|---|
| **start: fit() draws** | 0.097 | 0.0113 | 0 | 1.017 | 0.071 | 0.984 | 0.99 0.98 0.99 1.03 1.02 |
| standard, 1000 it | 5.964 | 1.063 | 6.07 | 0.022 | 1.692 | 0.722 | 0.20 0.016 0.006 0.025 0.003 |
| **reweighted, 1000 it** | 3.838 | 0.030 | 3.94 | 0.268 | 0.191 | **1.031** | 0.65 0.22 0.24 0.18 0.14 |
| matrix, 1000 it | 6.104 | 1.049 | 6.21 | 0.021 | 1.641 | 0.724 | 0.18 0.016 0.006 0.024 0.003 |
| precond (exact Hessian), 1000 it | 1.187 | 0.306 | 1.32 | 0.262 | 0.384 | 1.410 | 0.75 0.10 0.03 0.15 1.22 |
| **FLOOR: 400 exact draws** | **0.078** | **0.0045** | | **1.000** | **0.041** | 1.000 | 1.00 1.00 1.00 1.00 1.01 |

Lorenz's reweighted-kernel endpoint, 3.838 with sd ratio 1.031, is almost numerically identical to
investigation 4's FitzHugh–Nagumo figure of 3.805 with sd ratio 0.991. The failure is the same
size on a chaotic system as on a benign one, which is itself evidence that it has nothing to do
with the posterior's shape.

`moved` exceeds `energy` in every row: the ensemble does not merely fail to improve, it travels
further from its correct starting point than its final distance to the target — it walks past.

### Preconditioning with the exact Hessian does not help, and can hurt

The `precond` rows run SVGD in coordinates whitened by `H^-1` at the MAP, which is the constant
matrix-valued kernel `K(x,y) = Σ k_Σ(x,y)` with Σ the exact Laplace covariance. On lorenz it is
the best of the four (1.19 against 3.84–6.10) and still 15× the floor. On fn it is the **worst**,
and fails in the opposite direction: Stein R 4.40, marginal sds 2.4× too wide, and 6–15× too much
variance in the stiffest bands. Section 2 already explained why this was never going to work — a
perfect preconditioner produces `N(0, I)`, and `N(0, I)` is the target SVGD fails on hardest.

### HIV (d = 608, p = 5) — the best-conditioned system, and the worst result

The reference here has R-hat 1.0001 and **zero divergences**, the Hessian condition number is
4.0e2, and `fit()` starts the ensemble at 1.03× the K = 400 floor with Stein R exactly 1.000.
There is nothing left to blame.

| variant | energy | thEnergy | Stein R | thErr | sdrat | band profile |
|---|---|---|---|---|---|---|
| **start: fit() draws** | 0.111 | 0.0148 | 1.000 | 0.096 | 1.003 | 0.98 1.02 1.00 1.03 1.04 |
| standard, 1000 it | 6.602 | 1.218 | 0.083 | 0.346 | 0.084 | 0.40 0.001 0.001 0.001 0.013 |
| reweighted, 1000 it | 4.452 | 0.648 | 0.167 | 0.290 | 0.098 | 0.59 0.004 0.002 0.002 0.244 |
| matrix (diag Fisher), 1000 it | 1.3e6 | 1.1e6 | 2.3e14 | 2.6e5 | 1945 | **diverged** |
| precond (exact Hessian), 1000 it | 1.739 | 9.186 | 0.974 | 2.845 | 0.255 | 0.25 0.41 0.74 0.03 4.35 |
| **FLOOR: 400 exact draws** | **0.108** | **0.0056** | **1.000** | **0.0081** | 1.000 | 1.00 1.01 1.01 1.00 0.99 |

The diagonal-Fisher matrix kernel **diverges outright** on HIV, whose coordinates span 30 to 1e5 —
worth recording, since that variant is the one carried in `magi_msvgd/dev tests/`. The
exact-Hessian preconditioner gets the *global* Stein R to 0.974, which looks like success, and its
θ error is 2.8 reference sd against a floor of 0.008: R averaged over 608 coordinates is dominated
by the state block and says nothing about the 5 that matter. That is a second way the diagnostics
can mislead, alongside the sd ratio.

## 4. The same test with the optimizer removed

`investigation7/exp06_field.py`. Everything above runs Prodigy, so it could be argued that an
adaptive learning rate is amplifying a drift a smaller step would never express. The argument can
be closed off entirely.

In the mean-field limit SVGD's velocity vanishes identically when the ensemble law *is* the
target: `phi_p(x) = E_{y~p}[k(y,x) s(y) + grad_y k(y,x)] = 0` by integration by parts. So at exact
draws the field is finite-K sampling noise plus an O(1/K) systematic bias — and that bias is the
whole mechanism. Measured directly: take R = 8 independent K-particle subsamples of the
**reference**, evaluate `phi` at each, and report `d/dt log Var` band by band with a standard
error over replicates. No step size, no optimizer, no convergence test.

Standard kernel, `d/dt log Var` per unit flow time (± 1 se over 8 replicates):

| system | K | soft ← band → stiff | | | | |
|---|---|---|---|---|---|---|
| fn | 100 | −23.0 ±0.22 | −78.3 ±0.59 | −99.5 ±0.89 | −348 ±2.5 | −531 ±2.9 |
| fn | 400 | −5.90 ±0.02 | −20.2 ±0.06 | −25.5 ±0.06 | −89.2 ±0.37 | −136 ±0.36 |
| fn | 1600 | −1.51 ±0.003 | −5.11 ±0.014 | −6.41 ±0.007 | −22.5 ±0.03 | −34.4 ±0.04 |
| lorenz | 400 | −0.026 | −0.120 | −0.190 | −0.274 | −0.693 |
| hiv | 400 | −8.8e−6 | −3.0e−4 | −6.0e−3 | −1.4e−2 | **−37.5** |

Every band, every system, every K: **strictly negative, hundreds of standard errors from zero.**
The target is not a fixed point of the finite-K flow. Nothing about this depends on an optimizer.

Three further readings:

* **The bias is exactly O(1/K).** fn's stiffest band goes −531 → −136 → −34.4 as K goes
  100 → 400 → 1600: ratios 3.9 and 3.95 against a predicted 4. So more particles do help — at
  a rate that section 5 makes precise, and which is not nearly good enough.
* **The mean is unbiased.** The companion `drift` rows (motion of the ensemble mean) are
  consistent with zero everywhere: fn K = 400 band 1 gives +0.118 ± 0.097. SVGD at the target does
  not move the ensemble; it only squeezes it. The θ *errors* in section 3 are downstream of the
  squeeze, not a separate defect.
* **On the real posteriors the collapse IS anisotropic, and violently so.** The stiff-to-soft
  ratio of contraction rates is 23× on fn, 27× on lorenz, and **4.2 million×** on HIV. So
  investigation 4's word "anisotropic" describes what happens on these posteriors correctly —
  but section 2 shows anisotropy is not the *cause*, since removing it entirely (the `N(0, I)`
  row) does not help. Anisotropy decides which directions collapse first; the dimension decides
  that they collapse — or rather, as section 6 corrects, the dimension decides *how far*, given a
  bandwidth rule that tracks the ensemble.

**Cross-check on the driver.** `exp02_coldstart.py` first runs 50 Adam steps from the same start
through both `MSVGD.solve` and my own `msvgd7.run_svgd`, and reports a maximum relative difference
of **0.000e+00** — bit-identical. The kernel variants in exp01 and the field measurement above
therefore sit on the same code path as the shipped library.

## 5. The law: Var_SVGD / Var_target = ln(K) / d

`investigation7/exp07_scaling.py`, `exp08_law.py`. If the failure is dimensional, it should have a
law. Target `N(0, I_d)` throughout — isotropic, so nothing is left for a preconditioner to fix and
the equilibrium variance ratio is directly the number SVGD gets wrong. Started at exact draws, so
there is no burn-in to mistake for an equilibrium. Prodigy, 2000 iterations.

> **Read section 10 alongside this one.** The `Θ(1/d)` part of what follows is a known theorem
> (Ba et al., ICLR 2022), and it was measured here before that was checked. What is specific to
> this report is the `ln K` numerator, which belongs to the `h = Med/log K` bandwidth rule the
> library ships and not to the `h = Med` rule the theorem is stated for. `exp13_vs_theory.py`
> reproduces both, in the same harness, to within 1%.

Standard RBF kernel, mean variance ratio:

| d \ K | 50 | 100 | 400 | 1600 |
|---|---|---|---|---|
| 2 | 0.880 | 0.920 | 0.963 | 0.990 |
| 5 | 0.652 | 0.721 | 0.830 | 0.912 |
| 10 | 0.263 | 0.436 | 0.573 | 0.679 |
| 25 | 0.151 | 0.152 | 0.244 | 0.382 |
| 50 | 0.078 | 0.090 | 0.109 | 0.172 |
| 100 | 0.039 | 0.046 | 0.058 | 0.079 |
| 200 | 0.020 | 0.023 | 0.030 | 0.039 |
| 325 | 0.012 | 0.014 | 0.018 | 0.025 |
| 608 | 0.0064 | 0.0076 | 0.0098 | 0.0123 |

Multiply each entry by `d` and the table collapses onto a function of K alone:

| d \ K | 50 | 100 | 400 | 1600 |
|---|---|---|---|---|
| 50 | 3.915 | 4.512 | 5.452 | 8.602 |
| 100 | 3.912 | 4.605 | 5.774 | 7.924 |
| 200 | 3.912 | 4.582 | 5.914 | 7.756 |
| 325 | 3.917 | 4.601 | 5.956 | 8.072 |
| 608 | 3.890 | 4.605 | 5.978 | 7.469 |
| **ln K** | **3.912** | **4.605** | **5.991** | **7.378** |

**Var_SVGD / Var_target = ln(K) / d**, to three decimal places over an order of magnitude in d.
`exp08` sweeps K from 10 to 3200 at fixed d and confirms `(var·d)/ln K = 1.000, 0.998, …`.

### Why ln K, and what it means

`ln K` is not a coincidence: `MSVGD.pairwise_distance` sets the bandwidth by the median heuristic,
`h = median(||x−y||²) / ln K`. An ensemble with per-coordinate variance `v` in `d` dimensions has
`median ||x−y||² ≈ 2dv`, so `h = 2dv / ln K`; substituting the law gives **h = 2, independently of
d and K**. `exp08` reports the final bandwidth directly and gets 2.10–2.22.

So the mechanism is a feedback loop, not a property of the target: the ensemble contracts, the
median heuristic re-tightens the bandwidth around the contracted ensemble, which permits further
contraction, and the loop terminates when the bandwidth reaches twice the target's own variance
scale — by which point the ensemble occupies `ln(K)/d` of the target's volume per direction.

### What the law costs

Inverting it: reaching a fraction `f` of the correct posterior variance needs
`K = exp(f · d)` particles.

| system | d | K for f = 0.5 | K for f = 0.9 |
|---|---|---|---|
| fn | 325 | e^162 | e^292 |
| lorenz | 306 | e^153 | e^275 |
| hiv | 608 | e^304 | e^547 |

There is no particle budget **under the median heuristic**. This is not "mSVGD needs more
compute"; it is "mSVGD as the library ships it cannot be run on a problem of this dimension, at
any budget". Conversely the law says where it *is* usable as shipped: `d ≲ 5` reaches 0.83–0.91 at
K = 400–1600, and `d = 2` reaches 0.99.

The qualifier matters, and section 7 is about it: the median heuristic turns out to be the whole
of the problem, and replacing it changes the answer.

### The reweighted kernel obeys the same law with a better constant

| d | 50 | 100 | 200 | 325 | 608 |
|---|---|---|---|---|---|
| var·d (K = 400) | 24.98 | 28.36 | 19.79 | 17.56 | 17.74 |

Roughly `20/d` instead of `ln(K)/d` — a factor of 3–4 better and still `O(1/d)`. That is the
entire content of its historically better ranking, and it changes nothing qualitative: 20/325 is
6% of the correct variance.

### The parameter block is not spared

A reader might accept that the 300–600 state coordinates are mangled and still hope the p = 3–7
parameters — the only thing most users read — come out. They do not. Energy distance restricted to
θ and whitened by the reference's own θ covariance, at 1000 iterations, as a multiple of the
K = 400 θ floor:

| | fn | lorenz | hiv |
|---|---|---|---|
| start: fit() draws | 1.6× | 2.5× | 2.6× |
| standard RBF | **123×** | **236×** | **217×** |
| reweighted | 9.6× | 6.6× | 116× |
| max\|θ err\| in ref sd, standard | 1.01 | 1.69 | 0.35 |
| max\|θ err\| in ref sd, reweighted | 0.37 | 0.19 | 0.29 |
| (θ error floor) | 0.010 | 0.041 | 0.008 |

The θ marginal is the *least* damaged part of the ensemble and it is still an order of magnitude
past the floor at best. The reason is section 4's drift row: SVGD does not move the ensemble mean
at the target, so a short run leaves θ looking plausible; the damage arrives as the collapse in
the state block distorts the joint and drags θ with it, which is why the θ error *grows* with
iteration count (fn standard: 0.61 → 0.88 → 1.00 → 1.01 at 100/200/500/1000 iterations).

### Two ways the law could have been an artefact, both closed off

**The optimizer.** All of the above uses Prodigy, which adapts its own learning rate. Plain
gradient descent at a fixed 1e-2 for 2000 steps looks completely different — variance ratio 0.90
at d = 50, K = 400 where Prodigy gives 0.11 — which would be alarming if it meant the two
discretisations had different fixed points. They do not; SGD is simply nowhere near converged.
Run out 100× longer (`exp08` Part 3b):

| iterations | 2,000 | 8,000 | 40,000 | 200,000 | law: ln(400)/50 |
|---|---|---|---|---|---|
| var ratio, d = 50, K = 400 | 0.897 | 0.675 | 0.219 | **0.124** | **0.120** |

It converges to the law's value. Prodigy is not creating the collapse, only reaching it sooner.

The optimizer-free version of the same statement is `exp08` Part 3a: the SVGD velocity field
evaluated at exact draws from `N(0, I_d)` contracts the variance at

    d/dt log Var  =  −1.974e−2 (K = 100),  −4.981e−3 (K = 400)

**independently of d** — measured at d = 10, 50, 100 and 325, agreeing to three digits, with
standard errors of 1e−6 — and the two values are −2/K to within 1%. The target is not a fixed
point of the K-particle flow; it is contracted at rate 2/K in every direction, at every dimension.
The dimension does not set the rate; it sets where the contraction stops, because the equilibrium
is `ln(K)/d`.

**The bandwidth.** Since the mechanism is the median heuristic, the obvious question is whether
fixing the bandwidth cures it. `exp08` Part 2 holds `h` at multiples of `h* = 2d/ln K` (the median
heuristic evaluated at the *target* rather than at the contracted ensemble), d = 325, K = 400:

| h / h* | 0.1 | 0.3 | 1 | 3 | 10 | 100 |
|---|---|---|---|---|---|---|
| var ratio | 0.071 | 0.161 | 0.346 | 0.585 | 0.811 | **0.976** |

which looks like a cure. It is not one, and `exp11_bandwidth_control.py` is the control: as
`h → ∞` the RBF kernel tends to the constant 1 and the update degenerates to
`phi(x_i) → mean_j s(x_j)`, a rigid translation that **cannot change the ensemble's shape at all**.
Started at exact draws, an update that does nothing scores perfectly. The control repeats the
sweep from starting ensembles whose variance is deliberately wrong (0.0025×, 0.25×, 4×) — a
sampler must converge to 1 from all of them, a frozen ensemble keeps whatever it began with.

## 6. Correction: the bandwidth rule is the whole of the problem (on a Gaussian)

The paragraph above predicted that a large fixed bandwidth would "work" only by freezing the
ensemble. **That prediction is wrong**, and the control that was written to demonstrate it
demonstrated the opposite. `investigation7/exp11_bandwidth_control.py`, target `N(0, I_325)`,
K = 400, 5000 iterations, bandwidth fixed at multiples of `h* = 2d/ln K = 108.5`, started from
ensembles at 0.05×, 0.25×, 1× and 2× the correct standard deviation:

| h / h* | start sd | var ratio | var / var(start) | energy | × floor |
|---|---|---|---|---|---|
| 1 | 0.25 | 0.346 | 5.5 | 1.364 | 17.4 |
| 1 | 1.00 | 0.346 | 0.35 | 1.365 | 17.4 |
| 3 | 1.00 | 0.585 | 0.58 | 0.431 | 5.5 |
| 30 | 0.25 | 0.926 | 14.8 | 0.049 | 0.63 |
| 30 | 1.00 | 0.926 | 0.93 | 0.049 | 0.62 |
| 100 | 0.25 | 0.976 | 15.6 | 0.042 | 0.54 |
| 100 | 1.00 | 0.976 | 0.98 | 0.042 | 0.53 |
| 100 | 2.00 | 0.976 | 0.24 | 0.042 | 0.54 |
| 300 | 1.00 | 0.992 | 0.99 | 0.041 | **0.52** |
| 1000 | 2.00 | 0.997 | 0.25 | 0.042 | 0.53 |

Read the `var / var(start)` column: at h = 30 h* the ensemble started at a quarter of the correct
spread **expands by a factor of 14.8** and lands on the same 0.926 as the one started correctly,
and the one started at twice the spread contracts by 0.24 to the same place. It is an attractor,
not a frozen state.

And the energy distance is **0.52–0.63 of the K = 400 floor** — the ensemble is not merely as good
as 400 exact draws, it is *better*, which is what a quasi-uniform deterministic ensemble should be
against iid Monte-Carlo error. In 325 dimensions, with 400 particles.

So the honest statement of sections 2–5 is narrower than it looked:

> SVGD's collapse in high dimensions is a property of the **median-heuristic bandwidth**
> `h = median(||x−y||²)/ln K`, not of SVGD, not of the dimension, and not of the target. The rule
> is roughly **1.5–2.5 orders of magnitude too small** at d = 325, and because it is measured on
> the ensemble rather than on the target it tightens as the ensemble contracts, which is the
> positive feedback that carries it all the way to `ln(K)/d`.

Everything measured in sections 2–5 remains exactly as reported, because the median heuristic is
the only bandwidth rule `msvgd.MSVGD` implements and `bandwidth=-1` is its default. But the
diagnosis changes from "SVGD is unusable here" to "SVGD's default bandwidth is unusable here", and
those imply completely different remedies.

Two caveats, and section 7 tests both:

* This is an **isotropic** target. A single scalar `h` has one length scale; the MAGI posteriors
  span several orders of magnitude in scale (section 4's contraction rates differ by 4.2e6 across
  bands on HIV), so there may be no scalar `h` that serves them.
* `h*` was computed from the known target. A user has no reference. The practical analogue is the
  median heuristic evaluated on a *good* ensemble, which `fit()` supplies — but the multiplier
  needed is 30–300, and nothing so far says how to find it without a reference.

## 7. From a cold start, the way a user would run it

`investigation7/exp02_coldstart.py`, driving the **shipped** `msvgd.MSVGD` (the driver check
above confirms my own loop is bit-identical to it). The initial ensemble is built exactly as the
old `MAGI.solve` built it: `particles_init + 0.2 * N(0, I)`, `atol = rtol = 0`, and the
`k_schedule` row is the mitosis that gives mSVGD its name.

All timings in this table were taken with three systems running concurrently on 32 cores, so they
are roughly 2–10× the uncontended figures (`fit()` alone is 12.2 s on fn, 9.4 s on lorenz and
28.6 s on hiv, from `exp00`). The *ratios within a row block* are meaningful; the absolute numbers
are not.

| system | method | energy | thEnergy | Stein R | max\|θ err\| | sec |
|---|---|---|---|---|---|---|
| **fn** | `fit()` | **0.137** | **0.0087** | 1.250 | **0.077** | 30 |
| | adam 0.1, K = 200, 2000 it | 8.211 | 0.698 | 0.016 | 1.008 | 34 |
| | prodigy, K = 200, 2000 it | 9.217 | 0.697 | 0.016 | 1.004 | 32 |
| | prodigy, K = 50→100→200, mitosis | 7.185 | 0.732 | 0.015 | 1.021 | 25 |
| | prodigy, K = 400, 5000 it | 7.770 | 0.655 | 0.020 | 1.003 | 272 |
| | *floor / NUTS* | *0.082* | *0.0055* | *1.000* | *0.010* | *160* |
| **lorenz** | `fit()` | **0.097** | **0.0113** | 1.017 | **0.071** | 19 |
| | adam 0.1, K = 200, 2000 it | 6.381 | 1.133 | 0.018 | 1.703 | 30 |
| | prodigy, K = 200, 2000 it | 6.710 | 1.021 | 0.016 | 1.636 | 34 |
| | prodigy, mitosis 50→100→200 | 6.279 | 1.126 | 0.018 | 1.714 | 23 |
| | prodigy, K = 400, 5000 it | 6.581 | 0.956 | 0.027 | 1.597 | 292 |
| | *floor / NUTS* | *0.078* | *0.0045* | *1.000* | *0.041* | *116* |
| **hiv** | `fit()` | **0.111** | **0.0148** | 1.000 | **0.096** | 334 |
| | adam 0.1, K = 200, 2000 it | 20.36 | 5.076 | 0.007 | 3.728 | 31 |
| | prodigy, K = 200, 2000 it | 18.66 | 5.017 | 0.010 | 3.650 | 25 |
| | prodigy, mitosis 50→100→200 | 19.49 | 4.663 | 0.013 | 3.683 | 22 |
| | prodigy, K = 400, 5000 it | 16.79 | 15.25 | 2.738 | 2.726 | 198 |
| | *floor / NUTS* | *0.108* | *0.0056* | *1.000* | *0.008* | *252* |

Three things worth saying plainly.

* **Cold-start mSVGD lands in the same place as the fixed-point runs** — energy 6–9 on fn and
  lorenz, 19–20 on hiv, against floors of 0.08–0.11. Whether it is started at the answer or at a
  guess makes no difference, which is what "the fixed point is elsewhere" means operationally.
* **Mitosis does not help, and neither does more of anything.** 50→100→200 with splits scores the
  same as 200 flat; 400 particles for 5000 iterations (10× the compute) scores the same as 200 for
  2000. The law of section 5 says why: the equilibrium moves as `ln K`, so doubling the particles
  buys 15%. On HIV the longest run is the only one that does not simply collapse, and it goes the
  other way instead — Stein R 2.74, 15× too much variance in the stiffest band, θ energy 15.2
  against 5.0 for the short runs. Longer is not better there either.
* **The cost argument is not there either.** mSVGD is not faster than `fit()`, which is 60–190×
  more accurate, and it is within a factor of 5 of the *reference chain itself* — 96,000 NUTS
  draws in 116–252 s. On these problems, at this dimension, there is no budget at which mSVGD is
  the right thing to run.

## 8. Does investigation 4's number reproduce? Almost exactly.

Same system (FitzHugh–Nagumo), same K = 400, same kernels, same optimizer, same fixed iteration
counts — but different data (forward Euler → RK4) and a different GP hyperparameter fit.

| | inv 4 energy | inv 7 energy | inv 4 Stein R | inv 7 Stein R | inv 4 sd ratio | inv 7 sd ratio |
|---|---|---|---|---|---|---|
| start | 0.0847 | 0.1372 | — | 1.250 | 0.987 | 1.018 |
| standard, 200 it | 6.4898 | **6.5127** | 0.0245 | **0.0236** | 0.825 | 0.737 |
| standard, 1000 it | 6.8948 | **6.8469** | 0.0190 | **0.0194** | 0.748 | 0.692 |
| reweighted, 200 it | 3.8101 | 4.2472 | 0.2705 | 0.1965 | 0.962 | 0.965 |
| reweighted, 1000 it | 3.8046 | 4.6522 | 0.2394 | 0.1621 | 0.9909 | 0.9940 |
| reference floor | 0.0331 | 0.0366 | | | | |

The standard kernel agrees to **three significant figures** across a change of integrator and a
change of GP hyperparameter fit. The reweighted kernel is 10–20% worse on the corrected data but
the same qualitatively, including the 0.99 sd ratio.

That is exactly what sections 2 and 6 predict: the endpoint is set by the bandwidth rule and the
dimension, and d = 325 on both datasets. **Investigation 4's negative result survives intact.**
The GP bug and the integrator, which invalidated most of investigations 5 and 6, had no bearing
on this one. `investigation7/exp05_whatchanged.py` runs the 2×2 of {rk4, euler} × {fixed GP, old
GP} explicitly; the reproduction above already answers the question, so exp05 is confirmation
rather than evidence.

## 9. Does the bandwidth fix transfer to the real posteriors? Mostly — with one condition.

`investigation7/exp12_bandwidth_magi.py`. Same fixed-point setup as section 3 (K = 400 started at
`fit()` draws), but with the bandwidth held at multiples of `h0`, the median heuristic evaluated
**on the `fit()` ensemble** — which is a quantity a user actually has, unlike section 6's `h*`.
2000 iterations. Each setting is also run from the same ensemble shrunk 4× about its mean, so
"converged there" and "stayed where it started" remain distinguishable.

FitzHugh–Nagumo, `h0 = 0.402`:

| variant | energy | thEnergy | Stein R | max\|θ err\| | band profile |
|---|---|---|---|---|---|
| start: fit() draws | 0.137 | 0.0087 | 1.250 | 0.077 | 0.98 0.98 1.03 1.01 1.03 |
| start: shrunk 0.25× | 5.435 | 0.461 | 0.064 | 0.077 | 0.06 ×5 |
| h = 1·h0 (≈ the default) | 6.322 | 0.619 | 0.027 | 1.013 | 0.19 0.009 0.002 0.001 0.001 |
| h = 10·h0 | 1.449 | 0.102 | 0.326 | 0.460 | 0.86 0.47 0.36 0.07 0.01 |
| h = 100·h0 | 0.194 | 0.0052 | 0.754 | 0.050 | 0.88 0.78 0.79 0.66 0.59 |
| **h = 1000·h0** | **0.069** | 0.0351 | **0.966** | 0.256 | 0.93 0.89 0.93 0.90 0.89 |
| h = 1000·h0, from 0.25× start | 5.399 | 1.008 | 0.067 | 0.936 | 0.07 ×5 |
| **FLOOR: 400 exact draws** | **0.082** | **0.0055** | **1.000** | **0.010** | ~1.00 |

Lorenz, `h0 = 135.4`:

| variant | energy | thEnergy | Stein R | max\|θ err\| | band profile |
|---|---|---|---|---|---|
| start: fit() draws | 0.097 | 0.0113 | 1.017 | 0.071 | ~1.00 |
| h = 1·h0 | 5.325 | 1.034 | 0.032 | 1.723 | 0.28 0.024 0.008 0.037 0.004 |
| h = 10·h0 | 1.569 | 0.280 | 0.277 | 1.070 | 0.92 0.37 0.20 0.42 0.07 |
| h = 100·h0 | 0.110 | 0.0128 | 0.801 | 0.191 | 0.96 0.82 0.77 0.80 0.72 |
| **h = 1000·h0** | **0.052** | 0.0090 | **0.944** | 0.106 | 0.97 0.93 0.92 0.94 0.93 |
| h = 1000·h0, from 0.25× start | 5.215 | 0.992 | 0.066 | 1.081 | 0.06 ×5 |
| **FLOOR: 400 exact draws** | **0.078** | **0.0045** | **1.000** | **0.041** | ~1.00 |

**The fix transfers.** At 1000× the default bandwidth, mSVGD on the actual 306- and
325-dimensional MAGI posteriors reaches an energy distance of 0.052 and 0.069 against K = 400
floors of 0.078 and 0.082 — **below the floor on both**, i.e. indistinguishable from 400 exact
draws, or slightly better as a quasi-uniform ensemble should be. Stein R goes from 0.03 to 0.94
and 0.97. The band profile is flat at 0.89–0.97 instead of collapsing three orders of magnitude.
Anisotropy is evidently not the obstacle it looked like: one scalar bandwidth, if large enough,
serves posteriors whose contraction rates differ 27-fold across bands.

**The condition is the starting ensemble.** Every `0.25×` row fails: at h = 1000·h0 the repulsion
term carries a factor `2/h`, so an ensemble that starts too narrow expands far too slowly to
recover in 2000 iterations (in the isotropic case of section 6, 5000 iterations were enough). So
this is not a standalone sampler at these settings; it is a *refinement of an ensemble that is
already about the right size* — which `fit()` provides, in 9–29 s.

**And it does not beat `fit()` where it matters.** The full-dimensional energy improves on the
start (0.137 → 0.069 on fn, 0.097 → 0.052 on lorenz), but the θ block gets *worse*: θ energy
0.0087 → 0.0351 on fn, max |θ err| 0.077 → 0.256 against a floor of 0.010. On lorenz θ is roughly
unchanged. So the honest summary is: with a hand-tuned bandwidth 1000× the library default,
mSVGD can polish `fit()`'s state block to below the Monte-Carlo floor while slightly degrading its
parameters, for 30–40 s. Whether that is worth anything depends on wanting the states rather than
the parameters.

**How the multiplier would be chosen without a reference is not settled.** 1000× was found by a
sweep scored against the reference. Section 6 gives a principled candidate — pick `h` so that the
kernel's own length scale matches the target's, rather than the ensemble's — and Stein R is a
reference-free quantity that tracks the sweep monotonically (0.027 → 0.33 → 0.75 → 0.97 on fn),
so tuning `h` upward until R ≈ 1 is an obvious and untested procedure.

## 10. Against the literature — and the one-line change that buys 39×

The collapse itself is known. Ba, Erdogdu, Ghassemi, Sun, Suzuki, Wu & Zhang, *Understanding the
Variance Collapse of SVGD in High Dimensions*, ICLR 2022 (OpenReview `Qycd9j5Qp9J`; there is no
arXiv version) prove, for an isotropic Gaussian target in the proportional limit
`n, d → ∞`, `γ = d/n > 1`, Gaussian RBF kernel, under the **plain** median heuristic
`σ = sqrt(Med{||x_i−x_j||²}/2)`, i.e. `h = 2σ² = Med` with **no** `1/log n` factor:

> **Corollary 4.** `v_SVGD = (e − 1)^{-1} γ^{-1}`, i.e. **0.5820 · n/d**, whereas MMD-descent
> gives `v_MMD = 1`.

(Their result is conditional on an unproven near-orthogonality assumption on the particle
configuration, which they verify numerically rather than prove.)

**So the `1/d` scaling is theirs, not this report's, and section 5 should not have presented it as
a discovery.** What differs is the `n`-dependence and the constant, and the reason is that the two
bandwidth rules are different:

| | bandwidth | law | at K = 400, d = 800 |
|---|---|---|---|
| Ba et al. 2022, Cor. 4 | `h = Med` | `0.582 · n/d` | 0.291 |
| this report, sections 5 & 8 | `h = Med / ln K` (what `msvgd` ships) | `ln(K) / d` | 0.0075 |

Same `1/d`; **linear in `n` versus logarithmic in `K`**, which at K = 400 is a factor of 39 and at
K = 4000 a factor of 281. Neither law dominates the other as a matter of form -- they describe two
different algorithms -- but on any particle budget anyone would actually use, `0.582 n` is far
larger than `ln K`, so the shipped rule is the worse of the two.

`msvgd.MSVGD.pairwise_distance` implements the *other* convention — Liu & Wang (NeurIPS 2016)
recommend `h = Med / log n`, and that is what the library uses. So the two laws should differ, and
`investigation7/exp13_vs_theory.py` runs both conventions through the same harness:

| convention | K | d | γ = d/K | v measured | Ba et al. `0.582 n/d` | this report `ln K / d` |
|---|---|---|---|---|---|---|
| Ba (h = Med) | 400 | 800 | 2 | **0.29026** | **0.29099** | 0.00749 |
| msvgd (h = Med/lnK) | 400 | 800 | 2 | **0.00750** | 0.29099 | **0.00749** |
| Ba (h = Med) | 400 | 1600 | 4 | **0.14513** | **0.14549** | 0.00374 |
| msvgd (h = Med/lnK) | 400 | 1600 | 4 | **0.00371** | 0.14549 | **0.00374** |
| Ba (h = Med) | 200 | 400 | 2 | 0.28953 | 0.29099 | 0.01325 |
| msvgd (h = Med/lnK) | 200 | 400 | 2 | 0.01325 | 0.29099 | **0.01325** |
| Ba (h = Med) | 100 | 400 | 4 | 0.14404 | 0.14549 | 0.01151 |
| msvgd (h = Med/lnK) | 100 | 400 | 4 | 0.01150 | 0.14549 | **0.01151** |

Both laws are confirmed to within 0.3–1%, on the same code, at the same time. That validates the
harness against published theory, and it isolates a fact worth acting on:

> **The `1/ln K` in `pairwise_distance` costs a factor of `0.582 K / ln K` in variance fidelity.**
> At K = 400 that is **39×**; at K = 4000 it is 281×. Deleting it — one character class in one
> line — takes the variance ratio at K = 400, d = 800 from 0.0075 to 0.290.

Both remain `O(1/d)`, so this is not a cure; it is a 39× improvement in the constant, for free,
and it moves the library from a bandwidth rule nobody has analysed to one that has a theorem
attached to it.

### Where sections 6 and 9 sit relative to that theory

Ba et al. also test *fixed* bandwidths and conclude against them:

> **Corollary 6** (IMQ kernel): "When σ = √d, `v_SVGD < 1` and is decreasing as γ > 1 increases.
> When σ = 1, `v_SVGD → 0` … at a rate of `d^{-1/3}`." … "This corollary suggests that the IMQ
> kernel with fixed bandwidth is not a remedy to the variance collapse problem."

`σ = √d` means `h = 2d` — exactly the bandwidth that would be "correct" if the variance were 1.
The measurements in sections 6 and 9 do not contradict that, because they are in a regime Ba et
al. did not test: **h ≈ 100 d**, two orders of magnitude beyond `σ = √d`. The reason that regime
behaves differently is a two-line expansion. For `k = exp(−||x−y||²/h)` with `h ≫ 2dv`, expanding
to first order and using `(1/K)Σ_j x_j x_jᵀ ≈ v I` on a centred ensemble,

    phi(x_i)  ≈  −x̄  +  (2/h)(1 − v) x_i  +  O(h^{-2}),

whose fixed point is **v = 1**, approached at rate `2(1 − v)/h`. So a sufficiently large fixed
bandwidth has the *right* fixed point and merely reaches it slowly — which is exactly the pattern
measured: correct variance from every start in section 6, and in section 9 a correct answer from a
well-scaled start but not (within 2000 iterations) from a 4×-too-narrow one.

The `Θ(1/d)` collapse is established; the following are not, on the evidence of the literature
search behind the citations here, and should be read as this report's own measurements rather than
as known results:

* the `ln(K)/d` law for the `h = Med/log n` convention — the scaling in `d` is Ba et al.'s, the
  `ln K` in place of their `0.582 n` is not — and the 39× penalty it carries against the plain
  median heuristic;
* the observation that the median heuristic's equilibrium pins `h → 2` in units where the target
  has unit variance (`exp08` measures 1.79–2.26 across d = 50–325 and K = 10–3200);
* that a fixed `h ≳ 100 d` restores the variance to 0.93–0.99 and the *energy distance* to below
  the K-particle Monte-Carlo floor, in d = 325 and on the real MAGI posteriors.

Other relevant references, for the record: Zhuo et al., *Message Passing SVGD*, ICML 2018
(arXiv:1711.04425) prove only repulsive-force decay, not the variance claim they are usually cited
for; Korba et al., NeurIPS 2020 (arXiv:2006.09797) give mean-field rates only; Priser, Bianchi &
Salim, ICLR 2025 (arXiv:2406.11929) show *noisy* SVGD provably avoids the collapse; Liu & Wang,
*SVGD as Moment Matching*, NeurIPS 2018 (arXiv:1810.11693) show SVGD with **linear** kernels
estimates Gaussian means and variances exactly — i.e. the collapse belongs to the RBF/median
setup, not to SVGD's fixed-point structure.

On the density-reweighted kernel: Huang, Dong & Fang (OpenReview `k2CRIF8tJ7Y`) was **rejected**
from ICLR 2023 (scores 3/6/8), no reviewer raised variance or dispersion, and no source claims it
mitigates variance collapse. Its better ranking here (`≈20/d` against `ln K/d`) is an empirical
observation of this report, not a published property. The nearest published analogue is
h-SVGD, *Convergence Aspects of Hybrid Kernel SVGD*, TMLR 2025 (OpenReview `JZkbMSQDmD`), which
uses different kernels in the drift and repulsion terms, improves variance empirically, and
**proves it does not converge to the target in the mean-field limit** — a useful reminder that
"fixes the variance" and "has a convergence guarantee" are independent claims here.

## 11. Where mSVGD actually works: the profiled p-dimensional marginal

`investigation7/exp09_profiled_svgd.py`. Section 5's law is a statement about dimension, and
MAGI's dimension is a choice. The joint is `p + nD` = 306–608. The parameter block is `p` = 3–7,
where exp07 measures a variance ratio of 0.83–0.99 — SVGD's *good* regime. And
`profiled.ProfiledPosterior` already supplies the p-dimensional profiled log marginal

    log p̂(θ) = log p(θ, X*(θ)) − ½ log det H_XX(θ),

whose inner solve is a `jax.lax.scan` of Cholesky solves and is therefore differentiable. So SVGD
can be run on it directly, with autodiff gradients through the profile.

FitzHugh–Nagumo, p = 3, K = 64 particles, 400 iterations, cold start from Laplace θ draws at the
joint MAP (no knowledge of the answer):

| variant | θ energy | × floor | max\|θ err\| | sd ratio | sec |
|---|---|---|---|---|---|
| start: Laplace θ draws at the MAP | 0.6440 | 21.4 | 1.190 | 0.907 | — |
| `fit()` (the incumbent), K = 64 | 0.0758 | 2.52 | 0.239 | 1.091 | 96 |
| **profiled SVGD, standard RBF** | **0.00894** | **0.30** | **0.0118** | 0.931 | 1884 |
| profiled SVGD, reweighted | 0.01479 | 0.49 | 0.0159 | 1.010 | 1130 |
| FLOOR: 64 exact reference draws | 0.03013 | 1.00 | 0.0100 | 1.000 | — |

**Both kernels beat 64 exact draws**, by 2–3×, and the parameter mean lands at 0.012 reference sd
against the reference chain's own half-vs-half floor of 0.010. Same library, same default
bandwidth, same median heuristic — the only change is that the 322 state coordinates are
integrated out analytically instead of being carried as particles.

Lorenz, p = 3, same settings:

| variant | theta energy | x floor | max abs theta err | sd ratio | sec |
|---|---|---|---|---|---|
| start: Laplace theta draws at the MAP | 1.3682 | 50.8 | 1.953 | 0.843 | -- |
| `fit()`, K = 64 | 0.0416 | 1.55 | 0.175 | 1.010 | 9.5 |
| **profiled SVGD, standard RBF** | **0.00960** | **0.36** | **0.0267** | 0.962 | 792 |
| profiled SVGD, reweighted | 0.8751 | 32.5 | 1.013 | 1.579 | 665 |
| FLOOR: 64 exact reference draws | 0.02693 | 1.00 | 0.0405 | 1.000 | -- |

The standard kernel replicates: below the K = 64 floor, and a theta error of 0.027 against the
reference chain's own half-vs-half floor of 0.041, i.e. below the resolution of the reference
itself. **The reweighted kernel fails here** -- 32.5x the floor, barely moved from a start at
50.8x, with the spread 58% too wide. So the ranking of the two kernels inverts once the dimension
is small enough for either to work: the density reweighting exists to fight collapse, and where
there is no collapse to fight it is only a distortion. That is worth stating because every
comparison earlier in this report had the reweighted kernel ahead.

That is what the dimension law predicts, and it is the constructive half of this investigation:
**mSVGD's problem on MAGI is not mSVGD, it is being asked to carry a 300-to-600-dimensional state
vector as particles when that state vector has a closed-form conditional.**

The cost is the catch. 1884 s against `fit()`'s 96 s in the same contended run (12 s uncontended),
because every gradient backpropagates through three Gauss-Newton steps and a Cholesky of the
322×322 state block, for each of 64 particles, 400 times. Roughly 20–30× `fit()` for a 2.5–8×
better answer on a p = 3 problem. Whether that trade is worth taking is a judgement, but it is a
real trade, which nothing else in this report offers.

## 12. Experiment 5, explicitly: neither the GP fix nor the integrator matters

`investigation7/exp05_whatchanged.py`. The 2x2 of {RK4, forward Euler} x {fixed GP fit, the
pre-fix `fit_phisigma` reconstructed read-only in `investigation7/oldgp.py`} on FitzHugh-Nagumo,
started at `fit()` draws, 1000 iterations. Scoring is deliberately reference-free, since only one
cell has a reference: Stein R, and the band profile taken against each cell's own Laplace
covariance.

| cell | l/dt | Stein R, standard | Stein R, reweighted | band profile, standard |
|---|---|---|---|---|
| rk4 / fixed GP | 13.2, 12.5 | **0.0194** | 0.189 | 0.17 0.05 0.03 0.006 0.000 |
| rk4 / OLD GP | 13.4, 12.5 | **0.0192** | 0.207 | 0.17 0.09 0.03 0.015 0.000 |
| euler / fixed GP | 13.3, 12.5 | **0.0193** | 0.187 | 0.17 0.05 0.03 0.006 0.000 |
| euler / OLD GP | 13.4, 12.5 | **0.0193** | 0.223 | 0.17 0.10 0.03 0.015 0.000 |

Stein R agrees to three significant figures in all four cells, and the band profiles are
indistinguishable. The premise is confirmed on the way past: FitzHugh-Nagumo's GP fit really was
unaffected by the bug (l/dt = 13.4 before the fix, 13.2 after, against the failures of 0.001-0.16
that investigation 6 sec. 8 found on HIV and Hes1), and the integrator changes nothing either.

So the answer to "was investigation 4's result an artefact of the GP bug or of the data change?"
is **neither**. It was a property of the algorithm's bandwidth rule and the problem's dimension,
both of which are unchanged by either fix.

## 13. Conclusions

### Does investigation 4's negative result survive?

**Yes, unchanged.** On FitzHugh-Nagumo the standard kernel reproduces to three significant figures
(energy 6.847 vs 6.895, Stein R 0.0194 vs 0.0190) and the reweighted kernel to 10-20%, including
its 0.99 sd ratio. The 2x2 of {RK4, Euler} x {fixed GP, old GP} in section 12 gives Stein R
0.0192-0.0194 in every cell. Neither the GP hyperparameter fix nor the integrator change touches
this result. It extends to the two systems investigation 4 never tested, and HIV -- the
best-conditioned of the four, with a zero-divergence reference -- is the worst of the three.

### But the diagnosis in investigation 4 was wrong, in a way that matters

It called the failure **anisotropic collapse**, and inferred that from Stein R alone. Two
measurements contradict that as a *cause*:

* On a plain `N(0, I)` in the same dimension, with no MAGI anywhere in it, SVGD started at exact
  draws collapses to a flat 1.8% of the correct variance in every direction -- 89x the floor, the
  same as on the real posterior. Perfect isotropy does not help, which is also why preconditioning
  with the exact Hessian does not (section 3: it is the worst of the four kernels on fn).
* The collapse follows `Var/Var_target = ln(K)/d` on an isotropic Gaussian, over d from 50 to 608
  and K from 10 to 3200, to three decimals.

Anisotropy is real on the MAGI posteriors -- section 4 measures contraction rates differing by
23x on fn, 27x on lorenz and 4.2e6 on HIV -- but it decides *which* directions go first, not
whether they go. The cause is the bandwidth rule and the dimension.

### The mechanism, and the part that changes what should be done

The median heuristic sets `h = median(||x-y||^2) / ln K` **from the ensemble**, so as the ensemble
contracts the bandwidth contracts with it, and the loop terminates at `h ~= 2` in units where the
target has unit variance (measured 1.79-2.26 across all d and K). Two consequences:

1. **Deleting the `1/ln K`** -- the Liu & Wang (2016) convention the library follows -- recovers
   the `0.582 n/d` law that Ba et al. prove for the plain median heuristic. At K = 400 that is a
   **39x** improvement in variance fidelity for a one-line change (`exp13`, both laws confirmed to
   within 1% in the same harness). Still `O(1/d)`, so not a cure.
2. **Fixing the bandwidth at `h ~ 100 d`** -- two orders of magnitude beyond the `sigma = sqrt(d)`
   that Ba et al. tested and rejected -- gives the flow the *correct* fixed point `v = 1`
   (heuristic expansion in section 10, confirmed by measurement). On `N(0, I_325)` with K = 400
   the ensemble converges from 0.05x to 2x the correct spread to a variance ratio of 0.93-0.99 and
   an energy distance **0.52x the K-particle Monte-Carlo floor**. On the real 306- and
   325-dimensional MAGI posteriors, at `h = 1000 h0`, energy 0.052 and 0.069 against floors of
   0.078 and 0.082 -- also below the floor.

I wrote the prediction that a large fixed bandwidth would only work by freezing the ensemble, and
the control written to demonstrate that disproved it. That is recorded in section 6 as it happened.

### What I would actually recommend

1. **Do not use mSVGD on the joint MAGI posterior as it ships.** From a cold start it lands at
   80-190x the energy floor on all three systems, mitosis does not help, 10x the compute does not
   help, and it is not faster than `fit()`, which is 60-190x more accurate, nor much faster than
   the 96,000-draw NUTS reference itself.
2. **If mSVGD is kept, run it on the profiled p-dimensional marginal.** Section 11: on fn and
   lorenz, SVGD on `log p_hat(theta)` with 64 particles reaches **0.30x and 0.36x** the 64-draw
   energy floor and a theta error of 0.012-0.027 against reference floors of 0.010-0.041 --
   better than `fit()` by 2.5-8x on the same quantity. It costs 20-80x `fit()`'s wall clock, and
   the reweighted kernel must not be used there (it fails on lorenz, 32x the floor).
3. **Change the default bandwidth rule, or expose it.** The `1/ln K` costs 39x at K = 400 for
   nothing, and `bandwidth` is already an argument -- what is missing is any indication that the
   default is catastrophic above a few dimensions. Stein R tracks the bandwidth sweep
   monotonically (0.027 -> 0.33 -> 0.75 -> 0.97 on fn) and needs no reference, so "raise h until
   R ~ 1" is an obvious tuning rule.
4. **`_stein_R` should be reported by default and its target value documented.** It was the one
   diagnostic in the old code that saw this failure, and it reads 0.02 where the marginal standard
   deviations read 0.99.

### Traps, restated because they cost time here

* **Marginal standard deviations are worthless as a score.** The reweighted kernel ends at sd
  ratio 0.994 (fn) and 1.031 (lorenz) while sitting 50-60x the energy floor. Credible intervals
  drawn from those ensembles would look perfect.
* **Stein R averaged over all coordinates is dominated by the state block.** On HIV the
  exact-Hessian preconditioner reports R = 0.974 with a theta error of 2.8 reference sd against a
  floor of 0.008.
* **Every distance needs a floor at the ensemble's own K.** On fn, 2000 exact draws score 0.037
  and 400 exact draws score 0.082 -- a 2.2x difference that is pure Monte-Carlo error.
* **The whitened eigenvalue spectrum is uninformative at K ~ d.** For 4000 reference draws in
  d = 325 it already spans 0.52-1.64 from Marchenko-Pastur alone. The band profile (variance ratio
  along the reference covariance's own fixed eigenvectors) is unbiased at any K and was the
  diagnostic that made the anisotropy question answerable.

### What is not settled

* **How to choose the bandwidth without a reference.** The 1000x multiplier in section 9 came from
  a sweep scored against the reference. The Stein-R rule above is untested.
* **Whether the large-h regime is a sampler or only a polisher.** In section 9 it converges from a
  correctly-scaled start but not, within 2000 iterations, from one 4x too narrow. Section 6 shows
  it does converge on an isotropic target given 5000 iterations, so this may be only a budget
  question -- but it was not measured on the MAGI posteriors.
* **Whether below-the-floor energy at large h means the answer is right, or only that the
  ensemble is quasi-uniform.** Both are consistent with the data.
* **hes1** is untouched throughout: R-hat 1.76, 13% divergences, no usable reference.
* **The profiled-SVGD result is two systems and one particle count.** K = 64, 400 iterations, no
  sweep over either, and no run on HIV (p = 5, nD = 603, so roughly 8x the per-gradient cost).

### Note on the code that moved underneath this

All measurements here are float64 (`setup7.build(..., dtype=jnp.float64)`; `fit()` inherits it
because `put()` has been called). The `gauss_newton.py` Cholesky-ridge change made during this
investigation takes the float64 ridge from a fixed 1e-12 to `4096 * eps` = 9.1e-13, so nothing
above is affected by it; the reported float32 concern does not touch any number in this report.

