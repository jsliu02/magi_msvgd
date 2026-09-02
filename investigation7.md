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
| | *floor / NUTS* | *0.108* | *0.0056* | *1.000* | *0.008* | *252* |

Three things worth saying plainly.

* **Cold-start mSVGD lands in the same place as the fixed-point runs** — energy 6–9 on fn and
  lorenz, 19–20 on hiv, against floors of 0.08–0.11. Whether it is started at the answer or at a
  guess makes no difference, which is what "the fixed point is elsewhere" means operationally.
* **Mitosis does not help, and neither does more of anything.** 50→100→200 with splits scores the
  same as 200 flat; 400 particles for 5000 iterations (10× the compute) scores the same as 200 for
  2000. The law of section 5 says why: the equilibrium moves as `ln K`, so doubling the particles
  buys 15%.
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

