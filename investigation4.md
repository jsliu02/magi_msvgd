# Investigation 4 — obtaining a faithful posterior sample, given a fast and accurate MAP

Investigations 2 and 3 were built on a MAP with gradient norm ~1.3e3. The Gauss–Newton solver now
reaches ‖∇‖ ~1e-7 in 0.05 s, which changes what is worth trying: every Laplace-metric method
inherits the quality of the mode, and several earlier conclusions were artefacts of a bad one.
This investigation restarts from a correct MAP and asks what the fastest defensible route to a
faithful posterior is, prioritising speed, safety and theoretical justifiability.

**Headline.** The plain Laplace approximation `N(x_MAP, H⁻¹)` is wrong by **a full posterior
standard deviation** on two of the three FitzHugh–Nagumo ODE parameters, while its interval widths
are within a few percent — so nothing about the output looks wrong. A deterministic mean correction
costing **~1 s** cuts those errors to 0.01–0.09 sd (91–99% of the error removed), and comes with a
reference-free rule that correctly declines to apply it on the two settings where it would hurt. The covariance
was never the problem. The residual error is measurable *without a reference*, and there is a
checkable condition on the ODE under which no approximation is needed at all.

**Scope.** The quantitative claims hold at the baseline data density (41 observations), where the
reference is validated against an independent gold standard. As the data thin the method degrades,
and the point of §4 is that it says so on its own.

---

## 0. Master table (baseline setting, d = 325)

Mean error in posterior sd per dimension, against the validated reference; cost is wall clock after
JIT warm-up. The floor is what a second independent reference run would score.

| method | cost | bias | verdict |
|---|---|---|---|
| `N(x_MAP, H⁻¹)` — plain Laplace | 0.35 s | 0.1012 | mis-centred by ~1 sd on 2 of 3 ODE parameters |
| + third-order mean correction | 0.74 s | 0.0268 | good here, but hurts at 2 of 4 settings |
| + low-rank VI mean (§3.2) | 1.1 s | 0.0359 | its value is the certificate, not the mean |
| **+ midpoint of the two** | **1.4 s** | **0.0234** | **recommended: best and most robust** |
| + profile variance rescale, m=40 (§3.1) | 13 s | 0.0268 | correct, no gain on the joint covariance |
| preconditioned NUTS, 400×150, target 0.95 | 144 s | 0.0056 vs gold | exact; beats gold's own half-vs-half floor |
| mSVGD, reweighted kernel (§3.6) | 20 s | — (energy 3.80) | moves *away* from a correct starting point |
| reference floor | — | 0.0062 | |

The whole investigation in one line: **~1.4 s of arithmetic removes an error that a 144 s exact
sampler is otherwise needed to avoid, and the same arithmetic tells you when not to trust it.**

---

## 1. What the error actually is

Scored against the 8-chain 64,000-draw NUTS gold standard, with gold's own half-vs-half agreement
as the noise floor:

| quantity | `N(x_MAP, H⁻¹)` | + third order | gold half (floor) |
|---|---|---|---|
| θ_b mean error | **−0.971 sd** | −0.010 sd | +0.017 sd |
| θ_c mean error | **+1.032 sd** | +0.093 sd | −0.008 sd |
| θ_a mean error | +0.246 sd | +0.034 sd | +0.009 sd |
| trajectory max mean error | 1.015 sd | 0.115 sd | 0.042 sd |
| θ sd ratios | 1.00 / 1.01 / 0.94 | 0.99 / 1.01 / 0.94 | 1.00 / 0.99 / 1.01 |
| trajectory sd ratio (median) | 0.983 | 0.985 | 0.993 |

This is the kind of error that does not announce itself: the credible intervals have very nearly
the right *width*, so nothing looks wrong, but two of them are centred a full standard deviation
away. The correction is the standard third-order Laplace term

    E[x] ≈ x* − ½ H⁻¹ ∇ tr(H⁻¹ ∇²U),

evaluated with the exact Hessian from the Gauss–Newton machinery. It costs one gradient of a
scalar built from a Hessian — 0.39 s at d = 325, jitted.

This section characterises the error at baseline, where the third-order term is an excellent fix.
It is *not* the final recommendation: across all four settings it makes matters worse at two of
them, and the estimator that survives is its midpoint with the VI mean of §3.2, gated on the rule
of §4.2. Read §4.2 before using anything here.

Aggregate, calibrated metrics (bias = ‖Σ_g^(-1/2)(μ_q−μ_g)‖/√d, Förstner = affine-invariant
covariance distance, KL against the gold Gaussian):

| | bias | trace | Förstner | KL |
|---|---|---|---|---|
| `N(x_MAP, H⁻¹)` | 0.1008 | 1.0067 | 0.2243 | 4.86 |
| + third order | **0.0271** | 1.0067 | 0.2243 | 3.33 |
| + profile scale (m=40) | 0.0271 | 0.9890 | 0.2237 | 3.31 |
| floor (noiseless vs full chain) | 0.0042 | 1.0001 | 0.0767 | 0.48 |

### Calibrating the floors mattered

The raw covariance numbers are meaningless without a floor. In d = 325, affine-invariant metrics
are dominated by the error in *gold's own* covariance estimate. Splitting gold's 8 chains 4-and-4
gives Förstner 0.1538; inverting that against synthetic draws puts gold's effective sample size at
**27,791** (43% of its 64,000 draws), from which the floor for a noiseless approximation scored
against the full chain is **0.0767**. For scale, uniformly inflating gold's own covariance by 1.25×
scores 0.2231 — statistically indistinguishable from what the Laplace covariance scores. Any
comparison reported without this calibration is not interpretable.

---

## 2. The covariance is not the problem (and chasing it is a trap)

Three independent lines all say the mean is the whole story.

**The residual covariance error is real but hides in the null space of every marginal.** Scoring
the Laplace covariance against each half of gold separately gives per-direction log-variance errors
correlated at **0.975** — genuinely real, not noise (the pure-noise control is 0.852). But it is
concentrated in ~20 directions holding 17% of the posterior variance:

| variance-share bin | share | rms log-ratio |
|---|---|---|
| top 5 directions | 78.5% | 0.040 |
| next 20 | 17.3% | **0.245** |
| next 100 | 3.0% | 0.039 |
| remaining 200 | 1.2% | 0.023 |

Those 20 directions are combinations that cancel out of every θ marginal and every pointwise
trajectory band — which is why the user-facing table in §1 shows sds within 1–6% while the
aggregate Förstner sits at 3× its floor.

**A method that demonstrably fixes marginal variances does not move the joint covariance.** I built
Tierney–Kadane Laplace marginals along screened directions using a bordered Gauss–Newton profile
(details in §3). It cut the mean marginal-variance error from **4.68% to 1.66%**, fixing the two
worst directions from +8.7% → +0.8% and +6.7% → +1.0%. Its effect on the joint covariance was
Förstner 0.2243 → 0.2237, for 13 s. Not worth it.

**Sampled metrics cannot resolve the effect anyway.** At K=800 the standard error of `varwtd` is
comparable to the 3–5% effect being measured; it reported the scale correction as a slight *loss*
while the analytic comparison showed each corrected direction moving decisively toward truth. Any
analytic approximation should be scored analytically.

---

## 3. Methods tried

### 3.1 Screen-then-profile (works as designed; not worth its cost)

A reference-free screen ranks directions by how badly the quadratic model fails along them. Along
eigenvector `v_j` of `H`, with unit = 1 posterior sd,

    q_j = mean over t ∈ {−2,−1,1,2} of | U(μ + t·sd_j·v_j)/(t²/2) − 1 |,

zero iff the slice is exactly quadratic. It costs **4 batched log-density evaluations for all 325
directions (0.15 s)**, ranks them at Spearman **0.760** against the true error, and its top 5
capture **71%** of the variance-weighted Laplace error (top 10 → 85%).

The screen is a good *detector* and a terrible *estimator*: along the softest direction the slice is
185% steeper than quadratic at 1 sd, while the true marginal is only 8% too narrow. The gap is the
complement relaxing as you move, so what is needed is the profile, not the slice:

    log p_marg(z) = −U_prof(z) − ½ log det H_⊥(z),   U_prof(z) = min{ U(x) : vᵀ(x−μ) = z }.

Two identities make this cheap. The constrained Gauss–Newton step is the unconstrained step plus a
multiple of `A⁻¹v`, so it reuses the *same* Cholesky factor; and for unit `v`,
`det(NᵀHN) = det(H)·(vᵀH⁻¹v)`, so the restricted determinant never requires the 324-dimensional
basis to be formed. Cost: **0.31 s per direction.** Accurate (§2) — but pointless for the joint.

### 3.2 Gaussian variational inference (diverges naively; fixed by restricting it)

Minimising KL(q‖p) over Gaussians has stationarity conditions `E_q[∇log p] = 0` and
`Σ⁻¹ = E_q[−∇²log p]`, so the Laplace approximation is the zeroth iterate of a fixed point built
from quantities this pipeline already computes. The naive iteration **diverges**, for two reasons
with one cause:

- the mean step `Σ E_q[∇log p]` overshoots by ~1.6× (tau 0.2751 against a correct ~0.17), because
  the right Newton preconditioner is `E_q[H]`, not `H(x_MAP)`;
- the covariance step explodes because a 325×325 matrix average over 192 samples has
  O(√(d/n)) eigenvalue noise that inversion amplifies (trace 1.0067 → 2.1253 in one step).

Both are fixed by refusing to estimate what the samples cannot support. Restricted to the
m = 12 screened directions, `E_q[H]` is an m×m block — an easy estimate instead of an impossible
one — and since the subspace is spanned by eigenvectors of `H`, the corrected curvature is block
diagonal in `H`'s eigenbasis and inverts in closed form:

    A = H + V_S Δ V_Sᵀ,   A⁻¹ = Σ_{i∉S} v_i v_iᵀ/λ_i + V_S (diag(λ_S) + Δ)⁻¹ V_Sᵀ.

Only `V_SᵀH(x)V_S` is ever needed, so **12 Hessian-vector products per sample** replace a 325×325
assembly. The iteration then converges in 3 steps:

| iter | bias | certificate tau(step) |
|---|---|---|
| 0 (Laplace) | 0.1008 | — |
| 1 | 0.0388 | 0.1660 |
| 2 | 0.0340 | 0.0147 |
| 5 | 0.0339 | 0.0004 |

It also reveals *why* everything else was hard: **`E_q[H]/H(x_MAP)` is 6–10× in the soft
directions.** The average curvature over the posterior is an order of magnitude above the curvature
at the mode.

The VI mean (0.0339) is slightly *worse* than the third-order mean (0.0271): the reverse-KL
Gaussian optimum is genuinely a different point from the posterior mean. That turns out to be
useful — see §4.

**Antithetic pairing is load-bearing throughout.** The first attempt at `E_q[∇log p]` produced
garbage (tau 1.07 against a signal of 0.27) purely from Monte Carlo noise; the O(δ) term dominates
the variance and is odd, so evaluating at `μ ± δ` cancels it exactly. Spherical cubature is worse
than useless here — its 2d points sit at √d ≈ 18 posterior sds along a single axis, deep in the
region where the degree-6 polynomial dominates, and it diverged immediately.

### 3.3 Preconditioned NUTS (the reliable fallback)

With the exact Hessian as the metric, blackjax NUTS in whitened coordinates with an identity mass
matrix reaches the sampling floor:

| configuration | energy | varwtd | grad evals | wall clock | divergences |
|---|---|---|---|---|---|
| warm=50, n=5, K=400 | 0.0824 | 0.924 | 159 | 21 s | — |
| warm=200, n=50, K=400 | 0.0817 | 0.999 | 1622 | 74 s | 193 |
| warm=300, n=150, K=400, target 0.95 | — | — | — | 144 s | 33 |

The k=400 floor is ~0.083, so it is *at* the floor from the cheapest configuration onward. The 193
divergences at target_accept 0.8 are consistent with the 6–10× curvature variation found in §3.2;
tightening to 0.95 cuts them to 33.

**The 144 s configuration was validated against the independent 8-chain gold standard and beat
gold's own half-vs-half floor on every metric** — bias 0.0056 (floor 0.0095), Förstner 0.1358
(floor 0.1538), KL 1.50 (floor 1.90), θ means agreeing to 0.007–0.036 sd. Two unrelated sampler
configurations agreeing within noise is what licenses using this construction in settings where no
gold chain exists.

### 3.4 Combining the two mean estimates (a free improvement)

The third-order mean and the VI mean are both O(Λ)-accurate with *different* O(Λ²) residuals, so
their errors can partially cancel. Fitting the extrapolation `μ_3 + λ(μ_3 − μ_VI)` against the
reference gives an optimum at λ = −0.30 (baseline) and −0.50 (half) — negative in both cases,
meaning the truth lies *between* the two estimates. The optimum is not stable enough to tune, but
its tuning-free special case, the plain midpoint, is better than both estimators at both settings:

| setting | bias(third order) | bias(VI) | **bias(midpoint)** | floor |
|---|---|---|---|---|
| baseline | 0.0268 | 0.0359 | **0.0234** | 0.0062 |
| half | 0.0928 | 0.1010 | **0.0503** | 0.0081 |

At `half` this halves the error. It costs nothing beyond the VI solve already needed to compute the
`disagree` certificate. Two settings is thin evidence for a recommendation, so this is reported as
a consistent observation rather than the headline.

**A negative result worth recording.** I also tried a "self-consistent Laplace" step — recompute
`H` at `μ_3` and reapply the correction there. It diverges catastrophically (bias ~4e9), and the
reason is that the step is simply invalid: the third-order formula is derived *at the mode*, where
`∇U = 0`, so applying it at a non-stationary point drops the `−Σ∇U` term. Restoring that term
turns it into exactly the VI iteration of §3.2, so there is nothing new to be had this way.


### 3.5 As a sampler, not just as moments

Users need draws, not `(μ, Σ)`. Energy distance in Mahalanobis coordinates against the reference,
with the floor set by scoring one 2000-draw subsample of the reference against another:

| setting | `N(x_MAP, H⁻¹)` | + third order | + midpoint | floor |
|---|---|---|---|---|
| baseline | 0.1646 | 0.0403 | **0.0397** | 0.0331 |
| half | 0.2763 | 0.1382 | **0.0669** | 0.0332 |

At baseline the corrected Gaussian is at **1.2× the sampling floor** — at 2000 draws it is not
distinguishable from the reference chain. At `half` it is at 2× the floor: real residual error,
which the `disagree` certificate flagged in advance without seeing the reference.


### 3.6 mSVGD: a decisive negative result

Investigations 1–3 tried to make mSVGD represent this posterior faithfully. With a faithful
Gaussian now available in 0.5 s, the question changes: started *at* a correct answer, does mSVGD
stay there? A sampler whose fixed point is the posterior must.

| | energy | sd ratio | Stein R |
|---|---|---|---|
| corrected Gaussian (start, K=400) | 0.0847 | 0.987 | — |
| standard kernel, 200 iters | 6.4898 | 0.825 | 0.0245 |
| standard kernel, 1000 iters | 6.8948 | 0.748 | 0.0190 |
| reweighted kernel, 200 iters | 3.8101 | 0.962 | 0.2705 |
| reweighted kernel, 1000 iters | 3.8046 | **0.9909** | 0.2394 |
| reference subsample floor | 0.0331 | 1.000 | 1.000 |

It degrades the answer by 45–80× in energy distance and drives Stein's identity from 1 to
0.02–0.27. The reweighted kernel is the milder failure and is the more instructive one: it holds
the marginal standard deviations at **0.991 of the reference** — visually perfect credible
intervals — while the energy distance is 3.80 and Stein R is 0.24. This is the anisotropic collapse
documented in investigation.md, now shown in its cleanest form, since the starting point was known
to be correct and so the motion is unambiguously *away* from the target.

The conclusion is that on this problem mSVGD is not a sampler that converges slowly; its fixed
point is not the posterior. Any pipeline built on it inherits that, and no amount of better
initialisation helps. This is why the rest of this investigation abandons it.


---

## 4. Certifying the answer without a reference

Everything above needs gold to score. For deployment, the question is whether the method knows
when it is failing. Four computable certificates, none of which needs a reference:

| certificate | what it measures | cost |
|---|---|---|
| `dA` | `‖H_XX(θ,X₁) − H_XX(θ,X₂)‖/‖H_XX‖`; zero iff `p(X\|θ)` is exactly Gaussian | 2 Hessians |
| `q_max` | largest slice-curvature screen value | 4 batched log-p evals |
| `kappa_S` | max of `E_q[H]/H` over the screened subspace | 192 HVP batches |
| `disagree` | `‖μ_3rd − μ_VI‖` in posterior sd per dim | one VI solve |

`disagree` is the key one. The third-order mean is a Taylor truncation of the *posterior* mean; the
VI mean is the exact stationary point of the *reverse-KL Gaussian* objective. Neither is the truth
and they are derived from genuinely different arguments, so they can only agree closely if both are
near it. Measured against references:

| setting | q_max | kappa_S | tau_end | **disagree (no ref)** | bias MAP | bias 3rd | bias VI | floor |
|---|---|---|---|---|---|---|---|---|
| baseline | 4.95 | 10.4 | 0.0002 | **0.0344** | 0.1012 | 0.0268 | 0.0359 | 0.0062 |
| half (data thinned 2×) | 20.8 | 45.8 | 0.0003 | **0.0872** | 0.1390 | 0.0928 | 0.1010 | 0.0081 |
| quarter (thinned 4×) | 932 | 2121 | **5.21** | **3.3711** | 0.1972 | 0.3365 | 3.6425 | 0.0121 |

At baseline and half, `disagree` predicts the true bias to within 30% and 6% respectively. At
quarter everything fails at once and every certificate says so: `tau_end = 5.21` means the VI
iteration **did not converge at all** (against 2e-4 at baseline), `kappa_S = 2121` means the
average curvature over `q` is three orders of magnitude above the curvature at the mode, and
`disagree` is 100× its baseline value. The third-order correction there is worse than no correction
and VI diverges outright — but nothing has to be inferred from a reference to know that.

**Context for how hard `quarter` is.** The settings thin the observations, not the 161-point
discretisation grid: baseline has 41 observations of each of 2 states, half has 21, quarter has
**11**. So at quarter, 22 data points constrain 325 unknowns plus 3 ODE parameters, and the
posterior is correspondingly broad (tr Σ = 4.69 against 0.996 at baseline, min eig(H) = 0.387
against 4.94). This is close to an unidentified problem, not a regime where any Gaussian
approximation should be expected to work. The useful property is not that the method succeeds
there — it does not — but that it reports its own failure without being told.

**The references themselves degrade, and this limits what can be claimed.**

| setting | draws | sec | max R̂ | divergences | min eig(H) | tr(Σ) |
|---|---|---|---|---|---|---|
| baseline | 60000 | 144 | 1.0304 | 33 | 4.94 | 0.996 |
| half | 60000 | 507 | 1.1628 | 1334 | 2.65 | 1.424 |
| quarter | 60000 | 800 | **1.7654** | 715 | 0.387 | 4.686 |

At quarter, R̂ = 1.77 means the reference has not converged, so the "bias 3rd = 0.3365" figure is
measured against an unreliable target and should not be read as a quantitative claim — the
qualitative failure is established by the reference-free certificates alone. At half, R̂ = 1.16 is
marginal and those numbers are indicative rather than precise. Only baseline (R̂ = 1.03, and
independently validated against the 8-chain gold standard in §3.3) supports quantitative claims.

This is itself a finding about the fallback: **preconditioned NUTS at 400 chains × 150 draws does
not converge on the sparse-data settings either.** The recommendation to fall back to NUTS carries
the obligation to check its diagnostics rather than assume it worked; on `quarter` it needs far
longer chains than the configuration that suffices at baseline.

Sweeping the ODE's cubic coefficient α from the condition-(A) case to real FitzHugh–Nagumo shows
the whole suite is coherent and monotone:

| α | dA | kappa_S | third-order correction | disagree |
|---|---|---|---|---|
| 0.00 | 0.00e+00 | 1.18 | 0.00514 | 0.0012 |
| 0.25 | 2.30e-03 | 1.14 | 0.01024 | 0.0018 |
| 0.50 | 4.14e-02 | 4.20 | 0.11117 | 0.0379 |
| 0.75 | 8.38e-02 | 4.78 | 0.13592 | 0.0343 |
| 1.00 | 1.30e-01 | 10.40 | 0.17388 | 0.0344 |

At `dA = 0` exactly, `kappa_S = 1.18 ≈ 1` and `disagree = 0.0012`: every certificate independently
reports "Laplace is exact here", correctly. `dA` is computable **before any sampling or correction
is attempted**.

### 4.1 Is the certificate measuring the posterior or the knobs?

The low-rank VI solve has two choices made by judgement: subspace size `MS` and antithetic pair
count `NP`. If `disagree` moved with them it would be measuring the tuning, not the posterior.

| MS | NP | seed | bias(VI) | disagree | bias(midpoint) |
|---|---|---|---|---|---|
| 12 | 1024 | 0 | 0.0359 | 0.0344 | 0.0234 |
| 6 | 1024 | 0 | 0.0707 | 0.0638 | 0.0384 |
| **24** | 1024 | 0 | **0.0359** | **0.0344** | **0.0234** |
| 12 | 256 | 0 | 0.0366 | 0.0379 | 0.0229 |
| 12 | 2048 | 0 | 0.0353 | 0.0335 | 0.0232 |
| 12 | 1024 | 1 | 0.0349 | 0.0355 | 0.0229 |
| 12 | 1024 | 2 | 0.0375 | 0.0352 | 0.0240 |

`MS = 12` and `MS = 24` are **identical** — the screened subspace saturates, which independently
validates the screen: it finds a subspace that stops mattering once it is large enough. `MS = 6` is
materially worse, consistent with §3.1's finding that the top 10 directions carry 85% of the
variance-weighted error. `NP` from 256 to 2048 and three different seeds move `disagree` by only
±3%, so 256 pairs suffice and the seed dependence is negligible. This sweep was run at baseline only; the knobs were held fixed at `MS = 12`, `NP = 1024` for every other result in this document, so nothing reported here is the product of per-setting tuning.


### 4.3 Correction: score on theta, not on the 325-dimensional average

Section 4.2 concluded the midpoint was the estimator to use. That was scored on `bias`, an average
over all 325 coordinates -- 322 of which are trajectory states. MAGI exists to estimate the ODE
parameters, and on those three coordinates the ranking reverses. Largest |theta error| in
reference sd:

| setting | MAP | **third order** | midpoint | VI |
|---|---|---|---|---|
| baseline (validated) | 1.033 | **0.099** | 0.297 | 0.546 |
| half (R-hat 1.16) | 1.595 | **0.153** | 0.630 | 1.353 |
| noisy (R-hat 1.48) | 2.579 | 1.366 | **1.023** | 2.914 |
| quarter (R-hat 1.77) | 1.762 | **0.582** | diverged | -- |

The third-order mean is 3-4x better on theta at the two settings with usable references, while
losing only 13% on the all-coordinate average (0.027 vs 0.023 at baseline). That is the right
trade, and **the third-order mean is the default**. The midpoint remains available for callers
whose target is the trajectory rather than theta.

Two further consequences.

**The `noisy` verdict of §4.2 was also an aggregate artefact.** There the third-order correction
appeared to make things worse (0.300 -> 0.433 aggregate) but on theta it improves the largest
error from 2.579 to 1.366. Both are poor; the correction is not the reason.

**The gate may be conservative.** On theta the correction improved every setting tried, including
`quarter`, where the gate fires (1.762 -> 0.582). So the gate, calibrated on the aggregate, is
plausibly a false positive with respect to theta. `quarter`'s reference has R-hat = 1.77, so this
cannot be settled here; the gate is kept because reporting the mode is a safe fallback, and
`.mu3` is left on the result so the caller can look.

This is a correction to §4.2, which stands as written for the all-coordinate metric it used.

### 4.2 A decision rule, and a better estimator

At `quarter` and `noisy` the *uncorrected* Laplace mean beats the third-order-corrected one, so a
firing certificate must suppress the correction rather than merely flag it. Evaluating on all four
settings also shows the third-order mean is not the estimator to use:

| setting | \|Δ₃\| | \|Δ_VI\| | disagree | **disagree/\|Δ₃\|** | bias MAP | bias 3rd | **bias midpoint** |
|---|---|---|---|---|---|---|---|
| baseline | 0.174 | 0.162 | 0.0344 | **0.198** | 0.1012 | 0.0268 | **0.0234** |
| half | 0.314 | 0.286 | 0.0872 | **0.278** | 0.1390 | 0.0928 | **0.0503** |
| noisy | 0.701 | 0.646 | 0.1708 | **0.244** | 0.3000 | 0.4329 | **0.1699** |
| quarter | 1.671 | 3.812 | 3.3711 | **2.017** | 0.1972 | 0.3365 | 1.8459 |

Two things follow.

**The midpoint is the estimator, not the third-order mean.** It is better than the third-order mean
at all four settings, and better than the uncorrected MAP at three. `noisy` is the decisive case:
the third-order correction alone makes matters *worse* (0.300 → 0.433) while the midpoint improves
them by 1.8× (0.300 → 0.170). Averaging two estimators with different leading residuals is more
robust than either, and it costs nothing beyond the VI solve already needed for the certificate.

**The rule is `disagree/|Δ₃| < 0.5`.** This dimensionless ratio separates exactly the three settings
where the midpoint helps from the one where it does not — 0.198, 0.278, 0.244 against 2.017. It
makes the right call 4 for 4. An absolute threshold on `|Δ₃|` also separates them (0.174, 0.314,
0.701, 1.671) but needs its cut placed between 0.7 and 1.67 with nothing to anchor it there,
whereas the ratio is self-normalising and its natural cut at 0.5 sits in a gap spanning an order of
magnitude. I earlier proposed requiring both clauses; on this evidence the ratio alone is the
better rule, and adding the `|Δ₃| < 0.5` clause would wrongly suppress `noisy`.

Applied, the rule gives bias improvements of 4.3× (baseline), 2.8× (half) and 1.8× (noisy), and
correctly declines to touch `quarter`. Caveat: the `noisy` and `quarter` references have R̂ = 1.48
and 1.77, so those two rows are directional rather than precise, and four settings is a thin basis
for the constant 0.5.

The α sweep supplies five more points, all dense-data cases where the correction is known to help;
their `disagree/|Δ₃|` values are 0.23, 0.18, 0.34, 0.25, 0.20 — all comfortably under the cut.


---

## 5. Theory: where the non-Gaussianity comes from, exactly

With σ fixed, MAGI's negative log-posterior is *exactly* a sum of squares, `−2 log p = ‖R(x)‖²`,
and only the ODE block of `R` is nonlinear — through `f` alone. Differentiating twice,

    ∇²R_a = √b (Lkᵀ)_a ∇²f        (pointwise in the grid index),

so the residual's entire departure from linearity is the ODE's second derivative. Writing
`δ = x − x_MAP` and `E(δ) = R(x_MAP+δ) − R_MAP − Jδ`, stationarity gives `R_MAPᵀJ = 0`, hence the
**exact identity** (no truncation anywhere):

    log q(x) − log p(x) = (R_MAP + Jδ)ᵀ E(δ) + ½‖E(δ)‖² + const,   q = N(x_MAP, (JᵀJ)⁻¹).

**Condition (A).** If `f(·,θ)` is affine in the state `X` for every θ, then `R` is affine in `X` at
fixed θ, so `U(θ,X) = U(θ,X*(θ)) + ½(X−X*)ᵀAᵀA(X−X*)` is exactly quadratic in `X`. Therefore
`p(X|θ)` is **exactly Gaussian**, and integrating `X` out is exact rather than approximate:

    log p(θ) = −U(θ, X*(θ)) − ½ log det(A(θ)ᵀA(θ)) + const.

The 325-dimensional problem is then exactly a 3-dimensional one, solvable by quadrature with **no
MCMC and no Gaussian assumption on θ** — the only error is quadrature error, controlled by adding
nodes until the answer stops moving (it was stable at 7 Gauss–Hermite nodes per dimension).

Validated with three structurally different rules, which agree to all six printed decimals:

| rule | θ means | θ sds |
|---|---|---|
| Gauss–Hermite 9³ | 0.085448, 0.562092, 0.811322 | 0.014407, 0.013004, 0.012095 |
| Gauss–Hermite 13³ | 0.085448, 0.562092, 0.811322 | 0.014407, 0.013004, 0.012095 |
| uniform 21³ over ±5 sd | 0.085448, 0.562092, 0.811322 | 0.014407, 0.013004, 0.012095 |
| Laplace, for contrast | 0.085338, 0.562550, 0.810998 | 0.014400, 0.012995, 0.012096 |

The Gauss–Hermite rules share a Gaussian weight and so could in principle share a blind spot; the
uniform grid shares nothing with them but the integrand (its largest normalised weight is 0.0079,
so it is not concentrating on a few nodes). Against the exact answer the Laplace θ means are off by
0.008, 0.035 and 0.027 posterior sd — small, and consistent with the 0.005 correction size measured
at α = 0 in §4.

This is much weaker than requiring `f` affine in `(X,θ)` jointly — θ may still multiply states, as
it does here through `c(V+R)`, which is why α=0 measures `L2 = 6.76` rather than 0. It covers
linear compartment models, linear pharmacokinetics, and every constant-coefficient system.
Verified numerically: `dA = 0.00e+00` exactly at α=0 versus `1.30e-01` for the cubic ODE, and the
quadrature route confirms Laplace is essentially exact there (θ means within 0.008 sd).

A caveat on rigour: I also tried to turn the exact identity into a usable *a priori* bound via
`Λ = √b‖Lk‖L₂`. The measured values (Λ = 2012, 1588, 2548, 4173 across the α sweep) do not track
the actual error — `corr/Λ` varies by 17× — because a sup-norm bound over the posterior bulk is far
too loose in 325 dimensions. The computable certificates in §4 are the honest substitute: they are
measured rather than bounded, and they were validated against references.

### 5.1 Precision

MAGI's production default is float32, and the third-order correction is a gradient of a trace of a
Hessian — three derivative levels on a quantity already built from cancelling terms. It survives
single precision intact:

| | float64 | float32 | rel diff |
|---|---|---|---|
| MAP ‖grad‖ | 5.4e-07 | 2.9e-02 | 2.7e-05 |
| exact Hessian | — | — | 1.8e-06 |
| third-order correction | — | — | 6.7e-04 |
| correction size (tau) | 0.17388 | 0.17386 | gap 0.00007 |
| resulting bias | 0.0268 | 0.0268 | — |

The fp32 MAP stalls at ‖grad‖ 2.9e-02 rather than 5.4e-07, but that is far below the level that
affects anything downstream — the fp32-vs-fp64 gap in the correction is 0.00007 tau against a
correction of 0.174 and a floor of 0.006. No precision caveat is needed. This does depend on the
`jax.default_matmul_precision("highest")` guard in `magi_logdensity`: without it the fp32 gradient
loses ~4 significant digits, since the einsums' VJPs are matrix-matrix products.


---

## 6. Recommendation

| route | cost | when |
|---|---|---|
| exact quadrature over θ | seconds | `dA ≈ 0` (f affine in the state) |
| **GN MAP → exact Hessian → third-order mean and low-rank VI mean → take the midpoint** | **~1.4 s** | **default, gated on the rule below** |
| report `N(x_MAP, H⁻¹)` uncorrected, escalate | ~0.4 s | `disagree/|Δ₃| ≥ 0.5` |
| preconditioned NUTS | 144 s at baseline; more elsewhere | rule fires, or exactness required |

**The pipeline.** 0.05 s (Gauss–Newton MAP) + 0.30 s (exact Hessian and eigendecomposition) +
0.39 s (third-order mean) + ~0.7 s (low-rank VI mean, which also produces the certificate). Take
the midpoint of the two means and keep the Laplace covariance. Deterministic given a seed, and
insensitive to its two knobs (§4.1).

**The gate.** Compute `disagree/|Δ₃|`. Below 0.5, use the midpoint — it improved the mean by
4.3×, 2.8× and 1.8× on the three settings where it applies. At or above 0.5, report
`N(x_MAP, H⁻¹)` uncorrected and escalate to a sampler; on `quarter` this correctly avoids turning
a bias of 0.197 into one of 1.85.

**On escalating.** Preconditioned NUTS is exact and, at baseline, beats gold's own half-vs-half
floor in 144 s. But at 400 chains × 150 draws it did **not** converge on any of the three harder
settings (R̂ = 1.16, 1.48, 1.77). Escalation therefore means running NUTS *and reading its
diagnostics*, with far longer chains than baseline requires — not assuming it worked because it is
the exact method.

**Not recommended.** The profile/Tierney–Kadane machinery (correct, 13 s, no gain on the joint
covariance); unrestricted Gaussian VI (diverges); spherical cubature for any expectation in this
geometry (its nodes sit 18 posterior sds out); "self-consistent Laplace" (invalid off the mode);
and mSVGD, whose fixed point on this problem is not the posterior (§3.6).

**The single most important practical point.** `N(x_MAP, H⁻¹)` — the thing one writes down by
default — is mis-centred by a full posterior standard deviation on two of three ODE parameters
while its interval *widths* are within a few percent. Nothing about the output looks wrong. About a
second of arithmetic fixes it, and the same arithmetic tells you when it cannot.

## 7. Reproducing, and what would be worth productionising

Nothing in the codebase was modified for this investigation; everything lives in `investigation4/`.
`setup4.py` builds the MAP and exact Hessian per setting; `pipeline.py` is the reusable
deterministic pipeline plus metric-floor calibration; `profile_marg.py` is the bordered-GN
profiler. Experiments `exp01`–`exp21` are standalone and each carries its rationale in its
docstring. References are in `ref4_{setting}.npz`.

If any of this is to graduate into `magi_msvgd`, the ordering by value per line of code is:

1. **The third-order mean correction** (~6 lines on top of `GaussNewtonMAP`). This is the whole
   headline result and it needs the exact Hessian, which `gauss_newton.py` already assembles.
2. **`|Δ₃|` and the slice-curvature screen** (~10 lines, 4 batched log-density evaluations). The
   cheapest half of the safety story: it costs 0.15 s and catches the case where the correction
   should not be applied at all.
3. **The low-rank VI solve** (~25 lines) for the `disagree` certificate and the midpoint mean.
   Needs HVPs, not full Hessians, so it stays cheap.
4. **`dA`**, the condition-(A) test (2 Hessian evaluations). Worth running once per model, not per
   fit, and it identifies problems where an exact route exists.

The profiler in `profile_marg.py` is correct and reusable but earned its place only as a
measurement instrument — it should not go into the production path on the evidence here.
