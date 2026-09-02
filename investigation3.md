# Investigation 3 — Structure-Exploiting and Deterministic Methods for MAGI

A fresh start. Investigations 1 and 2 searched over *samplers* — kernels, metrics, integrators —
and converged on whitened Langevin. This one starts instead from the algebraic structure of the
MAGI posterior and asks what that structure permits, with a preference for deterministic methods
and for guarantees that are allowed to be MAGI-specific.

Scripts in `investigation3/`. Nothing in the existing codebase was modified.

---

## 0. Headline findings

1. **MAGI is exactly a nonlinear least-squares problem.** `−2 log p(x) = ‖R(x)‖² + const` with
   `R` stacking the GP-prior, observation and ODE residuals — verified to a relative
   **8.4e−12**. The first two blocks are exactly *linear*; every nonlinearity lives in the ODE
   block, and it acts *pointwise in time*. Everything below follows from those two facts.

2. **The MAP used throughout investigation 2 was never a stationary point.** It has gradient
   norm **1763**; Gauss-Newton reaches **2.2e−7** and a log-density 16.5 nats higher. The two
   points differ by only ‖dx‖ = 0.024, but in directions of curvature ~1e5 that is a large
   gradient. Several of investigation 2's conclusions rest on quantities evaluated there and
   need revising — see §3.

3. **A fully deterministic pipeline reaches the sampling floor.** Gauss-Newton MAP → exact
   Hessian → closed-form third-order mean correction gives **energy 0.062 against a 0.050
   floor**, with a mean error of **0.027 — below the 0.034 Monte Carlo floor of an 800-draw
   reference**, because it has no Monte Carlo error at all. No randomness anywhere, ~20 s.

4. **An exact conditional, under an explicit and mild condition on the ODE.** If `f` is affine
   in a subset of the parameters — true for mass-action kinetics, Lotka-Volterra and linear
   compartment models, and true of FitzHugh-Nagumo's `(a,b)` given `c` — those parameters are
   **exactly Gaussian given the rest, in closed form**. Validated to reproduce the reference
   marginals to 0.1%, and worth roughly 3–5× the particle count.

5. **Randomize-then-Optimize carries a genuine safety certificate but is not competitive here.**
   Its map is well posed — every solve converges to 1e−13 and the contraction certificate
   `κ < 1` holds for 99.8% of samples — but the importance weights give **ESS 5.4%**, dominated
   by `log|det(Q̄ᵀJ)|` accumulating the Jacobian's variation over 325 dimensions.

6. **Investigation 2's "failure regime" does not exist.** Re-run with a converged MAP, whitened
   MALA and pCN neither diverge nor freeze at σ = 0.5 — they are merely unconverged in budget.
   The catastrophic failures reported there came from whitening with the corrupted metric.

7. **The deterministic correction has an a-priori trust indicator.** `τ = ‖H^{1/2}δ‖/√d`, the
   correction measured in posterior standard deviations per dimension, is computable before any
   reference exists and correctly separates the case where the correction helps (τ = 0.17) from
   the one where it actively hurts (τ = 0.70).

---

## 1. The organizing structure

`lsq.py`. Writing out MAGI's log-density,

```
-2 log p(x) = ||R(x)||^2 + const,
R(x) = [ sqrt(b) Lc^T (X - mu) ;  (X - y)/sigma on observed ;  sqrt(b) Lk^T r(X, theta) ]
```

with `Lc Lcᵀ = C⁻¹`, `Lk Lkᵀ = K⁻¹`, `b = beta_inv` and `r = f(X,θ) − μ̇ − m(X−μ)`. Verified
numerically: the difference from `−log p` has relative standard deviation **8.4e−12** across test
points, i.e. it is a constant.

Two consequences drive everything else.

**The nonlinearity is confined and pointwise.** The GP and observation blocks are exactly linear
in `X`. The ODE block is nonlinear only through `f`, which is applied *separately at each of the
n = 161 grid points* to a 5-vector `(X_i, θ)`. So the entire nonlinear content of a
325-dimensional posterior is 161 copies of a map from 5 variables to 2.

**The posterior would be exactly Gaussian if `R` were affine.** All non-Gaussianity — and every
failure mode catalogued in investigations 1 and 2 — is controlled by the curvature of `R`, which
for a polynomial vector field is explicitly boundable.

### The analytic Jacobian

`jac.py`. Because only the ODE block moves, and only through the pointwise derivatives
`∂f/∂(X_i,θ)`, the residual Jacobian can be assembled from `n` small local derivatives plus
constant blocks rather than by automatic differentiation:

| | cost | accuracy |
|---|---|---|
| `jax.jacfwd` (325 JVPs) | 19 ms | — |
| analytic assembly | **0.08 ms** | rel. error **8.6e−17** |

A 240× speedup, exact to machine precision. This is what makes Gauss-Newton — and hence
everything in §3 and §5 — affordable inside a per-particle loop.

---

## 2. What is provable, and under which conditions

Stated up front because it is the point of the investigation. Let `f` be the ODE vector field.

**(S1) Unconditional.** `−2 log p = ‖R‖² + const` with `R` explicit. Hence the MAP is a
nonlinear least-squares problem and Gauss-Newton applies, with its quadratic local convergence.
No condition on the ODE.

**(S2) Condition (A): `f` affine in a subset `θ_A` of the parameters.** Then the ODE residual is
affine in `θ_A`, the ODE term is a quadratic form in `θ_A`, and since MAGI places a flat prior on
the parameters, `θ_A | (X, θ_rest)` is **exactly Gaussian with closed-form mean and covariance**.
No approximation. §6.

**(S3) Condition (B): `f` affine in `(X, θ)` jointly.** Then `R` is affine, the posterior is
*exactly* Gaussian, Gauss-Newton converges in one step, the third-order correction of §5 is
identically zero, and RTO has constant weights and is exact. The degenerate case, but it fixes
the sense in which every method here degrades: smoothly in the curvature of `f`.

**(S4) Certificate for RTO.** The map is well posed wherever
`κ(x) = ‖I − R̄⁻¹Q̄ᵀJ(x)‖ < 1`, which simultaneously gives geometric convergence of the solve,
invertibility of `Q̄ᵀJ` (so the weights exist) and a local diffeomorphism. `κ` is *measured per
sample at no extra cost*. It is also bounded a priori,

```
kappa  <=  ||Rb^-1|| sqrt(b) ||Lk||  *  L2 * rho
           \___ GP/data discretization only ___/   \__ODE__/
```

with `L₂` a bound on the second derivatives of `f` over the region — an explicit condition on the
specified ODE, and zero for affine `f`.

**(S5) Indicator, not a proof.** The third-order correction of §5 is the leading term of an
asymptotic expansion, so it is trustworthy only while that term is small. `τ = ‖H^{1/2}δ‖/√d`
measures exactly that and is computable before any reference exists. §7.

---

## 3. Correction: the MAP in investigation 2 was not converged

Every Laplace-based quantity in investigation 2 was evaluated at a point produced by Prodigy with
`atol=1e-7`. Gauss-Newton from the least-squares form disagrees:

| | log p | ‖∇ log p‖ | mean error |
|---|---|---|---|
| cached MAP (Prodigy, 30 000 iters) | −12.656 | **1763** | 0.333 |
| Gauss-Newton refined | **+3.805** | **2.2e−7** | **0.101** |

The points differ by ‖dx‖ = 0.024 — tiny — but the Hessian has eigenvalues up to 1e5, so a
0.024 displacement leaves a gradient of order 1e3. A first-order optimizer cannot solve a
d = 325 problem at condition number 2e4; Gauss-Newton, which the least-squares form supplies for
free, converges in tens of iterations.

Three of investigation 2's conclusions change.

- **The MAP-to-mean displacement is 0.101, not 0.335.** Its claim that "the entire error of the
  Laplace approximation is the mean" survives in kind, but the magnitude was inflated ~3× by
  optimizer error. `N(x*, H⁻¹)` scores energy **0.179**, not 1.446.
- **The σ = 0.5 "weak identification" diagnosis was an artifact.** A Hessian at a non-stationary
  point is not a Hessian. Re-evaluated at a converged MAP:

| setting | ‖∇‖ Prodigy | ‖∇‖ GN | min eig(H) | was (inv. 2) | tr(H⁻¹) | was |
|---|---|---|---|---|---|---|
| baseline σ=0.2 | 1575 | 2.2e−7 | 4.94 | 5.31 | 1.00 | 0.98 |
| half-obs | 2040 | 2.8e−7 | 2.65 | 1.78 | 1.42 | 1.68 |
| quarter-obs | 2638 | 2.1e−5 | 0.39 | 1.59 | 4.69 | 1.88 |
| **noisy σ=0.5** | 4530 | 5.0e−7 | **1.07** | **0.0078** | **3.97** | **133.2** |

  The smallest eigenvalue at σ = 0.5 is **1.07, not 0.0078** — 137× larger — and `tr(H⁻¹)` is
  **3.97, not 133**. The posterior there is not pathologically flat. Investigation 2's whitened
  samplers diverged at σ = 0.5 because they were whitening with a *corrupted metric* whose
  spurious tiny eigenvalue stretched one direction by 100×, not because of weak identification.
- **The "pre-flight check" was diagnosing optimizer failure**, not a property of the posterior.
  It was still catching something real, but the attribution was wrong.

What survives untouched: the samplers' own results. Whitened ULA/MALA/pCN reached energy ≈ 0.049
*using the corrupted metric*, which is direct confirmation of investigation 2's own argument that
the metric is only a preconditioner and does not bias the answer.

### Re-running the samplers with a converged MAP

`exp07_rerun.py`. Investigation 2's sharpest claim was that σ = 0.5 is a regime where *every*
method built on the Laplace metric fails — ULA diverging, MALA freezing 43% of its chains. Since
that metric had a spurious eigenvalue of 0.0078 (whitening stretches that direction by ~100×),
the claim had to be re-tested. With the Gauss-Newton MAP and the Hessian evaluated there
(k = 400, fixed step sizes, no adaptation):

| setting | sampler | energy | floor | varwtd | bias | accept |
|---|---|---|---|---|---|---|
| baseline | MALA | **0.0815** | 0.0828 | 0.945 | 0.051 | 0.88 |
| baseline | pCN | 0.0841 | 0.0828 | 0.894 | 0.052 | 0.77 |
| σ = 0.5 | MALA | 0.850 | 0.0521 | 0.953 | 0.250 | 0.08 |
| σ = 0.5 | pCN | 0.280 | 0.0521 | 0.378 | 0.134 | 0.52 |

**Neither sampler diverges or freezes at σ = 0.5.** Both run to completion and return finite,
sane ensembles. They are merely *unconverged* in the budget given — pCN reaches energy 0.28
against a 0.052 floor, and MALA's acceptance falls to 0.08 because the step size was fixed rather
than adapted.

So investigation 2's failure regime is **not a property of the posterior**. σ = 0.5 is a harder
problem that needs more compute (and, for MALA, step-size adaptation); the catastrophic,
qualitative failures reported there were caused by whitening with a corrupted metric. The
transience and stickiness analyses remain correct as mathematics — they are properties of ULA and
MALA on super-quadratic tails — but the specific MAGI setting used to demonstrate them was not
the weakly identified problem it was taken to be.

---

## 4. A fully deterministic pipeline

`exp02_meancorr.py`. Since the whole error of the Laplace approximation is the mean, a
deterministic estimate of the mean is a deterministic estimate of the posterior. The classical
Laplace expansion supplies one in closed form: expanding `U` to cubic order and taking Gaussian
expectations,

```
E[x]  =  x*  -  (1/2) H^-1 grad_x tr( H^-1 grad^2 U(x) ) |_{x*}  +  higher order
```

For MAGI every ingredient is exact and cheap. `∇²U = JᵀJ + Σ_a R_a ∇²R_a`, the analytic Jacobian
gives the first term, and the second is nonzero only on the ODE rows and only in the local
`(X_m, θ)` blocks — `n` small Hessians of a 5-variable function. The assembled Hessian matches
`jax.hessian` to **3.6e−16**, and the outer gradient is one autodiff pass, 3.4 s.

| mean estimate | error | randomness |
|---|---|---|
| MAP (Laplace centre) | 0.1008 | none |
| **MAP + deterministic correction** | **0.0271** | **none** |
| SAV-VI, 2.3 s (investigation 2) | 0.0513 | stochastic |
| mean of a full SVGD run | 0.0680 | stochastic |
| 800-draw sampling floor | 0.0340 | — |

The correction is **more accurate than the 800-draw Monte Carlo floor**, which is not paradoxical:
a deterministic estimate has no sampling error, so the floor does not apply to it.

Scored as a posterior representation:

| | energy | dev | sd ratio | profile |
|---|---|---|---|---|
| N(MAP, H⁻¹) | 0.179 | 4.0 | 0.983 | ≈1 |
| **N(MAP + correction, H⁻¹)** | **0.062** | 5.7 | 0.999 | ≈1 |
| NUTS floor | 0.050 | 3.5 | 1.003 | ≈1 |

Total cost ~20 s, entirely deterministic, no kernel, no chain, no step size, no tuning constant.

---

## 5. Randomize-then-Optimize: the map works, the weights do not

`rto.py`, `rto2.py`, `exp01*_rto.py`. RTO draws `ξ ~ N(0,I_d)` and *solves* `Q̄ᵀR(x) = ξ`, where
`J(x*) = Q̄R̄`. It is exact whenever `R` is affine, its pushforward density is available in closed
form so the sample can be exactly reweighted, and — unlike every method in investigations 1 and
2 — it involves **no Markov chain**, so neither transience nor stickiness can occur.

**The map is sound.** With full Gauss-Newton (the analytic Jacobian makes this affordable), all
400 solves converge to **1.6e−13**, and the certificate `κ < 1` holds for **99.8%** of samples
(max 1.038, mean 0.426). This is a real guarantee of well-posedness, obtained at no extra cost.

*The fixed-Jacobian variant is not viable*: its iteration matrix is `I − R̄⁻¹Q̄ᵀJ(x)` and `R̄⁻¹`
amplifies by `1/σ_min`, so convergence is governed by the worst-conditioned direction. It
diverged (residual 19.9 → 44.5). Recomputing the Jacobian removes that dependence entirely.

**The weights are not.** ESS is **21.6/400 (5.4%)**, and reweighting makes the answer *worse*
(energy 0.152 unweighted → 1.099 weighted). Decomposing `log w = −log|det(Q̄ᵀJ)| − ‖R_⊥‖²/2`:

| term | sd (nats) |
|---|---|
| `−log|det(Q̄ᵀJ(x))|` | **1.264** |
| `−‖R_⊥(x)‖²/2` | 0.472 |
| total | 1.384 |

The determinant term dominates. (I had expected the opposite — that the `N − d = 641`
null-space dimensions would drive it — and the measurement says otherwise.) With
`κ̄ = 0.43`, the Jacobian varies by ~40%, and the log-determinant accumulates that across all
325 dimensions into ~1.3 nats, which is enough to destroy the ESS.

So RTO is the right *shape* for this problem and carries the best safety certificate of anything
tried, but on MAGI at this dimension its reweighting is not usable. Unweighted it gives energy
0.152, worse than §4's deterministic 0.062 at far greater cost.

---

## 6. Exact parameter conditionals under Condition (A)

`exp03_rb.py`. If `f` is affine in `θ_A`, the ODE residual is `A₀ + M θ_A` and the ODE term is a
quadratic form, so with MAGI's flat parameter prior

```
theta_A | X, theta_rest  ~  N( -P^-1 q,  (beta_inv P)^-1 ),   P = M^T K^-1 M,  q = M^T K^-1 A0
```

exactly. For FitzHugh-Nagumo this holds for `(a,b)` given `c`, with
`M = [ (1/c)·1 , −(1/c)·R ]`. It holds for *all* parameters in mass-action kinetics,
Lotka-Volterra and linear compartment models.

**Validation** — applying the closed form to gold-standard `(X,c)` draws must reproduce the
reference marginals, and does:

| | 2.5% | 50% | 97.5% | width |
|---|---|---|---|---|
| a, closed form | 0.1575 | 0.1954 | 0.2341 | 0.0767 |
| a, NUTS reference | 0.1578 | 0.1954 | 0.2353 | 0.0776 |
| b, closed form | 0.1572 | 0.3254 | 0.4702 | **0.3129** |
| b, NUTS reference | 0.1569 | 0.3248 | 0.4699 | **0.3130** |

**Benefit.** The reported parameter marginals become an exact Gaussian mixture rather than a
particle histogram, removing all Monte Carlo error in exactly the quantities a MAGI user
reports. Interval width as % of the reference, averaged over 20 independent subsamples:

| k | a: Rao-Blackwellized | a: raw | b: RB | b: raw |
|---|---|---|---|---|
| 10 | **94.2** | 79.1 | **81.9** | 72.0 |
| 40 | **99.8** | 91.1 | **93.6** | 84.2 |
| 200 | 98.6 | 97.9 | 100.8 | 98.8 |
| 1000 | 100.2 | 99.9 | 101.6 | 101.0 |

At k = 40 the Rao-Blackwellized estimate is where the raw one gets to at k ≈ 200 — roughly a
3–5× saving in particles, for a few lines of linear algebra and no approximation.

---

## 7. When to trust the deterministic correction

`exp06_criterion.py`. The correction is an asymptotic term and helps only while it is small.
Measured in the posterior's own metric, `τ = ‖H^{1/2}δ‖/√d` is the correction expressed in
posterior standard deviations per dimension, and needs no reference:

| setting | τ | correction's effect on energy |
|---|---|---|
| baseline σ=0.2 | **0.174** | 0.179 → **0.062** (helps) |
| half-obs | 0.314 | (no reference available) |
| noisy σ=0.5 | **0.701** | 1.157 → **2.436** (hurts) |
| quarter-obs | 1.671 | (no reference available) |

τ correctly orders the two cases that can be checked. **It is an indicator, not a calibrated
threshold**: with two validated points I can say τ ≈ 0.17 was safe and τ ≈ 0.70 was not, and
nothing sharper. Reporting τ alongside the correction is free and lets a user see which regime
they are in.

Note that σ = 0.5 remains a harder problem — but for the opposite reason to the one investigation
2 gave. It is not that the posterior is flat (§3 shows it is not); it is that the posterior is
genuinely more non-Gaussian there, so the cubic term is no longer a perturbation.

---

## 7b. How fast can the least-squares solve be?

`exp08_solvers.py`, `exp09_scaling.py`. The pipeline above took 21 s, almost all of it wasted.
Three independent inefficiencies, each measured separately.

**The first-order warm start is unnecessary.** Gauss-Newton converges from the raw MAGI
initialization with no help at all:

```
||grad|| after k GN steps:  k=1: 3.1e3   k=3: 9.8   k=10: 5.6e-3   k=20: 7.1e-6   k=30: 2.2e-7
```

30 steps, **0.07 s**. The 20 000 Prodigy iterations that preceded it contributed nothing except
a worse starting point than they appeared to.

**The linear solve was 500× slower than necessary.** `jnp.linalg.lstsq` computes an SVD, which
is wildly over-engineered for a matrix with `cond(J) = 125`:

| solve | time | rel. error |
|---|---|---|
| `lstsq` / SVD (fp64) | 176.7 ms | 2.6e−14 |
| QR + triangular solve (fp64) | 5.7 ms | 8.2e−15 |
| **normal equations + Cholesky (fp64)** | **1.6 ms** | 2.9e−13 |
| normal equations + Cholesky (fp32) | 0.17 ms | 2.6e−4 |
| **fp32 Cholesky + one fp64 refinement** | **0.35 ms** | 3.2e−7 |

Normal equations square the condition number to 1.6e4, which in double precision costs about
four digits out of sixteen — irrelevant here, and it buys 110×. The fp32 variant with a single
fp64 refinement step is another 4.6× on hardware where fp64 runs at 1/64 rate, and would be the
choice at larger `n`.

**End to end:**

| stage | before | after |
|---|---|---|
| MAP | ~20 s (Prodigy + 60 GN with `lstsq`) | **0.07 s** |
| exact Hessian | 0.8 s | 0.03 s |
| third-order correction | 3.4 s | 2.71 s |
| **total** | **21.2 s** | **2.80 s** |

with identical output (bias 0.1008 → 0.0271, energy 0.178 → 0.060). The correction is now the
bottleneck, at 97% of the runtime: it autodifferentiates through a function that assembles a
325×325 Hessian. Contracting `Σ_jk Σ_jk U_ijk` directly from the local 5×5×5 tensors would avoid
that, and is the obvious next optimization if it ever matters.

**A banded solve is plausible but not established.** Both precision matrices have rapid
off-diagonal decay — at n=161, `C⁻¹` is numerically banded at width 27 and `K⁻¹` at 16 — which
would make a banded Cholesky O(n·b²) rather than O(n³). Whether that is an asymptotic win depends
on how `b` scales:

| step | n | bw(C⁻¹) | bw/n | bw(K⁻¹) | bw/n |
|---|---|---|---|---|---|
| 0.25 | 81 | 20 | 0.247 | 13 | 0.160 |
| 0.125 | 161 | 27 | 0.168 | 16 | 0.099 |
| 0.0625 | 321 | 38 | 0.118 | 102 | 0.318 |

`C⁻¹`'s *relative* bandwidth falls steadily with refinement (0.247 → 0.168 → 0.118), which is
what a banded solver needs. `K⁻¹`'s does not — it jumps to 102 at the finest grid, most likely
because the GP hyperparameters are refit at each discretization rather than held fixed. So the
banded route looks promising for fine grids but I have not established it, and would want the
hyperparameters controlled before claiming anything. The genuinely scalable route for a Matérn
kernel is its exact state-space form, which gives an O(n) block-tridiagonal solve independent of
the grid — a larger change, and the right one if `n` ever becomes the bottleneck.

---

## 8. Recommendations

1. **Adopt the least-squares formulation.** It is exact, it costs nothing, and it supplies (i) a
   properly converged MAP by Gauss-Newton — which alone takes `N(x*,H⁻¹)` from energy 1.446 to
   0.179 — (ii) an analytic Jacobian 240× faster than autodiff, and (iii) the Gauss-Newton
   Hessian for free. **The single most valuable change here is replacing the first-order MAP
   solve.** The current one is not finding the mode, and the replacement is *faster*: 30
   Gauss-Newton steps with a Cholesky solve of the normal equations reach ‖grad‖ = 2e−7 in
   **0.07 s**, against 20 s of Prodigy that ends at ‖grad‖ = 1763 (§7b). Do not use `lstsq`
   inside the loop — Cholesky is 110× faster at this conditioning.
2. **Then the deterministic correction**, ~3 s on top, taking energy to 0.062 against a 0.050
   floor with no randomness. Report `τ` with it.
3. **Rao-Blackwellize the parameters** wherever Condition (A) holds. Exact, cheap, and worth
   3–5× the particles on precisely the quantities that get reported.
4. **Do not pursue RTO further on this problem** unless the dimension drops substantially; the
   log-determinant term scales with `d` and there is no obvious way around it.

### Limitations

- The deterministic pipeline returns a *Gaussian* (with a corrected mean), so it cannot represent
  the marginal skew that investigation 2 measured (θ_b skew −0.219). Where shape matters, a
  sampler is still required.
- Its residual error is **entirely the mean**, not the covariance: substituting the true mean into
  the Gauss-Newton covariance gives energy 0.049 against a 0.049 floor, whereas substituting the
  true covariance into the corrected mean gives 0.056. Further work should target a fourth-order
  mean term, not the covariance.
- The σ = 0.5 re-runs used fixed step sizes; a tuned sampler would do better there, so those rows
  bound the method from below rather than characterising it.
- `τ` is an indicator with two validated points, not a calibrated criterion.
- All of this is one ODE system at one dimension. Condition (A) and the `L₂` bound are
  ODE-specific by design, but the *numbers* are FitzHugh-Nagumo's.
- The third-order correction needs `∇³U` contracted once; it is cheap here because `f` is
  pointwise and low-degree, and would not be for a general dense nonlinearity.

---

## Appendix: scripts

| file | purpose |
|---|---|
| `lsq.py` | least-squares form of the MAGI posterior, with the exactness check |
| `jac.py` | analytic residual Jacobian, with verification against autodiff |
| `rto.py`, `rto2.py` | RTO: fixed-Jacobian and full Gauss-Newton variants, weights, κ |
| `exp01*_rto.py` | RTO convergence, ESS, certificate |
| `exp02_meancorr.py` | deterministic third-order mean correction |
| `exp03_rb.py` | exact parameter conditionals under Condition (A) |
| `exp04_safety.py` | MAP convergence and Hessian spectra across settings |
| `exp05_final.py` | RTO weight decomposition; deterministic pipeline across settings |
| `exp06_criterion.py` | the τ indicator; Rao-Blackwellization at small k |
| `exp07_rerun.py` | investigation 2's whitened samplers, re-run with a converged MAP |
| `exp08_solvers.py` | linear-solve variants, warm-start necessity, precision bandwidth |
| `exp09_scaling.py` | bandwidth scaling with the grid; timed end-to-end pipeline |

`harness.py` and `laplace_cache.npz` are symlinked from `investigation2/`.
