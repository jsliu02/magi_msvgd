# Posterior Faithfulness of mSVGD on MAGI — Investigation Notes

Working notes on diagnosing and attempting to correct SVGD's variance
collapse / underdispersion on a high-dimensional ODE-parameter-inference
problem, benchmarked against a NUTS gold standard.

---

## Headline findings

1. Of six corrective techniques surveyed, **the density-reweighted kernel**
   (Huang, Dong & Fang 2023) gave the best credible-interval calibration and
   is now the default (`reweighted_kernel=True`). Sliced SVGD was actively
   harmful; MK-SVGD and matrix-valued SVGD were partial improvements;
   stacking any two techniques overshot rather than added value.

2. The reweighted kernel's `clip_exponent` is **not** an overflow guard. It is
   load-bearing: it acts as an implicit trust region, and it is standing in
   for the paper's Assumption [A3] (bounded density ratio), which this problem
   violates by hundreds of nats.

3. The kernel **can** be made exactly overflow-free (rank-1 factorization +
   log-domain shift, no free constant). This is worth doing on its own merits,
   but it does not remove the need for the clip: the binding constraint is
   float *dynamic range*, not magnitude.

4. Every principled, constant-free replacement for the clip that was tried
   either neutered the method or destabilized it. `clip_exponent = 20` is best
   understood as **a one-parameter dial tuned so the intervals cross NUTS**,
   not as a derived quantity.

5. A **Stein-identity diagnostic** `R` (target value 1, derived not tuned,
   free to compute, validated to 0.1% on the gold standard) shows the ensemble
   satisfies the identity at only **~28%** of the required level for the
   production kernel (~4% for the per-particle clip variant) — while all 325
   marginal standard deviations sit at ~84% of NUTS. The failure is
   **anisotropic collapse along stiff directions**, not a scale deficit, and
   matching θ interval widths against NUTS is a weak validation criterion.
   Now implemented: printed by `solve()` and left on `self.stein_R`.

6. Decomposing `R` along the NUTS principal axes confirms this directly: the
   standard kernel retains **~0.000** of the posterior variance in the stiffest
   40% of directions. Preconditioning is the one intervention that targets that
   mechanism, and it works — matrix + reweighted reaches R = 0.367 vs 0.281 at
   equal interval calibration (§6). This is the one case where **stacking does
   help**, contra finding 1, which was reached by scoring on θ widths alone.

---

## 1. Setup

### Problem

FitzHugh–Nagumo ODE parameter inference via MAGI.

```
dV/dt =  c(V - V³/3 + R)
dR/dt = -(1/c)(V - a + b·R)
```

- Inferred vector: `θ = (a, b, c)` plus the latent trajectory.
- Total dimension **d = 325** (3 parameters + 322 trajectory coordinates).
- Discretization `t ∈ [0, 20]`, step 0.125; observation noise `σ = (0.2, 0.2)`.
- Production runs in **fp32** on GPU; NUTS reference runs on CPU (fp64).
- Solver config used throughout: `k=200 → k_schedule=800`, Prodigy optimizer,
  `atol=0.1`, `rtol=0`, `max_iter=1000`.

### NUTS gold standard

Built with blackjax + window adaptation, saved to
`magi_msvgd/dev tests/nuts_gold_standard.npz` (166 MB).

| | |
|---|---|
| chains | 8 |
| warmup / sampling | 2000 / 8000 |
| total draws | 64,000 × 325 dims |
| target accept | 0.95 |
| **divergences** | **0 of 64,000** |
| R̂ (θ) | 1.0016, 1.0030, 1.0022 (max over all 325 dims: 1.0048) |
| ESS (θ) | 3598, 1220, 2271 (min over all dims: 1202) |
| θ posterior mean | (0.1958, 0.3212, 2.8981); truth (0.2, 0.2, 3.0) |
| θ 95% CI | a [0.158, 0.235], b [0.157, 0.470], c [2.781, 3.013] |

This is a clean chain by every standard diagnostic and is treated as ground
truth for *posterior fidelity* throughout.

### A recurring caveat: fidelity ≠ coverage

The gold standard itself has poor frequentist coverage of the true `θ` on `b`
and `c` (43% and 33% over 30 datasets; 52% / 55% over 100). This reproduced
across fp64/real-data and fp32/simulated runs, so it reflects the MAGI
posterior being narrow/biased relative to truth, **not** a sampler defect.

Consequence: **coverage of the true parameter and fidelity to the NUTS
posterior point in opposite directions.** All rankings below use fidelity
(width ratio to NUTS), per the stated priority.

---

## 2. Corrective techniques surveyed

### 2.1 Sliced SVGD (SKSD / S-SVGD) — *rejected, deleted*

Gong, Peng & Liu, ICLR 2021 (`sksd.pdf`).

- **Axis-aligned slicing.** Implemented as a drop-in kernel. Empirically
  confirmed the paper's own caveat (Section B.1, Eq. 20–21): axis-aligned
  slices are **blind to correlation**. On a correlated target it cannot
  represent the dependence structure at all. Verified by switching the smoke
  test to an uncorrelated target, where it behaves; a documented,
  non-asserting demo was kept for the correlated case.
- **maxSKSD-rg (optimized slice directions).** Eq. 6–11, Algorithm 2.
  Implemented; raw projected ascent and Adam ascent on the slice directions
  **both diverged exponentially**. The formula was re-derived against the
  paper and the `R = G = I` reduction checked exactly (residual 0.0), so this
  is a genuine negative result rather than a transcription bug.
- **On MAGI.** Axis-aligned sliced SVGD degraded results relative to the
  standard kernel.

All sliced-SVGD files were deleted.

### 2.2 Post-hoc Stein importance reweighting

Liu & Lee 2017. Build the KSD Gram matrix over final particles, solve a convex
QP over the simplex, report `ESS = 1/Σwᵢ²`.

Works as advertised and does widen the intervals — but see §2.7: it widens
*past* NUTS.

### 2.3 Tempering / flattening the target

Power tempering `p^β`, both at fixed `β` followed by reweighting, and with
ESS-targeted adaptive `β`. The flatten-then-correct idea is sound in principle
but did not produce a better width ratio than the reweighted kernel alone, and
adds a schedule to tune.

### 2.4 MK-SVGD (multiple-kernel SVGD) — *partial improvement*

Ai, Liu, He & Xu 2019 (`mksvgd.pdf`). Bandwidth ladder with weights
`wᵢ ∝ √(mean(Mᵢ))` from per-bandwidth KSD (Eq. 19/21).

Two problems found and fixed:

- A **sign error of mine** in the initial implementation
  (`phi_i = (K@score - dxkxy)/k`, should be `+`). Fixed by building the
  weighted sum from each kernel's already-validated `_combine` output;
  `n_kernels=1` equivalence to standard SVGD then exact (4.16e-17).
- **Wide bandwidth ladders collapse.** The paper's own Eq. 21 weighting
  concentrates essentially all mass on the smallest bandwidth
  (`norm_sq` 1180 vs 0.003). Mitigated with conservative defaults
  (`n_kernels=5`, `ratio=2.0`) and documented as a failure mode.

### 2.5 Matrix-valued SVGD — *partial improvement*

Wang, Tang, Bajaj & Liu, NeurIPS 2019 (`matrix_svgd.pdf`).
`K_Q(x,y) = Q⁻¹ exp(-‖x-y‖²_Q / 2h)` (Eq. 12–15), diagonal empirical-Fisher
preconditioner.

- Initially reported "converged after 1 iteration" — a false convergence.
  The `Q⁻¹`-rescaled gradients simply fell below `atol=0.1`. Re-run with a
  fixed iteration budget. **General pitfall: an absolute tolerance cannot be
  shared across differently-scaled updates.** (The same issue later motivated
  decoupling the monitored gradient from the optimizer input — §3.6.)
- Combined matrix + reweighted was implemented and run on the same 30
  datasets; the stack overshot. Both files were subsequently deleted.

Note: matrix SVGD was scored on θ interval widths, which **§6 shows is the
wrong criterion for it** — re-scored on the Stein diagnostic it is clearly
doing work, and its combination with the reweighted kernel is the best variant
found.

### 2.6 Density-reweighted kernel — *adopted as default*

Huang, Dong & Fang 2023, Eq. 24:

```
k(x,y) = p_*(x)^(-1/2) · k_base(x,y) · p_*(y)^(-1/2)
```

Reweighting by the target's inverse-sqrt density amplifies repulsion in
low-density regions, where the standard kernel's corrective gradient vanishes
as particle density → 0 — the mechanism behind variance collapse.

The product rule on the `p_*(x)^(-1/2)` factor, with `s(x) = ∇log p_*(x)`:

```
∇_x k(x,y) = k(x,y)·[∇_x log k_base(x,y) - 0.5·s(x)]
           = k(x,y)·[-2(x-y)/h - 0.5·s(x)]
```

The repulsion term is unchanged and the **drift coefficient is exactly
halved** — hence `drift=0.5` rather than `1.0` in `_combine`.

Now `msvgd.MSVGD._reweighted_svgd_update`, exposed as
`solve(..., reweighted_kernel=...)`, defaulted to `True` in `magi.MAGI`.

### 2.7 Study results

**B = 100 datasets** (`dev tests/b100_mk_study_results.json`, shorter NUTS):

| method | width % of NUTS (a, b, c) | \|dev\| | coverage % | sec |
|---|---|---|---|---|
| standard | [76.0, 85.8, 67.1] | 23.7 | [90, 75, 75] | 3.7 |
| **reweighted** | **[114.1, 116.3, 101.3]** | **10.6** | [99, 86, 84] | 4.1 |
| MK | [84.6, 93.5, 77.7] | 14.8 | [91, 81, 82] | 4.4 |
| NUTS | [100, 100, 100] | 0.0 | [95, 52, 55] | 27.4 |

**B = 30 datasets** (`dev tests/b30_reweighted_stein_results.json`, longer NUTS;
NUTS on CPU, SVGD on GPU):

| method | width % of NUTS (a, b, c) | \|dev\| | coverage % | sec |
|---|---|---|---|---|
| **reweighted** | **[109.8, 109.3, 99.1]** | **6.7** | [97, 70, 83] | 4.2 |
| reweighted + Stein | [118.4, 117.1, 103.3] | 12.9 | [100, 73, 87] | 4.6 |
| NUTS | [100, 100, 100] | 0.0 | [93, 43, 33] | 125.1 |

**Stacking does not help.** Stein reweighting on top of the reweighted kernel
pushes widths from ~109% to ~118% of NUTS. It buys frequentist coverage
(70→73% on `b`, 83→87% on `c`) purely by overshooting. Under a fidelity
criterion it is a regression. Same conclusion for matrix + reweighted.

**Ranking (fidelity):** reweighted ≫ MK > matrix > standard ≫ sliced.

---

## 3. The `clip_exponent` investigation

### 3.1 What it is

Two engineering additions to the paper, neither from it:

```python
ld       = logdensity - jnp.max(logdensity)                     # <= 0
reweight = jnp.exp(jnp.clip(-0.5*(ld[:,None] + ld[None,:]), max=clip_exponent))
```

`exp(-0.5·logdensity(x))` is defined only up to `logdensity`'s arbitrary
additive constant and would overflow outright, so the log-density is centered
by its per-batch max and the pairwise exponent clipped. Default `20.0`.

### 3.2 Motivating symptom

The reweighted kernel's reported max gradient is ~1 order of magnitude larger
than the standard kernel's near convergence. Investigated and found
**systemic**: the two kernels converge to genuinely different equilibria
(ensemble log-density spread 4.5 vs 340 nats). Reading the *standard* kernel's
reporter on the *reweighted* final state gives 13.7, versus 0.08 on the
standard final state. θ stabilizes by ~1000 iterations while the gradient
plateaus at 3–5, so **max grad is a poor convergence signal for this kernel**.

A rank-1 hypothesis for the gap was tested and **refuted**: the pairwise clip
breaks the `w ⊗ w` factorization (residual 4.85e8, not 0), and `max(w)/mean(w)`
was only 1.57.

### 3.3 The clip is load-bearing, not an overflow guard

Static evaluation at a fixed converged state:

```
   clip   max reweight   max|combined|  finite?
   20.0      4.852e+08     2.73041e+09     True
   40.0      2.354e+17      1.4031e+18     True
  100.0            inf             nan    False
    inf            inf             nan    False
```

Unclipped NaNs in fp32 **and** fp64. And it is a dynamical runaway, not merely
precision: raising the cap grows the ensemble log-density spread
**236 → 1755 → 2632 nats** (fp64, clip 20/100/400).

**Silent intermediate failure.** Before the NaN there is a regime where
gradients grow so large the optimizer step collapses and particles freeze at
initialization. fp64 clip=400 gives θ mean `[0.1632, 0.1632, 1.9722]`, matching
`theta_init = [0.16388468, 0.16205919, 1.9732542]` — collapsed intervals
(13–18% of NUTS) that look like a converged answer.

**fp32 usable window** (3 seeds, width % of NUTS):

| clip | a | b | c | note |
|---|---|---|---|---|
| 20 | 108.4 | 107.2 | 97.0 | sd [1.7, 1.3, 2.6] |
| 40 | 107.1 | 102.3 | 93.7 | sd [0.8, 4.3, 4.9] — indistinguishable from 20 |
| 60 | 58.9 | 13.3 | 17.9 | **frozen at init** |

So 20 sits inside the usable window with ~2× headroom. No default change was
warranted on these grounds.

### 3.4 An exact overflow-free reformulation

The reweight matrix is a symmetric rank-1 rescaling of the RBF matrix:

```
K = diag(w) · RBF · diag(w),    w_i = exp(-0.5·ld_i)
```

Every term in `_combine` is **linear in K**, so the update is **homogeneous of
degree 2 in w**: scaling `w → c·w` scales the whole update by `c²`. Choosing
`c = exp(-max u)` where `u = -0.5·ld_centered` gives

```
ŵ = exp(u - max u) ∈ (0, 1]      ⟹      K̂ ∈ [0, 1]
```

which **cannot overflow, by construction**. The discarded factor
`exp(2·max u)` is a single global scalar that is never materialized — and a
uniform scale on the whole kernel is exactly the class of rescaling already
shown to be invisible to Adam-family optimizers and provably invariant for the
monitored gradient.

Verified numerically:

```
global factor between forms      = 169.036   (predicted exp(2·max u) = 168.872)
MONITORED grad, current form     = 0.28338227
MONITORED grad, stable form      = 0.28310463
direction cosine between updates = 0.9999964833
```

**Structural catch:** this requires clipping **per-particle** (`u_i ≤ U`), not
pairwise. Pairwise clipping breaks the exact rank-1 factorization — the same
fact that refuted the rank-1 hypothesis in §3.2. The two are different
operations and are not interchangeable (§3.7).

### 3.5 Why that does not remove the clip

The shift trades overflow for underflow; it does not create dynamic range.

| | representable exp range |
|---|---|
| float32 | exp(-87.3) .. exp(88.7) → **176 nats** |
| float64 | exp(-708.4) .. exp(709.8) → **1418 nats** |
| needed, unclipped | ~28,000 nats |

Only ~5% of the required range is representable in fp64. With the stable form
and Adam(0.01), unclipped runs stay **finite** for 300 iterations (particles
~2.4, score ~1.4e4, K ≤ 1, updates ~30 — so the NaN seen with the production
solver was Prodigy-specific, not the kernel). But the ensemble spread grows
2397 → 28,160 nats, and:

```
after 300 iters: ld spread = 19120 nats
  w exactly 0        : 73.0% of particles
  K rows exactly 0   : 73.5% get NO kernel interaction
  effective participants: 54 of 200
```

The ensemble silently collapses onto a quarter of its particles, with the rest
receiving neither drift nor repulsion. Arguably worse than a NaN.

**Conclusion: overflow is a solvable representation problem; the clip is
solving a different problem** — bounding dispersion so the weight ratio stays
inside float range at all.

### 3.6 Related fix: decoupling the monitored gradient

The mean-normalization was initially described as "invisible" to Adam. That
was corrected: the constant is recomputed each iteration, so even Adam's
invariance is only approximate, and SGD / Adagrad / Prodigy are genuinely
affected. The implemented fix returns the update **unchanged** alongside
`scale = mean(reweight)`, which the caller divides by **for monitoring and
atol only, never for the optimizer step** — making the rescaling exactly
invisible to the trajectory for any optimizer, and letting one `atol`/`rtol`
serve both kernels.

### 3.7 The clip is Assumption [A3]

`u_i ≤ U` is exactly

```
max_j p_*(x_j) / p_*(x_i)  ≤  exp(2U)
```

i.e. the paper's **Assumption [A3]** (`sup_x p_t(x)/p_*(x) ≤ β`, warmness),
with `β = e^{2U}`. The paper *assumes* this; it does not enforce it. Its
experiments are low-dimensional, where A3 is plausible. At d = 325 the
reweight is an importance weight whose variance grows exponentially in
dimension, and A3 is violated by hundreds of nats.

So the quantity being bounded is the right one. The open question is the value.

### 3.8 Principled, constant-free alternatives — all failed

| approach | free constant? | width % of NUTS | \|dev\| |
|---|---|---|---|
| standard kernel (no reweight) | — | [76, 84, 68] | 24.1 |
| KDE reweight (`p_t`, per Prop 4.1) | **none** | [68, 73, 62] | 32.4 |
| Ionides (2008) truncation | **none** | NaN | — |
| ESS-targeted cap (ESS ≥ k/2) | ESS fraction | NaN | — |
| A3 bound, U = 4 | yes | [81, 73, 76] | 23.7 |
| A3 bound, U = 6.35 | yes | [85, 76, 99] | 13.2 |
| A3 bound, U = 9 | yes | [90, 84, 88] | 13.0 |
| A3 bound, U = √(d/2) = 12.75 | **derived** | [84, 71, 110] | 18.0 |
| A3 bound, U = 18 | yes | [74, 52, 134] | 35.9 |
| A3 bound, U = 25.5 | yes | [78, 59, 151] | 37.9 |
| per-particle U = 10 (stable form) | yes | [95, 94, 93] | 6.1 |
| **production pairwise clip = 20** | yes | **[108, 107, 97]** | **6.2** |

Diagnoses:

- **KDE reweight.** Prop 4.1 actually specifies reweighting by `p_t`, the
  *current particle density*; `p_*` is the paper's acknowledged approximation.
  `p̂_t(x_i) = (1/k)Σ_j k_base(x_i,x_j)` is already computed inside `_combine`
  and is **bounded in [1/k, 1]** since `k_base(x,x)=1`, giving `w ∈ [1, √k]`
  with no clip at all. Measured `w ∈ [4.96, 16.81]` against `√k = 28.3`, spread
  collapses to ~10 nats, max grad 0.09 — zero numerical pathology.
  **But it is empirically inert**: it reproduces the uncorrected kernel.
  *Reason:* a reweight acts only through its **variation across particles**
  (a uniform factor cancels, by the same homogeneity as §3.4). The KDE's
  variation is ~3×; `p_*^{-1/2}` at d=325 varies by hundreds of nats.
- **Ionides truncation** (`min(w, √k·w̄)`, scale-equivariant so evaluable in the
  shifted domain, no free constant) and **ESS targeting** both bound a
  *summary* of the weight distribution — the maximum, the effective sample
  size — but neither bounds its **range**, which is what drives the runaway.
  Both NaN'd.
- **√(d/2) typical-set scale.** For a d-dim posterior, `sd(log p) ≈ √(d/2)`;
  weight variation beyond the target's own intrinsic log-density fluctuation
  is stragglers, not signal. Gives U = 12.75 at d = 325. Lands within a factor
  of two of the empirical optimum but is **not confirmed** — the best measured
  U is ~0.5–0.7× it, and with 3-seed sd up to 24 the middle rows are not
  resolved from each other.

### 3.9 Conclusion

`clip_exponent` is a **one-parameter dial interpolating from under-dispersed to
over-dispersed**: U = 4 reproduces the uncorrected kernel, U = 25.5 overshoots
to 151% on `c`. Good calibration at 20 is the dial being tuned against the
answer, not the paper's mechanism operating as derived.

The tension looks intrinsic rather than an artifact of how the clip is
written: bounding the weights tightly enough to be safe destroys the variation
that does the correcting.

---

## 4. A Stein-identity diagnostic

### 4.1 Derivation

Every adaptive rule above lacked a **target value** that isn't "match NUTS."
Stein's identity supplies one. For `p` with `∫∇(p·f) = 0`, taking `f(x) = x`:

```
E_p[(x - μ)·s(x)ᵀ] = -I         where  s = ∇log p
```

Define

```
R  =  -(1/(k·d)) Σ_i (x_i - x̄)·s(x_i)       →   1  at the target
```

For a Gaussian target `N(μ, Σ)` and an ensemble with covariance `A`,
`s(x) = -Σ⁻¹(x-μ)`, so

```
R = (1/d)·tr(A Σ⁻¹)
```

— i.e. **R directly estimates the variance-inflation factor**, on a scale
where 1 is correct. The setpoint comes from the identity, not from
calibration, and both operands are already computed each step, so it is free.

*Convention note:* `magi.MAGI.gradient` returns `∇log p` **directly**;
`msvgd` negates it (`raw_grad = -self.gradient(...)`) so `_combine` yields a
descent direction for optax. Use `+gradient` as the score.

### 4.2 Validation

On 4000 gold-standard NUTS draws:

```
R_global = 0.9986        (identity predicts exactly 1.0)
per-coordinate R: median 0.996,  q05 0.765,  q95 1.246
theta coords    : 1.040, 0.732, 1.109
```

Finite-sample control — R = 1 is reachable at production ensemble size, so any
gap is not an artifact of k:

| draws | R_global | R_theta |
|---|---|---|
| 200 | 0.9906 | 1.3674 |
| **800** | **0.9991** | 1.1575 |
| 4000 | 1.0026 | 1.0209 |

### 4.3 Results

k = 800, 2 seeds:

| variant | width % of NUTS | R_global | R_theta |
|---|---|---|---|
| standard kernel | [71, 79, 64] | 0.018 | 0.027 |
| reweight U = 4 | [81, 72, 75] | 0.031 | 0.071 |
| reweight U = 9 | [94, 93, 84] | **0.039** | 0.052 |
| reweight U = 18 | [75, 53, 129] | 0.114 | 0.185 |
| reweight U = 25.5 | [76, 57, 146] | 0.245 | 0.154 |
| **NUTS (target)** | [100, 100, 100] | **1.000** | **1.000** |

R responds **monotonically** to U, so it is a valid control signal — but over
the entire admissible range it moves only `0.018 → 0.245` against a setpoint
of 1. The per-particle setting that best matches NUTS θ-widths satisfies
Stein's identity at **4%** of the required level.

**Correction (from the §6 run).** Every row above uses the *per-particle* clip
form (§3.4), needed for the rank-1 reformulation. The **production pairwise
clip = 20** was not in this sweep, and scores considerably better:
**R = 0.281 ± 0.031** over 5 seeds — ~7× the per-particle form, though still
3.5× short of the setpoint. This is consistent with §3.8's principle that a
reweight acts only through its variation across particles: the pairwise clip
binds only when *both* particles are low-density, so it preserves more of the
corrective variation than a per-particle cap at the same worst-case bound. The
two clip semantics are not interchangeable, and the pairwise one is better.

### 4.4 What it exposes: anisotropic collapse

R ≈ 0.04 with intact marginals isolates the failure. The marginals *are*
intact — per-coordinate sd, SVGD (U = 9) / NUTS:

```
theta block      (n =   3): [0.979 0.898 0.862]
trajectory block (n = 322): median 0.840,  q05 0.652,  q95 1.131
coords with sd < 25% of NUTS: 0.0%
coords with sd < 10% of NUTS: 0.0%
```

Nothing is collapsed in scale. Since `R = (1/d)·tr(AΣ⁻¹)`, R is small when the
ensemble lacks spread along directions where `Σ⁻¹` is large — the **stiff,
high-curvature directions**, which for a GP-smoothed trajectory are the
high-frequency modes.

A scalar-bandwidth RBF kernel cannot resolve directions whose scales differ by
orders of magnitude, and no amount of density reweighting changes that, since
the reweight is **isotropic per particle**.

**This also means θ-interval-width-vs-NUTS is a weak validation criterion:**
widths agree at 84–98% while the joint satisfies the identity at 4–28%.

§6 confirms the anisotropy reading directly by decomposing R along the NUTS
principal axes, rather than inferring it.

### 4.5 A dynamic controller for U

Implementable and well-posed, in the spirit of NUTS dual-averaging:

```
log U  ←  log U + η·(1 - R)      # R < 1 → widen the A3 bound; R > 1 → tighten
```

`η` sets convergence speed, not the fixed point, so no magic constant
survives. **But on this problem the controller's verdict is negative:** it
would drive U monotonically to the stability boundary and pin it there, and
calibration there is worse (146% on `c` at U = 25.5, where R is still 0.245).
Dynamic tuning removes the arbitrariness; it does not produce a good answer,
because no U is good by this criterion.

### 4.6 Caveats on R

- R = 1 is **necessary, not sufficient** — it is one moment condition.
  R ≈ 0.04 is nonetheless a decisive *failure* signal.
- The `tr(AΣ⁻¹)` reading is exact only for Gaussian targets.
- `R_theta` on a 3-dim block is noisy (NUTS itself gives 1.37 at k = 200).
  **`R_global` is the reliable one.**

---

## 6. Preconditioning, scored by R

Script: `dev tests/matrix_vs_reweighted_stein_R.py`
→ `matrix_vs_reweighted_stein_R_results.json`.

### 6.1 Method

Matrix-valued kernel `K_Q(x,y) = Q⁻¹ exp(-‖x-y‖²_Q / h)`. The divergence term
is `Q⁻¹ ∇k_Q = -2(x-y)/h · k_Q`, so **Q cancels in the repulsion** and survives
only as an elementwise factor on the drift. `_combine` is therefore reused
unchanged with `drift = Q⁻¹` broadcast over `dim`, and the Q-metric distances
come from feeding `sqrt(diag Q)`-scaled particles to `pairwise_distance`.

`Q` is the diagonal empirical Fisher, **normalized to mean 1**. An overall
scale on `Q` cancels in the kernel (the median-heuristic `h` absorbs it) but
not in the drift, so normalizing makes this a pure *anisotropy* correction and
removes a confound with overall step scale.

All variants run a **fixed iteration budget** (`atol=0`, `max_iter=1000`),
because `Q⁻¹`-rescaled gradients otherwise trip a shared absolute tolerance and
report false convergence — the §2.5 pitfall.

New metric: the **spectral profile**. Project particles onto the eigenbasis of
the NUTS covariance and take `var(ensemble) / eigenvalue` per direction,
binned softest → stiffest. For a Gaussian, R is exactly the mean of these
ratios, so this decomposes R by direction. NUTS covariance condition number:
2.05e4.

### 6.2 Results (3 seeds)

| variant | width % of NUTS | \|dev\| | R_global | R_theta | median sd ratio | spectral profile (soft → stiff) |
|---|---|---|---|---|---|---|
| standard | [75.5, 85.3, 67.3] | 24.0 | 0.023 | 0.032 | 0.756 | [0.032, 0.003, 0.002, **0.000**, **0.000**] |
| reweighted | [108.6, 106.4, 95.0] | 6.7 | 0.287 | 0.093 | 1.042 | [0.390, 0.110, 0.083, 0.141, 0.236] |
| matrix | [71.9, 93.5, 80.4] | 18.0 | 0.069 | 0.094 | 0.796 | [0.084, 0.013, 0.015, 0.073, 0.072] |
| **matrix + reweighted** | [96.6, 101.2, 113.1] | **5.9** | **0.348** | 0.053 | 1.041 | [0.241, 0.047, 0.049, **0.319**, **0.351**] |
| NUTS (target) | [100, 100, 100] | 0.0 | 1.000 | 1.000 | 1.000 | [1, 1, 1, 1, 1] |

5-seed confirmation of the top two:

| variant | R per seed | mean ± sd | \|dev\| |
|---|---|---|---|
| reweighted | 0.307, 0.304, 0.250, 0.238, 0.308 | **0.281 ± 0.031** | 6.1 |
| matrix + reweighted | 0.394, 0.383, 0.268, 0.425, 0.366 | **0.367 ± 0.053** | 6.7 |

### 6.3 Reading

- **The anisotropy diagnosis is confirmed outright.** The standard kernel holds
  essentially **zero** posterior variance (0.000, 0.000) in the stiffest 40% of
  directions, while its median marginal sd ratio is a respectable 0.756. The
  marginals hide a total collapse.
- **Preconditioning does what it was predicted to do.** Pure matrix lifts the
  two stiffest bins from 0.000 → 0.073, 0.072, and R from 0.023 → 0.069 (3×) —
  while *worsening* θ widths (|dev| 24.0 → 18.0 is an improvement, but its
  widths are still far off). Scored on widths alone it looks mediocre; scored
  on the mechanism it targets, it clearly works. This vindicates re-scoring.
- **The reweighted kernel is itself partly an anisotropy correction**, not just
  a scale correction — it reaches 0.141/0.236 in the stiff bins, better than
  pure preconditioning.
- **Stacking helps here**, contra §2.7. Matrix + reweighted is best on R
  (0.367 vs 0.281, ≈3 se over 5 seeds) at statistically indistinguishable
  interval calibration (|dev| 6.7 vs 6.1), and has the flattest profile — the
  only variant where the stiff bins (0.319, 0.351) exceed the mid bins. The
  earlier "do not stack" conclusion was an artifact of scoring on θ widths,
  where overshoot and genuine improvement are indistinguishable.
- **Still 3× short.** Even the best variant sits at R = 0.37 against 1.0. This
  narrows but does not close the gap.

---

## 7. Conclusions and next steps

**Settled.**

- Reweighted kernel is the best *single* technique on width fidelity and is a
  reasonable default — but its good calibration is tuned, not derived.
- Coverage and fidelity conflict here; fidelity is the stated priority.
- The overflow reformulation (§3.4) is exact and free of constants.
- The pairwise clip is meaningfully better than the per-particle clip at the
  same worst-case bound (R 0.281 vs ~0.039), so the rank-1 reformulation is
  **not** a free win — it costs accuracy to buy overflow safety (§4.3).
- The failure is anisotropic, established directly rather than inferred: the
  standard kernel holds ~0.000 of the posterior variance in the stiffest 40%
  of directions while its marginals look fine (§6.3).

**Done.**

1. **`R` is wired in** (`MSVGD._stein_R`). Computed from the raw score before
   any kernel rescales it, carried through the `_run_phase` loop, printed by
   the monitor line and the per-phase summary, and left on `self.stein_R`.
   Verified bit-for-bit behaviour-preserving over 15 configs, and validated to
   recover a known variance-inflation factor (0.2491/0.4983/1.9931 against
   0.25/0.5/2.0).
2. **Preconditioned comparison run and scored by `R`** — §6. Confirms the
   mechanism, and overturns the earlier "do not stack" conclusion for
   matrix + reweighted specifically.

**Revised.**

- *"Stacking overshoots; do not stack"* held for reweighted + Stein, and for
  matrix + reweighted **as scored on θ widths**. Under `R`, matrix + reweighted
  is the best variant found. The general lesson is that θ-width scoring cannot
  distinguish overshoot from genuine improvement.

**Dropped.**

- The proposed B = 30 "does `β` transfer across datasets and dimension?"
  experiment — transferability of `β` is the wrong question.

**Open.**

- Even the best variant reaches R = 0.37 against 1.0. What closes the
  remaining 3×? Candidates: a non-diagonal / low-rank preconditioner (the NUTS
  covariance has condition number 2e4, so a diagonal Q may be too coarse), or a
  mixture-of-preconditioners as in the original matrix-SVGD paper.
- Whether `matrix + reweighted` holds up across the B = 30 datasets, and its
  cost (it was ~equal in wall-clock here, but that is one problem size).
- Whether the MAGI posterior's poor frequentist coverage on `b`/`c` (§1) is a
  modelling issue independent of all of the above.

---

## Appendix: file map

| file | status |
|---|---|
| `msvgd/msvgd/msvgd.py` | production; `_reweighted_svgd_update`, `_stein_R`, `reweighted_kernel` toggle, `k_schedule` |
| `dev tests/matrix_vs_reweighted_stein_R.py` | §6 comparison, + `..._results.json` |
| `magi_msvgd/magi_msvgd/magi.py` | `MAGI(MSVGD)`; `reweighted_kernel=True` default |
| `dev tests/nuts_gold_standard.npz` | 8 × 8000 × 325 gold standard + diagnostics |
| `dev tests/b100_mk_study_results.json` | §2.7 table 1 |
| `dev tests/b30_reweighted_stein_results.json` | §2.7 table 2 |
| `dev tests/msvgd_mk.py`, `magi_mk.py` | MK-SVGD |
| `dev tests/magi_reweighted.py` | standalone reweighted variant |
| `dev tests/test_stein_reweight*.py` | post-hoc Stein reweighting |
| *(deleted)* | all sliced-SVGD files; all matrix-SVGD files; vmap-optimizer tests |
