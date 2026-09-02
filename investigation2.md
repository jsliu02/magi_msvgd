# Increasing the Faithfulness of mSVGD's Posterior Representation — Investigation 2

Follow-up to `investigation.md`. That investigation diagnosed the failure (anisotropic collapse
along stiff directions) but every fix tried was a variation on the SVGD kernel, and the best
result still satisfied Stein's identity at only ~0.37 of the required level. This investigation
widens the search: literature methods not previously tried, adaptations of them, and some
constructions that fall out of the diagnosis itself.

Scripts in `investigation2/`. Nothing in the existing codebase was modified; all variants are
built by transforming the target or monkeypatching within the experiment scripts.

---

## Headline findings

1. **At the baseline data density the FHN/MAGI posterior is Gaussian to within sampling
   resolution, and the Hessian at the MAP is its covariance.** `N(μ_NUTS, H⁻¹)` is
   statistically indistinguishable from the 64k-draw NUTS gold standard (energy 0.0184 vs a
   0.0188 floor at n=4000), and matches on per-coordinate sds and θ intervals too. This reframes
   everything: the 325-dimensional covariance is *available in closed form for ~3 seconds of
   compute*, and the inference problem reduces to locating the mean. **Scope, established in
   §13:** this holds for *well-identified* posteriors. It weakens as data becomes sparse (at a
   quarter of the observations the same construction is 18% under-dispersed in the leading
   directions) and fails outright at σ = 0.5, where it is 14× too wide.

2. **SVGD actively destroys a correct ensemble.** Initialized at exact Laplace samples
   (R = 1.15, profile ≈ 1), standard SVGD drives R to 0.03 and energy from 1.45 to 6.21. Smaller
   learning rates only slow it. The fixed point itself is collapsed, so no initialization,
   bandwidth, or annealing schedule can rescue it.

3. **Whitening by the Laplace metric is the single highest-leverage change.** Every method that
   reaches the sampling floor does so in those coordinates, and every method that fails, fails
   without them — IMQ diverges in x-space, all blockwise variants diverge in x-space, and
   ordinary SVGD collapses there.

4. **Given the metric, several methods reach the floor — and whitened ULA is the one that is
   good on every view.** Energy distance against a 0.048 floor: whitened ULA **0.049**, whitened
   IMQ SVGD 0.053, blockwise (B=1) 0.078, Laplace + a 2-second VI mean 0.083, against current
   production's **3.665**. But energy weights all 325 directions equally while the top 5 carry
   78.5% of the variance, so it must be read next to a variance-weighted ratio (§1) — on which
   ULA scores 0.933 and production 1.409 (43–56% over-dispersed in the leading directions, 88%
   collapsed in the other 305).

5. **The RBF kernel's failure is quantifiable and its two cures are independent.** In 325
   dimensions the median heuristic drives every off-diagonal Gram entry to ≈ 1/k, so each
   particle interacts with **fewer than 5 of its 800 neighbours**. Heavier tails (IMQ) or
   lower-dimensional kernels (blockwise) each fix it; both need whitening first.

6. **A NUTS-free stopping rule.** The whitened flow contracts monotonically, so it has an
   optimal stopping time and running to convergence makes it worse. The Stein diagnostic R
   locates it, with a target fixed by theory rather than a tuned patience. **Stop at R ≤ 1.05**:
   stopping at exactly 1 optimizes energy (0.053 vs 0.072) but over-contracts the dominant
   directions (varwtd 0.824 vs 0.911) and costs 50% on θ interval accuracy.

7. **Importance sampling cannot certify any of this** — ESS of 62, 34 and 1.1 out of 20 000 from
   a proposal at energy 0.11 — and **adding Langevin noise to SVGD is provably and empirically
   wrong** (R = 1.0 → 42 as noise grows), because the repulsion term already *is* the
   deterministic surrogate for the diffusion.

8. **There is a clean failure regime, and it is predictable for free.** At σ = 0.5 the smallest
   Hessian eigenvalue collapses by 680× and `tr(H⁻¹)` by 136×; the Laplace metric becomes
   inappropriate and *every* method built on it — IMQ and ULA alike — diverges immediately at
   every step size. `min eig(H)` is a pre-flight check that predicts this with no reference
   chain. Note the contrast: this approach fails **loudly**, while production at the same
   setting returns a plausible-looking but badly wrong ensemble.

9. **θ interval widths, the criterion used throughout `investigation.md`, rank methods
   backwards.** They put production above the recommended pipeline while the actual
   distributional error differs by 70×. Three separate experiments here reproduce that inversion.


---

## 1. Method and metrics

The previous investigation scored on θ interval widths and the Stein diagnostic R. Widths were
shown there to be a weak criterion, and R is a single Gaussian-flavoured moment condition, so
this investigation adds a genuine distributional metric.

| metric | meaning | target |
|---|---|---|
| `energy` | energy distance to the NUTS sample in NUTS-whitened coordinates | 0.048 (floor) |
| `varwtd` | variance-weighted ratio of ensemble to reference variance along NUTS principal axes | 1.020 (floor) |
| `R_global` | Stein-identity dispersion, `-(1/kd)Σ(xᵢ-x̄)·s(xᵢ)`; `= tr(AΣ⁻¹)/d` for Gaussians | 1 |
| `profile` | R decomposed along NUTS principal axes, binned softest → stiffest | [1,1,1,1,1] |
| `sd_ratio` | median per-coordinate sd ratio to the reference | 1.007 (floor) |
| `bias` | Mahalanobis distance of the ensemble mean from the NUTS mean, per-dim rms | 0.034 (floor) |
| `dev` | mean abs deviation of θ 95% CI widths from NUTS, in % | 2.1–3.5 (floor) |

Energy distance is a proper metric between distributions and assumes nothing about Gaussianity —
but it is **not** sufficient on its own here, for a reason discovered late and documented below.
`varwtd` was added in response and should be read alongside it. Every "floor" above is measured,
not assumed: an independent 800-draw subsample of the gold standard scored through the identical
pipeline.

**Calibration of the energy scale** (exact Gaussians at the NUTS moments, deliberately
perturbed by known amounts) — needed to read any of the tables below:

| perturbation | energy |
|---|---|
| none (= sampling floor at k=800) | 0.048 |
| all sd × 0.97 | 0.053 |
| all sd × 0.94 | 0.070 |
| all sd × 0.90 | 0.108 |
| all sd × 0.80 | 0.330 |
| mean shifted 0.10 sd | 0.167 |
| mean shifted 0.33 sd | 1.280 |

So energy ≈ 0.05 means "indistinguishable at this sample size", 0.11 is roughly a 10% scale
error or an 0.08 sd mean shift, and the production configuration's 3.67 is far outside the
range these perturbations cover.

*Figures for the same method sometimes differ slightly between tables (e.g. gauss-hybrid energy
0.110 in §9 vs 0.112 in §15; whitened ULA dev 3.2 in §4 vs 2.6 in §15). These come from
different experiments with different seeds and are within seed-to-seed noise; each table is
internally consistent, which is what the comparisons within it rely on.*

### A limitation of energy distance, found late and corrected for

Energy distance here is computed in Mahalanobis-whitened coordinates, which gives all 325
directions **equal weight** — while the top 5 eigendirections of the posterior carry **78.5% of
the total variance** (top 20: 95.3%). The metric therefore assigns the directions that dominate
the posterior about 1.5% of its attention. A constructed control makes the blindness explicit —
exact Gaussians at the NUTS moments, perturbed only in their leading directions:

| perturbation | energy | sd ratio | θ dev | profile |
|---|---|---|---|---|
| none | 0.0496 | 1.007 | 4.7 | [1, 1, 1, 1, 1] |
| **top-3 eigendirections × 0.3** | **0.0483** | 0.721 | 35.9 | [1, 1, 1, 1, 1] |
| **top-5 × 0.3** | **0.0492** | 0.665 | 44.5 | [1, 1, 1, 1, 1] |
| **top-20 × 0.6** | **0.0486** | 0.640 | 37.5 | [1, 1, 1, 1, 1] |
| all directions × 0.82 | 0.2735 | 0.799 | 20.9 | [0.67, …] |

Destroying 70% of the variance in the three dominant directions leaves energy **at the floor**,
and the binned profile is equally blind because 3 directions cannot move a 65-direction median.
Only `sd_ratio` and θ `dev` catch it.

This investigation therefore adds a **variance-weighted ratio** as the complementary view:

```
varwtd = Σ_k λ_k · (ensemble variance along axis k) / λ_k   /   Σ_k λ_k
```

i.e. the ensemble-to-reference variance ratio along the NUTS principal axes, weighted by how
much variance each axis actually carries. `varwtd = 1` is correct. It is reported alongside
energy from the master comparison onward, and it **changes the ranking** — see there.

*Every energy figure in this document should be read as "faithfulness across the bulk of the 325
directions", not as an unqualified distance.* The two views are reported together because
neither alone is sufficient: energy is blind to the leading directions, and `varwtd` is blind to
everything except second moments.

### A robustness bug found along the way

`MSVGD._mitotic_split` regularizes the ensemble covariance with an **absolute** `1e-6·I` ridge
before a Cholesky. With `k < dim` that covariance is exactly singular (rank ≤ k−1), so the ridge
is the only thing making the factorization succeed — and being absolute, it only works at the
particle scale it was tuned for (~1e-2, where 1e-6 is ~1% relative). Any reparameterization that
changes the particle scale breaks it: in whitened coordinates (spread ~50) it is negligible and
the Cholesky returns NaN; scaling it to stay relative makes it negligible in x-space instead.
There is no single ridge that works for both.

The fix used throughout investigation2 (`harness.patch_split`) drops the ridge and samples the
covariance-matched jitter inside the ensemble's own span via SVD, which is what a rank-deficient
Gaussian actually is — exact, scale-invariant, and it removes the isotropic 325-dimensional
noise floor the ridge was silently injecting. Verified behaviour-preserving in x-space
(energy 6.742 vs 6.738 standard; 3.665 vs 3.710 reweighted).

This is a latent bug in the current codebase, independent of everything else here.


---

## 2. Where the error actually is

**`exp01_whiten.py`, `exp02_decompose.py`**

At d = 325 the full dense Hessian is affordable — 0.8 s once compiled — versus the diagonal
empirical Fisher that `investigation.md` was restricted to. At the MAP (found in 3.1 s, `logp =
−12.656`, `θ = [0.2003, 0.2438, 2.9589]`) it is positive definite with eigenvalues in [5.31,
9.97e4], condition number 1.88e4.

The Laplace approximation it defines is already better than every SVGD variant:

| approximation | energy | bias | dev | profile soft → stiff |
|---|---|---|---|---|
| N(MAP, H⁻¹) | 1.446 | 0.335 | 6.6 | [0.962, 0.981, 1.015, 1.009, 1.014] |
| **N(μ_NUTS, H⁻¹)** | **0.048** | 0.034 | 5.3 | [0.963, 0.972, 1.019, 1.010, 1.017] |
| N(MAP, Σ_NUTS) | 1.447 | 0.336 | 0.8 | ≈ 1 |
| N(μ_NUTS, Σ_NUTS) | 0.047 | 0.033 | 1.7 | ≈ 1 |
| NUTS k=800 (floor) | 0.048 | 0.034 | 3.5 | ≈ 1 |

Reading down the rows attributes the error: swapping in the true covariance changes nothing
(1.446 → 1.447), swapping in the true mean takes it to the floor (1.446 → 0.048). **The entire
error of the Laplace approximation is the mean.** At higher resolution the conclusion holds —
at n = 4000 the NUTS-vs-NUTS floor is 0.0188 and N(μ, H⁻¹)-vs-NUTS is 0.0184.

The MAP-to-mean displacement is 6.0 whitened units (rms 0.333/dim) and is concentrated in the
stiffest directions:

```
displacement rms by stiffness bin (soft -> stiff): [0.218 0.093 0.142 0.148 0.676]
theta-block displacement: [0.0045 -0.0775 0.0608]   (NUTS sd [0.0197 0.0803 0.0592])
```

— i.e. about one posterior sd on `b` and `c`. This is ordinary posterior skew, not a bug.

**Consequence.** The 325-dimensional covariance structure that `investigation.md` spent its
entire effort trying to recover with kernel modifications is available in closed form for a few
seconds of compute. The inference problem is one of locating a mean.


---

## 3. SVGD destroys a correct ensemble

**`exp04_from_laplace.py`**

Every earlier run started far from the posterior, so "SVGD is underdispersed" was ambiguous
between *the fixed point is wrong* and *it never gets there*. Initializing at exact Laplace
samples — which §2 shows are indistinguishable from NUTS — settles it. Whitened coordinates
are used so the target is near-isotropic, which is the regime the RBF kernel is designed for;
this is SVGD's best case.

| optimizer / kernel | R at it = 0 | 50 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| adam 1e-2, standard | 1.153 | 0.509 | 0.076 | 0.031 | 0.034 |
| adam 1e-3, standard | 1.153 | 1.046 | 0.814 | 0.526 | 0.266 |
| sgd 1e-2, standard | 1.153 | 1.142 | 1.117 | 1.082 | 1.042 |
| **adam 1e-2, reweighted** | 1.153 | **0.797** | **0.783** | **0.778** | **0.773** |

Standard SVGD takes a correct ensemble and collapses it — energy 1.45 → 6.21. The learning-rate
rows show this is the flow, not the optimizer: smaller steps only slow the decay. **No
initialization strategy, bandwidth rule, or annealing schedule can fix standard SVGD, because
its fixed point is collapsed.**

The reweighted kernel behaves qualitatively differently: it reaches a genuine *stable* fixed
point at R ≈ 0.78, unchanged from iteration 50 to 1000, with a **flat** profile
[0.699, 0.713, 0.757, 0.763, 0.774]. In whitened coordinates its deficiency is uniform rather
than anisotropic — which matters, because uniform deficiency is exactly what a scalar
correction can fix exactly (§9).


---

## 4. Fixing SVGD itself: whitening + IMQ

**`exp06_noise.py`, `exp08_imq.py`**

Two changes, and the ablation shows both are required. Whitening removes the anisotropy the
scalar-bandwidth kernel cannot resolve. The IMQ kernel
`k(x,y) = (1 + ‖x−y‖²/h)^(−1/2)` (Gorham & Mackey 2017) replaces RBF's exponential decay with
polynomial tails — in 325 dimensions pairwise distances concentrate, so the RBF Gram matrix
degenerates toward the identity and particles stop interacting, while IMQ keeps them coupled.

| coordinates | kernel | it = 500 | 1500 | 4000 | bias @1500 |
|---|---|---|---|---|---|
| whitened | RBF | 1.417 | 1.359 | 1.236 | 0.322 |
| **whitened** | **IMQ** | 0.088 | **0.051** | 0.062 | **0.035** |
| x-space | RBF | 1.342 | 2.704 | 4.332 | 0.072 |
| x-space | IMQ | diverged | — | — | — |

(energy distance; floor 0.048)

- **IMQ without whitening diverges.** The heavy tails that help in a well-conditioned metric are
  destabilizing across a 1.9e4 condition number.
- **Whitening without IMQ never corrects the mean** — RBF's bias stays at 0.30–0.33, i.e. it
  never leaves the MAP, and its profile slowly decays.
- **Together they reach the floor**: energy 0.051 against 0.048, bias 0.035 against 0.034,
  R = 1.010, profile [0.921, 0.961, 0.993, 0.967, 0.980].

Against the current production configuration (x-space reweighted, energy 3.665) this is a
**72× reduction** in energy distance, and unlike §9's corrections it is a genuine particle
ensemble produced by an SVGD flow. Two qualifications established later: energy under-weights
the leading directions, where this configuration is 18% short (§15), and stopping at R ≤ 1.05
rather than R ≤ 1.0 trades a little energy for a lot of leading-direction accuracy (§11).

The x-space RBF row is worth noting separately: its energy *grows* with iteration
(1.342 → 2.704 → 4.332) while its θ interval widths stay near 90% of NUTS. This is
`investigation.md`'s "θ widths are a weak criterion" in its sharpest form — by the width
criterion the run looks stable, while it is in fact getting monotonically worse.

### Preconditioned ULA is a strong, much simpler baseline

The same sweep included the kernel-free limits. Pure gradient flow with no noise and no kernel
collapses to a point (energy 10.7), as it must. But **whitened ULA** — `x ← x + lr·∇log p +
√(2·lr)·z`, 800 parallel chains, no kernel at all — reaches the floor. Verified over 3 seeds and
a step-size sweep (`exp16_ula.py`, 2000 steps):

| step size | energy | R | bias | θ widths % | dev | sd ratio |
|---|---|---|---|---|---|---|
| 3e-3 | 0.055 | 0.999 | 0.043 | [89.8, 95.5, 96.4] | 6.1 | 0.979 |
| **1e-2** | **0.049** | 1.004 | 0.036 | [97.4, 99.7, 101.8] | **3.2** | 0.995 |
| NUTS floor | 0.048 | 1.003 | 0.034 | [103.2, 95.0, 97.7] | 3.5 | 1.003 |

At lr = 1e-2 it matches the sampling floor on **every** metric including θ interval widths,
where the SVGD recipe is ~10% narrow. ULA is biased by discretization at any finite step size,
and the lr sweep shows that bias appearing as lr shrinks toward the noise floor rather than
growing — at these step sizes discretization bias is below what 800 draws can resolve.

If the goal is simply a faithful posterior sample and the Hessian is affordable, **this is the
best and simplest thing in this investigation**: no kernel, no bandwidth, no mitosis, no
stopping rule.


---

## 5. Why IMQ works: the RBF Gram matrix degenerates

**`exp13_gram.py`**

In high dimension pairwise squared distances concentrate, so with the median heuristic
`h = median(L2)/log k` essentially every off-diagonal entry sees `L2/h ≈ log k`, giving

```
RBF:  exp(-L2/h)         ->  exp(-log k)       = 1/k
IMQ:  (1 + L2/h)^(-1/2)  ->  (1 + log k)^(-1/2)
```

Measured on real whitened ensembles, the predictions are almost exact:

| k | mean L2/h | (log k) | RBF off-diag | (1/k) | IMQ off-diag | ratio | RBF effective neighbours |
|---|---|---|---|---|---|---|---|
| 200 | 5.310 | 5.298 | 5.34e-3 | 5.0e-3 | 0.399 | 74.6 | 4.26 |
| 800 | 6.696 | 6.685 | 1.40e-3 | 1.25e-3 | 0.361 | 257.6 | 4.56 |
| 3200 | 8.087 | 8.071 | 3.73e-4 | 3.1e-4 | 0.332 | 892.2 | 4.93 |

Two things follow.

**The RBF Gram matrix is effectively the identity.** With 800 particles in 325 dimensions, each
particle interacts with **fewer than 5 others** (participation ratio 4.56). The repulsion term
is therefore negligible, each particle is driven independently by the drift, and the ensemble
collapses onto the MAP. This is the mechanism behind every collapse observed in
`investigation.md` and in §3 here.

**Adding particles does not fix it.** RBF off-diagonals shrink like `1/k` (because
`median/log k` pushes `L2/h` up as k grows) and the effective neighbour count stays pinned near
4.5 — so each particle's *interaction neighbourhood does not grow*, and IMQ's relative advantage
widens with k (74.6 → 257.6 → 892.2).

Empirically more particles do help, but only in the soft subspace. Ordinary x-space SVGD at
2000 iterations (`exp12_blocked.py`):

| k | R | energy | profile soft → stiff |
|---|---|---|---|
| 200 | 0.056 | 5.377 | [0.138, 0.010, 0.006, 0.001, 0.000] |
| 800 | 0.180 | 3.186 | [0.499, 0.157, 0.124, 0.001, 0.001] |
| 3200 | 0.433 | 1.277 | [0.756, 0.573, 0.537, 0.014, 0.004] |

A 16× increase in particles moves R from 0.06 to 0.43 and fills the *softest* directions
(0.138 → 0.756), while **the two stiffest bins stay at ~0.001–0.014 throughout**. Brute-force
particle scaling buys the easy directions and never touches the hard ones, so it is not a
practical route to faithfulness — but it is not literally counterproductive, and an earlier
draft of this note overstated that.

IMQ's off-diagonals sit at 0.33–0.40 regardless of k, so the Gram matrix stays dense and the
repulsion survives. That is the whole reason §4 works.

The x-space row of the same table adds a detail: there the coefficient of variation of `L2/h` is
0.52 versus 0.076 whitened. Anisotropy makes distances *heterogeneous* rather than concentrated,
which is a different pathology — and the one whitening removes.


---

## 6. Blockwise kernels: a second, independent fix

**`exp12_blocked.py`**

If the RBF failure is concentration (§5), the other escape is to never apply a kernel in 325
dimensions at all — partition the coordinates into blocks and give each block its own kernel
(message-passing / graphical SVGD, Zhuo et al. 2018). Sweeping block size over the divisors of
325, in whitened coordinates, with B = 325 recovering ordinary SVGD as a built-in control:

| block size B | energy | bias | dev |
|---|---|---|---|
| **1** | **0.078** | 0.059 | 11.0 |
| 5 | 0.578 | 0.206 | 14.1 |
| 13 | 1.048 | 0.281 | 14.8 |
| 25 | 1.204 | 0.302 | 14.5 |
| 65 | 1.296 | 0.314 | 14.3 |
| 325 (= ordinary SVGD) | 1.332 | 0.318 | 14.1 |

Strictly monotone: **the lower-dimensional the kernel, the better**, with the fully
coordinate-wise kernel (B=1) reaching energy 0.078 against ordinary SVGD's 1.332 — a 17×
improvement from nothing but restricting the kernel's dimension. The B=325 row reproduces
exp08's whitened-RBF result (1.332 vs 1.359), so the sweep is internally consistent.

So there are **two independent fixes for the same pathology**: heavier kernel tails (IMQ, 0.053)
or lower-dimensional kernels (B=1, 0.078). IMQ is slightly better and requires no choice of
block structure.

**Every x-space blockwise run diverged**, at every block size. Whitening remains the
prerequisite.

### This reconciles the sliced-SVGD failure in `investigation.md`

`investigation.md` (sliced SVGD) rejected axis-aligned sliced SVGD because it is *blind to
correlation* — demonstrated on a correlated target, and confirmed as harmful on MAGI. The B=1
row here is the same idea in kernel form, and it works. The difference is the whitening: axis-
aligned methods are blind to correlation, so removing the correlation first is precisely what
they need. That earlier negative result was a consequence of applying an axis-aligned method in
a strongly correlated coordinate system, not of the method itself.


---

## 7. How much preconditioner structure is required? (all of it)

**`exp11_precond.py`**

The recipe needs a metric, and the full dense Hessian costs O(d) gradient evaluations, O(d²)
memory and O(d³) to factor. Testing weaker structure:

| preconditioner | condition number | energy | θ width % of NUTS | sd ratio |
|---|---|---|---|---|
| **full H** | 1.88e4 | **0.053** | [86.7, 90.7, 88.4] | 0.931 |
| diag(H) | 3.05e1 | 0.595 | [38.6, 13.3, 19.6] | 0.179 |
| block-time (2×2 per time point + θ) | 3.05e1 | 0.596 | [38.7, 13.4, 19.8] | 0.179 |
| GP prior precision `β⁻¹C⁻¹` (Hessian-free) | 2.79e7 | diverged | — | — |
| identity (x-space) | — | diverged | — | — |

**The condition numbers explain the whole table.** `diag(H)` has condition number **30**, against
the full Hessian's **18 800**. The posterior's ill-conditioning is therefore almost entirely in
the *correlations*, not in differing marginal scales — so a diagonal preconditioner can remove at
most a factor of 30 out of 18 800, and a block-time preconditioner adds essentially nothing
(30.5 vs 30.45) because the dominant coupling is the GP prior's dense long-range structure, not
the ODE's local structure.

This retrospectively explains `investigation.md` (matrix SVGD): the diagonal empirical-Fisher
matrix-SVGD variant was never going to work, for reasons visible in one condition number.

The GP prior alone is far *more* ill-conditioned than the posterior (2.8e7 vs 1.9e4) — the data
regularizes the prior's stiffest modes — so the Hessian-free shortcut is not available.

**Consequence for scalability.** This is the main limitation of the recommended pipeline: it
needs the full dense Hessian, so it is comfortable to roughly d ~ 10³–10⁴ and awkward beyond.
A low-rank-plus-diagonal approximation capturing the leading correlations is the obvious next
thing to try and was not tested here.


---

## 8. How accurate must the metric be? (very)

**`exp18_hrobust.py`**

H is the expensive ingredient, so it matters whether a sloppy version suffices — that determines
whether the MAP solve can be truncated, whether H can be reused across nearby datasets, and
whether a low-rank approximation is viable.

| metric | energy | θ widths % | dev | sd ratio | profile soft → stiff |
|---|---|---|---|---|---|
| **exact H at the MAP** | **0.069** | [91.5, 95.0, 90.4] | 7.7 | 0.956 | [0.94, 0.98, 1.01, 0.98, 0.99] |
| H at MAP + 1 sd | diverged (iter 2) | — | — | — | — |
| H at MAP + 3 sd | diverged (iter 4) | — | — | — | — |
| 0.99·H + 0.01·diag(H) | 0.792 | [49.9, 20.9, 38.1] | 63.7 | 0.405 | [0.86, 0.97, 1.00, 0.99, 1.00] |
| 0.90·H + 0.10·diag(H) | 0.801 | [41.0, 14.4, 25.3] | 73.1 | 0.240 | [0.66, 0.96, 0.98, 0.98, 1.03] |
| 0.50·H + 0.50·diag(H) | 0.732 | [38.6, 13.6, 21.2] | 75.5 | 0.188 | [0.40, 1.04, 1.04, 0.99, 1.19] |

**A 1% admixture of the diagonal costs a factor of 11 in energy distance** (0.069 → 0.792) and
collapses the θ intervals to 21–50% of NUTS. The metric has essentially no tolerance.

**Evaluating H away from the MAP is worse still** — and the reason is instructive. The Frobenius
relative error is *tiny*:

```
point       relFro    min eig      max eig     cond        n<=0
MAP         0.0000    5.3074e+00   9.97e+04    1.88e+04       0
MAP+1sd     0.0042   -4.1590e+00   9.97e+04    9.97e+34       1   <- indefinite
MAP+3sd     0.0068    3.6982e-02   9.97e+04    2.70e+06       0
```

A 0.4% Frobenius perturbation flips the smallest eigenvalue **negative**. The Frobenius norm is
dominated by the large eigenvalues and is blind to exactly the small ones that whitening depends
on — `H^(−1/2)` amplifies the softest directions most, so any error there is magnified rather
than damped. Note the shrinkage rows fail in the **soft** bin (profile 0.86 → 0.40) while the
stiff bins stay near 1: a bad metric fails in the opposite direction from a missing metric.

**Practical consequences:**

- The MAP must be genuinely converged, and H checked positive definite. Do not truncate the MAP
  solve.
- Do not reuse H across datasets or shrink it toward anything.
- This **weakens the low-rank-plus-diagonal suggestion** made in the preconditioner section: any
  approximation that perturbs the small eigenvalues is likely to fail the same way. A low-rank
  correction would have to be built to preserve the *bottom* of the spectrum, not the top, which
  is the harder direction for randomized methods. Scalability beyond a dense Hessian remains the
  main open problem.


---

## 9. Post-hoc corrections

**`exp03_hybrid.py`, `exp05_pipeline.py`**

If the covariance is known analytically and SVGD's *mean* is good (bias 0.068 for the standard
kernel, versus 0.335 for the MAP), the two failure modes are complementary and can be combined
without changing the sampler at all.

| method | energy | R | bias | dev |
|---|---|---|---|---|
| raw standard SVGD | 6.742 | 0.022 | 0.068 | 27.4 |
| raw reweighted SVGD | 3.665 | 0.244 | 0.176 | 5.0 |
| affine-recalibrate reweighted (`T A Tᵀ = H⁻¹`) | 0.459 | 0.950 | 0.176 | 5.4 |
| **whitened reweighted + Stein inflation + SVGD mean** | **0.096** | 1.007 | 0.068 | 18.8 |
| **gauss-hybrid: N(mean_SVGD, H⁻¹)** | **0.110** | 1.114 | 0.077 | 5.0 |
| NUTS k=800 floor | 0.048 | 1.003 | 0.034 | 3.5 |

Notes:

- **The standard kernel gives a better mean than the reweighted one** (bias 0.068 vs 0.176), so
  the reweighting adopted in `investigation.md` to fix the variance actively *hurts* the only
  quantity the hybrid needs. Under this pipeline, `reweighted_kernel=False` is preferable.
- **Stein inflation** (`x ← μ + (x−μ)/√R`) sets R = 1 by construction, but is only *valid* when
  the profile is flat. It is, in whitened coordinates (§3), and there it works:
  profile [0.70…0.78] → [0.90, 0.92, 0.98, 0.99, 1.00]. Applied to the non-flat x-space ensemble
  as a control it is a disaster — θ widths blow to [215, 209, 191]% and R to 3.96 — because it
  must overshoot the soft directions to fill the stiff ones.
- **Affine recalibration** preserves whatever non-Gaussian shape the ensemble found, and is the
  right choice if the posterior is *not* Gaussian. It works on the reweighted ensemble
  (profile → ≈1) but not the standard one, whose stiff directions have essentially zero variance
  and so cannot be inflated back — `A^(−1/2)` amplifies numerical noise instead.

### The mean does not need SVGD at all

**`exp17_sav.py`.** Since §2 reduced the problem to locating a mean, and §9 obtained that mean
from an SVGD run, the obvious question is whether the SVGD run is necessary. It is not. Fixing
base samples `z_i` and maximizing the deterministic objective
`L(μ) = (1/n) Σ log p(μ + H^(−1/2) z_i)` with Adam gives the KL-optimal Gaussian mean directly:

| n samples | optimizer | bias | seconds |
|---|---|---|---|
| 256 | adam 1e-3 | 0.0764 | 1.5 |
| 256 | adam 1e-2 | 0.0745 | 1.4 |
| 1024 | adam 1e-3 | 0.0541 | 2.4 |
| **1024** | **adam 1e-2** | **0.0513** | **2.3** |
| — | (MAP) | 0.335 | — |
| — | (mean of a full SVGD run) | 0.068 | ~4 s + setup |
| — | (floor) | 0.034 | — |

**2.3 seconds beats the mean of a full SVGD run.** Scoring `N(μ_SAV, H⁻¹)` gives energy
**0.083**, dev 5.3, R 1.084 — better than gauss-hybrid's 0.110, with no SVGD anywhere in the
pipeline. Total cost end to end is roughly 2 s (MAP) + 1 s (Hessian) + 2 s (VI) ≈ **5 seconds**,
against 125 s for the NUTS reference and a 44× better energy distance than current production.


---

## 10. The recipe, and a stopping rule

**`exp09_verify.py`**

The whitened-IMQ flow slowly over-contracts, so unlike a converging optimizer it has an optimal
stopping time and running longer makes it worse. R locates that time with a target fixed by
theory rather than a tuned patience: **stop the first time R descends through 1.**

| config | stop iter | energy @ stop | best energy | best iter | energy @ 3000 |
|---|---|---|---|---|---|
| k=800 seed 0 | 1900 | 0.0525 | 0.0509 | 1200 | 0.0570 |
| k=800 seed 1 | 1700 | 0.0535 | 0.0524 | 1200 | 0.0588 |
| k=800 seed 2 | 2000 | 0.0530 | 0.0510 | 1200 | 0.0572 |
| k=800 seed 3 | 2200 | 0.0540 | 0.0518 | 1200 | 0.0569 |

The rule lands within 2–4% of the best achievable energy on every seed, and beats running to
3000 iterations. It costs nothing: both operands are already in the loop.

### Two caveats, both load-bearing

**The rule needs an over-dispersed start,** because it detects a *downward* crossing. Started
from a tight ball at the MAP (R < 1 from the outset) it fires at the first checkpoint.

**More importantly, the Laplace initialization is not a convenience — it is required *for the
SVGD flow*.** SVGD is a contraction here; it does not expand:

| initialization | best energy achieved |
|---|---|
| Laplace samples, `y ~ N(0,I)` | **0.051** |
| tight ball at the MAP (`0.05·N(0,I)`) | 6.86 |
| the standard MAGI initialization | diverged |

Starting collapsed, the flow never fills the posterior out — the same one-way behaviour §3
showed from the other side. (This is specific to the deterministic flow. Langevin is ergodic and
forgets its initialization: whitened ULA from a tight ball reaches an identical answer by 2000
steps. It is only *wider* starts that are unsafe there — 3·N(0,I) diverges.) The Laplace approximation supplies a correctly-dispersed starting
ensemble, and the flow's job is only to correct its mean and shape. That is a coherent division
of labour, but it does mean the recipe is Laplace-plus-SVGD, not SVGD alone.

**Particle count matters as it must:** k=200 reaches energy 0.106 against k=800's 0.051, since
the achievable floor itself scales with k.

### Recommended pipeline

```
1.  x_MAP  <- solve(is_MAP=True)                      #  ~2 s
2.  H      <- -hessian(logdensity)(x_MAP)             #  ~0.8 s at d=325;  CHECK H is PD
3.  y_0    ~ N(0, I)             in coordinates x = x_MAP + H^(-1/2) y
4.  iterate in y, monitoring R:
       IMQ SVGD flow            -- deterministic particles, best theta marginals
       or ULA (add sqrt(2*lr)*z) -- simpler, and best on every aggregate metric
5.  stop when R first descends through 1.05          #  not 1.0 -- see section 11
6.  return x = x_MAP + H^(-1/2) y
```

Every step is NUTS-free. Measured cost is 3.0 s of setup plus 0.2–0.7 s of iteration (see the
cost table under Recommendations), against 125 s for the NUTS reference in `investigation.md`'s
B=30 study — i.e. **faster than the current production configuration**, not slower.

Step 3 is load-bearing rather than a convenience: the flow only contracts, so it must start
over-dispersed, and `N(0, I)` in these coordinates is exactly the Laplace approximation. Step 5
detects a *downward* crossing, which is why step 3 matters for it too.


---

## 11. The stopping threshold is a real knob

**`exp14_threshold.py`** (3 seeds)

| stop at R ≤ | iters | energy | θ widths % of NUTS | dev | sd ratio |
|---|---|---|---|---|---|
| 1.30 | 50 | 1.018 | [99.5, 98.2, 92.8] | 4.6 | 0.970 |
| 1.20 | 50 | 1.018 | [99.5, 98.2, 92.8] | 4.6 | 0.970 |
| 1.10 | 183 | 0.461 | [98.0, 97.6, 92.3] | 5.0 | 0.967 |
| **1.05** | 617 | **0.072** | [94.4, 96.2, 91.2] | **6.5** | 0.955 |
| **1.00** | 1850 | **0.053** | [89.0, 92.7, 89.6] | 9.6 | 0.933 |
| NUTS floor | — | 0.048 | [103.2, 95.0, 97.7] | 3.5 | 1.003 |

There is a genuine trade-off, and it should be stated rather than hidden: **stopping at R = 1
optimizes the joint distribution; stopping at R ≈ 1.05 gives better θ interval widths.** At
R = 1 the θ intervals are ~10% narrow, which matters because those three numbers are what a MAGI
user reports.

The top two rows are instructive in their own right. At R ≤ 1.2 the θ widths look excellent
(99.5, 98.2, 92.8, dev 4.6) while the energy distance is 1.018 — the ensemble is still sitting
at the MAP with the wrong mean. Good widths, wrong location. This is the third time in this
investigation that θ widths have ranked things backwards.

**Recommendation:** stop at R ≤ 1.05 by default. It gives 93% of the energy improvement at
two-thirds better θ calibration and a third of the iterations.


---

## 12. Marginal shape fidelity: the case for particles

**`exp15_shape.py`**

The "posterior is Gaussian" result of §2 is a statement about the joint in Mahalanobis geometry,
where energy distance operates. It is **not** a claim that individual marginals are Gaussian —
and they are not. Measured on the 64k gold standard:

```
|skew| over 325 marginals : median 0.070,  q95 0.444,  max 0.811
theta skew                : [0.063, -0.219, -0.050]     (theta_b is 23 SE from zero)
|excess kurtosis|         : median 0.048,  q95 0.388,  max 1.140
```

A Gaussian resample cannot reproduce this by construction; a particle method can. Testing on the
θ marginals — 1-D Wasserstein distance in units of 1e-3 posterior sd, lower better:

| method | W₁(a) | W₁(b) | W₁(c) | sd % (a, b, c) |
|---|---|---|---|---|
| **whitened IMQ** | **97** | **489** | **469** | 92, 93, 90 |
| x-space reweighted (current) | 186 | 783 | 648 | 111, 108, 97 |
| gauss-hybrid | 295 | 976 | 975 | 105, 106, 92 |
| NUTS k=800 (floor) | 55 | 36 | 34 | 97, 99, 101 |

**Whitened IMQ is the best of the three on every θ marginal**, roughly 2× better than the current
production configuration and 2–3× better than the Gaussian resample. This is the concrete reason
to prefer the particle pipeline over gauss-hybrid despite their similar energy distances: the
particles carry marginal shape that a Gaussian cannot.

Two honest caveats. First, all three methods are far above the floor (34–55) on θ — the θ
marginals are the hardest part of this posterior and none of these methods nails them. Second,
**skewness is not resolvable at k=800**: its standard error is √(6/800) = 0.087 against a true
θ_b value of −0.219, so the per-method skew column is too noisy to adjudicate and is not
evidence for or against any method here.


---

## 13. Does any of this generalize?

**`exp10_generalize.py`**

Everything above was established on one FitzHugh–Nagumo dataset at one noise level and
observation density. Both control how well-identified the posterior is, so they are the natural
axes along which the Gaussianity result should fail if it is going to. Each setting gets its
**own independent NUTS reference** (4 chains × 3000 draws after 1000 warmup, CPU, fp64), so the
baseline row doubles as an independent replication of the whole investigation against a
different chain from the gold standard.

| setting | NUTS quality | N(MAP,H⁻¹) | **N(REFmean,H⁻¹)** | floor | production | whitened IMQ |
|---|---|---|---|---|---|---|
| baseline, all obs | R̂ 1.0019, 0 div | 1.242 | **0.049** | 0.044 | 3.633 | **0.051** |
| half the observations | R̂ 1.0024, 0 div | 1.605 | **0.051** | 0.045 | 2.410 | 0.130 |
| quarter of the observations | R̂ 1.0036, 0 div | 2.326 | **0.050** | 0.048 | 2.236 | 0.378 |
| **noisy, σ = 0.5** | R̂ 1.0069, 0 div | 28.60 | **24.61** | 0.045 | 8.620 | **diverged** |

*(energy distance against each setting's own reference)*

### It replicates, and then it breaks

**The baseline replicates against an independent chain.** Production 3.633 (vs 3.665 on the gold
standard), whitened IMQ 0.051 (vs 0.053), Gaussianity 0.049 against a 0.044 floor. None of the
headline numbers were an artifact of the particular reference used.

**Across observation density, Gaussianity looks stable by energy but is quietly degrading.**
0.049 / 0.051 / 0.050 against floors of 0.044 / 0.045 / 0.048 — yet the quantities energy is
blind to (§1) tell a different story:

| setting | N(REFmean,H⁻¹) sd ratio | its θ dev | floor sd ratio | floor dev |
|---|---|---|---|---|
| baseline | 0.987 | 6.9 | 1.034 | 4.4 |
| half the observations | 1.015 | 11.5 | 1.017 | 4.4 |
| **quarter of the observations** | **0.816** | **20.4** | 0.995 | 2.8 |

At a quarter of the observations the exact-mean Gaussian is 18% under-dispersed per-coordinate
with θ intervals 20% off, while its energy sits at the floor — precisely the failure mode the
constructed control in §1 showed energy cannot see.

**At σ = 0.5 the approach fails outright.** `N(μ, H⁻¹)` is **14× too wide** (sd ratio 13.9,
θ dev 1280), and *every* method built on the Laplace metric diverges — whitened IMQ at
lr = 1e-2, 1e-3 and 1e-4, and whitened ULA at all three as well, immediately. Since both
samplers fail identically, the problem is **the metric, not the flow**.

### The failure is predictable without a reference chain

The Hessian spectrum says so directly:

| setting | min eig(H) | cond(H) | tr(H⁻¹) | Laplace sd, flattest direction |
|---|---|---|---|---|
| baseline σ=0.2 | 5.307 | 1.9e4 | 0.98 | 0.43 |
| half the observations | 1.784 | 7.4e4 | 1.68 | 0.75 |
| quarter of the observations | 1.591 | 1.0e5 | 1.88 | 0.79 |
| **noisy σ=0.5** | **0.0078** | **3.1e7** | **133.2** | **11.36** |

At σ = 0.5 the smallest eigenvalue collapses by 680× and `tr(H⁻¹)` by 136×. The posterior has a
near-flat direction in which the quadratic approximation predicts an sd of 11.4, while the true
posterior is bounded there by the ODE's cubic term rather than by curvature. Whitening stretches
that direction by an order of magnitude and puts the initial ensemble straight into the region
where the cubic explodes — hence immediate divergence, and hence the 14× over-dispersion of the
Gaussian. Both symptoms have one cause, and `min eig(H)` predicts it for free.

### What to conclude

- The Gaussianity result and the recipe are **not general properties of MAGI posteriors**. They
  are properties of *well-identified* MAGI posteriors, and identification is measurable in
  advance from the Hessian spectrum.
- Across the identification sequence the recipe degrades gracefully then fails abruptly: 1.2× its
  floor at baseline, 2.9× at half, 7.9× at quarter, divergence at σ=0.5.
- **It fails loudly.** Divergence is impossible to miss. Contrast production at the same setting:
  energy 8.62 and sd ratio 0.696, i.e. badly wrong but returning a plausible-looking ensemble —
  the same silent-failure pattern as §15's 88% collapse. A method that refuses to run when its
  assumptions break is preferable to one that quietly returns the wrong answer.
- The honest scope of the recommendation is therefore: **use the Laplace metric where the pre-flight
  check passes, and fall back to NUTS where it does not.** At σ=0.5 nothing here beats NUTS, and
  nothing here pretends to.

## 13b. Projected SVGD (pSVGD), tested

**`exp22_psvgd_spectrum.py`, `exp23_psvgd.py`** — Chen & Ghattas 2020.

pSVGD attacks the same diagnosis as §5 from the other side: rather than fixing the kernel, it
reduces the dimension the kernel sees. It builds the gradient information matrix of the
*likelihood*, `H = E[∇log f ∇log fᵀ]`, solves the generalized eigenproblem `Hψ = λΓψ` against the
prior covariance `Γ`, runs SVGD on the top-`r` coefficients, and **freezes the remaining
`d − r` directions at prior draws**. Given that §6 found kernel dimension to be the single
strongest lever (B=1: 0.078 vs B=325: 1.332), this is the principled version of that idea and
deserved a test rather than the hand-wave it got in §16 of an earlier draft.

Implemented in prior-whitened coordinates, which is equivalent to the paper's formulation and
makes the coefficient prior exactly `N(0, I)`: with `Γ = Ls Lsᵀ` and `u = Ls⁻¹x`, diagonalize
`S = Lsᵀ H Ls`, keep `V_r`, and note that `∇_w log π(w) = V_rᵀ Lsᵀ ∇_x log p(x)` reproduces the
paper's Eq. 28 exactly — the `−w` prior term falls out on its own. MAGI's θ prior is improper, so
`Γ` uses the GP prior for the trajectory and a broad `N(0, 10²)` for θ; since `f` is *defined* as
`p / p₀`, this choice leaves the target posterior exactly unchanged.

### The premise does not hold, and the test does not depend on the sampler

Rather than tune a sampler, the assumption can be tested directly at its ceiling: keep the
**exact posterior** coefficients (from gold-standard draws, with the basis also built from them)
and freeze the complement at prior draws. No sampler can do better than this.

| r | energy | θ widths % of NUTS | sd ratio |
|---|---|---|---|
| 5 | 1019 | [5095, 2641, 3493] | 52.4 |
| 20 | 253 | [1783, 1367, 2777] | 26.1 |
| 50 | 78.3 | [551, 931, 955] | 9.63 |
| 100 | 23.1 | [226, 304, 313] | 3.19 |
| 200 | 0.671 | [111, 109, 112] | 1.13 |
| 300 | 0.050 | [105, 100, 101] | 1.00 |
| 324 | 0.051 | [105, 100, 101] | 1.00 |
| *true NUTS draws* | *0.051* | *[105, 100, 101]* | *1.00* |

The `r = 324` row reproducing the NUTS draws exactly confirms the projection machinery is
correct. **Acceptable accuracy needs `r ≈ 300` of 325 — an 8% dimension reduction, which is
useless.** A control isolates the cause: freezing the complement at its *posterior* value
instead of a prior draw gives energy 6.8 at `r=5` and 1.85 at `r=50`, against 1019 and 78.3. The
error is entirely the prior complement.

### Why: MAGI has no prior-dominated subspace

```
prior sd / posterior sd :  theta [507, 125, 169]
                           trajectory  median 61x,  q05 22x,  min 15x
coordinates where the prior is within 2x of the posterior:  0.0%
lambda > 1    in 73% of directions   (likelihood dominates the prior)
lambda < 0.01 in  7% of directions   (22 of 325 -- the exploitable complement)
```

pSVGD assumes the posterior equals the prior off a small informed subspace. Here the prior is
15--500× wider than the posterior in **every single coordinate**, so substituting prior draws
injects enormous excess variance — visible as the sd ratios of 9.6--52 above.

The structural reason is specific and worth stating, because it says when the method *would*
apply. In the paper's PDE example `d = 16,641` with 49 observations, so ~50 directions are
informed and >16,500 are genuinely prior-dominated. MAGI has 82 observations but also an **ODE
term, which sits on the likelihood side of the split and couples every coordinate**. It is not
Gaussian, so it cannot be folded into `Γ`. That single structural fact removes the
prior-dominated complement the method needs.

*(A direct run of the algorithm, adaptive basis and all, confirms this: across `r ∈ {5,…,325}`,
three optimizers and two initializations, the best energy achieved was 165 — worse than doing
nothing. But the ceiling calculation above is the informative result, since it holds for any
sampler.)*

---

## 13c. Safer and faster than MALA: HMC and pCN

**`exp25_faster.py`** — MALA is a random walk driven by a first-order integrator, so it is
neither the safest nor the fastest option available once the whitened metric is in hand.

**HMC** takes `L` leapfrog steps per proposal. In whitened coordinates the target is
near-isotropic, which is HMC's best case, and its dimensional scaling is better
(`d^-1/4` vs MALA's `d^-1/3`). **pCN** proposes `y' = sqrt(1-rho^2) y + rho*xi`, which preserves
`N(0,I)` *exactly*: unconditionally stable, no step-size restriction, and no gradients at all —
only density evaluations.

| method | evals/chain | accept | energy | varwtd | wall-clock |
|---|---|---|---|---|---|
| MALA | 2500 grad | 0.56 | 0.052 | 1.026 | 2.1 s |
| HMC ε=0.3 L=10, n=30 | **300 grad** | 0.66 | 0.075 | 1.028 | 1.5 s |
| HMC ε=0.3 L=10, n=100 | 1000 grad | 0.67 | 0.061 | 1.070 | 1.6 s |
| **pCN ρ=0.15** | 20000 density | 0.51 | **0.049** | 0.994 | **1.5 s** |
| pCN ρ=0.05 | 20000 density | 0.84 | 0.049 | 0.966 | 1.7 s |
| NUTS floor | — | — | 0.048 | 1.020 | 125 s |

**HMC is the most gradient-efficient**: 300 gradient evaluations for energy 0.075, against
MALA's 2500 for 0.052 — roughly an 8× reduction in gradient count, as its scaling predicts.
**pCN is the most accurate and the fastest in wall-clock**, reaching the sampling floor in 1.5 s,
because a density evaluation without autodiff is several times cheaper than a gradient and the
extra steps more than pay for themselves.

*(A pre-test predicted pCN's viability at no cost: acceptance is governed by the spread of
`r = log p − log N(x; mu, H^-1)`, measured at `sd(r) = 14` nats. That rules out the `rho = 1`
independence-sampler limit — acceptance would be `~e^-14` — while implying `rho <~ 0.07` works,
which is what the sweep found.)*

### pCN is structurally immune to the stickiness of §13b

This is the more important result. pCN's proposal **contracts toward the reference mean** by
`sqrt(1-rho^2)`, so a chain stranded in the far tail is pulled back rather than frozen — exactly
the failure mode that defeats every local-proposal method there. On the super-quadratic targets,
started from the same `N(0,I)` that froze MALA:

| p | exact E[x²] | MALA | HMC (ε=0.1, L=20) | **pCN (ρ=0.1)** |
|---|---|---|---|---|
| 6 | 0.43719 | raised: frozen chains | ratio 1.015, 0.3% frozen | **ratio 0.9960**, acc 0.89 |
| 10 | 0.39992 | raised: frozen chains | ratio 2.001, 38% frozen | **ratio 0.9958**, acc 0.87 |

HMC shares MALA's pathology, as it must — it is also a local Metropolis method, so from far out
every proposal is rejected. Smaller steps help at p=6 and not at p=10 (68% frozen at ε=0.3,
still 38% at ε=0.1). pCN recovers the correct answer to 0.4% in both.

### The one caveat

At MAGI σ=0.5, pCN stays finite and never freezes, but does not converge within the budget: the
whitened ensemble sd sits at 1.10–1.32, essentially still at its over-wide start, while the
acceptance rate (0.564 at ρ=0.02) looks perfectly healthy. So pCN converts a hard failure into a
slow one, which is preferable but still needs its own convergence check — the Stein diagnostic R
serves, since acceptance does not.

---

## 14. Negative results

Recorded because each rules out a plausible line of attack.

**Importance sampling cannot certify the answer.** Self-normalized IS against the exact target
using `N(mean_SVGD, H⁻¹)` as proposal — a proposal at energy 0.11 — gives ESS of **62.4, 33.6,
and 1.1 out of 20,000** across three seeds. The resampled output is *worse* than the proposal
(energy 0.54, 0.85, 22.6). In 325 dimensions IS degenerates even from a near-perfect proposal,
so there is no cheap way to turn any of these approximations into a certified one.

**Langevin noise added to SVGD over-disperses, monotonically.**

| noise multiplier | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|
| R | 1.01 | 3.23 | 10.57 | 23.56 | 42.42 | 67.39 |
| θ width % | 87 | 148–166 | 230–272 | 324–372 | 401–461 | 479–538 |

This is the expected behaviour rather than a surprise: SVGD's `∇·k` repulsion term *is* the
deterministic surrogate for the diffusion, so adding the diffusion back double-counts the
entropy. Worth recording because "SVGD underdisperses, so add noise" is an obvious idea and it
is exactly wrong.

**Bandwidth is not the lever.** Multiplying the median-heuristic bandwidth by 0.1, 10, or 100
changed nothing material (energy 35.8, 38.0, 42.6 at noise 1.0; the blowup is driven by the
noise, not the bandwidth). The RBF kernel's problem in high dimension is its tail *shape*, not
its scale — consistent with IMQ and with block size both being levers while bandwidth is not.

**More particles is not the lever either.** 16× more particles (200 → 3200) moves ordinary
x-space SVGD from R = 0.056 to 0.433 and fills the softest directions (0.138 → 0.756), but the
two stiffest bins stay at 0.001–0.014 throughout (§5). Brute force buys the easy directions and
never touches the hard ones.

**Stochastic natural-gradient VI on the mean diverges — but the fix is standard, and turns
this into the cheapest good method in the investigation.** The iteration
`μ ← μ + Σ·E_q[∇log p]` diverged both undamped and at damping 0.25 (`exp07_vi.py`). The cause is
not the step rule: fresh samples each iteration make `E_q[∇log p]` a noisy estimate, and Σ has
condition number 1.9e4, so the iterate random-walks. Fixing the base samples and optimizing the
resulting deterministic objective (sample-average approximation / common random numbers) is
stable and extremely fast — see the "The mean does not need SVGD at all" subsection of §9,
where it reaches bias 0.051 in 2.3 seconds.


---

## 15. Master comparison

### Pooled evidence that R is the right cheap diagnostic

Every scored ensemble produced by this investigation — 155 of them, across all experiments —
pooled to compare the two NUTS-free diagnostics against the ground-truth distributional error:

```
|log R|        vs log energy   Spearman rho = +0.845
theta-width dev vs log energy  Spearman rho = +0.497
```

**R predicts distributional error roughly twice as well as θ interval widths do.** More
usefully, it is a reliable *filter*:

| R band | n | median energy | min | max |
|---|---|---|---|---|
| R ∈ [0.95, 1.05] | 56 | 0.054 | 0.047 | 1.387 |
| R ∈ [0.82, 1.22] (excl. above) | 49 | 1.009 | 0.047 | 1.448 |
| R ∈ [0.37, 2.7] (excl. above) | 15 | 0.991 | 0.868 | 1.342 |
| outside | 35 | 4.866 | 2.085 | 319.8 |

All 50 ensembles achieving energy < 0.15 have **R ∈ [0.946, 1.133]**, so R ≈ 1 is effectively
*necessary*. It is not *sufficient*: the tightest band still contains a run at energy 1.387 —
the whitened-RBF flow, which sits at the MAP with correctly-scaled dispersion but the wrong
mean, so a single second-moment condition cannot see its error.

Practical rule: **R ≈ 1 rules a run in for further checking and R far from 1 rules it out**, but
confirming faithfulness needs a mean check too — which is exactly the `bias` column, and is why
both are reported throughout.

All on the same problem, k=800. Two complementary views (see §1): `energy` weights all 325
directions equally; `varwtd` weights them by the variance they carry, so it sees the leading
directions energy is blind to. `1.000` is correct for `varwtd`.

| method | energy | **varwtd** | R | bias | dev | needs |
|---|---|---|---|---|---|---|
| NUTS k=800 — *sampling floor* | **0.048** | **1.020** | 1.003 | 0.034 | 2.1 | — |
| **whitened ULA, lr 1e-2** | **0.049** | 0.933 | 1.004 | 0.036 | **2.6** | H |
| whitened IMQ SVGD, stop at R ≤ 1.0 | 0.053 | 0.824 | 0.999 | 0.036 | 9.6 | H |
| **whitened IMQ SVGD, stop at R ≤ 1.05** | 0.072 | 0.911 | 1.049 | — | 6.5 | H |
| whitened blockwise RBF, B=1 | 0.078 | — | 1.010 | 0.059 | 11.0 | H |
| Laplace + SAV-VI mean (no SVGD, ~5 s) | 0.083 | — | 1.084 | 0.061 | 5.3 | H only |
| whitened reweighted + Stein inflation + SVGD mean | 0.096 | — | 1.007 | 0.068 | 18.8 | H + 2 runs |
| **gauss-hybrid: N(mean_SVGD, H⁻¹)** | 0.112 | **0.993** | 1.114 | 0.077 | 6.4 | H + 1 run |
| affine-recalibrated reweighted | 0.459 | — | 0.950 | 0.176 | 5.4 | H + 1 run |
| whitened blockwise RBF, B=5 | 0.578 | — | 0.985 | 0.206 | 14.1 | H |
| whitened RBF SVGD (= blockwise B=325) | 1.332 | — | 0.990 | 0.318 | 14.1 | H |
| Laplace N(x_MAP, H⁻¹) | 1.446 | — | 1.153 | 0.335 | 6.6 | H only |
| **x-space reweighted — *current production*** | 3.665 | 1.409 | 0.244 | 0.176 | 5.0 | — |
| x-space standard SVGD | 6.742 | — | 0.022 | 0.068 | 27.4 | — |

**The two views disagree, and the disagreement is the point.** By energy, whitened IMQ at
R ≤ 1.0 (0.053) beats gauss-hybrid (0.112) two-to-one. By `varwtd` the order reverses:
gauss-hybrid is near-perfect in the dominant directions (0.993) while that IMQ setting is 18%
short (0.824). Per-eigen-block ratios make it concrete (3 seeds):

| method | varwtd | top-1 | top-5 | top-20 | rest | sd ratio |
|---|---|---|---|---|---|---|
| NUTS k=800 (floor) | 1.020 | 0.997 | 1.015 | 1.007 | 0.998 | 1.007 |
| **whitened ULA** | 0.933 | 0.885 | 0.916 | 0.963 | 1.005 | 0.985 |
| whitened IMQ @R ≤ 1.05 | 0.911 | 0.836 | 0.904 | 0.897 | 0.984 | 0.955 |
| whitened IMQ @R ≤ 1.0 | 0.824 | 0.735 | 0.809 | 0.845 | 0.965 | 0.933 |
| **gauss-hybrid** | **0.993** | 0.944 | 1.011 | 0.939 | 0.996 | 0.991 |
| x-space reweighted (production) | 1.409 | 1.434 | 1.555 | 1.282 | **0.119** | **1.037** |
| x-space standard SVGD | 0.674 | 0.724 | 0.744 | 0.288 | **0.001** | 0.746 |

*(production and standard rows are 3-seed means; sd over seeds ≤ 0.10 on every entry, and
≤ 0.024 on the `rest` column)*

Three things follow.

- **Whitened ULA is the only method strong on every view**: energy 0.049 (floor 0.048), dev 2.6
  (floor 2.1), varwtd 0.933. It is the recommendation.
- **The R ≤ 1.05 stopping threshold is now much better justified** than R ≤ 1.0: it trades energy
  0.053 → 0.072 for varwtd 0.824 → 0.911 and dev 9.6 → 6.5. Stopping at exactly R = 1
  over-contracts the directions that matter most.
- **Production's sd ratio of 1.037 is cancellation, not correctness.** It is 43–56%
  *over*-dispersed in the leading directions and **88% collapsed in the remaining 305**
  (rest = 0.119); the two errors average to a per-coordinate summary that looks nearly perfect.
  No single scalar would have caught this — and this is precisely the configuration
  `investigation.md` selected on θ-width calibration. Standard SVGD fails the same way but
  one-sidedly: `rest = 0.001`, essentially *zero* variance in 305 of 325 directions.

Note also how badly θ-width `dev` orders the table: it ranks production (5.0) above whitened IMQ
(9.6) while the distributional error differs by 50–70×. `investigation.md` flagged widths as a
weak criterion; here they are actively misleading.

## 16. Catalogue of ideas

Everything considered, whether or not it worked. Sources: literature, adaptations of it, and
constructions suggested by the diagnosis itself.

### Tested

| idea | origin | outcome |
|---|---|---|
| Full dense Laplace preconditioner (whitening) | adaptation of matrix-SVGD to a global exact metric | **works**; prerequisite for everything else |
| IMQ kernel | Gorham & Mackey 2017 | **works**, but only in whitened coordinates; diverges in x-space |
| Preconditioned ULA | classical | **works best overall**; matches the floor, needs no kernel |
| Blockwise / message-passing kernel | Zhuo et al. 2018 | **works** (B=1: 0.078 vs 1.332); monotone in block size; needs whitening |
| Sample-average-approximation VI for the mean | classical (common random numbers) | **works**; bias 0.051 in 2.3 s, no SVGD needed |
| Stein-identity stopping rule | novel; from `investigation.md`'s diagnostic | **works**; R ≤ 1.05 rather than R ≤ 1.0 |
| Laplace covariance + SVGD mean ("gauss-hybrid") | novel; from the §2 decomposition | **works**; best in the leading directions (varwtd 0.993) |
| Stein-calibrated scalar inflation | novel | **works when the profile is flat**; harmful otherwise |
| Affine recalibration `T A Tᵀ = H⁻¹` | novel | works on reweighted, fails on standard (zero-variance directions) |
| Variance-weighted spectral scoring | novel; forced by the metric blindness in §1 | **methodological fix**; reorders the ranking |
| Stochastic natural-gradient VI on the mean | classical | **fails** undamped and damped; MC noise × cond 1.9e4 random-walks |
| Langevin noise added to SVGD | Ye et al. 2020 (self-repulsive dynamics) | **fails**, monotonically — double-counts entropy |
| Self-normalized importance sampling | classical | **fails**; ESS 1–62 out of 20 000 |
| Bandwidth multiplier sweep (0.1×–100×) | standard tuning | no material effect — tail shape, not scale |
| Particle-count scaling | Ba et al. 2022 | partial: fills soft directions only, stiff bins stay ~0.004 |
| Diagonal / block-time / GP-prior preconditioners | scalability | **fail**; ill-conditioning is in the correlations (§7) |
| Shrinking or displacing H | scalability / robustness | **fails**; 1% diagonal admixture costs 11× (§8) |
| Projected SVGD (pSVGD) | Chen & Ghattas 2020 | **fails**; needs r=300 of 325, no prior-dominated subspace exists (§13b) |

### Considered and not pursued, with reasons


- **Stein Variational Newton** (Detommaso et al. 2018) — per-particle Hessian preconditioning. A
  single global Hessian already gives profile ≈ 1, and per-particle Hessians cost 800× more.
- **Mixture-of-preconditioners matrix SVGD** — same reason.
- **Normalizing-flow / Gaussianized SVGD** — at the baseline density the posterior is Gaussian to
  sampling resolution, so a flow has nothing to learn. **This is the one entry worth revisiting**:
  §13 shows Gaussianity degrading in the leading directions as data becomes sparse, and §12 shows
  real marginal skew, so a flow may earn its keep on sparser or more nonlinear problems where the
  Gaussian methods lose their advantage.
- **Second-order analytic mean correction** (third-derivative tensor contracted with `H⁻¹`) —
  would fix the MAP→mean displacement analytically, but needs `∇³U`; the SAV-VI iteration reaches
  the same fixed point using only gradients, in 2.3 s.
- **Annealing / tempering the drift-to-repulsion ratio** (D'Angelo & Fortuin 2021) — §3 shows the
  fixed point itself is wrong, and annealing changes the path, not the fixed point.
- **Randomized low-rank approximation of H** — the natural scalability route, but §8 shows the
  metric depends on the *smallest* eigenvalues, which randomized methods approximate worst.

## 17. Recommendations

### Status: implemented

The recommended pipeline is now `MSVGD.whitened_ula()` in `msvgd/msvgd.py`, together with two
private helpers (`_laplace_metric`, `_ula_steps`). Verified bit-for-bit behaviour-preserving for
`solve()` and scored against the gold standard on the real problem:

| | energy | varwtd | R | bias | dev | sd ratio | wall-clock |
|---|---|---|---|---|---|---|---|
| `whitened_ula()` | 0.050 | 0.972 | 0.993 | 0.037 | **2.3** | 1.008 | **3.6 s** |
| NUTS k=800 floor | 0.048 | 1.020 | 1.003 | 0.034 | 3.5 | 1.003 | 125 s |

Two things were added during implementation that the investigation had not established.

**The discretization bias is known in closed form and is removed by default.** Whitening makes
the target's precision the identity, so ULA's stationary inflation is the same in every
direction:

```
var(ULA) / var(target) = 1 / (1 - step_size/2)
```

Measured against that identity to within 0.3% over step sizes from 0.01 to 0.2 on a
40-dimensional anisotropic Gaussian. `bias_correct=True` divides it out — exact in the Gaussian
limit the method already assumes, no tuning constant — which takes R from 1.109 to 0.998 at
`step_size = 0.2` and makes the result largely insensitive to the step size.

**The positive-definiteness check doubles as a MAP-convergence check.** At σ = 0.5 the internal
MAP search returns a point whose Hessian has 4 negative eigenvalues, so the method refuses to run
*before* sampling rather than diverging 2000 steps later. Supplying a tightly-converged fp64 MAP
there gets past that check and then hits the divergence guard, which reports `min eig` and
`trace(H⁻¹)` and names weak identification as the cause. Both failure modes are raised, never
returned silently.

### The one change that matters

**Adopt the Laplace metric.** Compute `H = −∇²log p` at the MAP once (0.8 s at d=325; 3 s
including the MAP solve) and work in `x = x_MAP + H^(−1/2) y`. Every method that reaches the
sampling floor in this investigation does so in those coordinates, and every method that fails,
fails without them — including IMQ, every blockwise variant, and SVGD itself. Nothing else in
this investigation comes close to that leverage.

### Pre-flight check: when the Laplace metric is inappropriate

Before using any of this, inspect the spectrum of `H`. It costs nothing extra — you have already
eigendecomposed it — and it predicts the one regime where the whole approach fails (§13):

| setting | min eig(H) | cond(H) | tr(H⁻¹) | Laplace sd in flattest direction | outcome |
|---|---|---|---|---|---|
| baseline σ=0.2 | 5.307 | 1.9e4 | 0.98 | 0.43 | works, at the floor |
| half the observations | 1.784 | 7.4e4 | 1.68 | 0.75 | works |
| quarter of the observations | 1.591 | 1.0e5 | 1.88 | 0.79 | degraded |
| **noisy σ=0.5** | **0.0078** | **3.1e7** | **133.2** | **11.36** | **fails** |

When the smallest eigenvalue collapses, the posterior has a near-flat direction, the quadratic
approximation predicts enormous spread there, and the true posterior is in fact bounded by the
ODE nonlinearity rather than by curvature. Whitening then stretches that direction by an order
of magnitude and pushes the initial ensemble into the region where the cubic term explodes.

**Rule of thumb: if `tr(H⁻¹)` is orders of magnitude larger than at a well-identified reference
fit, or `min eig(H)` approaches zero, do not trust the Laplace metric.** This is a weak-
identification diagnostic, not a numerical one, and it requires no reference chain.

### Which sampler, given the metric

Ordered by what the user is optimizing for:

1. **Faithfulness, simplest — and the overall recommendation** — *whitened ULA*, lr 1e-2, 800
   chains, 2000 steps. Energy 0.049 (floor 0.048), dev 2.6 (floor 2.1), varwtd 0.933. It is the
   only method that is strong on all three views simultaneously. No kernel, no bandwidth, no
   mitosis, no stopping rule.
2. **Keeping a deterministic particle method** — *whitened IMQ SVGD* stopped at **R ≤ 1.05, not
   R ≤ 1.0**. Energy 0.072, varwtd 0.911, dev 6.5. Best θ marginal Wasserstein of any method
   tested. Stopping at exactly R = 1 looks better on energy (0.053) but over-contracts the
   dominant directions (varwtd 0.824).
3. **If the leading directions matter most** (trajectory uncertainty bands, θ intervals) —
   *gauss-hybrid*, `N(mean_SVGD, H⁻¹)`. Its varwtd of 0.993 is the best of any method, though its
   energy (0.112) is the worst of the good ones because it is Gaussian and so misses marginal
   shape.
4. **Cheapest acceptable** — *Laplace + SAV-VI mean*, ~5 s end to end, energy 0.083. No SVGD.
5. **Smallest change to the existing code** — keep the current solver, take its ensemble
   **mean**, and pair it with `H⁻¹`. A 33× energy improvement requiring no change to
   `msvgd.py` at all.

### Cost

Measured on the RTX 3090 (`exp19_timing.py`), k = 800, d = 325:

| stage | seconds |
|---|---|
| MAP solve (fp64) | 2.1 |
| dense Hessian 325×325 (fp64) | 0.8 |
| eigendecomposition | 0.0 |
| **setup subtotal** | **3.0** |
| whitened IMQ, per iteration | 0.0004 |
| whitened IMQ, ~600 it (R ≤ 1.05) | 0.2 |
| whitened ULA, 2000 steps | 0.5 |
| **total, setup + ULA** | **3.4** |
| **total, setup + IMQ** | **3.2** |
| reference: NUTS (`investigation.md` B=30 study) | 125.0 |

The setup dominates, and the sampling is nearly free. The whole pipeline is **~37× faster than
the NUTS reference** while being statistically indistinguishable from it, and it is *faster than
the current production configuration* (~4 s), not slower.

### Robustness caveat that should drive the choice

Options 1 and 2 use `H` **only as a preconditioner** — they target the exact posterior, so on a
genuinely non-Gaussian problem they degrade gracefully. Options 3–5 *return* a Gaussian, so they
are only as good as the Gaussianity assumption: §2 verified it at the baseline density, but §12
shows it already fails at the level of individual marginals (θ_b skew −0.219, 23 SE) and §13
shows it degrading in the leading directions as data becomes sparse.

This cuts against the varwtd ranking, which favours the Gaussian options — their advantage there
is real but is *conditional on the posterior actually being Gaussian*, and it is exactly the
assumption that fails first as you move away from this dataset. **Prefer 1 or 2 unless cost
dominates or you have separately verified Gaussianity for your problem** (which the §2
decomposition makes cheap to do: compare `N(μ, H⁻¹)` against a short reference chain on both
energy *and* sd ratio).

### Independent of all the above

- **Fix `_mitotic_split`.** The absolute `1e-6·I` ridge on a rank-deficient covariance is a
  latent bug: it works only at the particle scale it was tuned for and produces NaN under any
  reparameterization. The SVD-span sampling in `harness.patch_split` is exact, scale-invariant,
  and drops the isotropic noise the ridge was injecting.
- **Stop using θ interval widths as the primary criterion.** They ranked methods backwards three
  separate times here (§"Master comparison", §"The stopping threshold is a real knob",
  §"Fixing SVGD itself"). Score against a reference with energy distance, or at minimum report
  the spectral profile alongside.
- **Reconsider `reweighted_kernel=True` as the default.** Its entire benefit was θ-width
  calibration achieved through a tuned clip constant. Under every pipeline here the *standard*
  kernel is preferable: it gives a better mean (bias 0.068 vs 0.176), which is the only thing
  the hybrid needs, and the reweighting's variance correction is superseded by the Laplace
  metric.

### Limitations

- **Dense Hessian.** O(d) gradient evaluations, O(d²) memory, O(d³) factorization. Comfortable to
  d ~ 10³–10⁴. §7 shows the cheap alternatives (diagonal, block-time, GP-prior) all fail because
  the ill-conditioning lives in the correlations, and §8 shows the metric has almost no error
  tolerance and depends on the *smallest* eigenvalues — which is the hard end for randomized
  low-rank methods. **Scaling this beyond a dense Hessian is the main open problem** and I do not
  have a promising route to it.
- **The Hessian must be positive definite at the MAP,** and the MAP genuinely converged. It was
  here (min eigenvalue 5.31), but evaluating H just 1 sd away already produced an indefinite
  matrix (§8).
- **Weak identification breaks the whole approach, not just one sampler.** At σ = 0.5 (min
  eig(H) = 0.0078) the Laplace covariance is 14× too wide and both IMQ and ULA diverge at every
  step size (§13). Run the pre-flight check; where it fails, use NUTS. Nothing in this
  investigation improves on NUTS in that regime.
- **The Gaussian-returning options are conditional on Gaussianity**, which §13 shows degrading as
  data becomes sparse and collapsing entirely at σ = 0.5, and §12 shows already failing for
  individual marginals. Verify it for your problem before relying on options 3–5.
- **One ODE system, one dimension.** Everything is FitzHugh–Nagumo at d = 325, varied over noise
  and observation density (§13) but not over model. A stiffer or higher-dimensional system could
  behave differently, particularly for the dense-Hessian requirement.
- **k = 800 limits what any of this can resolve.** Marginal skew is not estimable at that size
  (se 0.087 against a true θ_b value of −0.219), so the marginal-shape comparison in §12 rests on
  Wasserstein distance rather than on moments.

## Appendix: scripts

All in `investigation2/`. Run from that directory; `harness.py` provides the problem, the
gold-standard reference, the metrics and the measured floors, and is imported by everything else.
`laplace_cache.npz` caches the MAP and Hessian eigendecomposition so only the first run pays for
them.

| script | what it establishes |
|---|---|
| `harness.py` | problem construction, metrics, floors, `patch_split()` |
| `exp01_whiten.py` | MAP + dense Hessian; the Laplace approximation as a control |
| `exp02_decompose.py` | the entire Laplace error is the mean |
| `exp03_hybrid.py` | Gaussianity at higher resolution; gauss-hybrid; affine recalibration |
| `exp04_from_laplace.py` | SVGD destroys a correct ensemble |
| `exp05_pipeline.py` | Stein inflation; the IS failure |
| `exp06_noise.py` | Langevin-noise sweep; ULA; first IMQ signal |
| `exp07_vi.py` | stochastic natural-gradient VI diverges |
| `exp08_imq.py` | energy-metric calibration; kernel × coordinates ablation |
| `exp09_verify.py` | seed/init/k robustness; R as a stopping rule |
| `exp10_generalize.py` | independent NUTS references at 4 data/noise settings |
| `exp11_precond.py` | full vs diagonal vs block vs GP-prior metric |
| `exp12_blocked.py` | blockwise kernels; particle-count scaling |
| `exp13_gram.py` | the Gram-matrix degeneracy that explains it all |
| `exp14_threshold.py` | the stopping threshold as a knob |
| `exp15_shape.py` | θ marginal Wasserstein and skew |
| `exp16_ula.py` | ULA across seeds and step sizes |
| `exp17_sav.py` | sample-average-approximation VI for the mean |
| `exp18_hrobust.py` | how exact the metric must be |
| `exp19_timing.py` | wall-clock cost of each stage |
| `exp20_varweighted.py` | variance-weighted re-scoring; reorders the ranking |
| `exp21_nongaussian.py` | non-Gaussianity: bias vs mixing vs transience |
| `exp22_psvgd_spectrum.py` | pSVGD premise: generalized eigenvalue decay |
| `exp23_psvgd.py` | pSVGD implementation and ceiling calculation |
| `exp24_gaussianity.py` | does the method assume Gaussianity? skew-normal + MAGI skew |
| `exp25_faster.py` | HMC and pCN against MALA, speed and safety |

Environment notes: SVGD/ULA runs use fp32 on the GPU (`CUDA_VISIBLE_DEVICES=0`, the RTX 3090);
MAP, Hessian and NUTS use fp64, with NUTS on CPU (`JAX_PLATFORMS=cpu`), matching the protocol in
`investigation.md`.
