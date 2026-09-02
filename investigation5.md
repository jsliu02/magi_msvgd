# Investigation 5 — correcting the posterior without assuming it is nearly Gaussian

Investigation 4 produced a pipeline that corrects the Laplace approximation's mean and reports a
reference-free certificate. It was developed entirely on FitzHugh–Nagumo. Run unchanged on the
four systems in `tests.py`, **it declines to apply on three of them**:

| system | dim | p | ‖∇log p‖ at MAP | min eig(H) | cond(H) | tr(Σ) | q_max | kappa_S | gate |
|---|---|---|---|---|---|---|---|---|---|
| fn | 325 | 3 | 5.3e-7 | 4.14 | 7.2e3 | 1.14 | 11.7 | 28.6 | **apply** |
| hes1 | 106 | 7 | 4.9e-8 | 1.7e-2 | 1.8e8 | 171.6 | 23.1 | 192 | WARN |
| hiv | 608 | 5 | **1.7e-2** | **5.4e-12** | **4.3e16** | 1.9e11 | 5.1e8 | 9.4e170 | WARN |
| lorenz | 306 | 3 | 2.0e-8 | 1.5e-3 | 8.8e4 | 5307 | 7.2 | 1.3e153 | WARN |

That is the overfitting, stated plainly, and it set the agenda: stop patching a Gaussian and find
something whose accuracy does not depend on the posterior being nearly Gaussian.

**Headline.** Profiling the states out and solving the small parameter problem directly reaches
the reference chain's *own* half-vs-half floor on FitzHugh–Nagumo — mean error 0.0087 against a
floor of 0.0281 and standard-deviation error 0.43% against 0.86% — where the shipped third-order
correction gets 0.1360 and 9.03%. It is 16× better on the mean, 21× better on the covariance, and
it is the first method in these investigations that corrects the covariance at all.

---

## 1. The test systems, and what they cost to run

`tests.py` spans regimes FitzHugh–Nagumo does not: **Hes1** has 7 parameters, a rational
nonlinearity, an `exp(X)` state parameterisation and a component that is *never observed*; **HIV**
has time-dependent forcing and states spanning 30 to 10^5; **Lorenz** is chaotic. Getting them to
run surfaced four issues, none of them in MAGI:

* `hiv_ode` returns an inhomogeneous list when `t` arrives with shape `(1,)`, which is how MAGI's
  `vmap` used to supply it: `eta` picks up the trailing axis so the first two components are
  `(1,)` while the third stays scalar. **Fixed in MAGI rather than worked around** — `magi.py`
  now hands the user's field a scalar `t`, which is what its own docstring promises ("ode should
  be written for a single observation at a single time point") and what `tests.py`'s own
  `ground_truth` already passes. Results on all four systems are bit-identical; only genuinely
  time-dependent fields, which were silently broken, change behaviour.
* HIV and Lorenz store the initial condition under `X0`, but `DynamicalSystem.ground_truth` reads
  `x0`.
* For HIV and Lorenz the observation times are **not** a float-exact subset of the discretisation
  set — mathematically they are (0.2 against 0.1; 0.1 against 0.025) but not in binary — so
  `discretize` raises. A nearest-grid-point match with a tolerance fixes it.
* `sample()` compares observation times against the integration grid with exact equality, which
  for HIV and Lorenz holds only at `step = 1e-4`.

`ground_truth` also overwrites its own `step` argument with a decimal count when `step is None`, so
it must always be passed explicitly.

### Two of the four systems are not identified

This is worth stating before any method is judged on them. Using `m.tau` as the observation mask
(note `m.x_init` is overwritten by the initialiser and is *not* the data):

| system | residual/σ at observations | trajectory error vs truth | log p(MAP) | log p(truth) |
|---|---|---|---|---|
| fn | 0.94 / 1.14 | 6% / 7% | −14.2 | −2421.0 |
| lorenz | 0.84 / 0.82 / 0.49 | 20% / 25% / 16% | −207.8 | −234.8 |
| hes1 | 0.35 / 0.57 / — | 11% / 15% / 100% | **+17.8** | −4.0 |
| hiv | **0.007 / 0.085 / 0.002** | 70% / 6% / 0.3% | **−843.9** | −1034.6 |

FitzHugh–Nagumo is the well-behaved case: residuals at the noise level and a 6% trajectory error.
HIV fits its observations to **0.007 σ**, 140 times tighter than the noise, and places θ at
(−11.9, −0.24, 0.26, 1789, 2.76) against a truth of (36, 0.108, 0.5, 1000, 3). Hes1 collapses five
of its seven parameters to ~0. In both cases the posterior genuinely prefers those points to the
truth, so this is a property of the model as configured, not a solver failure — but it means no
posterior approximation can be expected to look tidy there, and results on those two systems
measure the difficulty of a degenerate target rather than the quality of a method.

---

## 2. Conditioning, integration, and a rewritten `tests.py`

### The integration step was far too coarse

`tests.py` integrated with forward Euler and retained every substep in a `lax.scan`. Both were
limiting. Measuring the error at the observation times against an accurate reference, in units of
the observation noise:

| system | Euler @1e-6 | **RK4 @1e-3** | Euler f-evals @1e-6 | RK4 f-evals @1e-3 |
|---|---|---|---|---|
| fn | 6.8e-05 | **6.8e-11** | 20,000,000 | 80,000 |
| hes1 | 2.2e-06 | **7.5e-12** | 240,000,000 | 960,000 |
| hiv | 5.0e-03 | **5.2e-09** | 20,000,200 | 80,800 |
| lorenz | 1.7e-04 | **1.1e-08** | 2,500,100 | 10,400 |

At the step originally in use, HIV's data carried **2.6 σ of integration error** — larger than its
observation noise — so the posterior being studied was not the intended one. `tests.py` has been
rewritten: RK4 rather than Euler, and the solution retained only on the output grid rather than at
every substep, which takes the stored array from up to 5.8 GB (Hes1 at 1e-6) to a few kilobytes
independent of the step. Hes1's integration goes from 10.0 s to 0.14 s. The rewrite also fixes the
`x0`/`X0` inconsistency, replaces exact-float grid lookups with tolerance-based ones, and returns
`(D,)` from every field. Its correctness issues are listed in §1; none were in MAGI.

**Every number measured before this fix is superseded.** The rest of this section, and §3, are
re-measured; the references were rebuilt.

### Conditioning

With correctly integrated data the diagnosis changes. Hes1 is *not* ill-conditioned — its earlier
cond(H) = 1.8e8 was largely an artefact of the bad integration, and it is 2.2e3 once fixed. Only
HIV is genuinely hard, and there the Hessian at the stationary point is **indefinite** (smallest
eigenvalue −1.5e-10): the mode is a saddle, so no Gaussian approximation exists there at all.

| system | cond(A) | cond(DAD) | stock ‖∇log p‖ | scaled ‖∇log p‖ | stock s | scaled s |
|---|---|---|---|---|---|---|
| fn | 4.4e3 | 3.9e3 | 2.2e-07 | 2.8e-07 | 2.00 | **1.31** |
| hes1 | 2.2e3 | 1.0e2 | **3.8e-12** | 3.9e-09 | 1.82 | **1.21** |
| hiv | **4.1e17** | **8.4e10** | **1.7e-02** | **6.3e-09** | 3.40 | **2.38** |
| lorenz | 9.6e4 | 7.5e3 | 4.1e-09 | 4.3e-08 | 1.77 | **1.26** |

The normal equations square the condition number, so HIV's Jacobian — conditioned at 2e8 purely
because θ = (36, 0.108, 0.5, 1000, 3) are rate constants in mixed units — leaves A beyond float64
and its Cholesky meaningless. Symmetric diagonal scaling of the solve,

    D = diag(A)^(-1/2),   (D A D + λI) s = −D g,   δ = D s,

is identical in exact arithmetic and recovers seven orders there. It is **not** uniformly better:
on the three well-scaled systems the final gradient is slightly worse, by three orders on Hes1,
though every value involved is far below any tolerance a user would set. It is 1.3–1.5× faster
everywhere. Unconditional scaling is therefore the right default, but the honest summary is
"essential where it matters, harmless elsewhere", not "strictly better".

Three details cost more time than the change itself.

*The previous damping was not Marquardt's.* It was a uniform ridge `λ·trace(A)/dim·I`, not
`λ·diag(A)`. On a unit-diagonal matrix `λI` **is** `λ·diag(A)`, so scaling makes the damping
per-coordinate as a side effect — the point, since a uniform ridge damps stiff and soft directions
by wildly inappropriate amounts when the scaling is bad.

*The bounds on λ had to move with it.* λ is now relative, so the old absolute floor of 1e-12 sits
at the smallest eigenvalue of the scaled matrix and keeps perturbing the solve at the level of the
answer. The floor is now machine epsilon.

*The initial λ mattered more than the floor.* At `lm_init = 1e-8` HIV stalls at 1.5e-3; at 1e-10 or
below it reaches 1e-7 or better, and the other three are insensitive from 1e-8 to 1e-14. A rejected
step raises λ tenfold, so starting small is nearly free. The default is now 1e-10.

A diagonal floor set *relative to the largest entry* is tempting and wrong: HIV's legitimate
curvature range spans many orders, so any relative floor large enough to matter clips real
curvature. Only exactly-degenerate coordinates are excluded.

---

## 3. The profiled posterior

### The idea

Every metric in investigation 4 said the error that matters is in θ, and p is small: 3, 5, 7
across these systems. The 300–600 state dimensions are what make the problem hard. So do not
approximate in the hard directions at all — integrate them out:

    p(θ) ∝ ∫ exp(−U(θ, X)) dX ≈ exp(−U(θ, X*(θ))) · det H_XX(θ, X*(θ))^(−1/2),
    X*(θ) = argmin_X U(θ, X).

This is the Laplace approximation applied only to the **inner** integral, and it differs from the
joint Laplace in a way that matters. It is *exact* whenever f is affine in the state at fixed θ
(Condition A of investigation 4), where the joint Laplace still carries an O(Λ) mean error,
because the joint version linearises the θ–X coupling and this one does not. What remains is the
non-Gaussianity of p(X|θ) alone, which the GP prior and the data constrain far more tightly than
they constrain θ. And it makes no Gaussian assumption whatever about θ: skew, heavy tails and
curvature in the parameter marginals all survive.

The p-dimensional integral over θ is then done by self-normalised importance sampling from the
Laplace θ-marginal on a scrambled Sobol point set, so the result is deterministic given a seed and
comes with ESS and Pareto k̂ as reference-free diagnostics.

The output is not a Gaussian but a **mixture**, one Gaussian in X per θ node,

    p(x) ≈ Σ_i w_i δ(θ − θ_i) N(X; X*(θ_i), H_XX(θ_i)^(−1)),

from which every moment follows in closed form, including Cov(θ, X), which no Gaussian centred at
the mode gets right.

### It works

On FitzHugh–Nagumo, against a reference with R̂ = 1.007 over 96,000 draws:

| estimate | max \|θ error\| (ref sd) | max \|sd error\| | cost |
|---|---|---|---|
| MAP | 1.4721 | 9.03% | — |
| third-order (investigation 4) | 0.1360 | 9.03% | 5.4 s |
| **profiled** | **0.0087** | **0.43%** | **14 s** |
| reference half-vs-half floor | 0.0281 | 0.86% | |

Both errors are **below the floor**: at 96,000 draws the profiled posterior is not distinguishable
from a second independent reference run. ESS 435/512 (85%), k̂ = −0.87.

### Three things it needed

**Adapting the proposal is not optional.** The first pass had ESS of 3.8%–8.7%, and the reason is
diagnostic rather than technical: the proposal is centred at the mode, the mode is off-centre by
about 1.9 posterior standard deviations, and a shift of that size costs `exp(−1.9²/2) ≈ 0.17` in
ESS by itself. The low ESS *is* the bias being corrected. Three rounds of population Monte Carlo
on the weighted moments fix it:

| system | ESS round 0 | ESS final | k̂ | failed nodes |
|---|---|---|---|---|
| fn | 8.3% | **85%** | −0.87 | 0 |
| lorenz | 3.8% | **84%** | −0.39 | 58 → 0 |
| hes1 | 8.7% | **30%** | 0.13 | 0 → 6 |
| hiv | 0.8% | 0.8% | — | 0 |

**The inner solve must actually converge.** Accuracy is governed by it, not by the node count:

| inner iterations | max \|θ err\| | max \|sd err\| | cost |
|---|---|---|---|
| 3 | 0.0809 | 2.13% | 9.8 s |
| 4 | 0.0201 | 0.70% | 10.0 s |
| 6 | **0.0086** | **0.48%** | 12.2 s |

An unconverged X*(θ) biases `U_prof` and `log det H_XX` together. The residual inner gradient is
therefore reported as a diagnostic. A node ladder (early rounds only have to relocate the
proposal, so they do not need the full budget) cut the cost from 27 s to 14 s for free.

**A correctness check that needs no reference.** At θ = θ_MAP the inner profile must reproduce the
MAP trajectory exactly. Measured: 2e-9 (fn), 2e-11 (lorenz), 3e-17 (hiv).

### Where it does not work

On Hes1 — 7 parameters, `exp(X)` dynamics, one state never observed, five parameters collapsed —
every method is far from the floor:

| hes1 | max \|θ err\| | max \|sd err\| |
|---|---|---|
| MAP | 2.4996 | 835% |
| third-order | 1.0242 | 835% |
| profiled | **0.8930** | **386%** |
| floor | 0.0136 | 4.52% |

The profiled version halves the covariance error and improves the mean, but at 66× the floor that
is not a solution. Hes1 violates the premise directly: `p(X|θ)` is what the inner Laplace
approximates, and an `exp(X)` vector field with an unobserved component makes it strongly
non-Gaussian. The reference itself reports 6,129 divergences (6.4%), so how much of the remaining gap is
method and how much is reference is not settled.

On HIV the ESS stays at 1 through every round of adaptation. That is the degenerate posterior of
§1 rather than a failure of the estimator, and ESS = 1 reports it unambiguously.

---

## 4. Negative results

**Correcting the inner Laplace approximation by sampling makes things worse.** The exact inner
integral is `−U* − ½ log det H_XX + log E_ξ[e^{−Δ}]` with ξ ~ N(0, H_XX^(−1)); the profile drops
the last term. Estimating it with antithetic pairs — which cancel Δ's cubic part exactly, leaving
the quartic — and common random numbers across θ nodes is cheap and principled, and it fails:

| inner pairs | fn max \|θ err\| | hes1 ESS |
|---|---|---|
| 0 | **0.0086** | 10% |
| 4 | 0.3665 | 1% |
| 16 | 0.3311 | 1% |

Δ is a sum of quartic terms over 322–603 coordinates, so a handful of pairs cannot resolve its
mean, and injecting that noise into `log p̂(θ)` corrupts the weights multiplicatively. Common
random numbers do not rescue it because X*(θ) moves between nodes. The fact that fn reaches the
reference floor with the correction *switched off* is the useful part: it proves the inner Laplace
error is below the noise floor where the method works at all.

---

## 5. Status and what remains

Established: the conditioning fix, the profiled construction and its diagnostics, validation
against references on fn and hes1, and the characterisation of all four regimes. Outstanding: the
HIV and Lorenz references were still building at the time of writing, so the profiled posterior is
validated on two systems, one of which (hes1) is degenerate. Lorenz is the important remaining
test — chaotic, well-posed, ESS 84% — and will say whether the fn result generalises to a system
whose difficulty is dynamical rather than statistical.
