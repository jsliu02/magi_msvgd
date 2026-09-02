# Investigation 6 — what the fast MAP makes possible

> **CORRECTION (added after §1–§7 were written).** Sections 2, 4 and 5 below rest on the claim that
> two of the four benchmark systems are not identified and that HIV's posterior is improper. That
> was wrong. It was an artefact of the GP hyperparameter fit, which was starting its optimiser at
> `phi1 = phi2 = sigma = 1` regardless of the data's units and settling on lengthscales far below
> the grid spacing. With the fit put on a scale-free footing (§8), **all four systems are
> identified and every one recovers its true parameters to within 1.6 posterior standard
> deviations.** HIV's condition number falls from 4.1e17 to 4.0e2 and its null direction
> disappears. The corrected results are in §9, and §10 explains why Hes1 became harder
> to sample as a result; the earlier sections are left as written, since the
> reasoning in them is sound and only the premise was false.

Investigations 4 and 5 built a fast, exact Gauss–Newton mode solver and then used it for one
thing: a starting point for a Gaussian. This investigation asks what else a mode that costs one
second buys, and the answer turns out to be less about better approximations and more about
knowing whether the question being asked is well posed at all.

**Headline.** Two of the four test systems are not identified, and on one of them the mode is
already at the reference posterior mean while every correction makes it worse. A ~2-second
analysis built from the MAP and its exact Hessian says which parameters the data determines,
before any sampling. Where the problem *is* identified, profiling the states out and integrating
the parameters directly stays below the reference chain's own floor.

---

## 1. Is the returned point a mode at all?

Everything downstream — the Laplace covariance, the third-order correction, the whitening metric,
the profiled inner solve — assumes a strict local maximum. Three checks, all affordable only
because the MAP solve is fast:

* **curvature** — the exact Hessian's spectrum
* **escape** — if a genuinely negative direction exists, step along it and re-solve
* **globality** — re-solve from 8 dispersed starts, and separately by continuation in the prior
  tempering `beta_inv` (small `beta_inv` leaves an objective dominated by the observation term,
  with a benign landscape; following the solution up to the target keeps the iterate in that basin)

| system | log p | ‖∇log p‖ | negative dirs | escape | distinct optima from 8 starts | homotopy |
|---|---|---|---|---|---|---|
| fn | −6.9869 | 2.8e-07 | 0 | — | 1 | −6.9869 |
| hes1 | 26.5563 | 3.9e-09 | 0 | — | 1 | 26.5563 |
| hiv | −843.8808 | 6.3e-09 | 0 | — | 1 | −843.8808 |
| lorenz | −207.7921 | 4.3e-08 | 0 | — | 1 | −207.7921 |

The solver never lands on a saddle, multistart never finds a competing optimum, and the homotopy
agrees to four decimals everywhere. That is a stronger statement about the MAP than anything in
investigations 4 or 5, and it is worth having before building on it.

---

## 2. Which parameters are identified?

Investigation 5 reported HIV's Hessian as indefinite with 205 flat directions. That was measured
on the raw spectrum, which is the scale-naive error investigation 5 was itself about: an
eigenvalue of `H` has units, and HIV's coordinates span 30 to 10⁵. Two scale-free readings
instead — the spectrum of `D H D` with `D = diag(H)^(-1/2)`, and the marginal posterior standard
deviation of each parameter as a fraction of its own value.

| system | cond(DHD) | null dirs | identified | weak | not identified |
|---|---|---|---|---|---|
| fn | 4.7e3 | 0 | a, b, c | — | — |
| hes1 | 1.0e2 | 0 | — | — | **all 7** |
| hiv | 8.4e10 | 1 (100% on θ) | delta, N, c | — | lam, rho |
| lorenz | 7.9e3 | 0 | rho, sigma | beta | — |

HIV is *partially* identified: delta, N and c to 1.7–2.8% of their values, while lam and rho carry
posterior standard deviations of 4.3e5 and 2.5e3. Its single exact null direction lies entirely in
θ. Hes1 is identified in nothing — its mode collapses every parameter to ~1e-6 against true values
of 0.022 to 20, with posterior standard deviations of 0.3 to 3.5.

This costs about two seconds and needs no sampling, and it changes what may be asked. Reporting
"maximum θ error = 2.5 posterior sd" on a system where no parameter is identified measures the
reference chain's wandering, not the method.

### Proper, diffuse, or improper — three different things

A null direction of the quadratic model is not the same as an improper posterior: the density can
still decay at higher order. The distinction decides what a method should output, and whether a
reference chain can exist at all. Walking each parameter axis outward with the states re-profiled
at every step, so the walk follows the ridge rather than cutting across it, and recording the fall
in the profiled log-density:

| system | parameter | fall in log p̂ by 10⁴× its scale | verdict |
|---|---|---|---|
| hiv | **lam** | **0.07 nats** | **improper** |
| hiv | rho | 36 | diffuse |
| hiv | delta, N, c | 3.5e4 – 1.8e8 | identified |
| hes1 | all 7 | 157 – 1.3e9 | diffuse but **proper** |
| lorenz | beta | 1106 | weak |
| fn | a, b, c | 8e4 – ∞ | identified |

**HIV's posterior is improper in lam** — flat to within 0.05 nats while lam moves from −11.9 to
99,988. No posterior mean or variance for it exists, and no MCMC reference can converge along it.
That is why the HIV reference chain ran for over an hour without finishing: it was not slow, it was
futile, and the diagnosis above costs twelve seconds. Hes1, by contrast, is diffuse but proper
everywhere — a guess from its spectrum alone would have got that wrong.

The whole diagnosis — mode validity, globality, identifiability, properness — runs in 5 to 12
seconds per system and needs no sampler. It is the part of this investigation most worth keeping.

---

## 3. The profiled posterior, re-measured and rebuilt

### On the identified system it is at the floor

FitzHugh–Nagumo, correctly integrated data, reference R̂ = 1.0027 over 96,000 draws:

| estimate | max \|θ err\| (ref sd) | max \|sd err\| | sec |
|---|---|---|---|
| MAP | 1.0448 | 3.07% | — |
| third order (investigation 4) | 0.0629 | 3.07% | 2.5 |
| **profiled** | **0.0047** | **0.34%** | 12.4 |
| reference half-vs-half floor | 0.0120 | 0.93% | |

Below the floor on both mean and spread, and 13× better than the third-order correction. The
investigation-5 result survives the integration fix.

### A better proposal, and a structural fact that limits it

The first version drew θ from the joint Laplace marginal and adapted. On Hes1 that proposal is so
mismatched that the log importance weights span 8,400 nats and ESS collapses to 0.6%, recovering
to 16% only after ten rounds. Since p is 3–7 while the state space is 300–600, the profiled
marginal's own mode and curvature are directly computable: a Newton solve in p dimensions using
central differences through the profile, costing O(p²) profile solves per step.

This is worth doing for the mode. **The profile mode is not the joint MAP's θ** — on
FitzHugh–Nagumo it sits (−0.24, +1.01, −1.09) Laplace standard deviations away, which are
precisely the directions and magnitudes by which the joint mode is known to be biased. The joint
mode maximises `U(θ, X)`; the profiled mode maximises `U(θ, X*(θ)) + ½ log det H_XX(θ)`, and that
determinant — the volume of state space consistent with θ — is what the joint mode ignores.

It is **not** worth doing for the curvature, and the reason is structural. The second derivative of
`−U(θ, X*(θ))` at the mode is exactly the Schur complement `Σ_θθ^(-1)`, and the determinant term's
curvature turns out to be negligible: measured on Hes1 at three stencil sizes spanning two orders
of magnitude, the profile curvature equals the joint Laplace curvature to four decimals. So
profiling buys nothing locally; everything it gains is in the shape away from the mode.

Net effect on FitzHugh–Nagumo: same accuracy, 1.6× faster (9.6 s against 16.2 s).

### Faster, via the implicit function theorem

X*(θ) is defined by `∇_X U(θ, X*) = 0`, so `H_XX dX*/dθ + H_Xθ = 0` and the sensitivity is one
solve against a factorisation already in hand. Starting each node from
`X_MAP + (dX*/dθ)(θ − θ_MAP)` rather than from `X_MAP` puts it an order closer, which lets the
inner iteration count fall:

| inner iterations | 2 | 4 | 8 |
|---|---|---|---|
| from X_MAP | 0.0437 / 5.8 s | 0.0065 / 6.8 s | 0.0056 / 8.7 s |
| **from the predictor** | **0.0059 / 4.5 s** | 0.0063 / 6.1 s | 0.0065 / 9.1 s |

(max \|θ error\| in reference sd / wall clock; the floor is 0.0120.)

The full progression on FitzHugh–Nagumo, all at the same below-floor accuracy:
**16.2 s → 9.6 s (profile-mode proposal) → 4.5 s (+ predictor)**.

### On a chaotic system

Lorenz, reference R̂ = 1.0075 with 0.8% divergences over 96,000 draws:

| method | max \|θ err\| (ref sd) | max \|sd err\| | ESS | sec |
|---|---|---|---|---|
| MAP | 4.1388 | 111% | — | — |
| third order | **4.8615** | 111% | — | 2.0 |
| profiled v1 | 0.3408 | 10.4% | 83.6% | 10.8 |
| **profiled v2** | **0.3393** | **10.2%** | 75.5% | **4.7** |
| reference half-vs-half floor | 0.0251 | 0.91% | | |

The Gaussian pipeline fails here: the mode is 4.1 posterior standard deviations out and the
third-order correction makes it *worse*, driving sigma to 0.21 against a reference value of 6.21.
Profiling is 12× better on the mean and takes the spread error from 111% to 10%, with a healthy
ESS and a negative k̂ — but at 13× the floor it does not reach reference quality the way it does on
FitzHugh–Nagumo. The residual is presumably the inner Laplace approximation: chaotic dynamics make
`p(X | θ)` strongly non-Gaussian, which is precisely the assumption profiling rests on, and the
condition-(A) diagnostic of investigation 4 is the natural thing to check it against.

Note also that neither the reference nor any method recovers the true parameters here
(reference beta = 1.36 against a truth of 2.67). With 26 observations of a chaotic system the
posterior is genuinely far from the truth; the question being answered is whether a method
matches the posterior, not whether the posterior matches reality.

### A reference-free gate

Importance sampling at a few percent ESS is not an estimate. The gate is ESS ≥ 10% and
Pareto k̂ < 0.7, both computable without a reference. On Hes1 it fires (ESS 0.4–4.5%), and it is
right to: **the mode's θ error there is 0.0126 against a floor of 0.0181** — the MAP is already at
the reference mean — while both profiled versions make it substantially worse. Where nothing is
identified, the Laplace approximation is the correct answer and the method needs to say so.

---

## 4. A reference that cannot be used

Hes1's reference reports 17,269 divergences out of 96,000 draws — 18%. Divergent NUTS
under-explores, so its posterior standard deviations are likely biased *narrow*, and the apparent
"Laplace is 1.2–5.9× too wide" may be the reference being too narrow instead. Its half-vs-half
agreement of 4.91% does not settle this, since both halves can be stuck in the same way.

No covariance conclusion is drawn from Hes1 in this document. This is worth stating as a general
caution: on a weakly identified posterior, a long NUTS run is not automatically a gold standard,
and its divergence count is the first thing to read.


---

## 5. The missing term: a proper prior on θ

MAGI's posterior, as the paper states it, is

    p(θ, x(I) | ·) ∝ π_Θ(θ) × exp{ −½ Σ_d [ GP term + observation term + ODE term ] },

with "a general prior distribution π(·) on θ". **The implementation omits π_Θ entirely.** The
constructor's `theta_conf` argument reaches only `_initializer.py`, where it pulls the starting
point toward `theta_guess`; it never enters the log-density. The posterior therefore carries a
flat, improper prior on every parameter, which is exactly why §2 finds HIV's λ unbounded.

A Gaussian prior is another sum of squares, so restoring it costs one residual block,

    R ← [ ... ; sqrt(theta_conf) ⊙ (θ − theta_guess) ],

and the Gauss–Newton structure of the mode problem survives intact — the block is linear, so it
contributes a constant to `JᵀJ`, no second-derivative term to the exact Hessian, and it lifts the
θ block away from singularity as a side effect. With `theta_conf = 0` every result is reproduced
bit for bit, so the change is inert unless asked for.

### What it fixes on HIV

| prior sd (as a fraction of the guess) | null dirs | fall in log p̂ along λ | ESS | k̂ | delta | N |
|---|---|---|---|---|---|---|
| flat (the current default) | 1 | **0.07 nats** | 2.4% | 0.43 | 0.259 | 1788 |
| 100% | 0 | 5e7 | 8.1% | 0.11 | 0.481 | 962 |
| 50% | 0 | 2e8 | 16.1% | −0.06 | 0.481 | 963 |
| 20% | 0 | 1.3e9 | **33.0%** | −0.09 | 0.482 | 964 |
| *truth* | | | | | *0.5* | *1000* |

The null direction disappears, the posterior becomes proper, and the effective sample size
recovers fourteenfold — the collapse to ESS = 1 was never a defect of the estimator, it was the
estimator being asked to cover a direction of infinite width.

The more important effect is on the parameters the data *does* determine. Under the flat prior
delta and N sit at 0.259 and 1788 against a truth of 0.5 and 1000, despite §2 calling both
"identified to 1.7%". That was precision without accuracy: an unbounded λ drags the identified
parameters along the ridge, and the Laplace standard deviation, computed from curvature alone,
cannot see it. Restoring π_Θ moves them to 0.481 and 962.

### The control, and what actually drives it

The guess used above sat exactly on the true delta and N, so part of that improvement could have
been the prior injecting the answer. Re-centring it deliberately away from the truth separates the
two effects. λ and ρ are held at (30, 0.1) — close to their true (36, 0.108) — except in the last
row:

| prior centre (delta, N) | prior sd | delta | N | ESS |
|---|---|---|---|---|
| flat | — | 0.259 | 1788 | 2.4% |
| (0.25, 500) — **half** the truth | 20% | **0.470** | **940** | 59.4% |
| (0.5, 1000) — on the truth | 20% | 0.482 | 964 | 32.2% |
| (1.0, 2000) — twice the truth | 20% | 0.371 | 1257 | 13.2% |
| all ×3, so λ, ρ centred at (90, 0.3) | 20% | **0.176** | **2647** | 24.5% |
| *truth* | | *0.5* | *1000* | |

Centred at half the truth, delta and N land at 0.470 and 940 — near the truth and nowhere near the
prior centre. The prior is not supplying the answer.

What it is supplying is a location for the parameters the data cannot determine, and **that** is
what fixes the rest. In every row where λ and ρ are pinned near their true values, delta and N
recover to within 6–26% regardless of what the prior says about delta and N themselves. In the row
where λ and ρ are pinned three times too high, delta and N go to 0.176 and 2647 — worse than the
flat prior. Under the flat prior they are pinned nowhere at all, running off to λ = −11.9 and
ρ = −0.24, which are negative rate constants and physically impossible.

**This is the sharpest lesson of the investigation.** In a partially identified model the estimates
of the *identified* parameters are set by wherever the unidentified ones come to rest. An improper
prior does not merely leave one parameter undetermined; it corrupts parameters that look perfectly
well determined, and no curvature-based diagnostic can see it — the Laplace standard deviation for
delta is 0.4% of its value in every one of these rows. Only the ridge walk of §2 reports the
problem, and only a proper prior fixes it. On this evidence even a prior that did nothing but
enforce positivity of the rate constants would be a large improvement over the current default.

---

## 6. Recommendations

1. **Set a proper prior on θ.** `theta_conf = 0` is the current default and leaves the posterior
   improper whenever a parameter is unidentified — which, on the four test systems, is one in four.
   A prior standard deviation of 20–50% of the guess was enough on HIV to remove the null
   direction, make the posterior proper, recover the effective sample size fourteenfold, and move
   the *identified* parameters from 48% and 79% error to under 4%. I have not changed the default,
   since doing so would silently alter every existing result, but I would.
2. **Run the diagnosis before the inference.** Mode validity, globality by multistart and
   β-homotopy, identifiability from the scaled Hessian, and properness by ridge walk cost 5–12
   seconds total and need no sampler. They decide whether the question is well posed, and on two
   of four systems the answer changes what should be reported.
3. **Use the profiled posterior where the gate passes**, at 4.5 s and below the reference floor,
   and the Laplace approximation where it does not.
4. **Read a reference chain's divergence count before trusting it.** Hes1's reports 18%.
   On a weakly identified posterior a long NUTS run is not automatically a gold standard.

## 7. Summary across the four systems

| system | regime | best method | vs floor |
|---|---|---|---|
| fn | identified, well posed | profiled, 4.5 s | **below the floor** |
| lorenz | mostly identified, chaotic | profiled, 4.7 s | 13× floor, but 12× better than the mode |
| hes1 | nothing identified, reference unusable | the mode itself | mode is at the floor; corrections hurt |
| hiv | improper under a flat prior | none, until a prior is set | no reference can exist |

The profiled posterior wins wherever the problem is identified, by margins of 12× to 200×, and the
gate correctly declines on the one system where the mode is already the right answer. The
third-order correction of investigation 4 is superseded: it helps on FitzHugh–Nagumo and actively
hurts on Lorenz and Hes1.

## 8. What is not settled

No HIV reference exists or can exist under the flat prior. With a proper prior it could, and that
is the natural next measurement: the diagnosis says the posterior is then proper, so a chain
should converge, and the profiled result could be checked against it rather than against the truth.

The §5 confound has been checked and resolved: with the prior centred at half the true delta and N
the estimates still recover to 0.470 and 940, so the prior buys properness rather than the answer.
What remains open is how to choose the prior on the unidentified parameters, since that choice
determines the identified ones — a positivity constraint on rate constants is the obvious floor,
and the diagnosis of §2 says which parameters the choice will matter for.


---

## 9. Correction: the GP hyperparameter fit was the dominant error

The posterior trajectories look wrong on every system except FitzHugh–Nagumo. The cause is not the
posterior approximation — the fitted trajectory bands match the reference chain to within a few
percent everywhere — but the mode itself, and specifically what happens **between** observations.

On HIV's `T_U`, against a noise scale of σ = 3.16:

| | rms / σ |
|---|---|
| MAP vs data, at observed points | 0.007 |
| MAP vs truth, at observed points | 1.06 |
| MAP vs truth, at **unobserved** points | **74.5** |

The trajectory passes exactly through every observation and plunges to 9.5e-08 between them, where
the truth ranges 168 to 600. Observations sit at every other grid point, so the interleaved points
are held by the GP prior alone — and the fitted lengthscale is **1.05e-04** against a grid spacing
of 0.1.

### The cause

`fit_phisigma` optimises `log(phi1), log(phi2), log(sigma)` by BFGS from `x0 = zeros`, i.e. from
`phi1 = phi2 = sigma = 1` whatever the data's units are. HIV's `T_U` has a marginal variance of
5.6e4, so the start is five orders out, and BFGS — whose step sizes and convergence tolerances are
absolute in its own variables — settles on a lengthscale that fits the observations as white noise.

Trajectory quality is governed entirely by the resulting ratio of lengthscale to grid spacing:

| system | ℓ/dt (before) | trajectory error | ℓ/dt (after) | trajectory error |
|---|---|---|---|---|
| fn | 13, 12 | 3.5%, 7.2% | 13, 12 | 3.5%, **2.6%** |
| lorenz | 12, 9.1, 1.1 | 22.9%, 25.3%, 14.2% | 12, 9.1, 17 | **8.3%, 11.8%, 5.3%** |
| hes1 | 0.14, 0.15, 0.16 | 71.6%, 73.2%, 100.8% | 8.0, 6.3, 3.9 | **5.8%, 13.8%, 30.0%** |
| hiv | 0.001, 0.011, 12 | 69.5%, 6.4%, 0.3% | 50, 50, 8.7 | **0.4%, 0.9%, 0.1%** |

### The fix

The optimisation now runs on `phi = scale * exp(u)` from `u = 0`, where the scales are the data's
own marginal variance and the FFT lengthscale estimate the code already computes. A unit step then
means a factor of e regardless of units, so the optimiser behaves identically on every component
of every system.

A scale-free start alone is not sufficient. It pushed HIV's `T_I` lengthscale to 20.56 — the entire
20-unit time span — which makes `K^-1` **indefinite**, eigenvalues −24 to +11, and its Cholesky in
the solver NaN. Too long is as broken as too short: the Matern derivative kernel degenerates as the
lengthscale approaches the span. The fit is confined to `[2 dt, span/4]`, both ends being
properties of the discretisation rather than of the data.

### What it invalidates

| claim in §2, §4, §5 | status |
|---|---|
| HIV's posterior is improper in λ | **false** — 0 null directions, λ = 36.3 against a truth of 36 |
| Two of four systems are not identified | **false** — all four are, and recover their parameters |
| Hes1 collapses five of seven parameters to ~0 | **false** — all seven within 1.4 posterior sd of truth |
| No HIV reference chain can exist | **false** — it is well conditioned now |
| A proper prior rescues HIV's identified parameters | the effect was real but the cause was the GP |

HIV's condition number falls from 4.1e17 to 4.0e2, which also removes most of the motivation for
the Jacobi scaling of §2 on that system — though the scaling remains correct and costs nothing,
and the argument for it does not depend on any one benchmark.

`diagnose()` now reports ℓ/dt per state component and fails when it drops below 1, since nothing
else in the pipeline could see that the states were unconstrained.

### The lesson

Every quantitative claim in investigations 5 and 6 about which parameters the data determines was
measuring a hyperparameter-fitting bug. The bug was upstream of everything I was studying, it
produced plausible-looking posteriors, and the diagnostics I had built — mode validity,
identifiability, properness — all reported it faithfully as a property of the model. What exposed
it was looking at the trajectories, which none of those diagnostics examined and which I had not
plotted once in two investigations.


---

## 10. Why Hes1 became hard for NUTS

After the GP fix Hes1's reference went the wrong way: R-hat 1.005 → **1.76**, leapfrog steps per
chain 366k → **2.79M**. NUTS is taking maximum-depth trees and still not mixing.

### The measurement

The reference samples in coordinates whitened by the exact Hessian **at the mode**, which is exact
only if the curvature is the same everywhere. Writing `M = L' H(x) L`, `cond(M) = 1` at the mode by
construction; what matters is a standard deviation away:

| system | cond(M) over posterior | worst | min eig(M) | reference R-hat | divergences |
|---|---|---|---|---|---|
| hiv | **1.27** | 1.85 | +0.744 | **1.0001** | **0%** |
| fn | 324 | 3.1e3 | −4.99 | 1.006 | 1.4% |
| lorenz | 552 | 5.6e4 | −1.23 | 1.007 | 1.1% |
| hes1 | **1.8e6** | **3.0e8** | −6.11 | **1.76** | 13% |

The ordering is exact, and it is reproduced by **Laplace draws** rather than reference draws
(3.8e2 / 1.1e6 / 1.2 / 1.9e2), so the difficulty is knowable in advance from about eight extra
Hessian evaluations. This is now reported by `diagnose()`; it is not visible in the condition
number at the mode, which for Hes1 is 1.5e5 against the 1.1e6 away from it.

### The cause

Local curvature varies by a factor of 35.7 across Hes1's posterior, against **1.01** for HIV.
Regressing log of the largest eigenvalue on the coordinates:

| driver | correlation | slope |
|---|---|---|
| `max X[2]` — the peak of the **never-observed** component H | **+0.933** | 1.713 |
| parameter `a` | −0.902 | −97.6 |
| parameter `f` | +0.881 | 0.084 |

Hes1's states are log-scale (`P, M, H = exp(X)`) and `a` enters the field as `-a*H` and `-a*P`, so
a parameter multiplies an exponentiated state — and that state carries no data. A textbook funnel,
generated by the one component nothing observes.

### Why it appeared only now

Before the fix the parameters were unidentified and pinned near zero. With `a ~ 1e-9` the `a*H`
coupling was inert, the posterior was flat, and NUTS wandered it easily to R-hat 1.005 while
exploring a region that meant nothing. Now `a` is determined at 0.032 and the coupling is live.
**This is not a regression**: the problem went from flat, easy and meaningless to concentrated,
hard and correct.

### No fixed metric can sample it

A mass matrix must whiten every Hessian at once, so the obstruction is their mutual disagreement:

| system | pairwise cond(Hi^-1 Hj) | mode metric | averaged metric |
|---|---|---|---|
| hiv | 1.18 | 1.18 | 1.14 |
| fn | 2.1e3 | 3.8e2 | 59 |
| lorenz | 9.5e10 | 1.1e2 | 46 |
| hes1 | **1.7e13** | 3.7e5 | **5.2e11** |

On Hes1 the averaged metric is six orders *worse* than the mode's. Plain HMC or NUTS cannot work at
any budget. Lorenz is the near-miss worth noting: its pairwise figure is also enormous, yet a good
fixed metric still reduces it to 46.

### Options, in the order I would try them

1. **Observe H.** The funnel is generated by a component with no data; any observation of it
   largely dissolves the geometry. A change to the test case rather than the method, and by far
   the cheapest thing to establish first.
2. **Reparameterise to profiled coordinates** — sample `(theta, y)` with
   `X = X*(theta) + L(theta) y`, which removes the coupling by construction and is exactly what the
   profiled method already computes. Needs gradients through an argmin (implicit function theorem,
   already implemented) and through a log-determinant (third derivatives, available from
   `hess R_a = sqrt(b) (Lk^T)_a hess f`).
3. **Riemannian HMC with a SoftAbs metric** — the standard answer, same third derivatives, slow per
   step, but does not depend on the profiled construction being right.
4. **MCMC on the profiled theta density** — easy at p = 7, but it shares the inner Laplace
   approximation with the method under test, so it checks the parameter-space integration only and
   is **not** an independent reference.

The profiled posterior is structurally unaffected by this funnel, since it integrates the states
out at each theta rather than traversing the joint geometry: ESS 21% and khat −0.06 on Hes1 where
NUTS fails. Suggestive rather than established — without a reference the answer cannot be scored.
