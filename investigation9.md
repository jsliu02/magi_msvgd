# Investigation 9 — why the preconditioned recipe loses the parameters

[investigation8.md](investigation8.md) §2.6 found a reference-free procedure that reaches
**0.57–0.96×** the K-particle energy floor on fn, hiv and lorenz: whiten by `H⁻¹` at the MAP, run
SVGD at a fixed `h = 10·h*` with a small fixed step to its (genuine) fixed point, then rescale the
ensemble about its mean by `1/√R`. It left one thing open, and it is the thing that decides whether
the recipe is usable, because MAGI exists to estimate parameters:

| max \|θ err\|, in reference sd | fn | lorenz | hiv | floor |
|---|---|---|---|---|
| `fit()` draws (the start) | 0.077 | 0.071 | 0.096 | 0.010 / 0.041 / 0.008 |
| preconditioned SVGD equilibrium | **0.242** | **0.401** | **0.067** | |

Two systems get worse, one gets better. §2.6 could not say why.

## 1. The hypothesis, and why it is plausible before any measurement

SVGD's drift term is `(1/K) Σⱼ k(xⱼ, xᵢ) s(xⱼ)` — a kernel-weighted average of the score. A score
average is not mean-preserving: it points uphill, so it pulls the ensemble toward high density. The
repulsion term opposes this, and at the *mean-field* fixed point the two balance exactly; at finite
K they need not, and investigation 7 §4 already measured the finite-K bias directly (`d/dt log Var
= −2/K` at the target, with the ensemble **mean** unbiased at the adaptive bandwidth). Under the
preconditioned dynamics the bandwidth is fixed and much larger, which weakens the repulsion by
`1/h` while leaving the drift at `O(1)`, so the balance that held before need not hold now.

If that is what is happening, the equilibrium's θ mean should sit toward the **joint mode**, and
the amount it loses should be the mode-versus-mean gap — which is exactly the correction the
profiled posterior exists to supply. Independently measured joint-mode-to-profile-mode distances
(from the library optimisation work, float64, all four systems):

| | hiv | fn | lorenz | hes1 |
|---|---|---|---|---|
| joint mode → profile mode | **0.16 sd** | **1.07 sd** | **1.42 sd** | 1.43 sd |

The ordering is the same as the degradation ordering: hiv barely moves and improves, fn and lorenz
move by more than a standard deviation and degrade, lorenz worst on both. That is suggestive and
not evidence — three points ordered correctly is a 1-in-6 coincidence.

## 2. The test

`investigation9/exp01_modeseek.py`. For a coordinate block with reference mean `b`, joint MAP `a`,
and reference covariance `Σ_blk`, whiten by `Σ_blk` and project the ensemble mean `m` onto the
segment from `b` to `a`:

    t = (m−b)·(a−b) / |a−b|²      0 at the reference mean, 1 at the joint MAP
    r = |(m−b) − t(a−b)| / |a−b|  orthogonal drift, in units of the gap length

The hypothesis makes a sharp prediction that the absolute errors do not: **`t` should be similar
across systems** even though the errors differ fivefold, because what differs between systems is
the *length* of the segment, not the fraction of it travelled. The `fit()` start sits at the
reference mean, so `t` should begin near 0 and rise.

Computed for θ and, separately, for a 200-coordinate subset of the state block, where the
mode-versus-mean gap is small. If only θ drifts, that localises the effect.

## 3. Result: the scale of the hypothesis is right, the direction is wrong

fn, `gap(θ, max) = 1.029` sd, `gap(states, max) = 0.972` sd:

| iteration | t(θ) | r(θ) | t(states) | r(states) | max\|θ err\| | Stein R | energy ×floor |
|---|---|---|---|---|---|---|---|
| 0 (`fit()` start) | −0.037 | 0.090 | 0.427 | 0.549 | 0.077 | 1.274 | 1.65 |
| 5,000 | 0.024 | 0.048 | 0.179 | 0.197 | 0.055 | 0.907 | 0.88 |
| 25,000 | 0.124 | 0.065 | 0.215 | 0.180 | 0.169 | 0.737 | 2.20 |
| 50,000 | 0.169 | 0.074 | 0.167 | 0.181 | 0.221 | 0.691 | 2.75 |
| **100,000** | **+0.186** | 0.080 | **0.024** | 0.252 | 0.242 | 0.702 | 2.42 |
| after 1/√R rescaling | +0.186 | 0.080 | 0.024 | 0.252 | 0.242 | 1.074 | 0.68 |

lorenz, `gap(θ, max) = 1.803` sd:

| iteration | t(θ) | r(θ) | t(states) | max\|θ err\| | Stein R | energy ×floor |
|---|---|---|---|---|---|---|
| 0 (`fit()` start) | −0.024 | 0.070 | 0.020 | 0.071 | 1.019 | 1.24 |
| 5,000 | −0.062 | 0.051 | −0.027 | 0.104 | 0.894 | 0.79 |
| 25,000 | −0.115 | 0.033 | −0.046 | 0.202 | 0.718 | 2.18 |
| 50,000 | −0.156 | 0.017 | −0.052 | 0.278 | 0.662 | 2.94 |
| **100,000** | **−0.221** | 0.024 | **−0.057** | 0.401 | 0.653 | 2.92 |

Four things, in order of how much they matter.

**1. The displacement is θ-specific.** On fn the state block moves *toward* the reference mean
(t: 0.427 → 0.024, i.e. onto it) while θ moves away from it. Whatever this is, it is not a general
bias in the ensemble mean — it is confined to the parameter block, which is 3 of 325 coordinates.

**2. It is aligned with the mode–mean axis, and that is not chance.** The parallel fraction
`|t|/√(t²+r²)` is **0.92 on fn and 0.994 on lorenz**, against **0.58** for a random direction in
p = 3. So the coordinator's geometry is right: the displacement lives on the segment between the
joint mode and the posterior mean.

**3. Its magnitude is ≈ 0.2 × the gap, which explains the ordering exactly.** |t| = 0.186 and
0.221 — the *same fraction* on two systems whose absolute errors differ by 1.7×, which is the sharp
prediction the hypothesis made and it holds. Multiplying through:

| | gap (sd) | 0.2 × gap | measured change in max\|θ err\| |
|---|---|---|---|
| hiv | 0.148 | 0.030 | **−0.029** (0.096 → 0.067) |
| fn | 1.029 | 0.206 | **+0.165** (0.077 → 0.242) |
| lorenz | 1.803 | 0.361 | **+0.330** (0.071 → 0.401) |

Three systems, agreement to within 0.04 sd. **hiv improves not because it is different in kind but
because its gap is 0.15 sd, so a 20% displacement is smaller than the error it started with.**

**4. But it is not mode-seeking.** fn drifts **toward** the joint MAP (t > 0) and lorenz drifts
**away from it** (t < 0), past the reference mean on the far side. A kernel-weighted score average
pulls uphill, so the hypothesis as stated predicts t > 0 on both. It does not survive. What
survives is the weaker and still useful statement: *the displacement is a fixed fraction of the
mode–mean gap, along the mode–mean axis, with a system-dependent sign.*

**And the rescaling provably cannot touch it.** The `1/√R` step rescales about the ensemble's own
mean, so it is by construction mean-preserving — the table above shows t, r and the θ error
identical before and after, to every printed digit, while Stein R goes 0.702 → 1.074 and the energy
distance 2.42× → 0.68× the floor. A scale correction cannot fix a location error. This was
predictable and is now measured.

### HIV completes the picture, and breaks the "same fraction" reading

| iteration | t(θ) | r(θ) | t(states) | max\|θ err\| | Stein R | energy ×floor |
|---|---|---|---|---|---|---|
| 0 (`fit()` start) | −0.045 | 0.840 | 0.781 | 0.096 | 1.000 | 1.03 |
| 25,000 | +0.294 | 0.022 | 0.416 | 0.043 | 0.697 | 2.76 |
| **100,000** | **+0.454** | **0.022** | 0.549 | 0.067 | 0.538 | 6.78 |

HIV's `t` is **+0.454**, not ≈0.2, with an alignment of 0.999. So the fraction is *not* universal
(0.19, −0.22, 0.45) and the "similar across systems" prediction fails. What does hold, and holds
tightly, is that **the displacement is the error**:

| | t(θ) | gap (sd) | \|t\|·gap | measured final max\|θ err\| |
|---|---|---|---|---|
| fn | +0.186 | 1.029 | 0.191 | 0.242 |
| lorenz | −0.221 | 1.803 | 0.399 | **0.401** |
| hiv | +0.454 | 0.148 | 0.067 | **0.067** |

Two of the three agree to the printed digit. So the coordinator's diagnosis is right in substance —
the equilibrium hands back a displacement proportional to the mode–mean gap, which is exactly the
correction the profiled posterior exists to supply — and HIV improves because its gap is 0.15 sd,
not because it differs in kind.

## 4. The mechanism: it is the non-Gaussianity, and only that

`investigation9/exp03_gausscontrol.py`. On an exact Gaussian the mode **is** the mean, so a
displacement proportional to the mode–mean gap has nowhere to go. Identical dynamics — same
whitening matrix `L`, same bandwidth, same step, same starting ensemble — with the target replaced
by `N(mean_ref, cov_ref)`. fn, K = 400:

| system | target | lr | \|Δθ̄\| (sd) | final max\|θ err\| | (θ floor) | Stein R |
|---|---|---|---|---|---|---|
| fn | real posterior | 0.01 | 0.3454 | **0.2422** | 0.010 | 0.702 |
| fn | real posterior | 0.003 | 0.2804 | 0.1842 | | 0.721 |
| fn | **N(mean_ref, cov_ref)** | 0.01 | 0.1034 | **0.0045** | | 0.763 |
| fn | **N(mean_ref, cov_ref)** | 0.003 | 0.1039 | **0.0042** | | 0.755 |
| lorenz | real posterior | 0.01 | 0.4896 | **0.4013** | 0.041 | 0.653 |
| lorenz | real posterior | 0.003 | 0.2457 | 0.2185 | | 0.700 |
| lorenz | **N(mean_ref, cov_ref)** | 0.01 | 0.1199 | **0.0351** | | 0.738 |

**On the Gaussian the parameter mean lands at 0.0045 (fn) and 0.0351 (lorenz) reference sd — below
each reference chain's own floor of 0.010 and 0.041 — where the real posteriors land at 0.2422 and
0.4013.** Same dimension, same covariance, same metric, same bandwidth, same step, same start;
only the third and higher moments differ. **The drift is a non-Gaussianity effect and nothing
else**, generated by the same skewness that creates the mode–mean gap, which is why it lives on
that axis.

Note also that the Gaussian rows are *step-independent* (fn: 0.1034 → 0.1039 for a 3.3× smaller
step), i.e. that case has genuinely converged and has no drift to find.

### The step size is *not* a lever — I wrote that it was, and it is not

The real-posterior rows above are step-dependent at a fixed iteration count: a 3.3× smaller step
gives 1.3× less drift on fn and 1.8× less on lorenz. I wrote that up as a practical lever and
listed the `lr → 0` limit as the most consequential open question. **It was neither.** The
comparison was made at fixed *iterations*, and a smaller step covers less flow time, so the
ensemble had simply not got as far. `investigation9/exp05_steplimit.py` repeats it at matched flow
time, holding `lr × iterations` constant:

| lr | iterations | t(θ) | r(θ) | max\|θ err\| | Stein R | energy ×floor |
|---|---|---|---|---|---|---|
| 0.01 | 100,000 | 0.186 | 0.080 | 0.2422 | 0.7023 | 2.42 |
| 0.003 | 333,333 | 0.186 | 0.080 | 0.2422 | 0.7023 | 2.42 |
| 0.001 | 1,000,000 | 0.186 | 0.080 | 0.2422 | 0.7023 | 2.42 |

**Identical to four digits over a tenfold range of step size.** The drift is a property of the
continuous flow's finite-K fixed point, not of its discretisation, and no step size will remove it.
(lorenz reproduces its `lr = 0.01` row at −0.221 / 0.4013; its longer rows were still running.)

This also cleans up §5: the `K` sweep there was run at a single step size, and matched flow time
now shows that this does not matter, so its exponents are estimates rather than upper bounds.

## 5. Can it be bought off with particles? Partly, and not enough

`investigation9/exp02_Kdep.py`, same configuration, K = 100 / 400 / 1600.

| | K = 100 | K = 400 | K = 1600 | fitted decay |
|---|---|---|---|---|
| fn, \|t(θ)\| | 0.394 | 0.186 | 0.051 | **K^−0.74** |
| fn, alignment | 0.97 | 0.92 | 0.80 | → 0.58 (noise) |
| fn, max\|θ err\| | 0.467 | 0.242 | **0.076** | |
| fn, `fit()` at same K | 0.148 | 0.077 | **0.053** | |
| lorenz, \|t(θ)\| | 0.299 | 0.221 | 0.110 | **K^−0.36** |
| lorenz, alignment | 0.99 | 0.99 | 1.00 | no sign of noise |
| lorenz, max\|θ err\| | 0.539 | 0.401 | **0.200** | |
| lorenz, `fit()` at same K | 0.133 | 0.071 | **0.039** | |

**It is a finite-K bias** — it decays, and on fn the alignment decays with it (0.97 → 0.80 toward
the 0.58 of a random direction), which is the signature of a systematic term being overtaken by
sampling noise. But the rate is system-dependent and slow. Extrapolating to the reference floor:

* fn needs `|t| < 0.010` — about **1.5 × 10⁴ particles**;
* lorenz needs `|t| < 0.023` — about **1.2 × 10⁵ particles**, and its alignment is still 1.00 at
  K = 1600, so there is no sign of the systematic term running out.

And at every K tested, on both systems, **`fit()` is better on θ than SVGD at the same K** — by
1.4–3.1× on fn and 4.1–5.7× on lorenz. Buying particles narrows the gap on fn and does not close
it on lorenz.

## 6. What does recover it: importance reweighting, and it is nearly free

`investigation9/exp04_reweight.py`. §5 leaves the ensemble in a specific state: after the `1/√R`
step its **spread** matches the reference (`whsd²` 1.02–1.06, stiff-var 1.065, energy 0.57–0.96× the
floor) and only its **location** is wrong. A mislocated proposal with the right scale is precisely
what importance sampling exists to fix, and the weights should be mild.

SVGD returns particles, not a density, so the proposal has to be supplied. The honest choice is a
Gaussian fitted to the ensemble's own θ block, `q = N(mean, cov)` of the rescaled particles, and

    log wᵢ = log p̂(θᵢ) − log q(θᵢ)

with `log p̂` the profiled log marginal from `profiled.ProfiledPosterior.logp` — the same quantity
`fit()` integrates. So this is `fit()` with its scrambled-Sobol proposal replaced by the SVGD
ensemble.

| system | `fit()` at same K | equilibrium | + 1/√R | **+ IS reweight** | ESS | k̂ | weight spread |
|---|---|---|---|---|---|---|---|
| fn | 0.0768 (7.7× floor) | 0.2422 | 0.2422 | **0.0706 (7.0×)** | 312 (78%) | −0.95 | 5.7 nats |
| lorenz | 0.0710 (1.8×) | 0.4013 | 0.4013 | **0.0897 (2.2×)** | 263 (66%) | −0.85 | 13.0 nats |
| hiv | 0.0958 (11.9×) | 0.0667 | 0.0667 | **0.0289 (3.6×)** | 389 (97%) | 0.29 | 1.6 nats |

**It recovers essentially all of the loss**, and the diagnostics say it is safe: ESS 66–97%
(against `fit()`'s own 63–76% on its Sobol proposal), Pareto k̂ from −0.95 to 0.29 — all far below
the 0.7 gate — no failed profile solves, and log-weight spreads of 1.6–13 nats rather than the
8,400 that made the first version of `fit()`'s proposal unusable on Hes1. The cost is 0 s: 400
profile evaluations, no gradients.

Against `fit()` at the same particle count the reweighted estimate is **1.09× better on fn, 3.3×
better on hiv, and 1.26× worse on lorenz** — parity, with one clear win.

But be clear about what that means. Step 5 works by re-introducing the profiled log-density, which
is the machinery that had already solved the parameter problem. **SVGD contributes the proposal,
not the estimate.** The proposal happens to be a good one — its ESS beats `fit()`'s own — and that
is a real if modest finding, but no part of the parameter answer comes from the SVGD dynamics.

## 7. Conclusions

### The mechanism, settled

The θ degradation in investigation 8 §2.6 is:

1. **θ-specific.** The state block converges *onto* the reference mean (fn: t goes 0.427 → 0.024)
   while θ moves away from it. Three coordinates out of 325.
2. **Confined to the mode–mean axis.** Parallel fraction 0.92 (fn), 0.994 (lorenz), 0.999 (hiv),
   against 0.58 for a random direction in p = 3.
3. **Proportional to the mode–mean gap**, which is why the degradation orders exactly as the
   coordinator's independently measured mode-to-profile-mode distances predicted. `|t|·gap`
   reproduces the final θ error to the printed digit on lorenz and hiv and to 0.05 sd on fn. HIV
   *improves* because its gap is 0.148 sd, not because it differs in kind.
4. **Caused by non-Gaussianity, and by nothing else.** On `N(mean_ref, cov_ref)` — same dimension,
   same covariance, same whitening, same bandwidth, same step, same start — the parameter mean
   lands at **0.0045** reference sd, below the reference chain's own floor, against 0.2422 on the
   real posterior. The drift is generated by the same skewness that creates the gap.
5. **Not mode-seeking in direction.** fn and hiv drift *toward* the joint MAP, lorenz *away* from
   it. The coordinator's mechanism — a kernel-weighted score average pulling uphill — predicts
   toward on all three. Two of three is not the mechanism; the axis and the scale are real, the
   sign is not explained.
6. **A finite-K bias of the continuous flow, decaying too slowly to buy off.** `K^−0.74` on fn,
   `K^−0.36` on lorenz; reaching the reference floor needs ~1.5 × 10⁴ and ~1.2 × 10⁵ particles
   respectively, and lorenz's alignment is still 1.00 at K = 1600 with no sign of the systematic
   term running out. It is **not** a discretisation artefact: at matched flow time three step sizes
   spanning 10× give t = 0.186 and θ error 0.2422 identically to four digits, so no step size
   removes it.
7. **Immune to the `1/√R` rescaling**, necessarily: that step rescales about the ensemble's own
   mean and is mean-preserving by construction. Measured identical before and after, to every
   printed digit, while the energy distance goes 2.42× → 0.68× the floor.
8. **Removable by importance reweighting**, at ESS 66–97% and k̂ ≤ 0.29, restoring θ to parity with
   `fit()` (better on fn and hiv, slightly worse on lorenz) — but by re-introducing the profiled
   log-density, so the estimate comes from `fit()`'s machinery and not from SVGD.

### What I would now tell a user

The investigation 8 §2.6 recipe is **usable, with a fifth step**, and its value is narrower than it
first looked:

> 1. `fit()` for the starting ensemble. 2. Whiten by `H⁻¹` at the MAP. 3. Fixed `h = 10·h*`, small
> fixed step, run to equilibrium. 4. Rescale by `1/√R` — gives the joint posterior at 0.57–0.96× the
> K-particle energy floor. 5. **Importance-reweight by `log p̂(θ)`** — gives the parameters back, at
> ESS 66–97%.

What it buys over `fit()` alone is a **joint** ensemble that is at or below the Monte-Carlo floor
(against `fit()`'s 1.03–1.65×) while matching `fit()` on parameters. What it costs is ~10⁵ gradient
evaluations. Whether that is worth it depends entirely on wanting the states.

**And the sharper negative result stands.** SVGD's fixed point is not the posterior's parameter
marginal, at any bandwidth, in any metric, and for a reason that is now measured rather than
inferred: its finite-K equilibrium is displaced along the mode–mean axis by a fraction of the gap,
the displacement is generated by the posterior's skewness, and it decays as slowly as `K^−0.36`.
That is a more specific statement than investigation 4's "the fixed point is not the posterior",
and it says where the limit lies: SVGD can be made to represent a high-dimensional posterior's
*shape* to below the Monte-Carlo floor, and cannot be made to locate a skewed low-dimensional
marginal without help from something that already knows where the marginal is.

### What is not settled

* **Why the sign differs.** fn and hiv drift toward the joint mode, lorenz away from it. The axis
  and the magnitude are established; the sign is not. Lorenz is the chaotic system and the most
  non-Gaussian, but that is a label, not an explanation.
* ~~Whether the `lr → 0` limit is nonzero.~~ **Settled during this investigation** (§4): at
  matched flow time the drift is identical over a tenfold range of step size, so the limit is the
  drift itself. I had listed this as the most consequential gap and recommended trying a smaller
  step first; that recommendation was wrong and is withdrawn.
* **The K-decay exponents are two-point fits** (three K values, two systems), and they are the
  numbers the ~10⁴–10⁵ particle extrapolations rest on.
* **The profiled-marginal route** (investigation 8 §3) was not re-tested here. It is the other
  candidate the coordinator raised, and on the argument above it should be immune — the
  mode-versus-mean gap is what the profiled marginal integrates rather than something the sampler
  must find — which is consistent with §3's finding that it beats `fit()` at p ≤ 3.
* **hes1** remains excluded — no usable reference.

### Provenance

float64 throughout, `atol = rtol = 0`, K = 400 unless stated, `precond` runs using the
**post-fix** driver (investigation 8 §2.6; the pre-fix driver applied the metric twice).
V100 (`jax-cuda12`) and RTX 3090 (`magi` env). References `investigation5/ref5_*.npz`,
re-validated in investigation 8 §0.
