# Phase 5 build order — adaptive experiment design

Companion to `model-specification.md` and `test-plan.md`. Covers one question:
given a posterior and the data that produced it, **which experiment should be
run next, and at which time points?**

Phases 1–4 are implemented. The PyCCE backend, previously numbered 5, becomes
phase 6; numbering follows build order everywhere else in this project and the
adaptive engine is wanted first.

---

## 1. What the old implementation does

`adaptive_exp.py` on `old_rjmcmc_code`, 12.7 kB, twelve functions. It works and
the physics reasoning in it is sound. Summarised so the reimplementation can
keep what is right and be explicit about what changes.

### 1.1 Posterior to particles

`spin_bath_posterior` and `spin_bath_posterior_with_indices` turn the sampler's
output — a list of index lists — into a **weighted particle set** over distinct
baths. Each bath is canonicalised by sorting its $(A_\parallel, A_\perp)$ pairs,
so ordering is irrelevant, with optional rounding to merge near-duplicates.
Weights are multiplicities divided by the total; one representative index list
is kept per bath.

This is the right abstraction and the reimplementation keeps it. Everything
downstream sees particles, never a chain.

### 1.2 Expected information gain

`compute_EIG_particle` is a nested Monte-Carlo estimator of the mutual
information between the bath and the data:

1. precompute predictions for every particle under the candidate → $P$, shape
   $(K, N_t)$;
2. $M$ times: draw a "true" particle from the weights, simulate
   $d = P_{k} + \sigma\varepsilon$, evaluate the Gaussian log-likelihood of
   every particle, form $\log Z = \mathrm{logsumexp}(\log w + \ell)$, and
   accumulate $\ell_{k} - \log Z$;
3. average.

`compute_EIG_all_experiments` does the same across a list of candidates using
**common random numbers** — the same sampled particles and the same noise
realisation for every candidate. That is a deliberate and correct variance
reduction for a comparison, and it is kept.

### 1.3 Information density and time allocation

A second, cheaper design route:

- `information_density` — the weight-averaged variance of the predictions at
  each time point, divided by $\sigma^2$. Where the posterior's members
  disagree is where measuring separates them.
- `allocate_measurement_time` — averaging time $\propto \sqrt{\text{density}}$,
  normalised to a budget.
- `prune_timepoints` — drop points receiving under 5% of the maximum.
- `build_optimized_experiment` — reassemble with the surviving times and a
  per-point `data_weight`.

`optimize_experiment` chains these: dense grid → density → allocation → prune →
score by EIG → `utility = EIG / cost`.

### 1.4 Support

`merge_exp_params` combines single-experiment dicts; `prediction_matrix`
returns $(P, w)$; `make_dense_time_experiment` builds the design grid;
`plot_information_diagnostic` draws three panels — posterior predictions with
the mean and the truth, the information density with the selected points, and
the ground-truth coherence on both grids.

---

## 2. What changes, and why

Nothing above is wrong. The changes are about making it composable and about
measuring the things it assumes.

**Structure.** The old code is a module of functions bound to pandas
DataFrames, dict-shaped `exp_params`, and `calculate_coherence_with_T2` from
`rjmcmc`. The reimplementation depends on none of those: it takes a
`ParticleSet`, an `ExperimentSet` and a `ForwardModel`, which is what makes it
agnostic to how the posterior was obtained.

**Agnosticism, concretely.** The engine's only contact with the sampler is
`ParticleSet.from_trace`. A single chain, a pooled ensemble, a future sampler,
or an array typed in by hand all produce the same object, and the designer
cannot tell which. No import of `HybridDriver`, `Schedule`, `EnsembleRunner` or
any algorithm appears anywhere in `design/`, and a test asserts that.

**Two interchangeable utilities.** EIG and predictive variance are alternative
answers to "how informative is this candidate". The old code hard-wires their
roles — variance for point selection, EIG for scoring. They become
implementations of one `DesignUtility` interface, so either can do either job
and a third can be added without touching the designer.

**Things asserted that should be measured.**

- $\sqrt{\text{density}}$ allocation is called a "near-optimal D-design rule".
  Perhaps; it is untested here and the alternative — greedy sequential
  selection against the utility itself — is a direct comparison.
- The 5% prune threshold is a magic number.
- The nested MC estimator of EIG is **biased at finite $K$**: the same particle
  set supplies both the sampled truth and the evidence, so the estimator
  overstates the gain when the posterior has few members. The bias shrinks with
  $K$ and the size of the effect on *ranking* — which is all that matters for
  choosing a candidate — is measurable.

**Degenerate posteriors.** If the posterior has collapsed to one particle the
information density is identically zero and every candidate scores the same.
That is the correct answer and must not be a crash or a silent uniform grid; it
must be reported.

**Per-point measurement time has no home yet.** The old code attaches
`data_weight` to the experiment, which only means something if the likelihood
weights points by it — effectively a per-point $\sigma_j$. `Experiment` carries
one `sigma` per experiment, not per point. See §6.

---

## 3. Work units

| unit | what it adds | blocks |
|---|---|---|
| **5a** | `design/particles.py` — the posterior as weighted particles | everything |
| **5b** | `design/utility.py` — `DesignUtility`, EIG, predictive variance | 5c, 5d |
| **5c** | `design/selection.py` — choosing which points to keep | 5d |
| **5d** | `design/designer.py` — ranking candidates, proposing an experiment | T9 |
| **5e** | `post/plots.py` — the information diagnostic | — |

### 5a — `ParticleSet`

```
ParticleSet(site_idx, k, weight, dA_par, dA_perp, n_sites, k_max)
    .from_trace(trace, site_table, *, burn, stride, tol)
    .predictions(expset, site_table, model)   -> (K, n_points)
    .n_particles, .effective_size
```

Canonicalised **on couplings within `tol`, not on site indices**, for the same
reason detection is: symmetry-equivalent baths are one physical hypothesis, and
grouping by index would split it into six particles and inflate $K$.

`effective_size` is Kish's $1/\sum w^2$. A posterior of 400 draws that collapsed
to two distinct baths has an effective size near 2, and the EIG bias is a
function of that rather than of the draw count.

`predictions` evaluates every particle in **one vectorised pass** by building a
single `State` with one replica per particle — the trick `predictive_from_arrays`
already uses.

**Unit tests** — identical baths merge to one particle of weight 1; a symmetry
orbit merges; weights sum to 1 and follow multiplicity; `effective_size` is $K$
for uniform weights and near 1 for a spike; predictions have shape
$(K, n_\text{points})$ and match a per-particle loop; a one-particle posterior
is legal, not an error.

### 5b — `DesignUtility`

```
DesignUtility(ABC).score(predictions, weights, noise, rng) -> float
ExpectedInformationGain(n_draws=64, common_random=True)
PredictiveVariance()
```

Both take arrays only — no model, no table, no experiment. That is what makes
them unit-testable against cases with a known answer, and interchangeable.

**Unit tests** — EIG is zero when all particles predict identically, because
nothing is learnable; EIG rises with the separation between two particles'
predictions; EIG is bounded above by the prior entropy $-\sum w\log w$; with
`common_random` two candidates scored in one call share draws and the *ranking*
is stable across seeds where independent draws would flip it; predictive
variance is zero for identical predictions and scales as $1/\sigma^2$; both are
invariant to permuting the particles.

### 5c — `PointSelector`

```
PointSelector(ABC).select(predictions, weights, noise, budget) -> (indices, weight)
InformationDensity(power=0.5, prune_fraction=0.05)   # the old rule
GreedyUtility(utility)                               # sequential, exact
UniformThinning()                                    # the control
```

A selector returns **both** the points it kept and how much of the budget each
received, because the budget is total measurement time (§6, question 2). The
weights go straight into `Experiment.weight`, where the likelihood reads a
point measured four times as long as having a quarter the variance.

`InformationDensity` reproduces the old behaviour with its two magic numbers
exposed rather than buried. `GreedyUtility` is the honest comparison: pick the
point that most improves the utility, repeat.

**Unit tests** — every selector returns distinct in-range indices with
non-negative weights summing to the budget; `UniformThinning` on a uniform grid
returns evenly spaced indices with equal weights; `InformationDensity` puts its
time where the predictions disagree, not where they agree, on a constructed
case with one informative region; `GreedyUtility` never selects a point twice;
a budget of zero raises rather than returning an empty experiment.

### 5d — `ExperimentDesigner`

```
ExperimentDesigner(utility, selector, model, site_table)
    .rank(particles, candidates)  -> (n_candidates,) utility
    .propose(particles, candidates, *, budget, exclude=None) -> Experiment
```

`propose` picks the best candidate by `rank`, then spends `budget` on it with
the selector, and returns a real `Experiment` — points **and** their weights —
so the result goes straight back into the sampler.

**What the old data is for.** The posterior *is* the summary of the old data;
conditioning on it twice would double-count. The existing data enters in two
narrow ways only: `exclude` prevents re-proposing points already measured, and
the noise level is inherited. Joint design over old and new together is a
different calculation and is not this function.

**Unit tests** — `rank` returns one score per candidate; the proposed
experiment's weights sum to the budget; its `tau` is a subset of the chosen
candidate's; `exclude` removes points; a candidate list of one is legal;
an empty list raises; **a collapsed posterior is reported rather than
silently ranked** — one particle means every candidate is equally uninformative
and the designer says so; a source-level test asserts `design/` imports no
sampler, driver or scheduler.

### 5e — the information diagnostic

`plot_information_diagnostic(particles, expset, chosen, ...)`: posterior
predictions with the weighted mean, the information density with the selected
points marked, and the data. The old version took a ground-truth spin list; the
new one must work without one, since the purpose is measured data.

---

## 4. The ladder rung

### T9 — adaptive design beats a uniform grid

The claim the whole unit rests on, and it needs a control at both ends.

- **Positive.** Fit a bath, design a follow-up under a fixed **total
  measurement time**, simulate data from the truth at the chosen points with
  noise $\sigma/\sqrt{w_j}$, re-fit on the combined data, and show detection or
  residual improving relative to *the same total time* spread uniformly over
  the window.
- **Negative control.** An *anti-design* spending the same budget on the
  **least** informative points must do worse than uniform. Without it, a
  positive result is consistent with "any extra measurement helps", which is
  not the claim.

Equal *time*, not equal point count, is what makes the comparison fair: a
design that wins by measuring more is not a design.
- **Degenerate control.** With a collapsed posterior, adaptive and uniform must
  be indistinguishable; there is nothing to exploit.

Thresholds are set the standard way: run it, record the metric, record what the
metric reads with the mechanism disabled, and set the threshold between them
with margin. Expect seed dependence and pool accordingly — that has been true
of every comparison in this project so far.

---

## 5. What this can and cannot show

Adaptive design chooses where to measure so that the *posterior's members*
disagree most. That is a statement about the posterior, not about the bath. If
the posterior has missed the true configuration entirely — which
`test-plan.md` §5.10 documents happening, a single chain reporting a wrong
coupling at frequency 1.00 for 5,000 steps — then the design optimises the
separation of hypotheses that are all wrong, and will do it very efficiently.

The engine should therefore consume a **pooled ensemble** posterior, not one
chain, for the same reason the coupling posterior should. Worth stating in the
API docs rather than leaving to be discovered.

---

## 6. Open questions

1. **Resolved — per-point measurement time.** `Experiment.weight` carries the
   relative averaging time and the likelihood reads it as
   $\sigma_{\text{eff}} = \sigma_e/\sqrt{w_j}$, reducing exactly at $w = 1$.
   Specification §7.1; implemented ahead of the rest of this phase.
2. **Resolved — the design budget is total measurement time**, with
   `utility = EIG / cost` as in the original. Candidates are compared at equal
   spend rather than at equal point count, which is also what makes T9's
   uniform control fair: a design that wins by measuring more is not a design.
   This is the choice the per-point weights exist to express.
3. **Does the EIG bias change the ranking?** Measurable, and worth measuring
   before the estimator is trusted at small effective particle counts.
4. **Should `rank` cache predictions across calls?** A candidate list of 100
   dense grids times a few hundred particles is the expensive part, and the
   predictions are reusable while the posterior does not change.
