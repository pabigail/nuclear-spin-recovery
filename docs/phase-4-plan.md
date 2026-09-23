# Phase 4 build order

Companion to `model-specification.md` (what the model is) and `test-plan.md`
(what a test can show). This document covers only what phase 4 adds: ensembles,
the posterior summary package, cluster execution, the Wasserstein likelihood
variant, and the experimental-data rung.

Phases 1–3 are implemented. Phase 5 (PyCCE backend) is out of scope here.

**Status.** Units 4a–4d are implemented and pushed. Unit 4e is blocked on
experimental data files. Two planned ladder rungs, T7 and T8, were **not**
written; §4 records what stands in their place and what does not. The sections
below are left as they were written, with a status note at the head of each
unit, so the plan reads as a record of what was decided and when — including
where the outcome differed from the intention.

---

## 1. Ordering principle

The same one as phases 1–3: tests first and failing, then implementation, then
calibration against recorded runs with negative controls. Within the phase, work
is ordered by what unblocks what.

| unit | what it adds | blocks | status |
|---|---|---|---|
| **4a** | `post/` — posterior summaries extracted from the test harness | everything below | **done** |
| **4b** | `EnsembleRunner` — independent chains, pooled | 4c, T7 | **done** |
| **4c** | run configuration and Perlmutter submission | T6 at scale | **done** |
| **4d** | Wasserstein likelihood variant | T8 | **done** |
| **4e** | experimental data reader and T6 | — | **blocked on data** |

4a comes first because every other unit is measured through it. 4d is
independent of the rest and can move if something else turns out urgent.

---

## 2. What already exists, and must not be rewritten

Every metric in spec §9.2 is **already implemented** — in
`tests/theory/conftest.py`, as the `Metrics` dataclass and the `metrics`
fixture. None of it is in `src/`.

There is now a second copy. `notebooks/trans_dimensional_and_tempering.py`
defines its own `residual`, `residual_curve` and `detection_rates` because there
was nothing importable to use.

This matters beyond tidiness. The harness exists because a metric written inline
at the point of use produced the measurement error that prompted the test plan:
detection computed against a single final state rather than over the posterior.
Two inline copies is the same exposure, twice.

So 4a is an **extraction, not a new module**. The conftest harness moves to
`src/nuclear_spin_recovery/post/`, and the tests and the notebook both import it.
The existing theory tests are the safety net: they must pass unchanged, with no
threshold edited. A threshold that needs moving during this refactor means the
extraction changed a measurement, which is a defect, not a recalibration.

---

## 3. Work units

### 4a — `post/`

**Done.** 55 unit tests. The extraction was verified by the theory ladder passing 28/28 with no threshold edited. It immediately exposed a defect the three copies had hidden: `false_absence` matched by exact set membership while `detection_rate` matched within a tolerance, reporting FP = 0.224 on a run whose true value is 0.0 — all of it symmetry-orbit rounding in the fourth decimal. It also corrected a claim made in this plan's own reasoning: rounding does *not* merge a symmetry orbit, since its members differ in the fourth decimal. The match tolerance is what merges them.

**New files**

| file | contents |
|---|---|
| `post/__init__.py` | exports |
| `post/metrics.py` | `PosteriorSummary`, `summarize(trace, ...)` |
| `post/detection.py` | coupling matching, `detection_rate`, `false_absence` |
| `post/residual.py` | posterior-predictive signals and residual distribution |
| `post/plots.py` | trajectory diagnostics (spec §9.2, last paragraph) |

**Design points, each already forced by a measurement**

- Summaries are computed **over the posterior**, never a single state. The API
  takes a `Trace`, not a `State`; there is no entry point that accepts one
  configuration.
- Matching is on **couplings within a tolerance**, never on site index.
- Offsets come **from the trace**. Rebuilding a configuration from site indices
  alone pins them at zero and scores a relaxed run as if it had never been
  relaxed — the bug that made a relaxed run read 4.68σ against a diagnostic's
  1.82σ.
- `best_residual` and `median_residual` are both exposed, with the docstring
  stating which comparison each is valid for. A richer model has a *higher*
  median because a typical draw sits away from the prior mean.
- Reading a `Trace` property inside a loop is O(n²); `post/` reads each array
  once and passes arrays down.

**Unit tests** — `tests/unit/test_post.py`

- summary of a one-sample trace equals the hand-computed value
- detection rate is 1.0 for a spin present in every sample, 0.0 for one absent
- symmetry-equivalent sites at different indices score as detected
- a spin 0.05 kHz away matches at `tol=0.1`, one 0.5 kHz away does not
- offsets recorded in the trace reach the predictive signal
- `false_absence` of a trace whose every sample equals the mode is 0.0
- dimension discrepancy is `|mode(k) - k_true|`
- band grouping puts a 30 kHz spin in the 25–100 band, not 5–25
- plotting helpers return a `Figure` and draw the burn-in marker

**Done when** the theory ladder passes with `conftest.py` reduced to fixtures
that call `post/`, no threshold edited, and the notebook importing the same
functions.

### 4b — `EnsembleRunner`

**Done.** 55 unit tests, plus `Trace.save` and `Trace.load`. Seeds are derived per index as `SeedSequence([root, i])` rather than by spawning, which makes prefix stability structural. Two defects surfaced by running real data rather than by the tests: R-hat reported 0.994 — perfect convergence — for parameters that never moved, because the `within == 0` guard missed the ~1e-37 cancellation residue of a chain held at 3e-3; and `agreement()` raised on a single-ensemble result, which is the M = 1 point of the §5.2 sweep.

Spec §8.6. Several chains per dataset, differing in seed and initialization,
**never exchanging information** — the distinction from tempering, where rungs
swap by design. Post-burn-in samples pool into one posterior; the spread
*between* ensembles is the convergence diagnostic.

**New file** — `ensemble.py`

```
EnsembleRunner(schedule, n_ensembles, n_steps, n_burn, init)
    .run(target, seeds)        -> EnsembleResult
EnsembleResult
    .traces                    per-ensemble Trace, burn-in already discarded
    .pooled                    one posterior over all ensembles
    .agreement()               between-ensemble diagnostics
```

**Design points**

- Seeds derive from one root seed by a pure function, so a run is reproducible
  from the config alone and ensemble *j* is reproducible without running the
  others — required for job arrays, where each task runs one ensemble.
- Initialization policy is explicit and recorded, not implicit. §5.6 measured
  that chains starting above and below the truth settle at different *k*, so
  "where ensembles start" is a scientific choice: all-from-below hides the
  multimodality, spread-across-*k* exposes it. The runner takes an `init`
  strategy and stores which was used.
- Burn-in is discarded before pooling, never after.
- **Ensembles run as separate tasks and merge afterwards.** Each task writes one
  trace to disk; a separate entry point loads any number of them and pools. The
  runner therefore never needs all ensembles in one process, and a failed task is
  re-run alone rather than restarting the set.
- **`n_ensembles` is a swept hyperparameter, not the constant 5.** The merge
  entry point accepts any count, and §5.2 below is the study that sets it.

**Trace persistence** — `trace.save(path)` / `Trace.load(path)`, compressed
`.npz`. Measured on a realistic 25,000-step trace at `k_max = 64`: 39.4 MB in
memory, **6.8 MB compressed** (`site_idx` is padded with −1 and the offsets are
mostly zero, so it compresses 5.8×). Twenty ensembles are 136 MB — small enough
that pooling stays a local operation.

**Unit tests** — `tests/unit/test_ensemble.py`

- seed derivation is pure: same root seed gives the same per-ensemble seeds
- ensemble *j* run alone equals ensemble *j* from a full run, step for step
- pooled sample count is `n_ensembles × (n_steps − n_burn)`
- ensembles do not share state: mutating one trace leaves the others intact
- `agreement()` on identical traces reports zero spread
- a burn-in exceeding `n_steps` raises rather than silently pooling nothing
- a trace survives `save` then `load` unchanged, including offsets and the
  per-step algorithm labels
- merging *M* saved traces equals pooling the same *M* in one process
- merging traces with different `k_max` or `n_exp` raises rather than padding

### 4c — configuration and Perlmutter submission

**Done.** 43 unit tests, all passing on the first run. Shipped as specified, plus `configs/nv_ensemble.toml` and a generated `scripts/submit_perlmutter.sh`. The callable logic lives in `config.py` with the scripts as thin argparse wrappers, so it is importable and testable rather than reachable only through a subprocess.

**New files** — `config.py`, `scripts/submit_perlmutter.sh`,
`scripts/run_ensemble.py`, `scripts/merge_ensembles.py`

A run is specified by a config file, not a Python literal: site table and filter
thresholds, experiment definitions, schedule, ensemble count, step budget,
burn-in, root seed, output path. `config.py` loads it and builds the objects.

**Unit tests** — `tests/unit/test_config.py`

- round trip: config → objects → config is the identity
- the same config and root seed produce a byte-identical trace (the property the
  whole cluster story depends on)
- array index → ensemble seed is a pure, total function
- an unknown algorithm name raises, naming the known ones
- a schedule whose blocks do not sum to the declared cycle length raises

SLURM itself is not unit-testable; the script is generated and its text asserted.
Nothing in `tests/` submits a job.

**Submission shape: one ensemble per task.** A job array runs `run_ensemble.py`
with `SLURM_ARRAY_TASK_ID` selecting the ensemble seed; each task writes its own
trace; a separate merge step pools them. This is what makes `n_ensembles` a
sweepable knob rather than a constant baked into the runner — adding ensembles is
extending the array, and the pooled posterior is recomputed offline without
re-running anything.

**Perlmutter settings**

| setting | value |
|---|---|
| user | `pabigail` |
| project account | `m5305` |
| working directory | `/global/cfs/cdirs/m5305/pabigail/claude-rewrite/` |

A job array over ensembles, one core per task:

```bash
#!/bin/bash
#SBATCH -A m5305
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --array=0-19
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -D /global/cfs/cdirs/m5305/pabigail/claude-rewrite
#SBATCH -J nsr-ensemble
#SBATCH -o logs/%A_%a.out

srun -n 1 python scripts/run_ensemble.py \
     --config "$CONFIG" --ensemble "$SLURM_ARRAY_TASK_ID" --out runs/"$SLURM_ARRAY_JOB_ID"
```

then, once the array completes:

```bash
python scripts/merge_ensembles.py runs/<jobid> --out runs/<jobid>/pooled
```

**Why `shared` and not `regular`.** §5 measures one 25,000-step ensemble at
**3.9 minutes on a single core**. A Perlmutter CPU node has 128 cores, so a
`regular` (exclusive-node) allocation would burn 127 of them idling per task.
The `shared` QOS bills only the cores requested, which is what makes
ensemble-per-task economical at this problem size. `debug` is the right QOS
while testing the array mechanics.

**Why CFS is fine for output.** NERSC guidance generally steers heavy I/O to
`$SCRATCH`, but the volume here is negligible: 6.8 MB compressed per ensemble,
136 MB for twenty. Writing straight to the project directory avoids a
stage-out step, and CFS is not purged.

Two things to confirm against the allocation before the first real submission:
that `m5305` is permitted to use `shared` QOS, and the wall-clock limit to
request — 30 minutes is roughly 7× the measured single-ensemble runtime, which
leaves room for the full table and a larger *k* without being wasteful.

### 4d — Wasserstein likelihood

**Done**, with two departures from what this section specifies below. The penalty is additive in log space, not the product form — that form's logarithm is undefined across almost the whole state space the sampler visits. And ζ and the scale were collapsed into a single `weight`, because only their product ever entered the likelihood. Calibrated in test-plan §5.7: on well-aligned data **no weight improves recovery**, so the calibrated value is zero; on data carrying a diagnosed timing offset it is the better criterion over a window of offsets.

Spec §7.1. Adds `likelihood/wasserstein.py` implementing

```
L_mod = (1 - zeta) * exp(-sum (d - f)^2 / 2 sigma^2) - zeta * W(f, d)
```

No sampler changes: it installs through the existing `Likelihood` ABC, and
tempering already operates on whatever is installed.

**Unit tests** — `tests/unit/test_wasserstein.py`

- **at ζ = 0 it equals `GaussianL2` to machine precision**, on random states —
  the reduction is exact, not approximate
- ζ = 0 with an exact match gives 0, matching `GaussianL2`
- W is symmetric and zero for identical signals
- ζ outside [0, 1] raises
- the replica axis is handled, one value per replica

### 4e — experimental data and T6

**Blocked on data files.** The seam is in place and demonstrated: a `file` data mode raises `NotImplementedError` naming this unit, which `notebooks/configured_runs.py` exercises.

Gated on data files. The reader is designed so **parsing is separable**: a
`read_experiment(path)` function producing an `ExperimentSet`, tested against a
small committed fixture, with everything downstream already covered by phases
1–3.

Sequence when the files arrive: commit one small representative file as a
fixture, write parser tests against it, implement, then run T6.

---

## 4. New ladder rungs

**Status: neither was written.** T7 and T8 do not exist in `tests/theory/`; the
ladder there still ends at T5. What was built instead, and what that leaves
uncovered:

| rung | specified below | what exists |
|---|---|---|
| **T7** | ensemble agreement detects trapped chains | **nothing equivalent.** `test_ensemble.py` checks that `agreement()` computes what it should on synthetic traces, but nothing runs real chains and shows the diagnostic firing on the §5.6 dimension split. The negative control this rung exists for is unexercised. |
| **T8** | the Wasserstein variant reduces exactly | **covered, in the wrong place.** `test_zero_weight_gives_an_identical_accepted_path` compares the whole accepted trajectory against `GaussianL2` under one seed — exactly what T8 asks — but as a fast unit test on the four-site table, not a statistical rung. |

So T8's content is present and T7's is not. The honest reading: the ensemble
machinery is unit-tested and has never been demonstrated to detect the failure
it was built to detect. `notebooks/ensembles_and_agreement.py` shows ten real
chains disagreeing with a modal-$k$ spread of 2 and R-hat at 1.84, which is the
substance of T7 — but a notebook is not an assertion, and nothing fails if that
stops working.

Writing T7 is the outstanding test debt of this phase.

Numbering is topical and continues `test-plan.md` §4. Build order differs from
numbering: T7 and T8 were to land before T6, which waits on data.

### T7 — ensemble agreement detects trapped chains

The rung must show the diagnostic **fires**, not merely that it is quiet when
things are fine. A convergence diagnostic that never reports non-convergence is
indistinguishable from one that is not computed.

The negative control already exists and is reproducible: §5.6 records that
chains reaching a given *k* from above and from below disagree, stably to 30,000
steps.

**Initialization is spread across *k*** — ensembles start both above and below
the truth. This is the deliberate choice to *expose* the multimodality of §5.6
rather than hide it, and it is also what Gelman–Rubin requires: the statistic is
only meaningful from overdispersed starts. What protocol to use on experimental
data is a separate, later question, to be settled empirically rather than
assumed here.

- **Positive:** with the same spread-across-*k* policy and enough ensembles,
  pooled modal *k* and $R_i$ agree between independent *sets* of ensembles.
- **Negative control:** individual ensembles trapped above and below **disagree**,
  and `agreement()` reports it. §5.6 already measured this split and found it
  stable to 30,000 steps, so the control is reproducible rather than contrived.
- Pooling improves or matches the best single-ensemble residual.

### 5.2 — how many ensembles · calibration study

**Not run.** Everything it needs exists — prefix-stable seeds make the sweep
free, since $M$ ensembles are the first $M$ of a set already computed, and
`notebooks/ensembles_and_agreement.py` demonstrates the mechanics over
$M = 1, 2, 3, 5, 7, 10$ on one seed. What is missing is the seed-to-seed spread
at each $M$, without which no production number can be set: the notebook's own
sweep is non-monotone, with the pooled mode reading 6, then 8, then 6 again.

Not a pass/fail rung. A recorded sweep of detection accuracy against ensemble
count, which is what sets the production number in place of the inherited 5.

- Sweep $M = 1, 2, 5, 10, 20$ on simulated data at the §5.1 settings.
- Record pooled $R$ per coupling band, modal-*k* agreement, and best residual.
- Report the point of diminishing return, with the seed-to-seed spread at each
  $M$ — a mean improvement smaller than that spread is not an improvement.

The answer is expected to depend on the initialization policy, so the sweep is
run under spread-across-*k* and re-run if the policy changes.

### T8 — the Wasserstein variant reduces exactly

*Both assertions exist, as unit tests rather than as a ladder rung, and ζ is now
a single `weight`.*

- At weight 0 the sampler's accepted path is **identical** to `GaussianL2`'s,
  given the same seed — not merely similar. Exact reduction is the whole safety
  argument for adding the term.
  (`test_zero_weight_gives_an_identical_accepted_path`)
- At weight > 0 some configuration pair changes rank. Without this the parameter
  is decorative. (`test_the_weight_changes_the_ranking_of_some_pair`)

### T6 — experimental data · unchanged

As written in `test-plan.md` §4: criterion A only, no detection claim. Its
thresholds are set when the data lands, by the standard procedure — run the
rung, record the metric, record what the metric reads with the mechanism
disabled, set the threshold between them with margin.

---

## 5. Cluster sizing — measured, not assumed

Measured on this repo, full 3557-site table, stratified 12-spin bath, hybrid
schedule (50 site RWMH / 25 RJMCMC / 100 PT at J = 8 / 25 λ), single core:

| quantity | measured |
|---|---|
| per step | 9.40 ms |
| 25,000-step chain | **3.9 min** |
| trace, 25,000 steps at `k_max=64` | **60 MB** |
| `NeighborIndex` build, 3557 sites | 0.1 s |

So the published recipe — 5 ensembles × 25,000 steps — is **under four minutes
of wall clock** if the ensembles run in parallel, and about twenty minutes
serially on one laptop core.

Perlmutter is therefore not justified by the published recipe alone. It becomes
necessary at one of three scales, and which one drives the design:

1. **Many datasets.** Ensembles × datasets is where a 128-core node pays off.
2. **The PyCCE backend (phase 5),** where a forward evaluation stops being
   microseconds.
3. **Much longer chains,** if the dimension multimodality of §5.6 turns out to
   need them — though it was stable to 30,000 steps, which is evidence that
   length is not the fix.

The same measurement shows the full-table identifiability problem persists at
production scale: modal *k* reached 34 against a true 12, consistent with the
mode-17-against-8 recorded in §5.6. Ensembles will agree with each other on a
wrong *k*, and T7 is the rung that has to distinguish agreement from correctness.

---

## 6. Decisions and open questions

### Resolved

- **Between-ensemble statistic.** Gelman–Rubin $\hat{R}$ is computed for the
  scalars that have a stable identity across trans-dimensional samples — $\lambda$,
  $n$, $\sigma_e$ per experiment, and $k$. It is **not** computed for anything
  per-spin, where label switching means there is no "spin 3" to take a variance
  of, nor for $R_i$, which is a summary of an ensemble rather than a per-sample
  quantity. Those use spread of modal *k* and maximum pairwise $|\Delta R_i|$.
- **$\Delta R_i$ bands** are the three of `test-plan.md` §5.1, on coupling
  magnitude $\sqrt{A_\parallel^2 + A_\perp^2}$: **5–25**, **25–100**, **100–750** kHz.
  Consistent with §5.3, agreement is **asserted only above 100 kHz**, where
  $R = 0.83$–$0.92$ is stable; the 25–100 kHz transition band is recorded but not
  asserted, being measured at 0.16–0.56; below 25 kHz nothing is asserted at all,
  since those spins are not identifiable at these settings.
- **Initialization:** spread across *k*, to expose the multimodality. The
  experimental-data protocol is deferred to a later empirical study.
- **Submission:** one ensemble per task, merged and pooled afterwards.
  `n_ensembles` is swept, not fixed at 5; §5.2 is the study that sets it.

### Resolved by measurement — $\hat{R}$ is reported, never gated

The §5.6 dimension split was run as six ensembles spread across *k*, at two
chain lengths:

| chain length | modal *k* per ensemble | pooled mode | $\hat{R}$ on *k* | $\hat{R}$ on $\lambda$ |
|---|---|---|---|---|
| 4,000 | [8, 6, 6, 8, 6, 6] | **6** ✓ | 2.39 | 2.12 |
| 20,000 | [8, 6, 6, 8, 6, 6] | **6** ✓ | 2.68 | 2.24 |

Five times the chain length and $\hat{R}$ **rises**, while the pooled mode is
correct at both. The per-ensemble modes are identical across lengths, so this is
not burn-in.

The cause is structural, not a tuning failure. $\hat{R}$ asks whether each chain
has explored the whole posterior, and here the answer is permanently no: a single
chain cannot cross the dimension barrier. That is the reason ensembles exist, not
a defect in the run. **A conventional $\hat{R} < 1.01$ gate would reject a run
whose pooled posterior is correct.**

Therefore: $\hat{R}$ is computed per parameter and reported, and **nothing is
asserted on it**. What carries assertions is the stability of the *pooled*
summary between independent sets of ensembles, and the spread of modal *k* —
which read 2 here and held at both chain lengths.

$\hat{R}$ retains value as a trend. Recorded per parameter, a regression that
breaks $\lambda$ mixing stays visible, and a value approaching 1 would itself be
a finding.

### Resolved — config format

**Authored in TOML, recorded in JSON.** `tomllib` is stdlib at 3.11+, so reading
costs no dependency; writing TOML is *not* stdlib and would add `tomli-w`.
PyYAML is present in the development environment only transitively, via
JupyterLab, so depending on it is a latent break.

The runner writes the **resolved** config as JSON beside the results — stdlib in
both directions, diffable, and it captures defaults that were filled in rather
than only what was typed. The round-trip test compares parsed dicts, so no
writer dependency is needed at all.

Python-as-config is rejected: it cannot be recorded alongside results, it
executes arbitrary code on a shared filesystem, and it defeats the
"same config and root seed give a byte-identical trace" test, because a config
that is code is not data.

### Resolved — plotting stays in `post/`, lazily imported

`matplotlib` is a **dev extra, not a runtime dependency**. A module-level import
in `post/plots.py` would break `import nuclear_spin_recovery.post` on any
non-dev install, including a compute node that needs `post/` to write summaries
and should never pull a plotting stack.

Dropping plots entirely would defeat the purpose of 4a — the phase 3 notebook
already reimplemented these figures for want of an importable version.

So: plots live in `post/plots.py`, `matplotlib` is imported **inside each
function**, and it is declared as an optional `[plot]` extra. Package import
stays light and a missing matplotlib fails with a clear message rather than an
`ImportError` at import time.

One rule keeps the split testable: plotting functions take arrays or a
`PosteriorSummary` plus an optional `ax`, return the `Axes`, and **never compute
a metric**. Computation stays in `metrics.py`, so plots are tested for structure
rather than pixels.

### Resolved — Perlmutter

Account `m5305`, user `pabigail`, working directory
`/global/cfs/cdirs/m5305/pabigail/claude-rewrite/`. Job array over ensembles on
the `shared` QOS, one core per task, merged afterwards. Details and the reasoning
for `shared` over `regular` are in §3, unit 4c.

### Resolved after the fact — the Wasserstein weight

Two decisions this document did not anticipate, both forced by measurement
after 4d was written:

- The penalty is **additive in log space**. The published product form's
  logarithm is undefined wherever it is negative, which it already is at
  $E = -50$ for $\zeta = 0.1$ while ordinary traces sit near $E = -3000$.
- ζ and the scale are **one parameter**. Only their product entered the
  likelihood, so the two were perfectly degenerate; collapsing them stops a
  user tuning a knob that cannot independently do anything.

Both are recorded in `model-specification.md` §11, entries 6–8, so a later
disagreement with published results can be traced.

### Open

Nothing blocking implementation. Four things outstanding:

1. **T7 is unwritten** (§4). The ensemble diagnostic has never been shown to
   detect the failure it exists for. This is test debt, not a design question.
2. **The ensemble-count study (§5.2) has not been run**, so the production
   ensemble count is still the inherited 5 rather than a measured number.
   Everything it needs exists.
3. **`m5305`'s entitlement to the `shared` QOS**, to confirm at first
   submission rather than now.
4. **The wall-clock request**, once the production table and *k* range are
   fixed. 30 minutes is roughly 7× the measured single-ensemble runtime.

---

## 7. What phase 4 actually cost, and what it caught

Four units, each scaffolded as failing tests and reviewed before implementation.
The pattern earned its keep in a specific way worth recording: **every defect
found in this phase was found by running real data, not by the tests that had
been approved for the purpose.**

| unit | found by the tests | found by running it |
|---|---|---|
| 4a | — | `false_absence` matching by exact membership, reporting FP = 0.224 where the truth is 0.0 |
| 4b | — | R-hat reporting 0.994 for frozen parameters; `agreement()` raising at M = 1 |
| 4c | — | — (43 tests, all passed first run) |
| 4d | — | the default weight making the penalty inert; the product form being unusable |

In two of those cases the test that should have caught it existed and passed on
a value that happened to hide the bug: `np.ones` has exactly zero variance where
`3e-3` does not, and a substring search for `sbatch` matches a docstring. The
lesson is narrow and repeatable — **a constant chosen for convenience can make a
guard untestable** — and it is the one thing from this phase most worth carrying
into phase 5.
