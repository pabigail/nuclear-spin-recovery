# Rewrite

This branch (`claude-rewrite`) is a ground-up reimplementation of the sampler
that produced the results in arXiv:2506.19259 and arXiv:2506.18802. The physics
is unchanged; the software architecture, the test suite, and the written
specification are new.

## Timeline

| Period | Work | Location |
|---|---|---|
| 2024-07 – 2025-11 | Original research implementation, 89 commits | `main` |
| 2025-12 – 2026-06 | Continued research branches (`joss_submission`, `oop_mcmc_rewrite`, `exp_for_chris`) | public repo |
| 2026-09-16 – 2026-09-18 | Rewrite, 20 commits | imported here |

The rewrite was carried out in a separate private repository over three days and
imported into this branch on 2026-09-18 with `--allow-unrelated-histories`, so
both lineages remain visible in `git log`.

Development proceeded in phases, each one scaffolded as failing tests before any
implementation: phase 1 (data layer and analytic forward model), phase 2 (RWMH,
proposals, neighbour index, trace), phase 3 (RJMCMC, parallel tempering, hybrid
driver, DFT constraint relaxation). Phases 4 (ensemble runner, SLURM, posterior
metrics) and 5 (PyCCE backend) are not yet started.

## What changed

**Package layout.** `nuclear_spin_recover/` (7 modules, ~1230 lines) became
`src/nuclear_spin_recovery/` (21 modules, ~1750 lines) under a `src/` layout;
`setup.py` became `pyproject.toml`.

**Abstraction boundaries.** The rewrite introduces abstract base classes at the
four points where the science is expected to change:

- `ForwardModel` — `AnalyticCCE1` today, a PyCCE backend later, without touching
  the sampler.
- `Likelihood` — `GaussianL2` today; the noise model is swappable.
- `Proposal` — returns `(value, log_ratio)` so each proposal owns its own
  asymmetry correction rather than the acceptance step knowing about it.
- `Algorithm` — `RWMH`, `RJMCMC`, and `ParallelTempering` are composed by
  `HybridDriver` as a systematic-scan cycle of `Step(algorithm, n_steps)`.

**State carries a replica axis.** Every array in `State` has a leading replica
dimension, so parallel tempering advances all rungs in one vectorised step and
collapses to the cold chain at the end.

**Written specification.** `docs/model-specification.md` (638 lines) states the
Hamiltonian, forward model, priors, and sampling scheme as mathematics rather
than code, and records five conventions that had to be resolved against
conflicting published sources. `docs/test-plan.md` (423 lines) records what each
test can and cannot demonstrate, and the calibration runs behind every threshold.

**Test suite.** 8 test files became 21 (4074 lines, 390 tests), split into unit
tests and a `theory/` ladder T0–T5 that exercises the sampler against simulated
data with known ground truth.

**Two defects found by the new tests.** The RJMCMC birth ratio omitted the
combinatorial prior term. A prior uniform over configurations is not uniform over
*k*, since there are C(n,k) configurations of size *k*; including that factor
makes (k+1)/n_free cancel the proposal ratio exactly. Without it the log ratio
sat near +5.9 — a factor of 365 favouring every birth — and *k* ran to `k_max`
regardless of the data. Separately, `Trace` did not record the per-spin hyperfine
offsets, so every DFT-relaxed run was scored as though its constraint had never
been relaxed.

**One convention corrected.** The decoherence envelope is placed inside the
half-sum, following Jung et al., npj Quantum Information 7, 41 (2021), Eq. 5. The
two published forms differ by up to 0.465 in coherence; the inside placement is
the one for which full dephasing sends the signal to 1/2.

## What stayed the same

- **The physics.** CCE-1 analytic coherence under CPMG-N dynamical decoupling, in
  the Taminiau form, with a stretched-exponential envelope.
- **The inference strategy.** Metropolis-within-Gibbs over parameter blocks,
  reversible-jump moves over the number of spins, parallel tempering for the
  multimodal landscape.
- **The hyperfine table.** `nv-2.txt` is the identical file (blob `c05cb7b`),
  moved from `nuclear_spin_recover/io_files/` to the repository root.
- **Unit conventions.** Hyperfine in kHz, τ in ms, B_z in G, positions in Å,
  chosen so that kHz × ms is dimensionless.
- **The published results.** Nothing in this rewrite revises the conclusions of
  either paper. The two defects above were introduced by the rewrite's own
  earlier commits and caught before use; they are not errors in the published
  analysis.

## Research impact

The rewrite has not yet produced new scientific results — it is three days old
and phase 4 is unstarted. What it changes is what can be checked.

The calibration runs recorded in `docs/test-plan.md` §5.6 measure several
properties of this inference problem that were previously assumed rather than
quantified:

- **Dimension is only identifiable on a restricted table.** Posterior mode *k* is
  17 on the full 3557-site table and exactly 8 on the 165-site detectable subset.
- **Dimension is multimodal, and birth–death alone does not mix across it.**
  Chains initialised from below settle at 8; from above, at 10 — stable at 8k,
  16k, and 30k steps alike. This is not burn-in.
- **Parallel tempering's benefit is real but seed-dependent.** Pooled best
  residual improves 2.72σ → 1.76σ, ranging from 4.80 → 2.00 on one seed to no
  improvement on another.
- **Swap rate sets the replica count.** A four-rung geometric ladder swaps zero
  times; six rungs swap at 0.058.
- **The papers' σ² = 0.1 does not transfer to a bare sampler.** At that value the
  chain accepts 79% and sits at 31σ residual; σ = 0.02 is optimal on residual,
  detection rate, and acceptance simultaneously.

The recovery problem is ill-posed — many spin configurations reproduce the same
coherence signal within noise. The test suite therefore scores two separate
things: whether the forward model reproduces the data (residual against noise,
the only criterion available experimentally), and whether the posterior contains
the spins used to simulate it (detection rate over posterior samples, available
in simulation only). Detection is matched on couplings within 0.1 kHz and never
on site index, since symmetry-equivalent sites are physically indistinguishable.

## Claude's role

The rewrite was written by Claude (Anthropic's Claude Code) working
interactively under direction, across roughly three days. The division of labour:

**Directed by the author.** The physics, the model, the conventions, the phase
order, and the design of the test ladder T1–T5. The specification was written
from the author's description of the model and from the two source papers, then
reviewed and corrected before any code was written.

**Written by Claude.** The specification document, the test suite, the
implementation, and the calibration scripts — each phase scaffolded as failing
tests first, reviewed, then implemented.

**Corrected by the author.** Several substantive errors. Claude initially placed
the decoherence envelope outside the half-sum, following the application paper's
Eq. 6; the author identified Jung et al. Eq. 5 as the correct form. Claude
initially measured detection against the sampler's final state rather than the
posterior; the author corrected this, and it became the success criterion
recorded in §9.1 of the specification. The author also specified that recovery be
judged by signal fit and posterior containment rather than by configuration
accuracy, which is what makes the ill-posedness tractable to test.

**Errors Claude made and caught.** Five test-design errors, all the same shape —
asking a test to demonstrate something its measurement could not support. Two
conflated identifiability with sampler correctness; one asserted on a band where
both methods saturate; one asserted per-seed on a seed-dependent effect; one
counted zero swaps by construction, because identical tempering rungs make an
accepted swap indistinguishable from a rejected one. Claude also misdiagnosed one
calibration failure and retracted it. These are recorded in `docs/test-plan.md`
rather than quietly fixed, because the last one carries a general lesson: a test
that passes and a test that fails can both be measuring nothing, and the only way
to tell is to check what the metric reads when the mechanism is disabled.
