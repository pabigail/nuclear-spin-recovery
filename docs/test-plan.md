# Test plan

How correctness is established for this package, and in what order.

This document is the companion to `model-specification.md`: the specification
says what the model *is*, this says how we find out whether the implementation
samples it. It exists because the obvious test — "did the sampler recover the
configuration we simulated?" — is the wrong question for this problem, and a
suite built around it would pass while the sampler targeted the wrong
distribution.

---

## 1. What a recovery test can and cannot show

The inverse problem is ill-posed. Many distinct spin configurations reproduce the
same coherence signal to within the noise, so a run that lands on a different
configuration has not failed, and a run that lands on the right one has not
necessarily done anything right. Two criteria replace configuration equality;
they are the ones set out in specification §9.1.

**Criterion A — does the forward model reproduce the data?** The distribution of
RMS residual over posterior samples, measured against the noise level actually
present in the data. Bounded on both sides. Above the noise means information in
the data has not been extracted; well below it means the model is fitting noise,
which becomes possible as soon as $k$ is free to grow. Needs no ground truth, so
this is the only criterion available on experimental data.

> **Two different sigmas.** The $\sigma_e$ appearing in the likelihood
> (specification §7.1) is a *sampling temperature* as much as a noise estimate,
> and criterion A compares the residual against the noise **the data actually
> has**. The two are set independently, and §5 measures how performance depends
> on the former.

**Criterion B — does the posterior contain the spins that generated the data?**
The detection rate $R_i$ over post-burn-in posterior samples, matched on
hyperfine couplings within 0.1 kHz and grouped by coupling magnitude. Requires
ground truth, so it exists only in simulation. This is why simulated studies
carry the burden of quantifying accuracy: an experimental run inherits its
credibility from simulations performed under matched conditions — same pulse
number, field, sampling and noise level.

Three rules follow, and every test in the suite obeys them.

1. **Assert on the posterior, never on a single state.** Not the final state of a
   chain, not the modal sample. A spin absent from the modal configuration may
   appear in most samples, and the detection rate is defined over samples.
2. **Never assert $R = 1$ overall.** Weakly coupled spins are genuinely
   unidentifiable at a given set of experimental settings. Demanding their
   recovery encodes a false expectation and makes the suite fail for reasons the
   implementation cannot fix. $R \to 1$ is asserted only within a coupling band
   calibrated from simulation.
3. **Separate correctness from mobility.** A spin can be missing from the
   posterior because the data cannot identify it, or because the sampler could
   not reach it. These need different fixes and must not share a test.

---

## 2. Calibration policy

Every numeric threshold in the suite is derived from a recorded calibration run,
never tuned upward until the test passes. Each threshold carries, in the test
file: the value, the conditions that produced it, and the **negative control** —
what the same metric reads when the mechanism under test is disabled.

A calibration run that fails criterion A is not a calibration of anything, and
its detection rates must not be recorded as a band. If the residual does not
reach the noise, the chain has not fit the data, and $R$ is then measuring the
sampler's mobility and hyperparameters rather than what the data can identify.
Check criterion A first, every time.

A threshold the negative control also passes is not a test. T0 sets the pattern:
the correct kernel gives $\chi^2$ $p = 0.12$, the kernel with its proposal ratio
removed gives $p = 0.0$, and the threshold sits at $10^{-3}$ between them.

Goodness-of-fit thresholds are deliberately strict — $10^{-3}$ rather than the
conventional $0.01$ — because these tests run at fixed seeds. A threshold near
the nominal significance level converts the expected rate of chance rejections
into flakiness, while the alternatives being detected drive $p$ to zero, so
tightening costs no power.

---

## 3. Shared harness

`tests/theory/conftest.py` provides one metrics helper used by every rung, so no
test reimplements a measure. It returns:

- $R_i$ per reference spin, and $R$ aggregated by coupling band;
- RMS residual quantiles across posterior samples, in units of $\sigma_e$;
- the posterior over $k$;
- the sample-averaged false-absence rate $FP$ of specification §9.2;
- a posterior-predictive envelope: signals predicted from sampled
  configurations, for comparison against the data.

Centralising this is not tidiness. The measurement error that prompted this
document — computing detection against a single final state rather than over the
posterior — happened because the metric was written inline at the point of use.

---

## 4. The ladder

Each rung adds exactly one mechanism. Earlier rungs stay green.

### T0 — sampler invariance  ·  phase 2, passing

Targets whose stationary distribution is known in closed form, so the chain can
be checked against it without reference to any recovery.

- RWMH on a Gaussian reproduces its mean and standard deviation, and passes a
  KS test on samples thinned by $3\tau_{\text{int}}$ measured from the chain.
- Under a flat target, sites are visited uniformly, by $\chi^2$.
- The enumerated single-spin transition matrix is symmetric — an exact
  algebraic check with no statistical threshold.

This rung is the reason phase 2 precedes the recovery ladder at all. An
implementation that treats the occupancy-constrained walk as symmetric converges
to a stationary law proportional to $|N_R(x)|$ rather than uniform, yet passes
every rung below while reporting wrong posterior widths and model probabilities.

*Negative control:* proposal ratio removed → stationary law
$\propto |N_R(x)|$, $\chi^2$ $p = 0.0$.

### T1 — continuous parameter  ·  phase 3

$\lambda$ varies; spins and $k$ held at truth. Continuous RWMH.

- The 95% credible interval contains the true $\lambda$.
- The posterior concentrates: its width is a small fraction of the prior's.
- Criterion A: residual reaches $\sigma$.

$\lambda$ is a single identifiable scalar, so a credible-interval assertion is
meaningful here in a way it is not for configurations.

### T2a — discrete walk, local start  ·  phase 3

Sites vary; $k$ and $\lambda$ held at truth. Each spin starts a few Å from its
true site. Discrete RWMH.

This rung isolates the **acceptance rule**: the truth is within reach, so a
failure implicates the accept/reject logic rather than the chain's ability to
travel.

- Criterion B: $R > 0.60$ above 100 kHz (§5.4).
- Criterion A: median residual below $5\,\sigma$.

### T2b — discrete walk, global start  ·  phase 3

Identical, but every spin starts at a random site.

This rung **measures** rather than asserts. It records $R$ by band as the
single-block mobility baseline, and asserts only criterion A. Its recorded
numbers are what T4 must beat.

### T3 — trans-dimensional moves  ·  phase 3

$k$ varies; sites and $\lambda$ held. RJMCMC with birth and death from a fixed
candidate pool, so only the dimension kernel is under test.

- The posterior mode of $k$ equals $k_{\text{true}}$, from starts both above and
  below the true value — birth and death fail differently.
- Criterion A **with a floor**: residual not far below $\sigma$. Once $k$ is
  free, more spins always fit better; this is the assertion that the prior
  carried by $\gamma$ is doing its job, and without it the rung passes while $k$
  drifts upward.

### T4 — parallel tempering  ·  phase 3

$\lambda$ and sites vary; $k$ held. PT with a mixed inner schedule — continuous
RWMH on $\lambda$ and discrete RWMH on sites within each replica.

Tempering's claim is that it escapes local minima, which is only testable
against a baseline:

- $R$ by band **strictly exceeds T2b's recorded numbers** on identical data.
- Swap acceptance per adjacent rung falls in a sane band.
- The returned chain is $\beta_0$, and hot replicas never reach the trace.

### T5 — full hybrid, with the DFT constraint relaxed  ·  phase 3

Everything varies, including per-spin hyperfine offsets.

Truth is generated with couplings perturbed roughly 1 kHz off the table values —
the regime the methods paper's robustness study covers. Recovering on-table
truth would pass trivially with the offsets pinned at zero and would prove
nothing.

- The relaxed model achieves **both** a lower residual and a higher $R$ than the
  hard-constrained model on identical data.

### T7 — ensemble agreement  ·  phase 4, passing

Ensembles exist because a single chain cannot cross the dimension barrier. The
rung asserts that the diagnostic built to notice that **fires** on chains known
to be trapped, and **falls silent** when the trans-dimensional block that traps
them is removed.

- Modal $k$ spread $\ge 1$ and $\hat{R}$ on $k$ above 1.5, from ensembles
  spread across $k$.
- Spread exactly 0 and $\hat{R}$ undefined with birth-death moves deleted.
- **No claim that pooling improves the estimate.** Measured and false; §5.8.

### T6 — experimental data  ·  phase 4

Criterion A needs no ground truth, so the real CPMG-8 and CPMG-16 measurements at
311 G can be fit directly.

- Residual reaches the measured noise level.
- **No detection claim is made**, because none is available. What licenses
  belief in an experimental recovery is T2a–T5 run at matched settings, not
  anything this rung can show.

---

## 5. Calibrated thresholds

### 5.1 Conditions

All numbers below come from simulated NV data at the settings the papers use as
default: $N = 16$ CP pulses, $B_z = 311$ G, 250 interpulse spacings to 8 µs,
additive Gaussian noise $0.002$, $\lambda = 3$ µs, site table filtered at
strong 750 / weak 5 kHz. Baths are **stratified** across coupling magnitude —
random draws from the filtered table are dominated by weak couplings and leave
the strong bands with too few spins to calibrate against.

These characterise the **phase 2 sampler**: a single discrete RWMH block, sites
only, $k$ and $\lambda$ held at truth, started a few Å off. That is exactly the
T2a configuration, so it is the right calibration for that rung — and it is *not*
transferable to T4 or T5, which must be calibrated separately once tempering and
trans-dimensional moves exist.

### 5.2 Dependence on the likelihood's sigma

Three stratified baths of 12 spins, local start, 10,000 steps, 4,000 discarded.

| $\sigma$ | residual / noise | acceptance | $R$ 5–25 kHz | $R$ 25–100 | $R$ 100–750 |
|---:|---:|---:|---:|---:|---:|
| 0.002 | 2.89 | 9.4% | 0.004 | 0.250 | 0.833 |
| 0.006 | 4.21 | 17.9% | 0.013 | 0.557 | 0.913 |
| **0.020** | **1.97** | **33.5%** | 0.010 | 0.272 | **0.917** |
| 0.060 | 6.08 | 47.2% | 0.012 | 0.160 | 0.750 |
| 0.316 | 31.24 | 79.4% | 0.017 | 0.027 | 0.230 |

**$\sigma = 0.02$** is the optimum on every axis simultaneously: lowest residual,
highest detection in the identifiable band, and an acceptance rate of 33.5% —
inside the classic 23–44% window for random-walk Metropolis. It is the value the
suite uses for T2a and T2b.

At $\sigma = 0.316$ the chain accepts 79% of proposals and wanders: the residual
reaches 31 times the noise and detection collapses in every band. The methods
paper's $\sigma^2 = 0.1$ should therefore not be carried over to a bare sampler —
that value is tuned for the full hybrid, where tempering supplies the exploration
and the cold chain can afford a sharper likelihood. Whether it is optimal there
is a question for the T4 calibration, not an assumption to inherit.

### 5.3 The identifiable band

Reading down the good-$\sigma$ rows:

- **Below 25 kHz**, $R \approx 0.01$ regardless of $\sigma$. These spins are not
  identifiable at these settings, consistent with the application paper's
  detection floor of 7.8 kHz and its finding that the 0–25 kHz regime stays
  ill-posed however much data is added. **The suite asserts nothing here**, and
  a future change that "improves" this number should be treated as suspicious
  rather than celebrated.
- **25–100 kHz** is the transition, and it is noisy: 0.16 to 0.56 across
  $\sigma$ values at $n = 12$ per cell. Too unstable to carry a threshold.
- **Above 100 kHz**, $R = 0.83$–$0.92$ across the usable $\sigma$ range. This is
  the band where an assertion is fair.

Note that $R$ does not reach 1 even here, with the truth a few Å away. That is a
mobility limit of the single-block walk, and it is the headroom T4 must
demonstrate it can close.

### 5.4 Thresholds

| rung | assertion | measured | threshold |
|---|---|---|---|
| T2a | $R$, above 100 kHz | 0.83–0.92 | $> 0.60$ |
| T2a | median residual | 1.97 σ | $< 5\,\sigma$ |
| T2b | median residual | — | $< 8\,\sigma$ |
| T2b | $R$ by band | — | recorded, not asserted |
| T4 | $R$ above 100 kHz | — | $>$ T2b's recorded value |

Thresholds sit well below the measured values deliberately. These are
regression guards against a sampler that stops working, not a record of the best
performance seen — a threshold pinned to the observed number fails on seed
changes and teaches the reader to loosen it.

*Negative control, T2a:* at $\sigma = 0.316$ the same rung gives $R = 0.23$
above 100 kHz and a residual of 31 σ, failing both thresholds. A configuration
that cannot fit the data cannot pass.

### 5.5 What is not yet calibrated

T6 has no thresholds because the machinery it tests does not exist yet. It is
calibrated when its phase lands, by the same procedure: run the rung's own
configuration, record the metric, record what the metric reads with the
mechanism disabled, and set the threshold between them with margin.

### 5.6 Phase 3 calibration

Each number below decided a test's shape, not just its threshold. Four of the
five decisions were forced by a measurement contradicting what the test
originally assumed.

#### Model dimension is only identifiable on a detectable candidate table

RJMCMC run against baths of 8 spins, varying the pool births may draw from:

| candidate table | sites | k_true | posterior mode of k | 5–95% |
|---|---:|---:|---:|---|
| full, weak cutoff 5 kHz | 3557 | 8 | 17 | [11, 24] |
| weak cutoff 25 kHz | 767 | 8 | 9 | [8, 11] |
| **weak cutoff 100 kHz** | **165** | **8** | **8** | **[8, 9]** |

Below the detection floor a spurious spin changes the likelihood by less than
the noise, so it is accepted about half the time and $k$ random-walks upward.
On the full table the posterior mode measures identifiability, not the sampler.
T3 therefore runs against a restricted table, where a spurious spin costs
something.

#### Dimension is multimodal, and birth–death alone does not mix across it

Same bath, started under- and over-specified:

| start | 8,000 steps | 16,000 | 30,000 |
|---|---:|---:|---:|
| below ($k_0 = 4$) | 8 | 8 | 8 |
| above ($k_0 = 14$) | 10 | 10 | 10 |

Stable at every chain length, so this is not burn-in. From above the chain
finds every true spin — $R = 1.0$ across the detectable band — but never sheds
the last two extras. T3 asserts mode equality from below only, and *documents*
the over-specified case rather than demanding it pass. Closing that gap is what
tempering over a trans-dimensional block would buy.

#### Tempering helps, but not on every seed

Single discrete block against a six-rung ladder, same data and start:

| seed | residual, single | tempered | $R_{25-100}$ single | tempered |
|---|---:|---:|---:|---:|
| 21 | 1.73 | 1.62 | 0.482 | 0.566 |
| 22 | **4.80** | **2.00** | 0.370 | 0.743 |
| 23 | 1.64 | 1.66 | 0.546 | 0.545 |
| **pooled** | **2.72** | **1.76** | **0.466** | **0.618** |

Seed 22 is the claim in miniature: a chain stuck at 4.8σ, rescued to 2.0σ. Seed
23 shows nothing. T4 therefore asserts on the mean across three seeds. A
single-seed strict comparison would be flaky, and choosing the seed that shows
the effect would be worse than flaky. The 100–750 kHz band is unusable for this
comparison — both methods saturate near 1.0, leaving no headroom.

#### Swap rate sets the replica count

Rungs advanced 300 steps so they differ, then 400 swap attempts:

| ladder | coldest β | swap rate |
|---|---:|---:|
| geometric $2^{-j}$, J = 4 | 0.125 | **0.000** |
| geometric $2^{-j}$, J = 6 | 0.031 | 0.058 |
| $0.7^{j}$, J = 6 | 0.168 | 0.062 |
| $0.85^{j}$, J = 8 | 0.321 | 0.048 |

Hence `N_REPLICAS = 6`: a four-rung geometric ladder never exchanges at all.
The rates are low because the swap draws a *random* pair rather than an
adjacent one, following the methods paper — only a third of draws on six rungs
are adjacent, and wider gaps are almost never accepted. Raising this is an
obvious future improvement, and one the spec would have to record as a
departure.

#### Relaxation is compared at its best, not at its median

Truth generated with couplings perturbed ~1 kHz off-table:

| configuration | residual |
|---|---:|
| truth sites, offsets pinned at 0 | 3.47 σ |
| truth sites, **true** offsets | **1.00 σ** |
| constrained run, from truth sites | 3.47 σ |
| relaxed run, from truth sites | **1.71 σ** |

Three corrections were needed to get these. **Start at the true sites:** this
rung isolates relaxation, and starting from a wrong configuration measures
relaxation and configuration search together — the extra freedom lets the chain
fit the data with wrong sites, which inverts the comparison (best residual 4.53 σ
relaxed against 3.46 σ constrained). That is a real effect and worth its own
rung one day, but it is not the one T5 asks about. **Equal site-step budgets:**
the
relaxed schedule spends four of every six steps on offsets, so an equal *total*
budget gives it a third as many site moves and it loses on configuration search
rather than on the mechanism under test — that confound read 19.70 σ.
**Comparison at the best sample, not the median:** the constrained model pins
the offsets at the prior mean, which is the best point estimate when the
likelihood barely constrains them, while the relaxed model samples them, so a
typical draw is worse by construction. Comparing medians penalises the richer
model for exploring and reads 7.12 σ against 3.51 σ. The question a nested
model should be asked is whether it can reach a fit the constraint forbids, at
equal posterior-sample counts.

#### A measurement gap the rung exposed

T5 could not be measured at all until `Trace` recorded the hyperfine offsets.
The metrics harness rebuilds each posterior sample from its site indices, which
zeroes the offsets, so every relaxed run was scored as if its constraint had
never been relaxed. The symptom was a test reading 4.68 σ while a
final-state diagnostic on the same configuration read 1.82 σ: the diagnostic
carried the offsets, the trace had discarded them. A posterior sample is not
reproducible from site indices alone.

#### Chain lengths

| constant | value | basis |
|---|---:|---|
| `CHAIN_STEPS` | 8,000 | single-block rungs; k-recovery stable from 8,000 to 30,000 |
| `LADDER_STEPS` | 6,000 | tempered rungs cost `N_REPLICAS` inner steps each |
| `N_REPLICAS` | 6 | J = 4 never swaps; J = 6 swaps at 0.058 |
| `BURN` | 3,000 | |

Ladder wall time is about 4 minutes, dominated by T4's pooled comparison at
~110 s. The working loop remains `pytest -m "not slow"` at about 1 s.

---

### 5.7 The Wasserstein penalty weight

**Calibrated, and the calibrated value is zero.** A negative result, recorded
at the same length as a positive one so that nobody repeats the sweep assuming
it was never run.

The penalty carries a single weight $w$, in log-likelihood per unit of
normalised transport. The published form writes a factor $\zeta$ and a scale,
but only their product enters — `(zeta, scale)` of `(0.2, 5000)`, `(1.0, 1000)`
and `(0.5, 2000)` returned bitwise identical values — so the two were collapsed
into $w$ before this sweep was run.

Conditions as §5.1, on the 165-site detectable table, $k_{\text{true}} = 6$,
RJMCMC + parallel tempering at $J = 6$, 2,000 steps with 1,000 discarded, three
seeds. Weight 0 is `GaussianL2` exactly and is the negative control.

| $w$ | best residual per seed | median | $R$ per seed | mode of $k$ | accept |
|---:|---|---:|---|---|---:|
| **0** | [7.49, **0.92**, **0.92**] | **0.92** | [0.83, 1.00, 1.00] | [6, 6, 6] | 11.8% |
| $10^2$ | [7.49, 0.92, 0.92] | 0.92 | [0.83, 1.00, 1.00] | [6, 6, 6] | 11.8% |
| $10^3$ | [7.49, 7.49, 0.92] | 7.49 | [0.83, 0.83, 1.00] | [6, 6, 6] | 11.5% |
| $10^4$ | [7.49, 0.92, 8.04] | 7.49 | [0.83, 1.00, 0.83] | [6, 6, 7] | 12.5% |
| $10^5$ | [11.20, 9.79, 9.33] | 9.79 | [0.67, 0.83, 0.83] | [8, 8, 10] | 10.4% |
| $10^6$ | [9.57, 29.42, 28.94] | 28.94 | [0.83, 0.17, 0.33] | [8, 8, 6] | 10.4% |

The penalty is inert, then harmful. There is no weight at which it improves
either criterion.

#### Where the penalty starts to act at all

Chains run against `GaussianL2` and against each weight under one seed, compared
step by step:

| $w$ | first step at which the chain differs |
|---:|---|
| $10^0$ | never — bitwise identical |
| $10^1$ | never — bitwise identical |
| $10^2$ | step 563 |
| $10^3$ | step 44 |
| $10^4$ | step 5 |

Below about $10$ the term cannot flip a single accept/reject decision, so the
sampler is `GaussianL2` with extra arithmetic. Above about $10^3$ it flips
decisions immediately and the recovery degrades. The window in which it acts
without harming is narrow and, on these seeds, empty of benefit.

#### Why, given that the distance does discriminate

$\widehat{W}$ is not blind. Measured on the same data: $6\times10^{-5}$ at the
truth against $3.2\times10^{-2}$ at a random three-spin start — a factor of 517,
and in the right direction.

The residual discriminates far harder. $\log L$ runs from $-1$ at the truth to
$-3500$ at a random draw, a factor of thousands, and it is already sharpest
exactly where $\widehat{W}$ is flattest. So at a weight small enough to leave the
residual in charge the penalty contributes nothing, and at a weight large enough
to matter the sampler begins optimising $\widehat{W}$ instead — whose landscape
is flat enough that $k$ drifts upward (mode 8 to 10 against a true 6) and
detection falls away.

#### The regime the penalty was proposed for

The sweep above varies which spins are present. The penalty was proposed for a
different failure: features *displaced* rather than absent. That regime was
measured separately.

Data is simulated from the true bath and then recorded on a $\tau$ grid offset
by a whole number of sampling intervals — correct physics, systematic timing
error. The correct configuration is then scored against 400 random six-spin
candidates, and each criterion is asked how often a wrong candidate beats the
right one:

| offset | wrong candidate beats correct, least squares | …under $\widehat{W}$ |
|---:|---:|---:|
| 0 | 0.0% | 0.0% |
| 1 point (0.032 µs) | 4.8% | **0.2%** |
| 2 points (0.064 µs) | 35.0% | **12.8%** |
| 3 points (0.096 µs) | 65.0% | **24.2%** |
| 5 points (0.160 µs) | 70.0% | **44.8%** |
| 8 points (0.256 µs) | **20.8%** | 67.0% |

**The criteria cross over.** For offsets up to a few sampling intervals the
transport distance is markedly more robust — at three points it is misled a
quarter of the time against two thirds for least squares. At eight points the
ordering reverses: the correct signal has by then been transported far enough
that $\widehat{W}$ ranks it behind many random candidates, while least squares
recovers because a large offset degrades the random candidates too.

So the penalty does what it was proposed to do, in a window. Neither criterion
dominates the other, and the window's width is a property of the sampling
interval and the modulation, not a universal constant.

#### What this does and does not license

It does **not** show the penalty is useless. On displaced data it is
substantially the better criterion over a range of offsets, and the calibration
sweep simply does not contain that failure mode: those baths differ by which
spins are present, and both criteria rank them identically there.

It does show that **the default must stay $w = 0$**. On well-aligned data
the penalty is inert below $w \approx 10$ and harmful above $10^3$, with
no window of benefit. A user turning it on is departing from a calibrated
setting, and should be doing so because a timing offset has been *diagnosed* —
not as a general improvement. Having diagnosed one, they need their own sweep:
the useful weight depends on the offset, which is what the crossover shows.

---

### 5.8 T7 — ensemble agreement

Conditions as §5.1 on the 165-site detectable table, $k_{\text{true}} = 6$,
eight ensembles of 1,200 steps with 400 discarded, initialised
`spread_across_k` over $(3, 9)$ so chains approach from above and below.

**The diagnostic fires,** reproducibly across three root seeds:

| root seed | modal $k$ spread | $\hat{R}$ on $k$ |
|---:|---:|---:|
| 2026 | 2 | 2.39 |
| 11 | 2 | 2.34 |
| 404 | 2 | 1.94 |

Against a conventional pass mark of 1.01. Thresholds set at spread $\ge 1$ and
$\hat{R} > 1.5$, below the measured band with margin.

**And falls silent when the mechanism is removed.** With the trans-dimensional
block deleted, $k$ cannot vary: eight ensembles, modal $k = 6$ for every one of
them, spread 0, $\hat{R}$ undefined. That control is what makes the row above
mean something — a statistic that always reported disagreement would pass the
positive test too.

#### Two assertions from the plan that were dropped

`phase-4-plan.md` §4 specifies three checks for this rung. Two do not survive
contact with the definitions.

**"Pooling improves or matches the best single-ensemble residual" is a
tautology.** The pooled trace is the concatenation of the ensemble traces, so
its minimum residual is identically the smallest of theirs. It cannot fail. The
same holds for pooled $R$, which at equal sample counts is exactly the mean of
the per-ensemble values.

**"Pooling recovers the dimension" is false.** It was the obvious non-trivial
replacement, and measuring it refuted it:

| root seed | pooled $\lvert\text{mode}(k) - k_{\text{true}}\rvert$ | median over ensembles |
|---:|---:|---:|
| 2026 | **2** | 1.0 |
| 11 | **1** | 0.5 |
| 404 | 0 | 0.5 |

Pooling makes the modal dimension *worse* on two of three seeds. The pooled mode
is the mode of a mixture, and a mixture of ensembles that mostly over-count
produces an over-counting mode; nothing makes it converge on the truth.

§5.9 refines this: the measurement above is at $M = 8$, and pooling *does*
recover the dimension at $M = 20$, exactly, in six sets of six. The claim that
survives is narrower than "pooling fixes $k$" — it takes more ensembles than
anyone had tried.

This matters beyond the rung, because "run ensembles and pool them" is the
recommendation of spec §8.6. What pooling reliably provides is the **spread** —
the statement that the dimension is not settled — not a better point estimate of
it. The ensembles notebook says the same thing, and its own $M$-sweep flips
between 6 and 8 as ensembles are added.

---

### 5.9 How many ensembles

`phase-4-plan.md` §5.2 frames this as "which $M$ gives the best detection". That
question has no answer, and noticing why is most of the study: pooled $R$ is, at
equal sample counts, exactly the **mean** of the per-ensemble values, so its
expectation does not improve with $M$ at all. What improves is its variance.

The question actually put, therefore: **how many ensembles before the pooled
answer stops depending on which set you happened to run?**

Six independent sets of 20 ensembles, 1,000 steps with 400 discarded,
`spread_across_k` over $(3, 9)$, conditions otherwise as §5.1. Each $M$ row is a
*prefix* of the same 20 rather than a fresh draw — what prefix-stable seeds are
for. 455 s total.

| $M$ | pooled $R$, mean ± spread across sets | $\lvert\text{mode}(k)-6\rvert$ | best residual | modal-$k$ spread |
|---:|---|---|---:|---:|
| 1 | 0.857 ± 0.116 | 0.67 ± 0.75 | 7.76 | 0.0 |
| 2 | 0.831 ± 0.070 | 0.83 ± 0.69 | 4.48 | 0.8 |
| 3 | 0.851 ± 0.053 | 0.67 ± 0.75 | **0.92** | 0.8 |
| 5 | 0.857 ± 0.049 | 0.67 ± 0.75 | 0.92 | 1.5 |
| 10 | 0.868 ± 0.034 | 0.67 ± 0.47 | 0.92 | 1.8 |
| **20** | 0.879 ± **0.017** | **0.00 ± 0.00** | 0.92 | 2.0 |

**The mean $R$ barely moves** — 0.857 to 0.879 across a twentyfold change in
$M$ — exactly as the definition predicts.

**The spread falls as $1/\sqrt{M}$.** A log-log fit gives an exponent of
$-0.59$ against the $-0.5$ expected of independent samples; with six sets the
standard deviations are themselves noisy, so that is agreement, not a
discrepancy. It is also a check on the machinery: ensembles that shared state
would fall off more slowly.

**Three ensembles suffice for criterion A.** The best residual reaches 0.92 σ at
$M = 3$ and does not improve after. One ensemble finding the good mode is enough
for a fit; the rest buy precision on detection.

**Dimension needs twenty.** $\lvert\text{mode}(k)-6\rvert$ sits at 0.67 from
$M = 1$ through $M = 10$ and reaches **0.00 in all six sets** at $M = 20$. This
refines §5.8 rather than contradicting it: that measurement was at $M = 8$,
where pooling was indeed no better than a typical single ensemble. Pooling does
eventually recover the dimension — just not at the ensemble counts anyone had
tried.

*The modal-$k$ spread column is not comparable across rows.* It grows with $M$
because more ensembles give more chances of an outlier, not because agreement
worsens.

#### The recommendation

| if what matters is | use | what it costs |
|---|---|---|
| signal fit alone (criterion A) | $M = 3$ | best residual saturated |
| detection rate (criterion B) | $M = 10$ | $R$ to ±0.034 |
| the number of spins | $M = 20$ | modal $k$ exact in 6/6 sets |

The published $M = 5$ gives $R$ to ±0.049 and the modal dimension wrong in two
thirds of sets. It is defensible for a fit and not for a spin count.

**This study is one bath at one set of settings.** The $1/\sqrt{M}$ scaling
should transfer; the $M = 20$ threshold for dimension is a property of how
multimodal *this* posterior is, and a harder bath will need more.

---

## 6. Test layout

| location | contents | speed |
|---|---|---|
| `tests/unit/` | deterministic units: loaders, array invariants, acceptance algebra, proposal ratios | sub-second |
| `tests/theory/` | statistical rungs T0–T6, marked `slow`, fixed seeds | tens of seconds |

The working loop is `pytest -m "not slow"`. The full suite runs before every
commit.
