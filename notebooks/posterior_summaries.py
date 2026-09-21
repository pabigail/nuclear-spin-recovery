# %% [markdown]
# # Reading a posterior: detection, residual, and what not to measure
#
# The first three tutorials built a bath, sampled over it, and added the
# trans-dimensional and tempered machinery. All three ended by asking the same
# question — *did that work?* — and answered it with whatever arithmetic was to
# hand.
#
# That is the problem `post/` exists to fix. Every metric in specification §9.2
# used to live in a test fixture, and the phase 3 notebook grew a second copy
# because there was nothing importable. Two copies of a measurement is how the
# original error happened: detection computed against a single final state
# rather than over the posterior.
#
# | function | question it answers |
# |---|---|
# | `predictive_signals`, `residual_distribution` | **criterion A** — does the model reproduce the data? |
# | `detection_rate` | **criterion B** — does the posterior contain the true spins? |
# | `false_absence` | how stable is the modal configuration across the posterior? |
# | `summarize` | all of the above, from one `Trace` |
# | `post.plots` | the trajectory diagnostics of §9.2 |
#
# The theme throughout: **a posterior is a distribution, and every summary here
# is defined over it.** Judging a recovery by one configuration — the last state
# of a chain, or the modal sample — measures the wrong object.

# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    DiscreteLatticeWalk,
    Experiment,
    ExperimentSet,
    GaussianL2,
    HybridDriver,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    Schedule,
    SiteTable,
    State,
    Step,
    StretchedExponential,
    Target,
    Trace,
    simulate_dataset,
)
from nuclear_spin_recovery.post import (
    BANDS,
    band_index,
    by_band,
    couplings,
    detection_rate,
    matches,
    predictive_signals,
    residual_distribution,
    summarize,
)
from nuclear_spin_recovery.post import plots as postplots

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

# %% [markdown]
# ## 0. A posterior to read
#
# The same six-spin bath as the phase 3 tutorial, on the detectable subset of
# the table, sampled with the hybrid driver. Everything here is setup; the
# tutorial proper starts at §1.

# %%
DATA_NOISE, LIK_SIGMA, LAM = 0.002, 0.02, 3e-3
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
K_MAX, BURN = 32, 400

full_table = SiteTable.from_ivady_file(REPO / "nv-2.txt", strong_thresh=750.0,
                                       weak_thresh=5.0)
keep = np.hypot(full_table.a_par, full_table.a_perp) >= 100.0
table = SiteTable(
    distance=full_table.distance[keep], positions=full_table.positions[keep],
    a_par=full_table.a_par[keep], a_perp=full_table.a_perp[keep],
    isotope=full_table.isotope[keep], gyro=full_table.gyro[keep],
)
model = AnalyticCCE1(StretchedExponential())


def make_state(sites, *, sigma=LIK_SIGMA, tbl=None, k_max=K_MAX):
    tbl = table if tbl is None else tbl
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(tbl), n_exp=1, lam=np.array([[LAM]]),
        n_stretch=np.array([[1.0]]), sigma=np.array([[sigma]]), k_max=k_max,
    )


true_sites = np.sort(np.random.default_rng(7).choice(len(table), 6, replace=False))
blank = ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)])
data = simulate_dataset(make_state(true_sites, sigma=DATA_NOISE), blank, table,
                        model, sigma=DATA_NOISE, rng=np.random.default_rng(11))
target = Target(data, model, GaussianL2(), table)

walk = DiscreteLatticeWalk(NeighborIndex(table.positions, radius=6.0))
inner = Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)])
schedule = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 150),
    Step(ParallelTempering(inner, n_replicas=8), 400),
    Step(RWMH(ParameterBlock("sites"), walk), 100),
])

trace = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
t0 = time.time()
HybridDriver(schedule).run(
    make_state(np.random.default_rng(31).choice(len(table), 3, replace=False)),
    target, np.random.default_rng(23), n_total=1300, trace=trace)
print(f"{len(trace)} steps in {time.time() - t0:.1f} s, truth k = {len(true_sites)}")

# %% [markdown]
# ## 1. `summarize` — everything §9.2 defines, from one `Trace`

# %%
summary = summarize(trace, data, table, model, reference=true_sites,
                    burn=BURN, stride=10, noise=DATA_NOISE)

print(f"criterion A  best residual   : {summary.best_residual:6.2f} σ")
print(f"             median residual : {summary.median_residual:6.2f} σ")
print(f"criterion B  R_i per true spin: {np.round(summary.R_i, 2).tolist()}")
print(f"dimension    posterior mode   : {summary.k_mode}  "
      f"(discrepancy {summary.dimension_discrepancy(len(true_sites))})")
print(f"             false absence    : {summary.false_absence:.3f}")
print(f"predictive   shape            : {summary.predictive.shape}")

# %% [markdown]
# ### It takes a `Trace`, and refuses a `State`
#
# Not an accident of typing — an explicit check. A single configuration is the
# wrong object to summarise, and the refusal says so rather than failing later
# on a missing attribute.

# %%
try:
    summarize(make_state(true_sites), data, table, model, reference=true_sites)
except TypeError as exc:
    print("TypeError:", exc)

# %% [markdown]
# ## 2. Why the modal configuration is not the answer
#
# The most common single configuration in the posterior is a tempting thing to
# report. On this run it is missing a spin that **96% of posterior samples
# contain** — because "modal" picks one draw, and the posterior is a cloud.
#
# Whether that happens on any particular run is luck. That it *can* happen is
# the reason detection is defined over the posterior and not over one state.

# %%
post = trace.discard_burn_in(BURN)
modal_j = int(np.flatnonzero(post.k == summary.k_mode)[0])
modal = couplings(table, post.site_idx[modal_j, : summary.k_mode])

print(f"{'site':>6} {'|A| kHz':>9} {'R_i':>6}   in the modal configuration?")
for s, r in zip(true_sites, summary.R_i, strict=True):
    mag = np.hypot(table.a_par[s], table.a_perp[s])
    inmodal = matches((table.a_par[s], table.a_perp[s]), modal)
    print(f"{s:6d} {mag:9.1f} {r:6.2f}   {'yes' if inmodal else 'NO'}")

# %% [markdown]
# `false_absence` is that effect as a number: the fraction of posterior samples
# in which the modal configuration's own spins are missing.
#
# Two warnings come with it. It is **not** a false-positive rate in the
# classification sense, and must not be reported as one. And it has to match
# couplings within a tolerance, exactly as detection does — the version of this
# function that used exact set membership read **0.224** on a run whose every
# true spin had $R_i = 1.0$ and whose best residual was 0.92 σ. All of that
# 0.224 was symmetry-orbit rounding in the fourth decimal, and none of it was
# absence.

# %%
exact = np.mean([[c not in s for s in
                  [couplings(table, post.site_idx[j, : int(post.k[j])])
                   for j in range(len(post))]] for c in modal])
print(f"false absence, tolerance matched (correct): {summary.false_absence:.3f}")
print(f"false absence, exact set membership (bug) : {exact:.3f}")

# %% [markdown]
# ## 3. Detection is matched on couplings, never on site index
#
# The NV centre's $C_{3v}$ symmetry puts lattice sites into orbits that are
# spatially distinct and, to any coherence measurement, identical.
#
# `couplings` returns a set of $(A_\parallel, A_\perp)$ pairs — but rounding
# them does **not** merge an orbit. On the real table an orbit's members differ
# in the fourth decimal, an artefact of the DFT calculation rather than physics,
# so they survive as distinct entries. The merging happens later, at match time,
# through the tolerance.
#
# That is not a detail. It is exactly why every comparison in this module goes
# through `matches` rather than through set membership — and the bug in §2 was
# what happens when one of them forgets.

# %%
groups = table.symmetry_groups(tol=0.1)
orbit = np.flatnonzero(groups == groups[true_sites[0]])
print(f"site {true_sites[0]} lies in an orbit of {orbit.size}: {orbit.tolist()}")
print(f"A_par across the orbit: {np.round(table.a_par[orbit], 4).tolist()}")
print(f"as a rounded set        : {len(couplings(table, orbit))} distinct entries "
      f"-- rounding merges nothing")
print(f"pairwise within 0.1 kHz : "
      f"{all(matches((table.a_par[i], table.a_perp[i]), [(table.a_par[j], table.a_perp[j])]) for i in orbit for j in orbit)}"
      f"  <- the tolerance is what merges them")
print("\nmatching is within a tolerance, in kHz:")
print(f"  0.05 kHz away -> {matches((120.0, 45.0), [(120.05, 45.0)])}")
print(f"  0.50 kHz away -> {matches((120.0, 45.0), [(120.50, 45.0)])}")

# %% [markdown]
# `detection_rate` takes posterior samples and reference couplings, and returns
# $R_i$ — the fraction of samples containing each reference spin.

# %%
samples = [couplings(table, post.site_idx[j, : int(post.k[j])])
           for j in range(len(post))]
reference = list(zip(table.a_par[true_sites], table.a_perp[true_sites], strict=True))
print(f"R_i = {np.round(detection_rate(samples, reference), 3).tolist()}")
print(f"R   = {detection_rate(samples, reference).mean():.3f}")

# %% [markdown]
# ## 4. Bands, and why an empty one is `nan`
#
# Recovery quality depends strongly on coupling magnitude, so $R$ is reported
# by band. The three bands come from `docs/test-plan.md` §5.1, and only the
# last carries assertions: below 25 kHz spins are not identifiable at these
# settings, and 25–100 kHz is too noisy to threshold.

# %%
print(f"bands (kHz): {BANDS}")
for mag in (10.0, 30.0, 200.0, 2000.0):
    i = band_index(mag)
    print(f"  {mag:7.1f} kHz -> band {i}" + ("  (outside every band)" if i < 0 else ""))

# %%
print(f"R by band: {np.round(summary.by_band(), 3).tolist()}")
print("\nThe first two are nan, not 0.0, and that distinction matters: every")
print("reference spin here is above 100 kHz, so the low bands hold no data.")
print("Reporting 0.0 would read as 'detected nothing' instead of 'asked nothing'.")

# %% [markdown]
# The same guard applies to `by_band` directly.

# %%
values = np.array([1.0, 0.8])
magnitude = np.array([200.0, 300.0])
print(f"by_band with only high-band spins: {by_band(values, magnitude)}")

# %% [markdown]
# ## 5. The residual is a distribution, not a number
#
# Criterion A is the only one available on experimental data, where no ground
# truth exists. `predictive_signals` evaluates the forward model for every
# sampled configuration — offsets, $\lambda$ and stretch exponent all taken
# from the trace, because the posterior predictive integrates over everything
# that was sampled.
#
# Every draw becomes one replica of a single `State`, so the whole posterior is
# evaluated in one vectorised pass rather than a Python loop.

# %%
t0 = time.time()
predictive = predictive_signals(trace, data, table, model, stride=10)
residual = residual_distribution(data.data_all, predictive, DATA_NOISE)
print(f"{predictive.shape[0]} draws forward-modelled in {time.time() - t0:.2f} s")
print(f"residual: best {residual.min():.2f} σ, median {np.median(residual):.2f} σ, "
      f"worst {residual.max():.2f} σ")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

lo, hi = np.percentile(summary.predictive, [2.5, 97.5], axis=0)
ax1.fill_between(TAU * 1e3, lo, hi, color="steelblue", alpha=0.35,
                 label="95% of posterior draws")
ax1.plot(TAU * 1e3, data.data_all, lw=0.7, color="0.3", label="data")
ax1.set_xlabel(r"$\tau$ (µs)")
ax1.set_ylabel("coherence")
ax1.set_title("posterior predictive against the data")
ax1.legend(fontsize=8, loc="lower left")

ax2.hist(summary.residual, bins=30, color="steelblue", alpha=0.8)
ax2.axvline(1.0, color="crimson", ls="--", lw=1, label="1 σ")
ax2.axvline(summary.median_residual, color="k", ls="-", lw=1, label="median")
ax2.axvline(summary.best_residual, color="seagreen", ls="-", lw=1, label="best")
ax2.set_xlabel(r"RMS residual ($\sigma$)")
ax2.set_ylabel("draws")
ax2.set_title("criterion A is a distribution")
ax2.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ### `median_residual` and `best_residual` answer different questions
#
# The summary exposes both because using the wrong one silently inverts a
# comparison.
#
# A model with **extra sampled parameters** has a *higher* median residual than
# one holding those parameters at the prior mean — a typical draw sits away
# from that mean. Judged on the median, the richer model looks worse precisely
# because it is exploring. The question to ask is whether it can **reach** a fit
# the constrained model cannot, which is `best_residual`.
#
# Use the median to compare like against like; use the best to compare a model
# against one nested inside it.

# %%
ratio = summary.median_residual / summary.best_residual
print(f"median {summary.median_residual:6.2f} σ   "
      f"best {summary.best_residual:6.2f} σ   ratio {ratio:5.1f}×")
print("\nThe gap is the posterior's width, not a defect. Most draws carry a")
print("spurious spin or sit slightly off the optimum; a few reach it. Which")
print("number to quote depends entirely on the comparison being made -- and on")
print("a well-mixed run with no spurious dimension the two can coincide.")

# %% [markdown]
# ## 6. The diagnostic plots
#
# `post.plots` draws the trajectory diagnostics of §9.2. Each function takes
# arrays or a summary plus an optional `ax`, returns the `Axes`, and **computes
# no metric** — so the arithmetic stays testable without asserting on pixels.
#
# `matplotlib` is imported inside each function rather than at module scope: it
# is an optional `[plot]` extra, and a compute node running ensembles needs
# `post/` for summaries without a plotting stack.

# %%
steps = np.arange(0, len(trace), 10)
curve = residual_distribution(
    data.data_all,
    predictive_signals(trace, data, table, model, stride=10),
    DATA_NOISE)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
postplots.plot_residual(steps, curve, ax=axes[0, 0], burn=BURN, color="steelblue")
axes[0, 0].set_title("residual against step")
axes[0, 0].legend(fontsize=7)

postplots.plot_dimension(trace.k, ax=axes[0, 1], burn=BURN,
                         k_true=len(true_sites), color="darkorange")
axes[0, 1].set_title("dimension against step")
axes[0, 1].legend(fontsize=7)

postplots.plot_parameter(trace.log_prob, ax=axes[1, 0], burn=BURN,
                         label=r"$\log L$", color="0.4")
axes[1, 0].set_title("log-likelihood")

postplots.plot_detection_by_band(summary, ax=axes[1, 1])
axes[1, 1].set_title("detection by band (empty bands are gaps)")
plt.tight_layout()

# %% [markdown]
# ## 7. Bands across the full table
#
# The run above used the detectable subset, so every reference spin sat in one
# band. A stratified bath on the full 3557-site table populates all three, and
# shows what the band structure is for.
#
# This also reproduces a known result rather than a flattering one: on the full
# table the model dimension is **not** identifiable, and `dimension_discrepancy`
# reports it honestly.

# %%
mag_all = np.hypot(full_table.a_par, full_table.a_perp)
rng = np.random.default_rng(0)
strat = sorted({int(s) for lo, hi in BANDS
                for s in rng.choice(np.flatnonzero((mag_all >= lo) & (mag_all < hi)),
                                    4, replace=False)})
strat = np.array(strat)

data_f = simulate_dataset(make_state(strat, sigma=DATA_NOISE, tbl=full_table,
                                     k_max=64),
                          blank, full_table, model, sigma=DATA_NOISE,
                          rng=np.random.default_rng(3))
target_f = Target(data_f, model, GaussianL2(), full_table)
walk_f = DiscreteLatticeWalk(NeighborIndex(full_table.positions, radius=5.0))
sched_f = Schedule([
    Step(RWMH(ParameterBlock("sites"), walk_f), 60),
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=64)), 40),
])
trace_f = Trace(n_sites=len(full_table), k_max=64, n_exp=1)
t0 = time.time()
HybridDriver(sched_f).run(
    make_state(rng.choice(len(full_table), 8, replace=False), tbl=full_table,
               k_max=64),
    target_f, np.random.default_rng(5), n_total=2000, trace=trace_f)
print(f"{len(trace_f)} steps on the full table in {time.time() - t0:.1f} s")

# %%
summary_f = summarize(trace_f, data_f, full_table, model, reference=strat,
                      burn=1000, stride=25, noise=DATA_NOISE)
print(f"truth k = {len(strat)}, posterior mode = {summary_f.k_mode}, "
      f"discrepancy = {summary_f.dimension_discrepancy(len(strat))}")
print(f"best residual {summary_f.best_residual:.2f} σ")
print(f"\n{'band (kHz)':>14}  {'R':>6}  spins")
for (lo, hi), r in zip(BANDS, summary_f.by_band(), strict=True):
    n = int(((summary_f.magnitude >= lo) & (summary_f.magnitude < hi)).sum())
    print(f"{f'{lo:g}-{hi:g}':>14}  {r:6.2f}  {n}")

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6))
postplots.plot_detection_by_band(summary_f, ax=ax, color="darkorange")
ax.set_title("detection falls off with coupling magnitude")
plt.tight_layout()

# %% [markdown]
# The low band is where the papers' detection floor sits: below roughly 25 kHz
# a spin modulates the signal by less than the noise, and no amount of sampling
# recovers it. A future change that "improves" that number should be treated as
# suspicious rather than celebrated.

# %% [markdown]
# ## Where to go next
#
# What `post/` fixed: one implementation of every §9.2 metric, imported by the
# test ladder and by these notebooks alike, instead of three copies drifting
# apart. The extraction is verified by the ladder passing **with no threshold
# edited** — a threshold that had to move would have meant the extraction
# changed a measurement.
#
# It also turned up a defect the copies had hidden. `false_absence` matched by
# exact set membership while `detection_rate` matched within a tolerance, so
# symmetry-equivalent sites read as absences: FP = 0.224 on a run whose true
# value is 0.0. Two implementations of "is this spin in this sample" in one
# file, disagreeing.
#
# Still ahead in phase 4: `EnsembleRunner` (4b) for independent chains pooled
# after the fact, with $\hat{R}$ reported and never gated; run configuration and
# Perlmutter submission (4c); the Wasserstein likelihood variant (4d); and the
# experimental-data rung (4e). `docs/phase-4-plan.md` has the build order.
