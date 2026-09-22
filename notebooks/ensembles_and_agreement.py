# %% [markdown]
# # Ensembles: independent chains, pooled after the fact
#
# The tempering tutorial showed a single chain crossing a barrier it could not
# cross alone. This one deals with the barrier that **no** single chain crosses:
# the model dimension.
#
# `docs/test-plan.md` §5.6 records the measurement. Chains that reach a given
# $k$ from above and from below settle at different values and stay there —
# stable at 8,000, 16,000 and 30,000 steps alike. That is not burn-in and it is
# not fixed by running longer. Birth–death moves alone do not mix across
# dimension, so the answer has to come from several chains rather than one.
#
# | object | role |
# |---|---|
# | `derive_seeds` | per-ensemble seeds from one root, reproducible and prefix-stable |
# | `spread_across_k` | the initialisation policy — start chains above *and* below |
# | `EnsembleRunner` | `.run_one(i, …)` for a task, `.run(…)` for everything |
# | `Trace.save` / `.load` | the process boundary a job array crosses |
# | `merge_traces`, `load_ensembles` | the merge half of one-ensemble-per-task |
# | `EnsembleResult.agreement()` | R̂, modal-$k$ spread, per-band $\Delta R_i$ |
#
# The theme: **ensembles are how you find out whether to believe a run**, and
# the diagnostic reports rather than judges.

# %%
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    RHAT_PARAMETERS,
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    DiscreteLatticeWalk,
    EnsembleResult,
    EnsembleRunner,
    Experiment,
    ExperimentSet,
    GaussianL2,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    Schedule,
    SiteTable,
    State,
    Step,
    StretchedExponential,
    Target,
    derive_seeds,
    load_ensembles,
    merge_traces,
    rhat,
    simulate_dataset,
    spread_across_k,
)
from nuclear_spin_recovery.post import summarize

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

# %% [markdown]
# ## 0. The same six-spin problem

# %%
DATA_NOISE, LIK_SIGMA, LAM = 0.002, 0.02, 3e-3
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
K_MAX, N_STEPS, N_BURN, N_ENSEMBLES = 32, 1200, 400, 10

full_table = SiteTable.from_ivady_file(REPO / "nv-2.txt", strong_thresh=750.0,
                                       weak_thresh=5.0)
keep = np.hypot(full_table.a_par, full_table.a_perp) >= 100.0
table = SiteTable(
    distance=full_table.distance[keep], positions=full_table.positions[keep],
    a_par=full_table.a_par[keep], a_perp=full_table.a_perp[keep],
    isotope=full_table.isotope[keep], gyro=full_table.gyro[keep],
)
model = AnalyticCCE1(StretchedExponential())


def make_state(sites, *, sigma=LIK_SIGMA):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=1, lam=np.array([[LAM]]),
        n_stretch=np.array([[1.0]]), sigma=np.array([[sigma]]), k_max=K_MAX,
    )


true_sites = np.sort(np.random.default_rng(7).choice(len(table), 6, replace=False))
blank = ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)])
data = simulate_dataset(make_state(true_sites, sigma=DATA_NOISE), blank, table,
                        model, sigma=DATA_NOISE, rng=np.random.default_rng(11))
target = Target(data, model, GaussianL2(), table)

walk = DiscreteLatticeWalk(NeighborIndex(table.positions, radius=6.0))
inner = Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)])
schedule = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 100),
    Step(ParallelTempering(inner, n_replicas=8), 250),
    Step(RWMH(ParameterBlock("sites"), walk), 50),
])
print(f"{len(table)} candidate sites, truth k = {len(true_sites)}")

# %% [markdown]
# ## 1. Seeds: reproducible, and stable under extension
#
# A run has to be reproducible from its configuration alone, and ensemble $j$
# has to be reproducible **without running the others** — that is what a job
# array needs, since each task runs one ensemble in its own process.

# %%
print(f"derive_seeds(2026, 4) = {derive_seeds(2026, 4).tolist()}")
print(f"same call again       = {derive_seeds(2026, 4).tolist()}")
print(f"a different root      = {derive_seeds(2027, 4).tolist()}")

# %% [markdown]
# One property matters more than the others and is easy to get wrong.
# **Extending the count must not renumber the ensembles already run.**
#
# §5.2 of the plan sweeps the ensemble count over 1, 2, 5, 10, 20 to find how
# many are needed. If adding ensembles reshuffled the earlier seeds, every point
# on that curve would come from a different set of chains, and the curve would
# measure nothing.

# %%
many = derive_seeds(2026, 20)
for m in (1, 2, 5, 10):
    ok = np.array_equal(derive_seeds(2026, m), many[:m])
    print(f"  first {m:2d} of 20 == derive_seeds(root, {m:2d}):  {ok}")

# %% [markdown]
# Seeds are derived per index — `SeedSequence([root, i])` — rather than by
# spawning a batch. `SeedSequence.spawn` happens to be prefix-stable today, but
# per-index derivation makes the guarantee structural instead of a property of
# an internal counter that a numpy release could change.

# %% [markdown]
# ## 2. Initialisation is a scientific choice, so it is recorded
#
# `spread_across_k` starts ensembles at different $k$, cycling through the
# values given. Chains therefore approach the truth from above **and** from
# below.
#
# This is deliberate: it *exposes* the dimension multimodality rather than
# hiding it. Starting every chain from below would produce agreement that means
# nothing. It is also what Gelman–Rubin requires — the statistic is only
# meaningful from overdispersed starts.
#
# What policy to use on experimental data is a separate question, to be settled
# empirically rather than assumed here.

# %%
init = spread_across_k(make_state, (3, 9))
rng = np.random.default_rng(0)
print("ensemble index -> starting k:",
      [int(init(rng, i).k[0]) for i in range(6)], " (truth 6)")

# %% [markdown]
# ## 3. Running: one ensemble per task
#
# `run_one(i, …)` is what a SLURM array task calls. `run(…)` does all of them in
# this process — convenient for a notebook, not what the cluster uses.

# %%
runner = EnsembleRunner(schedule, n_ensembles=N_ENSEMBLES, n_steps=N_STEPS,
                        n_burn=N_BURN, init=init, init_name="spread_across_k")

t0 = time.time()
result = runner.run(target, root_seed=2026)
elapsed = time.time() - t0
print(f"{N_ENSEMBLES} ensembles x {N_STEPS} steps in {elapsed:.1f} s "
      f"({elapsed / N_ENSEMBLES:.1f} s each)")
print(f"init policy recorded: {result.init_name!r}")
print(f"each trace keeps {len(result.traces[0])} of {N_STEPS} steps "
      f"({N_BURN} discarded)")

# %% [markdown]
# `run_one` returns the **full** trace, burn-in included. A task writes what it
# sampled; how much to discard is decided at pooling, against diagnostics a
# single task cannot see.

# %%
alone = runner.run_one(1, target, root_seed=2026)
print(f"run_one keeps all {len(alone)} steps")
print(f"ensemble 1 alone == ensemble 1 of the full run: "
      f"{np.array_equal(np.asarray(alone.k)[N_BURN:], np.asarray(result.traces[1].k))}")

# %% [markdown]
# That equality is the property the whole cluster design rests on.

# %% [markdown]
# ## 4. Crossing the process boundary
#
# Ensembles merge after the fact, so traces have to survive being written and
# read back — offsets, per-step algorithm labels and all.

# %%
with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    result.save(tmp)
    files = sorted(p.name for p in tmp.glob("*.npz"))
    size = sum(p.stat().st_size for p in tmp.glob("ensemble_*.npz"))
    print(f"wrote {files[:3]} ... ({len(files)} files)")
    print(f"{N_ENSEMBLES} traces of {len(result.traces[0])} steps: "
          f"{size / 1e6:.2f} MB on disk")

    back = EnsembleResult.load(tmp)
    same = all(np.array_equal(np.asarray(a.site_idx), np.asarray(b.site_idx))
               and list(a.algorithm) == list(b.algorithm)
               for a, b in zip(result.traces, back.traces, strict=True))
    print(f"round trip exact, labels included: {same}")
    print(f"merging from disk == pooling in process: "
          f"{np.array_equal(np.asarray(merge_traces(load_ensembles(tmp)).k), np.asarray(result.pooled.k))}")

# %% [markdown]
# Compression earns its place because the padding compresses: `site_idx` beyond
# $k$ is $-1$ and the offsets are mostly zero. A 25,000-step trace at
# `k_max = 64` goes from 39.4 MB in memory to 6.8 MB on disk, so twenty
# ensembles are 136 MB and pooling stays a local operation.

# %% [markdown]
# ## 5. The chains disagree, and the spread is the point
#
# Individual chains settle on different dimensions and stay there. Pooling them
# gives a posterior over $k$ rather than a verdict — and on this run its mode
# lands on the truth.
#
# Do not read that as "pooling fixes dimension." §7 sweeps the ensemble count on
# these same chains and the pooled mode flips between 6 and 8 as ensembles are
# added. What ensembles reliably give is the **disagreement**: a modal-$k$
# spread of 2 says the dimension is not settled, and that statement is worth
# more than either chain's answer.

# %%
modes = [int(np.bincount(np.asarray(t.k)).argmax()) for t in result.traces]
pooled = result.pooled
pooled_mode = int(np.bincount(np.asarray(pooled.k)).argmax())

print(f"per-ensemble modal k : {modes}")
print(f"truth                : {len(true_sites)}")
print(f"pooled modal k       : {pooled_mode}   over {len(pooled)} draws")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
for t in result.traces:
    ax1.plot(np.asarray(t.k), lw=0.7, alpha=0.6)
ax1.axhline(len(true_sites), color="crimson", ls="--", lw=1.2, label="truth")
ax1.set_xlabel("step after burn-in")
ax1.set_ylabel("$k$")
ax1.set_title(f"{N_ENSEMBLES} independent chains")
ax1.legend(fontsize=8)

bins = np.arange(min(modes) - 1.5, max(modes) + 2.5)
ax2.hist(np.asarray(pooled.k), bins=bins, color="steelblue", alpha=0.85,
         density=True)
ax2.axvline(len(true_sites), color="crimson", ls="--", lw=1.2, label="truth")
ax2.set_xlabel("$k$")
ax2.set_ylabel("pooled density")
ax2.set_title("pooled posterior on $k$")
ax2.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 6. `agreement()` — reports, never judges
#
# Three things come back: R̂ per parameter, the spread of modal $k$, and the
# largest pairwise $\Delta R_i$ per coupling band.

# %%
agreement = result.agreement()
print(f"R-hat, computed for {RHAT_PARAMETERS}:")
for name, value in agreement.rhat.items():
    shown = "nan (never moved)" if np.isnan(value) else f"{value:.2f}"
    print(f"    {name:10s} {shown}")
print(f"\nmodal k spread : {agreement.k_mode_spread}")
print(f"ensembles      : {agreement.n_ensembles}")

# %% [markdown]
# ### Why R̂ is reported and never gated
#
# R̂ compares the variance *between* chain means against the variance *within*
# chains. Converged chains disagree no more than each one wanders, so R̂ → 1;
# the conventional threshold is 1.01.
#
# On this problem R̂ is structurally above that, and **running longer makes it
# worse**. Measured on six ensembles spread across $k$: R̂ on $k$ was 2.39 at
# 4,000 steps and 2.68 at 20,000 — while the pooled mode was correct at both
# lengths, and the per-ensemble modes were identical.
#
# The reason is not a tuning failure. R̂ asks whether *each chain* explored the
# whole posterior, and here the answer is permanently no: a single chain cannot
# cross the dimension barrier. That is why ensembles exist, not a defect in the
# run. **A 1.01 gate would reject a run whose pooled posterior is right.**
#
# So `Agreement` carries no pass/fail field, and a test asserts it never grows
# one.

# %% [markdown]
# ### R̂ on a parameter that never moved
#
# This schedule samples sites and dimension, not $\lambda$ or $\sigma$. Those
# come back `nan` rather than a number — and the guard that produces the `nan`
# is scaled rather than an exact comparison against zero.
#
# A chain held at a constant $3\times10^{-3}$ has a `ddof=1` variance of about
# $10^{-37}$: cancellation residue, not movement. An `== 0` test lets it through
# and reports $\hat{R} = \sqrt{(N-1)/N} = 0.994$ — which reads as *perfect
# convergence* for a parameter that never moved. That is the worst failure mode
# available to a diagnostic.

# %%
for value in (1.0, 0.02, 3e-3, 311.0):
    print(f"  frozen at {value:>8}: R-hat = {rhat(np.full((4, 800), value))}")
noisy = 3e-3 + np.random.default_rng(0).normal(0.0, 1e-5, size=(6, 400))
print(f"  real variation at the same scale: R-hat = {rhat(noisy):.4f}")

# %% [markdown]
# ### Detection spread
#
# $\Delta R_i$ needs $R_i$, which needs a reference — so it takes one
# `PosteriorSummary` per ensemble, from `post.summarize`.

# %%
summaries = [summarize(t, data, table, model, reference=true_sites, burn=0,
                       stride=20, noise=DATA_NOISE) for t in result.traces]
per_ensemble_R = np.array([s.R_i.mean() for s in summaries])
with_detection = result.agreement(summaries=summaries)

print(f"mean R per ensemble: {np.round(per_ensemble_R, 2).tolist()}")
print(f"largest pairwise ΔR_i per band: "
      f"{np.round(with_detection.max_delta_R, 2).tolist()}")
print("\nThe first two bands are nan: every reference spin here is above")
print("100 kHz, so those bands hold no data. Missing, not zero.")

# %% [markdown]
# $\Delta R_i$ is taken per reference spin and then maximised within the band,
# not averaged. One spin that half the ensembles find and half do not is the
# disagreement worth seeing, and averaging would dilute it against spins every
# ensemble agrees on.

# %% [markdown]
# ## 7. How many ensembles?
#
# Five is what the published runs used. It is a hyperparameter, not a constant,
# and this is the sweep that sets it — §5.2 of the plan.
#
# Because the seeds are prefix-stable, the sweep costs **nothing extra**: $M$
# ensembles are the first $M$ of the ten already run, not a fresh set each time.

# %%
sweep = []
for M in (1, 2, 3, 5, 7, 10):
    subset = EnsembleResult(traces=result.traces[:M], seeds=result.seeds[:M],
                            init_name=result.init_name)
    s = summarize(subset.pooled, data, table, model, reference=true_sites,
                  burn=0, stride=20, noise=DATA_NOISE)
    sweep.append((M, float(s.R_i.mean()), s.k_mode, s.best_residual,
                  subset.agreement().k_mode_spread))

print(f"{'M':>3} {'pooled R':>9} {'mode k':>7} {'best σ':>8} {'k spread':>9}")
for M, R, kmode, best, spread in sweep:
    print(f"{M:3d} {R:9.3f} {kmode:7d} {best:8.2f} {spread:9d}")

# %%
M, R, kmode, best, spread = (np.array(x) for x in zip(*sweep, strict=True))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
ax1.plot(M, R, "o-", color="steelblue")
ax1.set_xlabel("ensembles pooled")
ax1.set_ylabel("pooled $R$")
ax1.set_title("detection against ensemble count")
ax1.set_ylim(0, 1.05)

ax2.plot(M, kmode, "o-", color="darkorange", label="pooled mode $k$")
ax2.axhline(len(true_sites), color="crimson", ls="--", lw=1.2, label="truth")
ax2.set_xlabel("ensembles pooled")
ax2.set_ylabel("$k$")
ax2.set_title("dimension against ensemble count")
ax2.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# The pooled mode is **not monotone in $M$** here: it reads 6 at one ensemble,
# 8 through the middle of the sweep, and 6 again at ten. Detection wanders in a
# band rather than climbing. Neither curve is converged at this chain length,
# and that is the honest state of the measurement.
#
# So read this as an illustration of the *method*, not as the answer. A single
# realisation at one seed cannot set a production number — the real study
# repeats the sweep across seeds and reports the seed-to-seed spread at each
# $M$, because a mean improvement smaller than that spread is not an
# improvement. Tempering's benefit taught that lesson already
# (`docs/test-plan.md` §5.6), and the non-monotonicity above is the same
# warning arriving early.

# %% [markdown]
# ## 8. On the cluster
#
# Ensembles never exchange information, so they parallelise trivially. One
# ensemble per task, merged afterwards:
#
# ```bash
# #SBATCH -A m5305
# #SBATCH -C cpu
# #SBATCH -q shared
# #SBATCH --array=0-19
# #SBATCH -n 1 -c 1
# #SBATCH -t 00:30:00
# #SBATCH -D /global/cfs/cdirs/m5305/pabigail/claude-rewrite
#
# srun -n 1 python scripts/run_ensemble.py \
#      --config "$CONFIG" --ensemble "$SLURM_ARRAY_TASK_ID" --out runs/"$SLURM_ARRAY_JOB_ID"
# ```
#
# then `merge_ensembles.py` pools whatever the array produced.
#
# `shared` rather than `regular`, because one 25,000-step ensemble is **3.9
# minutes on a single core** and a Perlmutter CPU node has 128 of them — an
# exclusive allocation would idle 127 per task. Adding ensembles is extending
# the array, and the pooled posterior is recomputed offline without re-running
# anything.
#
# Those scripts arrive with 4c; the runner API above is already shaped for them.

# %% [markdown]
# ## Where to go next
#
# What ensembles buy: a way to tell agreement from correctness. Individual
# chains here settle on different dimensions and stay there; the pooled
# posterior recovers the truth anyway; and R̂ reports the disagreement without
# being allowed to veto the result.
#
# Still ahead in phase 4: run configuration and Perlmutter submission (4c), the
# Wasserstein likelihood variant (4d), and the experimental-data rung (4e).
# `docs/phase-4-plan.md` has the build order.
