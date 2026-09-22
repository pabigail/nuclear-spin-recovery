# %% [markdown]
# # Configured runs: one file, one seed, many machines
#
# Everything so far assembled a run in Python — build a table, wire a schedule,
# pick a seed, call the driver. That is fine in a notebook and wrong on a
# cluster, where the run happens in twenty separate processes over several
# hours and the only durable record of what was asked for is whatever was
# written down.
#
# `RunConfig` makes the file the specification.
#
# | piece | role |
# |---|---|
# | `RunConfig.from_toml` | read and validate a run specification |
# | `RunConfig.to_dict`, `.write_resolved` | record what was *actually* run, defaults included |
# | `.build_table`, `.build_experiments`, `.build_schedule`, `.build_runner` | turn the file into objects |
# | `run_ensemble(config, i, out)` | what one job-array task does |
# | `merge_run(config, out)` | pool the array afterwards |
# | `submission_script(config, …)` | generate the sbatch text |
#
# Two properties do the real work, and both are tested rather than asserted
# here: the same file and root seed give a **byte-identical** trace, and a
# merge over an incomplete array **refuses** rather than quietly publishing a
# posterior that is missing a third of its samples.

# %%
import json
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nuclear_spin_recovery import (
    DEFAULT_QOS,
    KNOWN_ALGORITHMS,
    PERLMUTTER_ACCOUNT,
    RunConfig,
    merge_run,
    run_ensemble,
    submission_script,
)

workdir = Path(tempfile.mkdtemp(prefix="nsr-config-"))
print(f"scratch directory: {workdir}")

# %% [markdown]
# ## 1. Authoring a run
#
# TOML, because `tomllib` is stdlib from 3.11 — reading costs no dependency.
# Writing TOML is *not* stdlib, which is why the **record** goes out as JSON
# rather than as a copy of this file. §2.
#
# This one is deliberately minimal: it names only what it must, and leans on
# defaults everywhere else.

# %%
MINIMAL = f"""
[table]
path = "{REPO / 'nv-2.txt'}"
strong_thresh = 750.0       # kHz -- drop sites where either component exceeds
weak_thresh = 5.0           # kHz -- drop sites where both fall below
coupling_cutoff = 100.0     # kHz -- the detectable subset, test-plan.md 5.6

[[experiment]]
n_pulses = 16
b_z = 311.0                 # G
tau_start = 3.2e-5          # ms
tau_stop = 8.0e-3
tau_count = 60

[data]
truth_k = 6
truth_seed = 7

[likelihood]
sigma = 0.02                # calibrated, not the data noise

[state]
lam = 3.0e-3                # ms
k_max = 32

[[schedule]]
algorithm = "rjmcmc"
n_steps = 20
k_max = 32

[[schedule]]
algorithm = "tempering"
n_steps = 30
n_replicas = 8
radius = 6.0

[ensemble]
n_ensembles = 3
n_steps = 150
n_burn = 50
root_seed = 2026
init_k = [3, 9]             # above and below, to expose the multimodality
"""

config_path = workdir / "run.toml"
config_path.write_text(MINIMAL)
config = RunConfig.from_toml(config_path)
print(f"{config.ensemble['n_ensembles']} ensembles x "
      f"{config.ensemble['n_steps']} steps, root seed "
      f"{config.ensemble['root_seed']}")

# %% [markdown]
# ## 2. What resolution adds
#
# The file above never mentions an isotope, a noise level, an output directory,
# a birth probability or an initialisation policy. The run uses all five.
#
# That gap is the entire argument for recording the **resolved** configuration
# rather than archiving the input file: six months later, the question is not
# what was typed but what was run.

# %%
typed = tomllib.loads(MINIMAL)
resolved = config.to_dict()

print("sections never written at all:", sorted(set(resolved) - set(typed)))
for name, section in resolved.items():
    if isinstance(section, dict):
        gained = sorted(set(section) - set(typed.get(name, {})))
        if gained:
            print(f"  [{name}] gained {gained}")
for i, (block, source) in enumerate(zip(resolved["schedule"], typed["schedule"],
                                        strict=True)):
    gained = sorted(set(block) - set(source))
    if gained:
        print(f"  [[schedule]] {i} ({block['algorithm']}) gained {gained}")

# %%
config.write_resolved(workdir / "resolved.json")
print((workdir / "resolved.json").read_text()[:420], "...")

# %% [markdown]
# The record round-trips, and is byte-stable: two writes of the same config
# produce identical files, so a diff between two runs shows only what actually
# differed.

# %%
config.write_resolved(workdir / "again.json")
print("byte-identical across writes:",
      (workdir / "resolved.json").read_bytes() == (workdir / "again.json").read_bytes())
with open(workdir / "resolved.json") as handle:
    reloaded = RunConfig.from_dict(json.load(handle))
print("JSON reloads into an equal config:", reloaded.to_dict() == config.to_dict())

# %% [markdown]
# ## 3. What the file refuses
#
# A configuration that is wrong should fail at load, loudly. The failure mode
# worth designing against is the quiet one: a typo'd key that gets ignored, a
# run that completes, and a record that looks correct while the sampler used a
# default nobody chose.

# %%
def try_load(label, mutate):
    bad = workdir / "bad.toml"
    bad.write_text(mutate(MINIMAL))
    try:
        RunConfig.from_toml(bad)
        print(f"{label:22s} -> accepted (!)")
    except (ValueError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"{label:22s} -> {type(exc).__name__}: {exc}")


try_load("typo in a key", lambda t: t.replace("n_ensembles", "n_ensembels"))
try_load("typo in a section", lambda t: t.replace("[state]", "[states]"))
try_load("unknown algorithm", lambda t: t.replace('"rjmcmc"', '"rjmcm"'))
try_load("missing required key", lambda t: t.replace("root_seed = 2026", ""))
try_load("burn-in eats the run", lambda t: t.replace("n_burn = 50", "n_burn = 200"))
try_load("empty block", lambda t: t.replace("n_steps = 20", "n_steps = 0"))

# %% [markdown]
# Each message names the offending key and, where it helps, the legal
# alternatives.

# %%
print("algorithms a schedule block may name:")
print(" ", ", ".join(KNOWN_ALGORITHMS))

# %% [markdown]
# ## 4. From file to objects

# %%
table = config.build_table()
expset = config.build_experiments()
schedule = config.build_schedule(table)
runner = config.build_runner(table)

print(f"table    : {len(table)} sites after both filters and the cutoff")
print(f"           |A| from {np.hypot(table.a_par, table.a_perp).min():.0f} kHz")
print(f"experiment: {expset.n_experiments} experiment, {expset.n_points} points")
print(f"schedule : {[s.algorithm.label for s in schedule]}, "
      f"{[s.n_steps for s in schedule]} steps")
print(f"runner   : {runner.n_ensembles} ensembles, {runner.n_steps} steps, "
      f"{runner.n_burn} burn-in, init {runner.init_name!r}")

# %% [markdown]
# The truth used to simulate the data is *in the config* — `truth_k` and
# `truth_seed` — so it is part of the provenance rather than living in a script
# somebody has since edited.
#
# Reading experimental data instead is `mode = "file"`, and says so plainly
# until unit 4e adds the reader.

# %%
from nuclear_spin_recovery import AnalyticCCE1, StretchedExponential

model = AnalyticCCE1(StretchedExponential())
data = config.build_data(table, model)
print(f"simulated data: {data.n_points} points, "
      f"noise {config.data['noise']}, truth k = {config.data['truth_k']}")

experimental = RunConfig.from_dict({**config.to_dict(),
                                    "data": {**config.data, "mode": "file"}})
try:
    experimental.build_data(table, model)
except NotImplementedError as exc:
    print(f"\nmode = 'file' -> NotImplementedError: {exc}")

# %% [markdown]
# ## 5. Determinism, which is what the cluster rests on
#
# A job array runs each ensemble in its own process, possibly on a different
# node, possibly days apart. Pooling them only means anything if ensemble $j$
# is the same chain wherever it ran.

# %%
first = run_ensemble(config, 0, out_dir=workdir / "det_a")
second = run_ensemble(config, 0, out_dir=workdir / "det_b")
identical = all(
    np.array_equal(np.asarray(getattr(first, f)), np.asarray(getattr(second, f)))
    for f in ("site_idx", "k", "lam", "dA_par", "log_prob"))
print(f"same config, same index, two processes' worth of work: "
      f"identical = {identical}")

other = RunConfig.from_dict({**config.to_dict(),
                             "ensemble": {**config.ensemble, "root_seed": 1}})
changed = run_ensemble(other, 0, out_dir=workdir / "det_c")
print(f"root_seed 2026 -> 1 changes the chain: "
      f"{not np.array_equal(np.asarray(first.k), np.asarray(changed.k))}")

# %% [markdown]
# ## 6. One ensemble per task
#
# `run_ensemble(config, i, out)` is the whole of what a SLURM array task does.
# Here the loop stands in for the scheduler.

# %%
run_dir = workdir / "runs" / "local"
t0 = time.time()
for index in range(config.ensemble["n_ensembles"]):
    trace = run_ensemble(config, index, out_dir=run_dir)
    print(f"  task {index}: {len(trace)} steps -> "
          f"ensemble_{index:03d}.npz")
print(f"{time.time() - t0:.1f} s total")
print("\nwritten:", sorted(p.name for p in run_dir.glob('*.npz')))

# %% [markdown]
# Each task writes the **full** trace, burn-in included. How much to discard is
# a pooling-time decision, made against diagnostics a single task cannot see.

# %% [markdown]
# ## 7. Merging — and refusing to
#
# `merge_run` discards burn-in, pools, and hands back an `EnsembleResult`.

# %%
result = merge_run(config, run_dir)
agreement = result.agreement()
print(f"pooled {len(result.traces)} ensembles, {len(result.pooled)} draws "
      f"({config.ensemble['n_steps']} - {config.ensemble['n_burn']} each)")
print(f"per-ensemble modal k: "
      f"{[int(np.bincount(np.asarray(t.k)).argmax()) for t in result.traces]} "
      f"(truth {config.data['truth_k']})")
print(f"k spread            : {agreement.k_mode_spread}")
print(f"R-hat on k          : {agreement.rhat['k']:.2f}")

# %% [markdown]
# Those modal $k$ values are wrong, and should be. Three chains of 150 steps is
# a demonstration of the plumbing, not a run: the production configuration in
# §10 asks for 25,000 steps apiece and a burn-in longer than this entire
# notebook's budget. What is being shown here is that the file, the tasks and
# the merge agree with one another — not that the sampler has converged.
#
# Now the part that matters on a cluster. A job array is not atomic: tasks time
# out, nodes drain, one of twenty simply fails. If `merge_run` pooled whatever
# happened to be on disk, the result would look completely healthy.

# %%
(run_dir / "ensemble_002.npz").unlink()
try:
    merge_run(config, run_dir)
except ValueError as exc:
    print(f"ValueError: {exc}")

# %%
partial = merge_run(config, run_dir, allow_partial=True)
print(f"\nwith allow_partial=True: pooled {len(partial.traces)} of "
      f"{config.ensemble['n_ensembles']}")

# %% [markdown]
# The escape hatch exists, but it has to be asked for by name. That is the
# whole design: the dangerous thing is possible and never the default.

# %% [markdown]
# ## 8. Submitting it
#
# `submission_script` generates the sbatch text. Nothing in the package submits
# anything — a test parses `config.py` and asserts it neither imports
# `subprocess` nor calls anything that could execute a scheduler.

# %%
sbatch = submission_script(config, config_path=config_path)
print(sbatch)

# %% [markdown]
# Three things in there are decisions rather than boilerplate.
#
# **`-q shared`, not `regular`.** One 25,000-step ensemble is 3.9 minutes on a
# single core, measured on the full 3557-site table. A Perlmutter CPU node has
# 128 cores, so an exclusive allocation would idle 127 of them per task.
#
# **`--array=0-N`, one ensemble per task.** Adding ensembles is extending the
# array; the pooled posterior is recomputed offline without re-running
# anything. That works only because the seeds are prefix-stable — the first $M$
# of twenty are exactly `derive_seeds(root, M)`.
#
# **`$SLURM_ARRAY_TASK_ID` is the ensemble index.** That one line is the entire
# bridge between the scheduler and the sampler.

# %%
print(f"account {PERLMUTTER_ACCOUNT}, QOS {DEFAULT_QOS}")
for override in ({}, {"qos": "debug", "time_limit": "00:05:00"}):
    text = submission_script(config, config_path=config_path, **override)
    quality = next(ln for ln in text.splitlines() if "-q " in ln)
    limit = next(ln for ln in text.splitlines() if "-t " in ln)
    label = str(override) if override else "defaults"
    print(f"  {label:44s} -> {quality.strip()}, {limit.strip()}")

# %% [markdown]
# ## 9. The scripts, run for real
#
# `scripts/run_ensemble.py` and `scripts/merge_ensembles.py` are thin argparse
# wrappers over the two functions above. Running them here as subprocesses is
# the closest a notebook gets to being a job array.

# %%
cli_dir = workdir / "runs" / "cli"
for index in range(config.ensemble["n_ensembles"]):
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_ensemble.py"),
         "--config", str(config_path), "--ensemble", str(index),
         "--out", str(cli_dir)],
        capture_output=True, text=True, check=True)
    print(done.stdout.strip())

# %%
done = subprocess.run(
    [sys.executable, str(REPO / "scripts" / "merge_ensembles.py"),
     "--config", str(config_path), "--out", str(cli_dir)],
    capture_output=True, text=True, check=True)
print(done.stdout)

# %% [markdown]
# Note what the task wrote alongside its trace.

# %%
print(sorted(p.name for p in cli_dir.iterdir()))

# %% [markdown]
# Every task writes `resolved.json`, not just the first. The bytes are
# identical, so the race is harmless — and a missing file means no task got far
# enough to start, which is worth knowing.

# %% [markdown]
# ## 10. The production configuration
#
# `configs/nv_ensemble.toml` is the same schema at full size: ten ensembles of
# 25,000 steps on the detectable table, `rjmcmc` + `tempering` at $J = 8$.
#
# It uses that pair rather than all three algorithms, because the comparison in
# `notebooks/algorithm_comparison.py` measured the pair at 0.92 σ against 10.61
# for all three — the third block competed for budget instead of adding a
# capability.

# %%
production = RunConfig.from_toml(REPO / "configs" / "nv_ensemble.toml")
print(f"ensembles  : {production.ensemble['n_ensembles']}")
print(f"steps      : {production.ensemble['n_steps']} "
      f"({production.ensemble['n_burn']} burn-in)")
print(f"schedule   : "
      f"{[b['algorithm'] for b in production.schedule]}")
print(f"init        : {production.ensemble['init_policy']} over "
      f"{production.ensemble['init_k']}")
print(f"\nestimated : {production.ensemble['n_steps'] * 9.4e-3 / 60:.1f} min "
      f"per ensemble at the measured 9.4 ms/step")

# %% [markdown]
# ## Where to go next
#
# What a config buys: a run that is reproducible from one file, a record that
# says what was actually used rather than what was typed, a load step that
# refuses typos instead of absorbing them, and a merge that refuses an
# incomplete array instead of publishing it.
#
# Still ahead in phase 4: the Wasserstein likelihood variant (4d), and the
# experimental-data reader with the T6 rung (4e) — which is the `mode = "file"`
# branch this notebook watched raise. `docs/phase-4-plan.md` has the build
# order.
