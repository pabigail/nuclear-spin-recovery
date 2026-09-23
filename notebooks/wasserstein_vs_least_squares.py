# %% [markdown]
# # Two ways to be wrong: transport distance against least squares
#
# Every likelihood in this package so far has scored a configuration one way:
# the sum of squared differences between the predicted and measured signal,
# point by point. That criterion has a blind spot it cannot see past — it
# compares $\tau_j$ to $\tau_j$ and nothing else, so a modulation dip displaced
# by one sampling interval is penalised exactly as hard as one that never
# appears.
#
# `WassersteinL2` adds an optimal-transport term for that case. This notebook
# measures what it buys, and the answer has two halves:
#
# | regime | which criterion is better |
# |---|---|
# | configurations differ by **which spins are present** | least squares; the penalty is inert, then harmful |
# | data carries a **systematic timing offset** | transport distance, over a window of offsets |
#
# The default stays $\zeta = 0$. What follows is why, and when to depart from it.

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
    WassersteinL2,
    signal_measure,
    simulate_dataset,
    wasserstein_signal_distance,
)
from nuclear_spin_recovery.post import summarize

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

# %%
DATA_NOISE, LIK_SIGMA, LAM, K_MAX = 0.002, 0.02, 3e-3, 32
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
DT = TAU[1] - TAU[0]

full = SiteTable.from_ivady_file(REPO / "nv-2.txt", strong_thresh=750.0,
                                 weak_thresh=5.0)
keep = np.hypot(full.a_par, full.a_perp) >= 100.0
table = SiteTable(distance=full.distance[keep], positions=full.positions[keep],
                  a_par=full.a_par[keep], a_perp=full.a_perp[keep],
                  isotope=full.isotope[keep], gyro=full.gyro[keep])
model = AnalyticCCE1(StretchedExponential())


def make_state(sites, sigma=LIK_SIGMA):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)), n_sites=len(table),
        n_exp=1, lam=np.array([[LAM]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[sigma]]), k_max=K_MAX)


def signal(sites, tau=TAU):
    """The coherence a configuration predicts, on any tau grid."""
    expset = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])
    return model.coherence(make_state(sites), expset, table)[0]


true_sites = np.sort(np.random.default_rng(7).choice(len(table), 6, replace=False))
blank = ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)])
data = simulate_dataset(make_state(true_sites, DATA_NOISE), blank, table, model,
                        sigma=DATA_NOISE, rng=np.random.default_rng(11))
obs = data.data_all
print(f"{len(table)} candidate sites, truth k = {len(true_sites)}, "
      f"sampling interval {DT * 1e3:.3f} µs")


def lsq(pred, observed=None):
    """The least-squares log-likelihood, in the sampler's units."""
    observed = obs if observed is None else observed
    return float(-0.5 * np.sum(((observed - pred) / LIK_SIGMA) ** 2))


# %% [markdown]
# ## 1. What each criterion looks at
#
# Least squares reads the signal vertically: the gap between the two curves at
# each $\tau$, squared and summed. Transport reads it horizontally: the dip
# depth $1-f$ becomes a mass distribution over $\tau$, and the cost is how far
# that mass has to move.

# %%
displaced = signal(true_sites, TAU + 3 * DT)     # correct physics, shifted data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
window = slice(0, 70)

ax1.plot(TAU[window] * 1e3, obs[window], lw=1.1, color="0.35", label="data")
ax1.plot(TAU[window] * 1e3, displaced[window], lw=1.1, color="crimson",
         label="a prediction, displaced by 3 points")
ax1.vlines(TAU[window] * 1e3, obs[window], displaced[window], color="steelblue",
           lw=0.6, alpha=0.6)
ax1.set_ylabel("coherence")
ax1.set_title("least squares measures the vertical gaps")
ax1.legend(fontsize=8, loc="lower left")

ax2.fill_between(TAU[window] * 1e3, 0, signal_measure(obs)[window], alpha=0.45,
                 color="0.5", label=r"data as mass: $1-d$")
ax2.fill_between(TAU[window] * 1e3, 0, signal_measure(displaced)[window],
                 alpha=0.45, color="crimson", label=r"prediction as mass: $1-f$")
ax2.set_xlabel(r"$\tau$ (µs)")
ax2.set_ylabel("mass")
ax2.set_title("transport measures how far the mass must slide along $\\tau$")
ax2.legend(fontsize=8, loc="upper right")
plt.tight_layout()

# %%
print(f"least squares : {lsq(displaced):10.1f}")
print(f"W-hat         : {wasserstein_signal_distance(displaced, obs, TAU):10.5f}")
print("\nfor reference, the correct undisplaced prediction:")
print(f"least squares : {lsq(signal(true_sites)):10.1f}")
print(f"W-hat         : "
      f"{wasserstein_signal_distance(signal(true_sites), obs, TAU):10.5f}")

# %% [markdown]
# ## 2. On well-aligned data, the penalty never helps
#
# The calibration sweep of `docs/test-plan.md` §5.7, reproduced at reduced
# length. Only the product $\zeta s$ enters the likelihood — `zeta=0.2,
# scale=5000` and `zeta=1.0, scale=1000` are bitwise identical — so the sweep
# runs over that product. Weight 0 **is** `GaussianL2`, exactly.

# %%
for pair in ((0.2, 5000.0), (1.0, 1000.0), (0.5, 2000.0)):
    value = WassersteinL2(zeta=pair[0], scale=pair[1]).log_prob(
        make_state(true_sites), data, model, table)[0]
    print(f"  zeta={pair[0]:<4} scale={pair[1]:<7.0f} -> {value:.9f}")
print("\nTwo parameters, one degree of freedom.")

# %%
walk = DiscreteLatticeWalk(NeighborIndex(table.positions, radius=6.0))
schedule = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 50),
    Step(ParallelTempering(
        Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)]),
        n_replicas=6), 50),
])
start = np.sort(np.random.default_rng(31).choice(len(table), 3, replace=False))
WEIGHTS = (0.0, 1e2, 1e3, 1e4, 1e5)
SEEDS = (17, 23)

t0 = time.time()
sweep = []
for weight in WEIGHTS:
    likelihood = (GaussianL2() if weight == 0.0
                  else WassersteinL2(zeta=1.0, scale=weight))
    target = Target(data, model, likelihood, table)
    best, detection, modes = [], [], []
    for seed in SEEDS:
        trace = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
        HybridDriver(schedule).run(make_state(start), target,
                                   np.random.default_rng(seed), n_total=1500,
                                   trace=trace)
        s = summarize(trace, data, table, model, reference=true_sites, burn=750,
                      stride=10, noise=DATA_NOISE)
        best.append(s.best_residual)
        detection.append(float(s.R_i.mean()))
        modes.append(s.k_mode)
    sweep.append((weight, np.median(best), np.mean(detection), modes))
print(f"{time.time() - t0:.0f} s\n")
print(f"{'zeta * scale':>13} {'median best σ':>14} {'mean R':>8} {'k mode':>10}")
for weight, med, det, modes in sweep:
    tag = "0  (GaussianL2)" if weight == 0 else f"{weight:.0e}"
    print(f"{tag:>13} {med:14.2f} {det:8.2f} {modes!s:>10}")
print(f"\ntruth k = {len(true_sites)}")

# %% [markdown]
# Inert, then harmful — but read the trend, not the rows. This sweep is
# abridged to two seeds and 1,500 steps to keep the notebook quick, and at that
# length the middle weights are within seed-to-seed noise of one another; only
# the collapse at the top is unambiguous.
#
# The full sweep in `docs/test-plan.md` §5.7 runs three seeds to $10^6$, where
# the median best residual reaches 28.94 σ against 0.92 for plain least squares
# and detection falls to 0.17 on one seed. Those are the numbers to quote.
#
# ### Why, when the distance does discriminate
#
# $\widehat{W}$ is not blind — it is 517 times larger at a random start than at
# the truth. The problem is that the residual discriminates *harder*, and is
# sharpest exactly where the transport distance is flattest.

# %%
print(f"{'configuration':22s} {'least squares':>14} {'W-hat':>9}")
for label, sites in (("truth", true_sites),
                     ("truth minus one spin", true_sites[:5]),
                     ("random 6", np.random.default_rng(31).choice(len(table), 6,
                                                                   replace=False)),
                     ("the 3-spin start", start)):
    pred = signal(sites)
    print(f"{label:22s} {lsq(pred):14.1f} "
          f"{wasserstein_signal_distance(pred, obs, TAU):9.5f}")
print("\nleast squares spans a factor of thousands; W-hat a factor of ~500,")
print("over a range too narrow to matter once it is scaled down enough not to")
print("overwhelm the residual.")

# %% [markdown]
# ## 3. On displaced data, least squares is the one that fails
#
# Now the regime the penalty was built for. The data is simulated from the true
# bath and recorded on a $\tau$ grid offset by a whole number of sampling
# intervals — correct physics, systematic timing error.
#
# The question put to each criterion: **how often does a wrong configuration
# beat the right one?**

# %%
def misranking_rate(offset_points, n_candidates=400, seed=0):
    """Fraction of random 6-spin candidates that outscore the true bath."""
    shifted = signal(true_sites, TAU + offset_points * DT)
    truth_lsq = lsq(signal(true_sites), shifted)
    truth_w = wasserstein_signal_distance(signal(true_sites), shifted, TAU)
    rng = np.random.default_rng(seed)
    beaten_lsq = beaten_w = 0
    for _ in range(n_candidates):
        candidate = signal(rng.choice(len(table), 6, replace=False))
        if lsq(candidate, shifted) > truth_lsq:
            beaten_lsq += 1
        if wasserstein_signal_distance(candidate, shifted, TAU) < truth_w:
            beaten_w += 1
    return beaten_lsq / n_candidates, beaten_w / n_candidates


OFFSETS = (0, 1, 2, 3, 5, 8)
t0 = time.time()
rates = np.array([misranking_rate(o) for o in OFFSETS])
print(f"{time.time() - t0:.0f} s\n")
print(f"{'offset':>22} {'least squares misled':>21} {'transport misled':>18}")
for offset, (a, b) in zip(OFFSETS, rates, strict=True):
    print(f"{f'{offset} pts ({offset * DT * 1e3:.3f} µs)':>22} "
          f"{a:20.1%} {b:17.1%}")

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.2))
shift_us = np.array(OFFSETS) * DT * 1e3
ax.plot(shift_us, rates[:, 0] * 100, "o-", color="crimson", lw=1.6,
        label="least squares")
ax.plot(shift_us, rates[:, 1] * 100, "o-", color="#023047", lw=1.6,
        label=r"transport $\widehat{W}$")
ax.set_xlabel(r"timing offset in the data (µs)")
ax.set_ylabel("wrong candidate wins (%)")
ax.set_title("which criterion is fooled by a displaced signal")
ax.legend(fontsize=9)
plt.tight_layout()

# %% [markdown]
# Two things in that curve, and the second is the interesting one.
#
# **Transport is markedly more robust to small offsets.** At three sampling
# intervals — under a tenth of a microsecond — least squares is misled by two
# thirds of random candidates and transport by a quarter. The penalty does what
# it was proposed to do.
#
# **The criteria cross over.** By eight intervals the ordering reverses: least
# squares recovers to 21% while transport degrades to 67%. The correct signal
# has been carried far enough that $\widehat{W}$ ranks it behind many random
# candidates, while a large offset degrades those candidates under least squares
# too.
#
# Neither criterion dominates. The window in which transport wins is set by the
# sampling interval and the modulation period, not by anything universal.

# %% [markdown]
# ## 4. What to actually do
#
# The penalty is a tool for a **diagnosed** problem, not a general improvement.
#
# - Leave $\zeta = 0$. On well-aligned data there is no weight that helps, and
#   weights above $\zeta s \approx 10^3$ actively destroy the recovery.
# - Below $\zeta s \approx 10$ the term cannot flip a single accept/reject
#   decision: the chain is bitwise identical to the least-squares one. If a
#   weight is set that low, the penalty is arithmetic with no effect.
# - Turn it on when a timing offset has been *identified* — not suspected — and
#   then sweep the weight yourself, because the useful value depends on the size
#   of the offset, which is what the crossover shows.
#
# And there is a better move available in that case. A diagnosed timing offset
# is a **forward-model** error, and modelling it as a fitted parameter is
# cleaner than paying for it with a distributional penalty that has its own
# failure mode.

# %%
target_plain = Target(data, model, GaussianL2(), table)
target_penalised = Target(data, model, WassersteinL2(zeta=1.0, scale=5.0), table)
state = make_state(start)
print("zeta*scale = 5, below the threshold where anything changes:")
print(f"  GaussianL2    {target_plain.log_prob(state)[0]:.6f}")
print(f"  WassersteinL2 {target_penalised.log_prob(state)[0]:.6f}")
print(f"  difference    "
      f"{abs(target_plain.log_prob(state)[0] - target_penalised.log_prob(state)[0]):.6f}")
print("\nNon-zero, and far too small to change any decision the sampler makes.")

# %% [markdown]
# ## Where to go next
#
# What was measured: on the recovery problem this package is built for, the
# transport penalty is inert below $\zeta s \approx 10$ and harmful above
# $10^3$, with no window of benefit — so the calibrated weight is zero. On data
# carrying a systematic timing offset it is substantially the better criterion
# over a window of offsets, and worse outside it.
#
# Both halves are in `docs/test-plan.md` §5.7, with the full sweep to $10^6$ and
# the step at which each weight first changes a sampler decision.
#
# The specification's §7.1 defines the distance — what measure is transported,
# and the two normalisations that make it a distance between distributions
# rather than between amplitudes — and §11 records the two places where this
# implementation departs from the published form.
