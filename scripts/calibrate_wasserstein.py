"""How does the transport penalty change recovery, as its weight varies?

Weight 0 is the negative control: it is GaussianL2 exactly.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "src")
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
    simulate_dataset,
    wasserstein_signal_distance,
)
from nuclear_spin_recovery.post import summarize

FULL = SiteTable.from_ivady_file("nv-2.txt", strong_thresh=750.0, weak_thresh=5.0)
KEEP = np.hypot(FULL.a_par, FULL.a_perp) >= 100.0
TABLE = SiteTable(distance=FULL.distance[KEEP], positions=FULL.positions[KEEP],
                  a_par=FULL.a_par[KEEP], a_perp=FULL.a_perp[KEEP],
                  isotope=FULL.isotope[KEEP], gyro=FULL.gyro[KEEP])
MODEL = AnalyticCCE1(StretchedExponential())
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
NOISE, LAM, SIGMA, K_MAX = 0.002, 3e-3, 0.02, 32
STEPS, BURN, SEEDS = 2000, 1000, (17, 23, 41)
WEIGHTS = (0.0, 1e2, 1e3, 1e4, 1e5, 1e6)


def state(sites, sigma=SIGMA):
    return State.from_sites(np.sort(np.asarray(list(sites), int)),
                            n_sites=len(TABLE), n_exp=1,
                            lam=np.array([[LAM]]), n_stretch=np.array([[1.0]]),
                            sigma=np.array([[sigma]]), k_max=K_MAX)


TRUE = np.sort(np.random.default_rng(7).choice(len(TABLE), 6, replace=False))
DATA = simulate_dataset(state(TRUE, NOISE),
                        ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)]),
                        TABLE, MODEL, sigma=NOISE, rng=np.random.default_rng(11))
WALK = DiscreteLatticeWalk(NeighborIndex(TABLE.positions, radius=6.0))
SCHEDULE = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 50),
    Step(ParallelTempering(Schedule([Step(RWMH(ParameterBlock("sites"), WALK), 1)]),
                           n_replicas=6), 50),
])
START = np.sort(np.random.default_rng(31).choice(len(TABLE), 3, replace=False))

# What the penalty can even see: transport cost at the truth against a draw.
w_truth = wasserstein_signal_distance(
    MODEL.coherence(state(TRUE), DATA, TABLE)[0], DATA.data_all, TAU)
w_random = wasserstein_signal_distance(
    MODEL.coherence(state(START), DATA, TABLE)[0], DATA.data_all, TAU)
print(f"W-hat at truth {w_truth:.5f}, at the start {w_random:.5f}, "
      f"ratio {w_random / max(w_truth, 1e-12):.0f}x\n", flush=True)

print(f"{'weight':>9} {'best sigma':>26} {'median':>7} {'R':>22} {'k mode':>14} "
      f"{'accept':>7}", flush=True)
t0 = time.time()
for weight in WEIGHTS:
    likelihood = (GaussianL2() if weight == 0.0
                  else WassersteinL2(weight=weight))
    target = Target(DATA, MODEL, likelihood, TABLE)
    best, R, kmode, acc = [], [], [], []
    for seed in SEEDS:
        trace = Trace(n_sites=len(TABLE), k_max=K_MAX, n_exp=1)
        HybridDriver(SCHEDULE).run(state(START), target,
                                   np.random.default_rng(seed),
                                   n_total=STEPS, trace=trace)
        summary = summarize(trace, DATA, TABLE, MODEL, reference=TRUE,
                            burn=BURN, stride=10, noise=NOISE)
        best.append(summary.best_residual)
        R.append(float(summary.R_i.mean()))
        kmode.append(summary.k_mode)
        sites = np.asarray(trace.site_idx)
        acc.append(float(np.mean(np.any(sites[1:] != sites[:-1], axis=1))))
    print(f"{weight:9.0f} {[round(b, 2) for b in best]!s:>26} "
          f"{np.median(best):7.2f} {[round(r, 2) for r in R]!s:>22} "
          f"{kmode!s:>14} {np.mean(acc):7.1%}", flush=True)
print(f"\ntruth k = {len(TRUE)}; weight 0 is GaussianL2 exactly.  "
      f"{time.time() - t0:.0f} s")
