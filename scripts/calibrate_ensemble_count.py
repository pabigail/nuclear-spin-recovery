"""How many ensembles are needed before the pooled estimate stops moving?

The question is about variance, not accuracy.  Pooled R is, at equal sample
counts, the mean of the per-ensemble values, so its expectation does not
improve with M -- its spread across independent sets does.  The study therefore
runs several independent *sets* of 20 ensembles and reports, at each M, how much
the pooled answer still depends on which set you happened to run.

Prefix-stable seeds make this nearly free: M ensembles are the first M of a set
already computed, not a fresh draw.
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
    simulate_dataset,
    spread_across_k,
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
STEPS, BURN = 1000, 400
MAX_M, COUNTS = 20, (1, 2, 3, 5, 10, 20)
ROOTS = (2026, 11, 404, 7, 55, 900)


def state(sites, sigma=SIGMA):
    return State.from_sites(np.sort(np.asarray(list(sites), int)),
                            n_sites=len(TABLE), n_exp=1,
                            lam=np.array([[LAM]]), n_stretch=np.array([[1.0]]),
                            sigma=np.array([[sigma]]), k_max=K_MAX)


TRUE = np.sort(np.random.default_rng(7).choice(len(TABLE), 6, replace=False))
DATA = simulate_dataset(state(TRUE, NOISE),
                        ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)]),
                        TABLE, MODEL, sigma=NOISE, rng=np.random.default_rng(11))
TARGET = Target(DATA, MODEL, GaussianL2(), TABLE)
WALK = DiscreteLatticeWalk(NeighborIndex(TABLE.positions, radius=6.0))
SCHEDULE = Schedule([
    Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 80),
    Step(ParallelTempering(
        Schedule([Step(RWMH(ParameterBlock("sites"), WALK), 1)]), n_replicas=6), 160),
])

t0 = time.time()
records = {m: {"R": [], "kerr": [], "best": [], "spread": []} for m in COUNTS}
for root in ROOTS:
    runner = EnsembleRunner(SCHEDULE, n_ensembles=MAX_M, n_steps=STEPS,
                            n_burn=BURN, init=spread_across_k(state, (3, 9)),
                            init_name="spread_across_k")
    full = runner.run(TARGET, root_seed=root)
    print(f"root {root}: {MAX_M} ensembles in {time.time() - t0:.0f} s", flush=True)
    for m in COUNTS:
        subset = EnsembleResult(traces=full.traces[:m], seeds=full.seeds[:m],
                                init_name=full.init_name)
        s = summarize(subset.pooled, DATA, TABLE, MODEL, reference=TRUE,
                      burn=0, stride=100, noise=NOISE)
        records[m]["R"].append(float(s.R_i.mean()))
        records[m]["kerr"].append(abs(s.k_mode - len(TRUE)))
        records[m]["best"].append(s.best_residual)
        records[m]["spread"].append(subset.agreement().k_mode_spread)

print(f"\n{'M':>3} {'pooled R  mean +/- spread':>26} {'|mode k - 6|':>16} "
      f"{'best sigma':>14} {'k spread':>10}")
for m in COUNTS:
    r = np.array(records[m]["R"])
    kerr = np.array(records[m]["kerr"])
    best = np.array(records[m]["best"])
    spread = np.array(records[m]["spread"])
    print(f"{m:3d} {r.mean():13.3f} +/- {r.std():7.3f} "
          f"{kerr.mean():10.2f} +/- {kerr.std():.2f} "
          f"{np.median(best):9.2f} {spread.mean():10.1f}")
print(f"\ntruth k = {len(TRUE)}; {len(ROOTS)} independent sets; "
      f"spread is the std across sets.  {time.time() - t0:.0f} s")
