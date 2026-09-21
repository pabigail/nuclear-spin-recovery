"""Shared metrics for the recovery ladder.

One helper, used by every rung, so no test reimplements a measure.  This is not
tidiness: the measurement error that prompted docs/test-plan.md -- computing
detection against a single final state rather than over the posterior --
happened because the metric was written inline at the point of use.

See docs/test-plan.md §3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    GaussianL2,
    SiteTable,
    State,
    StretchedExponential,
    Target,
    simulate_dataset,
)
from nuclear_spin_recovery.post import PosteriorSummary, summarize

REPO = Path(__file__).resolve().parents[2]

# Default experimental settings, matching the papers and the calibration in
# docs/test-plan.md §5.1.
N_PULSES, B_Z, LAM = 16, 311.0, 3e-3
DATA_NOISE = 0.002
LIK_SIGMA = 0.02          # calibrated, §5.2 -- not the data noise
TAU = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
MATCH_TOL = 0.1           # kHz, absorbs numerical differences between
                          # symmetry-related sites
BANDS = ((5, 25), (25, 100), (100, 750))


#: The ladder's metric object is now the package's own.  Extracting it out of
#: this file was the point of phase 4a: the notebook and the tests had grown
#: separate copies, which is how the posterior-vs-final-state error happened in
#: the first place.  See docs/phase-4-plan.md, unit 4a.
Metrics = PosteriorSummary


@pytest.fixture(scope="session")
def table():
    return SiteTable.from_ivady_file(
        REPO / "nv-2.txt", strong_thresh=750.0, weak_thresh=5.0)


@pytest.fixture(scope="session")
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture(scope="session")
def detectable_table(table):
    """Only sites a CPMG-16 experiment at 311 G can actually resolve.

    Below roughly 25 kHz a spin modulates the signal by less than the noise,
    so adding a spurious one barely changes the likelihood and is accepted
    about half the time.  On the full 3557-site table that random walk carries
    k far above the truth, which measures identifiability rather than the
    sampler.  Restricting the candidate pool makes a spurious spin cost
    something, so the model dimension becomes identifiable and the trans-
    dimensional machinery can be tested on its own.

    Measured (test-plan §5.6), k_true = 8: full table gives a posterior mode of
    17, a 25 kHz cutoff gives 9, and this one gives 8.
    """
    magnitude = np.hypot(table.a_par, table.a_perp)
    keep = magnitude >= 100.0
    return SiteTable(
        distance=table.distance[keep], positions=table.positions[keep],
        a_par=table.a_par[keep], a_perp=table.a_perp[keep],
        isotope=table.isotope[keep], gyro=table.gyro[keep])


@pytest.fixture
def make_state(table):
    def build(sites, lam=LAM, sigma=LIK_SIGMA, k_max=32, tbl=None):
        tbl = table if tbl is None else tbl
        return State.from_sites(
            np.sort(np.asarray(list(sites), dtype=int)),
            n_sites=len(tbl), n_exp=1,
            lam=np.array([[lam]]), n_stretch=np.array([[1.0]]),
            sigma=np.array([[sigma]]), k_max=k_max)
    return build


@pytest.fixture
def stratified_bath(table):
    """A bath spanning every coupling band.

    Random draws from the filtered table are dominated by weak couplings and
    leave the strong bands with too few spins to measure (test-plan §5.1).
    """
    magnitude = np.hypot(table.a_par, table.a_perp)

    def build(rng, per_band=4):
        sites = []
        for lo, hi in BANDS:
            pool = np.flatnonzero((magnitude >= lo) & (magnitude < hi))
            sites.extend(rng.choice(pool, size=min(per_band, pool.size), replace=False))
        return np.array(sorted(set(sites)))
    return build


@pytest.fixture
def simulated(table, model, make_state):
    """Truth, data and target for one simulated bath."""
    def build(sites, seed, lam=LAM, noise=DATA_NOISE, tbl=None):
        tbl = table if tbl is None else tbl
        truth = make_state(sites, lam=lam, sigma=noise, tbl=tbl)
        blank = ExperimentSet([Experiment(tau=TAU, n_pulses=N_PULSES, b_z=B_Z)])
        data = simulate_dataset(truth, blank, tbl, model, sigma=noise,
                                rng=np.random.default_rng(seed + 9000))
        return truth, data, Target(data, model, GaussianL2(), tbl)
    return build


@pytest.fixture
def metrics(table, model):
    """Compute every ladder metric from a trace. See test-plan §3.

    A thin adapter over :func:`nuclear_spin_recovery.post.summarize`; the
    measurement itself lives in the package now.
    """
    def compute(trace, truth, data, burn, stride=50, noise=DATA_NOISE, tbl=None):
        tbl = table if tbl is None else tbl
        reference = np.asarray(truth.site_idx[0, : int(truth.k[0])])
        return summarize(trace, data, tbl, model, reference=reference,
                         burn=burn, stride=stride, noise=noise, tol=MATCH_TOL)
    return compute
