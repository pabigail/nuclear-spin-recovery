"""Shared metrics for the recovery ladder.

One helper, used by every rung, so no test reimplements a measure.  This is not
tidiness: the measurement error that prompted docs/test-plan.md -- computing
detection against a single final state rather than over the posterior --
happened because the metric was written inline at the point of use.

See docs/test-plan.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    simulate_coherence,
    simulate_dataset,
)

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


@dataclass
class Metrics:
    """Everything a rung asserts on. Computed over the posterior, never a state."""

    R_i: np.ndarray            # detection rate per reference spin
    magnitude: np.ndarray      # coupling magnitude of each reference spin
    residual: np.ndarray       # RMS residual per sampled configuration, in units of sigma
    k_posterior: np.ndarray    # sampled values of k
    false_absence: float       # FP of spec §9.2
    predictive: np.ndarray     # (n_draws, n_points) posterior-predictive signals

    def R(self, lo, hi):
        """Mean detection rate within a coupling band."""
        sel = (self.magnitude >= lo) & (self.magnitude < hi)
        return float(self.R_i[sel].mean()) if sel.any() else float("nan")

    @property
    def median_residual(self):
        return float(np.median(self.residual))

    @property
    def best_residual(self):
        """Lowest residual any sampled configuration achieves.

        The statistic to use when comparing a model against one nested inside
        it.  A model with extra sampled parameters has a *higher* median
        residual than one holding them at the prior mean, because a typical
        draw sits away from that mean -- so a median comparison penalises the
        richer model for exploring.  What it should be asked is whether it can
        reach a fit the constrained model cannot.  Only meaningful between runs
        with the same number of posterior samples.
        """
        return float(np.min(self.residual))

    @property
    def k_mode(self):
        return int(np.bincount(self.k_posterior).argmax())


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
def metrics(table, model, make_state):
    """Compute every ladder metric from a trace. See test-plan §3."""
    def compute(trace, truth, data, burn, stride=50, noise=DATA_NOISE, tbl=None):
        tbl = table if tbl is None else tbl
        post = trace.discard_burn_in(burn)
        idx = range(0, len(post), max(1, stride))

        def couplings(sites):
            return set(zip(np.round(tbl.a_par[sites], 4),
                           np.round(tbl.a_perp[sites], 4)))

        samples = [couplings(post.site_idx[j, : int(post.k[j])]) for j in range(len(post))]
        ref = list(zip(tbl.a_par[np.asarray(truth.site_idx[0, : int(truth.k[0])])],
                       tbl.a_perp[np.asarray(truth.site_idx[0, : int(truth.k[0])])]))

        R_i = np.array([
            np.mean([any(abs(a - c) <= MATCH_TOL and abs(b - d) <= MATCH_TOL
                         for c, d in S) for S in samples])
            for a, b in ref])

        obs = data.data_all
        predictive, residual = [], []
        for j in idx:
            k = int(post.k[j])
            st = make_state(post.site_idx[j, :k], k_max=post.k_max, tbl=tbl)
            # Offsets must come from the trace.  Rebuilding from site indices
            # alone pins them at zero, which scores a relaxed run as if its
            # constraint had never been relaxed.
            st.dA_par[0, :k] = post.dA_par[j, :k]
            st.dA_perp[0, :k] = post.dA_perp[j, :k]
            pred = simulate_coherence(st, data, tbl, model)
            predictive.append(pred)
            residual.append(np.sqrt(np.mean((obs - pred) ** 2)) / noise)

        modal_k = int(np.bincount(post.k).argmax())
        modal_j = int(np.flatnonzero(post.k == modal_k)[0])
        modal = couplings(post.site_idx[modal_j, :modal_k])
        fa = (np.mean([[s not in S for S in samples] for s in modal])
              if modal else 0.0)

        return Metrics(
            R_i=R_i,
            magnitude=np.hypot(*np.array(ref).T) if ref else np.array([]),
            residual=np.array(residual),
            k_posterior=np.asarray(post.k),
            false_absence=float(fa),
            predictive=np.array(predictive),
        )
    return compute
