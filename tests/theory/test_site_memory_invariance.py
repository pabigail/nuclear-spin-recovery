"""Does a sampler with site memory still target the distribution it claims to?

Site memory changes what every move does to an offset: a hop exchanges it for
the destination's, a birth resumes one, a death leaves one behind, a tempering
swap exchanges whole memories.  Any of those done wrongly would still recover
a planted configuration, because recovery only needs the chain to end up near
the likelihood maximum.  What it would get wrong is the distribution.

So the likelihood is switched off.  With a flat target the sampler must
reproduce its own prior: at every site, the offset must follow that site's
prior, whose width differs from site to site, however the spins come and go.

The last test is the other half: with a likelihood, on the toy system the
design was worked out on, the right sites and couplings come back.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    DiscreteLatticeWalk,
    Envelope,
    Experiment,
    ExperimentSet,
    GaussianL2,
    HybridDriver,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    Schedule,
    SiteScaledOffset,
    SiteTable,
    State,
    Step,
    Target,
    Trace,
    gyromagnetic_ratio,
    modal_configuration,
    simulate_dataset,
)

pytestmark = pytest.mark.slow

#: As in test_sampler_invariance.py: strict, because these run at fixed seeds
#: and the faults they exist to catch drive the p-value to zero, not to 0.02.
P_MIN = 1e-3

# Five sites whose widths differ by a factor of twenty-four, one of them held
# up by the floor.  A kernel that applied one site's width at another would
# be visible at once.
A_PAR = np.array([120.0, -60.0, 20.0, 2.0, 80.0])
A_PERP = np.array([45.0, 30.0, 10.0, 1.0, 5.0])
N_SITES, K_MAX = len(A_PAR), 3
FRACTION, FLOOR = 0.1, 0.5
WIDTH = np.column_stack([np.maximum(FLOOR, FRACTION * np.abs(A_PAR)),
                         np.maximum(FLOOR, FRACTION * np.abs(A_PERP))])
POSITIONS = np.column_stack([np.arange(N_SITES, dtype=float),
                             np.zeros(N_SITES), np.ones(N_SITES)])
TABLE = SiteTable(
    distance=np.linalg.norm(POSITIONS, axis=1), positions=POSITIONS,
    a_par=A_PAR, a_perp=A_PERP, isotope=np.array(["13C"] * N_SITES),
    gyro=np.full(N_SITES, 6.7283))


class Flat:
    site_table = TABLE

    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


def run_prior_chain(prior, redraw, seed, n_steps):
    """Every move the package has, under a flat likelihood, with site memory."""
    offsets = RWMH(ParameterBlock("offsets"), SiteScaledOffset(
        100.0, TABLE, fraction_par=FRACTION, fraction_perp=FRACTION,
        floor=FLOOR, prior=prior, redraw_unoccupied=redraw))
    sites = RWMH(ParameterBlock("sites"),
                 DiscreteLatticeWalk(NeighborIndex(POSITIONS, radius=10.0)))
    jump = RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX))
    schedule = Schedule([
        Step(jump, 1), Step(sites, 1), Step(offsets, 6),
        Step(ParallelTempering(
            Schedule([Step(jump, 1), Step(sites, 1), Step(offsets, 2)]),
            n_replicas=3), 1),
    ])
    state = State.from_sites((0, 2), n_sites=N_SITES, n_exp=1,
                             lam=np.array([[3e-3]]), n_stretch=np.ones((1, 1)),
                             sigma=np.array([[0.1]]), k_max=K_MAX,
                             site_memory=True)
    trace = Trace(n_sites=N_SITES, k_max=K_MAX, n_exp=1)
    HybridDriver(schedule).run(state, Flat(), np.random.default_rng(seed),
                               n_steps, trace=trace)
    return trace


def offsets_by_site(trace, burn, thin):
    """Per site, the (d_par, d_perp) samples from thinned draws it was
    occupied in."""
    site = np.asarray(trace.site_idx)[burn::thin]
    k = np.asarray(trace.k)[burn::thin]
    d_par = np.asarray(trace.dA_par)[burn::thin]
    d_perp = np.asarray(trace.dA_perp)[burn::thin]
    live = np.arange(K_MAX)[None, :] < k[:, None]
    return [np.column_stack([d_par[live & (site == i)],
                             d_perp[live & (site == i)]])
            for i in range(N_SITES)]


def prior_cdf(prior, width, n_sigma=5.0):
    if prior == "flat":
        return stats.uniform(-width, 2 * width).cdf
    return stats.truncnorm(-n_sigma, n_sigma, scale=width).cdf


# Chain length and thinning are measured, not guessed.  At 120,000 steps
# thinned by 150 each site contributes 216 to 253 draws, with a lag-one
# autocorrelation of -0.10, and the smallest p-value over all forty
# combinations of site, component and setting is 0.04.  Scored against the
# width of another site instead, the same draws give 8e-48.
@pytest.mark.parametrize("redraw", [False, True])
@pytest.mark.parametrize("prior", ["gaussian", "flat"])
def test_each_sites_offset_follows_that_sites_prior(prior, redraw):
    trace = run_prior_chain(prior, redraw, seed=11, n_steps=120_000)
    samples = offsets_by_site(trace, burn=5_000, thin=150)
    for i in range(N_SITES):
        assert len(samples[i]) > 150, f"site {i} was hardly visited"
        for component in (0, 1):
            p = stats.kstest(samples[i][:, component],
                             prior_cdf(prior, WIDTH[i, component])).pvalue
            assert p > P_MIN, (
                f"site {i}, component {component}: offsets do not follow the "
                f"prior of width {WIDTH[i, component]} (p = {p:.2g})")


def test_the_test_above_can_tell_one_width_from_another():
    """Guards it: scored against the wrong site's prior, the same samples
    must fail."""
    trace = run_prior_chain("gaussian", False, seed=11, n_steps=120_000)
    samples = offsets_by_site(trace, burn=5_000, thin=150)
    wrong = stats.kstest(samples[3][:, 0], prior_cdf("gaussian", WIDTH[0, 0])).pvalue
    assert wrong < 1e-20


def test_sites_are_visited_evenly_and_spin_counts_stay_in_range():
    """Memory must not make some sites stickier than others."""
    trace = run_prior_chain("gaussian", False, seed=12, n_steps=120_000)
    site = np.asarray(trace.site_idx)[5_000::150]
    k = np.asarray(trace.k)[5_000::150]
    live = np.arange(K_MAX)[None, :] < k[:, None]
    visits = np.bincount(site[live], minlength=N_SITES)
    assert stats.chisquare(visits).pvalue > P_MIN
    assert set(np.unique(k)) == set(range(K_MAX + 1))


# ------------------------------------------------------------------ recovery

class NoEnvelope(Envelope):
    def __call__(self, tau, exp_id, lam, n_stretch):
        return np.ones_like(np.asarray(tau, dtype=float))


def test_recovers_three_relaxed_spins_from_ten_sites():
    """The toy system of notebooks/toy_three_spins_ten_sites, as a rung.

    Three spins, each sitting away from its table value, to be found among
    ten sites whose allowed range is ten percent of their own coupling.
    """
    centres = np.array([
        (100.0, 100.0), (10.0, 10.0), (100.0, 50.0), (60.0, 80.0),
        (150.0, 60.0), (40.0, 30.0), (75.0, 120.0), (130.0, 110.0),
        (25.0, 60.0), (55.0, 45.0)])
    truth_offsets = {2: (5.0, 2.0), 3: (-4.0, 5.0), 7: (8.0, -6.0)}
    n = len(centres)
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack([2 * np.cos(angle), 2 * np.sin(angle),
                                 np.full(n, 2.0)])
    table = SiteTable(
        distance=np.linalg.norm(positions, axis=1), positions=positions,
        a_par=centres[:, 0], a_perp=centres[:, 1],
        isotope=np.array(["13C"] * n),
        gyro=np.full(n, gyromagnetic_ratio("13C")))
    model = AnalyticCCE1(NoEnvelope())

    def make_state(sites):
        return State.from_sites(
            sites, n_sites=n, n_exp=1, lam=np.ones((1, 1)),
            n_stretch=np.ones((1, 1)), sigma=np.full((1, 1), 0.02), k_max=3,
            site_memory=True)

    truth = make_state(tuple(truth_offsets))
    for slot, (d_par, d_perp) in enumerate(truth_offsets.values()):
        truth.set_offset(0, slot, 0, d_par)
        truth.set_offset(0, slot, 1, d_perp)
    blank = ExperimentSet([Experiment(tau=np.linspace(3.2e-5, 8e-3, 250),
                                      n_pulses=4, b_z=311.0)])
    data = simulate_dataset(truth, blank, table, model, sigma=0.002,
                            rng=np.random.default_rng(3))
    target = Target(data, model, GaussianL2(), table)

    schedule = Schedule([
        Step(RWMH(ParameterBlock("sites"),
                  DiscreteLatticeWalk(NeighborIndex(positions, 10.0))), 1),
        Step(RWMH(ParameterBlock("offsets"), SiteScaledOffset(
            1.0, table, fraction_par=0.1, fraction_perp=0.1, prior="flat")), 5),
    ])
    trace = Trace(n_sites=n, k_max=3, n_exp=1)
    HybridDriver(schedule).run(make_state((0, 5, 9)), target,
                               np.random.default_rng(0), 15_000, trace=trace)

    modal = modal_configuration(trace, burn=3_000)
    assert modal.sites == (2, 3, 7)
    assert modal.share == 1.0
    for site, d_par, d_perp in zip(modal.sites, modal.d_par, modal.d_perp,
                                   strict=True):
        assert d_par == pytest.approx(truth_offsets[site][0], abs=1.0)
        assert d_perp == pytest.approx(truth_offsets[site][1], abs=1.0)
