"""The recovery ladder, T1 through T5.

Each rung adds exactly one mechanism.  Every assertion is on the posterior,
never on a final or modal state, and against the two criteria of specification
§9.1: does the forward model reproduce the data, and does the posterior contain
the spins that generated it.

Thresholds come from docs/test-plan.md §5.  They sit well below the measured
values on purpose: these are regression guards against a sampler that stops
working, not a record of the best performance seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    BirthDeathKernel,
    ContinuousReflected,
    DiscreteLatticeWalk,
    GaussianOffset,
    HybridDriver,
    NeighborIndex,
    ParallelTempering,
    ParameterBlock,
    RJMCMC,
    RWMH,
    Schedule,
    State,
    Step,
    Trace,
)
from .conftest import LAM, LIK_SIGMA

pytestmark = pytest.mark.slow

WALK_RADIUS = 5.0        # Angstrom; robust, and above the 1.54 A nearest neighbour
LOCAL_RADIUS = 3.0       # how far a T2a start is displaced from truth


def _trace(table, k_max=32):
    return Trace(n_sites=len(table), k_max=k_max, n_exp=1)


def _perturb(sites, rng, index):
    """Displace every spin to a nearby site, keeping them distinct."""
    out, taken = [], set()
    for s in sites:
        free = [c for c in index.neighbors(s) if c not in taken]
        pick = int(rng.choice(free)) if free else int(s)
        out.append(pick)
        taken.add(pick)
    return np.array(out)


# ══════════════════════════════════════════════════════ T1 — continuous block

def test_t1_lambda_posterior_covers_the_truth(table, simulated, make_state, metrics):
    """Spins and k at truth; only lambda varies.

    Lambda is a single identifiable scalar, so a credible-interval assertion is
    meaningful here in a way it is not for configurations.
    """
    rng = np.random.default_rng(0)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=0)

    start = make_state(sites, lam=1.2e-2)     # deliberately wrong
    trace = _trace(table)
    RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)).run(
        start, target, np.random.default_rng(1), n_steps=6000, trace=trace)

    post = trace.lam[2000:, 0]
    lo, hi = np.percentile(post, [2.5, 97.5])
    assert lo <= LAM <= hi


def test_t1_lambda_posterior_concentrates(table, simulated, make_state):
    """A posterior as wide as the prior has learned nothing."""
    rng = np.random.default_rng(0)
    sites = rng.choice(len(table), size=8, replace=False)
    _, _, target = simulated(sites, seed=0)

    trace = _trace(table)
    RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)).run(
        make_state(sites, lam=1.2e-2), target, np.random.default_rng(1),
        n_steps=6000, trace=trace)

    prior_width = 2e-2 - 1e-4
    assert np.std(trace.lam[2000:, 0]) < 0.05 * prior_width


def test_t1_fit_reaches_the_noise(table, simulated, make_state, metrics):
    rng = np.random.default_rng(0)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=0)

    trace = _trace(table)
    RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)).run(
        make_state(sites, lam=1.2e-2), target, np.random.default_rng(1),
        n_steps=6000, trace=trace)

    m = metrics(trace, truth, data, burn=2000)
    assert m.median_residual < 5.0


# ═══════════════════════════════════════════════════ T2a — discrete, local

def test_t2a_recovers_strong_couplings(table, simulated, stratified_bath,
                                       make_state, metrics):
    """Truth within reach, so a failure implicates the acceptance rule.

    Threshold from test-plan §5.4: measured R = 0.83-0.92 above 100 kHz,
    asserted at 0.60.  Nothing is asserted below 25 kHz, where R is ~0.01
    regardless -- those spins are not identifiable at these settings.
    """
    rng = np.random.default_rng(7)
    sites = stratified_bath(rng)
    truth, data, target = simulated(sites, seed=7)

    start = _perturb(sites, rng, NeighborIndex(table.positions, LOCAL_RADIUS))
    trace = _trace(table)
    RWMH(ParameterBlock("sites"),
         DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))).run(
        make_state(start), target, np.random.default_rng(7),
        n_steps=12000, trace=trace)

    m = metrics(trace, truth, data, burn=5000)
    assert m.R(100, 750) > 0.60
    assert m.median_residual < 5.0


def test_t2a_weak_couplings_are_not_asserted(table, simulated, stratified_bath,
                                             make_state, metrics):
    """Documents the floor rather than demanding recovery.

    A future change that raises this number should be treated as suspicious:
    the application paper puts the detection floor at 7.8 kHz for these
    settings, and the 0-25 kHz regime stays ill-posed however much data is
    added.
    """
    rng = np.random.default_rng(7)
    sites = stratified_bath(rng)
    truth, data, target = simulated(sites, seed=7)

    trace = _trace(table)
    RWMH(ParameterBlock("sites"),
         DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))).run(
        make_state(_perturb(sites, rng, NeighborIndex(table.positions, LOCAL_RADIUS))),
        target, np.random.default_rng(7), n_steps=12000, trace=trace)

    m = metrics(trace, truth, data, burn=5000)
    assert m.R(5, 25) < 0.30


# ══════════════════════════════════════════════════ T2b — discrete, global

def test_t2b_fits_from_a_random_start(table, simulated, stratified_bath,
                                      make_state, metrics):
    """Measures mobility; asserts only criterion A.

    Its detection numbers are the baseline T4 must beat, and are recorded
    rather than asserted.
    """
    rng = np.random.default_rng(11)
    sites = stratified_bath(rng)
    truth, data, target = simulated(sites, seed=11)

    start = rng.choice(len(table), size=len(sites), replace=False)
    trace = _trace(table)
    RWMH(ParameterBlock("sites"),
         DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))).run(
        make_state(start), target, np.random.default_rng(11),
        n_steps=12000, trace=trace)

    m = metrics(trace, truth, data, burn=5000)
    assert m.median_residual < 8.0


# ═══════════════════════════════════════════════════════ T3 — dimension

@pytest.mark.parametrize("k_start", ["above", "below"])
def test_t3_recovers_the_number_of_spins(table, simulated, make_state,
                                         metrics, k_start):
    """Birth and death fail differently, so start on both sides."""
    rng = np.random.default_rng(3)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=3)

    start = sites[:4] if k_start == "below" else np.concatenate(
        [sites, rng.choice(np.setdiff1d(np.arange(len(table)), sites),
                           size=6, replace=False)])
    trace = _trace(table)
    RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=32)).run(
        make_state(start), target, np.random.default_rng(3),
        n_steps=12000, trace=trace)

    m = metrics(trace, truth, data, burn=5000)
    assert m.k_mode == int(truth.k[0])


def test_t3_does_not_overfit(table, simulated, make_state, metrics):
    """With k free, more spins always fit better.

    A residual floor is the assertion that the prior carried by gamma is doing
    its job; without it this rung passes while k drifts upward.
    """
    rng = np.random.default_rng(3)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=3)

    trace = _trace(table)
    RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=32)).run(
        make_state(sites[:4]), target, np.random.default_rng(3),
        n_steps=12000, trace=trace)

    m = metrics(trace, truth, data, burn=5000)
    assert m.median_residual > 0.5
    assert m.k_posterior.max() < 32


# ═══════════════════════════════════════════════════════ T4 — tempering

def _single_block_baseline(table, simulated, sites, make_state, metrics, seed):
    """T2b's numbers, recomputed on the same data T4 will use."""
    truth, data, target = simulated(sites, seed=seed)
    rng = np.random.default_rng(seed)
    start = rng.choice(len(table), size=len(sites), replace=False)
    trace = _trace(table)
    RWMH(ParameterBlock("sites"),
         DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))).run(
        make_state(start), target, np.random.default_rng(seed),
        n_steps=12000, trace=trace)
    return metrics(trace, truth, data, burn=5000), start, truth, data, target


def test_t4_tempering_beats_the_single_block_baseline(
        table, simulated, stratified_bath, make_state, metrics):
    """Tempering's claim is that it escapes local minima.

    That is only testable against a baseline, so this rung is a comparison
    rather than an absolute threshold -- and it is the only form of the claim
    that can fail for the right reason.
    """
    sites = stratified_bath(np.random.default_rng(21))
    base, start, truth, data, target = _single_block_baseline(
        table, simulated, sites, make_state, metrics, seed=21)

    inner = Schedule([
        Step(RWMH(ParameterBlock("sites"),
                  DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))), 1),
    ])
    trace = _trace(table)
    ParallelTempering(inner, n_replicas=6).run(
        make_state(start), target, np.random.default_rng(21),
        n_steps=12000, trace=trace)

    hot = metrics(trace, truth, data, burn=5000)
    assert hot.R(100, 750) > base.R(100, 750)
    assert hot.median_residual <= base.median_residual


def test_t4_swap_acceptance_is_workable(table, simulated, stratified_bath,
                                        make_state):
    """A ladder nobody crosses is a ladder that does nothing."""
    sites = stratified_bath(np.random.default_rng(22))
    _, _, target = simulated(sites, seed=22)

    inner = Schedule([
        Step(RWMH(ParameterBlock("sites"),
                  DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))), 1),
    ])
    ladder = ParallelTempering(inner, n_replicas=6)
    st = make_state(sites).expand_replicas(6)
    rng = np.random.default_rng(22)

    swaps = 0
    for _ in range(400):
        before = st.site_idx.copy()
        st = ladder.attempt_swap(st, target, rng)
        swaps += not np.array_equal(before, st.site_idx)
    assert 0.05 < swaps / 400 < 0.95


def test_t4_mixed_inner_schedule_recovers_lambda_too(
        table, simulated, stratified_bath, make_state, metrics):
    """A rung advances both blocks before a swap is attempted."""
    sites = stratified_bath(np.random.default_rng(23))
    truth, data, target = simulated(sites, seed=23)

    inner = Schedule([
        Step(RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)), 1),
        Step(RWMH(ParameterBlock("sites"),
                  DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))), 1),
    ])
    trace = _trace(table)
    start = make_state(_perturb(sites, np.random.default_rng(23),
                                NeighborIndex(table.positions, LOCAL_RADIUS)),
                       lam=1.0e-2)
    ParallelTempering(inner, n_replicas=6).run(
        start, target, np.random.default_rng(23), n_steps=12000, trace=trace)

    post = trace.lam[5000:, 0]
    lo, hi = np.percentile(post, [2.5, 97.5])
    assert lo <= LAM <= hi


# ══════════════════════════════════════════ T5 — full hybrid, relaxed prior

def _off_table_truth(table, sites, rng, delta=1.0):
    """Perturb the couplings off their DFT values, the regime §5.3 targets."""
    return (rng.normal(0.0, delta, size=len(sites)),
            rng.normal(0.0, delta, size=len(sites)))


def test_t5_relaxation_beats_the_hard_constraint(
        table, simulated, stratified_bath, make_state, metrics, model):
    """Truth generated off-table, so pinned offsets cannot reproduce it.

    Recovering on-table truth would pass trivially with the offsets pinned at
    zero and would prove nothing.
    """
    from nuclear_spin_recovery import (Experiment, ExperimentSet, GaussianL2,
                                       Target, simulate_dataset)
    from .conftest import B_Z, DATA_NOISE, N_PULSES, TAU

    rng = np.random.default_rng(31)
    sites = stratified_bath(rng, per_band=3)
    truth = make_state(sites, sigma=DATA_NOISE)
    dpar, dperp = _off_table_truth(table, sites, rng, delta=1.0)
    truth.dA_par[0, : len(sites)] = dpar
    truth.dA_perp[0, : len(sites)] = dperp

    blank = ExperimentSet([Experiment(tau=TAU, n_pulses=N_PULSES, b_z=B_Z)])
    data = simulate_dataset(truth, blank, table, model, sigma=DATA_NOISE,
                            rng=np.random.default_rng(9031))
    target = Target(data, model, GaussianL2(), table)

    site_step = Step(RWMH(ParameterBlock("sites"),
                          DiscreteLatticeWalk(NeighborIndex(table.positions,
                                                            WALK_RADIUS))), 2)
    start = make_state(_perturb(sites, rng, NeighborIndex(table.positions,
                                                          LOCAL_RADIUS)))

    constrained = _trace(table)
    HybridDriver(Schedule([site_step])).run(
        start, target, np.random.default_rng(31), n_total=10000, trace=constrained)

    relaxed = _trace(table)
    HybridDriver(Schedule([
        site_step,
        Step(RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=1.0)), 2),
    ])).run(start, target, np.random.default_rng(31), n_total=10000, trace=relaxed)

    hard = metrics(constrained, truth, data, burn=4000)
    soft = metrics(relaxed, truth, data, burn=4000)
    assert soft.median_residual < hard.median_residual


def test_t5_offsets_stay_near_the_dft_values(
        table, simulated, stratified_bath, make_state, metrics):
    """The prior must keep offsets physically plausible.

    Offsets wandering far past the trusted DFT accuracy would mean the
    constraint had been abandoned rather than relaxed.
    """
    rng = np.random.default_rng(32)
    sites = stratified_bath(rng, per_band=3)
    truth, data, target = simulated(sites, seed=32)

    final = HybridDriver(Schedule([
        Step(RWMH(ParameterBlock("sites"),
                  DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))), 2),
        Step(RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=1.0)), 2),
    ])).run(make_state(sites), target, np.random.default_rng(32), n_total=8000)

    k = int(final.k[0])
    assert np.all(np.abs(final.dA_par[0, :k]) < 5.0)
    assert np.all(np.abs(final.dA_perp[0, :k]) < 5.0)


# ═══════════════════════════════════════════════ posterior predictive check

def test_posterior_predictive_envelope_brackets_the_data(
        table, simulated, make_state, metrics):
    """Criterion A in its proper form: predictions from sampled configurations."""
    rng = np.random.default_rng(41)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=41)

    trace = _trace(table)
    RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)).run(
        make_state(sites, lam=1.2e-2), target, np.random.default_rng(41),
        n_steps=6000, trace=trace)

    m = metrics(trace, truth, data, burn=2000)
    lo = m.predictive.min(axis=0)
    hi = m.predictive.max(axis=0)
    inside = np.mean((data.data_all >= lo - 4 * 0.002) &
                     (data.data_all <= hi + 4 * 0.002))
    assert inside > 0.9
