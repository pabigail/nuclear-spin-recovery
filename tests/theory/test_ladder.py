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
    EnsembleRunner,
    ExpectedInformationGain,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    ParticleSet,
    simulate_dataset,
    GaussianL2,
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
    Target,
    Trace,
    WassersteinL2,
    spread_across_k,
)
from nuclear_spin_recovery.post import summarize

from .conftest import B_Z, LAM, LIK_SIGMA, N_PULSES

pytestmark = pytest.mark.slow

WALK_RADIUS = 5.0        # Angstrom; robust, and above the 1.54 A nearest neighbour
LOCAL_RADIUS = 3.0       # how far a T2a start is displaced from truth

# Chain lengths, calibrated in docs/test-plan.md §5.6 rather than guessed.
CHAIN_STEPS = 8000       # single-block rungs
LADDER_STEPS = 6000      # tempered rungs, which cost n_replicas inner steps each
N_REPLICAS = 6           # J = 4 on a geometric ladder never swaps; J = 6 does
BURN = 3000


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
        start, target, np.random.default_rng(1), n_steps=5000, trace=trace)

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
        n_steps=5000, trace=trace)

    prior_width = 2e-2 - 1e-4
    assert np.std(trace.lam[2000:, 0]) < 0.05 * prior_width


def test_t1_fit_reaches_the_noise(table, simulated, make_state, metrics):
    rng = np.random.default_rng(0)
    sites = rng.choice(len(table), size=8, replace=False)
    truth, data, target = simulated(sites, seed=0)

    trace = _trace(table)
    RWMH(ParameterBlock("lam"), ContinuousReflected(4e-4, 1e-4, 2e-2)).run(
        make_state(sites, lam=1.2e-2), target, np.random.default_rng(1),
        n_steps=5000, trace=trace)

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
        n_steps=CHAIN_STEPS, trace=trace)

    m = metrics(trace, truth, data, burn=BURN)
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
        target, np.random.default_rng(7), n_steps=CHAIN_STEPS, trace=trace)

    m = metrics(trace, truth, data, burn=BURN)
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
        n_steps=CHAIN_STEPS, trace=trace)

    m = metrics(trace, truth, data, burn=BURN)
    assert m.median_residual < 8.0


# ═══════════════════════════════════════════════════════ T3 — dimension

def _run_rjmcmc(tbl, target, start, make_state, seed=3, n_steps=None):
    trace = Trace(n_sites=len(tbl), k_max=32, n_exp=1)
    RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=32)).run(
        make_state(start, tbl=tbl), target, np.random.default_rng(seed),
        n_steps=n_steps or CHAIN_STEPS, trace=trace)
    return trace


def _t3_case(detectable_table, simulated, make_state, k_start):
    tbl = detectable_table
    rng = np.random.default_rng(3)
    sites = rng.choice(len(tbl), size=8, replace=False)
    truth, data, target = simulated(sites, seed=3, tbl=tbl)
    start = (sites[:4] if k_start == "below"
             else np.concatenate([sites, rng.choice(
                 np.setdiff1d(np.arange(len(tbl)), sites), size=6, replace=False)]))
    return tbl, truth, data, target, start


def test_t3_recovers_the_number_of_spins(detectable_table, simulated, make_state,
                                         metrics):
    """From an under-specified start, births find the right dimension.

    Run against the detectable table, not the full one.  Below the detection
    floor a spurious spin barely changes the likelihood and is accepted about
    half the time, so k random-walks upward and the posterior mode measures
    identifiability rather than the sampler.  Measured (test-plan §5.6) with
    k_true = 8: the full table gives a mode of 17, this one gives 8.
    """
    tbl, truth, data, target, start = _t3_case(
        detectable_table, simulated, make_state, "below")
    m = metrics(_run_rjmcmc(tbl, target, start, make_state), truth, data,
                burn=BURN, tbl=tbl)
    assert m.k_mode == int(truth.k[0])


@pytest.mark.parametrize("k_start", ["above", "below"])
def test_t3_posterior_contains_the_truth(detectable_table, simulated, make_state,
                                         metrics, k_start):
    """Criterion B from either side: every simulated spin is in the posterior."""
    tbl, truth, data, target, start = _t3_case(
        detectable_table, simulated, make_state, k_start)
    m = metrics(_run_rjmcmc(tbl, target, start, make_state), truth, data,
                burn=BURN, tbl=tbl)
    assert m.R(100, 750) == pytest.approx(1.0)


def test_t3_over_specified_start_keeps_spurious_spins(
        detectable_table, simulated, make_state, metrics):
    """Documents a real limitation rather than demanding it be absent.

    Starting above the true dimension, the chain finds every true spin but
    never sheds the last few extras.  This is not burn-in: measured
    (test-plan §5.6) the posterior mode sits at 10 against a true 8 at 8,000,
    16,000 and 30,000 steps alike, while the same chain started below reaches
    exactly 8 at every length.  Dimension is multimodal and birth-death moves
    alone do not mix across it.

    If a change makes this pass with equality, dimension mixing has improved
    and this test should be tightened rather than deleted -- that is the
    outcome parallel tempering over a trans-dimensional block would buy.
    """
    tbl, truth, data, target, start = _t3_case(
        detectable_table, simulated, make_state, "above")
    m = metrics(_run_rjmcmc(tbl, target, start, make_state), truth, data,
                burn=BURN, tbl=tbl)
    assert m.k_mode >= int(truth.k[0])
    assert m.k_mode <= int(truth.k[0]) + 4


def test_t3_does_not_overfit(detectable_table, simulated, make_state, metrics):
    """With k free, more spins always fit better.

    A residual floor is the assertion that the prior carried by gamma is doing
    its job.  Without the combinatorial term in that kernel the birth ratio is
    about +5.9 on a table of thousands of sites -- a factor of 365 favouring
    every birth -- and k runs to k_max whatever the data says.
    """
    tbl, truth, data, target, start = _t3_case(
        detectable_table, simulated, make_state, "below")
    m = metrics(_run_rjmcmc(tbl, target, start, make_state), truth, data,
                burn=BURN, tbl=tbl)
    assert m.median_residual > 0.5
    assert m.k_posterior.max() < 32


# ═══════════════════════════════════════════════════════ T4 — tempering

def _run_sites(table, target, start, seed, n_steps, tempered, make_state, k_max=32):
    trace = Trace(n_sites=len(table), k_max=k_max, n_exp=1)
    walk = DiscreteLatticeWalk(NeighborIndex(table.positions, WALK_RADIUS))
    if tempered:
        ParallelTempering(
            Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)]),
            n_replicas=N_REPLICAS,
        ).run(make_state(start), target, np.random.default_rng(seed),
              n_steps=n_steps, trace=trace)
    else:
        RWMH(ParameterBlock("sites"), walk).run(
            make_state(start), target, np.random.default_rng(seed),
            n_steps=n_steps, trace=trace)
    return trace


def test_t4_tempering_beats_the_single_block_baseline(
        table, simulated, stratified_bath, make_state, metrics):
    """Tempering's claim is that it escapes local minima.

    Pooled over seeds, not asserted per seed.  Measured (test-plan §5.6): the
    benefit is real but variable -- one seed goes 4.80 sigma to 2.00, another
    shows nothing.  A single-seed strict comparison would be flaky, and picking
    the seed that shows the effect would be worse than flaky.

    The 100-750 kHz band is not used: both methods saturate near 1.0 there, so
    it has no headroom in which to show an improvement.
    """
    base_res, hot_res, base_R, hot_R = [], [], [], []
    for seed in (21, 22, 23):
        rng = np.random.default_rng(seed)
        sites = stratified_bath(rng)
        truth, data, target = simulated(sites, seed=seed)
        start = rng.choice(len(table), size=len(sites), replace=False)

        base = metrics(_run_sites(table, target, start, seed, LADDER_STEPS,
                                  False, make_state), truth, data, burn=BURN)
        hot = metrics(_run_sites(table, target, start, seed, LADDER_STEPS,
                                 True, make_state), truth, data, burn=BURN)
        base_res.append(base.median_residual)
        hot_res.append(hot.median_residual)
        base_R.append(base.R(25, 100))
        hot_R.append(hot.R(25, 100))

    assert np.mean(hot_res) < np.mean(base_res)
    assert np.mean(hot_R) > np.mean(base_R)


def test_t4_swap_acceptance_is_workable(table, simulated, stratified_bath,
                                        make_state):
    """A ladder nobody crosses is a ladder that does nothing.

    Rungs must diverge first: straight after expand_replicas every rung is
    identical, so an accepted swap leaves the state unchanged and cannot be
    told from a rejected one.  An earlier version of this test measured exactly
    that and counted zero swaps by construction.

    Measured rate here is about 0.06.  It is low because the swap draws a
    random pair rather than an adjacent one (spec §8.4, following the methods
    paper): only a third of draws on a six-rung ladder are adjacent, and wider
    gaps are almost never accepted.
    """
    sites = stratified_bath(np.random.default_rng(22))
    _, _, target = simulated(sites, seed=22)

    ladder = ParallelTempering(
        Schedule([Step(RWMH(ParameterBlock("sites"),
                            DiscreteLatticeWalk(
                                NeighborIndex(table.positions, WALK_RADIUS))), 1)]),
        n_replicas=N_REPLICAS)
    st = make_state(sites).expand_replicas(N_REPLICAS)
    rng = np.random.default_rng(22)
    for _ in range(200):
        st = ladder.step(st, target, rng)

    swaps = 0
    for _ in range(300):
        before = st.site_idx.copy()
        st = ladder.attempt_swap(st, target, rng)
        swaps += not np.array_equal(before, st.site_idx)
    assert 0.01 < swaps / 300 < 0.95


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
    ParallelTempering(inner, n_replicas=4).run(
        start, target, np.random.default_rng(23), n_steps=LADDER_STEPS, trace=trace)

    post = trace.lam[BURN:, 0]
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
    # Start at the true sites.  This rung isolates the relaxation mechanism;
    # starting from a wrong configuration measures relaxation and configuration
    # search at once, and the extra freedom lets the chain fit the data with
    # wrong sites, which is a different finding from the one being tested.
    # Measured (test-plan §5.6): from truth sites, 3.47 sigma constrained
    # against 1.71 relaxed; from perturbed sites the comparison inverts.
    start = make_state(sites)

    # Equal site-step budgets.  The relaxed schedule spends 4 of every 6 steps
    # on offsets, so an equal *total* budget would give it a third as many site
    # moves -- and it would lose on configuration search rather than on the
    # mechanism under test.  Measured: with that confound the relaxed arm reads
    # 19.70 sigma; with budgets equalised it reads 1.82 (test-plan §5.6).
    site_budget = 6000
    constrained = _trace(table)
    HybridDriver(Schedule([site_step])).run(
        start, target, np.random.default_rng(31), n_total=site_budget,
        trace=constrained)

    relaxed = _trace(table)
    HybridDriver(Schedule([
        site_step,
        Step(RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=1.0)), 4),
    ])).run(start, target, np.random.default_rng(31), n_total=site_budget * 3,
            trace=relaxed)

    # Equal posterior-sample counts, so "best residual" is a fair comparison:
    # the relaxed run is three times longer, and a minimum over more draws is
    # lower for free.
    hard = metrics(constrained, truth, data, burn=2500, stride=50)
    soft = metrics(relaxed, truth, data, burn=7500, stride=150)
    assert abs(len(soft.residual) - len(hard.residual)) <= 2

    # Compared at their best, not at their median.  The constrained model pins
    # the offsets at the prior mean, which is the best point estimate when the
    # likelihood barely constrains them; the relaxed model samples them, so its
    # typical draw is worse by construction.  The question is whether relaxing
    # lets the model reach a fit the constraint forbids.  Measured
    # (test-plan §5.6): 3.47 sigma constrained against 1.71 relaxed, with a
    # floor of 1.00 at the true offsets.
    assert soft.best_residual < hard.best_residual


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
    ])).run(make_state(sites), target, np.random.default_rng(32), n_total=6000)

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
        n_steps=5000, trace=trace)

    m = metrics(trace, truth, data, burn=2000)
    lo = m.predictive.min(axis=0)
    hi = m.predictive.max(axis=0)
    inside = np.mean((data.data_all >= lo - 4 * 0.002) &
                     (data.data_all <= hi + 4 * 0.002))
    assert inside > 0.9


# ---------------------------------------------------------------------------
# T7 — ensemble agreement detects trapped chains
# ---------------------------------------------------------------------------

ENSEMBLE_STEPS, ENSEMBLE_BURN, N_ENSEMBLES = 1200, 400, 8


def _ensemble_schedule(table, *, trans_dimensional):
    """The hybrid, with birth-death moves optionally removed.

    Removing them is the negative control: k cannot vary, so a diagnostic that
    reports disagreement in k must fall silent.
    """
    walk = DiscreteLatticeWalk(NeighborIndex(table.positions, radius=WALK_RADIUS))
    blocks = []
    if trans_dimensional:
        blocks.append(Step(RJMCMC(ParameterBlock("sites"),
                                  BirthDeathKernel(k_max=32)), 80))
    blocks.append(Step(ParallelTempering(
        Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)]),
        n_replicas=6), 160))
    return Schedule(blocks)


def _run_ensembles(table, target, make_state, *, k_values, trans_dimensional,
                   root_seed, n_ensembles=N_ENSEMBLES, n_steps=ENSEMBLE_STEPS):
    def build(sites):
        return make_state(sites, tbl=table)

    runner = EnsembleRunner(
        _ensemble_schedule(table, trans_dimensional=trans_dimensional),
        n_ensembles=n_ensembles, n_steps=n_steps, n_burn=ENSEMBLE_BURN,
        init=spread_across_k(build, k_values), init_name="spread_across_k")
    return runner.run(target, root_seed=root_seed)


def test_t7_agreement_fires_on_chains_trapped_at_different_dimensions(
        detectable_table, simulated, make_state):
    """The diagnostic must report the disagreement that is known to be there.

    §5.6 measured that chains reaching a given k from above and from below
    settle at different values and stay there.  Ensembles initialised across
    that split therefore *must* disagree, and `agreement()` must say so.  A
    convergence diagnostic that never reports non-convergence is
    indistinguishable from one that is not computed at all.

    Thresholds from §5.8: spread measured at 2 on each of three root seeds,
    R-hat on k at 1.94 to 2.39 against a conventional pass mark of 1.01.
    """
    tbl = detectable_table
    sites = np.sort(np.random.default_rng(7).choice(len(tbl), 6, replace=False))
    _truth, _data, target = simulated(sites, seed=7, tbl=tbl)

    result = _run_ensembles(tbl, target, make_state, k_values=(3, 9),
                            trans_dimensional=True, root_seed=2026)
    agreement = result.agreement()

    assert agreement.n_ensembles == N_ENSEMBLES
    assert agreement.k_mode_spread >= 1
    assert agreement.rhat["k"] > 1.5


def test_t7_agreement_falls_silent_when_dimension_cannot_move(
        detectable_table, simulated, make_state):
    """The negative control, and the reason the rung above means anything.

    With the trans-dimensional block removed, k is fixed by construction, so
    every ensemble must agree on it and R-hat must be undefined rather than
    large.  Measured: eight ensembles, modal k = 6 for all of them, spread 0,
    R-hat nan.

    Without this the fired diagnostic proves nothing: a statistic that always
    reports disagreement would pass the test above too.
    """
    tbl = detectable_table
    sites = np.sort(np.random.default_rng(7).choice(len(tbl), 6, replace=False))
    _truth, _data, target = simulated(sites, seed=7, tbl=tbl)

    result = _run_ensembles(tbl, target, make_state, k_values=(6,),
                            trans_dimensional=False, root_seed=2026,
                            n_ensembles=4, n_steps=600)
    agreement = result.agreement()

    modes = [int(np.bincount(np.asarray(t.k)).argmax()) for t in result.traces]
    assert modes == [6] * 4
    assert agreement.k_mode_spread == 0
    assert np.isnan(agreement.rhat["k"])


# ---------------------------------------------------------------------------
# T8 — the Wasserstein variant reduces exactly
# ---------------------------------------------------------------------------

T8_STEPS = 800


def _hybrid_for_t8(table):
    """RJMCMC, tempering and a cold site walk — every block at once.

    The point of running the full hybrid rather than a single block: the
    reduction has to survive replica expansion inside tempering and birth-death
    inside RJMCMC, not just a continuous update.
    """
    walk = DiscreteLatticeWalk(NeighborIndex(table.positions, radius=WALK_RADIUS))
    return Schedule([
        Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=32)), 40),
        Step(ParallelTempering(
            Schedule([Step(RWMH(ParameterBlock("sites"), walk), 1)]),
            n_replicas=6), 80),
        Step(RWMH(ParameterBlock("sites"), walk), 40),
    ])


def _t8_chain(table, target, start, likelihood, seed=17):
    schedule = _hybrid_for_t8(table)
    trace = Trace(n_sites=len(table), k_max=32, n_exp=1)
    HybridDriver(schedule).run(start, Target(target.expset, target.model,
                                             likelihood, target.site_table),
                               np.random.default_rng(seed),
                               n_total=T8_STEPS, trace=trace)
    return trace


def test_t8_zero_weight_reproduces_least_squares_through_the_whole_hybrid(
        detectable_table, simulated, make_state):
    """At weight 0 the sampler cannot tell the two likelihoods apart.

    Identical, not merely similar, and through every block: birth-death moves,
    a six-rung tempering ladder and a cold site walk.  Exact reduction is the
    entire safety argument for offering the variant — a user who leaves the
    penalty off must get the calibrated sampler, bit for bit.

    The unit suite checks the same property on a four-site table with one
    continuous block.  This rung checks that it survives the composition.
    """
    tbl = detectable_table
    sites = np.sort(np.random.default_rng(7).choice(len(tbl), 6, replace=False))
    _truth, _data, target = simulated(sites, seed=7, tbl=tbl)
    start = make_state(np.random.default_rng(31).choice(len(tbl), 3, replace=False),
                       tbl=tbl)

    plain = _t8_chain(tbl, target, start, GaussianL2())
    reduced = _t8_chain(tbl, target, start, WassersteinL2(weight=0.0))

    for field in ("site_idx", "k", "lam", "dA_par", "dA_perp", "log_prob"):
        assert np.array_equal(np.asarray(getattr(plain, field)),
                              np.asarray(getattr(reduced, field))), field
    assert list(plain.algorithm) == list(reduced.algorithm)


def test_t8_a_calibrated_weight_actually_changes_the_chain(
        detectable_table, simulated, make_state):
    """Above the acting threshold the penalty must do something.

    Without this the parameter is decorative: below some weight the chain is
    bitwise identical to the least-squares one, so a variant that never
    diverged would be indistinguishable from one that was never applied.

    The weight is 1e5 rather than the 1e3 of §5.7 because **the threshold is a
    property of the dataset, not only of the sampler**.  Measured on this rung's
    own data with this schedule: 1e3 never diverges in 800 steps, 1e4 first
    diverges at step 11, 1e5 at step 4.  §5.7 saw 1e3 act at step 44 on a
    different noise draw.  Two decades of margin, not one.
    """
    tbl = detectable_table
    sites = np.sort(np.random.default_rng(7).choice(len(tbl), 6, replace=False))
    _truth, _data, target = simulated(sites, seed=7, tbl=tbl)
    start = make_state(np.random.default_rng(31).choice(len(tbl), 3, replace=False),
                       tbl=tbl)

    plain = _t8_chain(tbl, target, start, GaussianL2())
    penalised = _t8_chain(tbl, target, start, WassersteinL2(weight=1e5))

    assert not np.array_equal(np.asarray(plain.site_idx),
                              np.asarray(penalised.site_idx))


# ---------------------------------------------------------------------------
# T9 — adaptive design beats a uniform grid at equal time
# ---------------------------------------------------------------------------

# One round of the design loop on the detectable table: fit a sparse, noisy
# first experiment, design a follow-up of the same total time two ways,
# measure each at the truth, refit on both.  Piloted: this first experiment
# leaves the posterior ambiguous but at the noise -- effective size 5.5,
# residual 1.00 sigma, three of six spins at R_i <= 0.16.  The protocol and its
# seed sweep live in scripts/calibrate_t9.py.

T9_SEED = 7
T9_FIRST_POINTS, T9_NOISE = 20, 0.02
T9_BUDGET = float(T9_FIRST_POINTS)
T9_DENSE = np.linspace(0.0, 8e-3, 250, endpoint=False) + 8e-3 / 250
T9_ENSEMBLES, T9_STEPS, T9_BURN = 4, 1200, 400

#: Paired margin on mean R_i, set from scripts/calibrate_t9.py.  None until
#: that run is recorded in test-plan Sec. 5.11.
T9_ADAPTIVE_OVER_UNIFORM = None


def _calibrated(value, name):
    if value is None:
        raise NotImplementedError(
            f"{name} is not calibrated yet: run scripts/calibrate_t9.py and "
            "record it in docs/test-plan.md Sec. 5.11")
    return value


def _t9_grid(n):
    return np.linspace(0.0, 8e-3, n, endpoint=False) + 8e-3 / n


def _t9_state(tbl, sites, n_exp, sigma=LIK_SIGMA):
    return State.from_sites(np.sort(np.asarray(list(sites), int)),
                            n_sites=len(tbl), n_exp=n_exp,
                            lam=np.full((1, n_exp), LAM),
                            n_stretch=np.ones((1, n_exp)),
                            sigma=np.full((1, n_exp), sigma), k_max=32)


def _t9_fit(tbl, model, data, root_seed):
    n_exp = data.n_experiments
    runner = EnsembleRunner(
        _ensemble_schedule(tbl, trans_dimensional=True),
        n_ensembles=T9_ENSEMBLES, n_steps=T9_STEPS, n_burn=T9_BURN,
        init=spread_across_k(lambda s: _t9_state(tbl, s, n_exp), (3, 9)),
        init_name="spread_across_k")
    return runner.run(Target(data, model, GaussianL2(), tbl),
                      root_seed=root_seed).pooled


def _t9_candidates():
    windows = np.array_split(T9_DENSE, 5)
    return [Experiment(tau=w, n_pulses=N_PULSES, b_z=B_Z) for w in windows] + [
        Experiment(tau=T9_DENSE, n_pulses=N_PULSES, b_z=B_Z)]


def _t9_unit_cost(experiment):
    """Every repetition costs the same.

    T9 was set up, and piloted, with the budget counted in repetitions: the
    follow-up gets as many as the first experiment had points.  Keeping that
    here keeps the pilot's numbers meaningful; the designer's own default
    charges 2 N tau.
    """
    return np.ones(len(experiment.tau))


def _t9_uniform():
    """The control: the budget spread evenly over the dense grid.  Needs no
    posterior."""
    idx = np.round(np.linspace(0, T9_DENSE.size - 1, T9_FIRST_POINTS)).astype(int)
    return Experiment(tau=T9_DENSE[idx], n_pulses=N_PULSES, b_z=B_Z,
                      sigma=T9_NOISE,
                      weight=np.full(idx.size, T9_BUDGET / idx.size))


def _t9_adaptive(tbl, model, particles, measured, rng):
    """The designer's answer, as a DesignResult."""
    return ExperimentDesigner(
        model, tbl, measured, utility=ExpectedInformationGain(n_draws=256),
        cost=_t9_unit_cost).propose(particles, _t9_candidates(),
                                    budget=T9_BUDGET, rng=rng, exclude=measured)


def _t9_design(kind, tbl, model, particles, measured, rng):
    if kind == "uniform":
        return _t9_uniform()
    proposed = _t9_adaptive(tbl, model, particles, measured, rng).experiment
    assert proposed is not None, "the designer found nothing to tell apart"
    return proposed


_T9_CACHE = {}


def _t9_round(tbl, model):
    """Mean R_i after refitting on each design; computed once per session."""
    if "round" in _T9_CACHE:
        return _T9_CACHE["round"]
    sites = np.sort(np.random.default_rng(T9_SEED).choice(len(tbl), 6,
                                                          replace=False))
    truth = _t9_state(tbl, sites, 1, sigma=T9_NOISE)
    first = simulate_dataset(
        truth, ExperimentSet([Experiment(tau=_t9_grid(T9_FIRST_POINTS),
                                         n_pulses=N_PULSES, b_z=B_Z)]),
        tbl, model, sigma=T9_NOISE, rng=np.random.default_rng(T9_SEED + 9000))
    particles = ParticleSet.from_trace(_t9_fit(tbl, model, first, T9_SEED), tbl,
                                       stride=10)
    out = {}
    for kind in ("adaptive", "uniform"):
        proposal = _t9_design(kind, tbl, model, particles, first,
                              np.random.default_rng(T9_SEED + 1))
        followup = simulate_dataset(truth, ExperimentSet([proposal]), tbl, model,
                                    sigma=T9_NOISE,
                                    rng=np.random.default_rng(T9_SEED + 5000))
        combined = ExperimentSet(first.experiments + followup.experiments)
        refit = _t9_fit(tbl, model, combined, T9_SEED + 100)
        after = summarize(refit, combined, tbl, model, reference=sites, burn=0,
                          stride=50, noise=T9_NOISE, tol=0.1)
        out[kind] = float(np.mean(after.R_i))
    _T9_CACHE["round"] = out
    return out


def test_t9_adaptive_design_beats_uniform_at_equal_time(detectable_table,
                                                        model):
    """The claim the design engine rests on.

    Equal *time*, not equal point count: the uniform follow-up spends the same
    budget evenly over the dense grid, so a design that won by measuring more
    would not be a design.  Both refits share their starts and root seed, so
    the comparison is paired.
    """
    r = _t9_round(detectable_table, model)
    margin = _calibrated(T9_ADAPTIVE_OVER_UNIFORM, "T9_ADAPTIVE_OVER_UNIFORM")
    assert r["adaptive"] - r["uniform"] >= margin, r


def test_t9_a_collapsed_posterior_is_declined(detectable_table, model):
    """The degenerate control.

    The plan asks that adaptive and uniform be indistinguishable when the
    posterior has collapsed.  The designer does something stronger: it
    declines, returning a result with no experiment, rather than an arbitrary
    design that would then have to be shown equal to uniform.  The uniform
    design needs no posterior and is still built.
    """
    tbl = detectable_table
    sites = np.sort(np.random.default_rng(T9_SEED).choice(len(tbl), 6,
                                                          replace=False))
    one = _t9_state(tbl, sites, 1, sigma=T9_NOISE)
    collapsed = ParticleSet(one.site_idx, one.k, [1.0], one.dA_par, one.dA_perp,
                            one.lam, one.n_stretch, one.sigma, len(tbl), 32)
    measured = ExperimentSet([Experiment(tau=_t9_grid(T9_FIRST_POINTS),
                                         n_pulses=N_PULSES, b_z=B_Z)])
    result = _t9_adaptive(tbl, model, collapsed, measured,
                          np.random.default_rng(0))
    assert result.experiment is None and not result.distinguishable
    assert _t9_uniform().weight.sum() == pytest.approx(T9_BUDGET)
