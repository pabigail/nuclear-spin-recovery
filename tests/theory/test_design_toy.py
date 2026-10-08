"""The designer on the system it was worked out on.

notebooks/adaptive_design_prototype.py prototyped the design procedure on a
ten-site toy lattice, standalone, before it was moved into the package.  This
is that problem again, run through the package, with the prototype's answers
as the reference: the same pulse number, the same delays, a gain within the
Monte Carlo scatter of the prototype's, and a proposed experiment that, once
simulated at the truth, moves the weight onto the true set of sites.

The posterior is four untempered chains, pooled.  They disagree, which is
what gives a design something to resolve.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from nuclear_spin_recovery import (
    RJMCMC,
    RWMH,
    AnalyticCCE1,
    BirthDeathKernel,
    DecouplingScaling,
    DiscreteLatticeWalk,
    ExpectedInformationGain,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    GaussianL2,
    HybridDriver,
    InformationDensity,
    NeighborIndex,
    ParameterBlock,
    ParticleSet,
    Schedule,
    SiteScaledOffset,
    SiteTable,
    State,
    Step,
    StretchedExponential,
    Target,
    Trace,
    gyromagnetic_ratio,
    merge_traces,
    simulate_dataset,
)

pytestmark = pytest.mark.slow

CENTRES = np.array([
    (100.0, 100.0), (10.0, 10.0), (100.0, 50.0), (60.0, 80.0), (150.0, 60.0),
    (40.0, 30.0), (75.0, 120.0), (130.0, 110.0), (25.0, 60.0), (55.0, 45.0)])
N_SITES, K_MAX = len(CENTRES), 6
TRUE = {2: (5.0, 2.0), 3: (-4.0, 5.0), 7: (8.0, -6.0)}
TRUE_SITES = tuple(sorted(TRUE))
B_Z, GAMMA, DATA_NOISE, LIK_SIGMA = 311.0, 2.0 / 3.0, 0.002, 0.02
LAMBDA_4 = 10 * np.pi / (gyromagnetic_ratio("13C") * B_Z)     # ten signal periods
GRID = np.linspace(0.05e-3, 16e-3, 320)
PULSES = (4, 8, 16, 32, 64)


def _decay(n_pulses):
    return LAMBDA_4 * (n_pulses / 4.0) ** (GAMMA - 1.0)


@pytest.fixture(scope="module")
def toy():
    """Table, model, first experiment, pooled particles and the budget."""
    angle = np.linspace(0.0, 2.0 * np.pi, N_SITES, endpoint=False)
    positions = np.column_stack([2 * np.cos(angle), 2 * np.sin(angle),
                                 np.full(N_SITES, 2.0)])
    table = SiteTable(
        distance=np.linalg.norm(positions, axis=1), positions=positions,
        a_par=CENTRES[:, 0], a_perp=CENTRES[:, 1],
        isotope=np.array(["13C"] * N_SITES),
        gyro=np.full(N_SITES, gyromagnetic_ratio("13C")))
    model = AnalyticCCE1(StretchedExponential())

    def make_state(sites, offsets=None, n_pulses=4):
        state = State.from_sites(
            tuple(int(s) for s in sites), n_sites=N_SITES, n_exp=1,
            lam=np.full((1, 1), _decay(n_pulses)), n_stretch=np.ones((1, 1)),
            sigma=np.full((1, 1), LIK_SIGMA), k_max=K_MAX, site_memory=True)
        for slot, (d_par, d_perp) in enumerate(offsets or ()):
            state.set_offset(0, slot, 0, d_par)
            state.set_offset(0, slot, 1, d_perp)
        return state

    def truth(n_pulses):
        return make_state(TRUE_SITES, [TRUE[s] for s in TRUE_SITES], n_pulses)

    first = ExperimentSet([Experiment(tau=np.linspace(3.2e-5, 8e-3, 250),
                                      n_pulses=4, b_z=B_Z)])
    data = simulate_dataset(truth(4), first, table, model, sigma=DATA_NOISE,
                            rng=np.random.default_rng(3))
    target = Target(data, model, GaussianL2(), table)

    def chain(seed):
        schedule = Schedule([
            Step(RJMCMC(ParameterBlock("sites"), BirthDeathKernel(k_max=K_MAX)), 10),
            Step(RWMH(ParameterBlock("sites"),
                      DiscreteLatticeWalk(NeighborIndex(positions, 10.0))), 10),
            Step(RWMH(ParameterBlock("offsets"), SiteScaledOffset(
                1.0, table, fraction_par=0.1, fraction_perp=0.1,
                prior="flat")), 40),
        ])
        trace = Trace(n_sites=N_SITES, k_max=K_MAX, n_exp=1)
        HybridDriver(schedule).run(make_state((0,)), target,
                                   np.random.default_rng(seed), 12_000,
                                   trace=trace)
        return trace.discard_burn_in(3_000)

    pooled = merge_traces([chain(seed) for seed in range(4)])
    particles = ParticleSet.from_trace(pooled, table, stride=20, tol=3.0)
    first_time = float(np.sum(2.0 * 4 * first.experiments[0].tau))
    return {"table": table, "model": model, "first": first, "truth": truth,
            "particles": particles, "budget": 2e-4 * first_time}


def _designer(toy, **kwargs):
    return ExperimentDesigner(
        toy["model"], toy["table"], toy["first"],
        utility=ExpectedInformationGain(n_draws=1000),
        envelope=DecouplingScaling(GAMMA), **kwargs)


def _candidates(tau=GRID, pulses=PULSES):
    return [Experiment(tau=tau, n_pulses=n, b_z=B_Z, sigma=DATA_NOISE)
            for n in pulses]


def _site_sets(particles, weight):
    out = collections.Counter()
    for i in range(particles.n_particles):
        key = tuple(sorted(int(s) for s in
                           particles.site_idx[i, : particles.k[i]]))
        out[key] += weight[i]
    return out


def test_the_toy_posterior_is_the_prototypes(toy):
    """128 baths on 6 sets of sites, the true set holding 47% of the weight."""
    particles = toy["particles"]
    sets = _site_sets(particles, particles.weight)
    assert particles.n_particles == 128
    assert len(sets) == 6
    assert sets[TRUE_SITES] == pytest.approx(0.474, abs=0.001)


def test_the_prototypes_design_is_reproduced(toy):
    """With the prototype's pruning cut-off: CPMG-64 at the same twenty delays.

    The delays are chosen deterministically and must match exactly.  The gain
    is a Monte Carlo estimate drawn differently from the prototype's, which
    scored 3.22; the two agree within their scatter.
    """
    result = _designer(toy, selector=InformationDensity(prune_fraction=0.2)
                       ).propose(toy["particles"], _candidates(),
                                 budget=toy["budget"],
                                 rng=np.random.default_rng(0))
    assert [d.tau.size for d in result.designs] == [180, 100, 24, 30, 20]
    out = result.experiment
    assert out.n_pulses == 64 and len(out) == 20
    assert out.tau.min() == pytest.approx(0.60e-3)
    assert out.tau.max() == pytest.approx(6.15e-3)
    assert np.sum(out.weight * 2.0 * 64 * out.tau) == pytest.approx(toy["budget"])
    assert result.gain == pytest.approx(3.22, abs=0.2)
    gains = [d.gain for d in result.designs]
    assert gains[0] < gains[1] < min(gains[2:]), "4 and 8 pulses trail"


def test_the_default_cut_off_spreads_thinner_and_gains_less(toy):
    """Measured, not assumed: at this budget the default pruning keeps over a
    hundred delays and gains about 0.6 nats less than a cut-off of 0.2."""
    default = _designer(toy).propose(toy["particles"], _candidates(),
                                     budget=toy["budget"],
                                     rng=np.random.default_rng(0))
    tighter = _designer(toy, selector=InformationDensity(prune_fraction=0.2)
                        ).propose(toy["particles"], _candidates(),
                                  budget=toy["budget"],
                                  rng=np.random.default_rng(0))
    assert len(default.experiment) > 100
    assert default.experiment.n_pulses == 32
    assert tighter.gain - default.gain > 0.3


def test_the_proposed_experiment_moves_the_weight_onto_the_truth(toy):
    """Simulate the design at the true bath and reweight the baths by it."""
    particles = toy["particles"]
    result = _designer(toy, selector=InformationDensity(prune_fraction=0.2)
                       ).propose(particles, _candidates(), budget=toy["budget"],
                                 rng=np.random.default_rng(0))
    out, design = result.experiment, result.chosen
    bare = ExperimentSet([Experiment(tau=out.tau, n_pulses=out.n_pulses,
                                     b_z=B_Z)])
    clean = toy["model"].coherence(toy["truth"](out.n_pulses), bare,
                                   toy["table"])[0]
    noise = out.sigma / np.sqrt(out.weight)
    measured = clean + np.random.default_rng(21).normal(0.0, noise)
    log_like = -0.5 * np.sum(
        ((measured - design.signals[:, design.index]) / noise) ** 2, axis=1)
    updated = particles.weight * np.exp(log_like - log_like.max())
    updated /= updated.sum()

    before = _site_sets(particles, particles.weight)[TRUE_SITES]
    after = _site_sets(particles, updated)[TRUE_SITES]
    assert before < 0.5
    assert after > 0.95


def test_delays_too_short_to_show_a_dip_are_declined(toy):
    short = np.linspace(0.02e-3, 0.25e-3, 40)
    result = _designer(toy).propose(toy["particles"],
                                    _candidates(short, pulses=(4, 8)),
                                    budget=toy["budget"],
                                    rng=np.random.default_rng(0))
    assert result.experiment is None
    assert abs(result.gain) < 0.01


def test_more_pulses_always_win_when_time_is_not_charged(toy):
    """Counted in repetitions the candidates are ordered by pulse number at
    every budget; counted in time they are not.  The second half is what the
    mandatory time cost exists for."""
    designer = ExperimentDesigner(
        toy["model"], toy["table"], toy["first"],
        utility=ExpectedInformationGain(n_draws=400),
        envelope=DecouplingScaling(GAMMA))
    cands = _candidates()
    own = np.array([designer.cost_of(c).sum() for c in cands])
    assert own / own[0] == pytest.approx([1, 2, 4, 8, 16])
    multiples = np.array([1e-5, 1e-4])
    by_repetitions = designer.gain_curve(toy["particles"], cands,
                                         multiples[:, None] * own[None, :])
    assert np.all(np.diff(by_repetitions, axis=1) > 0)
    by_time = designer.gain_curve(toy["particles"], cands, [toy["budget"]])[0]
    assert not np.all(np.diff(by_time) > 0), by_time
