"""Measurement time that depends on the point: the sequence-duration cost.

Two things carry this file.  The **reduction**: a selector given no cost, or
a cost of 1 everywhere, gives exactly what it gave before costs existed.  And
the **accounting**: with a cost, weights still mean relative repetitions, the
budget is ``sum_j w_j c_j``, and information is bought per unit time, so an
expensive point has to earn its cost.  The designer always charges: its
default cost is the free evolution of the sequence, ``2 N tau``.

The cost model itself is pinned to the forward model's tau convention: the
free evolution of CPMG-N is 2 N tau, because tau is half the pulse spacing in
the Taminiau form -- checked here against where the model puts a dip.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    InformationDensity,
    ParticleSet,
    SequenceDuration,
    StretchedExponential,
    single_spin_modulation,
    to_angular,
)

from .design_doubles import EvenSpread, TotalDensity

W3 = np.array([0.5, 0.3, 0.2])
SIGMA = 0.05
N_GRID = 20
B_Z = 311.0


def rng(seed=0):
    return np.random.default_rng(seed)


def scene(seed=0):
    P = np.zeros((3, N_GRID))
    P[:, 10:] = rng(seed).normal(0, 0.05, (3, N_GRID - 10))
    return P


def ladder_of_density(a):
    a = np.asarray(a, dtype=float)
    return np.vstack([np.zeros_like(a), 2 * a]), np.array([0.5, 0.5])


COSTLY = np.linspace(1.0, 4.0, N_GRID)

SELECTORS = {
    "density": lambda: InformationDensity(),
    "density_unpruned": lambda: InformationDensity(power=1.0, prune_fraction=0.0),
}


# --------------------------------------------------------------------------
# SequenceDuration
# --------------------------------------------------------------------------


def test_duration_is_overhead_plus_two_n_tau():
    exp = Experiment(tau=np.array([1e-3, 2e-3]), n_pulses=16, b_z=B_Z)
    np.testing.assert_allclose(SequenceDuration(5e-3)(exp),
                               [5e-3 + 32e-3, 5e-3 + 64e-3])


def test_with_no_overhead_given_the_duration_is_two_n_tau():
    """The default: the free evolution of the sequence and nothing else."""
    exp = Experiment(tau=np.array([1e-3, 2e-3]), n_pulses=16, b_z=B_Z)
    assert SequenceDuration().overhead == 0.0
    np.testing.assert_allclose(SequenceDuration()(exp), 2.0 * 16 * exp.tau)


def test_duration_grows_linearly_with_pulse_number():
    tau = np.array([1e-3])
    d = {n: SequenceDuration(0.0)(Experiment(tau=tau, n_pulses=n, b_z=B_Z))[0]
         for n in (4, 8, 16, 32, 64)}
    assert d[64] == pytest.approx(16 * d[4])


def test_evolution_per_pulse_is_adjustable():
    exp = Experiment(tau=np.array([1e-3]), n_pulses=8, b_z=B_Z)
    assert SequenceDuration(0.0, evolution_per_pulse=1.0)(exp)[0] == (
        pytest.approx(8e-3))


@pytest.mark.parametrize("kw", [{"overhead": -1e-3}, {"overhead": np.nan},
                                {"overhead": 0.0, "evolution_per_pulse": 0.0}])
def test_duration_rejects_bad_parameters(kw):
    with pytest.raises(ValueError):
        SequenceDuration(**kw)


def test_the_two_n_tau_convention_matches_the_forward_model():
    """tau is half the pi-pulse spacing: a weak spin's first dip sits at
    pi / (2 omega_L + A_par), not at twice that.  If this moved, the factor of
    2 in the duration would be wrong."""
    gyro = 6.7283
    tau = np.linspace(0.05e-3, 2e-3, 200_001)
    M = single_spin_modulation(tau, 20.0, 15.0, 16, B_Z, gyro)
    predicted = np.pi / (2 * gyro * B_Z + to_angular(20.0))
    assert tau[np.argmin(M)] == pytest.approx(predicted, rel=0.02)


# --------------------------------------------------------------------------
# selectors: the reduction and the accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SELECTORS)
def test_unit_cost_reproduces_no_cost_exactly(name):
    a_idx, a_w = SELECTORS[name]().select(scene(), W3, SIGMA, 4.0, rng(2))
    b_idx, b_w = SELECTORS[name]().select(scene(), W3, SIGMA, 4.0, rng(2),
                                          cost=np.ones(N_GRID))
    np.testing.assert_array_equal(a_idx, b_idx)
    np.testing.assert_array_equal(a_w, b_w)


@pytest.mark.parametrize("name", SELECTORS)
def test_the_budget_is_spent_in_time(name):
    idx, weight = SELECTORS[name]().select(scene(), W3, SIGMA, 6.0, rng(),
                                           cost=COSTLY)
    assert np.sum(np.asarray(weight) * COSTLY[idx]) == pytest.approx(6.0)
    assert np.all(np.asarray(weight) > 0)


@pytest.mark.parametrize("name", SELECTORS)
def test_a_constant_cost_divides_the_weights(name):
    """Every point three times as expensive buys a third of the repetitions
    and changes nothing else -- the information per unit time keeps its
    order."""
    a_idx, a_w = SELECTORS[name]().select(scene(), W3, SIGMA, 6.0, rng())
    b_idx, b_w = SELECTORS[name]().select(scene(), W3, SIGMA, 6.0, rng(),
                                          cost=np.full(N_GRID, 3.0))
    np.testing.assert_array_equal(a_idx, b_idx)
    np.testing.assert_allclose(b_w, np.asarray(a_w) / 3)


@pytest.mark.parametrize("bad", [np.zeros(N_GRID), -np.ones(N_GRID),
                                 np.full(N_GRID, np.nan), np.ones(N_GRID - 1)])
@pytest.mark.parametrize("name", SELECTORS)
def test_a_bad_cost_raises(name, bad):
    with pytest.raises(ValueError):
        SELECTORS[name]().select(scene(), W3, SIGMA, 4.0, rng(), cost=bad)


def test_density_allocates_by_information_per_unit_time():
    """Densities 1, 4, 16 at costs 1, 1, 4: rates 1, 4, 4; power 1/2 gives
    time in proportion 1, 2, 2, so a budget of 5 is times 1, 2, 2 and
    weights 1, 2, 0.5."""
    P, w = ladder_of_density([1.0, 2.0, 4.0])
    idx, weight = InformationDensity().select(P, w, 1.0, 5.0, rng(),
                                              cost=np.array([1.0, 1.0, 4.0]))
    assert list(idx) == [0, 1, 2]
    np.testing.assert_allclose(weight, [1.0, 2.0, 0.5])


def test_an_expensive_point_must_earn_its_cost():
    """Two points equally informative per repetition; one costs ten times as
    much.  It is given less of the time, and far fewer repetitions."""
    P, w = ladder_of_density([1.0, 1.0])
    idx, weight = InformationDensity().select(P, w, 1.0, 1.0, rng(),
                                              cost=np.array([10.0, 1.0]))
    assert list(idx) == [0, 1]
    time = np.asarray(weight) * np.array([10.0, 1.0])
    assert time[0] < time[1]
    assert weight[0] < weight[1] / 10


def test_a_costly_enough_point_is_dropped():
    """At a hundred times the cost its share falls under the pruning cut."""
    P, w = ladder_of_density([1.0, 1.0])
    idx, weight = InformationDensity(prune_fraction=0.2).select(
        P, w, 1.0, 1.0, rng(), cost=np.array([100.0, 1.0]))
    assert list(idx) == [1]
    assert weight[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the designer
# --------------------------------------------------------------------------

TAU = np.linspace(1e-4, 8e-3, 20)


@pytest.fixture
def setup(tiny_site_table):
    measured = ExperimentSet([
        Experiment(tau=np.linspace(2e-4, 6e-3, 12), n_pulses=8, b_z=B_Z),
        Experiment(tau=np.linspace(2e-4, 6e-3, 12), n_pulses=16, b_z=B_Z),
    ])
    n, k_max = 3, 8
    site_idx = np.full((n, k_max), -1)
    for i, c in enumerate([[0, 2], [0], [2]]):
        site_idx[i, : len(c)] = c
    particles = ParticleSet(
        site_idx=site_idx, k=np.array([2, 1, 1]), weight=W3,
        dA_par=np.zeros((n, k_max)), dA_perp=np.zeros((n, k_max)),
        lam=np.full((n, 2), 3e-3), n_stretch=np.ones((n, 2)),
        sigma=np.full((n, 2), 0.002), n_sites=4, k_max=k_max)
    model = AnalyticCCE1(StretchedExponential())
    return tiny_site_table, model, measured, particles


def unit_cost(experiment):
    return np.ones(len(experiment.tau))


def designer(setup, cost=unit_cost, selector=None):
    """Deterministic utility; unit cost unless the test is about another."""
    table, model, measured, _ = setup
    return ExperimentDesigner(model, table, measured, utility=TotalDensity(),
                              selector=selector, cost=cost, min_gain=0.0)


def test_the_designer_charges_two_n_tau_unless_told_otherwise(setup):
    """Leaving the cost out is not the same as charging nothing."""
    *_, particles = setup
    cands = [Experiment(tau=TAU, n_pulses=16, b_z=B_Z),
             Experiment(tau=TAU, n_pulses=8, b_z=B_Z)]
    default = designer(setup, cost=None).rank(particles, cands, budget=4.0,
                                              rng=rng())
    explicit = designer(setup, cost=lambda e: 2.0 * e.n_pulses * e.tau).rank(
        particles, cands, budget=4.0, rng=rng())
    uncharged = designer(setup).rank(particles, cands, budget=4.0, rng=rng())
    np.testing.assert_allclose(default, explicit, rtol=1e-12)
    assert not np.allclose(default, uncharged)


def test_the_same_delays_cost_twice_as_much_at_twice_the_pulses(setup):
    """Equal time buys half the repetitions at N = 16 that it buys at N = 8,
    delay for delay."""
    *_, particles = setup
    d = ExperimentDesigner(setup[1], setup[0], setup[2], utility=TotalDensity(),
                           selector=EvenSpread(20), min_gain=0.0)
    at_16, at_8 = (d.propose(particles, [Experiment(tau=TAU, n_pulses=n,
                                                    b_z=B_Z)], budget=4.0,
                             rng=rng()).experiment for n in (16, 8))
    np.testing.assert_allclose(at_16.weight, at_8.weight / 2, rtol=1e-12)


def test_doubling_every_cost_halves_predictive_variance(setup):
    """Half the repetitions, twice the variance per point."""
    *_, particles = setup
    cands = [Experiment(tau=TAU, n_pulses=16, b_z=B_Z)]
    a = designer(setup).rank(particles, cands, budget=4.0, rng=rng())[0]
    b = designer(setup, cost=lambda e: np.full(len(e.tau), 2.0)).rank(
        particles, cands, budget=4.0, rng=rng())[0]
    assert b == pytest.approx(a / 2)


def test_a_costlier_sequence_is_penalised_in_the_ranking(setup):
    """The same candidate, costed per pulse number: at equal time the longer
    sequence buys fewer repetitions.  With a cost that charges N = 16 four
    times what it charges N = 8 per point, a candidate's score at N = 16 is a
    quarter of what it would be uncosted."""
    *_, particles = setup
    cands = [Experiment(tau=TAU, n_pulses=16, b_z=B_Z)]
    plain = designer(setup).rank(particles, cands, budget=4.0, rng=rng())[0]
    costed = designer(setup, cost=lambda e: np.full(len(e.tau), e.n_pulses / 4)
                      ).rank(particles, cands, budget=4.0, rng=rng())[0]
    assert costed == pytest.approx(plain / 4)


def test_proposal_spends_the_budget_in_sequence_time(setup):
    *_, particles = setup
    cost = SequenceDuration(5e-3)
    out = designer(setup, cost=cost).propose(
        particles, [Experiment(tau=TAU, n_pulses=16, b_z=B_Z)], budget=2.0,
        rng=rng()).experiment
    assert np.sum(out.weight * cost(out)) == pytest.approx(2.0)


def test_an_overhead_is_added_to_every_repetition(setup):
    """With an even spread, each kept delay gets the same time whatever the
    overhead; the overhead only changes how many repetitions that buys."""
    *_, particles = setup
    cost = SequenceDuration(5e-3)
    out = designer(setup, cost=cost, selector=EvenSpread(5)).propose(
        particles, [Experiment(tau=TAU, n_pulses=16, b_z=B_Z)], budget=2.0,
        rng=rng()).experiment
    np.testing.assert_allclose(out.weight * cost(out), [0.4] * 5)
    np.testing.assert_allclose(cost(out), 5e-3 + 2.0 * 16 * out.tau)


def test_a_cost_that_returns_the_wrong_shape_raises(setup):
    *_, particles = setup
    with pytest.raises(ValueError):
        designer(setup, cost=lambda e: np.ones(3)).rank(
            particles, [Experiment(tau=TAU, n_pulses=16, b_z=B_Z)], budget=4.0,
            rng=rng())
