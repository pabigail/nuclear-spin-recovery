"""Point selectors: which points of a candidate to measure, and for how long.

The contract every selector shares is the one the designer and the likelihood
rely on: distinct points in increasing order, positive weights, and a total
that equals the budget -- because the budget is total measurement time, and
designs are compared at equal time (docs/phase-5-plan.md Sec. 6, question 2).

Beyond that, each selector has a known answer on a constructed case:
UniformThinning's spacing, InformationDensity's allocation by hand, and
GreedyUtility under predictive variance, which is additive and so reduces to
taking the top points by density.  Greedy under EIG is Monte Carlo; its
thresholds were measured on a reference implementation first, in the scene
below:

  first pick informative, 64 draws          30/30 scenes
  all four picks informative, 64 draws      26/30 -- not asserted
  all four picks informative, 1024 draws    30/30

docs/phase-5-plan.md, unit 5c.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    ExpectedInformationGain,
    GreedyUtility,
    InformationDensity,
    LeastInformative,
    NothingToLearn,
    PointSelector,
    PredictiveVariance,
    UniformThinning,
    information_density,
)

W3 = np.array([0.5, 0.3, 0.2])
SIGMA = 0.05
N_GRID = 20
INFORMATIVE = 10          # points [10, 20) carry the disagreement


def rng(seed=0):
    return np.random.default_rng(seed)


def scene(seed=0):
    """Three particles that agree on the first half of the grid and disagree,
    at about the noise level, on the second."""
    P = np.zeros((3, N_GRID))
    P[:, INFORMATIVE:] = rng(seed).normal(0, 0.05, (3, N_GRID - INFORMATIVE))
    return P


def collapsed():
    """One particle: nothing to learn."""
    return scene()[:1]


def ladder_of_density(a):
    """Two equal-weight particles at 0 and 2a: variance a^2 at each point, so
    at unit noise the information density is exactly a^2."""
    a = np.asarray(a, dtype=float)
    return np.vstack([np.zeros_like(a), 2 * a]), np.array([0.5, 0.5])


# Factories, not instances: constructors are stubs until 5c is implemented,
# and an instance built at collection time would error rather than fail.
SELECTORS = {
    "uniform": lambda: UniformThinning(4),
    "density": lambda: InformationDensity(),
    "greedy_pv": lambda: GreedyUtility(PredictiveVariance(), 4),
    "greedy_eig": lambda: GreedyUtility(ExpectedInformationGain(), 4),
    "least": lambda: LeastInformative(4),
}
DETERMINISTIC = ["uniform", "density", "greedy_pv", "least"]


# --------------------------------------------------------------------------
# the contract every selector keeps
# --------------------------------------------------------------------------


def test_point_selector_is_abstract():
    with pytest.raises(TypeError):
        PointSelector()


def test_selectors_are_point_selectors():
    for cls in (UniformThinning, InformationDensity, GreedyUtility,
                LeastInformative):
        assert issubclass(cls, PointSelector)


def test_nothing_to_learn_is_a_value_error():
    assert issubclass(NothingToLearn, ValueError)


@pytest.mark.parametrize("name", SELECTORS)
def test_indices_are_distinct_sorted_and_in_range(name):
    idx, _ = SELECTORS[name]().select(scene(), W3, SIGMA, 4.0, rng())
    idx = np.asarray(idx)
    assert idx.dtype.kind == "i"
    assert np.all(np.diff(idx) > 0)
    assert idx.min() >= 0 and idx.max() < N_GRID


@pytest.mark.parametrize("name", SELECTORS)
def test_weights_are_positive_and_spend_the_budget(name):
    idx, weight = SELECTORS[name]().select(scene(), W3, SIGMA, 7.5, rng())
    assert len(weight) == len(idx)
    assert np.all(np.asarray(weight) > 0)
    assert np.sum(weight) == pytest.approx(7.5)


@pytest.mark.parametrize("name", SELECTORS)
@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_a_budget_that_is_not_positive_raises(name, budget):
    with pytest.raises(ValueError):
        SELECTORS[name]().select(scene(), W3, SIGMA, budget, rng())


@pytest.mark.parametrize("name", SELECTORS)
def test_a_weight_per_particle_is_required(name):
    with pytest.raises(ValueError):
        SELECTORS[name]().select(scene(), np.array([0.5, 0.5]), SIGMA, 4.0,
                                 rng())


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_the_budget_scales_the_weights_and_not_the_points(name):
    a_idx, a_w = SELECTORS[name]().select(scene(), W3, SIGMA, 4.0, rng())
    b_idx, b_w = SELECTORS[name]().select(scene(), W3, SIGMA, 12.0, rng())
    np.testing.assert_array_equal(a_idx, b_idx)
    np.testing.assert_allclose(b_w, 3 * np.asarray(a_w))


# --------------------------------------------------------------------------
# UniformThinning -- the control
# --------------------------------------------------------------------------


def test_uniform_spacing_on_a_uniform_grid():
    idx, weight = UniformThinning(4).select(np.zeros((2, 10)), [0.5, 0.5],
                                            SIGMA, 2.0, rng())
    assert list(idx) == [0, 3, 6, 9]
    np.testing.assert_allclose(weight, [0.5] * 4)


def test_uniform_can_take_every_point():
    idx, _ = UniformThinning(N_GRID).select(scene(), W3, SIGMA, 4.0, rng())
    assert list(idx) == list(range(N_GRID))


@pytest.mark.parametrize("n_points", [0, N_GRID + 1])
def test_uniform_point_count_must_fit_the_grid(n_points):
    with pytest.raises(ValueError):
        UniformThinning(n_points).select(scene(), W3, SIGMA, 4.0, rng())


def test_uniform_does_not_look_at_the_predictions():
    a, _ = UniformThinning(5).select(scene(0), W3, SIGMA, 4.0, rng())
    b, _ = UniformThinning(5).select(scene(9), W3, SIGMA, 4.0, rng())
    np.testing.assert_array_equal(a, b)


def test_uniform_works_on_a_collapsed_posterior():
    """T9's degenerate control needs the uniform design to keep working."""
    idx, weight = UniformThinning(4).select(collapsed(), [1.0], SIGMA, 4.0,
                                            rng())
    assert len(idx) == 4
    assert np.sum(weight) == pytest.approx(4.0)


# --------------------------------------------------------------------------
# InformationDensity -- the old rule, its numbers exposed
# --------------------------------------------------------------------------


def test_density_allocation_by_hand():
    """Densities 0, 1, 4, 16; power 1/2 gives 0, 1, 2, 4; the zero is
    dropped and the rest share the budget in proportion."""
    P, w = ladder_of_density([0.0, 1.0, 2.0, 4.0])
    idx, weight = InformationDensity().select(P, w, 1.0, 7.0, rng())
    assert list(idx) == [1, 2, 3]
    np.testing.assert_allclose(weight, [1.0, 2.0, 4.0])


def test_density_power_one_allocates_in_proportion_to_density():
    P, w = ladder_of_density([1.0, 2.0, 4.0])
    idx, weight = InformationDensity(power=1.0).select(P, w, 1.0, 21.0, rng())
    assert list(idx) == [0, 1, 2]
    np.testing.assert_allclose(weight, [1.0, 4.0, 16.0])


def test_density_prune_fraction_drops_small_allocations():
    """Allocations 1, 2, 4 at power 1/2; a fraction of 0.3 sets the bar at
    1.2, which drops the first and renormalises the rest."""
    P, w = ladder_of_density([1.0, 2.0, 4.0])
    idx, weight = InformationDensity(prune_fraction=0.3).select(P, w, 1.0, 6.0,
                                                                rng())
    assert list(idx) == [1, 2]
    np.testing.assert_allclose(weight, [2.0, 4.0])


def test_density_drops_zero_density_even_without_pruning():
    P, w = ladder_of_density([0.0, 1.0, 0.0, 2.0])
    idx, _ = InformationDensity(prune_fraction=0.0).select(P, w, 1.0, 3.0,
                                                           rng())
    assert list(idx) == [1, 3]


def test_density_spends_its_time_where_the_particles_disagree():
    idx, _ = InformationDensity().select(scene(), W3, SIGMA, 4.0, rng())
    assert np.all(np.asarray(idx) >= INFORMATIVE)


def test_density_gives_more_time_to_more_disagreement():
    P = scene()
    idx, weight = InformationDensity().select(P, W3, SIGMA, 4.0, rng())
    dens = information_density(P, W3, SIGMA)[idx]
    order = np.argsort(dens)
    assert np.all(np.diff(np.asarray(weight)[order]) >= 0)


def test_density_never_selects_an_unmeasurable_point():
    noise = np.full(N_GRID, SIGMA)
    noise[15] = np.inf
    idx, _ = InformationDensity(prune_fraction=0.0).select(scene(), W3, noise,
                                                           4.0, rng())
    assert 15 not in list(idx)


@pytest.mark.parametrize("kw", [{"power": -0.5}, {"prune_fraction": -0.1},
                                {"prune_fraction": 1.0}])
def test_density_rejects_bad_parameters(kw):
    with pytest.raises(ValueError):
        InformationDensity(**kw)


def test_density_on_a_collapsed_posterior_raises():
    with pytest.raises(NothingToLearn):
        InformationDensity().select(collapsed(), [1.0], SIGMA, 4.0, rng())


def test_density_on_identical_predictions_raises():
    P = np.tile(scene()[0], (3, 1))
    with pytest.raises(NothingToLearn):
        InformationDensity().select(P, W3, SIGMA, 4.0, rng())


# --------------------------------------------------------------------------
# GreedyUtility -- the direct comparison
# --------------------------------------------------------------------------


def test_greedy_predictive_variance_takes_the_top_points_by_density():
    """Predictive variance is a sum over points, so greedy is exact for it."""
    P = scene()
    idx, weight = GreedyUtility(PredictiveVariance(), 5).select(
        P, W3, SIGMA, 5.0, rng())
    top = np.sort(np.argsort(information_density(P, W3, SIGMA))[-5:])
    np.testing.assert_array_equal(idx, top)
    np.testing.assert_allclose(weight, [1.0] * 5)


def test_greedy_can_take_every_point():
    idx, _ = GreedyUtility(PredictiveVariance(), N_GRID).select(
        scene(), W3, SIGMA, 4.0, rng())
    assert list(idx) == list(range(N_GRID))


@pytest.mark.parametrize("n_points", [0, N_GRID + 1])
def test_greedy_point_count_must_fit_the_grid(n_points):
    with pytest.raises(ValueError):
        GreedyUtility(PredictiveVariance(), n_points).select(
            scene(), W3, SIGMA, 4.0, rng())


def test_greedy_eig_first_pick_is_informative():
    """At the default 64 draws: 30 of 30 scenes on the reference."""
    greedy = GreedyUtility(ExpectedInformationGain(), 1)
    for seed in range(30):
        idx, _ = greedy.select(scene(seed), W3, SIGMA, 4.0, rng(seed))
        assert idx[0] >= INFORMATIVE, f"scene {seed}"


def test_greedy_eig_stays_informative_with_enough_draws():
    """At 64 draws a late pick strays on 4 of 30 scenes, where estimator noise
    exceeds the marginal gain; at 1024, on none."""
    greedy = GreedyUtility(ExpectedInformationGain(n_draws=1024), 4)
    for seed in range(30):
        idx, _ = greedy.select(scene(seed), W3, SIGMA, 4.0, rng(seed))
        assert np.all(np.asarray(idx) >= INFORMATIVE), f"scene {seed}: {idx}"


def test_greedy_eig_is_deterministic_given_the_seed():
    greedy = GreedyUtility(ExpectedInformationGain(), 4)
    a, _ = greedy.select(scene(), W3, SIGMA, 4.0, rng(3))
    b, _ = greedy.select(scene(), W3, SIGMA, 4.0, rng(3))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("utility_cls", [PredictiveVariance,
                                         ExpectedInformationGain])
def test_greedy_on_a_collapsed_posterior_raises(utility_cls):
    with pytest.raises(NothingToLearn):
        GreedyUtility(utility_cls(), 4).select(collapsed(), [1.0], SIGMA, 4.0,
                                               rng())


# --------------------------------------------------------------------------
# LeastInformative -- the anti-design, T9's negative control
# --------------------------------------------------------------------------


def test_least_informative_takes_the_lowest_density_points():
    """Densities 16, 1, 9, 0, 4: the lowest two are indices 3 and 1."""
    P, w = ladder_of_density([4.0, 1.0, 3.0, 0.0, 2.0])
    idx, weight = LeastInformative(2).select(P, w, 1.0, 6.0, rng())
    assert list(idx) == [1, 3]
    np.testing.assert_allclose(weight, [3.0, 3.0])


def test_least_informative_spends_its_time_where_the_particles_agree():
    idx, _ = LeastInformative(4).select(scene(), W3, SIGMA, 4.0, rng())
    assert np.all(np.asarray(idx) < INFORMATIVE)


def test_least_informative_breaks_ties_by_index():
    idx, _ = LeastInformative(3).select(collapsed(), [1.0], SIGMA, 3.0, rng())
    assert list(idx) == [0, 1, 2]


def test_least_informative_works_on_a_collapsed_posterior():
    """A control; T9's degenerate case needs it to keep working."""
    idx, weight = LeastInformative(4).select(collapsed(), [1.0], SIGMA, 4.0,
                                             rng())
    assert len(idx) == 4
    assert np.sum(weight) == pytest.approx(4.0)


def test_least_informative_never_chooses_an_unmeasurable_point():
    """Infinite noise has zero density, the lowest there is -- which is exactly
    why it must be excluded rather than ranked."""
    noise = np.full(N_GRID, SIGMA)
    noise[:2] = np.inf
    idx, _ = LeastInformative(3).select(collapsed(), [1.0], noise, 3.0, rng())
    assert list(idx) == [2, 3, 4]


def test_least_informative_raises_if_too_few_points_are_measurable():
    noise = np.full(N_GRID, np.inf)
    noise[:3] = SIGMA
    with pytest.raises(ValueError):
        LeastInformative(4).select(scene(), W3, noise, 4.0, rng())


@pytest.mark.parametrize("n_points", [0, N_GRID + 1])
def test_least_informative_point_count_must_fit_the_grid(n_points):
    with pytest.raises(ValueError):
        LeastInformative(n_points).select(scene(), W3, SIGMA, 4.0, rng())
