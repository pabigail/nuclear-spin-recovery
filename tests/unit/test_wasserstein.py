"""The optimal-transport likelihood variant.

Most of this file is about **normalisation**, because a Wasserstein distance is
defined between probability measures and a coherence signal is not one. Three
separate normalisations have to hold or the quantity is not a distance between
distributions at all:

- the weights carry unit mass, so rescaling either signal changes nothing;
- the result is divided by the tau span, so it does not depend on whether tau
  was recorded in ms or us;
- negative weights, which noisy data above unit coherence would produce, are
  repaired rather than transported.

The rest pins the exact reduction at zeta = 0, which is the whole safety
argument for adding the term at all.

Spec Sec. 7.1; docs/phase-4-plan.md unit 4d.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from nuclear_spin_recovery import (
    RWMH,
    AnalyticCCE1,
    ContinuousReflected,
    Experiment,
    ExperimentSet,
    GaussianL2,
    Likelihood,
    ParameterBlock,
    State,
    StretchedExponential,
    Target,
    Trace,
    WassersteinL2,
    signal_measure,
    simulate_dataset,
    wasserstein_signal_distance,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

TAU = np.linspace(0.0, 1.0, 201)


def bump(centre, width=0.05, tau=TAU):
    """A localised feature, as a coherence dip."""
    return 1.0 - np.exp(-0.5 * ((tau - centre) / width) ** 2)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def expset():
    return ExperimentSet(
        [Experiment(tau=np.linspace(1e-4, 8e-3, 40), n_pulses=16, b_z=311.0)]
    )


def make_state(sites, table, *, lam=3e-3, sigma=0.02, k_max=8, n_exp=1):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma), k_max=k_max,
    )


@pytest.fixture
def scene(tiny_site_table, expset, model):
    """Truth, data and a target on the hand-built table."""
    truth = make_state([0, 2], tiny_site_table)
    data = simulate_dataset(truth, expset, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    return truth, data


# --------------------------------------------------------------------------
# the measure: turning a signal into something transportable
# --------------------------------------------------------------------------


def test_measure_is_the_dip_depth():
    assert signal_measure(np.array([1.0, 0.75, 0.5])) == pytest.approx([0.0, 0.25, 0.5])


def test_measure_of_full_coherence_is_empty():
    """A signal pinned at 1 has no features to transport."""
    assert np.all(signal_measure(np.ones(10)) == 0.0)


def test_measure_is_non_negative_for_a_physical_signal():
    signal = 0.5 * (1.0 + np.cos(np.linspace(0, 20, 100)))
    assert np.all(signal_measure(signal) >= 0.0)


def test_noisy_data_above_unit_coherence_does_not_give_negative_weight():
    """Noise pushes data past 1; a negative weight makes transport meaningless."""
    assert np.all(signal_measure(np.array([1.004, 0.9, 1.0009])) >= 0.0)


def test_floor_raises_the_clipping_level():
    assert np.all(signal_measure(np.array([1.5, 0.2]), floor=0.05) >= 0.05)


# --------------------------------------------------------------------------
# normalisation -- the three things that make this a distance on distributions
# --------------------------------------------------------------------------


def test_rescaling_one_signals_mass_changes_nothing():
    """Unit-mass normalisation: W compares shapes, not amplitudes."""
    a, b = bump(0.3), bump(0.5)
    deep = 1.0 - 3.0 * (1.0 - a)          # same feature, three times as deep
    assert wasserstein_signal_distance(deep, b, TAU) == pytest.approx(
        wasserstein_signal_distance(a, b, TAU), rel=1e-9)


def test_the_result_does_not_depend_on_the_units_of_tau():
    """Divided by the span, so ms and us give the same number."""
    a, b = bump(0.3), bump(0.5)
    in_ms = wasserstein_signal_distance(a, b, TAU)
    in_us = wasserstein_signal_distance(a, b, TAU * 1000.0)
    assert in_us == pytest.approx(in_ms, rel=1e-9)


def test_a_pure_translation_costs_the_fraction_of_the_window_moved():
    """The calibration: a feature moved a fifth of the window scores 0.2."""
    assert wasserstein_signal_distance(bump(0.3), bump(0.5), TAU) == pytest.approx(
        0.2, abs=1e-3)


def test_the_distance_is_dimensionless_and_bounded():
    a, b = bump(0.02), bump(0.98)
    assert 0.0 <= wasserstein_signal_distance(a, b, TAU) <= 1.0


# --------------------------------------------------------------------------
# metric properties
# --------------------------------------------------------------------------


def test_identical_signals_are_at_zero_distance():
    assert wasserstein_signal_distance(bump(0.4), bump(0.4), TAU) == pytest.approx(0.0)


def test_distance_is_symmetric():
    a, b = bump(0.3), bump(0.7)
    assert wasserstein_signal_distance(a, b, TAU) == pytest.approx(
        wasserstein_signal_distance(b, a, TAU))


def test_distance_is_non_negative():
    assert wasserstein_signal_distance(bump(0.1), bump(0.9), TAU) >= 0.0


def test_triangle_inequality_holds():
    a, b, c = bump(0.2), bump(0.5), bump(0.8)
    assert wasserstein_signal_distance(a, c, TAU) <= (
        wasserstein_signal_distance(a, b, TAU)
        + wasserstein_signal_distance(b, c, TAU) + 1e-9)


def test_two_featureless_signals_agree_rather_than_erroring():
    """Both measures empty: nothing to transport is agreement, not a failure."""
    assert wasserstein_signal_distance(np.ones(50), np.ones(50), TAU[:50]) == \
        pytest.approx(0.0)


def test_one_empty_measure_against_a_featured_one_is_maximal():
    """No transport plan turns a featureless signal into a modulated one.

    Not covered by the approved scaffold; the case turned up while
    implementing, where the choice is between a nan, an exception, and the
    maximum. The maximum is the one that lets a sampler walk out of it.
    """
    assert wasserstein_signal_distance(np.ones(201), bump(0.5), TAU) == \
        pytest.approx(1.0)
    assert wasserstein_signal_distance(bump(0.5), np.ones(201), TAU) == \
        pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        wasserstein_signal_distance(bump(0.3), bump(0.5)[:-1], TAU)


def test_a_degenerate_tau_axis_raises():
    """A zero span cannot normalise anything."""
    with pytest.raises(ValueError):
        wasserstein_signal_distance(np.array([0.5, 0.5]), np.array([0.4, 0.6]),
                                    np.array([1.0, 1.0]))


# --------------------------------------------------------------------------
# the exact reduction at zeta = 0
# --------------------------------------------------------------------------


def test_zeta_zero_equals_the_gaussian_to_machine_precision(scene, tiny_site_table,
                                                            model):
    truth, data = scene
    gaussian = GaussianL2().log_prob(truth, data, model, tiny_site_table)
    variant = WassersteinL2(zeta=0.0).log_prob(truth, data, model, tiny_site_table)
    assert variant == pytest.approx(gaussian, rel=1e-15)


@pytest.mark.parametrize("sites", [(0,), (1, 2), (0, 2, 3), (1,)])
def test_the_reduction_holds_for_arbitrary_states(scene, tiny_site_table, model,
                                                  sites):
    _truth, data = scene
    state = make_state(sites, tiny_site_table)
    assert WassersteinL2(zeta=0.0).log_prob(state, data, model, tiny_site_table) == \
        pytest.approx(GaussianL2().log_prob(state, data, model, tiny_site_table),
                      rel=1e-15)


def test_the_reduction_ignores_the_scale(scene, tiny_site_table, model):
    """zeta = 0 must switch the term off, whatever it would have been scaled by."""
    _truth, data = scene
    state = make_state([0, 2], tiny_site_table)
    a = WassersteinL2(zeta=0.0, scale=1.0).log_prob(state, data, model,
                                                    tiny_site_table)
    b = WassersteinL2(zeta=0.0, scale=1e9).log_prob(state, data, model,
                                                    tiny_site_table)
    assert a == pytest.approx(b, rel=1e-15)


def test_zeta_zero_survives_a_degenerate_signal(scene, tiny_site_table, model):
    """`0 * nan` is nan: the penalty must be short-circuited, not multiplied out."""
    _truth, data = scene
    state = make_state([3], tiny_site_table)       # a nearly featureless spin
    assert np.all(np.isfinite(
        WassersteinL2(zeta=0.0).log_prob(state, data, model, tiny_site_table)))


def test_zeta_zero_gives_an_identical_accepted_path(scene, tiny_site_table, model):
    """Exact reduction means the sampler cannot tell the two apart."""
    _truth, data = scene

    def run(likelihood):
        target = Target(data, model, likelihood, tiny_site_table)
        sampler = RWMH(ParameterBlock("lam"),
                       ContinuousReflected(radius=2e-4, lower=5e-4, upper=2e-2))
        trace = Trace(n_sites=len(tiny_site_table), k_max=8, n_exp=1)
        sampler.run(make_state([0, 2], tiny_site_table), target,
                    np.random.default_rng(4), n_steps=60, trace=trace)
        return np.asarray(trace.lam)

    assert np.array_equal(run(GaussianL2()), run(WassersteinL2(zeta=0.0)))


# --------------------------------------------------------------------------
# behaviour with the penalty switched on
# --------------------------------------------------------------------------


def test_the_penalty_only_ever_lowers_the_log_likelihood(scene, tiny_site_table,
                                                         model):
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    penalised = WassersteinL2(zeta=0.5).log_prob(state, data, model, tiny_site_table)
    plain = GaussianL2().log_prob(state, data, model, tiny_site_table)
    assert np.all(penalised <= plain)


def test_the_penalty_grows_with_zeta(scene, tiny_site_table, model):
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    values = [WassersteinL2(zeta=z).log_prob(state, data, model,
                                             tiny_site_table)[0]
              for z in (0.0, 0.25, 0.5, 1.0)]
    assert all(b <= a + 1e-12 for a, b in pairwise(values))


def test_zeta_changes_the_ranking_of_some_pair(scene, tiny_site_table, model):
    """Without this the parameter is decorative."""
    _truth, data = scene
    left, right = make_state([0, 2], tiny_site_table), make_state([1, 3],
                                                                 tiny_site_table)

    def gap(likelihood):
        return (likelihood.log_prob(left, data, model, tiny_site_table)[0]
                - likelihood.log_prob(right, data, model, tiny_site_table)[0])

    assert gap(WassersteinL2(zeta=1.0)) != pytest.approx(gap(GaussianL2()))


def test_the_penalty_vanishes_when_the_signals_match(scene, tiny_site_table,
                                                     model):
    """Zero transport cost, so the variant agrees with the Gaussian even at zeta > 0."""
    truth, data = scene
    exact = simulate_dataset(truth, data, tiny_site_table, model, sigma=0.0,
                             rng=np.random.default_rng(0))
    assert WassersteinL2(zeta=0.7).log_prob(truth, exact, model,
                                            tiny_site_table) == pytest.approx(
        GaussianL2().log_prob(truth, exact, model, tiny_site_table), rel=1e-12)


def test_the_penalty_is_bounded_by_zeta_times_the_scale(scene, tiny_site_table,
                                                        model):
    """W is bounded by 1, so the penalty can never exceed zeta * scale.

    This is what makes the scale's meaning testable rather than folklore: it is
    the most the transport term is allowed to move the log-likelihood, and on
    real data it lands far below that ceiling. Measured at the default scale on
    NV data, the penalty is about a tenth of a percent of the residual term --
    present, but unable to change an acceptance decision until the scale is
    calibrated upward.
    """
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    zeta, scale = 0.5, 37.0
    gap = (GaussianL2().log_prob(state, data, model, tiny_site_table)
           - WassersteinL2(zeta=zeta, scale=scale).log_prob(
               state, data, model, tiny_site_table))
    assert np.all(gap >= 0.0)
    assert np.all(gap <= zeta * scale + 1e-12)


def test_zeta_outside_the_unit_interval_raises():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            WassersteinL2(zeta=bad)


def test_one_value_per_replica(scene, tiny_site_table, model):
    _truth, data = scene
    replicas = make_state([0, 2], tiny_site_table).expand_replicas(4)
    assert WassersteinL2(zeta=0.3).log_prob(replicas, data, model,
                                            tiny_site_table).shape == (4,)


# --------------------------------------------------------------------------
# why the published product form is not what is implemented
# --------------------------------------------------------------------------


def test_the_published_product_form_is_negative_where_the_sampler_lives():
    """Measured, not asserted: (1 - z) exp(E) - z W < 0 for ordinary states.

    Spec Sec. 7.1 prints the penalty as a product form. Its logarithm is
    undefined wherever it is negative, and E is large and negative for any
    configuration that does not already fit -- traces in this project routinely
    sit near -3000. The penalty is therefore applied additively in log space.
    """
    transport = 0.2
    for exponent in (-50.0, -3000.0):
        for zeta in (0.1, 0.5):
            assert (1.0 - zeta) * np.exp(exponent) - zeta * transport < 0.0


def test_the_implemented_form_is_finite_there(scene, tiny_site_table, model):
    """Same regime, this implementation: a number, not a nan."""
    _truth, data = scene
    bad = make_state([3], tiny_site_table)
    value = WassersteinL2(zeta=0.5).log_prob(bad, data, model, tiny_site_table)
    assert np.all(np.isfinite(value))


# --------------------------------------------------------------------------
# it is a Likelihood like any other
# --------------------------------------------------------------------------


def test_it_subclasses_the_likelihood_abc():
    assert issubclass(WassersteinL2, Likelihood)


def test_it_installs_in_a_target(scene, tiny_site_table, model):
    _truth, data = scene
    target = Target(data, model, WassersteinL2(zeta=0.2), tiny_site_table)
    assert np.all(np.isfinite(target.log_prob(make_state([0, 2], tiny_site_table))))


def test_tempering_scales_the_whole_thing(scene, tiny_site_table, model):
    """beta multiplies the penalised log-likelihood, not just the residual."""
    _truth, data = scene
    target = Target(data, model, WassersteinL2(zeta=0.4), tiny_site_table)
    state = make_state([1, 3], tiny_site_table)
    assert target.log_prob(state, beta=0.25) == pytest.approx(
        0.25 * target.log_prob(state, beta=1.0))
