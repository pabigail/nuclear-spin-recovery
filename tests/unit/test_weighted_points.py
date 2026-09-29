"""Per-point measurement weights.

Adaptive design produces a *subset of points with unequal averaging time*, so
the likelihood has to be able to say that one point was measured harder than
another. The form is a relative weight on top of the sampled per-experiment
noise,

    log L = -1/(2 sigma_e^2) sum_j w_j (d_j - f_j)^2,

so the effective noise at a point is sigma_e / sqrt(w_j): doubling the
repetitions halves the variance. A weight rather than a literal per-point sigma
because sigma_e is a **sampled** parameter — an explicit per-point array would
leave the sampler nothing to update.

The load-bearing property is that w = 1 reduces to the present likelihood
exactly, since every calibrated number in docs/test-plan.md was measured
without weights.

docs/phase-5-plan.md §6, question 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    GaussianL2,
    State,
    StretchedExponential,
    WassersteinL2,
    add_noise,
    simulate_dataset,
)
from nuclear_spin_recovery.post import residual_distribution

TAU = np.linspace(1e-4, 8e-3, 30)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


def make_state(sites, table, *, sigma=0.02, k_max=8, n_exp=1):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=n_exp, lam=np.full((1, n_exp), 3e-3),
        n_stretch=np.ones((1, n_exp)), sigma=np.full((1, n_exp), sigma),
        k_max=k_max)


@pytest.fixture
def scene(tiny_site_table, model):
    truth = make_state([0, 2], tiny_site_table)
    blank = ExperimentSet([Experiment(tau=TAU, n_pulses=16, b_z=311.0)])
    data = simulate_dataset(truth, blank, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    return truth, data


def reweighted(expset, weight):
    """The same data with a per-point weight attached."""
    return ExperimentSet([
        Experiment(tau=e.tau, n_pulses=e.n_pulses, b_z=e.b_z, data=e.data,
                   sigma=e.sigma, weight=weight)
        for e in expset.experiments])


# --------------------------------------------------------------------------
# the field itself
# --------------------------------------------------------------------------


def test_weight_defaults_to_absent():
    assert Experiment(tau=TAU, n_pulses=16, b_z=311.0).weight is None


def test_weight_must_match_the_grid():
    with pytest.raises(ValueError):
        Experiment(tau=TAU, n_pulses=16, b_z=311.0, weight=np.ones(5))


def test_a_negative_weight_raises():
    bad = np.ones_like(TAU)
    bad[3] = -1.0
    with pytest.raises(ValueError):
        Experiment(tau=TAU, n_pulses=16, b_z=311.0, weight=bad)


def test_a_non_finite_weight_raises():
    bad = np.ones_like(TAU)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        Experiment(tau=TAU, n_pulses=16, b_z=311.0, weight=bad)


def test_weight_all_is_ones_when_unset(scene):
    _truth, data = scene
    assert data.weight_all == pytest.approx(np.ones(data.n_points))


def test_weight_all_concatenates_ragged_experiments():
    two = ExperimentSet([
        Experiment(tau=TAU, n_pulses=8, b_z=311.0, weight=np.full(TAU.size, 2.0)),
        Experiment(tau=TAU[:10], n_pulses=16, b_z=311.0),
    ])
    out = two.weight_all
    assert out.shape == (TAU.size + 10,)
    assert out[:TAU.size] == pytest.approx(2.0)
    assert out[TAU.size:] == pytest.approx(1.0)     # unset means uniform


# --------------------------------------------------------------------------
# the reduction, which every calibrated number depends on
# --------------------------------------------------------------------------


def test_unit_weights_reproduce_the_unweighted_likelihood(scene,
                                                          tiny_site_table,
                                                          model):
    """Exact, not close: test-plan §5 was measured without weights."""
    truth, data = scene
    weighted = reweighted(data, np.ones(data.n_points))
    assert GaussianL2().log_prob(truth, weighted, model, tiny_site_table) == \
        pytest.approx(GaussianL2().log_prob(truth, data, model,
                                            tiny_site_table), rel=1e-15)


def test_absent_weights_reproduce_the_unweighted_likelihood(scene,
                                                            tiny_site_table,
                                                            model):
    """The default path must not merely agree, it must be the same path."""
    truth, data = scene
    before = GaussianL2().log_prob(truth, data, model, tiny_site_table)
    assert np.all(np.isfinite(before))


@pytest.mark.parametrize("sites", [(0,), (1, 2), (0, 2, 3)])
def test_the_reduction_holds_for_arbitrary_states(scene, tiny_site_table,
                                                  model, sites):
    _truth, data = scene
    state = make_state(sites, tiny_site_table)
    weighted = reweighted(data, np.ones(data.n_points))
    assert GaussianL2().log_prob(state, weighted, model, tiny_site_table) == \
        pytest.approx(GaussianL2().log_prob(state, data, model,
                                            tiny_site_table), rel=1e-15)


# --------------------------------------------------------------------------
# what the weight does
# --------------------------------------------------------------------------


def test_doubling_every_weight_doubles_the_log_likelihood(scene,
                                                          tiny_site_table,
                                                          model):
    """A uniform weight is a rescaling of sigma_e^2, and must act like one."""
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    single = GaussianL2().log_prob(state, reweighted(data, np.ones(data.n_points)),
                                   model, tiny_site_table)
    double = GaussianL2().log_prob(state, reweighted(data, np.full(data.n_points, 2.0)),
                                   model, tiny_site_table)
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_a_zero_weight_drops_a_point_entirely(scene, tiny_site_table, model):
    """The mechanism adaptive design needs: points not measured cost nothing."""
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    keep = np.ones(data.n_points)
    keep[10:] = 0.0
    kept = GaussianL2().log_prob(state, reweighted(data, keep), model,
                                 tiny_site_table)

    short = ExperimentSet([Experiment(tau=data.experiments[0].tau[:10],
                                      n_pulses=16, b_z=311.0,
                                      data=data.data_all[:10])])
    truncated = GaussianL2().log_prob(state, short, model, tiny_site_table)
    assert kept == pytest.approx(truncated, rel=1e-12)


def test_weights_match_a_hand_computation(scene, tiny_site_table, model):
    _truth, data = scene
    state = make_state([0, 2], tiny_site_table)
    rng = np.random.default_rng(1)
    weight = rng.uniform(0.2, 3.0, size=data.n_points)
    predicted = model.coherence(state, data, tiny_site_table)[0]
    expected = -0.5 * np.sum(
        weight * (data.data_all - predicted) ** 2) / state.sigma[0, 0] ** 2
    assert GaussianL2().log_prob(state, reweighted(data, weight), model,
                                 tiny_site_table)[0] == pytest.approx(expected)


def test_the_wasserstein_variant_honours_weights_too(scene, tiny_site_table,
                                                     model):
    """Its residual term is the same Gaussian and must not diverge from it."""
    _truth, data = scene
    state = make_state([1, 3], tiny_site_table)
    weight = np.full(data.n_points, 3.0)
    assert WassersteinL2(weight=0.0).log_prob(
        state, reweighted(data, weight), model, tiny_site_table) == \
        pytest.approx(GaussianL2().log_prob(
            state, reweighted(data, weight), model, tiny_site_table), rel=1e-15)


# --------------------------------------------------------------------------
# simulation and the residual have to agree with the likelihood
# --------------------------------------------------------------------------


def test_simulated_noise_shrinks_as_the_root_of_the_weight():
    """A point measured four times as long is half as noisy."""
    signal = np.zeros(4000)
    weight = np.full(4000, 4.0)
    noisy = add_noise(signal, 0.01, rng=np.random.default_rng(0), weight=weight)
    assert np.std(noisy) == pytest.approx(0.005, rel=0.08)


def test_simulated_noise_is_unchanged_at_unit_weight():
    signal = np.zeros(4000)
    plain = add_noise(signal, 0.01, rng=np.random.default_rng(0))
    weighted = add_noise(signal, 0.01, rng=np.random.default_rng(0),
                         weight=np.ones(4000))
    assert np.array_equal(plain, weighted)


def test_the_residual_is_weighted_too():
    """Otherwise criterion A and the likelihood would disagree about the fit."""
    observed = np.zeros(6)
    predictive = np.full((1, 6), 0.02)
    weight = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    half = residual_distribution(observed, predictive, 0.01, weight=weight)
    assert half == pytest.approx([2.0])


def test_the_residual_reduces_at_unit_weight():
    observed = np.zeros(4)
    predictive = np.full((1, 4), 0.02)
    assert residual_distribution(observed, predictive, 0.01,
                                 weight=np.ones(4)) == \
        pytest.approx(residual_distribution(observed, predictive, 0.01))
