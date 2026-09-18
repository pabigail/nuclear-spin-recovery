"""Synthetic data generation."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    State,
    StretchedExponential,
    add_noise,
    simulate_coherence,
    simulate_dataset,
)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def state(tiny_site_table):
    return State.from_sites(
        (0, 2),
        n_sites=len(tiny_site_table),
        n_exp=1,
        lam=np.array([[3e-3]]),
        n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]),
        k_max=8,
    )


def test_noiseless_matches_forward_model(model, state, single_experiment,
                                         tiny_site_table):
    got = simulate_coherence(state, single_experiment, tiny_site_table, model)
    expected = model.coherence(state, single_experiment, tiny_site_table)[0]
    assert got == pytest.approx(expected)


def test_add_noise_is_reproducible():
    signal = np.full(200, 0.5)
    a = add_noise(signal, 0.01, rng=np.random.default_rng(7))
    b = add_noise(signal, 0.01, rng=np.random.default_rng(7))
    assert a == pytest.approx(b)


def test_add_noise_differs_across_seeds():
    signal = np.full(200, 0.5)
    a = add_noise(signal, 0.01, rng=np.random.default_rng(1))
    b = add_noise(signal, 0.01, rng=np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_noise_std_matches_request():
    signal = np.zeros(200_000)
    got = add_noise(signal, 0.01, rng=np.random.default_rng(0))
    assert np.std(got) == pytest.approx(0.01, rel=0.02)


def test_noise_is_zero_mean():
    signal = np.zeros(200_000)
    got = add_noise(signal, 0.01, rng=np.random.default_rng(0))
    assert np.mean(got) == pytest.approx(0.0, abs=1e-3)


def test_zero_sigma_leaves_signal_untouched():
    signal = np.linspace(0.0, 1.0, 50)
    assert add_noise(signal, 0.0, rng=np.random.default_rng(0)) == pytest.approx(signal)


def test_per_experiment_sigma(two_experiments):
    """A per-experiment sigma must scale only its own points."""
    signal = np.zeros(two_experiments.n_points)
    got = add_noise(
        signal,
        np.array([1e-6, 1.0]),
        exp_id=two_experiments.exp_id,
        rng=np.random.default_rng(0),
    )
    assert np.all(np.abs(got[:3]) < 1e-4)
    assert np.any(np.abs(got[3:]) > 0.1)


def test_simulate_dataset_attaches_data(model, state, single_experiment,
                                        tiny_site_table):
    out = simulate_dataset(
        state, single_experiment, tiny_site_table, model, sigma=0.001,
        rng=np.random.default_rng(0),
    )
    assert out.data_all is not None
    assert len(out.data_all) == single_experiment.n_points


def test_simulate_dataset_does_not_mutate_input(model, state, single_experiment,
                                                tiny_site_table):
    simulate_dataset(
        state, single_experiment, tiny_site_table, model, sigma=0.001,
        rng=np.random.default_rng(0),
    )
    assert single_experiment.experiments[0].data is None


def test_simulated_residual_is_within_noise(model, state, single_experiment,
                                            tiny_site_table):
    out = simulate_dataset(
        state, single_experiment, tiny_site_table, model, sigma=0.001,
        rng=np.random.default_rng(0),
    )
    truth = simulate_coherence(state, single_experiment, tiny_site_table, model)
    assert np.std(out.data_all - truth) == pytest.approx(0.001, rel=0.3)
