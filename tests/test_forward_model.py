# tests/test_forward_model_parametrized_experiments.py

import numpy as np
import pytest

from nuclear_spin_recover.coherence_signal import (
    SingleCoherenceSignal,
    CoherenceSignalList,
)
from nuclear_spin_recover.forward_model import ForwardModel

# -----------------------------------------------------------------------------
# Fixtures: experiments and spin bath
# -----------------------------------------------------------------------------


@pytest.fixture
def single_experiment():
    class SingleExperiment:
        def __init__(self):
            self.times = np.linspace(0.0, 1.0, 50)

    return SingleExperiment()


@pytest.fixture
def batch_experiment():
    class SingleExperiment:
        def __init__(self):
            self.times = np.linspace(0.0, 1.0, 50)

    class BatchExperiment:
        def __init__(self):
            self.experiments = [SingleExperiment() for _ in range(3)]

    return BatchExperiment()


@pytest.fixture
def spin_bath():
    class SpinBath:
        def copy_with_noise(self, eps):
            return self

    return SpinBath()


# -----------------------------------------------------------------------------
# Forward model implementations to test
# -----------------------------------------------------------------------------


class DummyForwardModel:
    def compute_coherence(self, experiment, spin_bath):
        if spin_bath is None:
            raise AttributeError("spin_bath cannot be None")

        if hasattr(experiment, "times"):
            return SingleCoherenceSignal(
                times=experiment.times, coherence=np.ones_like(experiment.times)
            )
        elif hasattr(experiment, "experiments"):
            signals = [
                SingleCoherenceSignal(times=e.times, coherence=np.ones_like(e.times))
                for e in experiment.experiments
            ]
            return CoherenceSignalList(signals)
        else:
            raise TypeError("Unsupported experiment type")


# List all forward models here
FORWARD_MODELS = [
    DummyForwardModel(),
    # AnalyticForwardModel(),
    # PyCCEForwardModel(),
]


# -----------------------------------------------------------------------------
# Parametrize both model and experiment type
# -----------------------------------------------------------------------------


@pytest.fixture(params=FORWARD_MODELS)
def model(request) -> ForwardModel:
    return request.param


@pytest.fixture(params=["single", "batch"])
def experiment_type(request, single_experiment, batch_experiment):
    if request.param == "single":
        return single_experiment
    else:
        return batch_experiment


# -----------------------------------------------------------------------------
# Generic forward model tests
# -----------------------------------------------------------------------------


def test_model_has_compute_coherence(model):
    assert hasattr(model, "compute_coherence")
    assert callable(model.compute_coherence)


def test_compute_coherence_returns_correct_type(model, experiment_type, spin_bath):
    signal = model.compute_coherence(experiment_type, spin_bath)

    if hasattr(experiment_type, "times"):
        assert isinstance(signal, SingleCoherenceSignal)
    else:
        assert isinstance(signal, CoherenceSignalList)
        assert len(signal) == len(experiment_type.experiments)


def test_signal_shapes(model, experiment_type, spin_bath):
    signal = model.compute_coherence(experiment_type, spin_bath)

    if isinstance(signal, SingleCoherenceSignal):
        assert signal.times.shape == signal.coherence.shape
        assert signal.times.ndim == 1
    else:
        for s, e in zip(signal, experiment_type.experiments):
            assert s.times.shape == s.coherence.shape
            assert s.times.shape == e.times.shape


def test_deterministic(model, experiment_type, spin_bath):
    s1 = model.compute_coherence(experiment_type, spin_bath)
    s2 = model.compute_coherence(experiment_type, spin_bath)

    if isinstance(s1, SingleCoherenceSignal):
        np.testing.assert_allclose(s1.coherence, s2.coherence)
    else:
        for sig1, sig2 in zip(s1, s2):
            np.testing.assert_allclose(sig1.coherence, sig2.coherence)


# -----------------------------------------------------------------------------
# Failure tests
# -----------------------------------------------------------------------------


def test_invalid_experiment_raises(model, spin_bath):
    with pytest.raises(TypeError):
        model.compute_coherence("not an experiment", spin_bath)


def test_invalid_spin_bath_raises(model, single_experiment):
    with pytest.raises(AttributeError):
        model.compute_coherence(single_experiment, None)


# -----------------------------------------------------------------------------
# Example downstream usage: log-likelihood
# -----------------------------------------------------------------------------


def log_likelihood(model, experiment, spin_bath, data):
    signal = model.compute_coherence(experiment, spin_bath)
    residual = data - signal.coherence
    return -0.5 * np.sum(residual**2)


def test_log_likelihood(model, single_experiment, spin_bath):
    data = np.ones_like(single_experiment.times)
    ll = log_likelihood(model, single_experiment, spin_bath, data)
    assert np.isfinite(ll)
