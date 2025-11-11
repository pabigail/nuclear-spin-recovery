import pytest
import numpy as np
from scipy.stats import wasserstein_distance

# ----------------------------------------------------------------------
# Mock classes for ForwardModel and Experiment
# ----------------------------------------------------------------------

class MockExperiment:
    """Minimal stand-in for Experiment with len() and noise field."""
    def __init__(self, num_experiments=2, noise=None):
        self.num_experiments = num_experiments
        self.noise = np.array(noise) if noise is not None else np.ones(num_experiments)

    def __len__(self):
        return self.num_experiments


class MockForwardModel:
    """Mock forward model that returns deterministic simulated data."""
    def __init__(self, spins, experiment):
        self.spins = spins
        self.experiment = experiment

    def compute_coherence(self, idx):
        n = len(self.spins)
        return np.linspace(0, 1, 5) + 0.1 * idx + 0.01 * n


# ----------------------------------------------------------------------
# Imports of the classes under test
# ----------------------------------------------------------------------
from nuclear_spin_recover import (
    ErrorModel,
    L2Error,
    WassersteinError,
    CompositeError,
    GaussianLogLikelihoodFromError,
)

# ----------------------------------------------------------------------
# 1. Base class ErrorModel
# ----------------------------------------------------------------------

def test_error_model_requires_data():
    class DummyError(ErrorModel):
        def __call__(self, spins, forward_model):
            return 0.0

    with pytest.raises(ValueError):
        DummyError(None)


def test_error_model_repr():
    data = [np.zeros(5), np.ones(5)]
    model = L2Error(data)
    rep = repr(model)
    assert "L2Error" in rep
    assert "2 experiments" in rep


# ----------------------------------------------------------------------
# 2. L2Error tests
# ----------------------------------------------------------------------

def test_l2_error_computation():
    data = [np.linspace(0, 1, 5), np.linspace(0.1, 1.1, 5)]
    spins = [1, 2, 3]
    exp = MockExperiment(num_experiments=2)
    err_model = L2Error(data)

    model_instance = MockForwardModel(spins, exp)
    error = err_model(spins, model_instance)

    assert isinstance(error, float)
    assert error > 0.0
    assert np.isfinite(error)


def test_l2_error_length_mismatch_raises():
    data = [np.zeros(3)]
    spins = [1, 2, 3]
    exp = MockExperiment(num_experiments=1)
    err_model = L2Error(data)

    class BadModel(MockForwardModel):
        def compute_coherence(self, idx):
            return np.zeros(5)  # wrong length

    bad_model = BadModel(spins, exp)
    with pytest.raises(ValueError):
        err_model(spins, bad_model)


# ----------------------------------------------------------------------
# 3. WassersteinError tests
# ----------------------------------------------------------------------

def test_wasserstein_error_computation():
    data = [np.linspace(0, 1, 5), np.linspace(0.2, 1.2, 5)]
    spins = [1, 2]
    exp = MockExperiment(num_experiments=2)
    wass_err = WassersteinError(data)
    model_instance = MockForwardModel(spins, exp)
    w_err = wass_err(spins, model_instance)

    assert isinstance(w_err, float)
    assert w_err >= 0.0
    assert np.isfinite(w_err)


def test_wasserstein_error_handles_constant_data():
    """Should not crash on constant or zero data."""
    data = [np.zeros(5), np.zeros(5)]
    spins = [1]
    exp = MockExperiment(num_experiments=2)
    wass_err = WassersteinError(data)
    model_instance = MockForwardModel(spins, exp)
    val = wass_err(spins, model_instance)
    assert val == 0.0


# ----------------------------------------------------------------------
# 4. CompositeError tests
# ----------------------------------------------------------------------

def test_composite_error_linear_combination():
    data = [np.linspace(0, 1, 5), np.linspace(0.2, 1.2, 5)]
    spins = [1, 2]
    exp = MockExperiment(num_experiments=2)

    l2_err = L2Error(data)
    wass_err = WassersteinError(data)
    comp_err = CompositeError(data, [l2_err, wass_err], weights=[0.9, 0.1])

    model_instance = MockForwardModel(spins, exp)
    val = comp_err(spins, model_instance)

    assert isinstance(val, float)
    assert np.isfinite(val)
    # Should equal roughly weighted sum
    val_manual = 0.9 * l2_err(spins, model_instance) + 0.1 * wass_err(spins, model_instance)
    assert np.isclose(val, val_manual)


def test_composite_error_weight_mismatch_raises():
    data = [np.zeros(5)]
    l2_err = L2Error(data)
    wass_err = WassersteinError(data)
    with pytest.raises(ValueError):
        CompositeError(data, [l2_err, wass_err], weights=[1.0])


# ----------------------------------------------------------------------
# 5. GaussianLogLikelihoodFromError tests
# ----------------------------------------------------------------------

def test_gaussian_loglike_from_error_basic():
    data = [np.linspace(0, 1, 5), np.linspace(0.2, 1.2, 5)]
    spins = [1, 2, 3]
    exp = MockExperiment(num_experiments=2, noise=[0.05, 0.05])

    l2_err = L2Error(data)
    wass_err = WassersteinError(data)
    comp_err = CompositeError(data, [l2_err, wass_err], weights=[0.9, 0.1])

    loglike_model = GaussianLogLikelihoodFromError(
        data,
        base_error_model=comp_err,
        sigma_sq=0.05,
        as_negative=True,
    )

    model_instance = MockForwardModel(spins, exp)
    nll = loglike_model(spins, model_instance)

    assert isinstance(nll, float)
    assert np.isfinite(nll)
    assert nll < 0  # Negative log-likelihood should be negative


def test_gaussian_loglike_from_error_logsign():
    """Check as_negative flag flips sign correctly."""
    data = [np.linspace(0, 1, 5)]
    spins = [1]
    exp = MockExperiment(num_experiments=1, noise=[0.05])

    l2_err = L2Error(data)
    loglike_model_pos = GaussianLogLikelihoodFromError(
        data, base_error_model=l2_err, sigma_sq=0.05, as_negative=False
    )
    loglike_model_neg = GaussianLogLikelihoodFromError(
        data, base_error_model=l2_err, sigma_sq=0.05, as_negative=True
    )

    model_instance = MockForwardModel(spins, exp)
    ll_pos = loglike_model_pos(spins, model_instance)
    ll_neg = loglike_model_neg(spins, model_instance)

    assert np.isclose(ll_pos, -ll_neg)


def test_gaussian_loglike_infers_sigma_from_experiment():
    data = [np.linspace(0, 1, 5)]
    spins = [1, 2]
    exp = MockExperiment(num_experiments=1, noise=[0.2])

    l2_err = L2Error(data)
    model_instance = MockForwardModel(spins, exp)

    ll_model = GaussianLogLikelihoodFromError(
        data, base_error_model=l2_err, sigma_sq=None, as_negative=True
    )

    # Should infer sigma_sq from exp.noise
    val = ll_model(spins, model_instance)
    assert isinstance(val, float)
    assert np.isfinite(val)

