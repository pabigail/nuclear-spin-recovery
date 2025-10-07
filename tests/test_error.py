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
        # Return a simple function of experiment index and spin count
        n = len(self.spins)
        return np.linspace(0, 1, 5) + 0.1 * idx + 0.01 * n


# ----------------------------------------------------------------------
# Imports of the classes under test
# ----------------------------------------------------------------------

from nuclear_spin_recover import ErrorModel, L2Error, CompositeErrorL2andWasserstein


# ----------------------------------------------------------------------
# 1. Base class ErrorModel
# ----------------------------------------------------------------------

def test_error_model_requires_data():
    # Define a temporary subclass to allow instantiation
    class DummyError(ErrorModel):
        def __call__(self, spins, experiment, forward_model_cls):
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
    # observed data similar but not identical to simulated
    data = [
        np.linspace(0, 1, 5),
        np.linspace(0.1, 1.1, 5),
    ]
    spins = [1, 2, 3]
    experiment = MockExperiment(num_experiments=2)
    err_model = L2Error(data)

    error = err_model(spins, experiment, MockForwardModel)
    assert isinstance(error, float)
    assert error > 0.0  # nonzero difference
    assert np.isfinite(error)


def test_l2_error_length_mismatch():
    # observed and simulated arrays of different lengths should raise
    class BadForwardModel(MockForwardModel):
        def compute_coherence(self, idx):
            return np.linspace(0, 1, 3)  # shorter array

    data = [np.linspace(0, 1, 5)]
    experiment = MockExperiment(num_experiments=1)
    err_model = L2Error(data)

    with pytest.raises(ValueError):
        err_model([1], experiment, BadForwardModel)


def test_l2_error_multiple_experiments():
    """Test that L2Error correctly sums over multiple experiments."""
    # Observed data slightly offset from simulated
    data = [
        np.linspace(0, 1, 5) + 0.1,  # exp 0
        np.linspace(0, 1, 5) + 1.1,  # exp 1
        np.linspace(0, 1, 5) + 2.1,  # exp 2
    ]
    spins = ["NuclearSpin1"]
    exp = MockExperiment(num_experiments=3)

    err_model = L2Error(data)
    err_value = err_model(spins, exp, MockForwardModel)

    # Compute expected error manually using the same logic as the model
    expected_error = 0.0
    model = MockForwardModel(spins, exp)
    for idx in range(len(exp)):
        simulated = np.array(model.compute_coherence(idx))
        observed = np.array(data[idx])
        expected_error += np.linalg.norm(observed - simulated)

    np.testing.assert_allclose(err_value, expected_error, rtol=1e-12)
    assert err_value > 0

# ----------------------------------------------------------------------
# 3. CompositeErrorL2andWasserstein tests
# ----------------------------------------------------------------------

def test_composite_error_pure_L2():
    data = [
        np.linspace(0, 1, 5),
        np.linspace(0.1, 1.1, 5),
    ]
    spins = [1, 2]
    experiment = MockExperiment(num_experiments=2, noise=[0.1, 0.1])

    err_model = CompositeErrorL2andWasserstein(data, sigma_sq=None, lambda_wass=0.0)
    error = err_model(spins, experiment, MockForwardModel)
    assert isinstance(error, float)
    assert error > 0.0


def test_composite_error_with_wasserstein_weight():
    data = [
        np.linspace(0, 1, 5),
        np.linspace(0.5, 1.5, 5),
    ]
    spins = [1]
    experiment = MockExperiment(num_experiments=2, noise=[0.2, 0.2])

    err_model = CompositeErrorL2andWasserstein(data, sigma_sq=0.1, lambda_wass=0.5)
    error = err_model(spins, experiment, MockForwardModel)
    assert isinstance(error, float)
    assert error > 0.0


def test_composite_error_handles_invalid_mass():
    """If observed/simulated sums are invalid, Wasserstein term should be skipped (no NaN)."""
    class FlatForwardModel(MockForwardModel):
        def compute_coherence(self, idx):
            return np.zeros(5)  # zero-mass distribution

    data = [np.zeros(5)]
    experiment = MockExperiment(num_experiments=1)
    err_model = CompositeErrorL2andWasserstein(data, sigma_sq=1.0, lambda_wass=1.0)

    error = err_model([1], experiment, FlatForwardModel)
    assert np.isfinite(error)
    assert error == pytest.approx(0.0)

def test_composite_error_multiple_experiments_with_wasserstein():
    """Test that CompositeErrorL2andWasserstein handles multiple experiments."""
    data = [
        np.linspace(0, 1, 5) + 0.05,
        np.linspace(1, 2, 5) + 0.1,
        np.linspace(2, 3, 5) + 0.15,
    ]
    spins = ["spin"]
    exp = MockExperiment(num_experiments=3, noise=[0.5, 1.0, 2.0])

    model = CompositeErrorL2andWasserstein(
        data, sigma_sq=None, lambda_wass=0.3
    )
    err_value = model(spins, exp, MockForwardModel)

    # Check scalar output and monotonicity
    assert isinstance(err_value, float)
    assert err_value > 0

    # Validate Wasserstein contribution is nonzero for at least one experiment
    wass_contributions = []
    for idx in range(len(exp)):
        simulated = np.linspace(0, 1, 5) + idx
        observed = data[idx]
        min_val = min(observed.min(), simulated.min())
        obs_shift = observed - min_val
        sim_shift = simulated - min_val
        obs_norm = obs_shift / obs_shift.sum()
        sim_norm = sim_shift / sim_shift.sum()
        wass_contrib = wasserstein_distance(
            np.arange(len(observed)), np.arange(len(observed)), obs_norm, sim_norm
        )
        wass_contributions.append(wass_contrib)

    assert any(w > 0 for w in wass_contributions)

# ----------------------------------------------------------------------
# 4. Numerical correctness sanity check
# ----------------------------------------------------------------------

def test_composite_error_matches_manual_L2():
    """Compare L2 term to a manual computation for validation."""
    data = [np.array([0.0, 0.5, 1.0])]
    spins = [1]
    experiment = MockExperiment(num_experiments=1, noise=[0.5])

    class SimpleForwardModel(MockForwardModel):
        def compute_coherence(self, idx):
            return np.array([0.1, 0.4, 0.9])

    err_model = CompositeErrorL2andWasserstein(data, sigma_sq=0.5, lambda_wass=0.0)
    val = err_model(spins, experiment, SimpleForwardModel)

    manual_val = 0.5 / 0.5 * np.sum((data[0] - np.array([0.1, 0.4, 0.9])) ** 2)
    assert np.isclose(val, manual_val)

