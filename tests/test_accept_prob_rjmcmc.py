import pytest
import numpy as np
import random

from nuclear_spin_recover.acceptance_probabilities import AcceptProbRJMCMC
from nuclear_spin_recover.spin_bath import NuclearSpin, SpinBath
from nuclear_spin_recover.experiments import Experiment
from nuclear_spin_recover.forward_models import AnalyticCoherenceModel
from nuclear_spin_recover.proposals import DiscreteRJMCMCProposal
from nuclear_spin_recover.error import L2Error

# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def simple_spin_bath():
    """SpinBath with 5 nuclear spins along x-axis, explicit gyro=1.0."""
    spins = [
        NuclearSpin(spin_type="13C", x=float(i), y=0.0, z=0.0,
                    A_par=10.0, A_perp=5.0, gyro=1.0)
        for i in range(5)
    ]
    return SpinBath(spins=spins)

@pytest.fixture
def default_params():
    """Dummy MCMC params including arrays for continuous proposals."""
    return {
        "lambda_decoherence": 0.5,
        "A_par": np.array([10.0]*5, dtype=float),
        "A_perp": np.array([5.0]*5, dtype=float)
    }

@pytest.fixture
def experiment_simple():
    """Single simple experiment."""
    num_exps = 1
    timepoints = [np.linspace(0, 1, 5)]
    num_pulses = [1]
    mag_field = [0.1]
    lambda_decoherence = [10.0]
    return Experiment(num_exps, num_pulses, mag_field, lambda_decoherence, timepoints)

@pytest.fixture
def forward_model_factory(simple_spin_bath, experiment_simple):
    """Return a factory that creates a forward model instance from a class."""
    def factory(cls, **kwargs):
        return cls(spins=simple_spin_bath.spins, experiment=experiment_simple, **kwargs)
    return factory

@pytest.fixture
def error_model(forward_model_factory):
    """L2Error using observed coherence signals from AnalyticCoherenceModel."""
    model_instance = forward_model_factory(AnalyticCoherenceModel)
    observed = model_instance.compute_coherence()
    return L2Error(observed)

# --------------------------
# Tests
# --------------------------

def test_accept_prob_rjmcmc_instantiation(simple_spin_bath, error_model, default_params, forward_model_factory):
    """Instantiate AcceptProbRJMCMC with a proper DiscreteRJMCMCProposal."""
    prop = DiscreteRJMCMCProposal(
        spin_bath=simple_spin_bath,
        max_spins=5,
        spin_inds=np.arange(len(simple_spin_bath.spins)),
        birth=True,
        params=default_params
    )

    model_instance = forward_model_factory(AnalyticCoherenceModel)

    apr = AcceptProbRJMCMC(
        forward_model=model_instance,
        error_model=error_model,
        proposal=prop
    )

    assert isinstance(apr, AcceptProbRJMCMC)
    assert apr.proposal is prop
    assert apr.error_model is error_model
    assert apr.temperature == 1.0

def test_accept_prob_rjmcmc_compute_birth(simple_spin_bath, error_model, default_params, forward_model_factory):
    """Compute acceptance probability for a birth move."""
    prop = DiscreteRJMCMCProposal(
        spin_bath=simple_spin_bath,
        max_spins=5,
        spin_inds=np.arange(len(simple_spin_bath.spins)),
        birth=True,
        params=default_params
    )
    prop.propose()  # generate proposed spins

    model_instance = forward_model_factory(AnalyticCoherenceModel)

    apr = AcceptProbRJMCMC(
        forward_model=model_instance,
        error_model=error_model,
        proposal=prop
    )

    prob = apr.compute()
    assert 0.0 <= prob <= 1.0

def test_accept_prob_rjmcmc_compute_death(simple_spin_bath, error_model, default_params, forward_model_factory):
    """Compute acceptance probability for a death move."""
    # For death, assume spin bath has 2 spins
    extra_spin = NuclearSpin("13C", 0.1, 0.1, 0.1, 0.2, 0.1, gyro=1.0)
    simple_spin_bath.spins.append(extra_spin)
    default_params["A_par"] = np.append(default_params["A_par"], extra_spin.A_par)
    default_params["A_perp"] = np.append(default_params["A_perp"], extra_spin.A_perp)

    prop = DiscreteRJMCMCProposal(
        spin_bath=simple_spin_bath,
        max_spins=5,
        spin_inds=np.arange(len(simple_spin_bath.spins)),
        birth=False,
        params=default_params
    )
    prop.propose()

    model_instance = forward_model_factory(AnalyticCoherenceModel)

    apr = AcceptProbRJMCMC(
        forward_model=model_instance,
        error_model=error_model,
        proposal=prop
    )

    prob = apr.compute()
    assert 0.0 <= prob <= 1.0

def test_accept_prob_rjmcmc_invalid_proposal(error_model, forward_model_factory):
    """Raises TypeError if proposal is not DiscreteRJMCMCProposal."""
    model_instance = forward_model_factory(AnalyticCoherenceModel)
    with pytest.raises(TypeError):
        AcceptProbRJMCMC(model_instance, error_model, proposal=None)

def test_accept_prob_rjmcmc_reproducibility(simple_spin_bath, error_model, default_params, forward_model_factory):
    """Repeated calls with fixed seed produce identical acceptance probabilities."""
    spin_inds = np.arange(len(simple_spin_bath.spins))

    random.seed(42)
    np.random.seed(42)
    prop1 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5, spin_inds=spin_inds, birth=None, params=default_params)
    prop1.propose()
    model_instance1 = forward_model_factory(AnalyticCoherenceModel)
    apr1 = AcceptProbRJMCMC(model_instance1, error_model, prop1)
    prob1 = apr1.compute()

    random.seed(42)
    np.random.seed(42)
    prop2 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5, spin_inds=spin_inds, birth=None, params=default_params)
    prop2.propose()
    model_instance2 = forward_model_factory(AnalyticCoherenceModel)
    apr2 = AcceptProbRJMCMC(model_instance2, error_model, prop2)
    prob2 = apr2.compute()

    assert prob1 == pytest.approx(prob2, rel=1e-12)

