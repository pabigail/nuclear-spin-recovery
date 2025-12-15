import numpy as np
import pytest
from nuclear_spin_recover import (
        SpinBath,
        NuclearSpin,
        Experiment,
        ForwardModel,
        AnalyticCoherenceModel,
        ErrorModel,
        L2Error,
        GaussianLogLikelihoodFromError,
        DiscreteRJMCMCProposal,
        AcceptProb,
        AcceptProbRJMCMC)


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def simple_spins():
    """List of 5 13C nuclear spins along the x-axis."""
    spins = [
        NuclearSpin(
            spin_type="13C",
            x=float(i), y=0.0, z=0.0,
            A_par=10.0, A_perp=5.0, gyro=1.0
        )
        for i in range(5)
    ]
    return SpinBath(spins)

@pytest.fixture
def default_params():
    """Dummy MCMC params including arrays for continuous proposals."""
    return {
        "lambda_decoherence": 0.5,
        "A_par": np.array([10.0]*5, dtype=float),
        "A_perp": np.array([5.0]*5, dtype=float)
    }   

@pytest.fixture
def simple_experiment():
    # One experiment: 10 timepoints, Hahn echo (N=1), B=500 G
    timepoints = np.linspace(0, 1e-3, 10)  # seconds
    exp = Experiment(
        num_exps=1,
        num_pulses=[1],
        mag_field=[500.0],
        noise=[0.0],
        timepoints=[timepoints],
        lambda_decoherence=[1.0]  # replaced by lambda_decoherence in AnalyticCoherenceModel
    )
    # Patch in lambda_decoherence param for consistency
    exp.lambda_decoherence = np.array([1e-3])
    return exp

@pytest.fixture
def analytic_coherence_model(simple_spins, simple_experiment):
    """AnalyticCoherenceModel with 5 13C nuclear spins and one Hahn-echo experiment."""
    model = AnalyticCoherenceModel(spins=simple_spins.spins, experiment=simple_experiment)
    return model

@pytest.fixture
def multi_experiment():
    """Experiment with two pulse sequences and different magnetic fields."""
    exp = Experiment(
        num_exps=2,
        num_pulses=[1, 2],
        mag_field=[100.0, 200.0],
        noise=[0.0, 0.0],
        timepoints=[[0.0, 1.0], [0.0, 1.0]],
         
        lambda_decoherence=[1.0, 2.0],
    )
    return exp

@pytest.fixture
def analytic_coherence_model_multi(simple_spins, multi_experiment):
    """AnalyticCoherenceModel with 5 spins and two experiments."""
    return AnalyticCoherenceModel(spins=simple_spins.spins, experiment=multi_experiment)

@pytest.fixture
def discrete_rjmcmc_proposal(simple_spins, default_params):
    """Discrete RJMCMC proposal object for birth/death moves with 5-spin SpinBath."""
    spin_bath = simple_spins  # SpinBath fixture
    max_spins = len(spin_bath.spins)
    spin_inds = np.arange(3)  # start with 3 active spins as an example

    proposal = DiscreteRJMCMCProposal(
        spin_bath=spin_bath,
        max_spins=max_spins,
        spin_inds=spin_inds,
        birth=None,             # random birth/death move
        params=default_params
    )
    return proposal

@pytest.fixture
def gaussian_loglike_multi(analytic_coherence_model_multi):
    """Gaussian log-likelihood for AnalyticCoherenceModel with multiple experiments."""
    model = analytic_coherence_model_multi
    spins = model.spins
    experiment = model.experiment

    data = model.compute_coherence()
    error_model = L2Error(data)

    return GaussianLogLikelihoodFromError(
        data=data,
        base_error_model=error_model,
        sigma_sq=1e-4,
        as_negative=True
    )

def test_acceptprob_rjmcmc_returns_numeric(
    gaussian_loglike_multi,
    discrete_rjmcmc_proposal,
    analytic_coherence_model_multi,  # injects the instance
):
    """Check that AcceptProbRJMCMC.compute() returns a finite numeric value."""

    # Unpack Gaussian log-likelihood fixture
    error_model = gaussian_loglike_multi
    proposal = discrete_rjmcmc_proposal

    # Propose a birth/death move
    proposal.propose()

    # Pass the forward model instance (not the fixture function)
    accept_prob = AcceptProbRJMCMC(
        forward_model=analytic_coherence_model_multi,
        error_model=error_model,
        proposal=proposal,
        temperature=1.0,
    )

    prob = accept_prob.compute()
    assert isinstance(prob, float)
    assert np.isfinite(prob)

