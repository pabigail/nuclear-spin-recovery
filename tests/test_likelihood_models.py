import sys
from pathlib import Path
import numpy as np
import pytest 
from scipy.stats import poisson
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel, PoissonForwardModel
from hymcmcpy.likelihood_models import LikelihoodModel, PoissonLogLikelihood


###----------LikelihoodModel----------###
@pytest.fixture
def poisson_params():
    """Params instance with one lambda parameter"""
    names = ["lambda"]
    vals = [4.0]
    discrete = [False]
    return Params(names, vals, discrete)

@pytest.fixture
def poisson_forward_model(poisson_params):
    """PoissonForwardModel using the 'lambda' parameter"""
    return PoissonForwardModel(poisson_params, subset_param_names=["lambda"])

def test_likelihood_model_is_abstract(poisson_forward_model):
    """Test that you cannot instantiate LikelihoodModel directly"""
    with pytest.raises(TypeError):
        LikelihoodModel(poisson_forward_model, [1, 2, 3])


###----------PoissonLogLikelihood----------###

@pytest.fixture
def params():
    names = ["lambda", "theta"]
    vals = [4.0, 2.5]
    discrete = [False, False]
    return Params(names, vals, discrete)

@pytest.fixture
def forward_model(params):
    return PoissonForwardModel(params=params, subset_param_names=["lambda"])

@pytest.fixture
def data():
    return np.array([2, 3, 4, 5])

@pytest.fixture
def likelihood(forward_model, data):
    return PoissonLogLikelihood(forward_model, data)


def test_valid_initialization(likelihood):
    assert isinstance(likelihood, PoissonLogLikelihood)


def test_invalid_data_negative(forward_model):
    with pytest.raises(ValueError, match="Poisson data must be non-negative integers."):
        PoissonLogLikelihood(forward_model, data=[2, -1, 3])


def test_invalid_data_non_integer(forward_model):
    with pytest.raises(ValueError, match="Poisson data must be non-negative integers."):
        PoissonLogLikelihood(forward_model, data=[1.2, 2.5, 3])


class DummyForwardModel(ForwardModel):
    def __init__(self, params, subset_param_names):
        super().__init__(params, subset_param_names)

    def compute(self, *args, **kwargs):
        return np.ones(len(args[0])) if args else np.ones(1)


def test_multiple_subset_params_invalid(params):
    with pytest.raises(ValueError, match="PoissonLogLikelihood requires exactly one subset parameter name."):
        bad_model = DummyForwardModel(params, subset_param_names=["lambda", "theta"])
        PoissonLogLikelihood(bad_model, data=[2, 3])


def test_log_likelihood_matches_scipy(likelihood, data, params):
    lam = params["val"][params["name"] == "lambda"][0]
    expected_log_likelihood = np.sum(poisson.logpmf(data, mu=lam))
    assert np.isclose(likelihood.log_likelihood(), expected_log_likelihood)
