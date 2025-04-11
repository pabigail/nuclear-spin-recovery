import sys
from pathlib import Path
import numpy as np
import pytest 
from scipy.stats import poisson, nbinom
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel, PoissonForwardModel, NegativeBinomialForwardModel
from hymcmcpy.likelihood_models import LikelihoodModel, PoissonLogLikelihood, NegativeBinomialLogLikelihood


###----------LikelihoodModel----------###

def test_likelihood_model_abstract_instantiation():
    with pytest.raises(TypeError):
        LikelihoodModel(forward_model=None, data=[])


###----------PoissonLogLikelihood----------###

def test_poisson_log_likelihood_correctness():
    # Params with two lambda values
    param_names = ["lambda_1", "lambda_2"]
    vals = [2.0, 5.0]
    discrete = [False, False]
    params = Params(param_names, vals, discrete)

    # Observed data
    data = np.array([0, 1, 2, 3])

    # Forward model using lambda_1
    fm1 = PoissonForwardModel(params, lambda_param="lambda_1")
    pll1 = PoissonLogLikelihood(fm1, data, lambda_param="lambda_1")
    expected_ll1 = np.sum(poisson.logpmf(data, mu=2.0))
    assert np.isclose(pll1.log_likelihood(), expected_ll1)

    # Forward model using lambda_2
    fm2 = PoissonForwardModel(params, lambda_param="lambda_2")
    pll2 = PoissonLogLikelihood(fm2, data, lambda_param="lambda_2")
    expected_ll2 = np.sum(poisson.logpmf(data, mu=5.0))
    assert np.isclose(pll2.log_likelihood(), expected_ll2)

def test_poisson_log_likelihood_invalid_param():
    params = Params(["lambda"], [3.0], [False])
    data = [1, 2, 3]

    with pytest.raises(ValueError):
        PoissonLogLikelihood(PoissonForwardModel(params, lambda_param="lambda"), data, lambda_param="not_a_param")


def test_poisson_log_likelihood_invalid_data():
    params = Params(["lambda"], [3.0], [False])
    with pytest.raises(ValueError):
        PoissonLogLikelihood(PoissonForwardModel(params), data=[-1, 0.5, 2])


###----------NegativeBinomialLogLikelihood----------###

def test_negative_binomial_log_likelihood_correctness():
    # Parameters for r and p
    param_names = ["r", "p"]
    vals = [10.0, 0.25]
    discrete = [False, False]
    params = Params(param_names, vals, discrete)

    # Observed count data
    data = np.array([0, 1, 2, 5])

    # Forward model
    nb_fm = NegativeBinomialForwardModel(params, r_param="r", p_param="p")
    nb_ll = NegativeBinomialLogLikelihood(nb_fm, data, r_param="r", p_param="p")

    expected_ll = np.sum(nbinom.logpmf(data, n=10.0, p=0.25))
    assert np.isclose(nb_ll.log_likelihood(), expected_ll)


def test_negative_binomial_log_likelihood_invalid_params():
    params = Params(["r"], [10.0], [False])
    data = [0, 1, 2]

    with pytest.raises(ValueError):
        NegativeBinomialLogLikelihood(NegativeBinomialForwardModel(params, r_param="r", p_param="p"),
                                      data, r_param="r", p_param="p")


def test_negative_binomial_log_likelihood_invalid_data():
    params = Params(["r", "p"], [10.0, 0.5], [False, False])
    with pytest.raises(ValueError):
        NegativeBinomialLogLikelihood(
            NegativeBinomialForwardModel(params, r_param="r", p_param="p"),
            data=[-1, 1.5, 2]
        )
