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


###----------Test different likelihood models together----------###

def test_shared_params_across_likelihood_models():
    # Create Params with all needed variables
    param_names = ["lambda", "r", "p"]
    param_vals = [3.0, 5.0, 0.4]
    discrete = [False, False, False]
    params = Params(param_names, param_vals, discrete)

    # Observed count data
    data = np.array([0, 1, 2, 3, 4])

    # ----------------- Poisson -----------------
    pfm = PoissonForwardModel(params, lambda_param="lambda")
    pll = PoissonLogLikelihood(pfm, data, lambda_param="lambda")
    expected_poisson_ll = np.sum(poisson.logpmf(data, mu=3.0))
    computed_poisson_ll = pll.log_likelihood()
    assert np.isclose(computed_poisson_ll, expected_poisson_ll), "Poisson log-likelihood mismatch"

    # ----------------- Negative Binomial -----------------
    nbfm = NegativeBinomialForwardModel(params, r_param="r", p_param="p")
    nbll = NegativeBinomialLogLikelihood(nbfm, data, r_param="r", p_param="p")
    expected_nb_ll = np.sum(nbinom.logpmf(data, n=5.0, p=0.4))
    computed_nb_ll = nbll.log_likelihood()
    assert np.isclose(computed_nb_ll, expected_nb_ll), "Negative Binomial log-likelihood mismatch"


def test_poisson_and_nb_models_on_same_param():
    # One shared parameter: interpret as lambda for Poisson, r for NB
    param_names = ["shared_param"]
    param_vals = [4.0]  # λ or r
    discrete = [False]
    params = Params(param_names, param_vals, discrete)

    # For NB we'll need to add another param for 'p'
    params = Params(
        names=["shared_param", "p"],
        vals=[4.0, 0.6],  # shared_param = λ or r
        discrete=[False, False]
    )

    data = np.array([0, 1, 2, 3])

    # Poisson model using 'shared_param' as λ
    pfm = PoissonForwardModel(params, lambda_param="shared_param")
    pll = PoissonLogLikelihood(pfm, data, lambda_param="shared_param")
    expected_poisson_ll = np.sum(poisson.logpmf(data, mu=4.0))
    assert np.isclose(pll.log_likelihood(), expected_poisson_ll), "Poisson likelihood mismatch"

    # Negative Binomial model using 'shared_param' as r and 'p' as is
    nbfm = NegativeBinomialForwardModel(params, r_param="shared_param", p_param="p")
    nbll = NegativeBinomialLogLikelihood(nbfm, data, r_param="shared_param", p_param="p")
    expected_nb_ll = np.sum(nbinom.logpmf(data, n=4.0, p=0.6))
    assert np.isclose(nbll.log_likelihood(), expected_nb_ll), "Negative Binomial likelihood mismatch"
