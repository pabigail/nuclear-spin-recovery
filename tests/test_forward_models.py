import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import poisson, nbinom
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel, PoissonForwardModel, NegativeBinomialForwardModel

###----------ForwardModel----------###

class DummyModel(ForwardModel):
    def compute(self, *args, **kwargs):
        return "ok"


def test_forward_model_is_abstract():
    params = Params(['a'], [1.0], [False])
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ForwardModel(params)


def test_valid_param_names_length_2():
    params = Params(['a', 'b'], [1.0, 2.0], [False, True])
    model = DummyModel(params)
    assert model.params['name'][0] == 'a'
    assert model.params['name'][1] == 'b'
    assert model.params['val'][0] == 1.0
    assert model.params['val'][1] == 2.0
    assert model.params['discrete'][0] == False
    assert model.params['discrete'][1] == True
    assert model.compute() == "ok"


###----------Fixtures----------###
@pytest.fixture
def simple_params():
    names = ['lambda', 'r', 'p']
    vals = [3.0, 5.0, 0.4]
    discrete = [False, True, False]
    return Params(names, vals, discrete)


###----------PoissonForwardModel----------###

def test_poisson_forward_model_valid(simple_params):
    model = PoissonForwardModel(simple_params, lambda_param='lambda')
    result = model.compute(k=2)
    expected = poisson.pmf(2, mu=3.0)
    assert np.isclose(result, expected)


def test_poisson_invalid_param_name(simple_params):
    with pytest.raises(ValueError):
        PoissonForwardModel(simple_params, lambda_param='not_in_params')


def test_poisson_invalid_param_type(simple_params):
    with pytest.raises(TypeError):
        PoissonForwardModel(simple_params, lambda_param=123)  # not a string


def test_two_poisson_models_same_params():
    # Create shared Params object with two rate parameters
    names = ['lambda_1', 'lambda_2']
    vals = [2.0, 5.0]
    discrete = [False, False]
    params = Params(names, vals, discrete)

    # Create two models, each using a different lambda
    model1 = PoissonForwardModel(params, lambda_param='lambda_1')
    model2 = PoissonForwardModel(params, lambda_param='lambda_2')

    # Compute PMF at k=3
    result1 = model1.compute(k=3)
    result2 = model2.compute(k=3)

    # Expected values using SciPy directly
    expected1 = poisson.pmf(3, mu=2.0)
    expected2 = poisson.pmf(3, mu=5.0)

    # Assert that results match expected values
    assert np.isclose(result1, expected1), "Model 1 did not return expected value"
    assert np.isclose(result2, expected2), "Model 2 did not return expected value"
    assert not np.isclose(result1, result2), "Models should not return the same result"


###----------NegativeBinomialForwardModel----------###
def test_negative_binomial_forward_model_valid(simple_params):
    model = NegativeBinomialForwardModel(simple_params, r_param='r', p_param='p')
    result = model.compute(k=3)
    expected = nbinom.pmf(3, n=5.0, p=0.4)
    assert np.isclose(result, expected)


def test_negative_binomial_invalid_r_name(simple_params):
    with pytest.raises(ValueError):
        NegativeBinomialForwardModel(simple_params, r_param='nope', p_param='p')


def test_negative_binomial_invalid_p_name(simple_params):
    with pytest.raises(ValueError):
        NegativeBinomialForwardModel(simple_params, r_param='r', p_param='nope')


def test_negative_binomial_invalid_param_types(simple_params):
    with pytest.raises(TypeError):
        NegativeBinomialForwardModel(simple_params, r_param=3, p_param='p')

    with pytest.raises(TypeError):
        NegativeBinomialForwardModel(simple_params, r_param='r', p_param=None)
