import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import poisson
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel, PoissonForwardModel

###----------ForwardModel----------###

class DummyModel(ForwardModel):
    def compute(self, *args, **kwargs):
        return "ok"


def test_forward_model_is_abstract():
    params = Params(['a'], [1.0], [False])
    subset_param_names = ['a']
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ForwardModel(params, subset_param_names)


def test_valid_subset_param_names_does_not_raise():
    params = Params(['a', 'b'], [1.0, 2.0], [False, False])
    subset = ['a']
    model = DummyModel(params, subset)
    assert model.params['name'][0] == 'a'
    assert model.subset_param_names == ['a']
    assert model.compute() == "ok"


def test_entire_param_list_is_valid_subset():
    names = ['x', 'y']
    vals = [1.0, 2.0]
    discrete = [False, True]
    params = Params(names, vals, discrete)

    model = DummyModel(params, subset_param_names=names)
    assert set(model.subset_param_names) == set(names)


def test_invalid_subset_param_names_raises():
    params = Params(['a', 'b'], [1.0, 2.0], [False, False])
    subset = ['c']  # Not in params["name"]
    with pytest.raises(ValueError, match="subset_param_names must be a subset of params\\['name'\\]"):
        DummyModel(params, subset)


###----------PoissonForwardModel----------###

def test_valid_instantiation_and_compute_scalar():
    params = Params(['rate'], [4.0], [False])
    model = PoissonForwardModel(params, subset_param_names=['rate'])
    result = model.compute(2)
    expected = poisson.pmf(2, 4.0)
    assert np.isclose(result, expected)


def test_valid_instantiation_and_compute_array():
    params = Params(['lambda'], [3.0], [False])
    model = PoissonForwardModel(params, subset_param_names=['lambda'])
    result = model.compute([0, 1, 2])
    expected = poisson.pmf([0, 1, 2], 3.0)
    np.testing.assert_allclose(result, expected)


def test_invalid_subset_length():
    params = Params(['lambda'], [3.0], [False])
    with pytest.raises(ValueError, match="PoissonForwardModel requires exactly one parameter"):
        PoissonForwardModel(params, subset_param_names=['lambda', 'extra'])


def test_subset_param_name_not_in_params():
    params = Params(['theta'], [2.0], [True])
    subset_param_names=['lambda']  # Not present
    with pytest.raises(ValueError, match="subset_param_names must be a subset of params\\['name'\\]"):
        PoissonForwardModel(params, subset_param_names)


def test_compute_uses_correct_param_value():
    # We test that the correct param is used even when other params exist
    params = Params(['alpha', 'rate', 'beta'], [1.0, 5.0, 2.0], [False, False, False])
    model = PoissonForwardModel(params, subset_param_names=['rate'])
    result = model.compute(3)
    expected = poisson.pmf(3, 5.0)
    assert np.isclose(result, expected)


def test_multiple_instantiations_with_different_subset_param_names():
    # Create a Params object with multiple parameters
    params = Params(['rate', 'lambda', 'scale'], [3.0, 5.0, 2.0], [False, False, False])
    
    # Instantiate two PoissonForwardModel objects with different subset_param_names
    model1 = PoissonForwardModel(params, subset_param_names=['rate'])
    model2 = PoissonForwardModel(params, subset_param_names=['lambda'])
    
    # Compute the Poisson PMF for each model, using different observed counts (k)
    result1 = model1.compute(2)
    result2 = model2.compute(3)

    # Check if each model computes with the correct parameter
    expected1 = poisson.pmf(2, 3.0)  # model1 should use "rate"
    expected2 = poisson.pmf(3, 5.0)  # model2 should use "lambda"

    # Ensure the results match expectations
    assert np.isclose(result1, expected1), f"Expected {expected1}, got {result1}"
    assert np.isclose(result2, expected2), f"Expected {expected2}, got {result2}"

    # Verify that the parameters in `params` haven't been modified unexpectedly
    assert np.isclose(params['val'][0], 3.0), "Parameter 'rate' should remain 3.0"
    assert np.isclose(params['val'][1], 5.0), "Parameter 'lambda' should remain 5.0"
    assert np.isclose(params['val'][2], 2.0), "Parameter 'scale' should remain 2.0"
