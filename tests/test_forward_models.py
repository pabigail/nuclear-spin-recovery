import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import poisson
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params
from hymcmcpy.forward_models import ForwardModel, PoissonForwardModel

###----------ForwardModel----------###

def test_forward_model_cannot_be_instantiated():
    with pytest.raises(TypeError, match="Can't instantiate abstract class ForwardModel"):
        ForwardModel(params=None)  # Abstract class must raise on instantiation


###----------PoissonForwardModel----------###

def test_poisson_forward_model_computes_pmf_correctly():
    names = ['lambda']
    vals = [3.0]
    discrete = [False]
    params = Params(names, vals, discrete)

    model = PoissonForwardModel(params)

    k = np.array([0, 1, 2, 3])
    expected = poisson.pmf(k, mu=3.0)
    output = model.compute(k)

    np.testing.assert_allclose(output, expected, rtol=1e-6)

def test_poisson_forward_model_scalar_input():
    params = Params(['lambda'], [5.0], [False])
    model = PoissonForwardModel(params)

    k = 2
    expected = poisson.pmf(k, mu=5.0)
    output = model.compute(k)

    assert np.isclose(output, expected)

def test_poisson_forward_model_raises_if_lambda_missing():
    params = Params(['alpha'], [1.0], [False])
    model = PoissonForwardModel(params)

    with pytest.raises(ValueError, match="params must contain one entry with the name 'lambda'"):
        model.compute(1)
