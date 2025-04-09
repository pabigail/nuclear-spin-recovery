import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.append(str(Path(__file__).resolve().parent.parent))
from hymcmcpy.params import Params


def test_empty_raises_error():
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        Params([],[],[])


@pytest.mark.parametrize("names, vals, discrete", [
    (["alpha"], [1.0, 2.0], [True, False]),
    (["alpha", "beta"], [1.0], [True, False]),
    (["alpha", "beta"], [1.0, 2.0], [True])])
def test_mismatch_sizes_raises_error(names, vals, discrete):
    with pytest.raises(ValueError, match="All input arrays must be same length"):
        Params(names, vals, discrete)


def test_valid_inputs_single():
    names = ["alpha"]
    vals = [-1.8]
    discrete = [False]
    params = Params(names, vals, discrete)
    assert isinstance(params, Params)
    assert len(params) == 1
    np.testing.assert_array_equal(params["name"], np.array(names, dtype='U50'))
    np.testing.assert_array_almost_equal(params["val"], np.array(vals, dtype=np.float64))
    np.testing.assert_array_equal(params["discrete"], np.array(discrete, dtype=bool))


def test_valid_inputs_longer():
    names = ["alpha", "beta", "gamma", "delta"]
    vals = [3.2, 0.0, np.pi, -1]
    discrete = [True, False, True, True]
    params = Params(names, vals, discrete)
    assert isinstance(params, Params)
    assert len(params) == 4
    np.testing.assert_array_equal(params["name"], np.array(names, dtype='U50'))
    np.testing.assert_array_almost_equal(params["val"], np.array(vals, dtype=np.float64))
    np.testing.assert_array_equal(params["discrete"], np.array(discrete, dtype=bool))


def test_structured_field_access_by_index():
    names = ['x', 'y', 'z']
    vals = [1.1, 2.2, 3.3]
    discrete = [True, False, True]
    params = Params(names, vals, discrete)

    # Access first entry and check individual fields
    first = params[0]
    assert first['name'] == 'x'
    assert first['val'] == 1.1
    assert first['discrete'] == True

    # Access second entry
    second = params[1]
    assert second['name'] == 'y'
    assert second['val'] == 2.2
    assert second['discrete'] == False

    # Access entire field column across all entries
    names_field = params['name']
    vals_field = params['val']
    discrete_field = params['discrete']

    np.testing.assert_array_equal(names_field, np.array(names, dtype='U50'))
    np.testing.assert_array_almost_equal(vals_field, np.array(vals, dtype=np.float64))
    np.testing.assert_array_equal(discrete_field, np.array(discrete, dtype=bool))


def test_value_update_reflects_in_params():
    names = ['x', 'y']
    vals = [1.0, 2.0]
    discrete = [False, True]
    params = Params(names, vals, discrete)

    # Update 'val' of the first parameter
    params[0]['val'] = 42.0

    # Ensure it's updated in the array
    assert params[0]['val'] == 42.0
    assert params['val'][0] == 42.0

    # Update via field view
    params['val'][1] = 99.0
    assert params[1]['val'] == 99.0
