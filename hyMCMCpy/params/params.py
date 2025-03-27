import numpy as np


class Params(np.ndarray):
    """A subclass of `ndarray` for storing system parameters.

    This class provides a structured `ndarray` with a fixed datatype to store
    parameter names, values, and whether they belong to a discrete or continuous domain.

    Attributes:
        _dtype_params (np.dtype): The structured data type of the array, which includes:
        - `name` (str): The name of the parameter.
        - `val` (float): The numerical value of the parameter.
        - `discrete` (bool): Whether the parameter is from a discrete domain (`True`)
              or a continuous domain (`False`).

    Args:
        names (array-like): An array of parameter names (strings).
        vals (array-like): An array of parameter values (floats).
        discrete (array-like): A boolean array indicating whether each parameter is discrete.
    
    Raises:
        ValueError: If the input arrays have inconsistent lengths.
        
    """

    _dtype_params = np.dtype([('name', np.str_, 50), # max string length 50
                              ('val', np.float64),
                              ('discrete', np.bool_)])

    def __new__(cls, names, vals, discrete):
       
        # ensure inputs are numpy arrays and have same shape
        names = np.asarray(names, dtype='U50').reshape(-1)
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        discrete = np.asarray(discrete, dtype=bool).reshape(-1)

        # make sure lengths are the same
        if not (len(names) == len(vals) == len(discrete)):
            raise ValueError("All input arrays must be same length")

        obj = np.empty(len(names), dtype=cls._dtype_params).view(cls)
        obj['name'] = names
        obj['val'] = vals
        obj['discrete'] = discrete

        return obj




