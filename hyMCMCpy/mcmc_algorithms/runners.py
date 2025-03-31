import numpy as np
from hymcmcpy.params import Params

class MCMC_iterations(np.ndarray):
    """
    A structured NumPy array to store parameter values at different iterations of a hybrid MCMC walker.

    This class tracks the parameters, the algorithm used for each step, and whether the proposed step was accepted.
    It ensures that all input arrays have the same length and stores them as a structured NumPy array.

    Attributes:
        _dtype_params (np.dtype): Structured dtype with fields:
            - 'algorithm' (str): The MCMC algorithm used at each step.
            - 'accept' (bool): Whether the proposed step was accepted.
            - 'params' (Params): The parameter values at each iteration.

    Args:
        algorithms (array-like): A list or array of algorithm names (strings) used in each iteration.
        accepts (array-like): A boolean array indicating whether each proposed step was accepted.
        params (array-like): An array of `Params` objects representing the system parameters at each step.

    Raises:
        ValueError: If input arrays do not have the same length.

    Returns:
        MCMC_iterations: A structured array storing MCMC step details.
    """


    _dtype_params = np.dtype([('algorithm', np.str_, 50),
                              ('accept', np.bool_),
                              ('params', Params)])

    def __new__(cls, algorithms, accepts, params):

        # ensure inputs are array-like and have same shape
        algorithms = np.asarray(algorithms, dtype='U50').reshape(-1)
        accepts = np.asarray(accepts, dtype=bool).reshape(-1)
        params = np.asarray(params, dtype=Params).reshape(-1)

        if not (len(algorithms) == len(accepts) == len(params)):
            raise ValueError("All input arrays must be the same length")

        obj = np.empty(len(algorithms), dtype=cls._dtype_params).view(cls)
        obj['algorithm'] = algorithms
        obj['accept'] = accepts
        obj['params'] = params

        return obj
