import numpy as np

class Params(np.ndarray):
    """
    Subclass of ``ndarray`` containing information about the parameters in the system

    The subclass has fixed structured datatype::

        _dtype_params = np.dtype([('name', np.unicode_)
                                  ('val', np.float64),
                                  ('discrete', np.bool_)])

    Args:
        name (array-like): 
            Array of names of parameters

        val (array-like):
            Array of the current values of the parameters

        discrete (array-like):
            Boolean array if the parameter comes from a discrete (True) or continuous (False) domain
            
    """

    _dtype_params = np.dtype([('name', np.unicode_, 50), # max string length 50
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




