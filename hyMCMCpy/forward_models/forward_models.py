import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import poisson, nbinom



class ForwardModel(ABC):
    """
    This abstract class is takes in Params and other arguments to return data-like output. 
    Should not be initialized directly (use an implemented subclass)

    Args:
        params (Params): the parameter that seek to recover using MCMC methods
        subset_param_names (list[str]): Names of parameters used by this model.
        **kwargs (dict): specific parameters that are unchanging across instatiations (and not varying during MCMC or inference)

    """
    def __init__(self, params, subset_param_names, **kwargs):
        self.params = params
        self.subset_param_names = subset_param_names

        # check that subset_param_names is a subset of param["name"]
        all_names = set(params["name"])
        subset_names = set(subset_param_names)

        if not subset_names.issubset(all_names):
            raise ValueError("subset_param_names must be a subset of params['name'].")


    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Generates data-like output given specific parameters.
        """
        pass



###----------Generic Examples----------###

class PoissonForwardModel(ForwardModel):
    """
    Forward model for the Poisson distribution, using the rate parameter (lambda) from the `Params` class.

    Args:
        params (Params): An instance of the `Params` class
        subset_param_names (list[str]): Name of the parameter for PoissonForwardModel to act upon (must be length 1)
        **kwargs: Additional fixed parameters (optional).
    
    Raises:
        ValueError: If `params` does not contain an entry with the name "lambda".
    """
    def __init__(self, params, subset_param_names, **kwargs):
        if len(subset_param_names) != 1:
            raise ValueError("PoissonForwardModel requires exactly one parameter to act upon (given by subset_param_name)")

        super().__init__(params, subset_param_names, **kwargs)
        self.param_name = subset_param_names[0]


    def compute(self, k):
        """
        Compute the Poisson probability mass function (PMF) at a given k.

        Args:
            k (array-like or scalar): The observed count or counts.
        
        Returns:
            float or array-like: The Poisson PMF evaluated at k.
        
        Raises:
            ValueError: If `params` does not contain a valid "lambda" entry.
        """
        # get index of specified parameter
        idx = np.where(self.params["name"] == self.param_name)[0]
        if len(idx) == 0:
            raise ValueError(f"Parameter '{self.param_name}' note found in params.")

        # Extract lambda value from params
        lambda_val = self.params["val"][idx[0]]

        # Return Poisson PMF
        return poisson.pmf(k, lambda_val)


