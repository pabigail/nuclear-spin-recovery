import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import poisson, nbinom



class ForwardModel(ABC):
    """
    This abstract class is takes in Params and other arguments to return data-like output. 
    Should not be initialized directly (use an implemented subclass)

    Args:
        params (Params): the parameter that seek to recover using MCMC methods
        **kwargs (dict): specific parameters that are unchanging across instatiations

    """
    def __init__(self, params, **kwargs):
        self.params = params

    @abstractmethod
    def compute(self):
        """
        Generates data-like output given specific parameters.
        """
        pass



###----------Generic Examples----------###

class PoissonForwardModel(ForwardModel):
    """
    Forward model for the Poisson distribution, using the rate parameter (lambda) from the `Params` class.

    Args:
        params (Params): An instance of the `Params` class, which should contain one entry 
                         with the name "lambda" representing the rate parameter for the Poisson distribution.
        **kwargs: Additional fixed parameters (optional).
    
    Raises:
        ValueError: If `params` does not contain an entry with the name "lambda".
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)


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
        # Ensure the params contain a valid "lambda" entry
        if "lambda" not in self.params["name"]:
            raise ValueError("params must contain one entry with the name 'lambda'.")
        
        # Extract lambda value from params
        lambda_val = self.params["val"][0]

        # Return Poisson PMF
        return poisson.pmf(k, lambda_val)


