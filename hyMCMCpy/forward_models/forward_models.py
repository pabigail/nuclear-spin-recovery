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
    Forward model for the Poisson distribution using a single parameter from `Params`.

    This model uses the specified parameter (given in `subset_param_names`) as the 
    rate parameter (λ) for the Poisson distribution.

    Args:
        params (Params): An instance of the `Params` class containing all system parameters.
        subset_param_names (list[str]): A list with a single parameter name to be used 
                                        as the Poisson rate (must have length 1).
        **kwargs: Optional additional fixed parameters.

    Raises:
        ValueError: If `subset_param_names` does not contain exactly one parameter name.
    """

    def __init__(self, params, subset_param_names, **kwargs):
        if len(subset_param_names) != 1:
            raise ValueError("PoissonForwardModel requires exactly one parameter to act upon (given by subset_param_name)")

        super().__init__(params, subset_param_names, **kwargs)
        self.param_name = subset_param_names[0]

    def compute(self, k):
        """
        Compute the Poisson probability mass function (PMF) at the given count(s) `k`.

        The model evaluates the PMF using the value of the parameter specified in
        `subset_param_names`, treating it as the Poisson rate λ.

        Args:
            k (array-like or scalar): The observed count(s) at which to evaluate the PMF.

        Returns:
            float or array-like: The value(s) of the Poisson PMF evaluated at `k`.

        Raises:
            ValueError: If the specified parameter is not found in `params`.
        """
        # get index of specified parameter
        idx = np.where(self.params["name"] == self.param_name)[0]

        # Extract lambda value from params
        lambda_val = self.params["val"][idx[0]]

        # Return Poisson PMF
        return poisson.pmf(k, lambda_val)


