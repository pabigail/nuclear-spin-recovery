import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import poisson, nbinom



class ForwardModel(ABC):
    """
    Abstract base class for forward models that map parameter sets to simulated data-like output.

    This class should not be instantiated directly. Subclasses are responsible for defining how
    named parameters in a `Params` object map to internal model variables used in the computation.

    Args:
        params (Params): The parameter set to be used in the forward model (typically to be inferred via MCMC).
        **kwargs (dict): Additional fixed model configuration parameters. These remain constant across model instances and 
                         are not varied during MCMC or inference. Subclasses should use these to specify how to interpret
                         parameters from the `Params` object.
    
    Notes:
        Subclasses must define how variables from `params["name"]` are used internally in the model's logic. This allows
        flexible reuse of the forward model code with different parameter configurations.
    """
    
    def __init__(self, params, **kwargs):
        self.params = params

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Generates data-like output given specific parameters.
        """
        pass



###----------Generic Examples----------###

class PoissonForwardModel(ForwardModel):
    """
    Forward model for computing the Poisson probability mass function (PMF) 
    using a parameter from the `Params` object as the Poisson rate (λ).

    This model allows flexible specification of which parameter to treat as the rate
    by passing the name of the parameter in `Params`. This decouples the internal
    logic of the forward model from the specific naming used in parameter sets.

    Args:
        params (Params): An instance of the `Params` class containing parameter names and values.
        lambda_param (str): The name of the parameter in `Params["name"]` to use as the Poisson rate λ.

    Raises:
        TypeError: If `lambda_param` is not a string.
        ValueError: If `lambda_param` is not found in `Params["name"]`.
    """
    def __init__(self, params, lambda_param="lambda"):
        if not isinstance(lambda_param, str):
            raise TypeError(f"`lambda_param` must be a string, got {type(lambda_param).__name__}")

        if lambda_param not in params["name"]:
            raise ValueError(f"`lambda_param` '{lambda_param}' not found in Params['name']")

        super().__init__(params, lambda_param=lambda_param)
        self.lambda_param = lambda_param


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
        idx = np.where(self.params["name"] == self.lambda_param)[0]

        # Extract lambda value from params
        lambda_val = self.params["val"][idx[0]]

        # Return Poisson PMF
        return poisson.pmf(k, lambda_val)


class NegativeBinomialForwardModel(ForwardModel):
    """
    Forward model for computing the Negative Binomial probability mass function (PMF) 
    using parameters from the `Params` object.

    The model interprets two parameters from `Params`: one as the number of failures `r` 
    and one as the success probability `p`. These can be flexibly specified via their 
    names in the `Params` array.

    Args:
        params (Params): An instance of the `Params` class containing parameter names and values.
        r_param (str): The name of the parameter to be used as the number of failures `r`.
        p_param (str): The name of the parameter to be used as the success probability `p`.

    Raises:
        TypeError: If `r_param` or `p_param` are not strings.
        ValueError: If either parameter name is not found in `Params["name"]`.
    """

    def __init__(self, params, r_param="r", p_param="p"):
        if not isinstance(r_param, str):
            raise TypeError(f"`r_param` must be a string, got {type(r_param).__name__}")
        if not isinstance(p_param, str):
            raise TypeError(f"`p_param` must be a string, got {type(p_param).__name__}")

        if r_param not in params["name"]:
            raise ValueError(f"`r_param` '{r_param}' not found in Params['name']")
        if p_param not in params["name"]:
            raise ValueError(f"`p_param` '{p_param}' not found in Params['name']")

        super().__init__(params, r_param=r_param, p_param=p_param)
        self.r_param = r_param
        self.p_param = p_param

    def compute(self, k):
        """
        Compute the Negative Binomial PMF at count(s) `k`.

        The model uses the values from `Params` corresponding to `r_param` and `p_param` 
        as the number of failures and success probability, respectively.

        Args:
            k (int or array-like): Observed count(s) at which to evaluate the PMF.

        Returns:
            float or array-like: PMF value(s) evaluated at `k`.
        """
        idx_r = np.where(self.params["name"] == self.r_param)[0]
        idx_p = np.where(self.params["name"] == self.p_param)[0]

        r_val = self.params["val"][idx_r[0]]
        p_val = self.params["val"][idx_p[0]]

        return nbinom.pmf(k, r_val, p_val)
