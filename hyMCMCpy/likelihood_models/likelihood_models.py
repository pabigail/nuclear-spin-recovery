import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import poisson, nbinom

class LikelihoodModel(ABC):
    """
    Abstract base class for likelihood models.

    This class connects a forward model with observed data to compute
    a likelihood function for statistical inference (e.g., in MCMC).

    The user must specify which parameter names (as defined in the `Params` object)
    should be passed to the forward model. This ensures the correct mapping between
    model inputs and stored parameters during evaluation.

    Args:
        forward_model (ForwardModel): An instance of a forward model used
            to generate data-like output given parameters.
        data (array-like): Observed data against which the model's output
            will be compared.
        **kwargs: Additional keyword arguments that specify configuration,
            including parameter names required by the forward model.
    """
    def __init__(self, forward_model, data, **kwargs):
        self.forward_model = forward_model
        self.data = data

    @abstractmethod
    def log_likelihood(self):
        """
        Compute the log likelihood of the observed data given the forward model.
        Almost every real MCMC algorithm works better on log-likelihoods instead of 
        likelihoods due to underflow in probability products
        
        Notes:
        The forward model internally uses parameter names passed at initialization
        to extract values from `Params`. These names must align with the names
        declared in the `Params` object.

        Returns:
            float: The likelihood value.
        """
        pass



class PoissonLogLikelihood(LikelihoodModel):
    """
    Poisson log-likelihood model.

    Computes the log-likelihood of the observed count data under a
    Poisson-distributed forward model, suitable for numerical stability
    and MCMC sampling.

    Args:
        forward_model (ForwardModel): A forward model instance that computes
            P(k | lambda) via its `compute()` method or provides parameters.
        data (array-like): Observed count data, assumed to be integers.
        **kwargs: Additional keyword arguments (optional).
    """
    def __init__(self, forward_model, data, lambda_param="lambda"):
        if not isinstance(lambda_param, str):
            raise TypeError(f"`lambda_param` must be a string, got {type(lambda_param).__name__}")

        if lambda_param not in forward_model.params["name"]:
            raise ValueError(f"`lambda_param` '{lambda_param}' not found in forward_model.params['name']")

        super().__init__(forward_model, data)
        self.lambda_param = lambda_param

        data = np.asarray(data)
        if np.any(data < 0) or not np.all(np.equal(np.mod(data, 1), 0)):
            raise ValueError("Poisson data must be non-negative integers.")

        self.data = data

    def log_likelihood(self):
        """
        Compute the Poisson log-likelihood.

        This computes:
            log L = sum_k [ k * log(lambda) - lambda - log(k!) ]

        Returns:
            float: The sum of the log-likelihoods over all data points.
        """
        # Retrieve the lambda value from the params by name
        idx = np.where(self.forward_model.params["name"] == self.lambda_param)[0]
        lam = self.forward_model.params["val"][idx[0]]

        # Vectorized computation of log PMF
        log_pmf = poisson.logpmf(self.data, mu=lam)
        return np.sum(log_pmf)


class NegativeBinomialLogLikelihood(LikelihoodModel):
    """
    Negative Binomial log-likelihood model.

    Computes the log-likelihood of observed count data under a negative binomial
    distribution, often used to model overdispersed count data.

    Args:
        forward_model (ForwardModel): A forward model instance containing a `params` object.
        data (array-like): Observed count data, assumed to be non-negative integers.
        r_param (str): Name of the dispersion parameter (number of failures until experiment stops).
        p_param (str): Name of the success probability parameter (probability of success in each trial).
    """

    def __init__(self, forward_model, data, r_param="r", p_param="p"):
        for param_name, label in zip([r_param, p_param], ["r_param", "p_param"]):
            if not isinstance(param_name, str):
                raise TypeError(f"`{label}` must be a string, got {type(param_name).__name__}")
            if param_name not in forward_model.params["name"]:
                raise ValueError(f"`{label}` '{param_name}' not found in forward_model.params['name']")

        super().__init__(forward_model, data)
        self.r_param = r_param
        self.p_param = p_param

        data = np.asarray(data)
        if np.any(data < 0) or not np.all(np.equal(np.mod(data, 1), 0)):
            raise ValueError("Negative binomial data must be non-negative integers.")

        self.data = data

    def log_likelihood(self):
        """
        Compute the Negative Binomial log-likelihood.

        This computes:
            log L = sum_k [ log(choose(k + r - 1, k)) + r*log(1 - p) + k*log(p) ]

        Returns:
            float: The sum of the log-likelihoods over all data points.
        """
        param_names = self.forward_model.params["name"]
        param_vals = self.forward_model.params["val"]

        r_idx = np.where(param_names == self.r_param)[0][0]
        p_idx = np.where(param_names == self.p_param)[0][0]

        r = param_vals[r_idx]
        p = param_vals[p_idx]

        if not (0 < p < 1):
            raise ValueError(f"Invalid success probability `p={p}`. Must be in (0, 1).")
        if r <= 0:
            raise ValueError(f"Invalid dispersion parameter `r={r}`. Must be positive.")

        log_pmf = nbinom.logpmf(self.data, n=r, p=p)
        return np.sum(log_pmf)
