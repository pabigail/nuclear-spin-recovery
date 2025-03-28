import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import poisson

class LikelihoodModel(ABC):
    """
    Abstract base class for likelihood models.

    This class connects a forward model with observed data to compute 
    a likelihood function for statistical inference (e.g., in MCMC).

    Args:
        forward_model (ForwardModel): An instance of a forward model used 
            to generate data-like output given parameters.
        data (array-like): Observed data against which the model's output 
            will be compared.
        **kwargs: Additional keyword arguments for customization in subclasses.
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

    def __init__(self, forward_model, data, **kwargs):
        super().__init__(forward_model, data, **kwargs)

    def log_likelihood(self):
        """
        Compute the Poisson log-likelihood.

        This computes:
            log L = sum_k [ k * log(lambda) - lambda - log(k!) ]

        Returns:
            float: The sum of the log-likelihoods over all data points.
        """
        lam = self.forward_model.params["val"][0]
        k_data = np.asarray(self.data)

        if np.any(k_data < 0) or not np.all(np.equal(np.mod(k_data, 1), 0)):
            raise ValueError("Poisson data must be non-negative integers.")

        # Vectorized computation of log PMF
        log_pmf = poisson.logpmf(k_data, mu=lam)

        return np.sum(log_pmf)
