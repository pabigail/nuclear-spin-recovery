import numpy as np
from abc import ABC, abstractmethod
from hymcmcpy.params import Params

class MCMCStep(ABC):
    """
    Abstract base class for performing a single step of an MCMC (Markov Chain Monte Carlo) algorithm.

    This class defines the structure for MCMC steps, where each subclass must implement the `one_step()`
    method to perform a single iteration of the algorithm, including proposing new parameters and deciding
    whether to accept or reject them.

    Args:
        data (array-like): Observed data against which proposed parameters and forward model will be compared
        params (Params): The current parameters (of type `Params`) for the system.
        forward_model (ForwardModel): A forward model used to generate predictions or simulations.
        likelihood_model (LikelihoodModel): A likelihood model used to evaluate the likelihood of the observed data.
        param_names (str or list of str): The name(s) of the parameter(s) to act upon during this step.
        **kwargs: Additional fixed arguments for the step (optional).

    Attributes:
        params (Params): The current system parameters.
        forward_model (ForwardModel): The forward model used in the step.
        nickname (str): String to refer to algorithm
        likelihood_model (LikelihoodModel): The likelihood model used to evaluate the likelihood of the data.
    """

    def __init__(self, data, likelihood_model, param_names, **kwargs):
        self.data = data
        self.likelihood_model = likelihood_model
        self.param_names = param_names

    @property
    @abstractmethod
    def nickname(self):
        """Get the unique identifier nickname for the algorithm.

        Each subclass must define a unique `nickname`, which is a string 
        used to identify the algorithm.

        Returns:
            str: The unique nickname of the algorithm.
        """
        pass


    def __setattr(self, key, value):
        """Prevent modification of 'nickname' after instantiation."""
        if key == "nickname":
            raise AttributeError("Cannot modify 'nickname' after instantiation")
        super().__setattr__(key, value)

    @abstractmethod
    def one_step(self, current_params):
        """
        Perform one step of the MCMC algorithm, which involves proposing new parameters and
        deciding whether to accept or reject them based on the model and the likelihood.

        This method should return the proposed new parameters and a boolean indicating whether
        the new parameters were accepted.
        
        Args:
            params (Params): The current parameters (of type `Params`
        
        Returns:
            tuple: A tuple containing the proposed new parameters and a boolean indicating
                   whether the proposal was accepted (True) or rejected (False).
        """
        pass



class RWMHContinuousStep(MCMCStep):
    """
    A subclass of MCMCStep implementing one step of the Random Walk Metropolis-Hastings (RWMH) algorithm.

    In this step, the new parameter is proposed by adding Gaussian noise to the current parameter,
    and the acceptance decision is based on the difference in log-likelihoods.

    Args:
        params (Params): Current parameters of the system, should be a single continuous parameter.
        forward_model (ForwardModel): The forward model used for generating predictions.
        likelihood_model (LikelihoodModel): The likelihood model used to evaluate the log-likelihood.
        sigma_sq (float): The variance of the Gaussian noise used to propose new parameters.
        **kwargs: Additional fixed arguments (optional).

    Attributes:
        sigma_sq (float): The variance of the Gaussian noise used for proposing new parameters.
    """
    
    def __init__(self, data, likelihood_model, param_names, sigma_sq):
        super().__init__(data, likelihood_model, param_names)
        self.sigma_sq = sigma_sq

    # set nickname
    nickname = "rwmh_continuous"



    def one_step(self, current_params):
        """
        Perform one step of the Random Walk Metropolis-Hastings (RWMH) algorithm using log-likelihood.

        In this method, a new parameter is proposed by adding Gaussian noise to the current parameter.
        The new parameter is accepted or rejected based on the difference in log-likelihoods.

        Args:
            current_params (Params): The current parameters (of type `Params`

        Returns:
            tuple: A tuple containing the proposed new parameter as a new `Params` object and a boolean indicating 
                   whether the proposal was accepted (True) or rejected (False).
        """
        # Identify the parameter name from subset_param_names
        subset_param_names = self.likelihood_model.forward_model.subset_param_names
        if len(subset_param_names) != 1:
            raise ValueError("RWMH_continuous requires exactly one subset parameter name.")

        param_name = subset_param_names[0]

        # Extract the current parameter value based on the subset_param_name
        current_param = current_params[param_name][0]

        # Propose new parameter using Gaussian noise
        proposed_param = current_param + np.random.normal(0, np.sqrt(self.sigma_sq))

        # Create a new Params object with the proposed parameter value
        proposed_params = Params(current_params['name'], [proposed_param], current_params['discrete'])

        # Compute the log-likelihood for the current and proposed parameters
        current_log_likelihood = self.likelihood_model.log_likelihood()

        # Update the forward model with proposed parameters and compute the log-likelihood for the new parameters
        self.likelihood_model.forward_model.params = proposed_params
        proposed_log_likelihood = self.likelihood_model.log_likelihood()

        # Compute the acceptance ratio using log-likelihood difference
        log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood

        # Accept the new parameter with probability min(1, exp(log_acceptance_ratio))
        accepted = np.random.rand() < min(1, np.exp(log_acceptance_ratio))

        if accepted:
            # Return the new proposed parameters if accepted
            self.likelihood_model.forward_model.params = proposed_params
            return proposed_params, accepted
        else:
            # Return the original parameters if rejected
            self.likelihood_model.forward_model.params = current_params
            return current_params, accepted
