import numpy as np
from abc import ABC, abstractmethod
from hymcmcpy.params import Params


class MCMCIterations(np.ndarray):
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


class MCMCRunner(ABC):
    """
    Abstract base class for running a hybrid MCMC algorithm given an initial parameter set,
    a sequence of algorithms, and a total number of iterations.

    Args:
        data (array-like): Observed data against which proposed parameters and forward model will be compared.
        likelihood_model (LikelihoodModel): A likelihood model used to evaluate the likelihood of the observed data.
        initial_params (Params): The initial parameters for the MCMC run.
        schedule (list of tuples): A list of (MCMCStep algorithm, int num_iterations) tuples defining the
            sequence of algorithms and their respective iteration counts.
        total_iterations (int): The total number of MCMC iterations to perform.
        existing_iterations (MCMC_iterations, optional): An existing MCMC_iterations object to append new results.

    Returns:
        MCMC_iterations: A structured array storing MCMC step details.
    """
    def __init__(self, data, likelihood_model, schedule, total_iterations, initial_params=None, existing_iterations=None,  **kwargs):
        if initial_params is None and existing_iterations is None:
            raise ValueError("Either intial_params or existing_iterations must be provided")
        if initial_params is not None and existing_iterations is not None:
            raise ValueError("Only one of intial_params or existing_iterations can be provided")
        
        self.data = data
        self.likelihood_model = likelihood_model
        self.initial_params = initial_params
        self.schedule = schedule
        self.total_iterations = total_iterations
        self.existing_iterations = existing_iterations
        self.kwargs = kwargs
    
    @abstractmethod
    def instantiation_step(self, algorithm, **kwargs):
        """
        Instantiate an MCMCStep subclass based on the given algorithm nickname and additional parameters.

        Args:
            algorithm_nickname (str): The nickname of the MCMC algorithm to instantiate.
            **kwargs: Additional parameters required for the specific MCMC algorithm.

        Returns:
            MCMCStep: An instance of the corresponding MCMC algorithm.
        """
        pass


    def run(self):
        """
        Execute the MCMC sequence as per the defined schedule.

        Returns:
            MCMC_iterations: A structured array storing the params, whether accepted, 
            and the algorithm at each step of the MCMC iterations.
        """
        current_params = self.initial_params if self.initial_params is not None else self.existing_iterations['params'][-1]
        algorithms = []
        accepts = []
        params_list = []

        if self.existing_iterations:
            algorithms.extend(self.existing_iterations['algorithm'])
            accepts.extend(self.existing_iterations['accept'])
            params_list.extend(self.existing_iterations['params'])

        for algorithm, num_iter in self.schedule:
            step = self.instantiation_step(algorithm, **self.kwargs)
            
            for _ in range(num_iter):
                new_params, accepted = step.one_step(current_params)
                algorithms.append(algorithm.nickname)
                accepts.append(accepted)
                params_list.append(new_params)
                
                if accepted:
                    current_params = new_params

        return MCMCIterations(algorithms, accepts, params_list)


