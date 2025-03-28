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
        pass



