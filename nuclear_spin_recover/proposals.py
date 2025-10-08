import numpy as np
from abc import ABC, abstractmethod
import random
import copy


class Proposal(ABC):
    """
    Abstract base class for proposal mechanisms in trans-dimensional MCMC.

    Each subclass must implement `propose`, which returns:
        new_spin_inds, new_params
    given the current spin indices and parameter dictionary.
    """

    def __init__(self, spin_inds, params):
        """
        Parameters
        ----------
        spin_inds : list[int]
            Indices of nuclear spins (in SpinBath) currently included in the model.
        params : dict
            Dictionary of current model parameters (e.g., lambda_decoherence, A_par, A_perp, etc.).
        """
        self.spin_inds = np.array(spin_inds, dtype=int)
        self.params = params

    @abstractmethod
    def propose(self):
        """
        Generate a proposed set of spins and parameters.
        Must be implemented by subclasses.
        Returns
        -------
        new_spin_inds : np.ndarray
        new_params : dict
        """
        pass



