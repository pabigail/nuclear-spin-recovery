import numpy as np
from abc import ABC, abstractmethod
from .spin_bath import SpinBath, NuclearSpin


class ForwardModel(ABC):
    """
    Abstract base class for forward models that compute NV (or other qubit)
    coherence signals given a nuclear spin environment and experiment parameters.
    """

    def __init__(self, spins, experiment):
        """
        Parameters
        ----------
        spins : list of NuclearSpin
            The nuclear spin bath configuration.
        experiment : Experiment
            Experimental conditions (magnetic field, timepoints, pulse sequence, etc.).
        """
        if not isinstance(spins, (list, tuple)):
            raise TypeError("spins must be a list of NuclearSpin objects.")
        if not all(isinstance(s, NuclearSpin) for s in spins):
            raise TypeError("all items in spins must be NuclearSpin instances.")
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment instance.")

        self.spins = spins
        self.experiment = experiment

    @abstractmethod
    def compute_coherence(self, idx=0):
        """
        Compute the coherence signal for experiment `idx`.

        Parameters
        ----------
        idx : int, optional
            Index of the experiment (default: 0).

        Returns
        -------
        signal : np.ndarray
            Complex or real-valued coherence signal, shape = (len(timepoints),).
        """
        pass

    def __repr__(self):
        return (f"<ForwardModel with {len(self.spins)} spins and "
                f"{len(self.experiment)} experiments>")
