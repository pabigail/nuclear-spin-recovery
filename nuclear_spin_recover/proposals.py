import numpy as np
from abc import ABC, abstractmethod
import random
import copy
from nuclear_spin_recover import SpinBath, NuclearSpin


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



class DiscreteLatticeRWMHProposal:
    """
    Random-Walk Metropolis-Hastings (RWMH) proposal that selects
    a new discrete spin index within a local region of a SpinBath lattice.
    """

    def __init__(self, spin_bath, r, spin_inds=None):
        """
        Parameters
        ----------
        spin_bath : SpinBath
            The SpinBath containing all NuclearSpins.
        r : float
            Radius (in Å or lattice units) defining the local neighborhood.
        spin_inds : list[int], optional
            Indices of spins already part of the current model (to avoid reuse).
        """
        if not isinstance(spin_bath, SpinBath):
            raise TypeError("spin_bath must be a SpinBath instance.")
        if r <= 0:
            raise ValueError("Radius r must be positive.")
        
        self.spin_bath = spin_bath
        self.r = float(r)
        self.spin_inds = [] if spin_inds is None else list(spin_inds)

    def propose(self, spin_index=None):
        """
        Propose a new spin index within radius r of a given spin.

        Parameters
        ----------
        spin_index : int, optional
            Index of the reference spin in the SpinBath.
            If None, one is chosen uniformly at random.

        Returns
        -------
        new_spin_index : int
            The proposed new spin index within the local neighborhood.

        Raises
        ------
        ValueError
            If no valid candidate spins are found within radius r.
        """
        n_spins = len(self.spin_bath)
        if n_spins == 0:
            raise ValueError("SpinBath is empty — cannot propose a new spin.")

        # Choose reference spin index
        if spin_index is None:
            spin_index = np.random.randint(0, n_spins)
        elif not (0 <= spin_index < n_spins):
            raise IndexError(f"spin_index {spin_index} out of range (0, {n_spins-1}).")

        # Compute neighborhood
        dist_row = self.spin_bath.distance_matrix[spin_index]
        neighbors = np.where((dist_row <= self.r) & (dist_row > 0))[0]

        # Exclude spins already in the model
        candidates = [i for i in neighbors if i not in self.spin_inds]

        if not candidates:
            raise ValueError(
                f"No available candidate spins within radius {self.r} "
                f"of spin index {spin_index}."
            )

        # Choose a new spin index uniformly from candidates
        new_spin_index = np.random.choice(candidates)
        return new_spin_index

    def __repr__(self):
        return (
            f"DiscreteLatticeRWMHProposal("
            f"r={self.r}, "
            f"n_excluded={len(self.spin_inds)}, "
            f"n_total={len(self.spin_bath)})"
        )
