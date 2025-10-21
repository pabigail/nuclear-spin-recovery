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



class DiscreteLatticeRWMHProposal(Proposal):
    """
    Random-Walk Metropolis-Hastings (RWMH) proposal that selects
    a new discrete spin index within a local region of a SpinBath lattice.
    """

    def __init__(self, spin_bath, r, spin_inds, params):
        """
        Parameters
        ----------
        spin_bath : SpinBath
            The SpinBath containing all NuclearSpins in the Lattice
        r : float
            Radius (in Å or lattice units) defining the local neighborhood.
        """
        super().__init__(spin_inds, params)

        if not isinstance(spin_bath, SpinBath):
            raise TypeError("spin_bath must be a SpinBath instance.")
        if r <= 0:
            raise ValueError("Radius r must be positive.")
        
        self.spin_bath = spin_bath
        self.r = float(r)

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
        n_spins = len(self.spin_inds)
        if n_spins == 0:
            raise ValueError("NuclearSpins is empty — cannot propose a new spin.")

        # Choose reference spin index
        if spin_index is None:
            spin_index = np.random.choice(self.spin_inds)
        elif spin_index not in self.spin_inds:
            raise IndexError(f"spin_index {spin_index} is not in current spin_inds {self.spin_inds.tolist()}")

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


class ContinuousBounded2dRWMHProposal(Proposal):
    """
    Random-Walk Metropolis-Hastings (RWMH) proposal for continuous (A_perp, A_par)
    hyperfine parameters of a single nuclear spin, with a fixed reference center.

    This proposal perturbs the (A_perp, A_par) pair of one spin using Gaussian
    random walks centered on either the current MCMC value or, if not provided,
    the SpinBath reference. A hard boundary (max_radius) is always enforced
    relative to the SpinBath reference values.

    Attributes
    ----------
    spin_bath : SpinBath
        SpinBath containing all NuclearSpin objects.
    width_fac : float
        Fractional width for Gaussian proposal variance.
    max_radius : float
        Maximum allowed radial distance from reference (SpinBath) values.
    eps : float
        Minimum proposal width to avoid degenerate moves.
    """

    def __init__(self, spin_bath, spin_inds, params, width_fac,
                 max_radius=np.inf, eps=1e-12):
        """
        Parameters
        ----------
        spin_bath : SpinBath
            SpinBath containing all NuclearSpin objects.
        spin_inds : list[int]
            Indices of currently included spins in the model.
        params : dict
            Dictionary of current model parameters (must include "A_par" and "A_perp").
        width_fac : float
            Fractional width for Gaussian variance.
        max_radius : float, optional
            Maximum allowed distance from SpinBath (A_perp, A_par) reference.
        eps : float, optional
            Minimum width to avoid degenerate proposals.
        """
        super().__init__(spin_inds, params)

        if not hasattr(spin_bath, "__getitem__"):
            raise TypeError("spin_bath must support indexing (e.g., SpinBath).")

        self.spin_bath = spin_bath
        self.width_fac = float(width_fac)
        self.max_radius = float(max_radius)
        self.eps = float(eps)

    def propose(self, spin_index=None, A_perp_current=None, A_par_current=None):
        """
        Propose new (A_perp, A_par) for one spin using a bounded Gaussian random walk.

        Parameters
        ----------
        spin_index : int, optional
            Index within current spin_inds to perturb.
            If None, one spin is chosen uniformly at random.
        A_perp_current, A_par_current : float, optional
            Current MCMC values for (A_perp, A_par). If provided, the random walk
            starts from these values; otherwise, starts from SpinBath values.

        Returns
        -------
        new_spin_inds : np.ndarray
            Copy of current spin indices (unchanged).
        new_params : dict
            Copy of `params` with updated "A_perp" and "A_par" for the chosen spin.
        A_perp_prop, A_par_prop : float
            Proposed (A_perp, A_par) values for the perturbed spin.
        """
        n_spins = len(self.spin_inds)
        if n_spins == 0:
            raise ValueError("No spins in current model — cannot propose a continuous move.")

        # Choose which spin to perturb
        if spin_index is None:
            spin_ind = np.random.choice(n_spins)
        elif not (0 <= spin_index < n_spins):
            raise IndexError(f"spin_index {spin_index} is out of bounds (0–{n_spins-1}).")
        else:
            spin_ind = spin_index

        spin_idx = self.spin_inds[spin_ind]
        spin = self.spin_bath[spin_idx]

        # Reference (fixed) SpinBath values
        ref_A_perp = spin.A_perp
        ref_A_par = spin.A_par

        # Center of the random walk
        mu_perp = A_perp_current if A_perp_current is not None else ref_A_perp
        mu_par = A_par_current if A_par_current is not None else ref_A_par

        # Proposal widths
        sigma_perp = max(self.width_fac * abs(mu_perp), self.eps)
        sigma_par = max(self.width_fac * abs(mu_par), self.eps)

        # Draw proposed values
        A_perp_prop = np.random.normal(loc=mu_perp, scale=sigma_perp)
        A_par_prop = np.random.normal(loc=mu_par, scale=sigma_par)

        # Enforce hard boundary relative to SpinBath reference
        dA_perp = A_perp_prop - ref_A_perp
        dA_par = A_par_prop - ref_A_par
        radial_dist = np.sqrt(dA_perp**2 + dA_par**2)

        if radial_dist > self.max_radius:
            # Reflect back into allowed radius
            scale = self.max_radius / radial_dist
            A_perp_prop = ref_A_perp + dA_perp * scale
            A_par_prop = ref_A_par + dA_par * scale

        # Create new parameter set
        new_params = {k: np.copy(v) if isinstance(v, np.ndarray) else v
                      for k, v in self.params.items()}
        new_params["A_perp"][spin_ind] = A_perp_prop
        new_params["A_par"][spin_ind] = A_par_prop

        return self.spin_inds.copy(), new_params, A_perp_prop, A_par_prop

    def __repr__(self):
        return (
            f"ContinuousBounded2dRWMHProposal("
            f"width_fac={self.width_fac}, "
            f"max_radius={self.max_radius}, "
            f"eps={self.eps}, "
            f"n_spins={len(self.spin_inds)})"
        )
