import numpy as np
from abc import ABC, abstractmethod
import random
import copy
from copy import deepcopy
from nuclear_spin_recover import SpinBath, NuclearSpin


class Proposal(ABC):
    """
    Abstract base class for proposal mechanisms in trans-dimensional MCMC.

    Subclasses generate proposed spins/parameters, which are stored internally
    as `prop_spins` and `prop_params`. The current state is stored in
    `current_spins` and `current_params`.

    Proposal objects do not compute likelihoods or acceptance probabilities;
    they only generate new candidate states.
    """

    def __init__(self, spin_inds, params):
        """
        Parameters
        ----------
        spin_inds : list[int]
            Indices of nuclear spins currently included in the model.
        params : dict
            Dictionary of current model parameters (e.g., lambda_decoherence, A_par, A_perp, etc.).
        """
        self.current_spins = np.array(spin_inds, dtype=int)
        self.current_params = params

        self.prop_spins = None
        self.prop_params = None

    @abstractmethod
    def propose(self):
        """
        Generate a proposed set of spins and parameters.

        This should set `self.prop_spins` and `self.prop_params`.

        Returns
        -------
        prop_spins : np.ndarray
        prop_params : dict
        """
        pass

    def accept_prop(self):
        """Accept the proposed move: update current state and clear proposal."""
        if self.prop_spins is None or self.prop_params is None:
            raise ValueError("No proposal to accept.")
        self.current_spins = self.prop_spins.copy()
        self.current_params = {k: np.copy(v) if isinstance(v, np.ndarray) else v
                               for k, v in self.prop_params.items()}
        self.prop_spins = None
        self.prop_params = None

    def reject_prop(self):
        """Reject the proposed move: discard the proposal."""
        self.prop_spins = None
        self.prop_params = None


class DiscreteLatticeRWMHProposal(Proposal):
    """
    Random-Walk Metropolis-Hastings (RWMH) proposal for discrete spin indices
    within a local neighborhood of a SpinBath lattice.
    """

    def __init__(self, spin_bath, r, spin_inds, params):
        """
        Parameters
        ----------
        spin_bath : SpinBath
            The SpinBath containing all NuclearSpins in the lattice.
        r : float
            Radius defining the local neighborhood.
        spin_inds : list[int]
            Current spins included in the model.
        params : dict
            Current model parameters.
        """
        super().__init__(spin_inds, params)

        if not isinstance(spin_bath, SpinBath):
            raise TypeError("spin_bath must be a SpinBath instance.")
        if r <= 0:
            raise ValueError("Radius r must be positive.")

        self.spin_bath = spin_bath
        self.r = float(r)

        # Proposal state
        self.prop_spins = None
        self.prop_params = None

    def propose(self, spin_index=None):
        """
        Propose a new spin index within radius r of a given spin.

        Returns
        -------
        prop_spins : np.ndarray
            Copy of current spins with one proposed change.
        prop_params : dict
            Copy of current_params (unchanged for discrete spins).
        new_spin_index : int
            Proposed spin index within the local neighborhood.
        """
        n_spins = len(self.current_spins)
        if n_spins == 0:
            raise ValueError("No spins in current model — cannot propose a new spin.")

        # Choose reference spin
        if spin_index is None:
            spin_index = np.random.choice(self.current_spins)
        elif spin_index not in self.current_spins:
            raise IndexError(
                f"spin_index {spin_index} is not in current spins {self.current_spins.tolist()}"
            )

        # Neighborhood
        dist_row = self.spin_bath.distance_matrix[spin_index]
        neighbors = np.where((dist_row <= self.r) & (dist_row > 0))[0]

        # Exclude spins already in the model
        candidates = [i for i in neighbors if i not in self.current_spins]
        if not candidates:
            raise ValueError(
                f"No available candidate spins within radius {self.r} of spin index {spin_index}."
            )

        # Choose new spin index
        new_spin_index = np.random.choice(candidates)

        # Build proposed spin array and param dict
        self.prop_spins = self.current_spins.copy()
        # Replace spin_index with new_spin_index
        idx_in_current = np.where(self.current_spins == spin_index)[0][0]
        self.prop_spins[idx_in_current] = new_spin_index

        # For discrete spins, parameters do not change
        self.prop_params = {k: np.copy(v) if isinstance(v, np.ndarray) else v
                            for k, v in self.current_params.items()}

        return self.prop_spins.copy(), self.prop_params.copy(), new_spin_index


    def __repr__(self):
        return (
            f"DiscreteLatticeRWMHProposal("
            f"r={self.r}, "
            f"n_excluded={len(self.current_spins)}, "
            f"n_total={len(self.spin_bath)})"
        )


class ContinuousBounded2dRWMHProposal(Proposal):
    """
    Random-Walk Metropolis-Hastings (RWMH) proposal for continuous (A_perp, A_par)
    hyperfine parameters of nuclear spins, with a fixed reference center.
    """

    def __init__(self, spin_bath, spin_inds, params, width_fac, max_radius=np.inf, eps=1e-12):
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
            Fractional width for Gaussian proposal variance.
        max_radius : float, optional
            Maximum allowed distance from SpinBath reference values.
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

        # Proposal state
        self.prop_spins = None
        self.prop_params = None

    def propose(self, spin_index=None, A_perp_current=None, A_par_current=None):
        """
        Propose new (A_perp, A_par) for one spin using a bounded Gaussian random walk.

        Returns
        -------
        prop_spins : np.ndarray
            Copy of current spins (unchanged for continuous proposal).
        prop_params : dict
            Copy of current_params with updated A_perp_rw / A_par_rw.
        A_perp_prop, A_par_prop : float
            Proposed (A_perp, A_par) values for the perturbed spin.
        """
        n_spins = len(self.current_spins)
        if n_spins == 0:
            raise ValueError("No spins in current model — cannot propose a continuous move.")

        # Choose which spin to perturb
        if spin_index is None:
            spin_ind = np.random.choice(n_spins)
        elif not (0 <= spin_index < n_spins):
            raise IndexError(f"spin_index {spin_index} is out of bounds (0–{n_spins-1}).")
        else:
            spin_ind = spin_index

        spin_idx = self.current_spins[spin_ind]
        spin = self.spin_bath[spin_idx]

        # Reference SpinBath values (never change)
        ref_A_perp = spin.A_perp
        ref_A_par = spin.A_par

        # Current random-walk values (or default to reference)
        mu_perp = A_perp_current if A_perp_current is not None else self.current_params["A_perp"][spin_ind]
        mu_par = A_par_current if A_par_current is not None else self.current_params["A_par"][spin_ind]

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
            scale = self.max_radius / radial_dist
            A_perp_prop = ref_A_perp + dA_perp * scale
            A_par_prop = ref_A_par + dA_par * scale

        # Build proposed state
        self.prop_spins = self.current_spins.copy()
        self.prop_params = {k: np.copy(v) if isinstance(v, np.ndarray) else v
                            for k, v in self.current_params.items()}
        self.prop_params["A_perp"][spin_ind] = A_perp_prop
        self.prop_params["A_par"][spin_ind] = A_par_prop

        return self.prop_spins.copy(), self.prop_params.copy(), A_perp_prop, A_par_prop

    def __repr__(self):
            return (
                f"ContinuousBounded2dRWMHProposal("
                f"width_fac={self.width_fac}, "
                f"max_radius={self.max_radius}, "
                f"eps={self.eps}, "
                f"n_spins={len(self.current_spins)})"
            )

class ContinuousBounded1dRWMHProposal(Proposal):
    """
    Proposes a single continuous parameter 'lambda_decoherence' in (0,1]
    using a Gaussian random walk with reflection at boundaries.
    """
    def __init__(self, params, radius=0.1):
        # No spins affected, just a free parameter
        super().__init__(spin_inds=[], params=params)
        self.radius = radius

        # Initialize lambda_decoherence if None
        if self.current_params.get("lambda_decoherence") is None:
            self.current_params["lambda_decoherence"] = 1.0

    def propose(self):
        """
        Propose a new lambda_decoherence by Gaussian random walk with reflection.
        Returns:
            prop_spins : np.ndarray
                Empty array (no spins are affected)
            prop_params : dict
                Copy of current_params with updated lambda_decoherence
        """
        current_val = self.current_params["lambda_decoherence"]
        prop_val = current_val + np.random.normal(0, self.radius)

        # Reflect until within (0,1]
        while not (0 < prop_val <= 1):
            if prop_val > 1:
                prop_val = 2 - prop_val
            elif prop_val <= 0:
                prop_val = -prop_val + 1e-8  # tiny offset to stay >0

        # Build proposed params dict
        self.prop_params = deepcopy(self.current_params)
        self.prop_params["lambda_decoherence"] = prop_val

        # No spins affected
        self.prop_spins = np.array([])

        return self.prop_spins.copy(), self.prop_params.copy()


def remove_elements(original_list, elements_to_remove):
    return [element for element in original_list if element not in elements_to_remove]


class DiscreteRJMCMCProposal(Proposal):
    """
    Reversible-jump MCMC proposal for adding/removing nuclear spins.
    """

    def __init__(self, spin_bath, max_spins, spin_inds=None, birth=None, params=None):
        """
        Parameters
        ----------
        spin_bath : SpinBath
            Spin bath containing all NuclearSpins.
        max_spins : int
            Maximum allowed number of spins in the model.
        spin_inds : list[int], optional
            Currently included spin indices.
        birth : bool or None
            If True, force birth; if False, force death; if None, 50/50 chance.
        params : dict
            Current model parameters.
        """
        if spin_inds is None:
            spin_inds = []

        super().__init__(spin_inds=spin_inds, params=params or {})

        if not isinstance(spin_bath, SpinBath):
            raise TypeError("spin_bath must be a SpinBath instance.")
        self.spin_bath = spin_bath
        self.max_spins = max_spins
        self.birth = birth  # can be True, False, or None

    def _get_birth_proposal(self):
        all_spins = list(range(len(self.spin_bath.distance_matrix)))
        available_spins = remove_elements(all_spins, self.current_spins)
        if not available_spins:
            return list(self.current_spins)
        new_spin = random.choice(available_spins)
        return list(self.current_spins) + [new_spin]

    def _get_death_proposal(self):
        if len(self.current_spins) == 0:
            return np.array([], dtype=int)
        spins_copy = list(self.current_spins)
        idx_to_remove = random.randint(0, len(spins_copy) - 1)
        spins_copy.pop(idx_to_remove)
        return spins_copy

    def propose(self):
        """
        Generate a proposed spin set using RJMCMC birth/death move.

        Returns
        -------
        prop_spins : np.ndarray
            Proposed spin indices.
        prop_params : dict
            Copy of current parameters (unchanged here).
        """
        # Decide birth or death
        if self.birth is None:
            birth_move = random.choice([True, False])
        else:
            birth_move = self.birth

        n_spins = len(self.current_spins)

        # If at boundaries, no valid move — return unchanged
        if (birth_move and n_spins >= self.max_spins) or (not birth_move and n_spins == 0):
            self.prop_spins = np.array(self.current_spins, dtype=int)
            self.prop_params = deepcopy(self.current_params)
            return self.prop_spins, self.prop_params

        # Generate proposal
        if birth_move:
            self.prop_spins = np.array(self._get_birth_proposal(), dtype=int)
        else:
            self.prop_spins = np.array(self._get_death_proposal(), dtype=int)

        self.prop_params = deepcopy(self.current_params)
        return self.prop_spins, self.prop_params
