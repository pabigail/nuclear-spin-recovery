import numpy as np
import numbers
from dataclasses import dataclass, field
from typing import List, Iterable
import hashlib
import json

@dataclass
class SingleNuclearSpin:
    """
    Representation of a single nuclear spin.

    Parameters
    ----------
    position : array_like, shape (3,)
        Cartesian position of the nuclear spin (x, y, z).
    gyro : float
        Gyromagnetic ratio of the nuclear spin.
    A_parallel : float
        Parallel hyperfine coupling component.
    A_perp : float
        Perpendicular hyperfine coupling component.
    """
    position: np.ndarray
    gyro: numbers.Real
    A_parallel: numbers.Real
    A_perp: numbers.Real
    
    def __post_init__(self):
        # Position
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3-element vector")

        # Scalars
        for name, val in {
            "gyro": self.gyro,
            "A_parallel": self.A_parallel,
            "A_perp": self.A_perp,
        }.items():
            if not isinstance(val, numbers.Real) or isinstance(val, bool):
                raise TypeError(f"{name} must be a real number")

    def get_signature(self, tol: float = 1e-6) -> tuple:
        """
        Deterministic signature for the nuclear spin.

        Parameters
        ----------
        tol : float
            Tolerance used to discretize floating-point values.

        Returns
        -------
        tuple
            Hashable signature tuple.
        """
        return (
            tuple(np.round(self.position / tol).astype(int)),
            round(self.gyro / tol),
            round(self.A_parallel / tol),
            round(self.A_perp / tol),
        )

@dataclass
class NuclearSpinList:
    """
    A list of SingleNuclearSpin objects.
    """
    spins: List[SingleNuclearSpin] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.spins, list):
            raise TypeError("Spins must be provided as a list")
        for spin in self.spins:
            if not isinstance(spin, SingleNuclearSpin):
                raise TypeError("All elements must be SingleNuclearSpin objects")

    def append(self, spin: SingleNuclearSpin) -> "NuclearSpinList":
        if not isinstance(spin, SingleNuclearSpin):
            raise TypeError("Can only append SingleNuclearSpin objects")
        self.spins.append(spin)
        return self

    def delete(self, indices: Iterable[int]) -> None:
        if not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be integers")
        for i in sorted(indices, reverse=True):
            if i < 0 or i >= len(self.spins):
                raise IndexError("At least one spin index is out of range")
            del self.spins[i]

    def __getitem__(self, idx):
        return self.spins[idx]

    def __len__(self):
        return len(self.spins)

    def __iter__(self):
        return iter(self.spins)

    def get_signature(self, tol: float = 1e-6) -> str:
        """
        Deterministic SHA256 signature for the spin list.
        """
        sig_list = [spin.get_signature(tol) for spin in self.spins]
        sig_json = json.dumps(sig_list, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(sig_json.encode("utf-8")).hexdigest()

@dataclass
class FullSpinBath:
    """
    Full nuclear spin bath.

    Parameters
    ----------
    spins : NuclearSpinList
        List of nuclear spins in the bath.
    distance_matrix : ndarray, optional
        Precomputed pairwise distance matrix.
    metadata : dict, optional
        Additional bath-level information (material, defect type, etc.).
    """

    spins: NuclearSpinList
    distance_matrix: np.ndarray | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if not isinstance(self.spins, NuclearSpinList):
            raise TypeError("spins must be a NuclearSpinList")

        if self.distance_matrix is not None:
            self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
            n = len(self.spins)
            if self.distance_matrix.shape != (n, n):
                raise ValueError("Distance matrix must have shape (N, N)")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute and store the pairwise distance matrix.
        """
        positions = np.array([spin.position for spin in self.spins])
        diff = positions[:, None, :] - positions[None, :, :]
        self.distance_matrix = np.linalg.norm(diff, axis=-1)
        return self.distance_matrix

    def get_signature(self, tol: float = 1e-6) -> str:
        """
        Deterministic SHA256 signature for the full spin bath.
        """
        sig_dict = {
            "spins": self.spins.get_signature(tol),
            "distance_matrix": (
                None
                if self.distance_matrix is None
                else np.round(self.distance_matrix / tol).astype(int).tolist()
            ),
            "metadata": self.metadata,
        }

        sig_json = json.dumps(sig_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(sig_json.encode("utf-8")).hexdigest()
