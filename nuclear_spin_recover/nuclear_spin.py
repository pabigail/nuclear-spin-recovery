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
    gamma : float
        Gyromagnetic ratio of the nuclear spin.
    A_parallel : float
        Parallel hyperfine coupling component.
    A_perp : float
        Perpendicular hyperfine coupling component.
    """

    position: np.ndarray
    gamma: numbers.Real
    A_parallel: numbers.Real
    A_perp: numbers.Real


@dataclass
class NuclearSpinList:
    """
    A list of SingleNuclearSpin objects.
    """

    spins: List[SingleNuclearSpin] = field(default_factory=list)


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
