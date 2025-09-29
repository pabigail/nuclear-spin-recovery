import numpy as np
import pandas as pd
import pycce as pc
import pickle as pkl
import os

# pycce-compatible dtype
_dtype_bath = np.dtype([
    ('N', np.unicode_, 16),       # spin type/name
    ('xyz', np.float64, (3,)),    # position vector
    ('A', np.float64, (3, 3)),    # hyperfine tensor
    ('Q', np.float64, (3, 3))     # quadrupole tensor
])

class NuclearSpin:
    """
    Represents a single nuclear spin with coordinates, hyperfine couplings,
    and optional quadrupole tensor.
    """

    def __init__(self, spin_type, x=None, y=None, z=None,
                 A_par=None, A_perp=None,
                 A_xx=None, A_yy=None, A_zz=None,
                 A_xy=None, A_yz=None, A_xz=None,
                 w_L=None, Q=None, uncertainty=None):

        # Validate required parameters
        if spin_type is None:
            raise ValueError("spin_type must be provided.")
        if None in (x, y, z):
            raise ValueError("All coordinates x, y, z must be provided.")
        if w_L is None:
            raise ValueError("Larmor frequency must be provided.")

        # Determine hyperfine specification
        has_cartesian = all(v is not None for v in (A_xx, A_yy, A_zz, A_xy, A_yz, A_xz))
        has_axial = A_par is not None and A_perp is not None

        if not (has_cartesian or has_axial):
            raise ValueError(
                "Hyperfine must be fully specified either via "
                "A_xx, A_yy, A_zz, A_xy, A_yz, A_xz OR A_par and A_perp."
            )

        self.spin_type = spin_type
        self.xyz = np.array([x, y, z], dtype=float)
        self.w_L = w_L

        if has_cartesian:
            self.A = np.array([
                [A_xx, A_xy, A_xz],
                [A_xy, A_yy, A_yz],
                [A_xz, A_yz, A_zz]
            ])
            # Compute axial parameters from Cartesian tensor
            self.A_perp = np.sqrt(A_xz**2 + A_yz**2)
            self.A_par = A_zz
        else:  # has_axial
            # Construct Cartesian tensor from axial parameters
            self.A = np.array([
                [0.0, 0.0, A_perp],
                [0.0, 0.0, 0.0],
                [A_perp, 0.0, A_par]
            ])
            self.A_par = A_par
            self.A_perp = A_perp

        # Quadrupole tensor
        self.Q = Q if Q is not None else np.zeros((3, 3))

        # Validate and set uncertainty
        if uncertainty is None:
            self.uncertainty = None
        elif isinstance(uncertainty, (float, int)):
            self.uncertainty = float(uncertainty)
        elif isinstance(uncertainty, (list, np.ndarray)):
            arr = np.array(uncertainty, dtype=float)
            if arr.shape != (2,):
                raise ValueError("Uncertainty array must have length 2 (for A_par and A_perp).")
            self.uncertainty = arr
        else:
            raise TypeError("Uncertainty must be None, float, or array-like of length 2.")

    def to_record(self):
        """Return structured-array-compatible record (for pycce)."""
        return (self.spin_type, self.xyz, self.A, self.Q)

    def __repr__(self):
        return (f"NuclearSpin(type={self.spin_type}, xyz={self.xyz}, "
                f"w_L={self.w_L}, A_par={self.A_par}, A_perp={self.A_perp}, "
                f"uncertainty={self.uncertainty})")


class SpinBath:
    """
    A collection of NuclearSpins with dataframe-like convenience
    and distance matrix computation.
    """
    def __init__(self, spins=None, distance_matrix_file=None):
        self.spins = [] if spins is None else list(spins)
        self._distance_matrix = None

        if distance_matrix_file is not None and os.path.exists(distance_matrix_file):
            with open(distance_matrix_file, "rb") as f:
                self._distance_matrix = pickle.load(f)

    def add_spin(self, spin):
        assert isinstance(spin, NuclearSpin)
        self.spins.append(spin)

    @property
    def dataframe(self):
        """Return a pandas DataFrame view for convenience."""
        data = {
            "type": [s.spin_type for s in self.spins],
            "x": [s.xyz[0] for s in self.spins],
            "y": [s.xyz[1] for s in self.spins],
            "z": [s.xyz[2] for s in self.spins],
            "w_L": [s.w_L for s in self.spins],
            "A_par": [s.A_par for s in self.spins],
            "A_perp": [s.A_perp for s in self.spins]
        }
        return pd.DataFrame(data)

    @property
    def distance_matrix(self):
        """Compute pairwise distances between all spins."""
        if self._distance_matrix is None:
            coords = np.array([s.xyz for s in self.spins])
            diff = coords[:, None, :] - coords[None, :, :]
            self._distance_matrix = np.linalg.norm(diff, axis=-1)
        return self._distance_matrix

    def to_numpy(self):
        """Convert bath to numpy structured array compatible with pycce.BathArray."""
        arr = np.array([s.to_record() for s in self.spins], dtype=_dtype_bath)
        return arr

    def __getitem__(self, idx):
        return self.spins[idx]

    def __len__(self):
        return len(self.spins)

    def __repr__(self):
        return f"SpinBath(nspins={len(self)})"
