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
                 gyro=None, Q=None, uncertainty=None):

        # Validate required parameters
        if spin_type is None:
            raise ValueError("spin_type must be provided.")
        if None in (x, y, z):
            raise ValueError("All coordinates x, y, z must be provided.")
        if gyro is None:
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
        self.gyro = gyro

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
                f"gyro={self.gyro}, A_par={self.A_par}, A_perp={self.A_perp}, "
                f"uncertainty={self.uncertainty})")


class SpinBath:
    """
    A collection of NuclearSpins with dataframe-like convenience
    and distance matrix computation.
    """
    def __init__(self, spins=None, distance_matrix_file=None, save_distance_matrix=False):
        """
        Parameters
        ----------
        spins : list of NuclearSpin, optional
            Initial list of nuclear spins in the bath.
        distance_matrix_file : str, optional
            Path to a file containing a precomputed distance matrix.
            If provided and exists, the distance matrix will be loaded
            from this file.
        save_distance_matrix : bool, default=False
            - If False, do not save the computed distance matrix.
            - If True, a valid `distance_matrix_file` must be provided,
              otherwise an error is raised.
        """
        self.spins = [] if spins is None else list(spins)
        self._distance_matrix = None
        self._distance_matrix_file = distance_matrix_file
        self._save_distance_matrix = save_distance_matrix

        if self._save_distance_matrix and self._distance_matrix_file is None:
            raise ValueError("Must provide `distance_matrix_file` when save_distance_matrix=True.")

        if distance_matrix_file is not None and os.path.exists(distance_matrix_file):
            with open(distance_matrix_file, "rb") as f:
                self._distance_matrix = pkl.load(f)

    def add_spin(self, spin):
        assert isinstance(spin, NuclearSpin)

        # Check if there is already a spin at this position
        for existing in self.spins:
            if np.allclose(existing.xyz, spin.xyz, atol=1e-8):
                raise ValueError(
                    f"Position {spin.xyz} already occupied by {existing.spin_type}. "
                    "Use `update_spin` to change the properties of an existing spin."
                )
        self.spins.append(spin)

    def update_spin(self, xyz, **kwargs):
        """
        Update parameters of the spin at lattice position `xyz`.
        Allowed kwargs: A_par, A_perp, gyro, spin_type.
        """
        for spin in self.spins:
            if np.allclose(spin.xyz, xyz, atol=1e-8):
                if "A_par" in kwargs:
                    spin.A_par = kwargs["A_par"]
                if "A_perp" in kwargs:
                    spin.A_perp = kwargs["A_perp"]
                if "gyro" in kwargs:
                    spin.gyro = kwargs["gyro"]
                if "spin_type" in kwargs:
                    spin.spin_type = kwargs["spin_type"]
                return spin
        raise ValueError(f"No spin found at position {xyz} to update.")

    @property
    def dataframe(self):
        """Return a pandas DataFrame view for convenience."""
        data = {
            "type": [s.spin_type for s in self.spins],
            "x": [s.xyz[0] for s in self.spins],
            "y": [s.xyz[1] for s in self.spins],
            "z": [s.xyz[2] for s in self.spins],
            "gyro": [s.gyro for s in self.spins],
            "A_par": [s.A_par for s in self.spins],
            "A_perp": [s.A_perp for s in self.spins],
            "uncertainty_par": [],
            "uncertainty_perp": [],
        }

        for s in self.spins:
            if s.uncertainty is None:
                data["uncertainty_par"].append(np.nan)
                data["uncertainty_perp"].append(np.nan)
            elif isinstance(s.uncertainty, (float, int)):
                data["uncertainty_par"].append(float(s.uncertainty))
                data["uncertainty_perp"].append(float(s.uncertainty))
            elif isinstance(s.uncertainty, (list, np.ndarray)):
                arr = np.array(s.uncertainty, dtype=float)
                if arr.shape != (2,):
                    raise ValueError("Uncertainty array must have length 2.")
                data["uncertainty_par"].append(arr[0])
                data["uncertainty_perp"].append(arr[1])
            else:
                raise TypeError("Uncertainty must be None, float, or array-like of length 2.")

        return pd.DataFrame(data)

    @property
    def distance_matrix(self):
        """Compute (or load) the pairwise distances between all spins."""
        if self._distance_matrix is None:
            coords = np.array([s.xyz for s in self.spins])
            diff = coords[:, None, :] - coords[None, :, :]
            self._distance_matrix = np.linalg.norm(diff, axis=-1)

            if self._save_distance_matrix:
                with open(self._distance_matrix_file, "wb") as f:
                    pkl.dump(self._distance_matrix, f)

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
