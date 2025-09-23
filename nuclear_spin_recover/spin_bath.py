import numpy as np
import pandas as pd
import pycce as pc

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
    def __init__(self, spin_type, x, y, z,
                 A_par=None, A_perp=None,
                 A_xx=None, A_yy=None, A_zz=None,
                 A_xy=None, A_yz=None, A_xz=None,
                 w_L=None, Q=None):

        self.spin_type = spin_type
        self.xyz = np.array([x, y, z], dtype=float)
        self.w_L = w_L

        # Construct hyperfine tensor
        if (A_xx is not None) or (A_yy is not None) or (A_zz is not None):
            self.A = np.array([
                [A_xx or 0.0, A_xy or 0.0, A_xz or 0.0],
                [A_xy or 0.0, A_yy or 0.0, A_yz or 0.0],
                [A_xz or 0.0, A_yz or 0.0, A_zz or 0.0]
            ])
        elif A_par is not None and A_perp is not None:
            self.A = np.diag([A_perp, A_perp, A_par])
        else:
            self.A = np.zeros((3, 3))

        # Quadrupole tensor
        self.Q = Q if Q is not None else np.zeros((3, 3))

    def to_record(self):
        """Return structured-array-compatible record (for pycce)."""
        return (self.spin_type, self.xyz, self.A, self.Q)

    def __repr__(self):
        return f"NuclearSpin(type={self.spin_type}, xyz={self.xyz}, w_L={self.w_L})"



class SpinBath:
    """
    A collection of NuclearSpins with dataframe-like convenience
    and distance matrix computation.
    """
    def __init__(self, spins=None):
        self.spins = [] if spins is None else list(spins)

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
        }
        return pd.DataFrame(data)

    @property
    def distance_matrix(self):
        """Compute pairwise distances between all spins."""
        coords = np.array([s.xyz for s in self.spins])
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=-1)

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
