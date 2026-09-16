"""Lattice sites and ab initio hyperfine couplings.

A :class:`SiteTable` is the discrete domain over which nuclear spins walk.
Each site carries a position, an isotope identity, the gyromagnetic ratio
implied by that identity, and the secular hyperfine components derived from
the DFT tensor.  See docs/model-specification.md Sec. 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Column order of the Ivady-format hyperfine file (after the leading index).
IVADY_COLUMNS = (
    "distance", "x", "y", "z",
    "A_xx", "A_yy", "A_zz", "A_xy", "A_xz", "A_yz",
)

MHZ_TO_KHZ = 1000.0


def read_hyperfine_table(path):
    """Parse an Ivady-format hyperfine file into raw, unfiltered arrays.

    Returns a mapping with the keys of :data:`IVADY_COLUMNS`.  Distances and
    positions are in Angstrom; tensor components are converted from MHz (as
    stored) to kHz.  No filtering or derivation is performed here.
    """
    raise NotImplementedError


def secular_components(a_zz, a_xz, a_yz):
    """Derive ``(A_parallel, A_perp)`` from tensor components, in kHz.

    ``A_parallel = A_zz`` and ``A_perp = sqrt(A_xz**2 + A_yz**2)``, with the
    defect axis along z.
    """
    raise NotImplementedError


@dataclass
class SiteTable:
    """Candidate nuclear spin sites with their hyperfine couplings."""

    distance: np.ndarray      # (n_sites,)      Angstrom, as stored
    positions: np.ndarray     # (n_sites, 3)    Angstrom
    a_par: np.ndarray         # (n_sites,)      kHz
    a_perp: np.ndarray        # (n_sites,)      kHz
    isotope: np.ndarray       # (n_sites,)      str
    gyro: np.ndarray          # (n_sites,)      rad / (ms * G)

    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_ivady_file(cls, path, *, strong_thresh, weak_thresh, isotope="13C"):
        """Load a site table from an Ivady-format file.

        Sites are filtered twice: those with *either* component above
        ``strong_thresh`` are dropped, then those with *both* components
        below ``weak_thresh`` are dropped.  Both bounds are inclusive.
        """
        raise NotImplementedError

    @classmethod
    def from_ase(cls, atoms, *, strong_thresh, weak_thresh, defect_index=0):
        """Build a site table from an ASE ``Atoms`` object.

        Hyperfine couplings are taken in the point-dipole approximation
        relative to the defect at ``defect_index``.
        """
        raise NotImplementedError

    def symmetry_groups(self, tol=0.1):
        """Label sites whose couplings agree within ``tol`` kHz.

        Returns an integer array of group labels, one per site.  Sites in
        the same group are indistinguishable in coherence data.
        """
        raise NotImplementedError
