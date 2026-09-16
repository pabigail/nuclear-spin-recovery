"""Lattice sites and ab initio hyperfine couplings.

A :class:`SiteTable` is the discrete domain over which nuclear spins walk.
Each site carries a position, an isotope identity, the gyromagnetic ratio
implied by that identity, and the secular hyperfine components derived from
the DFT tensor.  See docs/model-specification.md Sec. 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import TWO_PI, gyromagnetic_ratio

#: Column order of the Ivady-format hyperfine file (after the leading index).
IVADY_COLUMNS = (
    "distance", "x", "y", "z",
    "A_xx", "A_yy", "A_zz", "A_xy", "A_xz", "A_yz",
)

MHZ_TO_KHZ = 1000.0

#: Which isotope each chemical species contributes to the bath.
_ISOTOPE_FOR_SYMBOL = {"C": "13C", "Si": "29Si"}

#: mu_0 / 4pi * gamma_e * hbar, in SI, with gamma_e the electron value.
_MU0_OVER_4PI = 1.0e-7                    # T m / A
_GAMMA_E = 1.76085963e11                  # rad / (s T)
_HBAR = 1.054571817e-34                   # J s


def _point_dipole_tensor(rel, gyro):
    """Secular tensor components in kHz for displacements ``rel`` (Angstrom).

    Uses the point-dipole form A_ij ~ (3 n_i n_j - delta_ij) / r**3, with the
    defect axis along z.  ``gyro`` is per-site, in rad / (ms * G).
    """
    r = np.linalg.norm(rel, axis=1)
    unit = rel / r[:, None]
    # rad/(ms G) -> rad/(s T): x1e3 for ms->s, x1e4 for G->T.
    gyro_si = np.asarray(gyro, dtype=float) * 1.0e7
    r_m = r * 1.0e-10
    pref = _MU0_OVER_4PI * _GAMMA_E * gyro_si * _HBAR / r_m**3   # rad/s
    to_khz = 1.0 / (TWO_PI * 1.0e3)
    nx, ny, nz = unit[:, 0], unit[:, 1], unit[:, 2]
    a_zz = pref * (3.0 * nz * nz - 1.0) * to_khz
    a_xz = pref * (3.0 * nx * nz) * to_khz
    a_yz = pref * (3.0 * ny * nz) * to_khz
    return a_zz, a_xz, a_yz


def read_hyperfine_table(path):
    """Parse an Ivady-format hyperfine file into raw, unfiltered arrays.

    Returns a mapping with the keys of :data:`IVADY_COLUMNS`.  Distances and
    positions are in Angstrom; tensor components are converted from MHz (as
    stored) to kHz.  No filtering or derivation is performed here.
    """
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != len(IVADY_COLUMNS) + 1:
        raise ValueError(
            f"expected {len(IVADY_COLUMNS) + 1} columns (leading index plus "
            f"{IVADY_COLUMNS}), got shape {raw.shape}"
        )
    # Column 0 is a row index, not data.
    table = {name: raw[:, i + 1] for i, name in enumerate(IVADY_COLUMNS)}
    for name in ("A_xx", "A_yy", "A_zz", "A_xy", "A_xz", "A_yz"):
        table[name] = table[name] * MHZ_TO_KHZ
    return table


def secular_components(a_zz, a_xz, a_yz):
    """Derive ``(A_parallel, A_perp)`` from tensor components, in kHz.

    ``A_parallel = A_zz`` and ``A_perp = sqrt(A_xz**2 + A_yz**2)``, with the
    defect axis along z.  A_parallel keeps its sign; A_perp is a magnitude.
    """
    a_par = np.asarray(a_zz, dtype=float)
    a_perp = np.hypot(np.asarray(a_xz, dtype=float), np.asarray(a_yz, dtype=float))
    return a_par, a_perp


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
        return len(self.a_par)

    @classmethod
    def from_ivady_file(cls, path, *, strong_thresh, weak_thresh, isotope="13C"):
        """Load a site table from an Ivady-format file.

        Sites are filtered twice: those with *either* component above
        ``strong_thresh`` are dropped, then those with *both* components
        below ``weak_thresh`` are dropped.  Both bounds are inclusive.
        """
        table = read_hyperfine_table(path)
        a_par, a_perp = secular_components(
            table["A_zz"], table["A_xz"], table["A_yz"]
        )
        keep = (np.abs(a_par) <= strong_thresh) & (a_perp <= strong_thresh)
        keep &= (np.abs(a_par) >= weak_thresh) | (a_perp >= weak_thresh)
        if not keep.any():
            raise ValueError(
                f"no sites survive strong_thresh={strong_thresh}, "
                f"weak_thresh={weak_thresh}"
            )
        positions = np.column_stack(
            [table["x"][keep], table["y"][keep], table["z"][keep]]
        )
        n = int(keep.sum())
        return cls(
            distance=table["distance"][keep],
            positions=positions,
            a_par=a_par[keep],
            a_perp=a_perp[keep],
            isotope=np.full(n, isotope),
            gyro=np.full(n, gyromagnetic_ratio(isotope)),
        )

    @classmethod
    def from_ase(cls, atoms, *, strong_thresh, weak_thresh, defect_index=0):
        """Build a site table from an ASE ``Atoms`` object.

        Hyperfine couplings are taken in the point-dipole approximation
        relative to the defect at ``defect_index``.
        """
        positions = np.asarray(atoms.get_positions(), dtype=float)
        symbols = list(atoms.get_chemical_symbols())
        keep = np.ones(len(positions), dtype=bool)
        keep[defect_index] = False

        rel = positions[keep] - positions[defect_index]
        isotope = np.array([_ISOTOPE_FOR_SYMBOL[s] for s in np.array(symbols)[keep]])
        gyro = np.array([gyromagnetic_ratio(i) for i in isotope])

        a_zz, a_xz, a_yz = _point_dipole_tensor(rel, gyro)
        a_par, a_perp = secular_components(a_zz, a_xz, a_yz)

        sel = (np.abs(a_par) <= strong_thresh) & (a_perp <= strong_thresh)
        sel &= (np.abs(a_par) >= weak_thresh) | (a_perp >= weak_thresh)
        if not sel.any():
            raise ValueError(
                f"no sites survive strong_thresh={strong_thresh}, "
                f"weak_thresh={weak_thresh}"
            )
        return cls(
            distance=np.linalg.norm(rel[sel], axis=1),
            positions=rel[sel],
            a_par=a_par[sel],
            a_perp=a_perp[sel],
            isotope=isotope[sel],
            gyro=gyro[sel],
        )

    def symmetry_groups(self, tol=0.1):
        """Label sites whose couplings agree within ``tol`` kHz.

        Returns an integer array of group labels, one per site.  Sites in
        the same group are indistinguishable in coherence data.

        Grouping is transitive: sites are merged if a chain of pairwise-close
        sites connects them.
        """
        n = len(self)
        labels = np.full(n, -1, dtype=int)
        group = 0
        for seed in range(n):
            if labels[seed] >= 0:
                continue
            labels[seed] = group
            stack = [seed]
            while stack:
                j = stack.pop()
                close = (
                    (np.abs(self.a_par - self.a_par[j]) <= tol)
                    & (np.abs(self.a_perp - self.a_perp[j]) <= tol)
                    & (labels < 0)
                )
                idx = np.flatnonzero(close)
                labels[idx] = group
                stack.extend(idx.tolist())
            group += 1
        return labels
