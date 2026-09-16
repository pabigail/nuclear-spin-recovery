"""Unit conventions and physical constants.

Units throughout the package are (kHz, ms, G, Angstrom), chosen so that
products such as ``omega * tau`` are dimensionless with no conversion factor
in the inner loop.  See docs/model-specification.md Sec. 2.

Hyperfine couplings are stored as ordinary frequencies in kHz, matching the
ab initio tables, and converted to angular frequency by ``2 * pi`` before
entering any trigonometric expression.  Gyromagnetic ratios are stored
already in angular units, rad / (ms * G), and carry no additional 2 * pi.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

#: Isotopes with a known gyromagnetic ratio.
SUPPORTED_ISOTOPES: tuple[str, ...] = ("13C", "29Si")


def to_angular(a_khz):
    """Convert a hyperfine coupling from kHz to angular frequency, rad/ms."""
    raise NotImplementedError


def gyromagnetic_ratio(isotope: str) -> float:
    """Nuclear gyromagnetic ratio for ``isotope``, in rad / (ms * G).

    Sign convention follows the reference implementation: positive for
    13C, so that omega_L = +gyro * B_z.  See spec Sec. 11.1.
    """
    raise NotImplementedError
