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


#: Nuclear gyromagnetic ratios, rad / (ms * G), matching PyCCE's isotope table.
#: Signs are physical: 29Si precesses opposite to 13C.
_GYRO = {
    "13C": 6.7282853242747445,
    "29Si": -5.3190301724949975,
}


def to_angular(a_khz):
    """Convert a hyperfine coupling from kHz to angular frequency, rad/ms."""
    return TWO_PI * np.asarray(a_khz, dtype=float)


def gyromagnetic_ratio(isotope: str) -> float:
    """Nuclear gyromagnetic ratio for ``isotope``, in rad / (ms * G).

    Sign convention follows the reference implementation: positive for
    13C, so that omega_L = +gyro * B_z.  See spec Sec. 11.1.
    """
    try:
        return _GYRO[isotope]
    except KeyError:
        raise KeyError(
            f"unknown isotope {isotope!r}; known: {sorted(_GYRO)}"
        ) from None
