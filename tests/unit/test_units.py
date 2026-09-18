"""Unit conventions and physical constants."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import units


def test_two_pi():
    assert units.TWO_PI == pytest.approx(2.0 * np.pi)


def test_to_angular_multiplies_by_two_pi():
    assert units.to_angular(1.0) == pytest.approx(2.0 * np.pi)
    assert units.to_angular(100.0) == pytest.approx(200.0 * np.pi)


def test_to_angular_is_vectorized():
    got = units.to_angular(np.array([1.0, 2.0]))
    assert got == pytest.approx(np.array([2.0 * np.pi, 4.0 * np.pi]))


def test_to_angular_preserves_sign():
    """A_par is routinely negative; the conversion must not take a modulus."""
    assert units.to_angular(-5.0) == pytest.approx(-10.0 * np.pi)


def test_khz_times_ms_is_dimensionless():
    """The unit triple is chosen so omega * tau needs no conversion factor.

    1 kHz expressed as angular frequency is 2*pi rad/ms, so over tau = 1 ms
    the accumulated phase is exactly 2*pi.
    """
    omega = units.to_angular(1.0)   # rad / ms
    tau = 1.0                       # ms
    assert omega * tau == pytest.approx(2.0 * np.pi)


def test_gyromagnetic_ratio_13c():
    """gamma/2pi = 10.7084 MHz/T, expressed in rad/(ms*G)."""
    assert units.gyromagnetic_ratio("13C") == pytest.approx(6.7283, rel=1e-3)


def test_gyromagnetic_ratio_29si_is_negative():
    """29Si has a negative gyromagnetic ratio; the sign is physical."""
    assert units.gyromagnetic_ratio("29Si") == pytest.approx(-5.319, rel=1e-3)


def test_gyromagnetic_ratio_rejects_unknown_isotope():
    with pytest.raises((KeyError, ValueError)):
        units.gyromagnetic_ratio("42Xx")


def test_supported_isotopes_are_resolvable():
    for iso in units.SUPPORTED_ISOTOPES:
        assert np.isfinite(units.gyromagnetic_ratio(iso))


def test_gyro_matches_pycce():
    """Cross-check our constants against PyCCE, when it is installed."""
    pc = pytest.importorskip("pycce")
    for iso in units.SUPPORTED_ISOTOPES:
        assert units.gyromagnetic_ratio(iso) == pytest.approx(
            pc.common_isotopes[iso].gyro, rel=1e-6
        )
