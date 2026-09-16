"""Analytic CCE-1 forward model.

Equivalent to CCE-1: nuclear spins modulate the electronic coherence
independently, with no nuclear-nuclear coupling.  See spec Sec. 4.1.
"""

from __future__ import annotations

import numpy as np

from ..units import to_angular
from .base import ForwardModel


def single_spin_modulation(tau, a_par, a_perp, n_pulses, b_z, gyro):
    """Modulation M_i contributed by one nuclear spin.

    Parameters are in package units: tau in ms, couplings in kHz, b_z in G,
    gyro in rad / (ms * G).  Couplings are converted to angular frequency
    internally; gyro is already angular.

    omega_L = +gyro * b_z, following the reference implementation rather than
    the sign printed in both papers.  See spec Sec. 11.1.

    All arguments broadcast against one another, so this serves both a single
    spin over a tau grid and a whole bath over every point at once.
    """
    a_par = to_angular(a_par)
    a_perp = to_angular(a_perp)
    tau = np.asarray(tau, dtype=float)

    omega_l = np.asarray(gyro, dtype=float) * np.asarray(b_z, dtype=float)
    omega_1 = a_par + omega_l
    omega = np.sqrt(omega_1**2 + a_perp**2)
    m_z = omega_1 / omega
    m_x = a_perp / omega

    alpha = omega * tau
    eta = omega_l * tau
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_e, sin_e = np.cos(eta), np.sin(eta)

    # The denominator is 1 + cos(phi) and vanishes on resonance.  Where the
    # numerator vanishes too -- notably whenever A_perp is zero, so m_x is
    # zero -- the ratio is identically zero and must not be evaluated.
    numer = m_x**2 * (1.0 - cos_a) * (1.0 - cos_e)
    denom = 1.0 + cos_a * cos_e - m_z * sin_a * sin_e
    ratio = np.zeros(np.broadcast(numer, denom).shape, dtype=float)
    live = numer != 0.0
    np.divide(numer, denom, out=ratio, where=live)

    phi = np.arccos(np.clip(denom - 1.0, -1.0, 1.0))
    return 1.0 - ratio * np.sin(np.asarray(n_pulses) * phi / 2.0) ** 2


class AnalyticCCE1(ForwardModel):
    """Closed-form coherence for a non-interacting nuclear spin bath."""

    def __init__(self, envelope):
        self.envelope = envelope

    def coherence(self, state, expset, site_table):
        tau = expset.tau_all[None, None, :]
        n_pulses = expset.n_pulses_per_point[None, None, :]
        b_z = expset.b_z_per_point[None, None, :]

        active = state._active_mask()[:, :, None]
        a_par = state.a_par_per_spin(site_table)[:, :, None]
        a_perp = state.a_perp_per_spin(site_table)[:, :, None]
        # Empty slots gather zeros, which would make omega zero and the
        # direction cosines 0/0.  Any nonzero gyro keeps them finite; their
        # A_perp is zero, so they contribute a factor of one either way.
        gyro = np.where(active, state.gyro_per_spin(site_table)[:, :, None], 1.0)

        modulation = single_spin_modulation(tau, a_par, a_perp, n_pulses, b_z, gyro)
        modulation = np.where(active, modulation, 1.0)
        product = np.prod(modulation, axis=1)

        envelope = self.envelope(
            np.broadcast_to(expset.tau_all, product.shape),
            expset.exp_id,
            state.lam,
            state.n_stretch,
        )
        # Envelope sits inside the half-sum: spec Sec. 4.2, Jung et al. Eq. 5.
        return 0.5 * (1.0 + product * envelope)
