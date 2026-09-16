"""Analytic CCE-1 forward model.

Equivalent to CCE-1: nuclear spins modulate the electronic coherence
independently, with no nuclear-nuclear coupling.  See spec Sec. 4.1.
"""

from __future__ import annotations

import numpy as np

from .base import ForwardModel


def single_spin_modulation(tau, a_par, a_perp, n_pulses, b_z, gyro):
    """Modulation M_i contributed by one nuclear spin.

    Parameters are in package units: tau in ms, couplings in kHz, b_z in G,
    gyro in rad / (ms * G).  Couplings are converted to angular frequency
    internally; gyro is already angular.

    omega_L = +gyro * b_z, following the reference implementation rather than
    the sign printed in both papers.  See spec Sec. 11.1.
    """
    raise NotImplementedError


class AnalyticCCE1(ForwardModel):
    """Closed-form coherence for a non-interacting nuclear spin bath."""

    def __init__(self, envelope):
        self.envelope = envelope

    def coherence(self, state, expset, site_table):
        raise NotImplementedError
