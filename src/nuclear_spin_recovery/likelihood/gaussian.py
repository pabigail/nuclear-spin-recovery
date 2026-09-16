"""Gaussian L2 likelihood.

Unnormalized: an exact match gives zero.  See spec Sec. 7.1.
"""

from __future__ import annotations

import numpy as np

from .base import Likelihood


class GaussianL2(Likelihood):
    """-sum_e (1 / 2 sigma_e**2) sum_j (d_ej - f_e(tau_ej))**2."""

    def log_prob(self, state, expset, model, site_table):
        raise NotImplementedError
