"""Empirical decoherence envelopes.

The envelope absorbs dephasing not captured by the explicitly modeled spins.
See docs/model-specification.md Sec. 4.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Envelope(ABC):
    """Multiplicative attenuation applied to the spin-bath modulation."""

    @abstractmethod
    def __call__(self, tau, exp_id, lam, n_stretch):
        """Envelope value at each point. (n_replicas, n_points)"""


class StretchedExponential(Envelope):
    """exp(-(tau / lam) ** n).

    Reduces to the exponential envelope of the application paper at n = 1.
    """

    def __call__(self, tau, exp_id, lam, n_stretch):
        tau = np.asarray(tau, dtype=float)
        exp_id = np.asarray(exp_id, dtype=int)
        lam_pts = _per_point(lam, exp_id)
        n_pts = _per_point(n_stretch, exp_id)
        return np.exp(-((tau / lam_pts) ** n_pts))


def _per_point(values, exp_id):
    """Spread a per-experiment (R, n_exp) array onto points. (R, n_points)

    A single column is treated as a global value shared by all experiments.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 1:
        return np.repeat(values, exp_id.size, axis=1)
    return values[:, exp_id]
