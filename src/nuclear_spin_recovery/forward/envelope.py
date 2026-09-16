"""Empirical decoherence envelopes.

The envelope absorbs dephasing not captured by the explicitly modeled spins.
See docs/model-specification.md Sec. 4.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
        raise NotImplementedError
