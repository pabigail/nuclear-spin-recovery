"""Proposal kernels for random-walk Metropolis-Hastings.

Every proposal reports both the proposed value and the log of the proposal
ratio ``r(z -> x) / r(x -> z)``, which the acceptance rule needs.  Returning
the ratio from the proposal -- rather than assuming symmetry at the call site
-- is what keeps an asymmetric kernel correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Proposal(ABC):
    """A reversible proposal kernel."""

    @abstractmethod
    def propose(self, rng, current, occupied=None):
        """Return ``(proposed, log_ratio)``.

        ``log_ratio`` is log r(z -> x) - log r(x -> z), zero for a symmetric
        kernel.
        """


class ContinuousReflected(Proposal):
    """Uniform step of at most ``radius``, reflected at the domain bounds.

    Reflection preserves symmetry, so the log proposal ratio is exactly zero.
    Applied to lambda, the stretch exponent, and sigma.
    """

    def __init__(self, radius, lower=0.0, upper=1.0):
        self.radius = float(radius)
        self.lower = float(lower)
        self.upper = float(upper)

    def propose(self, rng, current, occupied=None):
        raise NotImplementedError


class DiscreteLatticeWalk(Proposal):
    """Move one spin to an unoccupied site within ``radius``.

    The occupancy constraint makes this kernel asymmetric: the number of
    available neighbours differs between the current and proposed sites, so

        log_ratio = log |N_R(x) \\ O| - log |N_R(z) \\ O|

    with the moving spin excluded from the occupied set O.  Dropping this term
    yields a chain that still appears to recover correct configurations while
    targeting the wrong distribution.  See spec Sec. 8.2.
    """

    def __init__(self, neighbors):
        self.neighbors = neighbors

    @property
    def radius(self):
        raise NotImplementedError

    def propose(self, rng, current, occupied=None):
        """Propose a new site for the spin currently at ``current``.

        If no unoccupied neighbour exists the move is a no-op: the current
        site is returned with a zero log ratio.
        """
        raise NotImplementedError
