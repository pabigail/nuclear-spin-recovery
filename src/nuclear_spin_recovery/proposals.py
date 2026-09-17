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

    def log_prior(self, value):
        """Log prior density at ``value``.

        Zero for kernels whose parameter has no proper prior -- the lattice
        constraint and the prior on k enter elsewhere (spec Sec. 7.2).  The
        hyperfine offsets of Sec. 5.3 are the exception: their Gaussian prior is
        proper and must appear in the acceptance ratio.
        """
        return 0.0


class ContinuousReflected(Proposal):
    """Uniform step of at most ``radius``, reflected at the domain bounds.

    Reflection preserves symmetry, so the log proposal ratio is exactly zero.
    Applied to lambda, the stretch exponent, and sigma.
    """

    def __init__(self, radius, lower=0.0, upper=1.0):
        self.radius = float(radius)
        self.lower = float(lower)
        self.upper = float(upper)
        if self.radius < 0.0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        if self.lower >= self.upper:
            raise ValueError(f"lower {lower} must be below upper {upper}")

    def propose(self, rng, current, occupied=None):
        step = rng.uniform(-self.radius, self.radius, size=np.shape(current))
        return self._reflect(np.asarray(current, dtype=float) + step), 0.0

    def _reflect(self, x):
        """Fold x back into [lower, upper] by repeated reflection.

        Reflection, not clipping: clipping piles probability mass onto the
        boundary and destroys the symmetry the zero proposal ratio assumes.
        """
        span = self.upper - self.lower
        y = np.mod(x - self.lower, 2.0 * span)
        y = np.where(y > span, 2.0 * span - y, y)
        return self.lower + y


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
        return self.neighbors.radius

    def propose(self, rng, current, occupied=None):
        """Propose a new site for the spin currently at ``current``.

        If no unoccupied neighbour exists the move is a no-op: the current
        site is returned with a zero log ratio.
        """
        current = int(current)
        occupied = np.asarray(occupied, dtype=bool)

        candidates = self.neighbors.neighbors(current)
        # The spin being moved must not block its own departure.
        free = candidates[~occupied[candidates]] if candidates.size else candidates
        if free.size == 0:
            return current, 0.0

        proposed = int(rng.choice(free))
        forward = self.neighbors.count_available(current, occupied, ignore=current)
        reverse = self.neighbors.count_available(proposed, occupied, ignore=current)
        if forward == 0 or reverse == 0:
            return current, 0.0
        return proposed, float(np.log(forward) - np.log(reverse))


class GaussianOffset(ContinuousReflected):
    """Continuous walk over a hyperfine offset, under a Gaussian prior.

    Relaxes the hard *ab initio* constraint (spec Sec. 5.3): a spin's coupling
    becomes its table value plus an offset drawn against N(0, scale**2), so the
    prior is centred on the DFT prediction and its width encodes how far that
    prediction is trusted.

    Unlike the other kernels this one carries a proper prior, which enters the
    acceptance ratio explicitly.
    """

    def __init__(self, radius, scale, bound=None):
        bound = 5.0 * scale if bound is None else float(bound)
        super().__init__(radius, lower=-bound, upper=bound)
        self.scale = float(scale)

    def log_prior(self, value):
        raise NotImplementedError
