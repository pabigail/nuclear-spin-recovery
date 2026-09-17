"""Reversible-jump MCMC: birth and death of nuclear spins.

The number of spins is not known a priori, so the model dimension is inferred
alongside the parameters.  See docs/model-specification.md Sec. 8.3.

The prior on k is carried by the dimension kernel rather than appearing as a
separate factor in the acceptance ratio.  Swapping the kernel therefore changes
the effective prior on bath size without touching the sampler.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


class BirthDeathKernel:
    """Proposes k -> k+1 or k -> k-1, and reports the kernel ratio.

    Births draw a site uniformly from those unoccupied; deaths remove one of
    the k live spins uniformly.  Those combinatorial factors are where the
    effective prior on bath size lives.
    """

    def __init__(self, k_max, birth_prob=0.5):
        self.k_max = int(k_max)
        self.birth_prob = float(birth_prob)

    def propose(self, rng, k, n_free):
        """Return ``(k_proposed, move)`` with move in {"birth", "death"}.

        At k = 0 only birth is possible; at k_max only death.  A proposal that
        cannot be made returns the current k with move None.
        """
        raise NotImplementedError

    def log_ratio(self, k, move, n_free):
        """log gamma(k', k) - log gamma(k, k') for the proposed move."""
        raise NotImplementedError


class RJMCMC(Algorithm):
    """Trans-dimensional moves over the number of spins."""

    def __init__(self, block, kernel):
        self.block = block
        self.kernel = kernel

    def step(self, state, target, rng, beta=1.0):
        raise NotImplementedError
