"""Parallel tempering.

J replicas run at inverse temperatures beta_j = 2^-j, zero-indexed so replica 0
is the cold chain sampling the true posterior.  Hot replicas exist to carry
configurations across barriers; only the cold chain is retained.  See
docs/model-specification.md Sec. 8.4.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


def geometric_ladder(n_replicas):
    """beta_j = 2 ** -j for j = 0 .. n_replicas - 1, so beta_0 = 1."""
    raise NotImplementedError


class ParallelTempering(Algorithm):
    """Runs an inner schedule on each rung of a temperature ladder.

    The inner argument is a Schedule rather than a single algorithm: a rung
    typically advances both continuous and discrete blocks before a swap is
    attempted.
    """

    def __init__(self, inner, n_replicas=10, betas=None):
        self.inner = inner
        self.n_replicas = int(n_replicas)
        self.betas = betas

    @property
    def block(self):
        """PT has no single block; it inherits whatever its inner schedule updates."""
        raise NotImplementedError

    def step(self, state, target, rng, beta=1.0):
        """Advance every rung one inner pass, then attempt one swap.

        Expects a state already expanded to ``n_replicas`` replicas.
        """
        raise NotImplementedError

    def attempt_swap(self, state, target, rng):
        """Propose exchanging two rungs, accept or reject, return the state."""
        raise NotImplementedError

    def run(self, state, target, rng, n_steps, trace=None, beta=1.0):
        """Expand to the ladder, advance, then collapse to the cold chain."""
        raise NotImplementedError
