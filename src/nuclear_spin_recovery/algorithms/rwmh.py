"""Random-walk Metropolis-Hastings, continuous and discrete.

A single parameter is proposed from a domain-aware kernel and accepted with

    alpha = min(1, [L(p') / L(p)] * [prior ratio] * [proposal ratio])

See docs/model-specification.md Sec. 8.1 and Sec. 8.2.
"""

from __future__ import annotations

import numpy as np

from .base import Algorithm


class RWMH(Algorithm):
    """Metropolis-Hastings over a fixed-dimension parameter block."""

    def __init__(self, block, proposal):
        self.block = block
        self.proposal = proposal

    def step(self, state, target, rng, beta=1.0):
        """Propose and accept or reject, independently per replica."""
        raise NotImplementedError
