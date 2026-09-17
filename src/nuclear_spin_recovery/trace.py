"""Recorded chain history.

A :class:`Trace` is the concatenated output of every sub-algorithm in a
schedule.  Only the cold chain is ever recorded: hot tempering replicas exist
to ferry configurations across barriers and do not target the posterior.
See docs/model-specification.md Sec. 8.4 and Sec. 8.5.
"""

from __future__ import annotations

import numpy as np


class Trace:
    """Append-only record of states visited by the sampler.

    Arrays are indexed (n_steps, ...) and hold replica 0 only.
    """

    def __init__(self, n_sites, k_max, n_exp):
        self.n_sites = int(n_sites)
        self.k_max = int(k_max)
        self.n_exp = int(n_exp)

    def __len__(self) -> int:
        raise NotImplementedError

    def append(self, state, log_prob, algorithm=""):
        """Record the cold replica of ``state``.

        Values are copied, not referenced: the sampler mutates states in place
        between steps, and a trace holding views would silently rewrite its own
        history.
        """
        raise NotImplementedError

    @property
    def site_idx(self):
        """(n_steps, k_max) int"""
        raise NotImplementedError

    @property
    def k(self):
        """(n_steps,) int"""
        raise NotImplementedError

    @property
    def lam(self):
        """(n_steps, n_exp) float"""
        raise NotImplementedError

    @property
    def n_stretch(self):
        """(n_steps, n_exp) float"""
        raise NotImplementedError

    @property
    def sigma(self):
        """(n_steps, n_exp) float"""
        raise NotImplementedError

    @property
    def log_prob(self):
        """(n_steps,) float"""
        raise NotImplementedError

    @property
    def algorithm(self):
        """(n_steps,) str -- which sub-algorithm produced each step."""
        raise NotImplementedError

    def discard_burn_in(self, n_burn):
        """Return a new Trace holding the steps after ``n_burn``."""
        raise NotImplementedError
