"""The sampler state: nuclear spin configuration plus envelope parameters.

Spins are packed into slots ``[0:k)`` of fixed-width arrays; slots beyond k
are undefined.  A leading replica axis is 1 outside a tempering block and J
inside one.  See docs/model-specification.md Sec. 6.
"""

from __future__ import annotations

import numpy as np


class State:
    """Configuration of k nuclear spins and the per-experiment parameters.

    Arrays carry a leading replica axis R.

    site_idx   (R, k_max)   int    index into the SiteTable
    k          (R,)         int    active spin count
    occupied   (R, n_sites) bool   occupancy bitmap, derived from site_idx
    lam        (R, n_exp)   float  decay constant, ms
    n_stretch  (R, n_exp)   float  stretch exponent
    sigma      (R, n_exp)   float  noise std
    """

    def __init__(self, site_idx, k, lam, n_stretch, sigma, n_sites, k_max):
        # TODO(phase-1): validate shapes and derive the occupancy bitmap.
        # Assignment only, so fixtures can build a state to exercise methods.
        self.site_idx = site_idx
        self.k = k
        self.lam = lam
        self.n_stretch = n_stretch
        self.sigma = sigma
        self.n_sites = n_sites
        self.k_max = k_max

    @classmethod
    def from_sites(cls, sites, *, n_sites, n_exp, lam, n_stretch, sigma, k_max):
        """Build a single-replica state from a sequence of site indices."""
        raise NotImplementedError

    @property
    def n_replicas(self) -> int:
        raise NotImplementedError

    @property
    def n_exp(self) -> int:
        raise NotImplementedError

    @property
    def occupied(self):
        raise NotImplementedError

    def copy(self):
        """Deep copy; no array is shared with the original."""
        raise NotImplementedError

    def expand_replicas(self, n_replicas):
        """Return a state with R identical replicas, for a tempering block."""
        raise NotImplementedError

    def collapse_to_cold(self):
        """Return replica 0 only, discarding the hot chains."""
        raise NotImplementedError

    def gyro_per_spin(self, site_table):
        """Gyromagnetic ratio of each active spin. (R, k_max)

        Determined by the site, never sampled independently.  See spec Sec. 6.
        """
        raise NotImplementedError

    def a_par_per_spin(self, site_table):
        """Parallel hyperfine component of each active spin, kHz. (R, k_max)"""
        raise NotImplementedError

    def a_perp_per_spin(self, site_table):
        """Perpendicular hyperfine component of each active spin, kHz."""
        raise NotImplementedError

    def check_invariants(self):
        """Raise if the state is malformed.

        Checks that k is within bounds, that occupancy agrees with the active
        slice of site_idx, and that no site is doubly occupied.
        """
        raise NotImplementedError
