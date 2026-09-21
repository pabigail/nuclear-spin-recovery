"""Posterior summaries over a recorded chain.

Every summary here is computed **over the posterior**.  There is deliberately no
entry point that accepts a single :class:`~nuclear_spin_recovery.state.State`:
judging a recovery by one configuration -- the last state of a chain, or the
modal sample -- measures the wrong object, and a spin absent from the modal
configuration may still appear in most samples.  See spec Sec. 9.1.

The posterior is trans-dimensional, so there is no stable correspondence between
"spin 3" in one sample and "spin 3" in another.  Summaries are therefore defined
on couplings rather than by per-parameter averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import BANDS


@dataclass
class PosteriorSummary:
    """Everything spec Sec. 9.2 defines, computed over posterior samples."""

    R_i: np.ndarray            # detection rate per reference spin
    magnitude: np.ndarray      # coupling magnitude of each reference spin, kHz
    residual: np.ndarray       # RMS residual per draw, in units of the noise
    k_posterior: np.ndarray    # sampled values of k
    false_absence: float       # FP of spec Sec. 9.2
    predictive: np.ndarray     # (n_draws, n_points) posterior-predictive signals

    def R(self, lo, hi):
        """Mean detection rate within a coupling band."""
        raise NotImplementedError

    def by_band(self, bands=BANDS):
        """Mean detection rate in each band. (len(bands),)"""
        raise NotImplementedError

    @property
    def median_residual(self):
        """Typical fit quality. Use when comparing like against like."""
        raise NotImplementedError

    @property
    def best_residual(self):
        """Lowest residual any sampled configuration achieves.

        The statistic for comparing a model against one nested inside it.  A
        model with extra sampled parameters has a *higher* median residual than
        one holding them at the prior mean, because a typical draw sits away
        from that mean -- so a median comparison penalises the richer model for
        exploring.  What it should be asked is whether it can reach a fit the
        constrained model cannot.  Only meaningful between runs with the same
        number of posterior samples.
        """
        raise NotImplementedError

    @property
    def k_mode(self):
        """Modal sampled dimension."""
        raise NotImplementedError

    def dimension_discrepancy(self, k_true):
        """|mode(k) - k_true|; zero means the inferred dimension is correct."""
        raise NotImplementedError


def summarize(trace, expset, site_table, model, *, reference=None, burn=0,
              stride=1, noise=1.0, tol=None):
    """Compute every spec Sec. 9.2 metric from a recorded chain.

    ``trace`` is a :class:`~nuclear_spin_recovery.trace.Trace`.  Passing a State
    raises ``TypeError`` from an explicit check, rather than failing later on a
    missing attribute: judging a recovery by one configuration is a modelling
    error, and it should read as a refusal rather than as a bug.

    ``reference`` is the ground-truth site index array
    when one exists, and None on experimental data, where detection is not
    available and ``R_i`` comes back empty.

    Each trace array is read **once**.  Reading a Trace property inside a loop
    rebuilds it every access, which is O(n^2) in the chain length.
    """
    raise NotImplementedError
