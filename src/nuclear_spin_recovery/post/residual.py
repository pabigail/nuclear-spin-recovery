"""Posterior-predictive signals and the residual distribution.

The residual is a *distribution* over posterior samples, not a number attached
to one configuration.  Its lower edge is bounded below by the noise: a fit that
reaches the noise level has extracted the information the data contains, and
several distinct configurations will reach it equally.

This is criterion A of spec Sec. 9.1 -- the only criterion available on
experimental data, where no ground truth exists.
"""

from __future__ import annotations


def predictive_signals(trace, expset, site_table, model, *, stride=1, k_max=None):
    """Forward-model signal for each sampled configuration. (n_draws, n_points)

    Offsets are taken from the trace, not rebuilt from site indices.
    """
    raise NotImplementedError


def residual_distribution(observed, predictive, noise):
    """RMS residual of each predictive draw, in units of ``noise``. (n_draws,)"""
    raise NotImplementedError
