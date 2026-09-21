"""Posterior-predictive signals and the residual distribution.

The residual is a *distribution* over posterior samples, not a number attached
to one configuration.  Its lower edge is bounded below by the noise: a fit that
reaches the noise level has extracted the information the data contains, and
several distinct configurations will reach it equally.

This is criterion A of spec Sec. 9.1 -- the only criterion available on
experimental data, where no ground truth exists.
"""

from __future__ import annotations

import numpy as np

from ..state import State


def predictive_signals(trace, expset, site_table, model, *, stride=1, k_max=None):
    """Forward-model signal for each sampled configuration. (n_draws, n_points)

    Offsets are taken from the trace, not rebuilt from site indices.  So are
    lambda and the stretch exponent: the posterior predictive integrates over
    every sampled parameter, and holding the envelope at a nominal value would
    evaluate a configuration the chain never visited.
    """
    step = max(1, int(stride))
    return predictive_from_arrays(
        np.array(trace.site_idx)[::step],
        np.array(trace.k)[::step],
        np.array(trace.dA_par)[::step],
        np.array(trace.dA_perp)[::step],
        np.array(trace.lam)[::step],
        np.array(trace.n_stretch)[::step],
        np.array(trace.sigma)[::step],
        expset, site_table, model,
        k_max=trace.k_max if k_max is None else k_max,
    )


def predictive_from_arrays(site_idx, k, dA_par, dA_perp, lam, n_stretch, sigma,
                           expset, site_table, model, *, k_max=None):
    """Predictive signals from already-extracted trace arrays.

    Separated so that :func:`~nuclear_spin_recovery.post.metrics.summarize` can
    read each trace array exactly once and pass the arrays down, instead of
    handing the trace over to be read again.

    Every draw becomes one replica of a single :class:`State`, so the whole
    posterior is evaluated in one vectorised forward pass rather than a Python
    loop over draws.
    """
    site_idx = np.asarray(site_idx, dtype=int)
    if site_idx.shape[0] == 0:
        return np.empty((0, expset.n_points), dtype=float)
    state = State(
        site_idx=site_idx,
        k=np.asarray(k, dtype=int),
        lam=np.asarray(lam, dtype=float),
        n_stretch=np.asarray(n_stretch, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        n_sites=len(site_table),
        k_max=site_idx.shape[1] if k_max is None else int(k_max),
        dA_par=np.asarray(dA_par, dtype=float),
        dA_perp=np.asarray(dA_perp, dtype=float),
    )
    return model.coherence(state, expset, site_table)


def residual_distribution(observed, predictive, noise):
    """RMS residual of each predictive draw, in units of ``noise``. (n_draws,)"""
    observed = np.asarray(observed, dtype=float)
    predictive = np.atleast_2d(np.asarray(predictive, dtype=float))
    if predictive.size == 0:
        return np.empty(0, dtype=float)
    return np.sqrt(np.mean((observed[None, :] - predictive) ** 2, axis=1)) / noise
