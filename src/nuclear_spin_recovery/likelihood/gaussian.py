"""Gaussian L2 likelihood.

Unnormalized: an exact match gives zero.  See spec Sec. 7.1.
"""

from __future__ import annotations

import numpy as np

from .base import Likelihood


class GaussianL2(Likelihood):
    """-sum_e (1 / 2 sigma_e**2) sum_j w_ej (d_ej - f_e(tau_ej))**2.

    ``w`` is the per-point relative measurement time, one everywhere unless an
    experiment sets it, so this reduces exactly to the unweighted sum by
    default -- which matters, because every threshold in docs/test-plan.md was
    calibrated without weights.

    A point measured four times as long is half as noisy, so its effective
    sigma is sigma_e / sqrt(w_j) and its squared residual enters weighted by
    w_j.  A weight of zero drops the point, which is the mechanism adaptive
    design needs: a time not measured must cost nothing.
    """

    def log_prob(self, state, expset, model, site_table):
        predicted = model.coherence(state, expset, site_table)
        residual = expset.data_all[None, :] - predicted
        sigma = state.sigma[:, expset.exp_id]
        weight = expset.weight_all[None, :]
        return -0.5 * np.sum(weight * (residual / sigma) ** 2, axis=1)
