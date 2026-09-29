"""Synthetic coherence data generation."""

from __future__ import annotations

import numpy as np

from .experiment import Experiment, ExperimentSet


def simulate_coherence(state, expset, site_table, model):
    """Noiseless coherence for a known configuration. (n_points,)"""
    return model.coherence(state, expset, site_table)[0]


def add_noise(signal, sigma, exp_id=None, rng=None, weight=None):
    """Add Gaussian noise, optionally per-experiment and per-point.

    ``weight`` is the relative measurement time at each point: a point measured
    four times as long is half as noisy, so its noise is sigma / sqrt(w_j).
    Omitted or all-ones, the draw is identical to the unweighted one -- the
    same random numbers, not merely the same distribution.
    """
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim > 0 and exp_id is not None:
        sigma = sigma[np.asarray(exp_id, dtype=int)]
    draw = rng.normal(0.0, 1.0, size=signal.shape) * sigma
    if weight is not None:
        weight = np.asarray(weight, dtype=float)
        # A zero-weight point was not measured; leave the noiseless value
        # rather than dividing by zero.
        scale = np.divide(1.0, np.sqrt(weight), out=np.ones_like(weight),
                          where=weight > 0.0)
        draw = draw * scale
    return signal + draw


def simulate_dataset(state, expset, site_table, model, sigma, rng=None):
    """Return a new ExperimentSet with simulated noisy data attached.

    The input set is left untouched.
    """
    truth = simulate_coherence(state, expset, site_table, model)
    noisy = add_noise(truth, sigma, exp_id=expset.exp_id, rng=rng,
                      weight=expset.weight_all)
    pieces = expset.split(noisy)
    return ExperimentSet(
        [
            Experiment(
                tau=e.tau.copy(),
                n_pulses=e.n_pulses,
                b_z=e.b_z,
                data=piece,
                sigma=e.sigma,
                weight=e.weight,
            )
            for e, piece in zip(expset.experiments, pieces)
        ]
    )
