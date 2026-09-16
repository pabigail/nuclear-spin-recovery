"""Synthetic coherence data generation."""

from __future__ import annotations

import numpy as np

from .experiment import Experiment, ExperimentSet


def simulate_coherence(state, expset, site_table, model):
    """Noiseless coherence for a known configuration. (n_points,)"""
    return model.coherence(state, expset, site_table)[0]


def add_noise(signal, sigma, exp_id=None, rng=None):
    """Add Gaussian noise, optionally with a per-experiment sigma."""
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim > 0 and exp_id is not None:
        sigma = sigma[np.asarray(exp_id, dtype=int)]
    return signal + rng.normal(0.0, 1.0, size=signal.shape) * sigma


def simulate_dataset(state, expset, site_table, model, sigma, rng=None):
    """Return a new ExperimentSet with simulated noisy data attached.

    The input set is left untouched.
    """
    truth = simulate_coherence(state, expset, site_table, model)
    noisy = add_noise(truth, sigma, exp_id=expset.exp_id, rng=rng)
    pieces = expset.split(noisy)
    return ExperimentSet(
        [
            Experiment(
                tau=e.tau.copy(),
                n_pulses=e.n_pulses,
                b_z=e.b_z,
                data=piece,
                sigma=e.sigma,
            )
            for e, piece in zip(expset.experiments, pieces)
        ]
    )
