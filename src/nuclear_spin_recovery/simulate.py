"""Synthetic coherence data generation."""

from __future__ import annotations

import numpy as np


def simulate_coherence(state, expset, site_table, model):
    """Noiseless coherence for a known configuration. (n_points,)"""
    raise NotImplementedError


def add_noise(signal, sigma, exp_id=None, rng=None):
    """Add Gaussian noise, optionally with a per-experiment sigma."""
    raise NotImplementedError


def simulate_dataset(state, expset, site_table, model, sigma, rng=None):
    """Return a new ExperimentSet with simulated noisy data attached."""
    raise NotImplementedError
