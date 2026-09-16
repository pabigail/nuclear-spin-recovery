"""Dynamical decoupling experiments and their joint flattening.

Experiments differ in pulse number, field, and tau sampling, and need not
share a grid or even a grid length.  :class:`ExperimentSet` concatenates them
into flat arrays with an experiment index, so the joint likelihood is one
vectorized pass.  See docs/model-specification.md Sec. 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Experiment:
    """A single CPMG-N coherence measurement."""

    tau: np.ndarray            # (n_points,) ms
    n_pulses: int
    b_z: float                 # G
    data: np.ndarray | None = None      # (n_points,) measured coherence
    sigma: float | None = None          # noise std, initial value

    def __post_init__(self):
        # TODO(phase-1): validate tau > 0, b_z finite, n_pulses a non-negative
        # int, and len(data) == len(tau).  Left permissive so fixtures build.
        pass

    def __len__(self) -> int:
        raise NotImplementedError


@dataclass
class ExperimentSet:
    """Several experiments on the same defect, fit jointly."""

    experiments: list

    @property
    def n_experiments(self) -> int:
        raise NotImplementedError

    @property
    def n_points(self) -> int:
        raise NotImplementedError

    @property
    def tau_all(self):
        """All tau values, experiments concatenated in order. (n_points,)"""
        raise NotImplementedError

    @property
    def exp_id(self):
        """Experiment index for each point. (n_points,) int"""
        raise NotImplementedError

    @property
    def n_pulses_per_point(self):
        """N gathered onto points via exp_id. (n_points,)"""
        raise NotImplementedError

    @property
    def b_z_per_point(self):
        """B_z gathered onto points via exp_id. (n_points,)"""
        raise NotImplementedError

    @property
    def data_all(self):
        """Measured coherence, concatenated. (n_points,)"""
        raise NotImplementedError

    def split(self, flat):
        """Split a flat (n_points,) array back into per-experiment pieces."""
        raise NotImplementedError
