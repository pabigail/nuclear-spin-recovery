"""Dynamical decoupling experiments and their joint flattening.

Experiments differ in pulse number, field, and tau sampling, and need not
share a grid or even a grid length.  :class:`ExperimentSet` concatenates them
into flat arrays with an experiment index, so the joint likelihood is one
vectorized pass.  See docs/model-specification.md Sec. 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        self.tau = np.asarray(self.tau, dtype=float)
        if self.tau.ndim != 1 or self.tau.size == 0:
            raise ValueError("tau must be a non-empty 1-D array")
        if not np.all(self.tau > 0.0):
            raise ValueError("tau must be strictly positive (ms)")
        if int(self.n_pulses) != self.n_pulses or self.n_pulses < 0:
            raise ValueError("n_pulses must be a non-negative integer")
        self.n_pulses = int(self.n_pulses)
        if not np.isfinite(self.b_z):
            raise ValueError("b_z must be finite (G)")
        self.b_z = float(self.b_z)
        if self.data is not None:
            self.data = np.asarray(self.data, dtype=float)
            if self.data.shape != self.tau.shape:
                raise ValueError(
                    f"data has {self.data.shape} points, tau has {self.tau.shape}"
                )

    def __len__(self) -> int:
        return int(self.tau.size)


@dataclass
class ExperimentSet:
    """Several experiments on the same defect, fit jointly."""

    experiments: list

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    @property
    def n_points(self) -> int:
        self._require_non_empty()
        return int(sum(len(e) for e in self.experiments))

    def _require_non_empty(self):
        if not self.experiments:
            raise ValueError("ExperimentSet is empty")

    @property
    def tau_all(self):
        """All tau values, experiments concatenated in order. (n_points,)"""
        self._require_non_empty()
        return np.concatenate([e.tau for e in self.experiments])

    @property
    def exp_id(self):
        """Experiment index for each point. (n_points,) int"""
        self._require_non_empty()
        return np.concatenate(
            [np.full(len(e), i, dtype=int) for i, e in enumerate(self.experiments)]
        )

    @property
    def n_pulses_per_point(self):
        """N gathered onto points via exp_id. (n_points,)"""
        self._require_non_empty()
        return np.array([e.n_pulses for e in self.experiments])[self.exp_id]

    @property
    def b_z_per_point(self):
        """B_z gathered onto points via exp_id. (n_points,)"""
        self._require_non_empty()
        return np.array([e.b_z for e in self.experiments])[self.exp_id]

    @property
    def data_all(self):
        """Measured coherence, concatenated. (n_points,)"""
        self._require_non_empty()
        missing = [i for i, e in enumerate(self.experiments) if e.data is None]
        if missing:
            raise ValueError(f"experiments {missing} carry no data")
        return np.concatenate([e.data for e in self.experiments])

    def split(self, flat):
        """Split a flat (n_points,) array back into per-experiment pieces."""
        flat = np.asarray(flat)
        if flat.shape[-1] != self.n_points:
            raise ValueError(
                f"expected {self.n_points} points, got {flat.shape[-1]}"
            )
        bounds = np.cumsum([len(e) for e in self.experiments])[:-1]
        return list(np.split(flat, bounds, axis=-1))
