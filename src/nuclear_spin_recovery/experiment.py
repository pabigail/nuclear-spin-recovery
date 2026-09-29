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
    weight: np.ndarray | None = None    # (n_points,) relative measurement time

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
        if self.weight is not None:
            self.weight = np.asarray(self.weight, dtype=float)
            if self.weight.shape != self.tau.shape:
                raise ValueError(
                    f"weight has {self.weight.shape} points, tau has "
                    f"{self.tau.shape}"
                )
            if not np.all(np.isfinite(self.weight)):
                raise ValueError("weight contains non-finite values")
            if np.any(self.weight < 0.0):
                raise ValueError(
                    "weight must be non-negative; it is a relative measurement "
                    "time, and a negative one would reward disagreement"
                )

    def __len__(self) -> int:
        return int(self.tau.size)


#: Accepted spellings for the unit of ``tau``, and the factor to milliseconds.
TAU_UNITS = {"ms": 1.0, "us": 1e-3, "µs": 1e-3, "ns": 1e-6, "s": 1e3}


@dataclass
class ExperimentSet:
    """Several experiments on the same defect, fit jointly."""

    experiments: list

    @classmethod
    def from_arrays(cls, tau, coherence, n_pulses, b_z, *, sigma, tau_units,
                    weight=None):
        """One measured experiment, from plain arrays.

        The entry point for data that was not simulated: the user supplies the
        interpulse spacings, the measured coherence, the pulse number, the
        field, and the noise level.

        ``tau_units`` has **no default**, deliberately.  Interpulse spacings are
        quoted in microseconds as often as in milliseconds, and the package
        works in milliseconds; a default would silently accept a trace off by
        three orders of magnitude, which produces a confident fit to the wrong
        physics rather than an error.

        ``sigma`` is **required** for the same reason in reverse.  Criterion A
        -- the only criterion available without ground truth -- is the residual
        measured in units of the noise, and the noise cannot be recovered from
        a single dynamical-decoupling trace: at this sampling density every
        estimator is floored by the modulation itself.  Measured on simulated
        NV data at a true sigma of 0.002, successive differences give 0.062,
        second differences 0.016, and the scatter in the decayed tail 0.017 --
        all of them signal structure, overestimating by eight- to thirty-fold.
        The noise has to come from the measurement.

        ``weight`` is optional and is the relative averaging time at each point
        -- the output of an adaptive design, or the repetition counts of an
        unevenly acquired trace.  Omitted, every point counts once.
        """
        return cls([_from_arrays(tau, coherence, n_pulses, b_z, sigma=sigma,
                                 tau_units=tau_units, weight=weight)])

    @classmethod
    def from_records(cls, records, *, tau_units):
        """Several measured experiments, from a sequence of mappings.

        Each record carries ``tau``, ``coherence``, ``n_pulses``, ``b_z`` and
        ``sigma``, and may carry ``weight``.  They need not share a grid or a
        grid length, and need not agree on whether they are weighted.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one record is required")
        built = []
        for i, record in enumerate(records):
            missing = sorted({"tau", "coherence", "n_pulses", "b_z", "sigma"}
                             - set(record))
            if missing:
                raise ValueError(f"record {i} is missing {missing}")
            built.append(_from_arrays(
                record["tau"], record["coherence"], record["n_pulses"],
                record["b_z"], sigma=record["sigma"], tau_units=tau_units,
                weight=record.get("weight")))
        return cls(built)

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
    def weight_all(self):
        """Relative measurement time per point. (n_points,)

        One everywhere unless an experiment sets it, so the weighted
        likelihood reduces exactly to the unweighted one by default.

        A weight is how much averaging a point received relative to the rest,
        so the effective noise there is ``sigma_e / sqrt(w_j)``: doubling the
        repetitions halves the variance.  Expressed as a weight rather than as
        a per-point sigma because ``sigma_e`` is a *sampled* parameter -- a
        literal per-point array would leave nothing for the sampler to update.
        """
        self._require_non_empty()
        return np.concatenate([
            np.ones(len(e)) if e.weight is None else e.weight
            for e in self.experiments])

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


def _from_arrays(tau, coherence, n_pulses, b_z, *, sigma, tau_units,
                 weight=None):
    """One validated :class:`Experiment` from user arrays.

    Every check here guards a way a measured trace can be silently wrong rather
    than loudly broken.  A nan reaching the likelihood makes every acceptance
    ratio nan, so the chain freezes without reporting anything; spacings out of
    order mean the arrays were assembled from mismatched columns; the wrong
    unit produces a confident fit to physics three decades away.
    """
    if tau_units not in TAU_UNITS:
        raise ValueError(
            f"unknown tau_units {tau_units!r}; known units are "
            f"{sorted(TAU_UNITS)}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"sigma must be positive and finite, got {sigma}; it is the noise "
            f"the residual is measured against and cannot be inferred from a "
            f"single trace")

    tau = np.asarray(tau, dtype=float)
    coherence = np.asarray(coherence, dtype=float)
    if tau.shape != coherence.shape:
        raise ValueError(
            f"tau has {tau.shape} points and coherence has {coherence.shape}")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau contains non-finite values")
    if not np.all(np.isfinite(coherence)):
        raise ValueError(
            "coherence contains non-finite values; drop those points rather "
            "than passing them, or a nan reaches every acceptance ratio")
    if np.any(tau <= 0.0):
        raise ValueError("tau must be strictly positive")
    if np.any(np.diff(tau) <= 0.0):
        raise ValueError(
            "tau must be strictly increasing; out-of-order spacings usually "
            "mean the arrays were assembled from mismatched columns")

    if weight is not None:
        weight = np.asarray(weight, dtype=float)
        if weight.shape != tau.shape:
            raise ValueError(
                f"weight has {weight.shape} points and tau has {tau.shape}")

    return Experiment(tau=tau * TAU_UNITS[tau_units], n_pulses=n_pulses,
                      b_z=b_z, data=coherence, sigma=sigma, weight=weight)
