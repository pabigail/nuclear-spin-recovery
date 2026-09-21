"""Trajectory diagnostics.

Spec Sec. 9.2, last paragraph: residual, k, and individual parameters tracked
against trace step, with burn-in marked and ensembles overlaid.

``matplotlib`` is imported **inside each function**, not at module scope.  It is
an optional extra rather than a runtime dependency, and a compute node running
ensembles needs this package to write summaries without pulling a plotting
stack.  Install it with the ``[plot]`` extra.

These functions never compute a metric.  They take arrays or a
:class:`~nuclear_spin_recovery.post.metrics.PosteriorSummary`, take an optional
``ax``, and return the ``Axes`` -- so computation stays testable separately from
drawing.
"""

from __future__ import annotations


def _pyplot():
    """Import pyplot on demand, with an actionable message if it is absent."""
    raise NotImplementedError


def plot_residual(steps, residual, *, ax=None, burn=None, noise_level=1.0, **kwargs):
    """Residual against step count, with the noise level marked."""
    raise NotImplementedError


def plot_dimension(k, *, ax=None, burn=None, k_true=None, **kwargs):
    """Sampled k against step count."""
    raise NotImplementedError


def plot_parameter(values, *, ax=None, burn=None, truth=None, label=None, **kwargs):
    """One continuous parameter against step count."""
    raise NotImplementedError


def plot_detection_by_band(summary, *, ax=None, bands=None, **kwargs):
    """Detection rate per coupling band."""
    raise NotImplementedError
