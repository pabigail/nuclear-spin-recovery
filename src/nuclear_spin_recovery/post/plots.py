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

import numpy as np

from .detection import BANDS


def _pyplot():
    """Import pyplot on demand, with an actionable message if it is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "plotting requires matplotlib, which is an optional extra: "
            "pip install 'nuclear-spin-recovery[plot]'"
        ) from exc
    return plt


def _axes(ax):
    """The axes to draw on, creating a figure only when none was given."""
    return _pyplot().subplots()[1] if ax is None else ax


def _mark_burn_in(ax, burn):
    """Vertical rule at the end of burn-in, so the reader sees what was cut."""
    if burn is not None:
        ax.axvline(float(burn), color="0.6", ls=":", lw=1.2, label="burn-in")


def plot_residual(steps, residual, *, ax=None, burn=None, noise_level=1.0, **kwargs):
    """Residual against step count, with the noise level marked."""
    ax = _axes(ax)
    kwargs.setdefault("lw", 1.3)
    ax.plot(np.asarray(steps), np.asarray(residual), **kwargs)
    ax.axhline(float(noise_level), color="crimson", ls="--", lw=1.0,
               label=f"{noise_level:g} sigma")
    _mark_burn_in(ax, burn)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("RMS residual (sigma)")
    return ax


def plot_dimension(k, *, ax=None, burn=None, k_true=None, **kwargs):
    """Sampled k against step count."""
    ax = _axes(ax)
    kwargs.setdefault("lw", 0.9)
    ax.plot(np.asarray(k), **kwargs)
    if k_true is not None:
        ax.axhline(float(k_true), color="crimson", ls="--", lw=1.0, label="truth")
    _mark_burn_in(ax, burn)
    ax.set_xlabel("step")
    ax.set_ylabel("$k$")
    return ax


def plot_parameter(values, *, ax=None, burn=None, truth=None, label=None, **kwargs):
    """One continuous parameter against step count."""
    ax = _axes(ax)
    kwargs.setdefault("lw", 0.9)
    ax.plot(np.asarray(values), label=label, **kwargs)
    if truth is not None:
        ax.axhline(float(truth), color="crimson", ls="--", lw=1.0, label="truth")
    _mark_burn_in(ax, burn)
    ax.set_xlabel("step")
    if label:
        ax.set_ylabel(label)
    return ax


def plot_detection_by_band(summary, *, ax=None, bands=None, **kwargs):
    """Detection rate per coupling band.

    Bands with no reference spin come back nan from the summary and are drawn
    as a gap rather than as a zero bar, so missing data does not read as a
    failed recovery.
    """
    ax = _axes(ax)
    bands = BANDS if bands is None else bands
    heights = np.asarray(summary.by_band(bands), dtype=float)
    positions = np.arange(len(bands))
    kwargs.setdefault("color", "steelblue")
    ax.bar(positions[~np.isnan(heights)], heights[~np.isnan(heights)], **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in bands])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("coupling magnitude (kHz)")
    ax.set_ylabel("detection rate $R$")
    return ax
