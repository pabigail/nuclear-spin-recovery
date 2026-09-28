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

    Raises when the summary has no reference at all.  Detection is undefined
    without ground truth, and drawing empty axes would let a reader believe the
    question had been asked and answered negatively.  On measured data use
    :func:`plot_coupling_posterior` instead.
    """
    if np.asarray(summary.R_i).size == 0:
        raise ValueError(
            "this summary has no reference, so detection was never computed. "
            "Empty axes would read as 'asked and answered no'. On measured "
            "data use coupling_posterior() and plot_coupling_posterior()."
        )
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


def plot_posterior_predictive(summary, tau, observed, *, ax=None, band=95.0,
                              **kwargs):
    """Posterior-predictive signals against the measured data.

    Criterion A made visible, and the only one of the two that survives without
    ground truth.  Draws the central ``band`` percent of the predictive draws
    as a filled region with the data over it.
    """
    ax = _axes(ax)
    predictive = np.atleast_2d(np.asarray(summary.predictive, dtype=float))
    tau = np.asarray(tau, dtype=float)
    edge = (100.0 - float(band)) / 2.0
    lo, hi = np.percentile(predictive, [edge, 100.0 - edge], axis=0)
    ax.fill_between(tau, lo, hi, alpha=0.35, color="steelblue",
                    label=f"{band:g}% of posterior draws", **kwargs)
    ax.plot(tau, np.asarray(observed, dtype=float), lw=0.8, color="0.3",
            label="data")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("coherence")
    return ax


def plot_coupling_posterior(couplings, frequencies, *, ax=None, top=20,
                            **kwargs):
    """What the posterior contains, ranked by how often it contains it.

    The reference-free reading of a recovery: each bar is one coupling, its
    height the fraction of posterior samples containing it.

    A tall bar is a coupling the sampled posterior insists on -- which is the
    data speaking only if the chains mixed.  From a single chain a bar at 1.00
    can be a stuck walker, and looks identical to a certain one; see
    :func:`~nuclear_spin_recovery.post.detection.coupling_posterior` and
    test-plan Sec. 5.10.  Feed this pooled ensemble samples.
    """
    ax = _axes(ax)
    found = np.atleast_2d(np.asarray(couplings, dtype=float))
    frequencies = np.asarray(frequencies, dtype=float)
    shown = min(int(top), frequencies.size)
    positions = np.arange(shown)
    kwargs.setdefault("color", "steelblue")
    ax.bar(positions, frequencies[:shown], **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"({a:.0f}, {b:.0f})" for a, b in found[:shown]],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$(A_\parallel,\ A_\perp)$  (kHz)")
    ax.set_ylabel("fraction of posterior samples")
    return ax
