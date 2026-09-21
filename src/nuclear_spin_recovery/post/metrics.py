"""Posterior summaries over a recorded chain.

Every summary here is computed **over the posterior**.  There is deliberately no
entry point that accepts a single :class:`~nuclear_spin_recovery.state.State`:
judging a recovery by one configuration -- the last state of a chain, or the
modal sample -- measures the wrong object, and a spin absent from the modal
configuration may still appear in most samples.  See spec Sec. 9.1.

The posterior is trans-dimensional, so there is no stable correspondence between
"spin 3" in one sample and "spin 3" in another.  Summaries are therefore defined
on couplings rather than by per-parameter averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import State
from . import detection
from .detection import BANDS, MATCH_TOL, couplings, detection_rate, false_absence
from .residual import predictive_from_arrays, residual_distribution


@dataclass
class PosteriorSummary:
    """Everything spec Sec. 9.2 defines, computed over posterior samples."""

    R_i: np.ndarray            # detection rate per reference spin
    magnitude: np.ndarray      # coupling magnitude of each reference spin, kHz
    residual: np.ndarray       # RMS residual per draw, in units of the noise
    k_posterior: np.ndarray    # sampled values of k
    false_absence: float       # FP of spec Sec. 9.2
    predictive: np.ndarray     # (n_draws, n_points) posterior-predictive signals

    def R(self, lo, hi):
        """Mean detection rate within a coupling band.

        nan when no reference spin falls in the band: an empty band is missing
        data, not a detection rate of zero.
        """
        sel = (self.magnitude >= lo) & (self.magnitude < hi)
        return float(self.R_i[sel].mean()) if sel.any() else float("nan")

    def by_band(self, bands=BANDS):
        """Mean detection rate in each band. (len(bands),)"""
        return detection.by_band(self.R_i, self.magnitude, bands)

    @property
    def median_residual(self):
        """Typical fit quality. Use when comparing like against like."""
        return float(np.median(self.residual))

    @property
    def best_residual(self):
        """Lowest residual any sampled configuration achieves.

        The statistic for comparing a model against one nested inside it.  A
        model with extra sampled parameters has a *higher* median residual than
        one holding them at the prior mean, because a typical draw sits away
        from that mean -- so a median comparison penalises the richer model for
        exploring.  What it should be asked is whether it can reach a fit the
        constrained model cannot.  Only meaningful between runs with the same
        number of posterior samples.
        """
        return float(np.min(self.residual))

    @property
    def k_mode(self):
        """Modal sampled dimension."""
        return int(np.bincount(self.k_posterior).argmax())

    def dimension_discrepancy(self, k_true):
        """|mode(k) - k_true|; zero means the inferred dimension is correct."""
        return int(abs(self.k_mode - int(k_true)))


def summarize(trace, expset, site_table, model, *, reference=None, burn=0,
              stride=1, noise=1.0, tol=None):
    """Compute every spec Sec. 9.2 metric from a recorded chain.

    ``trace`` is a :class:`~nuclear_spin_recovery.trace.Trace`.  Passing a State
    raises ``TypeError`` from an explicit check, rather than failing later on a
    missing attribute: judging a recovery by one configuration is a modelling
    error, and it should read as a refusal rather than as a bug.

    ``reference`` is the ground-truth site index array when one exists, and None
    on experimental data, where detection is not available and ``R_i`` comes
    back empty -- empty, not zero, because the question cannot be asked.

    ``stride`` thins the draws used for the **residual** only.  Detection runs
    over every post-burn-in draw, because R_i is a frequency and thinning it
    would only add variance.

    Each trace array is read **once**.  Reading a Trace property inside a loop
    rebuilds it every access, which is O(n^2) in the chain length.

    Detection matches sample couplings against the *table* values, with offsets
    excluded, while the predictive signal uses the sampled offsets in full.
    That split is deliberate: R_i asks whether the spin at a given site was
    found, and letting a relaxed coupling drift past the match tolerance would
    conflate site identification with coupling refinement.  The offsets are
    still scored, through the residual.
    """
    if isinstance(trace, State):
        raise TypeError(
            "summarize() takes a Trace, not a State. Posterior summaries are "
            "defined over the posterior; a single configuration -- the last "
            "state of a chain, or the modal sample -- measures the wrong "
            "object (spec Sec. 9.1)."
        )
    if not hasattr(trace, "log_prob"):
        raise TypeError(f"summarize() takes a Trace, got {type(trace).__name__}")

    tol = MATCH_TOL if tol is None else tol
    burn = int(burn)
    n_steps = len(trace)
    if burn >= n_steps:
        raise ValueError(
            f"discarding {burn} of {n_steps} steps would leave nothing"
        )

    # One read per array, then slice.  Reading inside the loops below is the
    # O(n^2) trap that cost 38 s of a 45 s suite.
    site_idx = np.array(trace.site_idx)[burn:]
    k = np.array(trace.k)[burn:]
    dA_par = np.array(trace.dA_par)[burn:]
    dA_perp = np.array(trace.dA_perp)[burn:]
    lam = np.array(trace.lam)[burn:]
    n_stretch = np.array(trace.n_stretch)[burn:]
    sigma = np.array(trace.sigma)[burn:]
    k_max = trace.k_max

    samples = [couplings(site_table, site_idx[j, : int(k[j])])
               for j in range(len(k))]

    if reference is None:
        ref_pairs, magnitude = [], np.array([])
    else:
        ref = np.asarray(list(reference), dtype=int)
        ref_pairs = list(zip(np.asarray(site_table.a_par, dtype=float)[ref],
                             np.asarray(site_table.a_perp, dtype=float)[ref],
                             strict=True))
        magnitude = (np.hypot(*np.array(ref_pairs).T) if ref_pairs
                     else np.array([]))

    step = max(1, int(stride))
    predictive = predictive_from_arrays(
        site_idx[::step], k[::step], dA_par[::step], dA_perp[::step],
        lam[::step], n_stretch[::step], sigma[::step],
        expset, site_table, model, k_max=k_max,
    )
    residual = residual_distribution(expset.data_all, predictive, noise)

    modal_k = int(np.bincount(k).argmax()) if len(k) else 0
    modal = set()
    if len(k):
        where = np.flatnonzero(k == modal_k)
        if where.size:
            modal = couplings(site_table, site_idx[int(where[0]), :modal_k])

    return PosteriorSummary(
        R_i=detection_rate(samples, ref_pairs, tol),
        magnitude=magnitude,
        residual=residual,
        k_posterior=np.asarray(k),
        false_absence=false_absence(samples, modal, tol),
        predictive=np.asarray(predictive),
    )
