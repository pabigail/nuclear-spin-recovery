"""Relaxed couplings: what the sampler moved the DFT values to.

Spec Sec. 5.3.  Detection (:mod:`.detection`) asks whether the spin at a site
was found, and deliberately matches on the *table* couplings so that a relaxed
value drifting past the match tolerance is not counted as a miss.  That leaves
the other half of a relaxed run unreported: where each coupling ended up, and
whether it is closer to an independent measurement than the DFT value it
started from.  This module is that half.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class RelaxedCouplings:
    """Per-site posterior of the relaxed couplings.

    Every array has one entry per site of the table.  Means and spreads are
    over the draws in which the site was occupied, and are ``nan`` for a site
    that never was.
    """

    occupancy: np.ndarray      # (n_sites,)  share of draws with a spin here
    a_par: np.ndarray          # (n_sites,)  mean relaxed A_parallel, kHz
    a_perp: np.ndarray         # (n_sites,)  mean relaxed A_perp, kHz
    a_par_std: np.ndarray      # (n_sites,)
    a_perp_std: np.ndarray     # (n_sites,)
    d_par: np.ndarray          # (n_sites,)  mean offset from the table, kHz
    d_perp: np.ndarray         # (n_sites,)

    def occupied(self, min_occupancy=0.5):
        """Sites occupied in at least ``min_occupancy`` of the draws."""
        return np.flatnonzero(self.occupancy >= float(min_occupancy))


@dataclass
class ModalConfiguration:
    """The most often visited set of sites, and its best draw."""

    sites: tuple               # occupied sites, ascending
    share: float               # fraction of draws on exactly this set
    step: int                  # trace index of the highest-likelihood such draw
    d_par: np.ndarray          # (len(sites),) offsets at that draw, kHz
    d_perp: np.ndarray         # (len(sites),)


def _live(trace, burn):
    """Site index, offsets and liveness of every slot after burn-in."""
    burn = int(burn)
    if burn >= len(trace):
        raise ValueError(
            f"discarding {burn} of {len(trace)} steps would leave nothing")
    site = np.asarray(trace.site_idx)[burn:]
    k = np.asarray(trace.k)[burn:]
    live = np.arange(site.shape[1])[None, :] < k[:, None]
    return (site, np.asarray(trace.dA_par)[burn:],
            np.asarray(trace.dA_perp)[burn:], live)


def relaxed_couplings(trace, site_table, burn=0):
    """Where each site's couplings sat, over the draws it was occupied in.

    Pass a pooled ensemble trace where there is one: a single chain reports
    the spread of the mode it is in, not of the posterior.
    """
    site, d_par, d_perp, live = _live(trace, burn)
    n_sites = len(site_table)
    where = site[live]

    def moments(values):
        total = np.zeros(n_sites)
        square = np.zeros(n_sites)
        np.add.at(total, where, values[live])
        np.add.at(square, where, values[live] ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
            std = np.sqrt(np.maximum(square / count - mean**2, 0.0))
        return mean, std

    count = np.bincount(where, minlength=n_sites).astype(float)
    par_mean, par_std = moments(d_par)
    perp_mean, perp_std = moments(d_perp)
    return RelaxedCouplings(
        occupancy=count / site.shape[0],
        a_par=np.asarray(site_table.a_par, dtype=float) + par_mean,
        a_perp=np.asarray(site_table.a_perp, dtype=float) + perp_mean,
        a_par_std=par_std, a_perp_std=perp_std,
        d_par=par_mean, d_perp=perp_mean,
    )


def modal_configuration(trace, burn=0):
    """The most visited set of sites after burn-in, and its best draw.

    With the number of spins free, the highest-likelihood draw of a whole run
    is biased towards extra spins: one more parameter never makes the best fit
    worse.  The set of sites the chain spends most of its time on is not, so
    the configuration reported here is that set, with the offsets of the
    highest-likelihood draw among the steps spent on it.
    """
    site, d_par, d_perp, live = _live(trace, burn)
    log_prob = np.asarray(trace.log_prob)[int(burn):]
    sets = [tuple(sorted(int(s) for s in row[mask]))
            for row, mask in zip(site, live, strict=True)]
    counts = Counter(sets)
    modal, n = counts.most_common(1)[0]
    on_modal = np.array([s == modal for s in sets])
    best = int(np.flatnonzero(on_modal)[np.argmax(log_prob[on_modal])])
    order = np.argsort(site[best][live[best]])
    return ModalConfiguration(
        sites=modal, share=n / len(sets), step=best + int(burn),
        d_par=d_par[best][live[best]][order],
        d_perp=d_perp[best][live[best]][order],
    )


def compare_couplings(relaxed, site_table, measured, *, sites=None,
                      min_occupancy=0.5):
    """DFT, relaxed and measured couplings side by side.

    ``measured`` is an (m, 2) array of independently measured
    ``(A_parallel, A_perp)`` pairs in kHz.  Each is paired with one recovered
    site, and the distance from the measurement is reported for the table
    value and for the relaxed value, so that relaxation can be seen to have
    helped, or not.

    The pairing is made on the **table** couplings, not the relaxed ones:
    pairing on the relaxed values would choose, for every measurement, the
    site relaxation had moved closest to it, and would report an improvement
    whether or not there was one.  Pass ``sites`` to give the pairing
    yourself, one site per measurement, when it is known.

    Returns a dict of arrays, one row per measurement.  ``site`` is -1, and
    the other columns ``nan``, for a measurement left unpaired because there
    were fewer recovered sites than measurements.
    """
    measured = np.atleast_2d(np.asarray(measured, dtype=float))
    if measured.shape[1] != 2:
        raise ValueError(
            f"measured must be (m, 2) pairs of (A_par, A_perp), "
            f"got shape {measured.shape}")
    table = np.column_stack([np.asarray(site_table.a_par, dtype=float),
                             np.asarray(site_table.a_perp, dtype=float)])
    m = measured.shape[0]
    paired = np.full(m, -1, dtype=int)
    if sites is not None:
        sites = np.asarray(sites, dtype=int)
        if sites.shape != (m,):
            raise ValueError(
                f"sites must name one site per measurement, got "
                f"{sites.shape[0] if sites.ndim else 0} for {m}")
        paired[:] = sites
    else:
        candidates = relaxed.occupied(min_occupancy)
        if candidates.size:
            cost = np.linalg.norm(
                measured[:, None, :] - table[candidates][None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            paired[rows] = candidates[cols]

    found = paired >= 0
    safe = np.where(found, paired, 0)
    dft = np.where(found[:, None], table[safe], np.nan)
    rel = np.where(found[:, None],
                   np.column_stack([relaxed.a_par[safe], relaxed.a_perp[safe]]),
                   np.nan)
    return {
        "site": paired,
        "measured": measured,
        "dft": dft,
        "relaxed": rel,
        "dft_error": np.linalg.norm(dft - measured, axis=1),
        "relaxed_error": np.linalg.norm(rel - measured, axis=1),
    }
