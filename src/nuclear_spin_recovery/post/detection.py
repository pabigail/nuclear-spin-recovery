"""Detection rate, coupling matching, and coupling-magnitude bands.

Posterior summaries are defined on **couplings**, never on site index: the NV
centre's symmetry puts lattice sites into orbits that are spatially distinct but
hyperfine-identical, and no coherence measurement can tell them apart.  Scoring
by index reports a physically correct answer as a miss.

Matching carries a tolerance because symmetry-related sites differ in the table
at the fourth decimal, from the DFT calculation rather than from physics.

See docs/model-specification.md Sec. 9.2 and docs/test-plan.md Sec. 5.1.
"""

from __future__ import annotations

import numpy as np

#: Coupling-magnitude bands, kHz, from docs/test-plan.md Sec. 5.1.  Detection is
#: asserted only in the last of these; see Sec. 5.3 for why the others are
#: recorded but not asserted.
BANDS = ((5.0, 25.0), (25.0, 100.0), (100.0, 750.0))

#: kHz.  Absorbs numerical differences between symmetry-related sites.
MATCH_TOL = 0.1


#: Decimal places couplings are rounded to before set membership.  Absorbs
#: float noise in the table without merging genuinely distinct sites.
_ROUND = 4


def couplings(site_table, sites, dA_par=None, dA_perp=None):
    """The (A_par, A_perp) pairs of ``sites``, offsets included if given.

    Offsets must be passed when they were sampled.  Rebuilding from site
    indices alone pins them at zero, which scores a relaxed run as though its
    constraint had never been relaxed.

    Returns a set of rounded pairs.  Rounding does **not** merge a symmetry
    orbit: on the NV table an orbit's members differ in the fourth decimal
    (519.7523 against 519.8493 kHz, say), so they survive as distinct entries.
    Merging happens at match time, through the ``tol`` of :func:`matches` --
    which is why every comparison in this module has to go through it rather
    than through set membership.
    """
    sites = np.asarray(list(sites), dtype=int)
    a_par = np.asarray(site_table.a_par, dtype=float)[sites]
    a_perp = np.asarray(site_table.a_perp, dtype=float)[sites]
    if dA_par is not None:
        a_par = a_par + np.asarray(dA_par, dtype=float)[: sites.size]
    if dA_perp is not None:
        a_perp = a_perp + np.asarray(dA_perp, dtype=float)[: sites.size]
    return set(zip(np.round(a_par, _ROUND), np.round(a_perp, _ROUND), strict=True))


def matches(pair, candidates, tol=MATCH_TOL):
    """Whether ``pair`` is within ``tol`` of any coupling in ``candidates``."""
    a, b = pair
    return any(abs(a - c) <= tol and abs(b - d) <= tol for c, d in candidates)


def detection_rate(samples, reference, tol=MATCH_TOL):
    """Fraction of posterior samples containing each reference spin. (n_ref,)

    ``samples`` is a sequence of coupling collections, one per posterior draw;
    ``reference`` is the coupling list of the spins being looked for.  This is
    R_i of spec Sec. 9.2, and it is defined over the posterior -- never over a
    single configuration.
    """
    reference = list(reference)
    if not reference:
        return np.empty(0, dtype=float)
    samples = list(samples)
    if not samples:
        return np.zeros(len(reference), dtype=float)
    return np.array(
        [np.mean([matches(ref, s, tol) for s in samples]) for ref in reference],
        dtype=float,
    )


def false_absence(samples, modal, tol=MATCH_TOL):
    """FP of spec Sec. 9.2: mean absence of the modal spins across samples.

    Not a false-positive rate in the classification sense, and not to be
    reported as one.

    Matched within ``tol``, exactly as :func:`detection_rate` is.  Exact set
    membership looks equivalent and is not: symmetry-equivalent sites carry
    table couplings that differ in the fourth decimal, so a sample sitting on a
    different member of the same orbit reads as an absence.  Measured on a run
    where every true spin had R_i = 1.0, k_mode was correct and the best
    residual was 0.92 sigma, exact membership reported FP = 0.224 where the
    true value is 0.0 -- all of it orbit rounding, none of it absence.
    """
    modal = list(modal)
    samples = list(samples)
    if not modal or not samples:
        return 0.0
    return float(np.mean(
        [[not matches(c, sample, tol) for sample in samples] for c in modal]))


def band_index(magnitude, bands=BANDS):
    """Index of the band containing ``magnitude``, or -1 if outside them all.

    Lower edges are inclusive, upper edges exclusive, so the bands tile without
    double-counting a value that sits exactly on a boundary.
    """
    for i, (lo, hi) in enumerate(bands):
        if lo <= magnitude < hi:
            return i
    return -1


def by_band(values, magnitude, bands=BANDS):
    """Mean of ``values`` within each band. (len(bands),)

    Bands with no members give nan rather than zero: an empty band is missing
    data, and averaging it as zero would understate detection.
    """
    values = np.asarray(values, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    out = np.full(len(bands), np.nan)
    for i, (lo, hi) in enumerate(bands):
        sel = (magnitude >= lo) & (magnitude < hi)
        if sel.any():
            out[i] = values[sel].mean()
    return out
