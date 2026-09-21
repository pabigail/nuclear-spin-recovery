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

#: Coupling-magnitude bands, kHz, from docs/test-plan.md Sec. 5.1.  Detection is
#: asserted only in the last of these; see Sec. 5.3 for why the others are
#: recorded but not asserted.
BANDS = ((5.0, 25.0), (25.0, 100.0), (100.0, 750.0))

#: kHz.  Absorbs numerical differences between symmetry-related sites.
MATCH_TOL = 0.1


def couplings(site_table, sites, dA_par=None, dA_perp=None):
    """The (A_par, A_perp) pairs of ``sites``, offsets included if given.

    Offsets must be passed when they were sampled.  Rebuilding from site
    indices alone pins them at zero, which scores a relaxed run as though its
    constraint had never been relaxed.
    """
    raise NotImplementedError


def matches(pair, candidates, tol=MATCH_TOL):
    """Whether ``pair`` is within ``tol`` of any coupling in ``candidates``."""
    raise NotImplementedError


def detection_rate(samples, reference, tol=MATCH_TOL):
    """Fraction of posterior samples containing each reference spin. (n_ref,)

    ``samples`` is a sequence of coupling collections, one per posterior draw;
    ``reference`` is the coupling list of the spins being looked for.  This is
    R_i of spec Sec. 9.2, and it is defined over the posterior -- never over a
    single configuration.
    """
    raise NotImplementedError


def false_absence(samples, modal):
    """FP of spec Sec. 9.2: mean absence of the modal spins across samples.

    Not a false-positive rate in the classification sense, and not to be
    reported as one.
    """
    raise NotImplementedError


def band_index(magnitude, bands=BANDS):
    """Index of the band containing ``magnitude``, or -1 if outside them all."""
    raise NotImplementedError


def by_band(values, magnitude, bands=BANDS):
    """Mean of ``values`` within each band. (len(bands),)

    Bands with no members give nan rather than zero: an empty band is missing
    data, and averaging it as zero would understate detection.
    """
    raise NotImplementedError
