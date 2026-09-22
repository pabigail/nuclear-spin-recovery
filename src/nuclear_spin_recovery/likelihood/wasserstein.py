"""Optimal-transport penalty on top of the Gaussian likelihood.

Spec Sec. 7.1 offers an optional variant that adds a distributional term to the
pointwise residual, so that a prediction whose features sit in almost the right
place is preferred over one whose features are absent.  The L2 residual cannot
express that: a modulation dip shifted by one sampling interval is penalised as
heavily as one that is missing.

Two things have to be pinned down before a Wasserstein distance between
*signals* means anything, and neither is in the published form.

**What the distributions are.**  A coherence signal is not a probability
measure.  It becomes one here by reading the dip depth ``1 - f`` as a
non-negative weight over tau, so the mass sits where the bath modulates the
signal and the transport cost is "how far along tau would the features have to
move".  Reading the *values* as samples instead -- the other obvious choice --
gives a quantity blind to tau, which scores a correctly-shaped signal and a
scrambled one alike.

**How it is normalised.**  Weights are normalised to unit mass, so the distance
is invariant to rescaling either signal, and the result is divided by the tau
span, so it is dimensionless rather than carrying milliseconds.  Without the
second step the penalty's size would depend on whether tau was recorded in ms
or us.

See docs/phase-4-plan.md, unit 4d.
"""

from __future__ import annotations

from .base import Likelihood


def signal_measure(signal, floor=0.0):
    """Non-negative weight over tau from a coherence signal. (n_points,)

    The dip depth ``1 - signal``, clipped at ``floor``.  Noisy data can exceed
    one, which would make the weight negative and the transport problem
    meaningless; clipping is the least-surprising repair and the clipped mass
    is a rounding error next to the modulation.
    """
    raise NotImplementedError


def wasserstein_signal_distance(a, b, tau, floor=0.0):
    """Normalised 1-Wasserstein distance between two signals' tau-measures.

    Dimensionless and in [0, 1]: the transport cost divided by the span of
    ``tau``, so a feature displaced by the whole window scores 1 and the value
    does not change if tau is re-expressed in different units.

    Returns 0 when both measures are empty -- two signals pinned at full
    coherence carry no features to transport, which is agreement, not an error.
    """
    raise NotImplementedError


class WassersteinL2(Likelihood):
    """Gaussian L2 with an optional transport penalty.

        log L = -1/2 sum_j ((d_j - f_j) / sigma)^2  -  zeta * scale * W

    At ``zeta = 0`` this is :class:`~nuclear_spin_recovery.likelihood.gaussian.
    GaussianL2` exactly -- not approximately, and the short-circuit matters:
    ``0 * nan`` is ``nan``, so a degenerate signal must not be able to poison a
    run that asked for no penalty.

    **This is not the product form printed in spec Sec. 7.1.**  That form,
    ``(1 - zeta) exp(E) - zeta W``, is negative whenever ``zeta W`` exceeds
    ``(1 - zeta) exp(E)``, and since E is large and negative for any
    configuration that does not already fit -- measured at -3000 on ordinary
    draws -- its logarithm is undefined for almost every state the sampler
    visits.  The penalty is therefore applied additively in log space, which
    agrees with the product form wherever that form is defined at all, composes
    with tempering, and is finite everywhere.

    ``scale`` makes the penalty commensurate with the residual term, which
    grows with the number of points; the default scales with the data.
    """

    def __init__(self, zeta=0.0, scale=None, floor=0.0):
        raise NotImplementedError

    def log_prob(self, state, expset, model, site_table):
        """Log-likelihood per replica. (n_replicas,)"""
        raise NotImplementedError
