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

import numpy as np
from scipy.stats import wasserstein_distance

from .base import Likelihood


def signal_measure(signal, floor=0.0):
    """Non-negative weight over tau from a coherence signal. (n_points,)

    The dip depth ``1 - signal``, clipped at ``floor``.  Noisy data can exceed
    one, which would make the weight negative and the transport problem
    meaningless; clipping is the least-surprising repair and the clipped mass
    is a rounding error next to the modulation.
    """
    return np.clip(1.0 - np.asarray(signal, dtype=float), floor, None)


def wasserstein_signal_distance(a, b, tau, floor=0.0):
    """Normalised 1-Wasserstein distance between two signals' tau-measures.

    Dimensionless and in [0, 1]: the transport cost divided by the span of
    ``tau``, so a feature displaced by the whole window scores 1 and the value
    does not change if tau is re-expressed in different units.

    Returns 0 when both measures are empty -- two signals pinned at full
    coherence carry no features to transport, which is agreement, not an error
    -- and 1 when exactly one is empty, since no transport plan turns a
    featureless signal into a modulated one.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"signals have shapes {a.shape} and {b.shape}")
    if a.shape != tau.shape:
        raise ValueError(f"signal shape {a.shape} does not match tau {tau.shape}")
    span = float(np.ptp(tau))
    if span <= 0.0:
        raise ValueError("tau spans zero, so the distance cannot be normalised")

    weight_a, weight_b = signal_measure(a, floor), signal_measure(b, floor)
    mass_a, mass_b = float(weight_a.sum()), float(weight_b.sum())
    if mass_a <= 0.0 and mass_b <= 0.0:
        return 0.0
    if mass_a <= 0.0 or mass_b <= 0.0:
        return 1.0
    # scipy normalises the weights to unit mass internally, which is what makes
    # this a distance between distributions rather than between amplitudes.
    return wasserstein_distance(tau, tau, weight_a, weight_b) / span


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
    with tempering, and is finite everywhere.  The specification records the
    same decision.

    ``scale`` converts a transport distance into the units of the residual
    term, and it is a **free parameter that has to be calibrated**, like zeta
    and like sigma_e before it.  Nothing in the physics fixes how many
    sigma-squared a full-window displacement is worth.

    The default, the number of data points, is a starting point and not a
    working value.  Measured on NV data at the settings of test-plan Sec. 5.1,
    with 250 points and zeta = 0.1: W runs from 0.0001 at the truth to 0.0153
    for a randomly drawn six-spin configuration, so the penalty spans 0.00 to
    0.38 while the residual term spans -1 to -3500.  That is about a tenth of a
    percent of the quantity it is meant to modify -- the term is present but
    cannot change an acceptance decision.  W is small because the envelope
    dominates the dip-depth distribution, leaving little mass to transport even
    between quite different configurations.

    A scale of order ``n_points / W_typical`` is where the term starts to
    matter.  Calibrate it the way sigma_e was calibrated: sweep, record the
    residual and the detection rate, and check what the metric reads with the
    mechanism disabled.
    """

    def __init__(self, zeta=0.0, scale=None, floor=0.0):
        self.zeta = float(zeta)
        if not 0.0 <= self.zeta <= 1.0:
            raise ValueError(f"zeta must lie in [0, 1], got {zeta}")
        #: None means "the number of data points", resolved per call.
        self.scale = scale
        self.floor = float(floor)

    def log_prob(self, state, expset, model, site_table):
        """Log-likelihood per replica. (n_replicas,)"""
        predicted = model.coherence(state, expset, site_table)
        residual = expset.data_all[None, :] - predicted
        sigma = state.sigma[:, expset.exp_id]
        gaussian = -0.5 * np.sum((residual / sigma) ** 2, axis=1)
        if self.zeta == 0.0:
            # Short-circuit rather than multiply by zero: the penalty can be
            # nan for a degenerate signal, and 0 * nan is nan.
            return gaussian
        scale = expset.n_points if self.scale is None else float(self.scale)
        return gaussian - self.zeta * scale * self._transport(predicted, expset)

    def _transport(self, predicted, expset):
        """Summed normalised transport cost per replica. (n_replicas,)

        Computed per experiment and summed.  Concatenating the grids first
        would let mass move between experiments, which is not a thing that can
        happen: each has its own tau axis, its own span, and its own pulse
        number.
        """
        observed = expset.split(expset.data_all)
        out = np.zeros(predicted.shape[0], dtype=float)
        for r in range(predicted.shape[0]):
            pieces = expset.split(predicted[r])
            out[r] = sum(
                wasserstein_signal_distance(piece, seen, experiment.tau,
                                            self.floor)
                for piece, seen, experiment
                in zip(pieces, observed, expset.experiments, strict=True)
            )
        return out
