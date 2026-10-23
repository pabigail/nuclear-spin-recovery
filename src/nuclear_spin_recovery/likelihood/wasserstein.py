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


def level_measures(a, b):
    """The pre-rewrite reading of two signals as weights. Two (n_points,) arrays.

    Each signal's height above the *joint* minimum of the pair, as in
    ``WassersteinError`` of the original research code (``error.py``).  The
    mass sits where coherence is high, so most of it is unmodulated background
    and a translated dip moves very little of it; it responds to amplitude and
    envelope more than to where the features are.  Kept for comparison with
    that code, not as a recommendation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    low = min(a.min(), b.min())
    return a - low, b - low


#: Ways to read a pair of signals as transportable weights.
MEASURES = ("dip", "level")


def wasserstein_signal_distance(a, b, tau, floor=0.0, measure="dip"):
    """Normalised 1-Wasserstein distance between two signals' tau-measures.

    Dimensionless and in [0, 1]: the transport cost divided by the span of
    ``tau``, so a feature displaced by the whole window scores 1 and the value
    does not change if tau is re-expressed in different units.

    Returns 0 when both measures are empty -- two signals pinned at full
    coherence carry no features to transport, which is agreement, not an error
    -- and 1 when exactly one is empty, since no transport plan turns a
    featureless signal into a modulated one.

    ``measure`` is ``"dip"`` (the dip depth, :func:`signal_measure`) or
    ``"level"`` (the pre-rewrite reading, :func:`level_measures`; ``floor`` is
    ignored).  The pre-rewrite code also placed the mass at the sample index
    rather than at tau; on a uniform grid that only multiplies the distance by
    ``n_points - 1``, which a scale absorbs.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
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

    if measure == "dip":
        weight_a, weight_b = signal_measure(a, floor), signal_measure(b, floor)
    else:
        weight_a, weight_b = level_measures(a, b)
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

        log L = -1/2 sum_j ((d_j - f_j) / sigma)^2  -  weight * W

    At ``weight = 0`` this is :class:`~nuclear_spin_recovery.likelihood.
    gaussian.GaussianL2` exactly -- not approximately, and the short-circuit
    matters: ``0 * nan`` is ``nan``, so a degenerate signal must not be able to
    poison a run that asked for no penalty.

    **One parameter, not two.**  Spec Sec. 7.1 writes the penalty with a
    weighting factor zeta and a scale s, but only their product ever enters the
    log-likelihood -- ``zeta=0.2, scale=5000`` and ``zeta=1.0, scale=1000``
    returned bitwise identical values -- so the two were perfectly degenerate
    and are collapsed here into ``weight``.  Its units are log-likelihood per
    unit of normalised transport, and it spans decades: the calibration in
    docs/test-plan.md Sec. 5.7 sweeps it from 0 to 1e6.

    **This is not the product form printed in spec Sec. 7.1.**  That form,
    ``(1 - zeta) exp(E) - zeta W``, is negative whenever the penalty exceeds
    ``(1 - zeta) exp(E)``, and since E is large and negative for any
    configuration that does not already fit -- measured at -3000 on ordinary
    draws -- its logarithm is undefined for almost every state the sampler
    visits.  The penalty is therefore applied additively in log space, which
    agrees with the product form wherever that form is defined at all, composes
    with tempering, and is finite everywhere.  The specification records the
    same decision.

    ``weight`` is a **calibrated parameter, and its calibrated value is zero**.
    On well-aligned data no weight improves either criterion: below about 10 the
    term cannot flip a single accept/reject decision, and above about 1e3 the
    recovery degrades.  It earns its place only on data carrying a diagnosed
    timing offset, where it is the better criterion over a window of offsets.
    See docs/test-plan.md Sec. 5.7 for both measurements.
    """

    def __init__(self, weight=0.0, floor=0.0):
        self.weight = float(weight)
        if self.weight < 0.0:
            raise ValueError(
                f"weight must be non-negative, got {weight}; it multiplies a "
                f"distance, so a negative value would reward disagreement"
            )
        self.floor = float(floor)

    def log_prob(self, state, expset, model, site_table):
        """Log-likelihood per replica. (n_replicas,)"""
        predicted = model.coherence(state, expset, site_table)
        residual = expset.data_all[None, :] - predicted
        sigma = state.sigma[:, expset.exp_id]
        gaussian = -0.5 * np.sum((residual / sigma) ** 2, axis=1)
        if self.weight == 0.0:
            # Short-circuit rather than multiply by zero: the penalty can be
            # nan for a degenerate signal, and 0 * nan is nan.
            return gaussian
        return gaussian - self.weight * self._transport(predicted, expset)

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


class MixedWassersteinL2(Likelihood):
    """Gaussian L2 and transport, mixed by a fraction ``zeta`` in [0, 1].

    Per experiment e, with S_e = sum_j (d_j - f_j)**2 and W_e the normalised
    transport distance of :func:`wasserstein_signal_distance`:

        form="penalty":  log L = -sum_e [ S_e + zeta/(1-zeta) s_e W_e ] / 2 sigma_e**2
        form="mixture":  log L = -sum_e [ (1-zeta) S_e + zeta s_e W_e ] / 2 sigma_e**2

    **What makes zeta a fraction.**  ``s_e`` converts transport into units of
    squared residual, so that ``zeta = 0.5`` weighs the two terms equally.  With
    ``scale="flat"`` it is fixed by the data before sampling: the featureless
    prediction f = 1 is charged the same by both terms,

        s_e = S_e(f = 1) / W_e(f = 1),

    which for the dip measure is just S_e(f = 1), since a flat signal carries no
    dip and so sits at the maximal distance 1.  A number (or one per
    experiment) can be given instead.

    **The two forms differ only in temperature.**  Both put the two terms in the
    ratio (1 - zeta) : zeta.  ``"penalty"`` keeps the Gaussian term whole and
    adds the transport on top, so sigma keeps its meaning and zeta = 1 is not
    allowed; it is :class:`WassersteinL2` with
    ``weight = zeta/(1-zeta) s_e / 2 sigma_e**2``.  ``"mixture"`` scales the
    Gaussian by (1 - zeta), which at zeta = 0.5 is the same as doubling sigma**2
    -- a broader posterior whatever the transport does -- and allows the pure
    transport limit zeta = 1.  It is the form of ``CompositeError`` in the
    pre-rewrite code, which mixed the two errors linearly inside the exponent.

    zeta = 0 is :class:`~nuclear_spin_recovery.likelihood.gaussian.GaussianL2`
    exactly, in both forms, for the same reason as in :class:`WassersteinL2`.

    The transport normalises mass away, so it cannot tell a shallow dip from a
    deep one; near zeta = 1 the number of spins is poorly identified.
    """

    FORMS = ("penalty", "mixture")

    def __init__(self, zeta=0.0, scale="flat", form="penalty", measure="dip",
                 floor=0.0):
        self.zeta = float(zeta)
        if not 0.0 <= self.zeta <= 1.0:
            raise ValueError(f"zeta is a fraction in [0, 1], got {zeta}")
        if form not in self.FORMS:
            raise ValueError(f"form must be one of {self.FORMS}, got {form!r}")
        if form == "penalty" and self.zeta == 1.0:
            raise ValueError(
                "zeta = 1 leaves no Gaussian term to keep whole; use "
                "form='mixture' for the pure transport limit"
            )
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
        if isinstance(scale, str):
            if scale != "flat":
                raise ValueError(f"scale must be 'flat' or a number, got {scale!r}")
        else:
            scale = np.asarray(scale, dtype=float)
            if scale.ndim > 1 or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError("a numeric scale must be positive and finite, "
                                 "one number or one per experiment")
        self.scale = scale
        self.form = form
        self.measure = measure
        self.floor = float(floor)
        self._cached = (None, None)

    def scales(self, expset):
        """s_e, the transport-to-squared-residual conversion. (n_experiments,)

        Computed once per ExperimentSet and cached; build a new set rather than
        editing a set's data in place.
        """
        if not isinstance(self.scale, str):
            return np.broadcast_to(self.scale, (expset.n_experiments,)).astype(float)
        cached_set, cached = self._cached
        if cached_set is expset:
            return cached
        out = np.empty(expset.n_experiments)
        weights = _point_weights(expset)
        for e, (seen, experiment) in enumerate(
                zip(expset.split(expset.data_all), expset.experiments, strict=True)):
            flat = np.ones_like(seen)
            w = 1.0 if weights is None else expset.split(weights)[e]
            squared = float(np.sum(w * (seen - flat) ** 2))
            moved = wasserstein_signal_distance(flat, seen, experiment.tau,
                                                self.floor, self.measure)
            if not (squared > 0.0 and moved > 0.0):
                raise ValueError(
                    f"experiment {e} cannot set a 'flat' scale (squared residual "
                    f"{squared}, transport {moved}); pass a numeric scale"
                )
            out[e] = squared / moved
        self._cached = (expset, out)
        return out

    def log_prob(self, state, expset, model, site_table):
        """Log-likelihood per replica. (n_replicas,)"""
        predicted = model.coherence(state, expset, site_table)
        residual = expset.data_all[None, :] - predicted
        sigma = state.sigma[:, expset.exp_id]
        squared = (residual / sigma) ** 2
        weights = _point_weights(expset)
        if weights is not None:
            squared = weights[None, :] * squared
        gaussian = -0.5 * np.sum(squared, axis=1)
        if self.zeta == 0.0:
            # Short-circuit, as in WassersteinL2: 0 * nan is nan.
            return gaussian
        transport = self._transport(predicted, expset)            # (n_rep, n_exp)
        charge = -0.5 * np.sum(self.scales(expset)[None, :] * transport
                               / state.sigma ** 2, axis=1)
        if self.form == "penalty":
            return gaussian + self.zeta / (1.0 - self.zeta) * charge
        return (1.0 - self.zeta) * gaussian + self.zeta * charge

    def _transport(self, predicted, expset):
        """Normalised transport per replica and experiment. (n_replicas, n_exp)"""
        observed = expset.split(expset.data_all)
        out = np.zeros((predicted.shape[0], expset.n_experiments))
        for r in range(predicted.shape[0]):
            for e, (piece, seen, experiment) in enumerate(zip(
                    expset.split(predicted[r]), observed, expset.experiments,
                    strict=True)):
                out[r, e] = wasserstein_signal_distance(
                    piece, seen, experiment.tau, self.floor, self.measure)
        return out


def _point_weights(expset):
    """Per-point measurement weights where the ExperimentSet carries them.

    Later versions of ExperimentSet add ``weight_all`` (relative measurement
    time per point); on those the Gaussian term is weighted exactly as in
    GaussianL2, and here, without it, every point counts once.
    """
    weights = getattr(expset, "weight_all", None)
    return None if weights is None else np.asarray(weights, dtype=float)
