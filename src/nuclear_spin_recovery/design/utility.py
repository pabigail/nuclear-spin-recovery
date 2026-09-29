"""How informative a candidate experiment is, given the posterior's particles.

Two answers to one question, behind one interface so that either can rank
candidates or choose points and a third can be added without touching the
designer.  Both take arrays only -- predictions, weights, noise -- and never a
model, table or experiment, which is what makes them testable against cases
with a known answer.

**Noise** is broadcastable to ``(n_points,)`` and is the *effective* standard
deviation at each point, ``sigma_e / sqrt(w_j)`` when the point carries a
measurement weight.  An infinite value is a point not measured -- the weight-
zero case -- and contributes nothing, exactly as it contributes nothing to the
likelihood.  Zero, negative or nan noise raises.

**ExpectedInformationGain** is the mutual information between the bath and the
data, in nats, by the Monte-Carlo estimator of ``adaptive_exp.py``: draw a
"true" particle by weight, simulate data from it, and average
``log p(d | k) - log p(d)``.  The evidence sums exactly over the particle set,
so the estimate is of the information about *which particle* -- a discrete
stand-in for the continuous posterior, and biased upward at small effective
size for that reason (docs/phase-5-plan.md Sec. 6, question 3).

Per draw the term is at most ``-log w_k``.  For uniform weights that is
``log K`` for every draw, so the estimate never exceeds the prior entropy;
for skewed weights the sampled ``k`` frequencies fluctuate and a single
estimate at 64 draws was measured at 1.04 nats against an entropy of 0.80.
The bound holds in expectation, and the estimate converges on it.

**Common random numbers.**  :meth:`DesignUtility.score_many` scores several
candidates with the same sampled particles and the same standard-normal noise
draws, truncated to each candidate's length.  Measured on two candidates
0.012 nats apart at 64 draws: shared draws ranked them correctly on 40 of 40
seeds, independent draws on 18.  Ranking is all the designer uses the number
for, so this is the property that matters.

See docs/phase-5-plan.md, unit 5b.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DesignUtility(ABC):
    """A scalar measure of how informative one candidate is."""

    @abstractmethod
    def score_many(self, predictions, weights, noise, rng):
        """Score several candidates in one call. (n_candidates,)

        ``predictions`` is a sequence of ``(K, n_points_c)`` arrays, one per
        candidate, each row a particle; ``noise`` is a sequence of the same
        length, each entry broadcastable to that candidate's points.  Point
        counts may differ between candidates.
        """

    def score(self, predictions, weights, noise, rng):
        """Score one candidate. ``predictions`` is ``(K, n_points)``."""
        raise NotImplementedError


def information_density(predictions, weights, noise):
    """Where the particles disagree, per point, in units of the noise. (n_points,)

    The weighted variance of the predictions at each point divided by the
    noise variance there.  Zero where every particle predicts the same, which
    is where measuring cannot separate them; zero at an unmeasured point.
    """
    raise NotImplementedError


class ExpectedInformationGain(DesignUtility):
    """Monte-Carlo mutual information between bath and data, in nats.

    ``n_draws`` simulated datasets per candidate.  ``common_random`` shares
    them across the candidates of one :meth:`score_many` call; switch it off
    only to measure what it buys.
    """

    def __init__(self, n_draws=64, common_random=True):
        raise NotImplementedError

    def score_many(self, predictions, weights, noise, rng):
        raise NotImplementedError


class PredictiveVariance(DesignUtility):
    """Total information density: the sum over points of
    :func:`information_density`.

    Deterministic -- ``rng`` is accepted for the interface and ignored -- and
    cheap, so it is the natural choice for scoring single points in a greedy
    selector.  It rewards disagreement without asking whether that
    disagreement is resolvable, which EIG does; the two can rank differently,
    and which is better for which job is what T9 is for.
    """

    def score_many(self, predictions, weights, noise, rng):
        raise NotImplementedError
