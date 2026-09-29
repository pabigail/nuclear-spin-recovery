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

import numpy as np
from scipy.special import logsumexp

#: Cap on the (draws, particles, points) residual block held at once.  Keeps
#: memory flat for large K without changing a single number.
_BLOCK_ELEMENTS = 4_000_000


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
        return float(self.score_many([predictions], weights, [noise], rng)[0])


def _weights(weights, n_particles):
    """Validated, normalised weights. (K,)"""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n_particles:
        raise ValueError(f"{n_particles} particles but {w.size} weights")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    total = w.sum()
    if total <= 0.0:
        raise ValueError("weights sum to zero")
    return w / total


def _noise(noise, n_points):
    """Per-point effective noise. (n_points,)  Infinite is unmeasured."""
    s = np.broadcast_to(np.asarray(noise, dtype=float), (n_points,))
    if np.any(np.isnan(s)) or np.any(s <= 0):
        raise ValueError("noise must be positive; use inf for an unmeasured point")
    return s


def _candidates(predictions, noise):
    """Pair each candidate's (K, n) predictions with its noise."""
    predictions = [np.atleast_2d(np.asarray(p, dtype=float)) for p in predictions]
    noise = list(noise)
    if len(noise) != len(predictions):
        raise ValueError(
            f"{len(predictions)} candidates but {len(noise)} noise entries")
    n_particles = {p.shape[0] for p in predictions}
    if len(n_particles) > 1:
        raise ValueError("candidates disagree on the number of particles")
    return [(p, _noise(n, p.shape[1])) for p, n in zip(predictions, noise,
                                                      strict=True)]


def information_density(predictions, weights, noise):
    """Where the particles disagree, per point, in units of the noise. (n_points,)

    The weighted variance of the predictions at each point divided by the
    noise variance there.  Zero where every particle predicts the same, which
    is where measuring cannot separate them; zero at an unmeasured point.
    """
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    w = _weights(weights, P.shape[0])
    s = _noise(noise, P.shape[1])
    mean = w @ P
    variance = w @ (P - mean) ** 2
    return variance / s ** 2


class ExpectedInformationGain(DesignUtility):
    """Monte-Carlo mutual information between bath and data, in nats.

    ``n_draws`` simulated datasets per candidate.  ``common_random`` shares
    them across the candidates of one :meth:`score_many` call; switch it off
    only to measure what it buys.
    """

    def __init__(self, n_draws=64, common_random=True):
        if int(n_draws) < 1:
            raise ValueError(f"n_draws must be at least 1, got {n_draws}")
        self.n_draws = int(n_draws)
        self.common_random = bool(common_random)

    def score_many(self, predictions, weights, noise, rng):
        cands = _candidates(predictions, noise)
        if not cands:
            return np.empty(0, dtype=float)
        n_particles = cands[0][0].shape[0]
        w = _weights(weights, n_particles)
        n_max = max(P.shape[1] for P, _ in cands)

        def draw():
            truth = rng.choice(n_particles, size=self.n_draws, p=w)
            return truth, rng.standard_normal((self.n_draws, n_max))

        truth, eps = draw()
        out = np.empty(len(cands), dtype=float)
        for c, (P, s) in enumerate(cands):
            if c > 0 and not self.common_random:
                truth, eps = draw()
            # Columns are masked rather than dropped from eps, so a point with
            # infinite noise leaves every other point's draw where it was.
            measured = np.isfinite(s)
            out[c] = self._gain(P[:, measured], w, s[measured],
                                truth, eps[:, : P.shape[1]][:, measured])
        return out

    @staticmethod
    def _gain(P, w, s, truth, eps):
        """Mean of log p(d | k) - log p(d) over the simulated datasets."""
        n_draws = truth.size
        if P.shape[0] == 1 or P.shape[1] == 0:
            return 0.0
        Q = P / s                                   # (K, n), in noise units
        data = Q[truth] + eps                       # (M, n)
        with np.errstate(divide="ignore"):
            log_w = np.log(w)                       # -inf for a zero weight
        total = 0.0
        block = max(1, _BLOCK_ELEMENTS // max(1, Q.size))
        for lo in range(0, n_draws, block):
            d = data[lo:lo + block]
            ll = -0.5 * np.sum((d[:, None, :] - Q[None, :, :]) ** 2, axis=2)
            log_evidence = logsumexp(log_w[None, :] + ll, axis=1)
            own = ll[np.arange(d.shape[0]), truth[lo:lo + block]]
            total += np.sum(own - log_evidence)
        return float(total / n_draws)


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
        cands = _candidates(predictions, noise)
        return np.array([information_density(P, weights, s).sum()
                         for P, s in cands], dtype=float)
