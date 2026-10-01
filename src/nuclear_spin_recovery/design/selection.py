"""Which points of a candidate to measure, and for how long.

A selector spends a **budget of measurement time** over a candidate's points
and returns both the points it kept and the time each received, as relative
weights summing to the budget.  One unit of weight is one point measured for
the baseline averaging time, so the effective noise at a kept point is
``sigma / sqrt(w_j)`` and the weights go straight into
:attr:`~nuclear_spin_recovery.experiment.Experiment.weight`.  Comparing designs
at equal budget rather than equal point count is what makes T9 fair: a design
that wins by measuring more is not a design (docs/phase-5-plan.md Sec. 6,
question 2).

**Points can cost different amounts of time.**  Every ``select`` takes an
optional ``cost``, the time of one unit of weight at each point -- for CPMG,
:class:`~nuclear_spin_recovery.design.cost.SequenceDuration`.  The budget is
then total time, ``sum_j w_j c_j``, and a weight still means relative
repetitions, so the likelihood is untouched.  Information per unit time at a
point is its density divided by its cost: an expensive point has to earn it.
With no cost every point costs 1 and the budget is the sum of the weights, so
every result below is exactly what it was before costs existed.

Selectors:

- :class:`InformationDensity` -- the rule from ``adaptive_exp.py``: time
  proportional to ``density ** power``, then points below ``prune_fraction``
  of the largest allocation dropped.  Both of its numbers were buried
  constants there and are parameters here.
- :class:`GreedyUtility` -- the direct comparison: add the point that most
  improves a :class:`~nuclear_spin_recovery.design.utility.DesignUtility`,
  repeat.  Every kept point receives an equal share of the time.
- :class:`UniformThinning` -- the control: evenly spaced in index, equal time.
- :class:`LeastInformative` -- the anti-design: the lowest-density points,
  equal time.  T9's negative control, without which "adaptive beats uniform"
  is consistent with "any extra measurement helps".

**Nothing to learn is an error, not a grid.**  If the particles agree at every
point -- one particle, or several that predict identically -- there is no
information to allocate by, and a selector that relies on it raises
:class:`NothingToLearn` rather than silently returning something uniform.  The
designer is what reports it.  :class:`UniformThinning` and
:class:`LeastInformative` never raise: they are T9's controls, and its
degenerate case needs them to keep working.

**A measured caveat on greedy EIG.**  The EIG estimate is Monte Carlo, and
adding an informative point can *lower* it even though the true value cannot
fall; an uninformative point leaves it exactly unchanged, and so wins that
comparison.  Measured on 30 scenes with 3 particles, the informative region
being half the grid: at 64 draws the first pick was informative on 30 of 30,
but a pick of four strayed on 4 of 30 -- always a late step, where the best
informative addition scored below the current set (0.747 against 0.831, for
one).  At 1024 draws, 0 of 30.  Use more draws for greedy selection than for
ranking candidates.

See docs/phase-5-plan.md, unit 5c.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utility import DesignUtility, _noise, _weights, information_density

#: A greedy first step scoring at or below this has found nothing.  EIG on
#: identical predictions is zero only to rounding -- the evidence carries
#: log(sum w), and normalised weights sum to 1 +- 1e-16 -- so exact zero is
#: not a usable test for it.  In nats, or in units of density.
_NOTHING = 1e-12


class NothingToLearn(ValueError):
    """The particles agree everywhere, so no point is more informative than
    another.  Raised by a selector that allocates by information; the designer
    reports it as a collapsed posterior."""


class PointSelector(ABC):
    """Spend a measurement budget over a candidate's points."""

    @abstractmethod
    def select(self, predictions, weights, noise, budget, rng, cost=None):
        """Choose points and their measurement time.

        ``predictions`` is ``(K, n_points)``; ``noise`` the **unit-weight**
        noise, broadcastable to ``(n_points,)``, infinite where a point cannot
        be measured; ``budget`` the total time to spend, positive; ``cost``
        the time of one unit of weight at each point, positive, or None for 1.

        Returns ``(indices, weight)``: distinct indices in increasing order,
        so tau stays sorted, and positive weights with
        ``sum(weight * cost[indices]) == budget``.
        """


def _inputs(predictions, weights, noise, budget, cost=None):
    """Validated ``(P, w, sigma, budget, cost)`` shared by every selector."""
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    w = _weights(weights, P.shape[0])
    s = _noise(noise, P.shape[1])
    budget = float(budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    return P, w, s, budget, _cost(cost, P.shape[1])


def _cost(cost, n_points):
    """Per-point cost of one unit of weight. (n_points,)  None is all ones."""
    if cost is None:
        return np.ones(n_points)
    c = np.asarray(cost, dtype=float)
    if c.shape != (n_points,):
        raise ValueError(f"cost has shape {c.shape}, expected ({n_points},)")
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("cost must be finite and positive at every point")
    return c


def _equal_time(budget, cost, idx):
    """Weights giving every point in ``idx`` the same share of the time."""
    return (budget / len(idx)) / cost[idx]


def _point_count(n_points, n_grid):
    if not 1 <= n_points <= n_grid:
        raise ValueError(f"cannot choose {n_points} of {n_grid} points")


class UniformThinning(PointSelector):
    """``n_points`` indices evenly spaced over the grid, equal time each.

    Spacing is in index, not in tau: the selector never sees tau, and on the
    uniform grids this project uses the two are the same thing.
    """

    def __init__(self, n_points):
        self.n_points = int(n_points)

    def select(self, predictions, weights, noise, budget, rng, cost=None):
        P, _, _, budget, c = _inputs(predictions, weights, noise, budget, cost)
        n_grid = P.shape[1]
        _point_count(self.n_points, n_grid)
        # Spacing of (n_grid - 1) / (n_points - 1) >= 1 keeps rounded indices
        # distinct.
        idx = np.round(np.linspace(0, n_grid - 1, self.n_points)).astype(int)
        return idx, _equal_time(budget, c, idx)


class LeastInformative(PointSelector):
    """The ``n_points`` measurable points of lowest information per unit time,
    equal time each.

    Ties -- most often among points of zero density -- go to the lower index,
    so the choice is deterministic.  Points with infinite noise cannot be
    measured and are never chosen; if fewer than ``n_points`` can be, this
    raises ValueError.  Never raises :class:`NothingToLearn`.
    """

    def __init__(self, n_points):
        raise NotImplementedError

    def select(self, predictions, weights, noise, budget, rng, cost=None):
        raise NotImplementedError


class InformationDensity(PointSelector):
    """Time proportional to ``(information_density / cost) ** power``, pruned.

    Density divided by cost is information per unit time; with no cost it is
    the density itself, and this is the old rule exactly.  Each point's weight
    is the time it receives divided by its cost.  ``power = 0.5`` is the old
    rule, described there as a near-optimal
    D-design allocation; that claim is T9's to test.  Points whose allocation
    falls below ``prune_fraction`` of the largest are dropped and the budget is
    renormalised over the survivors.  A point of zero density receives zero
    time and is always dropped, whatever the fraction.
    """

    def __init__(self, power=0.5, prune_fraction=0.05):
        if power < 0:
            raise ValueError(f"power must be non-negative, got {power}")
        if not 0 <= prune_fraction < 1:
            raise ValueError(
                f"prune_fraction must be in [0, 1), got {prune_fraction}")
        self.power = float(power)
        self.prune_fraction = float(prune_fraction)

    def select(self, predictions, weights, noise, budget, rng, cost=None):
        P, w, s, budget, c = _inputs(predictions, weights, noise, budget, cost)
        rate = information_density(P, w, s) / c
        if not np.any(rate > 0):
            raise NothingToLearn(
                "the particles agree at every point; there is no information "
                "to allocate time by")
        # Masked, not raised to the power directly: 0 ** 0 is 1, which at
        # power 0 would hand time to points that cannot separate anything.
        allocation = np.where(rate > 0, rate ** self.power, 0.0)
        keep = (allocation > 0) & (
            allocation >= self.prune_fraction * allocation.max())
        idx = np.flatnonzero(keep)
        time = budget * allocation[idx] / allocation[idx].sum()
        return idx, time / c[idx]


class GreedyUtility(PointSelector):
    """Build the point set one point at a time, by a design utility.

    Each step scores every remaining point added to the current set, in one
    :meth:`~nuclear_spin_recovery.design.utility.DesignUtility.score_many`
    call so that common random numbers apply, and keeps the best.  Every point
    gets the same share of the time, ``budget / n_points``, and is scored at
    the weight that buys: ``share / cost``.
    Raises :class:`NothingToLearn` if the first step finds no point that scores
    above zero.
    """

    def __init__(self, utility, n_points):
        if not isinstance(utility, DesignUtility):
            raise TypeError(f"{type(utility).__name__} is not a DesignUtility")
        self.utility = utility
        self.n_points = int(n_points)

    def select(self, predictions, weights, noise, budget, rng, cost=None):
        P, w, s, budget, c = _inputs(predictions, weights, noise, budget, cost)
        n_grid = P.shape[1]
        _point_count(self.n_points, n_grid)
        point_weight = (budget / self.n_points) / c
        effective = s / np.sqrt(point_weight)

        chosen = []
        remaining = list(range(n_grid))
        for step in range(self.n_points):
            sets = [sorted([*chosen, j]) for j in remaining]
            scores = self.utility.score_many(
                [P[:, cols] for cols in sets], w,
                [effective[cols] for cols in sets], rng)
            best = int(np.argmax(scores))
            if step == 0 and scores[best] <= _NOTHING:
                raise NothingToLearn(
                    "no single point scores above zero; the particles agree "
                    "wherever they can be measured")
            chosen.append(remaining.pop(best))
        idx = np.array(sorted(chosen), dtype=int)
        return idx, point_weight[idx]
