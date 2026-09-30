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

Three selectors:

- :class:`InformationDensity` -- the rule from ``adaptive_exp.py``: time
  proportional to ``density ** power``, then points below ``prune_fraction``
  of the largest allocation dropped.  Both of its numbers were buried
  constants there and are parameters here.
- :class:`GreedyUtility` -- the direct comparison: add the point that most
  improves a :class:`~nuclear_spin_recovery.design.utility.DesignUtility`,
  repeat.  Every kept point receives an equal share of the budget.
- :class:`UniformThinning` -- the control: evenly spaced in index, equal time.

**Nothing to learn is an error, not a grid.**  If the particles agree at every
point -- one particle, or several that predict identically -- there is no
information to allocate by, and a selector that relies on it raises
:class:`NothingToLearn` rather than silently returning something uniform.  The
designer is what reports it.  :class:`UniformThinning` never raises: it is the
control, and T9's degenerate case needs it to keep working.

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
    def select(self, predictions, weights, noise, budget, rng):
        """Choose points and their measurement time.

        ``predictions`` is ``(K, n_points)``; ``noise`` the **unit-weight**
        noise, broadcastable to ``(n_points,)``, infinite where a point cannot
        be measured; ``budget`` the total weight to spend, positive.

        Returns ``(indices, weight)``: distinct indices in increasing order,
        so tau stays sorted, and positive weights summing to ``budget``.
        """


def _inputs(predictions, weights, noise, budget):
    """Validated ``(P, w, sigma, budget)`` shared by every selector."""
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    w = _weights(weights, P.shape[0])
    s = _noise(noise, P.shape[1])
    budget = float(budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    return P, w, s, budget


def _point_count(n_points, n_grid):
    if not 1 <= n_points <= n_grid:
        raise ValueError(f"cannot choose {n_points} of {n_grid} points")


class UniformThinning(PointSelector):
    """``n_points`` indices evenly spaced over the grid, equal weight each.

    Spacing is in index, not in tau: the selector never sees tau, and on the
    uniform grids this project uses the two are the same thing.
    """

    def __init__(self, n_points):
        self.n_points = int(n_points)

    def select(self, predictions, weights, noise, budget, rng):
        P, _, _, budget = _inputs(predictions, weights, noise, budget)
        n_grid = P.shape[1]
        _point_count(self.n_points, n_grid)
        # Spacing of (n_grid - 1) / (n_points - 1) >= 1 keeps rounded indices
        # distinct.
        idx = np.round(np.linspace(0, n_grid - 1, self.n_points)).astype(int)
        return idx, np.full(self.n_points, budget / self.n_points)


class InformationDensity(PointSelector):
    """Time proportional to ``information_density ** power``, pruned.

    ``power = 0.5`` is the old rule, described there as a near-optimal
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

    def select(self, predictions, weights, noise, budget, rng):
        P, w, s, budget = _inputs(predictions, weights, noise, budget)
        density = information_density(P, w, s)
        if not np.any(density > 0):
            raise NothingToLearn(
                "the particles agree at every point; there is no information "
                "to allocate time by")
        # Masked, not raised to the power directly: 0 ** 0 is 1, which at
        # power 0 would hand time to points that cannot separate anything.
        allocation = np.where(density > 0, density ** self.power, 0.0)
        keep = (allocation > 0) & (
            allocation >= self.prune_fraction * allocation.max())
        idx = np.flatnonzero(keep)
        return idx, budget * allocation[idx] / allocation[idx].sum()


class GreedyUtility(PointSelector):
    """Build the point set one point at a time, by a design utility.

    Each step scores every remaining point added to the current set, in one
    :meth:`~nuclear_spin_recovery.design.utility.DesignUtility.score_many`
    call so that common random numbers apply, and keeps the best.  Points are
    scored at the weight they will finally receive, ``budget / n_points``.
    Raises :class:`NothingToLearn` if the first step finds no point that scores
    above zero.
    """

    def __init__(self, utility, n_points):
        if not isinstance(utility, DesignUtility):
            raise TypeError(f"{type(utility).__name__} is not a DesignUtility")
        self.utility = utility
        self.n_points = int(n_points)

    def select(self, predictions, weights, noise, budget, rng):
        P, w, s, budget = _inputs(predictions, weights, noise, budget)
        n_grid = P.shape[1]
        _point_count(self.n_points, n_grid)
        share = budget / self.n_points
        effective = s / np.sqrt(share)

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
        return np.array(sorted(chosen), dtype=int), np.full(self.n_points, share)
