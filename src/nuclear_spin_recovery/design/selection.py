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


class UniformThinning(PointSelector):
    """``n_points`` indices evenly spaced over the grid, equal weight each.

    Spacing is in index, not in tau: the selector never sees tau, and on the
    uniform grids this project uses the two are the same thing.
    """

    def __init__(self, n_points):
        raise NotImplementedError

    def select(self, predictions, weights, noise, budget, rng):
        raise NotImplementedError


class InformationDensity(PointSelector):
    """Time proportional to ``information_density ** power``, pruned.

    ``power = 0.5`` is the old rule, described there as a near-optimal
    D-design allocation; that claim is T9's to test.  Points whose allocation
    falls below ``prune_fraction`` of the largest are dropped and the budget is
    renormalised over the survivors.  A point of zero density receives zero
    time and is always dropped, whatever the fraction.
    """

    def __init__(self, power=0.5, prune_fraction=0.05):
        raise NotImplementedError

    def select(self, predictions, weights, noise, budget, rng):
        raise NotImplementedError


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
        raise NotImplementedError

    def select(self, predictions, weights, noise, budget, rng):
        raise NotImplementedError
