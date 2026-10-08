"""Which points of a candidate to measure, and for how long.

A selector spends a **budget of measurement time** over a candidate's points
and returns both the points it kept and the time each received, as relative
weights.  One unit of weight is one point measured for the baseline averaging
time, so the effective noise at a kept point is ``sigma / sqrt(w_j)`` and the
weights go straight into
:attr:`~nuclear_spin_recovery.experiment.Experiment.weight`.

**Points cost different amounts of time.**  ``select`` takes ``cost``, the
time of one unit of weight at each point -- for CPMG,
:class:`~nuclear_spin_recovery.design.cost.SequenceDuration`, ``2 N tau``.
The budget is total time, ``sum_j w_j c_j``, and a weight still means relative
repetitions, so the likelihood is untouched.  Information per unit time at a
point is its density divided by its cost: an expensive point has to earn it.

:class:`InformationDensity` is the rule from ``adaptive_exp.py``: time
proportional to ``(density / cost) ** power``, then points below
``prune_fraction`` of the largest allocation dropped.  Both of its numbers
were buried constants there and are parameters here.

**Nothing to learn is an error, not a grid.**  If the particles agree at every
point -- one particle, or several that predict identically -- there is no
information to allocate by, and the selector raises :class:`NothingToLearn`
rather than silently returning something uniform.  The designer turns that
into a candidate with no design.

See docs/phase-5-plan.md, unit 5c.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utility import _noise, _weights, information_density

#: Information density at or below this is none at all: the particles agree
#: to rounding.  In units of density.
_NOTHING = 1e-12


class NothingToLearn(ValueError):
    """The particles agree everywhere, so no point is more informative than
    another.  Raised by a selector that allocates by information; the designer
    reports it as a candidate that can tell nothing apart."""


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
