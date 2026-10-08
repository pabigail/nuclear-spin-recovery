"""Stand-ins for the designer's two pluggable parts, for tests only.

The package ships one utility and one selector.  Both are fine for designing
experiments and awkward for testing the designer around them: the expected
information gain is a Monte Carlo estimate, so exact relations -- twice the
budget, twice the score -- hold only approximately, and the density selector's
choice of delays depends on the scene.  These two are deterministic and
transparent, and exist so that the designer's own arithmetic can be checked
exactly.
"""

from __future__ import annotations

import numpy as np

from nuclear_spin_recovery import DesignUtility, PointSelector, information_density


class TotalDensity(DesignUtility):
    """The sum over measured points of the information density.

    Deterministic; ``rng`` is ignored.  A point with infinite noise is not
    measured and adds nothing.
    """

    def score_many(self, predictions, weights, noise, rng):
        return np.array([information_density(p, weights, s).sum()
                         for p, s in zip(predictions, noise, strict=True)])


class EvenSpread(PointSelector):
    """``n_points`` delays evenly spaced in index, equal time each."""

    def __init__(self, n_points):
        self.n_points = int(n_points)

    def select(self, predictions, weights, noise, budget, rng, cost=None):
        n_grid = np.atleast_2d(predictions).shape[1]
        cost = np.ones(n_grid) if cost is None else np.asarray(cost, float)
        idx = np.round(np.linspace(0, n_grid - 1, self.n_points)).astype(int)
        return idx, (float(budget) / self.n_points) / cost[idx]
