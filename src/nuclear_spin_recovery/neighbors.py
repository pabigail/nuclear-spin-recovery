"""Radius-limited neighbour lists over lattice sites.

Precomputed once per site table and reused by the discrete random walk, which
proposes a move to another site within ``radius`` of the current one.  The
radius is bounded below by the nearest-neighbour distance of the lattice
(1.54 Angstrom in diamond).  See docs/model-specification.md Sec. 8.2.
"""

from __future__ import annotations

import numpy as np


class NeighborIndex:
    """Sites within a fixed radius of each site, excluding the site itself."""

    def __init__(self, positions, radius):
        self.positions = np.asarray(positions, dtype=float)
        self.radius = float(radius)

    def __len__(self) -> int:
        raise NotImplementedError

    def neighbors(self, site):
        """Indices of sites within ``radius`` of ``site``, ascending."""
        raise NotImplementedError

    def count_available(self, site, occupied, ignore=None):
        """Number of unoccupied neighbours of ``site``.

        ``ignore`` is the site of the spin currently being moved, which must
        not count against its own move.  This count is what makes the discrete
        proposal asymmetric; see :class:`~nuclear_spin_recovery.proposals.
        DiscreteLatticeWalk`.
        """
        raise NotImplementedError
