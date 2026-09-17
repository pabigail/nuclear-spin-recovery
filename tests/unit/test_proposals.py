"""Proposal kernels and their proposal ratios."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import ContinuousReflected, DiscreteLatticeWalk, NeighborIndex

LINE = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------- continuous

def test_continuous_stays_in_bounds(rng):
    prop = ContinuousReflected(radius=0.3, lower=0.0, upper=1.0)
    x = 0.5
    for _ in range(500):
        x, _ = prop.propose(rng, x)
        assert 0.0 <= x <= 1.0


def test_continuous_ratio_is_zero(rng):
    """Reflection preserves symmetry, so the ratio is exactly zero."""
    prop = ContinuousReflected(radius=0.3, lower=0.0, upper=1.0)
    _, log_ratio = prop.propose(rng, 0.5)
    assert log_ratio == pytest.approx(0.0)


def test_continuous_step_within_radius(rng):
    prop = ContinuousReflected(radius=0.1, lower=0.0, upper=1.0)
    for _ in range(200):
        proposed, _ = prop.propose(rng, 0.5)
        assert abs(proposed - 0.5) <= 0.1 + 1e-12


def test_continuous_reflects_at_upper(rng):
    """A step past the boundary folds back inside, it is not clipped."""
    prop = ContinuousReflected(radius=0.2, lower=0.0, upper=1.0)
    seen = [prop.propose(rng, 0.95)[0] for _ in range(300)]
    assert max(seen) <= 1.0
    assert any(s < 0.95 for s in seen)
    assert not any(s == pytest.approx(1.0) for s in seen)   # no pile-up at the bound


def test_continuous_reflects_at_lower(rng):
    prop = ContinuousReflected(radius=0.2, lower=0.0, upper=1.0)
    seen = [prop.propose(rng, 0.05)[0] for _ in range(300)]
    assert min(seen) >= 0.0


def test_continuous_zero_radius_is_identity(rng):
    prop = ContinuousReflected(radius=0.0, lower=0.0, upper=1.0)
    proposed, _ = prop.propose(rng, 0.42)
    assert proposed == pytest.approx(0.42)


def test_continuous_is_unbiased():
    prop = ContinuousReflected(radius=0.1, lower=0.0, upper=1.0)
    rng = np.random.default_rng(3)
    steps = [prop.propose(rng, 0.5)[0] - 0.5 for _ in range(4000)]
    assert np.mean(steps) == pytest.approx(0.0, abs=0.01)


def test_continuous_is_reproducible():
    prop = ContinuousReflected(radius=0.1, lower=0.0, upper=1.0)
    a = [prop.propose(np.random.default_rng(5), 0.5)[0] for _ in range(3)]
    b = [prop.propose(np.random.default_rng(5), 0.5)[0] for _ in range(3)]
    assert a == pytest.approx(b)


def test_continuous_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        ContinuousReflected(radius=0.1, lower=1.0, upper=0.0)


def test_continuous_rejects_negative_radius():
    with pytest.raises(ValueError):
        ContinuousReflected(radius=-0.1, lower=0.0, upper=1.0)


# --------------------------------------------------------------- discrete

@pytest.fixture
def walk():
    return DiscreteLatticeWalk(NeighborIndex(LINE, radius=1.5))


def test_discrete_exposes_radius(walk):
    assert walk.radius == pytest.approx(1.5)


def test_discrete_proposes_a_neighbour(walk, rng):
    free = np.zeros(4, dtype=bool)
    for _ in range(50):
        site, _ = walk.propose(rng, 1, occupied=free)
        assert site in (0, 2)


def test_discrete_never_proposes_an_occupied_site(walk, rng):
    occupied = np.array([True, False, False, False])
    for _ in range(50):
        site, _ = walk.propose(rng, 1, occupied=occupied)
        assert site == 2


def test_discrete_never_proposes_the_current_site(walk, rng):
    free = np.zeros(4, dtype=bool)
    for _ in range(50):
        site, _ = walk.propose(rng, 2, occupied=free)
        assert site != 2


def test_discrete_ratio_is_zero_when_symmetric(walk, rng):
    """Sites 1 and 2 each have two free neighbours, so the ratio cancels."""
    free = np.zeros(4, dtype=bool)
    _, log_ratio = walk.propose(rng, 1, occupied=free)
    assert log_ratio == pytest.approx(0.0)


def test_discrete_ratio_is_asymmetric_at_the_edge(walk, rng):
    """Site 0 has one free neighbour, site 1 has two: log(1/2).

    This is the term that a symmetric-proposal implementation omits, and its
    absence is invisible to every recovery test.
    """
    free = np.zeros(4, dtype=bool)
    site, log_ratio = walk.propose(rng, 0, occupied=free)
    assert site == 1
    assert log_ratio == pytest.approx(np.log(1.0) - np.log(2.0))


def test_discrete_ratio_uses_availability_not_adjacency(walk, rng):
    """Occupying a neighbour of the target changes the reverse count."""
    occupied = np.array([False, False, False, True])   # site 3 taken
    site, log_ratio = walk.propose(rng, 1, occupied=occupied)
    if site == 2:
        # N(1) free = {0, 2} -> 2 ; N(2) free excluding the mover = {1} -> 1
        assert log_ratio == pytest.approx(np.log(2.0) - np.log(1.0))


def test_discrete_excludes_the_mover_from_occupancy(walk, rng):
    """The spin being moved must not block its own return path."""
    occupied = np.zeros(4, dtype=bool)
    occupied[1] = True                    # the mover itself sits at 1
    site, log_ratio = walk.propose(rng, 1, occupied=occupied)
    assert site in (0, 2)
    assert log_ratio == pytest.approx(0.0)


def test_discrete_no_free_neighbour_is_a_no_op(walk, rng):
    occupied = np.array([True, True, True, False])
    site, log_ratio = walk.propose(rng, 1, occupied=occupied)
    assert site == 1
    assert log_ratio == pytest.approx(0.0)


def test_discrete_isolated_site_is_a_no_op(walk, rng):
    """Site 3 is beyond the radius of every other site."""
    free = np.zeros(4, dtype=bool)
    site, log_ratio = walk.propose(rng, 3, occupied=free)
    assert site == 3
    assert log_ratio == pytest.approx(0.0)


def test_discrete_is_reproducible():
    walk = DiscreteLatticeWalk(NeighborIndex(LINE, radius=1.5))
    free = np.zeros(4, dtype=bool)
    a = walk.propose(np.random.default_rng(11), 1, occupied=free)[0]
    b = walk.propose(np.random.default_rng(11), 1, occupied=free)[0]
    assert a == b


def test_discrete_covers_all_free_neighbours(walk):
    """Over many draws the kernel must reach every available neighbour."""
    rng = np.random.default_rng(2)
    free = np.zeros(4, dtype=bool)
    seen = {walk.propose(rng, 1, occupied=free)[0] for _ in range(200)}
    assert seen == {0, 2}
