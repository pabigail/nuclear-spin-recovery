"""Radius-limited neighbour lists."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import NeighborIndex

# Four sites on a line at x = 0, 1, 2, 5.
LINE = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [5.0, 0, 0]])


@pytest.fixture
def line():
    return NeighborIndex(LINE, radius=1.5)


def test_len_is_site_count(line):
    assert len(line) == 4


def test_neighbors_within_radius(line):
    assert line.neighbors(0).tolist() == [1]
    assert line.neighbors(1).tolist() == [0, 2]
    assert line.neighbors(2).tolist() == [1]


def test_isolated_site_has_none(line):
    assert line.neighbors(3).tolist() == []


def test_site_is_not_its_own_neighbour(line):
    for i in range(4):
        assert i not in line.neighbors(i).tolist()


def test_neighbors_are_sorted(line):
    assert line.neighbors(1).tolist() == sorted(line.neighbors(1).tolist())


def test_relation_is_symmetric():
    rng = np.random.default_rng(0)
    idx = NeighborIndex(rng.uniform(0, 5, size=(30, 3)), radius=2.0)
    for i in range(30):
        for j in idx.neighbors(i):
            assert i in idx.neighbors(j).tolist()


def test_matches_brute_force():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 5, size=(40, 3))
    idx = NeighborIndex(pos, radius=1.8)
    for i in range(40):
        d = np.linalg.norm(pos - pos[i], axis=1)
        expected = np.flatnonzero((d <= 1.8) & (np.arange(40) != i))
        assert idx.neighbors(i).tolist() == expected.tolist()


def test_radius_below_nearest_neighbour_gives_none():
    idx = NeighborIndex(LINE, radius=0.5)
    for i in range(4):
        assert idx.neighbors(i).tolist() == []


def test_radius_is_inclusive():
    """A site exactly at the radius counts, matching the table filters."""
    idx = NeighborIndex(LINE, radius=1.0)
    assert idx.neighbors(0).tolist() == [1]


def test_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        NeighborIndex(LINE, radius=0.0)


def test_count_available_excludes_occupied(line):
    occupied = np.zeros(4, dtype=bool)
    assert line.count_available(1, occupied) == 2
    occupied[0] = True
    assert line.count_available(1, occupied) == 1


def test_count_available_ignores_the_moving_spin(line):
    """A spin must not count itself as blocking its own move."""
    occupied = np.zeros(4, dtype=bool)
    occupied[1] = True
    assert line.count_available(0, occupied, ignore=1) == 1
