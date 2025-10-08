import numpy as np
import pytest

from nuclear_spin_recover import NuclearSpin, SpinBath, DiscreteLatticeRWMHProposal


@pytest.fixture
def simple_spin_bath():
    """Create a SpinBath with 5 nuclear spins equally spaced along x-axis."""
    spins = [
        NuclearSpin(
            spin_type="13C",
            x=float(i), y=0.0, z=0.0,
            A_par=10.0, A_perp=5.0,
            gyro=1.0  # arbitrary nonzero
        )
        for i in range(5)
    ]
    return SpinBath(spins=spins)


def test_initialization(simple_spin_bath):
    """Ensure the proposal initializes correctly."""
    proposal = DiscreteLatticeRWMHProposal(
        spin_bath=simple_spin_bath, r=2.0, spin_inds=[0, 1]
    )
    assert isinstance(proposal.spin_bath, SpinBath)
    assert proposal.r == 2.0
    assert proposal.spin_inds == [0, 1]
    assert "DiscreteLatticeRWMHProposal" in repr(proposal)


def test_random_reference_spin(simple_spin_bath, monkeypatch):
    """Ensure proposal picks a random reference spin if none provided."""
    proposal = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5)

    # Force np.random.randint to always pick index 2
    monkeypatch.setattr(np.random, "randint", lambda n: 2)
    new_idx = proposal.propose()

    # Should return a valid neighbor of spin 2 (index 1 or 3)
    assert new_idx in [1, 3]


def test_local_neighbors(simple_spin_bath):
    """Ensure neighborhood selection works correctly."""
    proposal = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5)

    # For spin 2, within r=1.5 should include indices 1 and 3
    dist_row = simple_spin_bath.distance_matrix[2]
    neighbors = np.where((dist_row <= 1.5) & (dist_row > 0))[0]
    assert set(neighbors) == {1, 3}

    new_idx = proposal.propose(spin_index=2)
    assert new_idx in neighbors


def test_excludes_existing_spins(simple_spin_bath):
    """Ensure proposal excludes spins already in spin_inds."""
    proposal = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[3])

    # For spin 2, neighbors are [1, 3], but 3 is already in spin_inds
    new_idx = proposal.propose(spin_index=2)
    assert new_idx == 1


def test_no_candidates_raises(simple_spin_bath):
    """Ensure ValueError is raised when no candidate spins exist within radius."""
    proposal = DiscreteLatticeRWMHProposal(simple_spin_bath, r=0.4)
    with pytest.raises(ValueError, match="No available candidate spins"):
        proposal.propose(spin_index=2)


def test_invalid_inputs(simple_spin_bath):
    """Ensure invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError, match="Radius must be positive"):
        DiscreteLatticeRWMHProposal(simple_spin_bath, r=-1.0)

    proposal = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.0)
    with pytest.raises(IndexError, match="Spin index out of range"):
        proposal.propose(spin_index=99)

