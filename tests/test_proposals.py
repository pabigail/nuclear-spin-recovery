import numpy as np
import pytest

from nuclear_spin_recover import NuclearSpin, SpinBath, DiscreteLatticeRWMHProposal


@pytest.fixture
def simple_spin_bath():
    """Create a SpinBath with 5 nuclear spins equally spaced along the x-axis."""
    spins = [
        NuclearSpin(
            spin_type="13C",
            x=float(i), y=0.0, z=0.0,
            A_par=10.0, A_perp=5.0,
            gyro=1.0,
        )
        for i in range(5)
    ]
    return SpinBath(spins=spins)


@pytest.fixture
def default_params():
    """Return a dummy parameter dictionary."""
    return {"lambda_decoherence": 0.5}


def test_initialization(simple_spin_bath, default_params):
    """Ensure the proposal initializes correctly."""
    proposal = DiscreteLatticeRWMHProposal(
        spin_bath=simple_spin_bath, r=2.0, spin_inds=[0, 1], params=default_params
    )
    assert isinstance(proposal.spin_bath, SpinBath)
    assert proposal.r == 2.0
    assert np.all(proposal.spin_inds == np.array([0, 1]))
    assert proposal.params == default_params
    assert "DiscreteLatticeRWMHProposal" in repr(proposal)


def test_random_reference_spin(simple_spin_bath, default_params, monkeypatch):
    """Ensure proposal picks a random reference spin if none provided."""
    # Only some spins are "in model" to leave available neighbors
    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.5, spin_inds=[0, 2, 4], params=default_params
    )

    # Force np.random.randint to always pick index 2
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 2)
    new_idx = proposal.propose()

    # Should return a valid neighbor of spin 2 (index 1 or 3)
    assert new_idx in [1, 3]


def test_local_neighbors(simple_spin_bath, default_params):
    """Ensure neighborhood selection works correctly."""
    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.5, spin_inds=[0, 2, 4], params=default_params
    )

    dist_row = simple_spin_bath.distance_matrix[2]
    neighbors = np.where((dist_row <= 1.5) & (dist_row > 0))[0]
    assert set(neighbors) == {1, 3}

    new_idx = proposal.propose(spin_index=2)
    assert new_idx in neighbors


def test_excludes_existing_spins(simple_spin_bath, default_params):
    """Ensure proposal excludes spins already in spin_inds."""
    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.5, spin_inds=[2,3], params=default_params
    )

    new_idx = proposal.propose(spin_index=2)
    # 2’s neighbors are [1, 3]; 3 is excluded → only 1 remains
    assert new_idx == 1


def test_no_candidates_raises(simple_spin_bath, default_params):
    """Ensure ValueError is raised when no candidate spins exist within radius."""
    # spin_inds contains all spins, so no candidates will remain
    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.5, spin_inds=[0, 1, 2, 3, 4], params=default_params
    )
    with pytest.raises(ValueError, match="No available candidate spins"):
        proposal.propose(spin_index=2)


def test_invalid_inputs(simple_spin_bath, default_params):
    """Ensure invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError, match="Radius r must be positive"):
        DiscreteLatticeRWMHProposal(simple_spin_bath, r=-1.0, spin_inds=[], params=default_params)

    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.0, spin_inds=[0, 1, 3], params=default_params
    )
    with pytest.raises(IndexError, match=r"spin_index 99 is not in current spin_inds"):
        proposal.propose(spin_index=99)
