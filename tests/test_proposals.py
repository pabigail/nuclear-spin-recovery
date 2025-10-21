import numpy as np
import pytest

from nuclear_spin_recover import NuclearSpin, SpinBath, DiscreteLatticeRWMHProposal, ContinuousBounded2dRWMHProposal


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
        simple_spin_bath, r=1.5, spin_inds=[0, 2], params=default_params
    )

    dist_row = simple_spin_bath.distance_matrix[2]
    neighbors = np.where((dist_row <= 1.5) & (dist_row > 0))[0]
    # For the 3-spin bath, neighbors of spin 2 are {0, 1}
    assert set(neighbors) == {1, 3}


def test_excludes_existing_spins(simple_spin_bath, default_params):
    """Ensure proposal excludes spins already in spin_inds."""
    proposal = DiscreteLatticeRWMHProposal(
        simple_spin_bath, r=1.5, spin_inds=[2], params=default_params
    )

    new_idx = proposal.propose(spin_index=2)
    # Neighbors are {0, 1}, and only 1 is not in spin_inds
    assert new_idx in {0, 1}


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


@pytest.fixture
def simple_spin_bath_2():
    """Construct a SpinBath with a few NuclearSpin objects."""
    spins = [
        NuclearSpin("13C", x=0.0, y=0.0, z=0.0, A_par=2.0, A_perp=1.0, gyro=10.705),
        NuclearSpin("13C", x=1.0, y=0.0, z=0.0, A_par=5.0, A_perp=3.0, gyro=10.705),
        NuclearSpin("13C", x=0.0, y=1.0, z=0.0, A_par=-1.0, A_perp=0.5, gyro=10.705),
    ]
    return SpinBath(spins)


@pytest.fixture
def base_params_2():
    """Simple parameter dictionary matching 3 spins."""
    return {
        "A_par": np.array([2.0, 5.0, -1.0]),
        "A_perp": np.array([1.0, 3.0, 0.5]),
    }


def test_proposal_returns_expected_fields(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2

    prop = ContinuousBounded2dRWMHProposal(
        simple_spin_bath,
        spin_inds=np.array([0, 1, 2]),
        params=base_params,
        width_fac=0.1,
        max_radius=2.0,
    )

    new_inds, new_params, A_perp_prop, A_par_prop = prop.propose(spin_index=0)

    assert isinstance(new_inds, np.ndarray)
    assert isinstance(new_params, dict)
    assert isinstance(A_perp_prop, float)
    assert isinstance(A_par_prop, float)
    assert "A_perp" in new_params and "A_par" in new_params
    assert len(new_params["A_perp"]) == len(base_params["A_perp"])
    assert len(new_params["A_par"]) == len(base_params["A_par"])


def test_proposal_does_not_mutate_spinbath(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2

    prop = ContinuousBounded2dRWMHProposal(
        simple_spin_bath, [0], base_params, width_fac=0.1, max_radius=5.0
    )

    A_perp_ref = simple_spin_bath[0].A_perp
    A_par_ref = simple_spin_bath[0].A_par

    _, _, A_perp_prop, A_par_prop = prop.propose()

    # Proposal must not modify SpinBath
    assert simple_spin_bath[0].A_perp == A_perp_ref
    assert simple_spin_bath[0].A_par == A_par_ref
    # Proposed values are distinct with high probability
    assert (A_perp_prop != A_perp_ref) or (A_par_prop != A_par_ref)


def test_proposal_uses_current_values(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2

    prop = ContinuousBounded2dRWMHProposal(
        simple_spin_bath, [0], base_params, width_fac=0.0
    )

    # If width_fac=0, and we pass current values, proposal should equal them
    current_A_perp, current_A_par = 10.0, 20.0
    _, _, A_perp_prop, A_par_prop = prop.propose(
        A_perp_current=current_A_perp, A_par_current=current_A_par
    )

    # Proposal should equal the current values (width=0)
    assert np.isclose(A_perp_prop, current_A_perp)
    assert np.isclose(A_par_prop, current_A_par)


def test_boundary_reflection(simple_spin_bath_2, base_params_2):
    """Proposed values exceeding max_radius should be reflected."""
    np.random.seed(0)

    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2
    prop = ContinuousBounded2dRWMHProposal(
        simple_spin_bath, [0], base_params, width_fac=1.0, max_radius=0.5
    )

    # For reproducibility, propose far away by manually biasing
    _, _, A_perp_prop, A_par_prop = prop.propose(
        A_perp_current=10.0, A_par_current=10.0
    )

    ref_A_perp = simple_spin_bath[0].A_perp
    ref_A_par = simple_spin_bath[0].A_par
    radial_dist = np.sqrt((A_perp_prop - ref_A_perp)**2 + (A_par_prop - ref_A_par)**2)

    assert np.isclose(radial_dist, prop.max_radius, atol=1e-8)


def test_randomness_and_reproducibility(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2
    np.random.seed(123)
    prop1 = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], base_params, 0.1)
    res1 = prop1.propose(spin_index=0)

    np.random.seed(123)
    prop2 = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], base_params, 0.1)
    res2 = prop2.propose(spin_index=0)

    # With same seed, results should match exactly
    assert np.allclose(res1[2:], res2[2:])


def test_invalid_spin_index_raises(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0, 1], base_params, 0.1)
    with pytest.raises(IndexError):
        prop.propose(spin_index=10)


def test_no_spins_raises(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [], base_params, 0.1)
    with pytest.raises(ValueError):
        prop.propose()


def test_repr(simple_spin_bath_2, base_params_2):
    simple_spin_bath = simple_spin_bath_2
    base_params = base_params_2
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0, 1], base_params, 0.1)
    s = repr(prop)
    assert "width_fac" in s
    assert "max_radius" in s
    assert "n_spins" in s
