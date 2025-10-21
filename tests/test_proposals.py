import numpy as np
import pytest
import random
from nuclear_spin_recover import (
    NuclearSpin,
    SpinBath,
    DiscreteLatticeRWMHProposal,
    ContinuousBounded2dRWMHProposal,
    ContinuousBounded1dRWMHProposal,
    DiscreteRJMCMCProposal
)

# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def simple_spin_bath():
    """SpinBath with 5 nuclear spins along x-axis."""
    spins = [
        NuclearSpin(spin_type="13C", x=float(i), y=0.0, z=0.0,
                    A_par=10.0, A_perp=5.0, gyro=1.0)
        for i in range(5)
    ]
    return SpinBath(spins=spins)

@pytest.fixture
def default_params():
    """Dummy MCMC params including arrays for continuous proposals."""
    return {
        "lambda_decoherence": 0.5,
        "A_par": np.array([10.0]*5, dtype=float),
        "A_perp": np.array([5.0]*5, dtype=float)
    }

# -------------------------------
# Discrete Lattice Proposal Tests
# -------------------------------
def test_initialization(simple_spin_bath, default_params):
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=2.0, spin_inds=[0,1], params=default_params)
    assert isinstance(prop.spin_bath, SpinBath)
    assert prop.r == 2.0
    assert np.all(prop.current_spins == np.array([0,1]))
    assert prop.current_params == default_params
    assert "DiscreteLatticeRWMHProposal" in repr(prop)

def test_random_reference_spin(simple_spin_bath, default_params, monkeypatch):
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[0,2,4], params=default_params)
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 2)
    _, _, new_idx = prop.propose()
    assert new_idx in [1,3]

def test_local_neighbors(simple_spin_bath, default_params):
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[0,2], params=default_params)
    dist_row = simple_spin_bath.distance_matrix[2]
    neighbors = np.where((dist_row <= 1.5) & (dist_row > 0))[0]
    assert set(neighbors) == {1,3}

def test_excludes_existing_spins(simple_spin_bath, default_params):
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[2], params=default_params)
    _, _, new_idx = prop.propose(spin_index=2)
    assert int(new_idx) in {0,1,3}

def test_no_candidates_raises(simple_spin_bath, default_params):
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[0,1,2,3,4], params=default_params)
    with pytest.raises(ValueError, match="No available candidate spins"):
        prop.propose(spin_index=2)

def test_invalid_inputs(simple_spin_bath, default_params):
    with pytest.raises(ValueError):
        DiscreteLatticeRWMHProposal(simple_spin_bath, r=-1.0, spin_inds=[], params=default_params)
    prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.0, spin_inds=[0,1,3], params=default_params)
    with pytest.raises(IndexError):
        prop.propose(spin_index=99)

# -------------------------------
# Continuous 2D Proposal Tests
# -------------------------------
def test_proposal_returns_expected_fields(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0,1,2], default_params, width_fac=0.1, max_radius=2.0)
    new_inds, new_params, A_perp_prop, A_par_prop = prop.propose(spin_index=0)
    assert isinstance(new_inds, np.ndarray)
    assert isinstance(new_params, dict)
    assert isinstance(A_perp_prop, float)
    assert isinstance(A_par_prop, float)
    assert "A_perp" in new_params and "A_par" in new_params
    assert len(new_params["A_perp"]) == len(default_params["A_perp"])

def test_proposal_does_not_mutate_spinbath(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], default_params, width_fac=0.1, max_radius=5.0)
    A_perp_ref = simple_spin_bath[0].A_perp
    A_par_ref = simple_spin_bath[0].A_par
    _, _, A_perp_prop, A_par_prop = prop.propose()
    assert simple_spin_bath[0].A_perp == A_perp_ref
    assert simple_spin_bath[0].A_par == A_par_ref
    assert (A_perp_prop != A_perp_ref) or (A_par_prop != A_par_ref)

def test_proposal_uses_current_values(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], default_params, width_fac=0.0)
    _, _, A_perp_prop, A_par_prop = prop.propose(A_perp_current=10.0, A_par_current=20.0)
    assert np.isclose(A_perp_prop, 10.0)
    assert np.isclose(A_par_prop, 20.0)

def test_boundary_reflection(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], default_params, width_fac=1.0, max_radius=0.5)
    _, _, A_perp_prop, A_par_prop = prop.propose(A_perp_current=10.0, A_par_current=10.0)
    ref_A_perp = simple_spin_bath[0].A_perp
    ref_A_par = simple_spin_bath[0].A_par
    radial_dist = np.sqrt((A_perp_prop - ref_A_perp)**2 + (A_par_prop - ref_A_par)**2)
    assert np.isclose(radial_dist, prop.max_radius, atol=1e-8)

def test_randomness_and_reproducibility(simple_spin_bath, default_params):
    np.random.seed(123)
    prop1 = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], default_params, 0.1)
    res1 = prop1.propose(spin_index=0)
    np.random.seed(123)
    prop2 = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0], default_params, 0.1)
    res2 = prop2.propose(spin_index=0)
    assert np.allclose(res1[2:], res2[2:])

def test_invalid_spin_index_raises(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0,1], default_params, 0.1)
    with pytest.raises(IndexError):
        prop.propose(spin_index=10)

def test_no_spins_raises(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [], default_params, 0.1)
    with pytest.raises(ValueError):
        prop.propose()

def test_repr(simple_spin_bath, default_params):
    prop = ContinuousBounded2dRWMHProposal(simple_spin_bath, [0,1], default_params, 0.1)
    s = repr(prop)
    assert "width_fac" in s and "max_radius" in s and "n_spins" in s

# -------------------------------
# Combined Discrete + Continuous MCMC Chain Test
# -------------------------------
def test_combined_mcmc_chain(simple_spin_bath, default_params, monkeypatch):
    # Deterministic random numbers
    class FixedRNG:
        def __init__(self):
            self.choice_seq = [1, 3]
            self.choice_idx = 0
            self.normal_seq = [5.1, 10.1, 5.2, 10.2]
            self.normal_idx = 0

        def choice(self, a, size=None, replace=True, p=None):
            val = self.choice_seq[self.choice_idx]
            self.choice_idx += 1
            return val

        def normal(self, loc=0.0, scale=1.0, size=None):
            val = self.normal_seq[self.normal_idx]
            self.normal_idx += 1
            return val

    rng = FixedRNG()
    monkeypatch.setattr(np.random, "choice", rng.choice)
    monkeypatch.setattr(np.random, "normal", rng.normal)

    # Discrete
    disc_prop = DiscreteLatticeRWMHProposal(simple_spin_bath, r=1.5, spin_inds=[0,2], params=default_params)
    _, _, new_spin = disc_prop.propose(spin_index=2)
    disc_prop.prop_spins = np.array([0,2,new_spin])
    disc_prop.prop_params = disc_prop.current_params.copy()
    disc_prop.accept_prop()
    assert new_spin in disc_prop.current_spins


# -------------------------------
# ContinuousBounded1dRWMHProposal Test
# -------------------------------
def test_proposal_initialization(default_params):
    prop = ContinuousBounded1dRWMHProposal(params=default_params, radius=0.1)
    assert np.isclose(prop.current_params["lambda_decoherence"], 0.5)
    assert prop.radius == 0.1
    assert prop.current_spins.size == 0  # no spins

def test_proposal_reflection(default_params):
    np.random.seed(0)
    prop = ContinuousBounded1dRWMHProposal(params=default_params, radius=1.0)

    # Propose multiple times to hit boundaries
    for _ in range(100):
        _, new_params = prop.propose()
        val = new_params["lambda_decoherence"]
        assert 0 < val <= 1, f"Value out of bounds: {val}"

def test_proposal_reproducibility(default_params):
    np.random.seed(123)
    prop1 = ContinuousBounded1dRWMHProposal(params=default_params, radius=0.1)
    _, p1 = prop1.propose()

    np.random.seed(123)
    prop2 = ContinuousBounded1dRWMHProposal(params=default_params, radius=0.1)
    _, p2 = prop2.propose()

    assert np.isclose(p1["lambda_decoherence"], p2["lambda_decoherence"])

def test_accept_reject(default_params):
    prop = ContinuousBounded1dRWMHProposal(params=default_params, radius=0.1)
    _, p_new = prop.propose()
    # Accept
    prop.accept_prop()
    assert np.isclose(prop.current_params["lambda_decoherence"], p_new["lambda_decoherence"])

    # Propose again
    _, p_new2 = prop.propose()
    # Reject
    prop.reject_prop()
    assert np.isclose(prop.current_params["lambda_decoherence"], p_new["lambda_decoherence"])


# -------------------------------
# Helper comparisons
# -------------------------------
def spins_equal(a, b):
    return np.array_equal(np.array(a), np.array(b))


def params_equal(p1, p2):
    for key in p1:
        if key not in p2:
            return False
        v1, v2 = p1[key], p2[key]
        if isinstance(v1, np.ndarray):
            if not np.allclose(v1, v2):
                return False
        elif v1 != v2:
            return False
    return True


# -------------------------------
# Tests
# -------------------------------
def test_birth_proposal(simple_spin_bath, default_params):
    spin_inds = np.array([0, 1, 2], dtype=int)
    prop = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                  spin_inds=spin_inds, birth=True,
                                  params=default_params)

    spins_prop, params_prop = prop.propose()
    assert len(spins_prop) == len(spin_inds) + 1
    assert "lambda_decoherence" in params_prop
    assert isinstance(params_prop["lambda_decoherence"], float)


def test_death_proposal(simple_spin_bath, default_params):
    spin_inds = np.array([0, 1, 2, 3], dtype=int)
    prop = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                  spin_inds=spin_inds, birth=False,
                                  params=default_params)
    spins_prop, params_prop = prop.propose()

    assert len(spins_prop) == len(spin_inds) - 1
    assert "lambda_decoherence" in params_prop
    assert isinstance(params_prop["lambda_decoherence"], float)


def test_birth_when_at_max(simple_spin_bath, default_params):
    spin_inds = np.arange(5)
    prop = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                  spin_inds=spin_inds, birth=True,
                                  params=default_params)
    spins_prop, params_prop = prop.propose()

    assert spins_equal(spins_prop, spin_inds)
    assert params_equal(params_prop, default_params)


def test_reproducible_birth_death(simple_spin_bath, default_params):
    spin_inds = np.array([0, 1], dtype=int)
    random.seed(123)
    np.random.seed(123)
    prop1 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=None,
                                   params=default_params)
    spins1, params1 = prop1.propose()

    random.seed(123)
    np.random.seed(123)
    prop2 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=None,
                                   params=default_params)
    spins2, params2 = prop2.propose()

    assert spins_equal(spins1, spins2)
    assert params_equal(params1, params2)


def test_reproducible_birth(simple_spin_bath, default_params):
    spin_inds = np.array([0, 1], dtype=int)
    random.seed(42)
    np.random.seed(42)
    prop1 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=True,
                                   params=default_params)
    spins1, params1 = prop1.propose()

    random.seed(42)
    np.random.seed(42)
    prop2 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=True,
                                   params=default_params)
    spins2, params2 = prop2.propose()

    assert spins_equal(spins1, spins2)
    assert params_equal(params1, params2)


def test_reproducible_death(simple_spin_bath, default_params):
    spin_inds = np.array([0, 1, 2], dtype=int)
    random.seed(99)
    np.random.seed(99)
    prop1 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=False,
                                   params=default_params)
    spins1, params1 = prop1.propose()

    random.seed(99)
    np.random.seed(99)
    prop2 = DiscreteRJMCMCProposal(simple_spin_bath, max_spins=5,
                                   spin_inds=spin_inds, birth=False,
                                   params=default_params)
    spins2, params2 = prop2.propose()

    assert spins_equal(spins1, spins2)
    assert params_equal(params1, params2)
