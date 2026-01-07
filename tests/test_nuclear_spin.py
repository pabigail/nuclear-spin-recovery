import sys
import numpy as np
import pytest
from nuclear_spin_recover.nuclear_spin import SingleNuclearSpin, NuclearSpinList, FullSpinBath


def test_single_nuclear_spin_valid():
    spin = SingleNuclearSpin(
        position=[0.0, 1.0, 2.0],
        gamma=10.705,
        A_parallel=1.2,
        A_perp=0.8,
    )

    assert np.allclose(spin.position, [0.0, 1.0, 2.0])
    assert spin.gamma == 10.705
    assert spin.A_parallel == 1.2
    assert spin.A_perp == 0.8

@pytest.mark.parametrize(
    "position",
    [
        [0.0, 1.0],          # wrong length
        [0.0, 1.0, 2.0, 3.0],
        "not a vector",
    ],
)
def test_single_nuclear_spin_invalid_position(position):
    with pytest.raises(ValueError, match="Position must be a 3-element vector"):
        SingleNuclearSpin(
            position=position,
            gamma=1.0,
            A_parallel=1.0,
            A_perp=1.0,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma", True),
        ("A_parallel", False),
        ("A_perp", "bad"),
    ],
)
def test_single_nuclear_spin_invalid_scalars(field, value):
    kwargs = dict(
        position=[0, 0, 0],
        gamma=1.0,
        A_parallel=1.0,
        A_perp=1.0,
    )
    kwargs[field] = value

    with pytest.raises(TypeError, match=f"{field} must be a real number"):
        SingleNuclearSpin(**kwargs)

def test_single_nuclear_spin_signature_deterministic():
    spin1 = SingleNuclearSpin([0, 0, 0], 1.0, 2.0, 3.0)
    spin2 = SingleNuclearSpin([0, 0, 0], 1.0, 2.0, 3.0)

    assert spin1.get_signature() == spin2.get_signature()


@pytest.fixture
def spin():
    return SingleNuclearSpin([0, 0, 0], 1.0, 2.0, 3.0)


def test_nuclear_spin_list_valid(spin):
    spins = NuclearSpinList([spin])
    assert len(spins) == 1
    assert spins[0] is spin

def test_nuclear_spin_list_invalid_container():
    with pytest.raises(TypeError, match="Spins must be provided as a list"):
        NuclearSpinList("not a list")

def test_nuclear_spin_list_invalid_element(spin):
    with pytest.raises(TypeError, match="All elements must be SingleNuclearSpin"):
        NuclearSpinList([spin, "bad"])

def test_nuclear_spin_list_append(spin):
    spins = NuclearSpinList()
    spins.append(spin)

    assert len(spins) == 1

def test_nuclear_spin_list_append_invalid():
    spins = NuclearSpinList()
    with pytest.raises(TypeError, match="Can only append SingleNuclearSpin"):
        spins.append("bad")

def test_nuclear_spin_list_delete(spin):
    spins = NuclearSpinList([spin, spin])
    spins.delete([0])

    assert len(spins) == 1


def test_nuclear_spin_list_delete_out_of_range(spin):
    spins = NuclearSpinList([spin])
    with pytest.raises(IndexError):
        spins.delete([1])

def test_nuclear_spin_list_signature_deterministic(spin):
    spins1 = NuclearSpinList([spin])
    spins2 = NuclearSpinList([spin])

    assert spins1.get_signature() == spins2.get_signature()

@pytest.fixture
def spin_list(spin):
    return NuclearSpinList([spin, spin])

def test_full_spin_bath_valid(spin_list):
    bath = FullSpinBath(spins=spin_list)
    assert bath.spins is spin_list
    assert bath.distance_matrix is None

def test_full_spin_bath_invalid_spins():
    with pytest.raises(TypeError, match="spins must be a NuclearSpinList"):
        FullSpinBath(spins="bad")

def test_full_spin_bath_distance_matrix_shape(spin_list):
    dm = np.zeros((2, 2))
    bath = FullSpinBath(spins=spin_list, distance_matrix=dm)

    assert bath.distance_matrix.shape == (2, 2)

def test_full_spin_bath_invalid_distance_matrix(spin_list):
    with pytest.raises(ValueError, match="Distance matrix must have shape"):
        FullSpinBath(
            spins=spin_list,
            distance_matrix=np.zeros((3, 3)),
        )

def test_full_spin_bath_compute_distance_matrix(spin_list):
    bath = FullSpinBath(spins=spin_list)
    dm = bath.compute_distance_matrix()

    assert dm.shape == (2, 2)
    assert np.allclose(dm, dm.T)
    assert np.allclose(np.diag(dm), 0.0)

def test_full_spin_bath_metadata_validation(spin_list):
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        FullSpinBath(spins=spin_list, metadata="bad")

def test_full_spin_bath_signature_deterministic(spin_list):
    bath1 = FullSpinBath(spins=spin_list)
    bath2 = FullSpinBath(spins=spin_list)

    assert bath1.get_signature() == bath2.get_signature()

