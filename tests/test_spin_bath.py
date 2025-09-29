import pytest
import numpy as np
from nuclear_spin_recover import NuclearSpin  # replace with actual module name

def test_cartesian_initialization():
    spin = NuclearSpin(
        spin_type="C13",
        x=1.0, y=2.0, z=3.0,
        A_xx=0.1, A_yy=0.2, A_zz=0.3,
        A_xy=0.01, A_yz=0.02, A_xz=0.03,
        w_L=5.0
    )
    # Check coordinates
    np.testing.assert_array_equal(spin.xyz, [1.0, 2.0, 3.0])
    # Check Cartesian tensor
    expected_A = np.array([
        [0.1, 0.01, 0.03],
        [0.01, 0.2, 0.02],
        [0.03, 0.02, 0.3]
    ])
    np.testing.assert_allclose(spin.A, expected_A)
    # Check computed axial parameters
    expected_A_perp = np.sqrt(0.03**2 + 0.02**2)
    expected_A_par = 0.3
    assert np.isclose(spin.A_perp, expected_A_perp)
    assert np.isclose(spin.A_par, expected_A_par)

def test_axial_initialization():
    spin = NuclearSpin(
        spin_type="N14",
        x=0.0, y=0.0, z=0.0,
        A_par=0.5, A_perp=0.1,
        w_L=2.0
    )
    # Check coordinates
    np.testing.assert_array_equal(spin.xyz, [0.0, 0.0, 0.0])
    # Check Cartesian tensor filled correctly
    expected_A = np.array([
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.5]
    ])
    np.testing.assert_allclose(spin.A, expected_A)
    # Check axial parameters
    assert spin.A_par == 0.5
    assert spin.A_perp == 0.1

def test_missing_required_parameters():
    # Missing coordinates
    with pytest.raises(ValueError):
        NuclearSpin(spin_type="H1", x=0.0, y=1.0, w_L=1.0)
    # Missing w_L
    with pytest.raises(ValueError):
        NuclearSpin(spin_type="H1", x=0.0, y=1.0, z=2.0)
    # Missing hyperfine
    with pytest.raises(ValueError):
        NuclearSpin(spin_type="H1", x=0.0, y=1.0, z=2.0, w_L=1.0)

def test_to_record_and_repr():
    spin = NuclearSpin(
        spin_type="C13",
        x=1.0, y=2.0, z=3.0,
        A_xx=0.1, A_yy=0.2, A_zz=0.3,
        A_xy=0.01, A_yz=0.02, A_xz=0.03,
        w_L=5.0
    )
    record = spin.to_record()
    assert record[0] == "C13"
    np.testing.assert_array_equal(record[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(record[2], spin.A)
    np.testing.assert_allclose(record[3], spin.Q)
    # Check __repr__ contains key info
    rep = repr(spin)
    assert "C13" in rep and "w_L=5.0" in rep

@pytest.mark.parametrize("uncertainty", [
    None,
    0.1,
    [0.1, 0.2],
    np.array([0.05, 0.15]),
])
def test_uncertainty_field_accepts_valid_inputs(uncertainty):
    spin = NuclearSpin(
        spin_type="13C",
        x=0.0, y=0.0, z=1.0,
        A_par=2.0, A_perp=0.5,
        w_L=1.5,
        uncertainty=uncertainty
    )
    # Check it is stored correctly
    if uncertainty is None:
        assert spin.uncertainty is None
    elif isinstance(uncertainty, (list, np.ndarray)):
        np.testing.assert_allclose(spin.uncertainty, uncertainty)
    else:
        assert spin.uncertainty == uncertainty


def test_uncertainty_invalid_type():
    with pytest.raises(TypeError, match="Uncertainty must be None, float, or array-like"):
        NuclearSpin(
            spin_type="13C",
            x=0.0, y=0.0, z=1.0,
            A_par=2.0, A_perp=0.5,
            w_L=1.5,
            uncertainty="not a number"
        )


def test_uncertainty_array_wrong_length():
    with pytest.raises(ValueError, match="Uncertainty array must have length 2"):
        NuclearSpin(
            spin_type="13C",
            x=0.0, y=0.0, z=1.0,
            A_par=2.0, A_perp=0.5,
            w_L=1.5,
            uncertainty=[0.1, 0.2, 0.3]  # too long
        )
