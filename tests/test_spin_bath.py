import pytest
import numpy as np
import pandas as pd
from nuclear_spin_recover import NuclearSpin, SpinBath  # replace with actual module name


#----------NuclearSpin tests----------
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

#----------SpinBath tests----------
def make_spin(**kwargs):
    """Helper to make a simple spin with defaults overridden by kwargs."""
    defaults = dict(
        spin_type="13C",
        x=0.0, y=0.0, z=0.0,
        A_par=2.0, A_perp=0.5,
        w_L=1.5,
        uncertainty=None,
    )
    defaults.update(kwargs)
    return NuclearSpin(**defaults)


def test_dataframe_no_uncertainty():
    bath = SpinBath([make_spin(x=0), make_spin(x=1)])
    df = bath.dataframe
    assert "uncertainty_par" in df.columns
    assert "uncertainty_perp" in df.columns
    assert np.all(np.isnan(df["uncertainty_par"]))
    assert np.all(np.isnan(df["uncertainty_perp"]))


def test_dataframe_with_uncertainty_float():
    bath = SpinBath([make_spin(x=0, uncertainty=0.1), make_spin(x=1)])
    df = bath.dataframe
    assert np.isclose(df.loc[0, "uncertainty_par"], 0.1)
    assert np.isclose(df.loc[0, "uncertainty_perp"], 0.1)
    assert np.isnan(df.loc[1, "uncertainty_par"])
    assert np.isnan(df.loc[1, "uncertainty_perp"])


def test_dataframe_with_uncertainty_array():
    bath = SpinBath([
        make_spin(x=0, uncertainty=[0.1, 0.2]),
        make_spin(x=1, uncertainty=None),
    ])
    df = bath.dataframe
    assert np.isclose(df.loc[0, "uncertainty_par"], 0.1)
    assert np.isclose(df.loc[0, "uncertainty_perp"], 0.2)
    assert np.isnan(df.loc[1, "uncertainty_par"])
    assert np.isnan(df.loc[1, "uncertainty_perp"])


def test_distance_matrix_correctness():
    spin1 = make_spin(x=0, y=0, z=0)
    spin2 = make_spin(x=1, y=0, z=0)
    bath = SpinBath([spin1, spin2])
    dmat = bath.distance_matrix
    assert dmat.shape == (2, 2)
    assert np.isclose(dmat[0, 1], 1.0)
    assert np.isclose(dmat[1, 0], 1.0)
    assert np.isclose(dmat[0, 0], 0.0)


def test_len_and_getitem():
    spins = [make_spin(x=i) for i in range(3)]
    bath = SpinBath(spins)
    assert len(bath) == 3
    assert bath[0].xyz[0] == 0.0
    assert bath[1].xyz[0] == 1.0
    assert bath[2].xyz[0] == 2.0

def test_spinbath_mixed_uncertainty():
    # No uncertainty
    s1 = NuclearSpin("C13", x=0.0, y=0.0, z=0.0, w_L=1.0, A_par=0.5, A_perp=0.2)

    # Scalar uncertainty
    s2 = NuclearSpin("N14", x=1.0, y=0.0, z=0.0, w_L=2.0, A_par=0.6, A_perp=0.3,
                 uncertainty=0.1)

    # Tuple/array uncertainty
    s3 = NuclearSpin("H1", x=0.0, y=1.0, z=0.0, w_L=3.0, A_par=0.7, A_perp=0.4,
                 uncertainty=[0.05, 0.15])
    bath = SpinBath([s1, s2, s3])
    df = bath.dataframe

    # Should always include both uncertainty columns
    assert "uncertainty_par" in df.columns
    assert "uncertainty_perp" in df.columns

    # Row 1: NaN
    assert np.isnan(df.loc[0, "uncertainty_par"])
    assert np.isnan(df.loc[0, "uncertainty_perp"])

    # Row 2: scalar replicated
    assert df.loc[1, "uncertainty_par"] == pytest.approx(0.1)
    assert df.loc[1, "uncertainty_perp"] == pytest.approx(0.1)

    # Row 3: tuple split
    assert df.loc[2, "uncertainty_par"] == pytest.approx(0.05)
    assert df.loc[2, "uncertainty_perp"] == pytest.approx(0.15)

    # Types and shapes check
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3  # three spins

def make_spin(x, y=0.0, z=0.0, spin_type="13C", A_par=1.0, A_perp=0.5, w_L=1.0, uncertainty=None):
    """Helper to create a NuclearSpin with defaults overridden."""
    return NuclearSpin(spin_type=spin_type, x=x, y=y, z=z,
                       A_par=A_par, A_perp=A_perp, w_L=w_L, uncertainty=uncertainty)


def test_add_duplicate_spin_raises():
    bath = SpinBath()
    spin1 = make_spin(0.0, 0.0, 0.0, spin_type="13C")
    spin2 = make_spin(0.0, 0.0, 0.0, spin_type="14N")  # same coords, different type

    bath.add_spin(spin1)
    with pytest.raises(ValueError, match="already occupied"):
        bath.add_spin(spin2)

    # even same type duplicates are forbidden
    spin3 = make_spin(0.0, 0.0, 0.0, spin_type="13C")
    with pytest.raises(ValueError, match="already occupied"):
        bath.add_spin(spin3)


def test_update_existing_spin():
    bath = SpinBath()
    spin = make_spin(0.0, 0.0, 0.0, spin_type="13C", A_par=1.0, A_perp=0.5, w_L=1.0)
    bath.add_spin(spin)

    # Update the spin
    updated = bath.update_spin([0.0, 0.0, 0.0], A_par=2.0, A_perp=0.6, w_L=1.5, spin_type="14N")
    assert updated.A_par == 2.0
    assert updated.A_perp == 0.6
    assert updated.w_L == 1.5
    assert updated.spin_type == "14N"

    # Dataframe reflects update
    df = bath.dataframe
    assert df.loc[0, "A_par"] == 2.0
    assert df.loc[0, "A_perp"] == 0.6
    assert df.loc[0, "w_L"] == 1.5
    assert df.loc[0, "type"] == "14N"


def test_update_nonexistent_spin_raises():
    bath = SpinBath()
    # Try to update a spin at a site that does not exist
    with pytest.raises(ValueError, match="No spin found"):
        bath.update_spin([1.0, 1.0, 1.0], A_par=2.0)


def test_mixed_spin_add_and_update():
    bath = SpinBath()
    spin1 = make_spin(0.0, spin_type="13C")
    spin2 = make_spin(1.0, spin_type="14N")
    bath.add_spin(spin1)
    bath.add_spin(spin2)

    # Attempting to add a duplicate at 0.0, 0.0, 0.0 fails
    with pytest.raises(ValueError):
        bath.add_spin(make_spin(0.0, spin_type="14N"))

    # Update spin1
    updated = bath.update_spin([0.0, 0.0, 0.0], A_par=3.0)
    assert updated.A_par == 3.0
