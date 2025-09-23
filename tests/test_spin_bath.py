# tests/test_spin_bath.py

import numpy as np
import pytest

from nuclear_spin_recover import SpinBath, NuclearSpin


def test_nuclearspin_init_and_tensor_diag():
    spin = NuclearSpin(
        spin_type="C13",
        x=0.0, y=0.0, z=0.0,
        A_par=10.0, A_perp=5.0,
        w_L=3.2
    )
    assert spin.spin_type == "C13"
    assert np.allclose(spin.xyz, [0.0, 0.0, 0.0])
    assert np.allclose(np.diag(spin.A), [5.0, 5.0, 10.0])
    assert spin.w_L == 3.2
    assert spin.Q.shape == (3, 3)


def test_nuclearspin_init_full_tensor():
    spin = NuclearSpin(
        spin_type="H1",
        x=1.0, y=2.0, z=3.0,
        A_xx=1.0, A_yy=2.0, A_zz=3.0,
        A_xy=0.1, A_xz=0.2, A_yz=0.3
    )
    assert np.isclose(spin.A[0, 0], 1.0)
    assert np.isclose(spin.A[1, 1], 2.0)
    assert np.isclose(spin.A[2, 2], 3.0)
    assert np.isclose(spin.A[0, 1], 0.1)
    assert np.isclose(spin.A[0, 2], 0.2)
    assert np.isclose(spin.A[1, 2], 0.3)
    assert np.allclose(spin.A, spin.A.T, atol=1e-12)  # should be symmetric


def test_spinbath_add_and_len():
    s1 = NuclearSpin("C13", 0.0, 0.0, 0.0, A_par=10, A_perp=5)
    s2 = NuclearSpin("H1", 1.0, 0.0, 0.0, A_par=20, A_perp=2)
    bath = SpinBath()
    bath.add_spin(s1)
    bath.add_spin(s2)
    assert len(bath) == 2
    assert bath[0].spin_type == "C13"
    assert bath[1].spin_type == "H1"


def test_spinbath_dataframe_and_distances():
    s1 = NuclearSpin("C13", 0.0, 0.0, 0.0, A_par=10, A_perp=5)
    s2 = NuclearSpin("H1", 1.0, 0.0, 0.0, A_par=20, A_perp=2)
    bath = SpinBath([s1, s2])

    df = bath.dataframe
    assert set(df.columns) == {"type", "x", "y", "z", "w_L"}
    assert np.isclose(df.loc[0, "x"], 0.0)
    assert np.isclose(df.loc[1, "x"], 1.0)

    dist_mat = bath.distance_matrix
    assert dist_mat.shape == (2, 2)
    assert np.isclose(dist_mat[0, 1], 1.0)
    assert np.isclose(dist_mat[1, 0], 1.0)
    assert np.isclose(dist_mat[0, 0], 0.0)


def test_spinbath_numpy_conversion():
    s1 = NuclearSpin("C13", 0.0, 0.0, 0.0, A_par=10, A_perp=5)
    s2 = NuclearSpin("H1", 1.0, 0.0, 0.0, A_par=20, A_perp=2)
    bath = SpinBath([s1, s2])

    arr = bath.to_numpy()
    assert arr.dtype.names == ("N", "xyz", "A", "Q")
    assert arr.shape == (2,)
    assert np.allclose(arr[0]["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(arr[1]["xyz"], [1.0, 0.0, 0.0])

