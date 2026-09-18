"""Site table loading, derivation, filtering, and symmetry."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import SiteTable, lattice, read_hyperfine_table, secular_components

# First data row of nv-2.txt, in file order after the leading index column.
# distance, x, y, z (Angstrom), then A_xx A_yy A_zz A_xy A_xz A_yz (MHz).
ROW1_RAW = dict(
    distance=0.0,
    x=4.35002475133745e-6,
    y=2.60769167081455e-6,
    z=0.00220701949809748,
    A_xx=-2.73941058220000,
    A_yy=-2.73941113586667,
    A_zz=-2.35024128193333,
    A_xy=-6.56735931075692e-7,
    A_xz=-9.01146863938167e-6,
    A_yz=-8.91296312599543e-6,
)

# Second data row: a strongly coupled near neighbour.
ROW2_RAW = dict(
    distance=2.7435414268309173,
    x=-1.32959698127540,
    y=0.767648118123368,
    z=-2.27153444136687,
    A_xx=167.994215580000,
    A_yy=129.279069986667,
    A_zz=117.999170533333,
    A_xy=-33.5279664061995,
    A_xz=22.0676632344587,
    A_yz=-12.7407126660784,
)


def test_column_order_constant():
    assert lattice.IVADY_COLUMNS == (
        "distance", "x", "y", "z",
        "A_xx", "A_yy", "A_zz", "A_xy", "A_xz", "A_yz",
    )


def test_read_table_row_count(nv2_path):
    table = read_hyperfine_table(nv2_path)
    assert len(table["distance"]) == 19924


def test_read_table_golden_row_positions(nv2_path):
    """Pins the column mapping, including the leading index column."""
    table = read_hyperfine_table(nv2_path)
    for key in ("distance", "x", "y", "z"):
        assert table[key][1] == pytest.approx(ROW2_RAW[key], rel=1e-12)


def test_read_table_converts_mhz_to_khz(nv2_path):
    """Tensor components are stored in MHz and must come back as kHz."""
    table = read_hyperfine_table(nv2_path)
    assert table["A_zz"][1] == pytest.approx(ROW2_RAW["A_zz"] * 1000.0, rel=1e-12)
    assert table["A_xz"][1] == pytest.approx(ROW2_RAW["A_xz"] * 1000.0, rel=1e-12)


def test_read_table_does_not_convert_positions(nv2_path):
    """Positions are already Angstrom; only the tensor is rescaled."""
    table = read_hyperfine_table(nv2_path)
    assert table["x"][0] == pytest.approx(ROW1_RAW["x"], rel=1e-12)


def test_secular_components_hand_computed():
    a_par, a_perp = secular_components(a_zz=3.0, a_xz=4.0, a_yz=3.0)
    assert a_par == pytest.approx(3.0)
    assert a_perp == pytest.approx(5.0)


def test_secular_components_perp_is_non_negative():
    _, a_perp = secular_components(a_zz=-10.0, a_xz=-4.0, a_yz=-3.0)
    assert a_perp == pytest.approx(5.0)


def test_secular_components_par_keeps_sign():
    """A_parallel is signed and the sign matters in m_z."""
    a_par, _ = secular_components(a_zz=-7.0, a_xz=0.0, a_yz=0.0)
    assert a_par == pytest.approx(-7.0)


def test_secular_components_vectorized():
    a_par, a_perp = secular_components(
        a_zz=np.array([3.0, -7.0]),
        a_xz=np.array([4.0, 0.0]),
        a_yz=np.array([3.0, 0.0]),
    )
    assert a_par == pytest.approx(np.array([3.0, -7.0]))
    assert a_perp == pytest.approx(np.array([5.0, 0.0]))


def test_from_file_drops_strongly_coupled(nv2_path):
    """Sites with either component above strong_thresh are removed."""
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    assert np.all(np.abs(table.a_par) <= 750.0)
    assert np.all(table.a_perp <= 750.0)


def test_from_file_drops_doubly_weak(nv2_path):
    """A site survives only if at least one component reaches weak_thresh."""
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    reaches = (np.abs(table.a_par) >= 5.0) | (table.a_perp >= 5.0)
    assert np.all(reaches)


def test_strong_filter_uses_or_not_and(nv2_path):
    """A site with huge A_perp and negligible A_par must still be removed.

    Catches an and/or inversion in the strong-coupling filter, which would
    silently admit strongly coupled spins the method is not meant to model.
    """
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=250.0, weak_thresh=5.0)
    assert not np.any(table.a_perp > 250.0)
    assert not np.any(np.abs(table.a_par) > 250.0)


def test_thresholds_are_inclusive(nv2_path):
    """Bounds are <= strong and >= weak, matching the reference loader."""
    loose = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    tight = SiteTable.from_ivady_file(nv2_path, strong_thresh=250.0, weak_thresh=10.0)
    assert len(tight) < len(loose)


def test_tighter_weak_threshold_removes_sites(nv2_path):
    a = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    b = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=50.0)
    assert len(b) < len(a)


def test_distance_is_stored_not_recomputed(nv2_path):
    """Spec Sec. 11.5: the file's distance column is used as given.

    Row 0 has distance 0.0 while |(x,y,z)| is about 0.0022 Angstrom, so a
    recomputing loader is detectable.
    """
    table = read_hyperfine_table(nv2_path)
    assert table["distance"][0] == 0.0
    assert np.linalg.norm([table["x"][0], table["y"][0], table["z"][0]]) > 0.0


def test_isotope_and_gyro_are_populated(nv2_path):
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    assert np.all(table.isotope == "13C")
    assert np.all(table.gyro > 0)
    assert len(table.gyro) == len(table)


def test_positions_shape(nv2_path):
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    assert table.positions.shape == (len(table), 3)


def test_all_arrays_same_length(nv2_path):
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    n = len(table)
    for arr in (table.distance, table.a_par, table.a_perp, table.isotope, table.gyro):
        assert len(arr) == n


def test_impossible_thresholds_raise(nv2_path):
    """Excluding every site is a user error, not an empty success."""
    with pytest.raises(ValueError):
        SiteTable.from_ivady_file(nv2_path, strong_thresh=1.0, weak_thresh=1e9)


def test_symmetry_groups_pair_identical_sites(tiny_site_table):
    labels = tiny_site_table.symmetry_groups(tol=0.1)
    assert labels[0] == labels[1]


def test_symmetry_groups_separate_differing_sites(tiny_site_table):
    labels = tiny_site_table.symmetry_groups(tol=0.1)
    assert labels[0] != labels[2]
    assert labels[0] != labels[3]


def test_symmetry_groups_respect_tolerance(tiny_site_table):
    """A tolerance wide enough to span A_perp 45 vs 10 merges those sites."""
    labels = tiny_site_table.symmetry_groups(tol=50.0)
    assert labels[0] == labels[2]


def test_symmetry_groups_length(tiny_site_table):
    labels = tiny_site_table.symmetry_groups(tol=0.1)
    assert len(labels) == len(tiny_site_table)


def test_len_matches_array_length(tiny_site_table):
    assert len(tiny_site_table) == 4


def test_from_ase_builds_a_table():
    ase_build = pytest.importorskip("ase.build")
    atoms = ase_build.bulk("C", "diamond", a=3.567, cubic=True) * (3, 3, 3)
    table = SiteTable.from_ase(atoms, strong_thresh=1e9, weak_thresh=0.0)
    assert len(table) > 0
    assert table.positions.shape[1] == 3


def test_from_ase_excludes_the_defect_site():
    """The defect itself is not a candidate nuclear site."""
    ase_build = pytest.importorskip("ase.build")
    atoms = ase_build.bulk("C", "diamond", a=3.567, cubic=True) * (2, 2, 2)
    table = SiteTable.from_ase(atoms, strong_thresh=1e9, weak_thresh=0.0)
    assert len(table) == len(atoms) - 1
