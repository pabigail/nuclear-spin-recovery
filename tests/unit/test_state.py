"""State construction, invariants, and replica handling."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import State


def make_state(sites=(0, 2), n_sites=4, n_exp=2, k_max=8):
    return State.from_sites(
        sites,
        n_sites=n_sites,
        n_exp=n_exp,
        lam=np.full((1, n_exp), 1.0),
        n_stretch=np.full((1, n_exp), 1.0),
        sigma=np.full((1, n_exp), 0.1),
        k_max=k_max,
    )


def test_from_sites_sets_k():
    assert make_state(sites=(0, 2)).k[0] == 2


def test_from_sites_packs_active_slots():
    st = make_state(sites=(0, 2))
    assert st.site_idx[0, :2].tolist() == [0, 2]


def test_from_sites_sets_occupancy_bitmap():
    st = make_state(sites=(0, 2), n_sites=4)
    assert st.occupied[0].tolist() == [True, False, True, False]


def test_occupancy_shape_is_n_sites():
    st = make_state(sites=(0,), n_sites=7)
    assert st.occupied.shape == (1, 7)


def test_empty_bath_is_valid():
    """k = 0 is a legitimate model, not a degenerate case."""
    st = make_state(sites=())
    assert st.k[0] == 0
    assert not st.occupied[0].any()
    st.check_invariants()


def test_duplicate_sites_rejected():
    """The occupancy constraint is hard; two spins may not share a site."""
    with pytest.raises(ValueError):
        make_state(sites=(1, 1))


def test_check_invariants_catches_occupancy_mismatch():
    st = make_state(sites=(0, 2))
    st.occupied[0, 1] = True          # claim a site no spin occupies
    with pytest.raises(ValueError):
        st.check_invariants()


def test_check_invariants_catches_k_over_max():
    st = make_state(sites=(0, 2), k_max=8)
    st.k[0] = 9
    with pytest.raises(ValueError):
        st.check_invariants()


def test_check_invariants_passes_on_good_state():
    make_state(sites=(0, 2)).check_invariants()


def test_n_replicas_starts_at_one():
    assert make_state().n_replicas == 1


def test_n_exp_matches_construction():
    assert make_state(n_exp=3).n_exp == 3


def test_per_experiment_array_shapes():
    st = make_state(n_exp=3)
    assert st.lam.shape == (1, 3)
    assert st.sigma.shape == (1, 3)
    assert st.n_stretch.shape == (1, 3)


def test_copy_is_deep():
    st = make_state(sites=(0, 2))
    other = st.copy()
    other.site_idx[0, 0] = 3
    other.lam[0, 0] = 99.0
    assert st.site_idx[0, 0] == 0
    assert st.lam[0, 0] == pytest.approx(1.0)


def test_expand_replicas_gives_identical_copies():
    st = make_state(sites=(0, 2)).expand_replicas(4)
    assert st.n_replicas == 4
    for r in range(1, 4):
        assert st.site_idx[r].tolist() == st.site_idx[0].tolist()
        assert st.k[r] == st.k[0]


def test_expand_replicas_does_not_alias():
    st = make_state(sites=(0, 2)).expand_replicas(3)
    st.site_idx[1, 0] = 3
    assert st.site_idx[0, 0] == 0


def test_collapse_to_cold_returns_replica_zero():
    st = make_state(sites=(0, 2)).expand_replicas(4)
    st.lam[0, 0] = 7.0
    st.lam[1, 0] = 9.0
    cold = st.collapse_to_cold()
    assert cold.n_replicas == 1
    assert cold.lam[0, 0] == pytest.approx(7.0)


def test_gyro_per_spin_follows_the_site(tiny_site_table):
    """Gamma_n is determined by the site, never sampled. Spec Sec. 6."""
    st = make_state(sites=(0, 2), n_sites=len(tiny_site_table))
    gyro = st.gyro_per_spin(tiny_site_table)
    assert gyro[0, 0] == pytest.approx(tiny_site_table.gyro[0])
    assert gyro[0, 1] == pytest.approx(tiny_site_table.gyro[2])


def test_gyro_per_spin_handles_mixed_isotopes():
    """A mixed bath must return a different gyro for each isotope."""
    from nuclear_spin_recovery import SiteTable

    table = SiteTable(
        distance=np.array([1.0, 2.0]),
        positions=np.zeros((2, 3)),
        a_par=np.array([10.0, 10.0]),
        a_perp=np.array([5.0, 5.0]),
        isotope=np.array(["13C", "29Si"]),
        gyro=np.array([6.7283, -5.319]),
    )
    st = make_state(sites=(0, 1), n_sites=2)
    gyro = st.gyro_per_spin(table)
    assert gyro[0, 0] != pytest.approx(gyro[0, 1])


def test_hyperfine_per_spin_follows_the_site(tiny_site_table):
    st = make_state(sites=(0, 2), n_sites=len(tiny_site_table))
    assert st.a_par_per_spin(tiny_site_table)[0, 1] == pytest.approx(
        tiny_site_table.a_par[2]
    )
    assert st.a_perp_per_spin(tiny_site_table)[0, 1] == pytest.approx(
        tiny_site_table.a_perp[2]
    )


def test_site_out_of_range_rejected():
    with pytest.raises((ValueError, IndexError)):
        make_state(sites=(0, 99), n_sites=4)
