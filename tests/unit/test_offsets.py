"""Relaxing the ab initio constraint: per-spin hyperfine offsets.

Spec Sec. 5.3.  A spin's coupling becomes its table value plus a continuous
offset under a Gaussian prior centred on the DFT prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    GaussianOffset,
    ParameterBlock,
    RWMH,
    SiteTable,
    State,
    StretchedExponential,
)


class Flat:
    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


def make_state(table, sites=(0, 2), dA_par=None, dA_perp=None, k_max=4):
    return State.from_sites(
        sites, n_sites=len(table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.1]]), k_max=k_max)


# -------------------------------------------------------------------- state

def test_offsets_default_to_zero(tiny_site_table):
    st = make_state(tiny_site_table)
    assert st.dA_par.shape == (1, 4)
    assert np.all(st.dA_par == 0.0)
    assert np.all(st.dA_perp == 0.0)


def test_zero_offsets_reproduce_the_table(tiny_site_table):
    """With the constraint unrelaxed the model must reduce exactly."""
    st = make_state(tiny_site_table)
    assert st.a_par_per_spin(tiny_site_table)[0, 0] == pytest.approx(
        tiny_site_table.a_par[0])


def test_offset_shifts_the_coupling(tiny_site_table):
    st = make_state(tiny_site_table)
    st.dA_par[0, 0] = 2.5
    st.dA_perp[0, 0] = -1.5
    assert st.a_par_per_spin(tiny_site_table)[0, 0] == pytest.approx(
        tiny_site_table.a_par[0] + 2.5)
    assert st.a_perp_per_spin(tiny_site_table)[0, 0] == pytest.approx(
        tiny_site_table.a_perp[0] - 1.5)


def test_offsets_are_per_slot_not_per_site(tiny_site_table):
    """An offset belongs to the spin, and travels with it when it moves."""
    st = make_state(tiny_site_table, sites=(0, 2))
    st.dA_par[0, 1] = 3.0
    assert st.a_par_per_spin(tiny_site_table)[0, 0] == pytest.approx(
        tiny_site_table.a_par[0])
    assert st.a_par_per_spin(tiny_site_table)[0, 1] == pytest.approx(
        tiny_site_table.a_par[2] + 3.0)


def test_inactive_slots_contribute_nothing(tiny_site_table):
    st = make_state(tiny_site_table, sites=(0,))
    st.dA_par[0, 3] = 99.0
    assert st.a_par_per_spin(tiny_site_table)[0, 1:].tolist() == [0.0, 0.0, 0.0]


def test_copy_carries_offsets(tiny_site_table):
    st = make_state(tiny_site_table)
    st.dA_par[0, 0] = 4.0
    other = st.copy()
    other.dA_par[0, 0] = 9.0
    assert st.dA_par[0, 0] == pytest.approx(4.0)


def test_expand_replicas_carries_offsets(tiny_site_table):
    st = make_state(tiny_site_table)
    st.dA_par[0, 0] = 4.0
    wide = st.expand_replicas(3)
    assert wide.dA_par.shape == (3, 4)
    assert np.all(wide.dA_par[:, 0] == 4.0)


# ----------------------------------------------------------------- proposal

def test_offset_proposal_is_symmetric():
    prop = GaussianOffset(radius=0.2, scale=1.0)
    _, log_ratio = prop.propose(np.random.default_rng(0), 0.0)
    assert log_ratio == pytest.approx(0.0)


def test_offset_prior_peaks_at_zero():
    """The prior is centred on the DFT value, so zero offset is most likely."""
    prop = GaussianOffset(radius=0.2, scale=1.0)
    assert prop.log_prior(0.0) > prop.log_prior(0.5)
    assert prop.log_prior(0.5) > prop.log_prior(2.0)


def test_offset_prior_is_gaussian():
    prop = GaussianOffset(radius=0.2, scale=2.0)
    expected = -0.5 * (1.5 / 2.0) ** 2
    assert prop.log_prior(1.5) - prop.log_prior(0.0) == pytest.approx(expected)


def test_offset_prior_is_symmetric_in_sign():
    prop = GaussianOffset(radius=0.2, scale=1.0)
    assert prop.log_prior(-0.8) == pytest.approx(prop.log_prior(0.8))


def test_other_kernels_have_a_flat_prior():
    """Only the offsets carry a proper prior; the rest enter elsewhere."""
    from nuclear_spin_recovery import ContinuousReflected, DiscreteLatticeWalk
    from nuclear_spin_recovery import NeighborIndex

    assert ContinuousReflected(0.1, 0.0, 1.0).log_prior(0.4) == pytest.approx(0.0)
    line = np.array([[float(i), 0, 0] for i in range(4)])
    assert DiscreteLatticeWalk(NeighborIndex(line, 1.5)).log_prior(2) == pytest.approx(0.0)


def test_scale_bounds_the_offset_domain():
    """Default bounds sit several prior widths out, so reflection is rare."""
    prop = GaussianOffset(radius=0.1, scale=1.0)
    assert prop.upper >= 3.0
    assert prop.lower == pytest.approx(-prop.upper)


# ------------------------------------------------------------------ sampling

def test_offsets_block_updates_only_offsets(tiny_site_table):
    mover = RWMH(ParameterBlock("offsets"), GaussianOffset(0.2, 1.0))
    st = make_state(tiny_site_table)
    out = mover.step(st, Flat(), np.random.default_rng(0))
    assert out.site_idx.tolist() == st.site_idx.tolist()
    assert out.lam[0, 0] == pytest.approx(st.lam[0, 0])


def test_offsets_move_under_a_flat_target(tiny_site_table):
    mover = RWMH(ParameterBlock("offsets"), GaussianOffset(0.2, 1.0))
    st = make_state(tiny_site_table)
    for _ in range(80):
        st = mover.step(st, Flat(), np.random.default_rng(0))
    assert not np.all(st.dA_par[0, :2] == 0.0)


def test_prior_pulls_offsets_toward_zero(tiny_site_table):
    """Under a flat likelihood the posterior over offsets is the prior itself."""
    mover = RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=0.5))
    st = make_state(tiny_site_table)
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(4000):
        st = mover.step(st, Flat(), rng)
        seen.append(float(st.dA_par[0, 0]))
    assert np.mean(seen) == pytest.approx(0.0, abs=0.15)
    assert np.std(seen) == pytest.approx(0.5, rel=0.3)


def test_prior_ratio_enters_acceptance(tiny_site_table):
    """Without the prior term the offsets would random-walk to the bounds."""
    tight = RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=0.2))
    loose = RWMH(ParameterBlock("offsets"), GaussianOffset(0.3, scale=5.0))
    rng = np.random.default_rng(1)

    def spread(mover):
        st = make_state(tiny_site_table)
        vals = []
        for _ in range(3000):
            st = mover.step(st, Flat(), rng)
            vals.append(float(st.dA_par[0, 0]))
        return np.std(vals)

    assert spread(tight) < spread(loose)


# ------------------------------------------------------------ forward model

def test_forward_model_uses_the_offsets(tiny_site_table, single_experiment):
    model = AnalyticCCE1(StretchedExponential())
    st = make_state(tiny_site_table)
    base = model.coherence(st, single_experiment, tiny_site_table)
    st.dA_par[0, 0] = 25.0
    shifted = model.coherence(st, single_experiment, tiny_site_table)
    assert not np.allclose(base, shifted)


def test_zero_offsets_match_the_constrained_model(tiny_site_table, single_experiment):
    """The relaxed model must reduce exactly, not approximately."""
    model = AnalyticCCE1(StretchedExponential())
    st = make_state(tiny_site_table)
    with_zeros = model.coherence(st, single_experiment, tiny_site_table)
    st.dA_par[:] = 0.0
    st.dA_perp[:] = 0.0
    assert model.coherence(st, single_experiment, tiny_site_table) == pytest.approx(
        with_zeros)
