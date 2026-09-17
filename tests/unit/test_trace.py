"""Chain recording."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import State, Trace


def make_state(sites=(0, 2), n_sites=4, n_exp=1, k_max=8, lam=3e-3):
    return State.from_sites(
        sites,
        n_sites=n_sites, n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1),
        k_max=k_max,
    )


@pytest.fixture
def trace():
    return Trace(n_sites=4, k_max=8, n_exp=1)


def test_starts_empty(trace):
    assert len(trace) == 0


def test_append_grows(trace):
    trace.append(make_state(), log_prob=-1.0)
    trace.append(make_state(), log_prob=-2.0)
    assert len(trace) == 2


def test_records_configuration(trace):
    trace.append(make_state(sites=(1, 3)), log_prob=-1.0)
    assert trace.k[0] == 2
    assert trace.site_idx[0, :2].tolist() == [1, 3]


def test_records_continuous_parameters(trace):
    trace.append(make_state(lam=5e-3), log_prob=-1.0)
    assert trace.lam[0, 0] == pytest.approx(5e-3)
    assert trace.n_stretch[0, 0] == pytest.approx(1.0)
    assert trace.sigma[0, 0] == pytest.approx(0.1)


def test_records_log_prob(trace):
    trace.append(make_state(), log_prob=-7.5)
    assert trace.log_prob[0] == pytest.approx(-7.5)


def test_records_algorithm_label(trace):
    trace.append(make_state(), log_prob=-1.0, algorithm="rwmh:lam")
    assert trace.algorithm[0] == "rwmh:lam"


def test_append_copies_rather_than_aliases(trace):
    """The sampler mutates states between steps; a trace of views would
    silently rewrite its own history."""
    st = make_state(sites=(0, 2), lam=3e-3)
    trace.append(st, log_prob=-1.0)
    st.site_idx[0, 0] = 3
    st.lam[0, 0] = 99.0
    assert trace.site_idx[0, 0] == 0
    assert trace.lam[0, 0] == pytest.approx(3e-3)


def test_records_only_the_cold_replica(trace):
    """Hot tempering replicas do not target the posterior (spec Sec. 8.4)."""
    st = make_state(lam=3e-3).expand_replicas(4)
    st.lam[0, 0] = 1e-3
    st.lam[1:, 0] = 9e-3
    trace.append(st, log_prob=-1.0)
    assert trace.lam[0, 0] == pytest.approx(1e-3)


def test_array_shapes(trace):
    for _ in range(5):
        trace.append(make_state(), log_prob=-1.0)
    assert trace.site_idx.shape == (5, 8)
    assert trace.k.shape == (5,)
    assert trace.lam.shape == (5, 1)
    assert trace.log_prob.shape == (5,)


def test_multi_experiment_shapes():
    tr = Trace(n_sites=4, k_max=8, n_exp=3)
    tr.append(make_state(n_exp=3), log_prob=-1.0)
    assert tr.lam.shape == (1, 3)
    assert tr.sigma.shape == (1, 3)


def test_discard_burn_in(trace):
    for i in range(10):
        trace.append(make_state(lam=float(i)), log_prob=-float(i))
    kept = trace.discard_burn_in(4)
    assert len(kept) == 6
    assert kept.lam[0, 0] == pytest.approx(4.0)


def test_discard_burn_in_leaves_original_intact(trace):
    for i in range(10):
        trace.append(make_state(), log_prob=-float(i))
    trace.discard_burn_in(4)
    assert len(trace) == 10


def test_discard_all_raises(trace):
    for _ in range(3):
        trace.append(make_state(), log_prob=-1.0)
    with pytest.raises(ValueError):
        trace.discard_burn_in(3)


def test_append_rejects_mismatched_k_max(trace):
    with pytest.raises(ValueError):
        trace.append(make_state(k_max=16), log_prob=-1.0)


def test_append_rejects_mismatched_n_exp(trace):
    with pytest.raises(ValueError):
        trace.append(make_state(n_exp=2), log_prob=-1.0)


def test_empty_bath_is_recordable(trace):
    trace.append(make_state(sites=()), log_prob=-1.0)
    assert trace.k[0] == 0
