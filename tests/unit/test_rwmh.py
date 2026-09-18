"""Random-walk Metropolis-Hastings."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    ContinuousReflected,
    DiscreteLatticeWalk,
    NeighborIndex,
    ParameterBlock,
    RWMH,
    State,
    Trace,
)

LINE = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])


class FlatTarget:
    """Every configuration equally likely."""

    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


class LamTarget:
    """Log-density increasing in lambda, so proposals up are always accepted."""

    def __init__(self, sign=1.0):
        self.sign = sign

    def log_prob(self, state, beta=1.0):
        return beta * self.sign * 1e6 * state.lam[:, 0]


def make_state(sites=(0,), n_sites=4, lam=0.5, k_max=4, n_exp=1):
    return State.from_sites(
        sites, n_sites=n_sites, n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1),
        k_max=k_max,
    )


@pytest.fixture
def lam_mover():
    return RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))


@pytest.fixture
def site_mover():
    return RWMH(ParameterBlock("sites"), DiscreteLatticeWalk(NeighborIndex(LINE, 1.5)))


# ------------------------------------------------------------------- basics

def test_step_returns_a_state(lam_mover):
    out = lam_mover.step(make_state(), FlatTarget(), np.random.default_rng(0))
    assert isinstance(out, State)


def test_step_does_not_mutate_the_input(lam_mover):
    st = make_state(lam=0.5)
    lam_mover.step(st, LamTarget(), np.random.default_rng(0))
    assert st.lam[0, 0] == pytest.approx(0.5)


def test_step_preserves_invariants(site_mover):
    out = site_mover.step(make_state(sites=(0, 2)), FlatTarget(),
                          np.random.default_rng(0))
    out.check_invariants()


def test_continuous_block_leaves_sites_fixed(lam_mover):
    st = make_state(sites=(0, 2))
    out = lam_mover.step(st, FlatTarget(), np.random.default_rng(0))
    assert out.site_idx.tolist() == st.site_idx.tolist()
    assert out.k.tolist() == st.k.tolist()


def test_discrete_block_leaves_continuous_fixed(site_mover):
    st = make_state(sites=(0,), lam=0.5)
    out = site_mover.step(st, FlatTarget(), np.random.default_rng(0))
    assert out.lam[0, 0] == pytest.approx(0.5)
    assert out.sigma[0, 0] == pytest.approx(0.1)


def test_discrete_block_preserves_spin_count(site_mover):
    st = make_state(sites=(0, 2))
    out = site_mover.step(st, FlatTarget(), np.random.default_rng(0))
    assert out.k[0] == 2


def test_discrete_block_keeps_sites_distinct(site_mover):
    st = make_state(sites=(0, 1))
    rng = np.random.default_rng(0)
    for _ in range(50):
        st = site_mover.step(st, FlatTarget(), rng)
        active = st.site_idx[0, : st.k[0]]
        assert len(set(active.tolist())) == len(active)


# -------------------------------------------------------- accept and reject

def test_uphill_proposals_are_always_accepted(lam_mover):
    """A target steeply increasing in lambda drives it to the upper bound."""
    st = make_state(lam=0.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        st = lam_mover.step(st, LamTarget(sign=+1.0), rng)
    assert st.lam[0, 0] == pytest.approx(1.0, abs=0.06)


def test_downhill_proposals_are_rejected(lam_mover):
    """Reversing the target drives lambda the other way."""
    st = make_state(lam=0.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        st = lam_mover.step(st, LamTarget(sign=-1.0), rng)
    assert st.lam[0, 0] == pytest.approx(0.0, abs=0.06)


def test_flat_target_accepts_everything(lam_mover):
    """With no likelihood gradient the chain is the proposal alone."""
    st = make_state(lam=0.5)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(100):
        st = lam_mover.step(st, FlatTarget(), rng)
        seen.add(round(float(st.lam[0, 0]), 9))
    assert len(seen) > 50


def test_proposal_ratio_enters_acceptance():
    """A kernel reporting a large negative log ratio must suppress moves.

    Catches an implementation that computes the ratio but never uses it.
    """
    class Biased(ContinuousReflected):
        def propose(self, rng, current, occupied=None):
            return min(current + 0.01, self.upper), -1e6

    mover = RWMH(ParameterBlock("lam"), Biased(0.05, 0.0, 1.0))
    st = make_state(lam=0.5)
    rng = np.random.default_rng(0)
    for _ in range(100):
        st = mover.step(st, FlatTarget(), rng)
    assert st.lam[0, 0] == pytest.approx(0.5)


def test_beta_tempers_acceptance(lam_mover):
    """At beta = 0 every proposal is accepted regardless of the target."""
    st = make_state(lam=0.5)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(100):
        st = lam_mover.step(st, LamTarget(sign=-1.0), rng, beta=0.0)
        seen.add(round(float(st.lam[0, 0]), 9))
    assert len(seen) > 50


def test_is_reproducible(lam_mover):
    def chain(seed):
        st = make_state(lam=0.5)
        rng = np.random.default_rng(seed)
        for _ in range(30):
            st = lam_mover.step(st, FlatTarget(), rng)
        return float(st.lam[0, 0])

    assert chain(4) == pytest.approx(chain(4))


# ---------------------------------------------------------------- replicas

def test_replicas_advance_independently(lam_mover):
    st = make_state(lam=0.5).expand_replicas(4)
    out = lam_mover.step(st, FlatTarget(), np.random.default_rng(0))
    assert out.n_replicas == 4
    assert len(set(np.round(out.lam[:, 0], 9).tolist())) > 1


def test_replica_shapes_are_preserved(site_mover):
    st = make_state(sites=(0, 2)).expand_replicas(3)
    out = site_mover.step(st, FlatTarget(), np.random.default_rng(0))
    assert out.site_idx.shape == st.site_idx.shape
    assert out.k.shape == st.k.shape


# --------------------------------------------------------------------- run

def test_run_records_every_step(lam_mover):
    trace = Trace(n_sites=4, k_max=4, n_exp=1)
    lam_mover.run(make_state(), FlatTarget(), np.random.default_rng(0),
                  n_steps=25, trace=trace)
    assert len(trace) == 25


def test_run_labels_the_trace(lam_mover):
    trace = Trace(n_sites=4, k_max=4, n_exp=1)
    lam_mover.run(make_state(), FlatTarget(), np.random.default_rng(0),
                  n_steps=5, trace=trace)
    assert all("lam" in label for label in trace.algorithm)


def test_run_returns_the_final_state(lam_mover):
    out = lam_mover.run(make_state(), FlatTarget(), np.random.default_rng(0),
                        n_steps=10)
    assert isinstance(out, State)


def test_run_without_a_trace_is_allowed(lam_mover):
    lam_mover.run(make_state(), FlatTarget(), np.random.default_rng(0), n_steps=5)


def test_run_zero_steps_is_a_no_op(lam_mover):
    st = make_state(lam=0.5)
    out = lam_mover.run(st, FlatTarget(), np.random.default_rng(0), n_steps=0)
    assert out.lam[0, 0] == pytest.approx(0.5)
