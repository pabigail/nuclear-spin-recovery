"""Parallel tempering: the ladder, the swap rule, and the cold chain."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    ContinuousReflected,
    ParallelTempering,
    ParameterBlock,
    RWMH,
    Schedule,
    State,
    Step,
    Trace,
    geometric_ladder,
)


class Flat:
    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


class SharpLam:
    """Narrow target in lambda: cold rungs are nearly frozen, hot ones are not."""
    def log_prob(self, state, beta=1.0):
        return beta * (-0.5 * ((state.lam[:, 0] - 0.5) / 0.005) ** 2)


def make_state(lam=0.5, n_sites=6, k_max=4, n_exp=1):
    return State.from_sites(
        (0, 2), n_sites=n_sites, n_exp=n_exp,
        lam=np.full((1, n_exp), lam), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1), k_max=k_max)


def lam_schedule(radius=0.05):
    return Schedule([Step(RWMH(ParameterBlock("lam"),
                               ContinuousReflected(radius, 0.0, 1.0)), n_steps=1)])


@pytest.fixture
def pt():
    return ParallelTempering(lam_schedule(), n_replicas=4)


# -------------------------------------------------------------------- ladder

def test_geometric_ladder_starts_at_one():
    """Zero-indexed, so beta_0 = 1 is the cold chain (spec Sec. 11.3)."""
    betas = geometric_ladder(5)
    assert betas[0] == pytest.approx(1.0)


def test_geometric_ladder_halves():
    assert geometric_ladder(4) == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_geometric_ladder_is_descending():
    betas = geometric_ladder(8)
    assert np.all(np.diff(betas) < 0)


def test_default_ladder_matches_replica_count(pt):
    assert len(pt.betas) == 4


def test_custom_ladder_accepted():
    custom = [1.0, 0.6, 0.2]
    ladder = ParallelTempering(lam_schedule(), n_replicas=3, betas=custom)
    assert list(ladder.betas) == pytest.approx(custom)


def test_ladder_must_start_at_one():
    with pytest.raises(ValueError):
        ParallelTempering(lam_schedule(), n_replicas=3, betas=[0.5, 0.25, 0.1])


def test_ladder_must_descend():
    with pytest.raises(ValueError):
        ParallelTempering(lam_schedule(), n_replicas=3, betas=[1.0, 0.1, 0.5])


def test_ladder_length_must_match_replicas():
    with pytest.raises(ValueError):
        ParallelTempering(lam_schedule(), n_replicas=4, betas=[1.0, 0.5])


# ---------------------------------------------------------------- expansion

def test_run_returns_the_cold_chain_only(pt):
    out = pt.run(make_state(), Flat(), np.random.default_rng(0), n_steps=5)
    assert out.n_replicas == 1


def test_run_records_only_the_cold_chain(pt):
    trace = Trace(n_sites=6, k_max=4, n_exp=1)
    pt.run(make_state(), Flat(), np.random.default_rng(0), n_steps=7, trace=trace)
    assert len(trace) == 7
    assert trace.lam.shape == (7, 1)


def test_step_advances_every_rung(pt):
    expanded = make_state().expand_replicas(4)
    out = pt.step(expanded, Flat(), np.random.default_rng(0))
    assert out.n_replicas == 4
    assert len(set(np.round(out.lam[:, 0], 9).tolist())) > 1


def test_single_replica_degenerates_to_the_inner_schedule():
    solo = ParallelTempering(lam_schedule(), n_replicas=1)
    out = solo.run(make_state(), Flat(), np.random.default_rng(0), n_steps=5)
    assert out.n_replicas == 1


# --------------------------------------------------------------- swap rule

def test_swap_permutes_configurations(pt):
    """A swap exchanges rungs; it must not invent or lose a configuration."""
    st = make_state().expand_replicas(4)
    st.lam[:, 0] = [0.1, 0.2, 0.3, 0.4]
    out = pt.attempt_swap(st, Flat(), np.random.default_rng(0))
    assert sorted(np.round(out.lam[:, 0], 9).tolist()) == pytest.approx(
        [0.1, 0.2, 0.3, 0.4])


def test_equal_temperatures_always_swap():
    """With beta_a == beta_b the acceptance ratio is exactly one."""
    flat_ladder = ParallelTempering(lam_schedule(), n_replicas=2, betas=[1.0, 1.0])
    st = make_state().expand_replicas(2)
    st.lam[:, 0] = [0.1, 0.9]
    swapped = 0
    for seed in range(40):
        out = flat_ladder.attempt_swap(st, SharpLam(), np.random.default_rng(seed))
        swapped += out.lam[0, 0] != pytest.approx(0.1)
    assert swapped == 40


def test_swap_uses_the_temperature_gap():
    """log alpha = (beta_a - beta_b)(logL(p_b) - logL(p_a)); a large adverse
    gap must suppress the exchange."""
    ladder = ParallelTempering(lam_schedule(), n_replicas=2, betas=[1.0, 0.001])
    st = make_state().expand_replicas(2)
    st.lam[:, 0] = [0.5, 0.95]        # cold sits at the mode, hot far away
    swapped = 0
    for seed in range(60):
        out = ladder.attempt_swap(st, SharpLam(), np.random.default_rng(seed))
        swapped += out.lam[0, 0] != pytest.approx(0.5)
    assert swapped == 0


def test_favourable_swap_is_accepted():
    """If the hot rung holds the better configuration, the exchange should happen."""
    ladder = ParallelTempering(lam_schedule(), n_replicas=2, betas=[1.0, 0.001])
    st = make_state().expand_replicas(2)
    st.lam[:, 0] = [0.95, 0.5]        # cold far from the mode, hot at it
    out = ladder.attempt_swap(st, SharpLam(), np.random.default_rng(0))
    assert out.lam[0, 0] == pytest.approx(0.5)


# --------------------------------------------------------------- behaviour

def test_hot_rungs_move_more_than_cold(pt):
    """The whole point of the ladder: flattening buys exploration."""
    st = make_state(lam=0.5).expand_replicas(4)
    rng = np.random.default_rng(0)
    start = st.lam[:, 0].copy()
    for _ in range(60):
        st = pt.step(st, SharpLam(), rng)
    travel = np.abs(st.lam[:, 0] - start)
    assert travel[-1] > travel[0]


def test_inner_may_be_a_multi_block_schedule():
    """A rung typically advances several blocks before a swap is attempted."""
    from nuclear_spin_recovery import DiscreteLatticeWalk, NeighborIndex

    line = np.array([[float(i), 0, 0] for i in range(6)])
    inner = Schedule([
        Step(RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0)), 1),
        Step(RWMH(ParameterBlock("sites"), DiscreteLatticeWalk(NeighborIndex(line, 1.5))), 1),
    ])
    ladder = ParallelTempering(inner, n_replicas=3)
    out = ladder.run(make_state(), Flat(), np.random.default_rng(0), n_steps=10)
    assert out.n_replicas == 1
    out.check_invariants()


def test_is_reproducible(pt):
    def chain(seed):
        out = pt.run(make_state(), Flat(), np.random.default_rng(seed), n_steps=20)
        return float(out.lam[0, 0])
    assert chain(3) == pytest.approx(chain(3))


def test_invariants_survive_the_ladder(pt):
    out = pt.run(make_state(), Flat(), np.random.default_rng(0), n_steps=30)
    out.check_invariants()
