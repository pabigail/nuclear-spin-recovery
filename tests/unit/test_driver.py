"""The schedule and the hybrid driver."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    ContinuousReflected,
    DiscreteLatticeWalk,
    HybridDriver,
    NeighborIndex,
    ParameterBlock,
    RWMH,
    Schedule,
    State,
    Step,
    Trace,
)

LINE = np.array([[float(i), 0, 0] for i in range(6)])


class Flat:
    def log_prob(self, state, beta=1.0):
        return np.zeros(state.n_replicas)


def make_state(n_exp=1):
    return State.from_sites(
        (0, 2), n_sites=6, n_exp=n_exp,
        lam=np.full((1, n_exp), 0.5), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1), k_max=4)


def lam_step(n=3):
    return Step(RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0)), n)


def site_step(n=2):
    return Step(RWMH(ParameterBlock("sites"),
                     DiscreteLatticeWalk(NeighborIndex(LINE, 1.5))), n)


@pytest.fixture
def schedule():
    return Schedule([lam_step(3), site_step(2)])


@pytest.fixture
def driver(schedule):
    return HybridDriver(schedule)


# --------------------------------------------------------------------- Step

def test_step_holds_its_algorithm_and_count():
    algo = RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))
    step = Step(algo, 25)
    assert step.algorithm is algo
    assert step.n_steps == 25


def test_step_rejects_a_non_positive_count():
    algo = RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))
    with pytest.raises(ValueError):
        Step(algo, 0)


# ----------------------------------------------------------------- Schedule

def test_schedule_length_is_the_step_count(schedule):
    assert len(schedule) == 2


def test_schedule_iterates_in_order(schedule):
    blocks = [s.algorithm.block.name for s in schedule]
    assert blocks == ["lam", "sites"]


def test_steps_per_cycle_sums_the_counts(schedule):
    assert schedule.steps_per_cycle == 5


def test_empty_schedule_rejected():
    with pytest.raises(ValueError):
        Schedule([]).steps_per_cycle


# ------------------------------------------------------------------- driver

def test_run_records_the_whole_budget(driver):
    trace = Trace(n_sites=6, k_max=4, n_exp=1)
    driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=20, trace=trace)
    assert len(trace) == 20


def test_run_returns_the_final_state(driver):
    out = driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=10)
    assert isinstance(out, State)
    out.check_invariants()


def test_run_cycles_the_schedule(driver):
    """Two cycles of [lam x3, sites x2] must alternate in that pattern."""
    trace = Trace(n_sites=6, k_max=4, n_exp=1)
    driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=10, trace=trace)
    labels = list(trace.algorithm)
    assert labels[:5] == ["rwmh:lam"] * 3 + ["rwmh:sites"] * 2
    assert labels[5:] == ["rwmh:lam"] * 3 + ["rwmh:sites"] * 2


def test_budget_need_not_divide_the_cycle(driver):
    """A budget that stops mid-cycle stops there rather than overrunning."""
    trace = Trace(n_sites=6, k_max=4, n_exp=1)
    driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=7, trace=trace)
    assert len(trace) == 7
    assert list(trace.algorithm)[-1] == "rwmh:lam"


def test_zero_budget_is_a_no_op(driver):
    st = make_state()
    out = driver.run(st, Flat(), np.random.default_rng(0), n_total=0)
    assert out.lam[0, 0] == pytest.approx(0.5)


def test_state_carries_across_blocks(driver):
    """Each block starts where the previous one stopped -- the chain is one chain."""
    trace = Trace(n_sites=6, k_max=4, n_exp=1)
    driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=30, trace=trace)
    # lambda only changes during lam blocks, so it must be constant within a
    # sites block rather than resetting between blocks
    labels = np.array(list(trace.algorithm))
    lam = trace.lam[:, 0]
    sites_blocks = np.flatnonzero(labels == "rwmh:sites")
    assert np.all(np.isin(lam[sites_blocks], lam))


def test_run_without_a_trace_is_allowed(driver):
    driver.run(make_state(), Flat(), np.random.default_rng(0), n_total=10)


def test_is_reproducible(driver):
    def chain(seed):
        out = driver.run(make_state(), Flat(), np.random.default_rng(seed), n_total=25)
        return float(out.lam[0, 0]), out.site_idx.tolist()
    assert chain(2) == chain(2)


def test_single_block_schedule_matches_the_bare_algorithm():
    """A one-element schedule must not change what the algorithm does."""
    algo = RWMH(ParameterBlock("lam"), ContinuousReflected(0.05, 0.0, 1.0))
    direct = algo.run(make_state(), Flat(), np.random.default_rng(1), n_steps=12)
    viasched = HybridDriver(Schedule([Step(algo, 12)])).run(
        make_state(), Flat(), np.random.default_rng(1), n_total=12)
    assert direct.lam[0, 0] == pytest.approx(viasched.lam[0, 0])
