"""Algorithm interface, parameter blocks, and the target wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    Algorithm,
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    GaussianL2,
    ParameterBlock,
    State,
    StretchedExponential,
    Target,
)
from nuclear_spin_recovery.algorithms import base


def test_algorithm_is_abstract():
    with pytest.raises(TypeError):
        Algorithm()


def test_block_names_are_the_state_arrays():
    assert set(base.BLOCK_NAMES) == {"sites", "lam", "n_stretch", "sigma"}


@pytest.mark.parametrize("name", ["sites", "lam", "n_stretch", "sigma"])
def test_valid_block_names_accepted(name):
    assert ParameterBlock(name).name == name


def test_unknown_block_name_rejected():
    with pytest.raises(ValueError):
        ParameterBlock("temperature")


def test_sites_block_is_discrete():
    assert ParameterBlock("sites").is_discrete


@pytest.mark.parametrize("name", ["lam", "n_stretch", "sigma"])
def test_other_blocks_are_continuous(name):
    assert not ParameterBlock(name).is_discrete


def test_blocks_compare_by_name():
    assert ParameterBlock("lam") == ParameterBlock("lam")
    assert ParameterBlock("lam") != ParameterBlock("sigma")


# ------------------------------------------------------------------ Target

@pytest.fixture
def target(tiny_site_table):
    tau = np.linspace(1e-4, 8e-3, 20)
    model = AnalyticCCE1(StretchedExponential())
    probe = ExperimentSet([Experiment(tau=tau, n_pulses=16, b_z=311.0)])
    st = _state(tiny_site_table)
    pred = model.coherence(st, probe, tiny_site_table)[0]
    fitted = ExperimentSet(
        [Experiment(tau=tau, n_pulses=16, b_z=311.0, data=pred)]
    )
    return Target(fitted, model, GaussianL2(), tiny_site_table)


def _state(table, sites=(0,), lam=3e-3, n_exp=1):
    return State.from_sites(
        sites, n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), 0.1),
        k_max=8,
    )


def test_target_matches_the_likelihood(target, tiny_site_table):
    st = _state(tiny_site_table)
    direct = GaussianL2().log_prob(st, target.expset, target.model, tiny_site_table)
    assert target.log_prob(st) == pytest.approx(direct)


def test_target_returns_one_value_per_replica(target, tiny_site_table):
    st = _state(tiny_site_table).expand_replicas(3)
    assert target.log_prob(st).shape == (3,)


def test_target_tempering_scales_log_prob(target, tiny_site_table):
    """A tempered target is the log-likelihood times beta (spec Sec. 8.4)."""
    st = _state(tiny_site_table, sites=(2,))
    cold = target.log_prob(st, beta=1.0)[0]
    hot = target.log_prob(st, beta=0.25)[0]
    assert hot == pytest.approx(0.25 * cold)


def test_target_beta_one_is_the_default(target, tiny_site_table):
    st = _state(tiny_site_table, sites=(2,))
    assert target.log_prob(st)[0] == pytest.approx(target.log_prob(st, beta=1.0)[0])
