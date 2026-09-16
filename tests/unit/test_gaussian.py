"""Gaussian L2 likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    GaussianL2,
    State,
    StretchedExponential,
)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def lik():
    return GaussianL2()


def _state(table, n_exp=1, sigma=0.1, sites=(0,)):
    return State.from_sites(
        sites,
        n_sites=len(table),
        n_exp=n_exp,
        lam=np.full((1, n_exp), 3e-3),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma),
        k_max=8,
    )


def _fitted_set(model, state, table, tau, n_pulses=16, b_z=311.0):
    """An experiment whose data is exactly the model prediction."""
    eset = ExperimentSet([Experiment(tau=tau, n_pulses=n_pulses, b_z=b_z)])
    pred = model.coherence(state, eset, table)[0]
    return ExperimentSet(
        [Experiment(tau=tau, n_pulses=n_pulses, b_z=b_z, data=pred)]
    )


def test_exact_match_is_zero(model, lik, tiny_site_table):
    """The spec's form is unnormalized: a perfect fit gives exactly zero."""
    st = _state(tiny_site_table)
    tau = np.linspace(1e-4, 8e-3, 20)
    eset = _fitted_set(model, st, tiny_site_table, tau)
    assert lik.log_prob(st, eset, model, tiny_site_table)[0] == pytest.approx(0.0)


def test_mismatch_is_negative(model, lik, tiny_site_table):
    st = _state(tiny_site_table)
    tau = np.linspace(1e-4, 8e-3, 20)
    eset = _fitted_set(model, st, tiny_site_table, tau)
    eset.experiments[0].data = eset.experiments[0].data + 0.05
    assert lik.log_prob(st, eset, model, tiny_site_table)[0] < 0.0


def test_halving_sigma_quadruples_penalty(model, lik, tiny_site_table):
    tau = np.linspace(1e-4, 8e-3, 20)
    st = _state(tiny_site_table, sigma=0.1)
    eset = _fitted_set(model, st, tiny_site_table, tau)
    eset.experiments[0].data = eset.experiments[0].data + 0.05

    wide = lik.log_prob(st, eset, model, tiny_site_table)[0]
    st_narrow = _state(tiny_site_table, sigma=0.05)
    narrow = lik.log_prob(st_narrow, eset, model, tiny_site_table)[0]
    assert narrow == pytest.approx(4.0 * wide, rel=1e-9)


def test_returns_one_value_per_replica(model, lik, tiny_site_table):
    tau = np.linspace(1e-4, 8e-3, 10)
    st = _state(tiny_site_table)
    eset = _fitted_set(model, st, tiny_site_table, tau)
    expanded = st.expand_replicas(4)
    assert lik.log_prob(expanded, eset, model, tiny_site_table).shape == (4,)


def test_joint_equals_sum_of_parts(model, lik, tiny_site_table):
    """The joint log-likelihood is additive across experiments."""
    tau_a = np.linspace(1e-4, 8e-3, 12)
    tau_b = np.linspace(1e-4, 6e-3, 7)
    st = _state(tiny_site_table, n_exp=2)

    probe = ExperimentSet([
        Experiment(tau=tau_a, n_pulses=8, b_z=311.0),
        Experiment(tau=tau_b, n_pulses=16, b_z=311.0),
    ])
    pred = model.coherence(st, probe, tiny_site_table)[0]
    pieces = probe.split(pred)

    joint = ExperimentSet([
        Experiment(tau=tau_a, n_pulses=8, b_z=311.0, data=pieces[0] + 0.02),
        Experiment(tau=tau_b, n_pulses=16, b_z=311.0, data=pieces[1] - 0.03),
    ])
    total = lik.log_prob(st, joint, model, tiny_site_table)[0]

    st_a = _state(tiny_site_table, n_exp=1)
    st_b = _state(tiny_site_table, n_exp=1)
    only_a = ExperimentSet([joint.experiments[0]])
    only_b = ExperimentSet([joint.experiments[1]])
    part = (
        lik.log_prob(st_a, only_a, model, tiny_site_table)[0]
        + lik.log_prob(st_b, only_b, model, tiny_site_table)[0]
    )
    assert total == pytest.approx(part, rel=1e-9)


def test_per_experiment_sigma_is_applied(model, lik, tiny_site_table):
    """A noisier experiment must be down-weighted relative to a cleaner one."""
    tau = np.linspace(1e-4, 8e-3, 12)
    st = _state(tiny_site_table, n_exp=2)
    probe = ExperimentSet([
        Experiment(tau=tau, n_pulses=8, b_z=311.0),
        Experiment(tau=tau, n_pulses=8, b_z=311.0),
    ])
    pred = model.coherence(st, probe, tiny_site_table)[0]
    pieces = probe.split(pred)
    eset = ExperimentSet([
        Experiment(tau=tau, n_pulses=8, b_z=311.0, data=pieces[0] + 0.05),
        Experiment(tau=tau, n_pulses=8, b_z=311.0, data=pieces[1] + 0.05),
    ])
    st.sigma[0, 0] = 0.1
    st.sigma[0, 1] = 0.2
    total = lik.log_prob(st, eset, model, tiny_site_table)[0]

    st_equal = _state(tiny_site_table, n_exp=2, sigma=0.1)
    equal = lik.log_prob(st_equal, eset, model, tiny_site_table)[0]
    assert total > equal      # the wider sigma penalizes its residual less


def test_is_a_likelihood_subclass():
    from nuclear_spin_recovery import Likelihood

    assert isinstance(GaussianL2(), Likelihood)
