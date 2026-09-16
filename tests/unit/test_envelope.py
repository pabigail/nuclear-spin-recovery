"""Stretched-exponential decoherence envelope."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import StretchedExponential


@pytest.fixture
def env():
    return StretchedExponential()


def test_n_equals_one_is_plain_exponential(env):
    tau = np.array([[1e-3, 2e-3]])
    exp_id = np.array([0, 0])
    lam = np.array([[2e-3]])
    n = np.array([[1.0]])
    assert env(tau, exp_id, lam, n) == pytest.approx(np.exp(-tau / 2e-3))


def test_n_equals_two_is_gaussian(env):
    tau = np.array([[1e-3, 2e-3]])
    exp_id = np.array([0, 0])
    lam = np.array([[2e-3]])
    n = np.array([[2.0]])
    assert env(tau, exp_id, lam, n) == pytest.approx(np.exp(-((tau / 2e-3) ** 2)))


def test_value_at_zero_is_one(env):
    got = env(np.array([[0.0]]), np.array([0]), np.array([[1e-3]]), np.array([[1.0]]))
    assert got == pytest.approx(1.0)


def test_monotonically_decreasing(env):
    tau = np.linspace(0.0, 8e-3, 25)[None, :]
    exp_id = np.zeros(25, dtype=int)
    got = env(tau, exp_id, np.array([[3e-3]]), np.array([[1.0]]))[0]
    assert np.all(np.diff(got) < 0)


def test_bounded_in_unit_interval(env):
    tau = np.linspace(0.0, 8e-3, 25)[None, :]
    exp_id = np.zeros(25, dtype=int)
    got = env(tau, exp_id, np.array([[1e-3]]), np.array([[1.5]]))
    assert np.all(got >= 0.0) and np.all(got <= 1.0)


def test_per_experiment_lam_lands_on_right_points(env):
    """Each experiment's lambda must apply only to its own points."""
    tau = np.array([[1e-3, 1e-3]])
    exp_id = np.array([0, 1])
    lam = np.array([[1e-3, 1e9]])      # second experiment barely decays
    n = np.array([[1.0, 1.0]])
    got = env(tau, exp_id, lam, n)[0]
    assert got[0] == pytest.approx(np.exp(-1.0))
    assert got[1] == pytest.approx(1.0, abs=1e-6)


def test_per_experiment_n_lands_on_right_points(env):
    tau = np.array([[2e-3, 2e-3]])
    exp_id = np.array([0, 1])
    lam = np.array([[2e-3, 2e-3]])
    n = np.array([[1.0, 2.0]])
    got = env(tau, exp_id, lam, n)[0]
    assert got[0] == pytest.approx(np.exp(-1.0))
    assert got[1] == pytest.approx(np.exp(-1.0))   # tau == lam, both give e^-1


def test_global_n_broadcasts(env):
    """A scalar exponent applies to every experiment."""
    tau = np.array([[1e-3, 2e-3]])
    exp_id = np.array([0, 1])
    lam = np.array([[2e-3, 4e-3]])
    got_scalar = env(tau, exp_id, lam, np.array([[1.0]]))
    got_explicit = env(tau, exp_id, lam, np.array([[1.0, 1.0]]))
    assert got_scalar == pytest.approx(got_explicit)


def test_replica_axis_is_preserved(env):
    tau = np.array([1e-3, 2e-3])[None, :]
    exp_id = np.array([0, 0])
    lam = np.array([[2e-3], [4e-3]])
    n = np.array([[1.0], [1.0]])
    got = env(np.repeat(tau, 2, axis=0), exp_id, lam, n)
    assert got.shape == (2, 2)
    assert got[0, 0] != pytest.approx(got[1, 0])


def test_is_an_envelope_subclass():
    from nuclear_spin_recovery import Envelope

    assert isinstance(StretchedExponential(), Envelope)
