"""Experiment construction and the joint flattening."""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import Experiment, ExperimentSet


def test_len_is_point_count():
    exp = Experiment(tau=np.array([1e-3, 2e-3]), n_pulses=8, b_z=311.0)
    assert len(exp) == 2


def test_n_experiments(two_experiments):
    assert two_experiments.n_experiments == 2


def test_n_points_sums_ragged_grids(two_experiments):
    assert two_experiments.n_points == 8


def test_tau_all_concatenates_in_order(two_experiments):
    expected = np.array([1e-3, 2e-3, 3e-3, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3])
    assert two_experiments.tau_all == pytest.approx(expected)


def test_exp_id_labels_ragged_grids(two_experiments):
    assert two_experiments.exp_id == pytest.approx(np.array([0, 0, 0, 1, 1, 1, 1, 1]))


def test_exp_id_is_integer(two_experiments):
    assert np.issubdtype(two_experiments.exp_id.dtype, np.integer)


def test_n_pulses_gathered_onto_points(two_experiments):
    expected = np.array([8, 8, 8, 16, 16, 16, 16, 16])
    assert two_experiments.n_pulses_per_point == pytest.approx(expected)


def test_b_z_gathered_onto_points(two_experiments):
    assert two_experiments.b_z_per_point == pytest.approx(np.full(8, 311.0))


def test_gathers_differ_when_fields_differ():
    """Two experiments at different fields must not share a gathered B_z."""
    eset = ExperimentSet(
        [
            Experiment(tau=np.array([1e-3]), n_pulses=8, b_z=311.0),
            Experiment(tau=np.array([1e-3]), n_pulses=8, b_z=403.0),
        ]
    )
    assert eset.b_z_per_point == pytest.approx(np.array([311.0, 403.0]))


def test_split_round_trips(two_experiments):
    flat = np.arange(two_experiments.n_points, dtype=float)
    pieces = two_experiments.split(flat)
    assert len(pieces) == 2
    assert pieces[0] == pytest.approx(np.array([0.0, 1.0, 2.0]))
    assert pieces[1] == pytest.approx(np.array([3.0, 4.0, 5.0, 6.0, 7.0]))


def test_split_rejects_wrong_length(two_experiments):
    with pytest.raises(ValueError):
        two_experiments.split(np.zeros(3))


def test_data_all_concatenates():
    eset = ExperimentSet(
        [
            Experiment(tau=np.array([1e-3, 2e-3]), n_pulses=8, b_z=311.0,
                       data=np.array([0.9, 0.8])),
            Experiment(tau=np.array([1e-3]), n_pulses=16, b_z=311.0,
                       data=np.array([0.7])),
        ]
    )
    assert eset.data_all == pytest.approx(np.array([0.9, 0.8, 0.7]))


def test_data_all_raises_when_data_missing(two_experiments):
    """Fixtures carry no data; asking for it should be a clear error."""
    with pytest.raises((ValueError, TypeError)):
        _ = two_experiments.data_all


def test_single_experiment_matches_unflattened(single_experiment):
    exp = single_experiment.experiments[0]
    assert single_experiment.tau_all == pytest.approx(exp.tau)
    assert np.all(single_experiment.exp_id == 0)


def test_rejects_data_length_mismatch():
    with pytest.raises(ValueError):
        Experiment(
            tau=np.array([1e-3, 2e-3]),
            n_pulses=8,
            b_z=311.0,
            data=np.array([0.9]),
        )


def test_rejects_non_positive_tau():
    with pytest.raises(ValueError):
        Experiment(tau=np.array([0.0, 1e-3]), n_pulses=8, b_z=311.0)


def test_rejects_negative_pulse_count():
    with pytest.raises(ValueError):
        Experiment(tau=np.array([1e-3]), n_pulses=-8, b_z=311.0)


def test_rejects_empty_set():
    with pytest.raises(ValueError):
        _ = ExperimentSet([]).n_points
