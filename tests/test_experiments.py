import pytest
import numpy as np
from nuclear_spin_recover import Experiment  # adjust import path as needed


def test_single_experiment_initialization():
    exp = Experiment(
        num_exps=1,
        num_pulses=4,
        mag_field=100,
        noise=0.01,
        timepoints=[0, 1, 2],
        T2=0.5
    )

    assert len(exp) == 1
    assert np.all(exp.mag_field == [100])
    assert np.all(exp.num_pulses == [4])
    assert np.allclose(exp.noise, [0.01])
    assert np.all(exp.timepoints[0] == np.array([0, 1, 2]))
    assert np.allclose(exp.T2, [0.5])


def test_multiple_experiment_initialization():
    exp = Experiment(
        num_exps=2,
        num_pulses=[4, 8],
        mag_field=[100, 200],
        noise=[0.01, 0.02],
        timepoints=[[0, 1, 2], [0, 1, 2]],
        T2=[0.5, 0.7]
    )

    assert len(exp) == 2
    assert np.all(exp.mag_field == [100, 200])
    assert np.all(exp.num_pulses == [4, 8])
    assert np.allclose(exp.noise, [0.01, 0.02])
    assert np.all(exp.timepoints[1] == np.array([0, 1, 2]))
    assert np.allclose(exp.T2, [0.5, 0.7])


def test_getitem_returns_dict():
    exp = Experiment(
        num_exps=2,
        num_pulses=[4, 8],
        mag_field=[100, 200],
        noise=[0.01, 0.02],
        timepoints=[[1, 2, 3], [4, 5, 6]],
        T2=[0.5, 0.7]
    )
    first = exp[0]
    assert isinstance(first, dict)
    assert first["mag_field"] == 100
    assert first["num_pulses"] == 4
    assert np.allclose(first["timepoints"], [1, 2, 3])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch for noise"):
        Experiment(
            num_exps=2,
            num_pulses=[4, 8],
            mag_field=[100, 200],
            noise=[0.01],  # wrong length
            timepoints=[[0, 1, 2], [0, 1, 2]],
            T2=[0.5, 0.7]
        )


def test_default_T2():
    exp = Experiment(
        num_exps=2,
        num_pulses=[4, 8],
        mag_field=[100, 200],
        noise=[0.01, 0.02],
        timepoints=[[0, 1, 2], [0, 1, 2]],
        T2=None
    )
    assert np.all(exp.T2 == [1, 1])

