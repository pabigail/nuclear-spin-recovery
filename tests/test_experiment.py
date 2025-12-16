import sys
import numpy as np
import pytest
from nuclear_spin_recover.experiment import Experiment


def test_experiment_imported_instantiated():
    assert "nuclear_spin_recover.experiment" in sys.modules

    tau = [[1.0, 2.0, 3.0]]
    pulses = [{"X": np.pi}]
    exp = Experiment(
        experiment_name=["CPMG"], num_exps=1, tau=tau, mag_field=[10], pulses=pulses
    )
    assert isinstance(exp, Experiment)
    assert exp.experiment_name == ["CPMG"]


def test_exp_not_empty():
    with pytest.raises(
        ValueError, match="Number of experiments must be a positive integer"
    ):
        Experiment(
            experiment_name=[],
            num_exps=0,
            tau=[],
            mag_field=[],
            pulses=[],
        )

    with pytest.raises(
        ValueError, match="Number of experiments must be a positive integer"
    ):
        Experiment(
            experiment_name=[],
            num_exps=-3,
            tau=[],
            mag_field=[],
            pulses=[],
        )


def test_inconsistencies_between_array_lengths_exp_instantiation():
    experiment_name = ["CPMG", "CPMG", "CPMG"]
    tau = [[1.0, 2.0], [2.0, 2.5, 3.0], [0.2, 4.6]]
    mag_field = [3.0, 2.5, 100]
    pulses = [{"X": np.pi, "Y": np.pi}, {"Z": 0.5 * np.pi}, {"X": np.pi}]

    with pytest.raises(
        ValueError,
        match="Lists of experiments must all be the same length (and equal to the number of experiments",
    ):
        Experiment(
            experiment_name=experiment_name,
            num_exps=2,
            tau=tau,
            mag_field=mag_field,
            pulses=pulses,
        )

    with pytest.raises(
        ValueError,
        match="Lists of experiments must all be the same length (and equal to the number of experiments",
    ):
        Experiment(
            experiment_name=experiment_name,
            num_exps=4,
            tau=tau,
            mag_field=mag_field,
            pulses=pulses,
        )

    with pytest.raises(
        ValueError,
        match="Lists of experiments must all be the same length (and equal to the number of experiments",
    ):
        Experiment(
            experiment_name=experiment_name[0:1],
            num_exps=2,
            tau=tau,
            mag_field=mag_field[0:1],
            pulses=pulses[0:1],
        )


def test_experiment_positive_magfield():

    with pytest.raises(ValueError, match="Magnetic field must be >= 0"):
        Experiment(
            experiment_name=["CPMG"],
            num_exps=1,
            tau=[[1.0, 2.0]],
            mag_field=[-5.6],
            pulses=[{"X": np.pi}],
        )

    exp_mag_0 = Experiment(
        experiment_name=["CPMG"],
        num_exps=1,
        tau=[[1.0, 2.2]],
        mag_field=[0],
        pulses=[{"X": np.pi}],
    )
    assert exp_mag_0.mag_field[0] == 0

    exp_float_mag_field = Experiment(
        experiment_name=["CPMG"],
        num_exps=1,
        tau=[[1.0, 2.0]],
        mag_field=[3.2],
        pulses=[{"Y": np.pi}],
    )
    assert exp_float_mag_field.mag_field[0] == 3.2

    exp_int_mag_field = Experiment(
        experiment_name=["CPMG"],
        tau=[[1.0, 2.0, 3.0]],
        mag_field=[100],
        pulses=[{"Z": np.pi}],
    )
    assert exp_int_mag_field.mag_field == 100

    with pytest.raises(ValueError, match="Magnetic field must be >= 0"):
        Experiment(
            exp_name=["CPMG", "XY8"],
            tau=[[1.0], [2.0]],
            mag_field=[0.3, -200],
            pulses=[{"X": np.pi}, {"Y": 0.5 * np.pi}],
        )


def test_num_exps_integer():
    experiment_name = ["CPMG"]
    tau = ([[1.0]],)
    mag_field = [311]
    pulses = [{"X": np.pi}]

    with pytest.raises(
        TypeError, match="Number of experiments must be positive integer"
    ):
        Experiment(
            experiment_name=experiment_name,
            num_exps=1.0,
            tau=tau,
            mag_field=mag_field,
            pulses=pulses,
        )

    with pytest.raises(
        TypeError, match="Number of experiments must be positive integer"
    ):
        Experiment(
            experiment_name=experiment_name,
            num_exps=1.5,
            tau=tau,
            mag_field=mag_field,
            pulses=pulses,
        )

    exp = Experiment(
        experiment_name=experiment_name,
        num_exps=1,
        tau=tau,
        mag_field=mag_field,
        pulses=pulses,
    )

    assert isinstance(exp.num_exps, int)


def test_all_tau_positive():
    experiment_name = ["CPMG", "XY8"]
    num_exps = 2
    mag_field = [300, 500]
    pulses = [{"X": np.pi}, {"Y": np.pi}]

    with pytest.raises(ValueError, match="Interpulse times must be > 0"):
        Experiment(experiment_name=experiment_name,
                   num_exps=num_exps,
                   tau=[[0.0], [1.0]],
                   mag_field=mag_field,
                   pulses=pulses,
        )

    with pytest.raises(ValueError, match="Interpulse times must be > 0"):
        Experiment(experiment_name=experiment_name,
                   num_exps=num_exps,
                   tau=[[1.0], [0.0]],
                   mag_field=mag_field,
                   pulses=pulses,
        )

    with pytest.raises(ValueError, match="Interpulse times must be > 0"):
        Experiment(experiment_name=[experiment_name[0]],
                   num_exps=1,
                   tau=[[-0.1]],
                   mag_field=[mag_field[0]],
                   pulses=[pulses[0]],
        )

    exp = Experiment(
            experiment_name=["CPMG"],
            num_exps=1,
            tau=[[0.00001]],
            mag_field=[111],
            pulses=[pulses[0]],
        )

    assert exp.tau[[0]] == 0.0001

def test_exp_names_are_strings():
    
    with pytest.raises(TypeError, match="Experiment names must be nonempty strings"):
        Experiment(experiment_name=[""],
                   num_exps=1,
                   tau=[[1.0]],
                   mag_field=[1.0],
                   pulses=[{"X": np.pi}],
        )

    with pytest.raises(TypeError, match="Experiment names must be nonempty strings"):
        Experiment(experiment_name=[0],
                   num_exps=1,
                   tau=[[1.0]],
                   mag_field=[100],
                   pulses=[{"X": np.pi}],
        )

    with pytest.raises(TypeError, match="Experiment names must be nonempty strings"):
        Experiment(experiment_name=[True],
                   num_exps=1,
                   tau=[[0.005]],
                   mag_field=[200],
                   pulses=[{"Z": np.pi}],
        )

def test_pulse_dicts_proper_format():

    exp_no_pulses = Experiment(experiment_name=["FID"],
                               num_exps=1,
                               tau=[[1.0, 2.0, 3.0]],
                               mag_field=[311],
                               pulses=[{}],
    )
    assert isinstance(exp_no_pulses.pulses[0], dict)
    assert len(exp_no_pulses.pulses[0]) == 0

    experiment_name = ["XY8"]
    mag_field = [311]
    tau = [[1.0, 2.0]]

    with pytest.raises(ValueError, match="Invalid pulse axis"):
        Experiment(experiment_name=experiment_name,
                   num_exps=1,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=[{"A": np.pi}],
        )

    with pytest.raises(ValueError, match="Pulse must be a single number (float) in radians"):
        Experiment(experiment_name=experiment_name,
                   num_exps=1,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=[{"X": "pi"}],
        )

    with pytest.raises(ValueError, match="Pulse must be a single number (float) in radians"):
        Experiment(experiment_name=experiment_name,
                   num_exps=1,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=[{"X": True}],
        )

def test_experiment_signature_deterministic():
    exp = Experiment(experiment_name=["CPMG"],
                     num_exps=1,
                     tau=[[1.0, 2.0]],
                     mag_field=[300],
                     pulses=[{"Z": np.pi}],
            )

    sig1 = exp.get_signature()
    sig2 = exp.get_signature()

    assert sig1 == sig2


def test_experiment_signature_equal_for_identical_inputs():
    exp1 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=[{"X": np.pi}],
            )

    exp2 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=[{"X": np.pi}],
            )

    sig1 = exp1.get_signature()
    sig2 = exp2.get_signature()

    assert sig1 == sig2

def test_experiment_signature_unequal_similar_inputs():
    exp1 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=[{"X": np.pi}],
            )

    exp2 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.5]],
                      mag_field=[2.0],
                      pulses=[{"X": np.pi}],
            )

    exp3 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.0]],
                      mag_field=[2.1],
                      pulses=[{"X": np.pi}],
            )

    exp4 = Experiment(experiment_name=["CPMG"],
                      num_exps=1,
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=[{"Y": np.pi}],
            )

    sig1 = exp1.get_signature()
    sig2 = exp2.get_signature()
    sig3 = exp3.get_signature()
    sig4 = exp4.get_signature()

    assert sig1 != sig2
    assert sig1 != sig3
    assert sig1 != sig4
    assert sig2 != sig3
    assert sig2 != sig4
    assert sig3 != sig4


def test_update_experiment_fields():
    exp = Experiment(experiment_name=["CPMG"],
                     num_experiments=1,
                     tau=[[1.0, 1.2]],
                     mag_field=[100],
                     pulses=[{"X": np.pi}],
            )

    assert exp.num_experiments == 1
    assert exp.tau[0][0] == 1.0
    assert exp.tau[0][1] == 1.2
    assert exp.mag_field[0] == 100
    assert exp.pulses[0]["X"] == np.pi

    exp.update_mag_field(0, [200])
    assert exp.mag_field[0] == 200

    exp.update_tau(0, [1.1, 1.3])
    assert exp.tau[0][0] == 1.1
    assert exp.tau[0][1] == 1.3

    exp.update_pulses(0, {"Y": np.pi, "Z": np.pi})
    assert exp.pulses[0]["Y"] == np.pi
    assert exp.pulses[0]["Z"] == np.pi

    with pytest.raises(IndexError, match="Update index beyond number of experiments"):
        exp.update_mag_field(1, [200])

