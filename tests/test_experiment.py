import sys
import numpy as np
import pytest
from nuclear_spin_recover.experiment import Experiment

def test_experiment_imported_instantiated():
    assert "nuclear_spin_recover.experiment" in sys.modules

    tau = [[1.0, 2.0, 3.0]]
    pulses = [{"X": np.pi}]
    exp = Experiment(experiment_name=["CPMG"], num_exps = 1, tau=tau, mag_field=[10], pulses=pulses)
    assert isinstance(exp, Experiment)
    assert exp.experiment_name == ["CPMG"]


def test_exp_not_empty():
    with pytest.raises(ValueError, match="Number of experiments must be a positive integer"):
        Experiment(experiment_name=[],
                   num_exps=0,
                   tau=[],
                   mag_field=[],
                   pulses=[],
        )

    with pytest.raises(ValueError, match="Number of experiments must be a positive integer"):
        Experiment(experiment_name=[],
                   num_exps=-3,
                   tau=[],
                   mag_field=[],
                   pulses=[],
        )


def test_inconsistencies_between_array_lengths_exp_instantiation():
    experiment_name=["CPMG", "CPMG", "CPMG"]
    tau = [[1.0, 2.0],
           [2.0, 2.5, 3.0],
           [0.2, 4.6]]
    mag_field = [3.0, 2.5, 100]
    pulses = [{"X": np.pi, "Y": np.pi},
              {"Z": 0.5*np.pi},
              {"X": np.pi}]

    with pytest.raises(ValueError, match="Lists of experiments must all be the same length (and equal to the number of experiments"):
        Experiment(experiment_name=experiment_name,
                   num_exps=2,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=pulses)

    with pytest.raises(ValueError, match="Lists of experiments must all be the same length (and equal to the number of experiments"):
        Experiment(experiment_name=experiment_name,
                   num_exps=4,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=pulses)

    with pytest.raises(ValueError, match="Lists of experiments must all be the same length (and equal to the number of experiments"):
        Experiment(experiment_name=experiment_name[0:1],
                   num_exps=2,
                   tau=tau,
                   mag_field=mag_field[0:1],
                   pulses=pulses[0:1]
        )


def test_experiment_positive_magfield():

    with pytest.raises(ValueError, match="Magnetic field must be >= 0"):
        Experiment(experiment_name=["CPMG"], 
                   num_exps = 1,
                   tau=[[1.0, 2.0]], 
                   mag_field = [-5.6], 
                   pulses=[{"X": np.pi}]
        )

    exp_mag_0 = Experiment(experiment_name=["CPMG"],
                           num_exps = 1,
                           tau=[[1.0, 2.2]],
                           mag_field = [0],
                           pulses = [{"X": np.pi}]
                )
    assert exp_mag_0.mag_field[0] == 0

    exp_float_mag_field = Experiment(experiment_name=["CPMG"],
                                     num_exps= 1,
                                     tau=[[1.0, 2.0]],
                                     mag_field = [3.2],
                                     pulses = [{"Y": np.pi}]
                            )
    assert exp_float_mag_field.mag_field[0] == 3.2

    exp_int_mag_field = Experiment(experiment_name=["CPMG"],
                                   tau=[[1.0, 2.0, 3.0]],
                                   mag_field = [100],
                                   pulses = [{"Z": np.pi}]
                        )
    assert exp_int_mag_field.mag_field == 100

    with pytest.raises(ValueError, match="Magnetic field must be >= 0"):
        Experiment(exp_name=["CPMG", "XY8"],
                   tau=[[1.0], [2.0]],
                   mag_field=[0.3, -200],
                   pulses=[{"X": np.pi}, {"Y": 0.5*np.pi}]
        )
                                

def test_num_exps_integer():
    experiment_name = ["CPMG"]
    tau = [[1.0]],
    mag_field = [311]
    pulses = [{"X": np.pi}]

    with pytest.raises(TypeError, match="Number of experiments must be positive integer"):
        Experiment(experiment_name=experiment_name,
                   num_exps=1.0,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=pulses
        )

    with pytest.raises(TypeError, match="Number of experiments must be positive integer"):
        Experiment(experiment_name=experiment_name,
                   num_exps=1.5,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=pulses
        )

    exp = Experiment(experiment_name=experiment_name,
                     num_exps=1,
                     tau=tau,
                     mag_field=mag_field,
                     pulses=pulses
            )

    assert isinstance(exp.num_exps, int)
