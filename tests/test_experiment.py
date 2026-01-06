import sys
import numpy as np
import pytest
from nuclear_spin_recover.pulse import Pulse, PulseSequence
from nuclear_spin_recover.experiment import SingleExperiment, BatchExperiment


def test_experiment_imported_instantiated():
    assert "nuclear_spin_recover.experiment" in sys.modules

    tau = [1.0, 2.0, 3.0]
    pulses = PulseSequence([Pulse("X", np.pi)])
    sing_exp = SingleExperiment(
        experiment_name="CPMG", tau=tau, mag_field=10, pulses=pulses, lambda_decoherence=1,
    )
    assert isinstance(sing_exp, SingleExperiment)
    assert sing_exp.experiment_name == "CPMG"

    batch_exp = BatchExperiment([sing_exp])
    assert isinstance(batch_exp, BatchExperiment)


def test_experiment_positive_magfield():

    with pytest.raises(ValueError, match="Magnetic field must be >= 0"):
        SingleExperiment(
            experiment_name="CPMG",
            tau=[1.0, 2.0],
            mag_field=-5.6,
            pulses=PulseSequence([Pulse("X", np.pi)]),
        )

    exp_mag_0 = SingleExperiment(
        experiment_name="CPMG",
        tau=[1.0, 2.2],
        mag_field=0,
        pulses=PulseSequence([Pulse("X", np.pi)]),
    )
    assert exp_mag_0.mag_field == 0

    exp_float_mag_field = SingleExperiment(
        experiment_name="CPMG",
        tau=[1.0, 2.0],
        mag_field=3.2,
        pulses=PulseSequence([Pulse("Y", np.pi)]),
    )
    assert exp_float_mag_field.mag_field == 3.2

    exp_int_mag_field = SingleExperiment(
        experiment_name="CPMG",
        tau=[1.0, 2.0, 3.0],
        mag_field=100,
        pulses=PulseSequence([Pulse("Z", np.pi)]),
    )
    assert exp_int_mag_field.mag_field == 100


@pytest.mark.parametrize("good_lambda", [1e-6, 0.2, 0.999, 1.0])
def test_lambda_decoherence_proper_values(good_lambda):

    pulse_sequence=PulseSequence([Pulse("X", np.pi)])
    num_exps=1
    tau=[0.1, 0.2, 0.3]
    mag_field=100
    exp_name="CPMG"

    exp = SingleExperiment(
            experiment_name=exp_name,
            tau=tau,
            mag_field=mag_field,
            pulses=pulse_sequence,
            lambda_decoherence=good_lambda)

    assert exp.lambda_decoherence == good_lambda

    exp_default = SingleExperiment(
            experiment_name=exp_name,
            tau=tau,
            mag_field=mag_field,
            pulses=pulse_sequence)

    assert exp_default.lambda_decoherence == 1.0


@pytest.mark.parametrize(
        "bad_lambda", 
        [0.0, -0.5, 1.1, 100],
        )
def test_invalid_lambda_decoherence(bad_lambda):
    with pytest.raises(ValueError, match=r"Lambda decoherence must be in \(0\.0, 1\.0\]"):
        SingleExperiment(
                experiment_name="CPMG",
                tau=[0.1],
                mag_field=200,
                pulses=PulseSequence([]),
                lambda_decoherence=bad_lambda)

def test_all_tau_positive():

    exp1 = SingleExperiment(
            experiment_name="CPMG",
            tau=[0.00001],
            mag_field=111,
            pulses=PulseSequence([Pulse("X", np.pi)]),
        )

    assert exp1.tau[0] == 0.00001

    with pytest.raises(ValueError, match="All interpulse spacings must be > 0"):
        SingleExperiment(
                experiment_name="CPMG",
                tau=[0.0],
                mag_field=0.5,
                pulses=PulseSequence([Pulse("Y", np.pi)]),
            )

    with pytest.raises(ValueError, match="All interpulse spacings must be > 0"):
        SingleExperiment(experiment_name="CPMG",
                         tau=[-0.04],
                         mag_field=10,
                         pulses=PulseSequence([Pulse("Z", np.pi)]),
                         )

    with pytest.raises(ValueError, match="All interpulse spacings must be > 0"):
        SingleExperiment(experiment_name="CPMG",
                         tau=[0.02, 0.0, 0.4],
                         mag_field=15,
                         pulses=PulseSequence([Pulse("X", np.pi)]),
                         )

def test_tau_list_real_numbers():

    with pytest.raises(TypeError, match="Tau must be a list of interpulse spacings > 0"):
        SingleExperiment(
                experiment_name="CPMG",
                tau=0.3,
                mag_field=100,
                pulses=PulseSequence([]),
                )

    with pytest.raises(TypeError, match="All interpulse spacings must be real numbers > 0"):
        SingleExperiment(
                experiment_name="CPMG",
                tau=[True],
                mag_field=100,
                pulses=PulseSequence([]),
                )


def test_exp_names_are_strings():
    
    with pytest.raises(ValueError, match="Experiment names must be nonempty strings"):
        SingleExperiment(experiment_name="",
                   tau=[1.0],
                   mag_field=1.0,
                   pulses=PulseSequence([Pulse("X", np.pi)]),
        )

    with pytest.raises(TypeError, match="Experiment names must be nonempty strings"):
        SingleExperiment(experiment_name=0,
                   tau=[1.0],
                   mag_field=100,
                   pulses=PulseSequence([Pulse("X", np.pi)]),
        )

    with pytest.raises(TypeError, match="Experiment names must be nonempty strings"):
        SingleExperiment(experiment_name=True,
                   tau=[0.005],
                   mag_field=200,
                   pulses=PulseSequence([Pulse("Z", np.pi)]),
        )

def test_pulses_proper_objects():

    exp_no_pulses = SingleExperiment(experiment_name="FID",
                               tau=[1.0, 2.0, 3.0],
                               mag_field=311,
                               pulses=PulseSequence([]),
    )
    assert len(exp_no_pulses.pulses) == 0

    experiment_name = "XY8"
    mag_field = 311
    tau = [1.0, 2.0]

    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        SingleExperiment(experiment_name=experiment_name,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=PulseSequence([Pulse("A", np.pi)]),
        )

    with pytest.raises(ValueError, match="Angle must be a real number"):
        SingleExperiment(experiment_name=experiment_name,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=PulseSequence([Pulse("X", "pi")]),
        )

    with pytest.raises(ValueError, match="Angle must be a real number"):
        SingleExperiment(experiment_name=experiment_name,
                   tau=tau,
                   mag_field=mag_field,
                   pulses=PulseSequence([Pulse("X", True)]),
        )

    with pytest.raises(TypeError, match="Pulse sequence must be a list of Pulses"):
        SingleExperiment(experiment_name=experiment_name,
                         mag_field=mag_field,
                         tau=tau,
                         pulses=PulseSequence([1, 2, 3]),
                         )

    with pytest.raises(TypeError, match="Pulses field of SingleExperiment must be a PulseSequence"):
        SingleExperiment(experiment_name=experiment_name,
                         tau=tau,
                         mag_field=mag_field,
                         pulses=Pulse("X", np.pi),
        )

    with pytest.raises(TypeError, match="Pulses field of SingleExperiment must be a PulseSequence"):
        SingleExperiment(experiment_name=experiment_name,
                         tau=tau,
                         mag_field=mag_field,
                         pulses=[Pulse("Z", 0.5*np.pi)],
        )

def test_experiment_signature_deterministic():
    exp = SingleExperiment(experiment_name="CPMG",
                     tau=[1.0, 2.0],
                     mag_field=300,
                           pulses=PulseSequence([Pulse("Z", np.pi), 
                                                 Pulse("X", np.pi)]),
            )

    sig1 = exp.get_signature()
    sig2 = exp.get_signature()

    assert sig1 == sig2


def test_experiment_signature_equal_for_identical_inputs():
    exp1 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=PulseSequence([Pulse("X", np.pi)]),
            )

    exp2 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=PulseSequence([Pulse("X", np.pi)]),
            )

    sig1 = exp1.get_signature()
    sig2 = exp2.get_signature()

    assert sig1 == sig2

def test_experiment_signature_unequal_similar_inputs():
    exp1 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=PulseSequence([Pulse("X", np.pi)]),
            )

    exp2 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.5]],
                      mag_field=[2.0],
                      pulses=PulseSequence([Pulse("X", np.pi)]),
            )

    exp3 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.0]],
                      mag_field=[2.1],
                      pulses=PulseSequence[Pulse("X", np.pi)],
            )

    exp4 = SingleExperiment(experiment_name=["CPMG"],
                      tau=[[1.0]],
                      mag_field=[2.0],
                      pulses=PulseSequence[Pulse("Y", np.pi)],
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
    exp = SingleExperiment(experiment_name=["CPMG"],
                     tau=[[1.0, 1.2]],
                     mag_field=[100],
                     pulses=PulseSequence([Pulse("X", np.pi)]),
            )

    assert exp.tau[0][0] == 1.0
    assert exp.tau[0][1] == 1.2
    assert exp.mag_field[0] == 100
    assert exp.pulses[0].axis == np.pi

    exp.update_mag_field(0, [200])
    assert exp.mag_field[0] == 200

    exp.update_tau(0, [1.1, 1.3])
    assert exp.tau[0][0] == 1.1
    assert exp.tau[0][1] == 1.3

    exp.update_pulses(0, PulseSequence([Pulse("Y", np.pi), 
                                        Pulse("Z", np.pi)]))
    assert exp.pulses[0].angle == np.pi
    assert exp.pulses[0].angle == np.pi

    with pytest.raises(IndexError, match="Update index out of range of the number of experiments"):
        exp.update_mag_field(1, [200])

