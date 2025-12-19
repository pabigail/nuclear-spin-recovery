import sys
import numpy as np
import pytest
from nuclear_spin_recover.pulse import Pulse, PulseSequence


def test_pulse_imported_instantiated():
    assert "nuclear_spin_recover.pulse" in sys.modules

    pulse = Pulse("X", np.pi)
    assert isinstance(pulse, Pulse)

    pulse_sequence = PulseSequence([pulse, pulse, pulse])
    assert isinstance(pulse_sequence, PulseSequence)


def test_pulse_initialize():
    with pytest.raises(TypeError):
        Pulse()

    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        Pulse(np.pi, "X")

    with pytest.raises(TypeError):
        Pulse("Z")

    with pytest.raises(TypeError):
        Pulse(np.pi)


def test_pulse_axis():
    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        Pulse("A", np.pi)

    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        Pulse(10, np.pi)

    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        Pulse("XY", 0.5 * np.pi)


def test_pulse_access_fields():
    pulse_x = Pulse("X", np.pi)
    pulse_y = Pulse("Y", 0.5 * np.pi)
    pulse_z = Pulse("Z", -1 * np.pi)

    assert pulse_x.axis == "X"
    assert pulse_y.axis == "Y"
    assert pulse_z.axis == "Z"

    assert pulse_x.angle == np.pi
    assert pulse_y.angle == 0.5 * np.pi
    assert pulse_z.angle == -1 * np.pi


def test_pulse_update_fields():

    pulse = Pulse("X", np.pi)
    assert pulse.axis == "X"
    assert pulse.angle == np.pi

    pulse.update_axis("Y")
    assert pulse.axis == "Y"

    with pytest.raises(ValueError, match="Axis must be 'X', 'Y', or 'Z'"):
        pulse.update_axis("W")

    pulse.update_angle(-0.5 * np.pi)
    assert pulse.angle == -0.5 * np.pi

    with pytest.raises(ValueError, match="Angle must be a real number"):
        pulse.update_angle("Z")


def test_pulse_sequence_empty():
    seq = PulseSequence([])
    assert seq.pulses == []
    assert len(seq.pulses) == 0


def test_pulse_sequence_one():
    pulse = Pulse("X", np.pi)
    seq = PulseSequence([pulse])
    assert isinstance(seq.pulses[0], Pulse)
    assert seq.pulses[0] == pulse
    assert len(seq.pulses) == 1


def test_pulse_sequence_two():
    pulse_1 = Pulse("Y", np.pi)
    pulse_2 = Pulse("Z", 0.5 * np.pi)

    seq = PulseSequence([pulse_1, pulse_2])
    assert len(seq.pulses) == 2
    assert seq.pulses[0].axis == "Y"
    assert seq.pulses[0].angle == np.pi
    assert seq.pulses[1].axis == "Z"
    assert seq.pulses[1].angle == 0.5 * np.pi

    seq.pulses[1].update_angle(np.pi)
    assert seq.pulses[1].angle == np.pi


def test_initialize_pulse_sequence():
    with pytest.raises(TypeError, match="Pulse sequence must be a list of Pulses"):
        PulseSequence([{"X": np.pi}])

    with pytest.raises(TypeError, match="Pulse sequence must be a list of Pulses"):
        pulse = Pulse("Y", np.pi)
        PulseSequence([pulse, {"Z": np.pi}])


def test_update_pulse_sequence():
    pulse_x = Pulse("X", np.pi)
    pulse_y = Pulse("Y", 0.5 * np.pi)
    pulse_z = Pulse("Z", -1 * np.pi)

    seq = PulseSequence([pulse_x, pulse_y])
    assert seq.pulses[0].axis == "X"
    assert seq.pulses[0].angle == np.pi
    assert seq.pulses[1].axis == "Y"
    assert seq.pulses[1].angle == 0.5 * np.pi
    assert len(seq.pulses) == 2

    seq.pulses[0] = pulse_z
    assert seq.pulses[0].axis == "Z"
    assert seq.pulses[0].angle == -1 * np.pi
    assert seq.pulses[1].axis == "Y"
    assert seq.pulses[1].angle == 0.5 * np.pi
    assert len(seq.pulses) == 2


def test_append_pulses():
    pulse_x = Pulse("X", np.pi)
    pulse_y = Pulse("Y", -1 * np.pi)

    seq = PulseSequence([])
    assert len(seq.pulses) == 0

    seq.append(pulse_x)
    assert seq.pulses[0].axis == "X"
    assert seq.pulses[0].angle == np.pi
    assert len(seq.pulses) == 1

    seq.append(pulse_y)
    assert seq.pulses[1].axis == "Y"
    assert seq.pulses[1].angle == -1 * np.pi
    assert len(seq.pulses) == 2

    seq.append(pulse_x)
    assert seq.pulses[2].axis == "X"
    assert seq.pulses[2].angle == np.pi

    with pytest.raises(TypeError, match="Can only append Pulse objects"):
        seq.append({"X": np.pi})

    with pytest.raises(TypeError, match="Can only append Pulse objects"):
        seq.append(100)


def test_delete_single_pulse_from_pulse_seq():

    seq = PulseSequence([Pulse("X", np.pi), Pulse("Y", -1 * np.pi)])
    assert len(seq.pulses) == 2
    assert seq.pulses[0].axis == "X"
    assert seq.pulses[0].angle == np.pi
    assert seq.pulses[1].axis == "Y"
    assert seq.pulses[1].angle == -1 * np.pi

    seq.delete([0])
    assert len(seq.pulses) == 1
    assert seq.pulses[0].axis == "Y"
    assert seq.pulses[0].angle == -1 * np.pi

    seq.delete([0])
    assert len(seq.pulses) == 0


def test_delete_multiple_pulses_from_pulse_seq():

    seq = PulseSequence(
        [
            Pulse("Z", np.pi),
            Pulse("X", -1 * np.pi),
            Pulse("Y", 0.5 * np.pi),
            Pulse("Z", -0.5 * np.pi),
        ]
    )
    assert len(seq.pulses) == 4

    seq.delete([1, 3])
    assert len(seq.pulses) == 2
    assert seq.pulses[0].axis == "Z"
    assert seq.pulses[0].angle == np.pi
    assert seq.pulses[1].axis == "Y"
    assert seq.pulses[1].angle == 0.5 * np.pi

    seq.delete([0, 1])
    assert len(seq.pulses) == 0


def test_delete_indices_out_of_range_invalid():

    seq = PulseSequence([Pulse("X", np.pi), Pulse("Y", -0.5 * np.pi)])
    with pytest.raises(TypeError, match="Indices must be integers"):
        seq.delete(["X"])
    with pytest.raises(TypeError, match="Indices must be integers"):
        seq.delete([np.pi])
    with pytest.raises(IndexError, match="At least one Pulse index is out of range"):
        seq.delete([-1])
    with pytest.raises(IndexError, match="At least one Pulse index is out of range"):
        seq.delete([0, 2])
    with pytest.raises(IndexError, match="At least one Pulse index is out of range"):
        seq.delete([5])


def test_delete_indices_out_of_order():

    seq = PulseSequence(
        [
            Pulse("X", np.pi),
            Pulse("Y", -1 * np.pi),
            Pulse("Z", np.pi),
            Pulse("Z", 0.5 * np.pi),
        ]
    )

    assert len(seq.pulses) == 4

    seq.delete([3, 1])
    assert len(seq.pulses) == 2
    assert seq.pulses[0].axis == "X"
    assert seq.pulses[0].angle == np.pi
    assert seq.pulses[1].axis == "Z"
    assert seq.pulses[1].angle == np.pi


def test_signature_consistent():

    seq_1 = PulseSequence([])
    seq_2 = PulseSequence([])
    assert len(seq_1.pulses) == 0
    assert len(seq_2.pulses) == 0
    assert seq_1.get_signature() == seq_2.get_signature()

    seq_1.append(Pulse("X", np.pi))
    seq_2.append(Pulse("X", np.pi))
    assert len(seq_1.pulses) == 1
    assert len(seq_2.pulses) == 1
    assert seq_1.get_signature() == seq_2.get_signature()


def test_signature_unique():

    seq_1 = PulseSequence([])
    seq_2 = PulseSequence([Pulse("Z", 0.5 * np.pi)])
    assert seq_1.get_signature() != seq_2.get_signature()

    seq_1.append(Pulse("Z", -0.5 * np.pi))
    assert len(seq_1.pulses) == 1
    assert len(seq_2.pulses) == 1
    assert seq_1.get_signature() != seq_2.get_signature()

    seq_3 = PulseSequence([Pulse("X", 0.5 * np.pi)])
    assert seq_3.get_signature() != seq_2.get_signature()

    seq_3.append(Pulse("X", np.pi))
    seq_2.append(Pulse("X", np.pi))
    assert len(seq_3.pulses) == 2
    assert len(seq_2.pulses) == 2
    assert seq_3.get_signature() != seq_2.get_signature()
