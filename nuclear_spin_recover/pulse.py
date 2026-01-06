from dataclasses import dataclass, field
from typing import Literal, List, Iterable
import numbers

Axis = Literal["X", "Y", "Z"]


@dataclass(frozen=False)
class Pulse:
    """
    Single control pulse applied along a Cartesian axis.

    A pulse is defined by an axis of rotation and a rotation angle.
    Pulses are typically combined into a :class:`PulseSequence` to
    construct dynamical decoupling or control protocols.

    Parameters
    ----------
    axis : {"X", "Y", "Z"}
        Axis about which the pulse is applied.
    angle : float
        Rotation angle (in radians).

    Raises
    ------
    ValueError
        If `axis` is not one of ``{"X", "Y", "Z"}`` or if `angle`
        is not a real number.
    """

    axis: Axis
    angle: float

    axis_error_message = "Axis must be 'X', 'Y', or 'Z'"
    angle_error_message = "Angle must be a real number"

    def __post_init__(self):
        if self.axis not in {"X", "Y", "Z"}:
            raise ValueError(self.axis_error_message)

        if not (
            isinstance(self.angle, numbers.Real) and not isinstance(self.angle, bool)
        ):
            raise ValueError(self.angle_error_message)

    def update_angle(self, new_angle: numbers.Real) -> None:
        """
        Update the pulse rotation angle.

        Parameters
        ----------
        new_angle : float
            New rotation angle (in radians).

        Raises
        ------
        ValueError
            If `new_angle` is not a real number.
        """
        if not isinstance(new_angle, numbers.Real):
            raise ValueError(self.angle_error_message)
        self.angle = new_angle

    def update_axis(self, new_axis: Axis) -> None:
        """
        Update the pulse rotation axis.

        Parameters
        ----------
        new_axis : {"X", "Y", "Z"}
            New axis of rotation.

        Raises
        ------
        ValueError
            If `new_axis` is not one of ``{"X", "Y", "Z"}``.
        """
        if new_axis not in {"X", "Y", "Z"}:
            raise ValueError(self.axis_error_message)
        self.axis = new_axis


@dataclass
class PulseSequence:
    pulses: List[Pulse] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.pulses, list):
            raise TypeError("Pulse sequence must be a list of Pulses")
        for pulse in self.pulses:
            if not isinstance(pulse, Pulse):
                raise TypeError("Pulse sequence must be a list of Pulses")

    def append(self, pulse: Pulse) -> "PulseSequence":
        if not isinstance(pulse, Pulse):
            raise TypeError("Can only append Pulse objects")
        self.pulses.append(pulse)

    def delete(self, indices: Iterable[int]) -> None:
        if not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be integers")

        for i in sorted(indices, reverse=True):
            if i < 0 or i >= len(self.pulses):
                raise IndexError("At least one Pulse index is out of range")
            del self.pulses[i]

    def __getitem__(self, idx):
        return self.pulses[idx]

    def __len__(self):
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)

    def get_signature(self, angle_tol=1e-6) -> tuple:
        return tuple((p.axis, round(p.angle / angle_tol)) for p in self.pulses)
