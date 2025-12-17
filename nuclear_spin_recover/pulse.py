from dataclasses import dataclass, field
from typing import Literal, List

Axis = Literal["X", "Y", "Z"]

@dataclass(frozen=False)
class Pulse:
    axis: Axis
    angle: float


@dataclass
class PulseSequence:
    pulses: List[Pulse] = field(default_factory=list)
