import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from .pulse import Pulse, PulseSequence
import numbers

@dataclass
class SingleExperiment:
    experiment_name: str
    tau: np.ndarray
    mag_field: float
    pulses: PulseSequence
    lambda_decoherence: numbers.Real = 1.0

    def __post__init(self):
        if not isinstance(self.mag_field, numbers.Real):
            raise(TypeError, "Magnetic field must be a real number")
        if self.mag_field < 0:
            raise(ValueError, "Magnetic field must be >= 0")
        if not isinstance(self.lambda_decoherence, numbers.Real):
            raise(TypeError, "Lambda decoherence must be a real number")
        if self.lambda_decoherence <= 0 or self.lambda_decoherence > 1.0:
            raise(ValueError, "Lambda decoherence must be in (0.0, 1.0]")


@dataclass
class BatchExperiment:
    experiments: List[SingleExperiment]
