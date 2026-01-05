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



@dataclass
class BatchExperiment:
    experiments: List[SingleExperiment]
