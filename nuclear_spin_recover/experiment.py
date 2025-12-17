import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class SingleExperiment:
    experiment_name: str
    tau: np.ndarray
    mag_field: float
    pulses: dict

    def signature(self) -> tuple:
        return (self.experiment_name, tuple(self.tau), self.mag_field, tuple(self.pulses.items()))


@dataclass
class BatchExperiment:
    experiments: List[SingleExperiment]
