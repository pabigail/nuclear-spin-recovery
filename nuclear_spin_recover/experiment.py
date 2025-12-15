import numpy as np
from dataclasses import dataclass

@dataclass
class Experiment:
    experiment_name: str
    tau: np.ndarray
    mag_field: float
    pulses: dict

    def signature(self) -> tuple:
        return (self.experiment_name, tuple(self.tau), self.mag_field, tuple(self.pulses.items()))
