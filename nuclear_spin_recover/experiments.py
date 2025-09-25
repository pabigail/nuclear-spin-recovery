import numpy as np
from collections.abc import Iterable

def is_array_like(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


class Experiment:
    """
    Stores parameters for one or more experiments.
    Each attribute is a numpy array where each entry corresponds to one experiment.
    """
    def __init__(self, num_exps, num_pulses, mag_field, noise, timepoints, T2=None):
        self.num_experiments = num_exps

        self.mag_field = np.array(mag_field) if is_array_like(mag_field) else np.array([mag_field])
        self.noise = np.array(noise) if is_array_like(noise) else np.array([noise])
        self.num_pulses = np.array(num_pulses) if is_array_like(num_pulses) else np.array([num_pulses])

        # ensure list-of-lists structure for timepoints
        if is_array_like(timepoints[0]):
            self.timepoints = [np.array(tp) for tp in timepoints]
        else:
            self.timepoints = np.array([timepoints])

        if T2 is None:
            T2 = [1 for _ in range(len(self.mag_field))]

        self.T2 = np.array(T2) if is_array_like(T2) else np.array([T2])

        # Consistency check
        expected_len = self.num_experiments
        for name, arr in [
            ("mag_field", self.mag_field),
            ("noise", self.noise),
            ("num_pulses", self.num_pulses),
            ("timepoints", self.timepoints),
            ("T2", self.T2),
        ]:
            if len(arr) != expected_len:
                raise ValueError(
                    f"Length mismatch for {name}: expected {expected_len}, got {len(arr)}"
                )

    def __len__(self):
        """Number of experiments stored"""
        return self.num_experiments

    def __getitem__(self, idx):
        """Return a dict of parameters for experiment `idx`"""
        return {
            "mag_field": self.mag_field[idx],
            "noise": self.noise[idx],
            "num_pulses": self.num_pulses[idx],
            "timepoints": self.timepoints[idx],
            "T2": self.T2[idx],
        }

    def __repr__(self):
        return (f"<Experiment with {self.num_experiments} experiments: "
                f"fields={list(self.__dict__.keys())}>")

