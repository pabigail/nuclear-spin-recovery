import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Any, Iterable
from .pulse import Pulse, PulseSequence
import numbers
import hashlib
import json


@dataclass
class SingleExperiment:
    """
    A class representing a single experiment 

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., 'CPMG', 'XY8'). Must be non-empty.
    tau : list of float
        List of interpulse spacings (in arbitrary time units). All values must be real and > 0.
    mag_field : float
        Magnetic field strength. Must be a real number >= 0.
    pulses : PulseSequence
        Pulse sequence object containing one or more pulses.
    lambda_decoherence : float, optional
        Decoherence factor, must be in the interval (0.0, 1.0]. Default is 1.0.

    Raises
    ------
    TypeError
        If any input has the wrong type (e.g., `tau` is not a list, `mag_field` is not a number).
    ValueError
        If any numeric constraints are violated (negative magnetic field, non-positive tau, empty experiment name, etc.).
    """
    experiment_name: str
    tau: np.ndarray
    mag_field: float
    pulses: PulseSequence
    lambda_decoherence: numbers.Real = 1.0

    def __post_init__(self):
        if not isinstance(self.mag_field, numbers.Real):
            raise TypeError("Magnetic field must be a real number")
        if self.mag_field < 0:
            raise ValueError("Magnetic field must be >= 0")
        if not isinstance(self.lambda_decoherence, numbers.Real):
            raise TypeError("Lambda decoherence must be a real number")
        if self.lambda_decoherence <= 0 or self.lambda_decoherence > 1.0:
            raise ValueError("Lambda decoherence must be in (0.0, 1.0]")
        if not isinstance(self.tau, list):
            raise TypeError("Tau must be a list of interpulse spacings > 0")
        if not all(
            isinstance(t, numbers.Real) and not isinstance(t, bool) for t in self.tau
        ):
            raise TypeError("All interpulse spacings must be real numbers > 0")
        if not all(t > 0 for t in self.tau):
            raise ValueError("All interpulse spacings must be > 0")
        if not isinstance(self.experiment_name, str):
            raise TypeError("Experiment names must be nonempty strings")
        if self.experiment_name == "":
            raise ValueError("Experiment names must be nonempty strings")
        if not isinstance(self.pulses, PulseSequence):
            raise TypeError("Pulses field of SingleExperiment must be a PulseSequence")

    def update_mag_field(self, new_mag_field: float) -> None:
        """
        Update the magnetic field value.

        Parameters
        ----------
        new_mag_field : float
            New magnetic field value. Must be a real number >= 0.

        Raises
        ------
        TypeError
            If `new_mag_field` is not a real number.
        ValueError
            If `new_mag_field` is negative.
        """
        if not isinstance(new_mag_field, numbers.Real) or isinstance(
            new_mag_field, bool
        ):
            raise TypeError("Magnetic field must be a real number")
        if new_mag_field < 0:
            raise ValueError("Magnetic field must be >= 0")
        self.mag_field = new_mag_field

    def update_tau(self, new_tau: list) -> None:
        """
        Update the interpulse spacings.

        Parameters
        ----------
        new_tau : list of float
            New list of interpulse spacings. All values must be real and > 0.

        Raises
        ------
        TypeError
            If `new_tau` is not a list of real numbers.
        ValueError
            If any value in `new_tau` is <= 0.
        """
        if not isinstance(new_tau, list):
            raise TypeError("Tau must be a list of real numbers > 0")
        if not all(
            isinstance(t, numbers.Real) and not isinstance(t, bool) for t in new_tau
        ):
            raise TypeError("All interpulse spacings must be real numbers > 0")
        if not all(t > 0 for t in new_tau):
            raise ValueError("All interpulse spacings must be > 0")
        self.tau = new_tau

    def update_pulses(self, new_pulses: "PulseSequence") -> None:
        """
        Update the pulse sequence.

        Parameters
        ----------
        new_pulses : PulseSequence
            New pulse sequence object.

        Raises
        ------
        TypeError
            If `new_pulses` is not a PulseSequence.
        """
        if not isinstance(new_pulses, PulseSequence):
            raise TypeError("Pulses field must be a PulseSequence")
        self.pulses = new_pulses

    def get_signature(self) -> str:
        """
        Return a deterministic signature string for this experiment.

        Converts all relevant fields to JSON-compatible structures,
        then hashes them to produce a fixed-length string.

        Returns
        -------
        str
            SHA256 hash of the JSON representation of the experiment fields.
        """
        # Convert pulses to list of dicts
        pulses_list = [{"axis": p.axis, "angle": float(p.angle)} for p in self.pulses]

        # Build a serializable dict
        sig_dict = {
            "experiment_name": self.experiment_name,
            "tau": [float(t) for t in self.tau],
            "mag_field": float(self.mag_field),
            "lambda_decoherence": float(self.lambda_decoherence),
            "pulses": pulses_list,
        }

        # Convert to JSON string with sorted keys for determinism
        sig_json = json.dumps(sig_dict, sort_keys=True, separators=(",", ":"))

        # Return a hash (SHA256) to make signature compact
        return hashlib.sha256(sig_json.encode("utf-8")).hexdigest()



@dataclass
class BatchExperiment:
    """
    A batch of SingleExperiment objects.

    Parameters
    ----------
    experiments : list of SingleExperiment
        List of experiments in the batch.

    Methods
    -------
    append(experiment)
        Append a SingleExperiment to the batch.
    delete(indices)
        Delete experiments at specified indices.
    get_signature()
        Return a deterministic SHA256 hash string representing the batch.
    """

    experiments: List[SingleExperiment] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.experiments, list):
            raise TypeError("Experiments must be provided as a list")
        for exp in self.experiments:
            if not isinstance(exp, SingleExperiment):
                raise TypeError("All elements must be SingleExperiment instances")

    def append(self, experiment: SingleExperiment) -> "BatchExperiment":
        """Append a SingleExperiment to the batch."""
        if not isinstance(experiment, SingleExperiment):
            raise TypeError("Can only append SingleExperiment objects")
        self.experiments.append(experiment)
        return self

    def delete(self, indices: Iterable[int]) -> None:
        """Delete experiments at the specified indices."""
        if not all(isinstance(i, int) for i in indices):
            raise TypeError("Indices must be integers")
        for i in sorted(indices, reverse=True):
            if i < 0 or i >= len(self.experiments):
                raise IndexError("At least one experiment index is out of range")
            del self.experiments[i]

    def __getitem__(self, idx):
        return self.experiments[idx]

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        return iter(self.experiments)

    def get_signature(self) -> str:
        """
        Return a deterministic SHA256 hash string for the batch of experiments.

        Each SingleExperiment's signature is obtained via its `get_signature()`
        method. The sequence of signatures is converted to a JSON string and
        then hashed to produce a single compact, deterministic hash for the batch.

        Returns
        -------
        str
            SHA256 hash string representing the batch of experiments.
        """
        sig_list = [exp.get_signature() for exp in self.experiments]
        sig_json = json.dumps(sig_list, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(sig_json.encode("utf-8")).hexdigest()
