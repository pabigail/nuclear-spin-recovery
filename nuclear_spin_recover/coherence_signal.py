from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence
import numpy as np


@dataclass(frozen=True)
class SingleCoherenceSignal:
    """
    Coherence signal of a single experiment.

    Represents the time-dependent coherence of a spin system
    measured or simulated for a single experiment. This class
    enforces that `times` and `coherence` are 1D arrays of the
    same length and converts inputs to float arrays.

    Parameters
    ----------
    times : np.ndarray
        1D array of time points corresponding to the experiment
        (in arbitrary time units). Must be the same length as `coherence`.
    coherence : np.ndarray
        1D array of coherence values corresponding to `times`.
        Typically normalized between 0 and 1. Must have the same
        shape as `times`.

    Raises
    ------
    ValueError
        If `times` or `coherence` are not 1D arrays, or if they
        have different lengths.

    Notes
    -----
    This dataclass is frozen, so its fields are immutable. Both
    `times` and `coherence` arrays are automatically converted
    to NumPy float arrays during initialization.
    """

    times: np.ndarray
    coherence: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        coherence = np.asarray(self.coherence, dtype=float)

        if times.ndim != 1 or coherence.ndim != 1:
            raise ValueError("times and coherence must be 1D arrays")

        if times.shape != coherence.shape:
            raise ValueError("times and coherence must have the same shape")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "coherence", coherence)


class CoherenceSignalList(Sequence[SingleCoherenceSignal]):
    """
    Sequence of coherence signals corresponding to a batch of experiments.

    This class wraps multiple `SingleCoherenceSignal` objects, providing
    sequence-like behavior (`len`, iteration, indexing) while ensuring that
    all elements are valid `SingleCoherenceSignal` instances.

    Parameters
    ----------
    signals : Iterable[SingleCoherenceSignal]
        Iterable of `SingleCoherenceSignal` objects. Typically, each element
        corresponds to one experiment in a `BatchExperiment`.

    Raises
    ------
    ValueError
        If the input iterable is empty.
    TypeError
        If any element of the iterable is not a `SingleCoherenceSignal`.

    Attributes
    ----------
    _signals : tuple of SingleCoherenceSignal
        Internal immutable storage of the coherence signals.

    Methods
    -------
    __len__()
        Return the number of signals in the list.
    __iter__()
        Return an iterator over the signals.
    __getitem__(idx)
        Index into the list to retrieve a `SingleCoherenceSignal`.
    """

    def __init__(self, signals: Iterable[SingleCoherenceSignal]):
        self._signals = tuple(signals)

        if not self._signals:
            raise ValueError("CoherenceSignalList cannot be empty")

        for sig in self._signals:
            if not isinstance(sig, SingleCoherenceSignal):
                raise TypeError("All elements must be SingleCoherenceSignal")

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[SingleCoherenceSignal]:
        return iter(self._signals)

    def __getitem__(self, idx) -> SingleCoherenceSignal:
        return self._signals[idx]
