from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence
import numpy as np


@dataclass(frozen=True)
class SingleCoherenceSignal:
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
    def __init__(self, signals: Iterable[SingleCoherenceSignal]):
        self._signals = tuple(signals)

        if not self._signals:
            raise ValueError("CoherenceSignalList cannot be empty")

        for sig in self._signals:
            if not isinstance(sig, SingleCoherenceSignal):
                raise TypeError(
                    "All elements must be SingleCoherenceSignal"
                )

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[SingleCoherenceSignal]:
        return iter(self._signals)

    def __getitem__(self, idx) -> SingleCoherenceSignal:
        return self._signals[idx]

