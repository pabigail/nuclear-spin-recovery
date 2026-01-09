from typing import Protocol, Union
from nuclear_spin_recover.coherence_signal import (
    SingleCoherenceSignal,
    CoherenceSignalList,
)
from nuclear_spin_recover.experiment import (
        SingleExperiment,
        BatchExperiment,
        )
from nuclear_spin_recover.nuclear_spin import FullSpinBath

Experiment = Union["SingleExperiment", "BatchExperiment"]


class ForwardModel(Protocol):
    def compute_coherence(
        self,
        experiment: Experiment,
        spin_bath: "FullSpinBath",
    ) -> SingleCoherenceSignal | CoherenceSignalList:
        ...
