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
    """
    Protocol for a forward model that computes the coherence signal
    of a nuclear spin bath given an experiment.

    Any concrete implementation of this protocol must implement
    the `compute_coherence` method, which returns a coherence signal
    corresponding to the input experiment and spin bath.

    This allows different forward models (e.g., analytic, PyCCE-based,
    or ML-based) to be used interchangeably as long as they provide
    this method.
    """

    def compute_coherence(
        self,
        experiment: Experiment,
        spin_bath: "FullSpinBath",
    ) -> SingleCoherenceSignal | CoherenceSignalList:
        """
        Compute the coherence signal for a given experiment and spin bath.

        Parameters
        ----------
        experiment : SingleExperiment or BatchExperiment
            The experimental setup to simulate. A `SingleExperiment` returns a
            `SingleCoherenceSignal`, while a `BatchExperiment` returns a
            `CoherenceSignalList` with one signal per sub-experiment.
        spin_bath : FullSpinBath
            The nuclear spin bath environment to simulate, containing all
            relevant spins and interactions.

        Returns
        -------
        SingleCoherenceSignal or CoherenceSignalList
            The coherence signal(s) corresponding to the input experiment.
            - `SingleCoherenceSignal` if the input is a `SingleExperiment`
            - `CoherenceSignalList` if the input is a `BatchExperiment`

        Notes
        -----
        This method must be implemented by any concrete forward model.
        It should be deterministic for identical inputs and raise
        appropriate exceptions for invalid inputs.
        """
        ...
