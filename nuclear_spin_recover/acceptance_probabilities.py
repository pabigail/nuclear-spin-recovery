from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np
from .experiments import Experiment
from .forward_models import ForwardModel
from .error import ErrorModel
from .proposals import Proposal


class AcceptProb(ABC):
    """
    Abstract base class for computing acceptance probabilities in trans-dimensional MCMC.

    Parameters
    ----------
    forward_model : type
        Concrete subclass of ForwardModel (not abstract).
    experiment : Experiment
        Experiment containing observed data and conditions.
    error_model : ErrorModel
        Concrete instance of an ErrorModel subclass.
    proposal : Proposal
        Concrete instance of a Proposal subclass.
    """

    def __init__(self, forward_model, experiment, error_model, proposal):
        # Experiment can be any instance of Experiment
        if not isinstance(experiment, Experiment):
            raise TypeError(f"experiment must be an Experiment instance, got {type(experiment)}")
        self.experiment = experiment

        # Must be concrete subclass of ForwardModel
        if not inspect.isclass(forward_model) or not issubclass(forward_model, ForwardModel):
            raise TypeError(f"forward_model must be a subclass of ForwardModel, got {forward_model}")
        if inspect.isabstract(forward_model):
            raise TypeError(f"forward_model must be a concrete subclass, got abstract class {forward_model}")
        self.forward_model_cls = forward_model

        # Must be concrete instance of ErrorModel
        if not isinstance(error_model, ErrorModel):
            raise TypeError(f"error_model must be an instance of ErrorModel, got {type(error_model)}")
        if inspect.isabstract(error_model.__class__):
            raise TypeError(f"error_model must be a concrete subclass, got abstract class {error_model.__class__}")
        self.error_model = error_model

        # Must be concrete instance of Proposal
        if not isinstance(proposal, Proposal):
            raise TypeError(f"proposal must be an instance of Proposal, got {type(proposal)}")
        if inspect.isabstract(proposal.__class__):
            raise TypeError(f"proposal must be a concrete subclass, got abstract class {proposal.__class__}")
        self.proposal = proposal

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the acceptance probability for the current proposed move.

        Returns
        -------
        float
            Acceptance probability between 0 and 1.
        """
        pass
