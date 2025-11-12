from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np
import inspect
from .experiments import Experiment
from .forward_models import ForwardModel
from .error import ErrorModel
from .proposals import Proposal, DiscreteRJMCMCProposal


class AcceptProb(ABC):
    """
    Abstract base class for computing acceptance probabilities in trans-dimensional MCMC.

    Parameters
    ----------
    forward_model : ForwardModel instance
        Concrete instance of a ForwardModel subclass (not abstract).
    error_model : ErrorModel
        Concrete instance of an ErrorModel subclass.
    proposal : Proposal
        Concrete instance of a Proposal subclass.
    temperature : float, optional
        Sampling temperature for annealed or tempered MCMC (default = 1.0).
    """

    def __init__(self, forward_model, error_model, proposal, temperature=1.0):
        # --- ForwardModel validation ---
        if not isinstance(forward_model, ForwardModel):
            raise TypeError(
                f"forward_model must be an instance of ForwardModel, got {type(forward_model)}"
            )
        if inspect.isabstract(forward_model.__class__):
            raise TypeError(
                f"forward_model must be a concrete ForwardModel subclass, got abstract {forward_model.__class__}"
            )
        self.forward_model = forward_model  # store the instance

        # --- ErrorModel validation ---
        if not isinstance(error_model, ErrorModel):
            raise TypeError(
                f"error_model must be an instance of ErrorModel, got {type(error_model)}"
            )
        if inspect.isabstract(error_model.__class__):
            raise TypeError(
                f"error_model must be a concrete subclass, got abstract {error_model.__class__}"
            )
        self.error_model = error_model

        # --- Proposal validation ---
        if not isinstance(proposal, Proposal):
            raise TypeError(
                f"proposal must be an instance of Proposal, got {type(proposal)}"
            )
        if inspect.isabstract(proposal.__class__):
            raise TypeError(
                f"proposal must be a concrete subclass, got abstract {proposal.__class__}"
            )
        self.proposal = proposal

        # --- Temperature ---
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"temperature must be a positive float, got {temperature}")
        self.temperature = float(temperature)

    def _check_proposal_validity(self):
        """
        Ensure the proposal object contains a valid proposed state.
        """
        if self.proposal.prop_spins is None or self.proposal.prop_params is None:
            raise ValueError(
                "Proposal must contain valid proposed spins and parameters. "
                "Call `proposal.propose()` before computing acceptance probability."
            )

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

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}("
            f"ForwardModel={self.forward_model.__class__.__name__}, "
            f"ErrorModel={self.error_model.__class__.__name__}, "
            f"Proposal={self.proposal.__class__.__name__}, "
            f"T={self.temperature:.3f})>"
        )


class AcceptProbRJMCMC(AcceptProb):
    """
    Acceptance probability for reversible-jump MCMC birth/death moves.

    Accepts a ForwardModel instance instead of a class.
    Only compatible with DiscreteRJMCMCProposal.
    """

    def __init__(self, forward_model: ForwardModel,
                 error_model: ErrorModel,
                 proposal: 'DiscreteRJMCMCProposal',
                 temperature: float = 1.0):
        if not isinstance(proposal, DiscreteRJMCMCProposal):
            raise TypeError("AcceptProbRJMCMC only works with DiscreteRJMCMCProposal.")

        super().__init__(
            forward_model=forward_model,
            error_model=error_model,
            proposal=proposal,
            temperature=temperature
        )

    def compute(self) -> float:
        """Compute Metropolis-Hastings acceptance probability for RJMCMC birth/death move."""
        self._check_proposal_validity()

        curr_spins = [self.proposal.spin_bath.spins[i] for i in self.proposal.current_spins]
        prop_spins = [self.proposal.spin_bath.spins[i] for i in self.proposal.prop_spins]


        # Compute likelihood/error for current and proposed
        log_L_curr = -self.error_model(curr_spins, self.forward_model)
        log_L_prop = -self.error_model(prop_spins, self.forward_model)

        # RJMCMC birth/death step
        log_a = min(0.0, log_L_prop - log_L_curr)

        # Apply temperature scaling
        if self.temperature != 1.0:
            log_a /= self.temperature

        return float(np.exp(log_a))

