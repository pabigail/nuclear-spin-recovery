"""Algorithm interface and the parameter partition.

Every algorithm updates one named block of parameters and leaves the rest
fixed.  That partition -- p updated, q held -- is the mechanism by which the
algorithms compose into a hybrid sampler.  See spec Sec. 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

#: Parameter blocks an algorithm may update.
BLOCK_NAMES = ("sites", "lam", "n_stretch", "sigma")


@dataclass(frozen=True)
class ParameterBlock:
    """Names the parameters an algorithm updates."""

    name: str

    def __post_init__(self):
        # TODO(phase-2): reject names outside BLOCK_NAMES.  Left permissive so
        # fixtures build; validation is what test_algorithms exercises.
        pass

    @property
    def is_discrete(self) -> bool:
        raise NotImplementedError


class Target:
    """The distribution being sampled.

    Bundles everything the acceptance ratio needs, so an algorithm depends on
    a single object rather than on the forward model, the likelihood and the
    data separately.
    """

    def __init__(self, expset, model, likelihood, site_table):
        self.expset = expset
        self.model = model
        self.likelihood = likelihood
        self.site_table = site_table

    def log_prob(self, state, beta=1.0):
        """Tempered log-likelihood per replica. (n_replicas,)

        ``beta`` is the inverse temperature; beta = 1 is the true posterior.
        """
        raise NotImplementedError


class Algorithm(ABC):
    """One MCMC update rule applied to one parameter block."""

    @abstractmethod
    def step(self, state, target, rng):
        """Advance the chain one step, returning a new State."""

    def run(self, state, target, rng, n_steps, trace=None, beta=1.0):
        """Apply ``n_steps`` steps, recording each to ``trace`` if given."""
        raise NotImplementedError
