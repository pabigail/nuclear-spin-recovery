"""Algorithm interface and the parameter partition.

Every algorithm updates one named block of parameters and leaves the rest
fixed.  That partition -- p updated, q held -- is the mechanism by which the
algorithms compose into a hybrid sampler.  See spec Sec. 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Parameter blocks an algorithm may update.
BLOCK_NAMES = ("sites", "lam", "n_stretch", "sigma", "offsets")


@dataclass(frozen=True)
class ParameterBlock:
    """Names the parameters an algorithm updates."""

    name: str

    def __post_init__(self):
        if self.name not in BLOCK_NAMES:
            raise ValueError(
                f"unknown parameter block {self.name!r}; known: {list(BLOCK_NAMES)}"
            )

    @property
    def is_discrete(self) -> bool:
        return self.name == "sites"


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
        return beta * self.likelihood.log_prob(
            state, self.expset, self.model, self.site_table
        )


class Algorithm(ABC):
    """One MCMC update rule applied to one parameter block."""

    @abstractmethod
    def step(self, state, target, rng):
        """Advance the chain one step, returning a new State."""

    def run(self, state, target, rng, n_steps, trace=None, beta=1.0):
        """Apply ``n_steps`` steps, recording each to ``trace`` if given."""
        for _ in range(int(n_steps)):
            state = self.step(state, target, rng, beta=beta)
            if trace is not None:
                trace.append(state, target.log_prob(state, beta=1.0), self.label)
        return state

    @property
    def label(self) -> str:
        """Short name recorded alongside each trace entry."""
        block = getattr(self, "block", None)
        name = getattr(block, "name", "?")
        return f"{type(self).__name__.lower()}:{name}"
