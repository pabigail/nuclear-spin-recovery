"""The hybrid driver: a schedule of algorithms, cycled.

The sampler cycles deterministically through user-specified blocks, each naming
an algorithm and a step count, and concatenates their output into one trace.
This is a systematic-scan Metropolis-within-Gibbs composition: each block leaves
the target invariant, so the composition does.  See spec Sec. 8.5.

The schedule is the primary user-facing interface -- which algorithm updates
which parameters is a user decision, not a property of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Step:
    """One block of the schedule: an algorithm and how many steps of it."""

    algorithm: object
    n_steps: int

    def __post_init__(self):
        # TODO(phase-3): require n_steps >= 1 and an Algorithm instance.
        pass


class Schedule:
    """An ordered list of Steps, cycled by the driver."""

    def __init__(self, steps):
        self.steps = list(steps)

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    @property
    def steps_per_cycle(self) -> int:
        """Total sampler steps in one pass through the schedule."""
        raise NotImplementedError


class HybridDriver:
    """Cycles a schedule until the step budget is spent."""

    def __init__(self, schedule):
        self.schedule = schedule

    def run(self, state, target, rng, n_total, trace=None):
        """Advance the chain n_total steps, recording each to ``trace``.

        A budget that does not divide the cycle length stops partway through a
        cycle rather than overrunning.
        """
        raise NotImplementedError
