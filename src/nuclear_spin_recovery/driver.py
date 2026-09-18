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
        self.n_steps = int(self.n_steps)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")


class Schedule:
    """An ordered list of Steps, cycled by the driver."""

    def __init__(self, steps):
        self.steps = list(steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def steps_per_cycle(self) -> int:
        """Total sampler steps in one pass through the schedule."""
        if not self.steps:
            raise ValueError("schedule is empty")
        return sum(step.n_steps for step in self.steps)


class HybridDriver:
    """Cycles a schedule until the step budget is spent."""

    def __init__(self, schedule):
        self.schedule = schedule

    def run(self, state, target, rng, n_total, trace=None):
        """Advance the chain n_total steps, recording each to ``trace``.

        A budget that does not divide the cycle length stops partway through a
        cycle rather than overrunning.
        """
        remaining = int(n_total)
        if remaining and not self.schedule.steps:
            raise ValueError("schedule is empty")
        while remaining > 0:
            for step in self.schedule:
                if remaining <= 0:
                    break
                take = min(step.n_steps, remaining)
                state = step.algorithm.run(state, target, rng, n_steps=take,
                                           trace=trace)
                remaining -= take
        return state
