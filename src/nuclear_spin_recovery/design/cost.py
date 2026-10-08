"""What a measurement point costs in time.

A design budget is total measurement time, and points are not equally
expensive: one repetition of a CPMG-N sequence at delay tau takes the free
evolution of the sequence plus a fixed per-shot overhead (optical
initialisation, readout, waiting).  At the same tau, CPMG-64 takes sixteen
times the evolution of CPMG-4, so a design that ignored this would favour
long sequences for free.

A cost model maps an :class:`~nuclear_spin_recovery.experiment.Experiment` to
the time of **one unit of weight** at each of its points, ``(n_points,)``.
Selectors and the designer then spend a budget ``sum_j w_j c_j`` -- the
weights keep their meaning as relative repetitions, so the likelihood is
untouched, and only what a weight costs changes.  With no cost model every
point costs 1 and the budget is the sum of the weights, exactly as before.

**The free evolution is 2 N tau, not N tau.**  The forward model is the closed
form of Taminiau *et al.* for the sequence ``(tau - pi - 2 tau - pi - tau)^(N/2)``,
in which tau is *half* the spacing between pi pulses.  Measured on the model
itself rather than assumed: a weakly coupled spin's first dip sits at
``pi / (2 omega_L + A_par)`` -- 0.734 us against 0.729 predicted for
A_par = 20 kHz at 311 G -- and not at twice that, where it would be if tau
were the full spacing.  ``evolution_per_pulse`` is exposed for a lab whose
timing differs, but the default is the one the forward model implies.
"""

from __future__ import annotations

import numpy as np


class SequenceDuration:
    """Time of one repetition of CPMG-N at each delay, in ms.

        c(tau) = overhead + evolution_per_pulse * N * tau

    ``overhead`` is the per-shot time outside the free evolution, in ms --
    initialisation, readout and any wait.  It defaults to zero, which makes the
    cost the free evolution alone, ``2 N tau``.  That is a floor, not an
    estimate: a real shot has overhead, and leaving it out favours many short
    repetitions over few long ones.  Set it from the instrument when it is
    known.  ``evolution_per_pulse`` is 2 for the forward model's tau
    convention; see the module docstring.
    """

    def __init__(self, overhead=0.0, evolution_per_pulse=2.0):
        overhead = float(overhead)
        if not np.isfinite(overhead) or overhead < 0:
            raise ValueError(f"overhead must be non-negative, got {overhead}")
        evolution_per_pulse = float(evolution_per_pulse)
        if not np.isfinite(evolution_per_pulse) or evolution_per_pulse <= 0:
            raise ValueError("evolution_per_pulse must be positive, got "
                             f"{evolution_per_pulse}")
        self.overhead = overhead
        self.evolution_per_pulse = evolution_per_pulse

    def __call__(self, experiment):
        tau = np.asarray(experiment.tau, dtype=float)
        return self.overhead + self.evolution_per_pulse * experiment.n_pulses * tau
