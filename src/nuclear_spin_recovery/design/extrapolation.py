"""An envelope for a pulse number the posterior has not measured.

The designer predicts a candidate with the decay constant the posterior holds
for the measured experiment at the same pulse number and field.  At the start
of an experimental cycle that is a real restriction: after a CPMG-4 survey the
posterior knows lambda at N = 4 and nowhere else, so it could only ever propose
more CPMG-4.

Choosing a new pulse number needs an assumption about how the decay moves with
N, and this module makes it explicit and opt-in.  Under dynamical decoupling
the coherence time in *total* evolution time is commonly modelled as growing
as a power of the pulse number, ``T(N) ~ N**gamma``.  One repetition of CPMG-N
lasts ``2 N tau`` (see :mod:`~nuclear_spin_recovery.design.cost`), so in tau
the decay constant is

    lambda_N = lambda_ref * (N / N_ref) ** (gamma - 1),

which *shrinks* with N whenever gamma < 1: more pulses reach the same total
time at a smaller tau.  ``gamma`` is a property of the sample and its bath and
has no default.

The extrapolation is used only where the posterior has nothing better.  A
pulse number that has been measured uses its own sampled lambda, never the
scaled one, so after a round of measurement the assumption retreats to the
pulse numbers still unmeasured.
"""

from __future__ import annotations

import numpy as np


class DecouplingScaling:
    """Extrapolate the envelope to an unmeasured pulse number.

    :class:`~nuclear_spin_recovery.design.designer.ExperimentDesigner` picks
    the reference: the measured experiment at the same field whose pulse
    number is nearest in log N, the closest in the scaling's own terms.  Its
    stretch exponent and noise carry over unchanged; only lambda scales.
    """

    def __init__(self, gamma):
        gamma = float(gamma)
        if not np.isfinite(gamma):
            raise ValueError(f"gamma must be finite, got {gamma}")
        self.gamma = gamma

    def scale(self, lam_ref, n_ref, n_pulses):
        """lambda at ``n_pulses`` from lambda measured at ``n_ref``."""
        if n_ref <= 0 or n_pulses <= 0:
            raise ValueError("the scaling needs a positive pulse number at both "
                             f"ends, got {n_ref} and {n_pulses}")
        return np.asarray(lam_ref, dtype=float) * (n_pulses / n_ref) ** (
            self.gamma - 1.0)
