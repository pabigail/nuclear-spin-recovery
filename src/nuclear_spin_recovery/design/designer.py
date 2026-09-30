"""Rank candidate experiments against a posterior and propose the next one.

:meth:`ExperimentDesigner.propose` picks the best candidate by
:meth:`~ExperimentDesigner.rank`, spends the budget on it with a
:class:`~nuclear_spin_recovery.design.selection.PointSelector`, and returns a
real :class:`~nuclear_spin_recovery.experiment.Experiment` -- points **and**
their weights -- so the result goes straight back into the sampler.

**Candidates are compared at equal time.**  Ranking spreads the same budget
evenly over each candidate's grid before scoring it, so a candidate with more
points carries more noise per point.  A design that wins by measuring more is
not a design (docs/phase-5-plan.md Sec. 6, question 2).

**Where a candidate's envelope comes from.**  Predictions need lambda, the
stretch exponent and sigma, and these are per-experiment and sampled -- lambda
in particular changes with pulse number, since decoupling extends coherence
(spec Sec. 4.2).  A candidate therefore takes, particle by particle, the
envelope sampled for the **measured experiment with the same pulse number and
field**.  A candidate at a pulse number or field never measured has no sampled
envelope and raises: guessing one would design against a decay the posterior
knows nothing about.  The old ``adaptive_exp.py`` had every candidate carry its
own T2 and noise by hand; here they come from the posterior.

**Noise** is the particle-weighted mean of the sampled sigma for the matched
experiment, unless the candidate sets ``sigma`` itself -- which is how to ask
what a longer-averaged repeat would buy.

**What the old data is for.**  The posterior *is* the summary of the old data;
conditioning on it again would double-count.  The old data enters in two
narrow ways only: ``measured`` supplies the envelope and noise, and
``exclude`` stops points already measured from being proposed again.  Joint
design over old and new together is a different calculation and is not this
one.

**A collapsed posterior is reported, not ranked.**  With one particle, or
particles that agree at every candidate point, every candidate is equally
uninformative; ranking them would return an arbitrary winner.  Both methods
raise :class:`~nuclear_spin_recovery.design.selection.NothingToLearn`.  The test
is on the information density, which is deterministic, and not on the
utility's score: where the particles agree to 1e-8 of the noise, predictive
variance was measured at 1.8e-17 but the EIG estimate at -2.7e-10 -- Monte
Carlo noise, of either sign, larger than any threshold that means zero.

See docs/phase-5-plan.md, unit 5d.
"""

from __future__ import annotations


class ExperimentDesigner:
    """Choose the next experiment from a list of candidates.

    ``measured`` is the :class:`~nuclear_spin_recovery.experiment.
    ExperimentSet` the posterior was fitted to; its experiment order is the
    order of the particles' envelope columns.  ``utility`` is a
    :class:`~nuclear_spin_recovery.design.utility.DesignUtility`, ``selector``
    a :class:`~nuclear_spin_recovery.design.selection.PointSelector`; anything
    else raises TypeError.
    """

    def __init__(self, utility, selector, model, site_table, measured):
        raise NotImplementedError

    def rank(self, particles, candidates, *, budget, rng):
        """Utility of each candidate at equal measurement time. (n_candidates,)

        Every candidate is scored in one ``score_many`` call, so common random
        numbers apply across the comparison.  An empty list raises
        ValueError; a list of one is legal.
        """
        raise NotImplementedError

    def propose(self, particles, candidates, *, budget, rng, exclude=None):
        """The next experiment to run.

        Points of ``exclude`` -- usually the measured set -- are removed from
        any candidate with the same pulse number and field before ranking, so
        nothing already measured is proposed again; a candidate left with no
        points is dropped, and if none remain this raises ValueError.

        Returns an Experiment on the chosen candidate's pulse number and field,
        its tau a subset of that candidate's, its weights from the selector
        summing to ``budget``, its sigma the noise it was designed against, and
        no data.
        """
        raise NotImplementedError
