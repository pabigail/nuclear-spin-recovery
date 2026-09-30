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

import numpy as np

from ..experiment import Experiment, ExperimentSet
from ..post.residual import predictive_from_arrays
from .selection import _NOTHING, NothingToLearn, PointSelector
from .utility import DesignUtility, information_density

#: Relative tolerance for "the same field" and "the same tau".  Both are
#: recorded values, not computed ones, so anything looser would merge points
#: an experimenter meant as distinct.
_SAME = 1e-9


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
        if not isinstance(utility, DesignUtility):
            raise TypeError(f"{type(utility).__name__} is not a DesignUtility")
        if not isinstance(selector, PointSelector):
            raise TypeError(f"{type(selector).__name__} is not a PointSelector")
        self.utility = utility
        self.selector = selector
        self.model = model
        self.site_table = site_table
        self.measured = measured

    def _matched_experiment(self, cand):
        """Index of the measured experiment whose envelope ``cand`` uses."""
        same_n = [e.n_pulses == cand.n_pulses for e in self.measured.experiments]
        if not any(same_n):
            raise ValueError(
                f"no measured experiment at n_pulses={cand.n_pulses}, so the "
                "posterior holds no envelope for that pulse number")
        for i, e in enumerate(self.measured.experiments):
            if same_n[i] and np.isclose(e.b_z, cand.b_z, rtol=_SAME, atol=0):
                return i
        raise ValueError(
            f"no measured experiment at n_pulses={cand.n_pulses} and "
            f"b_z={cand.b_z} G; the posterior holds no envelope at that field")

    def _prepare(self, particles, candidates, budget):
        """Predictions and unit-weight noise for each candidate."""
        candidates = list(candidates)
        if not candidates:
            raise ValueError("no candidates to rank")
        budget = float(budget)
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        if particles.lam.shape[1] != self.measured.n_experiments:
            raise ValueError(
                f"particles carry envelopes for {particles.lam.shape[1]} "
                f"experiments, the measured set has "
                f"{self.measured.n_experiments}")

        prepared = []
        for cand in candidates:
            e = self._matched_experiment(cand)
            # Rebuilt bare: the candidate's own data or weights, if it has
            # any, are not what is being predicted.
            bare = ExperimentSet([Experiment(tau=cand.tau, n_pulses=cand.n_pulses,
                                             b_z=cand.b_z)])
            P = predictive_from_arrays(
                particles.site_idx, particles.k, particles.dA_par,
                particles.dA_perp, particles.lam[:, [e]],
                particles.n_stretch[:, [e]], particles.sigma[:, [e]],
                bare, self.site_table, self.model, k_max=particles.k_max)
            sigma = (float(cand.sigma) if cand.sigma is not None
                     else float(particles.weight @ particles.sigma[:, e]))
            prepared.append((cand, P, sigma))

        # Decided on density, which is deterministic; an EIG estimate on
        # agreeing particles is Monte Carlo noise of either sign.
        informative = particles.n_particles > 1 and any(
            information_density(P, particles.weight, sigma).max() > _NOTHING
            for _, P, sigma in prepared)
        if not informative:
            raise NothingToLearn(
                "the posterior has collapsed: its particles agree at every "
                "candidate point, so every candidate is equally uninformative")
        return prepared, budget

    def rank(self, particles, candidates, *, budget, rng):
        """Utility of each candidate at equal measurement time. (n_candidates,)

        Every candidate is scored in one ``score_many`` call, so common random
        numbers apply across the comparison.  An empty list raises
        ValueError; a list of one is legal.
        """
        prepared, budget = self._prepare(particles, candidates, budget)
        return self._score(particles, prepared, budget, rng)

    def _score(self, particles, prepared, budget, rng):
        # Equal time: each candidate's points share the whole budget evenly.
        noise = [np.full(P.shape[1], sigma / np.sqrt(budget / P.shape[1]))
                 for _, P, sigma in prepared]
        return np.asarray(self.utility.score_many(
            [P for _, P, _ in prepared], particles.weight, noise, rng),
            dtype=float)

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
        candidates = [self._without_excluded(c, exclude) for c in candidates]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            raise ValueError("every candidate point has already been measured")

        prepared, budget = self._prepare(particles, candidates, budget)
        scores = self._score(particles, prepared, budget, rng)
        cand, P, sigma = prepared[int(np.argmax(scores))]

        idx, weight = self.selector.select(P, particles.weight, sigma, budget,
                                           rng)
        idx = np.asarray(idx, dtype=int)
        order = np.argsort(cand.tau[idx], kind="stable")
        return Experiment(tau=cand.tau[idx][order], n_pulses=cand.n_pulses,
                          b_z=cand.b_z, sigma=sigma,
                          weight=np.asarray(weight, dtype=float)[order])

    @staticmethod
    def _without_excluded(cand, exclude):
        """``cand`` minus points ``exclude`` already holds, or None if empty."""
        if exclude is None:
            return cand
        done = [e.tau for e in exclude.experiments
                if e.n_pulses == cand.n_pulses
                and np.isclose(e.b_z, cand.b_z, rtol=_SAME, atol=0)]
        if not done:
            return cand
        done = np.concatenate(done)
        keep = ~np.any(np.isclose(cand.tau[:, None], done[None, :], rtol=_SAME,
                                  atol=0), axis=1)
        if not keep.any():
            return None
        if keep.all():
            return cand
        return Experiment(tau=cand.tau[keep], n_pulses=cand.n_pulses,
                          b_z=cand.b_z, sigma=cand.sigma)
