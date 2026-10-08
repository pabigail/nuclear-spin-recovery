"""Choose the next experiment: a pulse number and its delays, together.

:meth:`ExperimentDesigner.propose` takes the posterior as a weighted set of
baths and a list of candidate experiments, and returns **one experiment** --
a pulse number, the delays to measure at, and the weight of each -- or none,
when nothing on the list can tell the baths apart.

**Every candidate gets its own best delays.**  The baths are simulated on each
candidate's grid, and a :class:`~nuclear_spin_recovery.design.selection.
PointSelector` spends the budget on the delays where they disagree most per
unit of time.  The finished designs are then compared by a
:class:`~nuclear_spin_recovery.design.utility.DesignUtility`, by default the
expected information gain, and the best is returned.  The pulse number and the
delays are therefore chosen together: a candidate is judged on what it does
when measured well, not on its whole grid measured evenly.

**Candidates are compared at equal time.**  ``budget`` is a total measurement
time, and every candidate's design spends exactly that:
``sum(weight * cost) == budget``.  A design that wins by measuring for longer
is not a design.

**Time is always charged.**  ``cost`` maps a candidate to the time of one unit
of weight at each of its points, and defaults to
:class:`~nuclear_spin_recovery.design.cost.SequenceDuration` with no overhead:
one repetition of CPMG-N at delay tau takes ``2 N tau``.  At the same tau a
repetition at N = 64 takes sixteen times as long as at N = 4, so within one
budget it is repeated a sixteenth as often and is four times as noisy.  A long
sequence wins only if what it resolves is worth the time.

**Nothing to tell apart is an answer.**  If the best candidate's expected gain
is below ``min_gain``, :meth:`~ExperimentDesigner.propose` returns a
:class:`DesignResult` whose ``experiment`` is None.  That covers a posterior
that has collapsed to one bath, baths that agree at every candidate point, and
baths that differ by far less than the noise the budget can buy.  The
threshold is in nats and is not zero: the gain is a Monte Carlo estimate and
scatters around its true value, so a design that gains nothing can score a few
thousandths of a nat, of either sign.

**Where a candidate's envelope comes from.**  Predictions need lambda, the
stretch exponent and sigma, and these are per-experiment and sampled -- lambda
in particular changes with pulse number, since decoupling extends coherence
(spec Sec. 4.2).  A candidate therefore takes, particle by particle, the
envelope sampled for the **measured experiment with the same pulse number and
field**.  A candidate at a pulse number or field never measured has no sampled
envelope and raises: guessing one would design against a decay the posterior
knows nothing about -- unless the designer is given ``envelope``, a
:class:`~nuclear_spin_recovery.design.extrapolation.DecouplingScaling`.  Then
an unmeasured pulse number takes lambda scaled from the measured experiment at
the same field nearest in log N.  A measured pulse number always uses its own
sampled lambda; the scaling only fills gaps.

**The design is about the spins, not the envelope.**  The envelope -- lambda
and the stretch exponent -- is a nuisance parameter: it has to be there to
predict a signal, but nothing physical is learned from it.  By default every
particle is therefore predicted with **one shared envelope**, the
particle-weighted posterior mean for the matched experiment.  Hypotheses then
differ only where their spins differ, and expected information gain is
information about the bath alone.  With per-particle envelopes instead, two
baths that happen to carry different lambdas disagree wherever the envelope
dominates -- late tau above all -- and the design pays to learn lambda.
``shared_envelope=False`` restores the per-particle form.

**Noise** is the particle-weighted mean of the sampled sigma for the matched
experiment, unless the candidate sets ``sigma`` itself -- which is how to ask
what a longer-averaged repeat would buy.

**What the old data is for.**  The posterior *is* the summary of the old data;
conditioning on it again would double-count.  The old data enters in two
narrow ways only: ``measured`` supplies the envelope and noise, and
``exclude`` stops points already measured from being proposed again.

**Common random numbers.**  Every candidate's design is scored in one call,
with the same simulated baths and the same noise draws, and a delay keeps its
draw wherever it sits on its candidate's grid.  Candidates are then compared
on their designs and not on their luck.

See docs/phase-5-plan.md, unit 5d, and notebooks/adaptive_design_prototype.py,
where this procedure was worked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..experiment import Experiment, ExperimentSet
from ..post.residual import predictive_from_arrays
from .cost import SequenceDuration
from .selection import InformationDensity, NothingToLearn, PointSelector
from .utility import DesignUtility, ExpectedInformationGain, information_density

#: Relative tolerance for "the same field" and "the same tau".  Both are
#: recorded values, not computed ones, so anything looser would merge points
#: an experimenter meant as distinct.
_SAME = 1e-9

#: Default threshold on the expected information gain, in nats, below which no
#: experiment is proposed.  For scale, ruling out half of a set of equally
#: likely baths gains log 2 = 0.69.
MIN_GAIN = 0.05


@dataclass
class CandidateDesign:
    """What one candidate would be, measured as well as the budget allows.

    ``candidate`` is the experiment as offered, after any excluded points were
    removed; ``signals`` the simulated signal of every bath on its whole grid,
    ``(n_baths, n_grid)``; ``density`` the information density there at unit
    weight, and ``cost`` the time of one unit of weight.  ``index`` picks the
    delays the design measures out of that grid, and ``weight`` is the weight
    of each.  A candidate on which the baths agree everywhere has no design:
    ``index`` is empty and ``gain`` is zero.
    """

    candidate: Experiment
    signals: np.ndarray
    density: np.ndarray
    cost: np.ndarray
    sigma: float
    index: np.ndarray
    weight: np.ndarray
    gain: float = 0.0
    chosen: bool = False

    @property
    def n_pulses(self) -> int:
        return self.candidate.n_pulses

    @property
    def tau(self):
        """The delays the design measures. (n_points,)"""
        return self.candidate.tau[self.index]

    @property
    def time(self):
        """Measurement time spent at each of those delays. (n_points,)"""
        return self.weight * self.cost[self.index]

    @property
    def rate(self):
        """Information density per unit time, over the whole grid. (n_grid,)"""
        return self.density / self.cost

    def experiment(self):
        """The design as an :class:`Experiment`: delays, weights, design noise."""
        return Experiment(tau=self.tau, n_pulses=self.candidate.n_pulses,
                          b_z=self.candidate.b_z, sigma=self.sigma,
                          weight=self.weight)


@dataclass
class DesignResult:
    """The answer to "what should be measured next?".

    ``experiment`` is the one experiment to run, or None when no candidate's
    best design is expected to gain ``min_gain``.  ``designs`` holds the design
    worked out for every candidate, in the order they were offered, so the
    comparison behind the answer can be inspected without recomputing it.
    """

    experiment: Experiment | None
    designs: list
    budget: float
    min_gain: float

    @property
    def distinguishable(self) -> bool:
        """Whether some candidate can tell the baths apart."""
        return self.experiment is not None

    @property
    def chosen(self):
        """The :class:`CandidateDesign` behind ``experiment``, or None."""
        return next((d for d in self.designs if d.chosen), None)

    @property
    def gain(self) -> float:
        """Expected information gain of the best design, in nats."""
        return max((d.gain for d in self.designs), default=0.0)

    def __str__(self):
        header = (f"{'pulses':>6s} {'points':>7s} {'delays (us)':>15s} "
                  f"{'gain (nats)':>12s}")
        lines = [header]
        for d in self.designs:
            span = (f"{d.tau.min() * 1e3:.2f} - {d.tau.max() * 1e3:.2f}"
                    if d.tau.size else "none")
            lines.append(f"{d.n_pulses:6d} {d.tau.size:7d} {span:>15s} "
                         f"{d.gain:12.3f}{'   <- chosen' if d.chosen else ''}")
        if self.experiment is None:
            lines.append(
                f"\nNo experiment on this list can tell the baths apart: the "
                f"best gains {self.gain:.3f} nats, under the threshold of "
                f"{self.min_gain} nats.")
        else:
            e = self.experiment
            lines.append(
                f"\nRun CPMG-{e.n_pulses} at {len(e)} delays between "
                f"{e.tau.min() * 1e3:.2f} and {e.tau.max() * 1e3:.2f} us.")
        return "\n".join(lines)


class ExperimentDesigner:
    """Choose the next experiment from a list of candidates.

    ``measured`` is the :class:`~nuclear_spin_recovery.experiment.
    ExperimentSet` the posterior was fitted to; its experiment order is the
    order of the particles' envelope columns.

    ``utility`` compares finished designs and ``selector`` chooses each
    candidate's delays; they default to
    :class:`~nuclear_spin_recovery.design.utility.ExpectedInformationGain` and
    :class:`~nuclear_spin_recovery.design.selection.InformationDensity`.
    ``cost`` is called on a candidate and returns the time of one unit of
    weight at each point; it defaults to ``2 N tau``.  ``envelope``, if given,
    extrapolates lambda to pulse numbers not yet measured.
    ``shared_envelope`` (default True) predicts every particle with the
    posterior-mean envelope, so that only the spins are designed for.
    ``min_gain`` is the expected information gain, in nats, below which no
    experiment is proposed.
    """

    def __init__(self, model, site_table, measured, *, utility=None,
                 selector=None, cost=None, envelope=None, shared_envelope=True,
                 min_gain=MIN_GAIN):
        utility = ExpectedInformationGain() if utility is None else utility
        selector = InformationDensity() if selector is None else selector
        if not isinstance(utility, DesignUtility):
            raise TypeError(f"{type(utility).__name__} is not a DesignUtility")
        if not isinstance(selector, PointSelector):
            raise TypeError(f"{type(selector).__name__} is not a PointSelector")
        min_gain = float(min_gain)
        if not np.isfinite(min_gain) or min_gain < 0:
            raise ValueError(f"min_gain must be non-negative, got {min_gain}")
        self.utility = utility
        self.selector = selector
        self.model = model
        self.site_table = site_table
        self.measured = measured
        self.cost = SequenceDuration() if cost is None else cost
        self.envelope = envelope
        self.shared_envelope = bool(shared_envelope)
        self.min_gain = min_gain

    def cost_of(self, experiment):
        """Time of one unit of weight at each point of ``experiment``."""
        c = np.asarray(self.cost(experiment), dtype=float)
        if c.shape != experiment.tau.shape or not np.all(np.isfinite(c)) \
                or np.any(c <= 0):
            raise ValueError("cost must return a positive finite value per point")
        return c

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

    def _envelope_for(self, particles, cand):
        """Each particle's (lam, n_stretch, sigma) for ``cand``, as (K, 1).

        The matched measured experiment's sampled values when there is one;
        otherwise, with an ``envelope``, lambda scaled from the measured
        experiment at the same field nearest in log N.  With a shared envelope
        lambda and the stretch exponent are replaced by their particle-weighted
        means, the same for every particle.
        """
        lam, n_stretch, sigma = self._per_particle_envelope(particles, cand)
        if self.shared_envelope:
            w = particles.weight
            lam = np.full_like(lam, w @ lam[:, 0])
            n_stretch = np.full_like(n_stretch, w @ n_stretch[:, 0])
        return lam, n_stretch, sigma

    def _per_particle_envelope(self, particles, cand):
        try:
            e = self._matched_experiment(cand)
        except ValueError:
            if self.envelope is None:
                raise
            same_field = [
                (i, m.n_pulses) for i, m in enumerate(self.measured.experiments)
                if m.n_pulses > 0
                and np.isclose(m.b_z, cand.b_z, rtol=_SAME, atol=0)]
            if not same_field or cand.n_pulses <= 0:
                raise
            ref, n_ref = min(same_field, key=lambda p: (
                abs(np.log(p[1] / cand.n_pulses)), p[1]))
            lam = self.envelope.scale(particles.lam[:, [ref]], n_ref,
                                      cand.n_pulses)
            return lam, particles.n_stretch[:, [ref]], particles.sigma[:, [ref]]
        return (particles.lam[:, [e]], particles.n_stretch[:, [e]],
                particles.sigma[:, [e]])

    def designs(self, particles, candidates, *, budget, rng, exclude=None):
        """The best design of every candidate, scored. (list of CandidateDesign)

        Points of ``exclude`` -- usually the measured set -- are removed from
        any candidate with the same pulse number and field first, so nothing
        already measured is proposed again; a candidate left with no points is
        dropped, and if none remain this raises ValueError.

        Each remaining candidate is given the delays and weights its selector
        chooses for ``budget``, and all of the designs are scored in one call
        so that they share their random numbers.
        """
        candidates = [self._without_excluded(c, exclude) for c in candidates]
        if not candidates:
            raise ValueError("no candidates to design for")
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            raise ValueError("every candidate point has already been measured")
        budget = float(budget)
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        if particles.lam.shape[1] != self.measured.n_experiments:
            raise ValueError(
                f"particles carry envelopes for {particles.lam.shape[1]} "
                f"experiments, the measured set has "
                f"{self.measured.n_experiments}")

        out = []
        for cand in candidates:
            lam, n_stretch, sig = self._envelope_for(particles, cand)
            # Rebuilt bare: the candidate's own data or weights, if it has
            # any, are not what is being predicted.
            bare = ExperimentSet([Experiment(tau=cand.tau, n_pulses=cand.n_pulses,
                                             b_z=cand.b_z)])
            signals = predictive_from_arrays(
                particles.site_idx, particles.k, particles.dA_par,
                particles.dA_perp, lam, n_stretch, sig,
                bare, self.site_table, self.model, k_max=particles.k_max)
            sigma = (float(cand.sigma) if cand.sigma is not None
                     else float(particles.weight @ sig[:, 0]))
            cost = self.cost_of(cand)
            try:
                idx, weight = self.selector.select(
                    signals, particles.weight, sigma, budget, rng, cost=cost)
            except NothingToLearn:
                idx, weight = np.array([], dtype=int), np.array([])
            out.append(CandidateDesign(
                candidate=cand, signals=signals,
                density=information_density(signals, particles.weight, sigma),
                cost=cost, sigma=sigma, index=np.asarray(idx, dtype=int),
                weight=np.asarray(weight, dtype=float)))

        # Scored on the whole grid with infinite noise where a delay is not
        # measured, so a delay's noise draw does not depend on which other
        # delays the design kept.
        scored = [d for d in out if d.index.size and particles.n_particles > 1]
        if scored:
            noise = []
            for d in scored:
                effective = np.full(d.candidate.tau.size, np.inf)
                effective[d.index] = d.sigma / np.sqrt(d.weight)
                noise.append(effective)
            gains = self.utility.score_many(
                [d.signals for d in scored], particles.weight, noise, rng)
            for d, gain in zip(scored, gains, strict=True):
                d.gain = float(gain)
        return out

    def rank(self, particles, candidates, *, budget, rng, exclude=None):
        """Expected gain of each candidate's best design. (n_candidates,)

        In the order of :meth:`designs`, which drops a candidate that
        ``exclude`` empties.
        """
        return np.array([d.gain for d in self.designs(
            particles, candidates, budget=budget, rng=rng, exclude=exclude)])

    def propose(self, particles, candidates, *, budget, rng, exclude=None):
        """The next experiment to run, or the finding that there is none.

        Returns a :class:`DesignResult`.  Its ``experiment`` is on the chosen
        candidate's pulse number and field, its tau a subset of that
        candidate's in increasing order, its weights spending ``budget`` --
        ``sum(weight * cost) == budget`` -- its sigma the noise it was
        designed against, and it carries no data.  It is None when the best
        design gains less than ``min_gain``.
        """
        designs = self.designs(particles, candidates, budget=budget, rng=rng,
                               exclude=exclude)
        best = max(designs, key=lambda d: d.gain)
        experiment = None
        if best.index.size and best.gain >= self.min_gain:
            best.chosen = True
            experiment = best.experiment()
        return DesignResult(experiment=experiment, designs=designs,
                            budget=float(budget), min_gain=self.min_gain)

    def gain_curve(self, particles, candidates, budgets, *, seed=0,
                   exclude=None):
        """Expected gain of every candidate's best design, against budget.

        ``budgets`` is ``(n_budgets,)``, the same for every candidate, or
        ``(n_budgets, n_candidates)`` to give each candidate its own -- which
        is how to compare at equal numbers of repetitions, by making each
        candidate's budget a multiple of its own ``cost_of(c).sum()``.
        Returns ``(n_budgets, n_candidates)``.

        Every evaluation starts from the same ``seed``, so one candidate's
        curve is not roughened by a change of random numbers between budgets.
        """
        candidates = list(candidates)
        budgets = np.asarray(budgets, dtype=float)
        shared = budgets.ndim == 1
        if not shared and budgets.shape[1:] != (len(candidates),):
            raise ValueError(
                f"budgets has shape {budgets.shape}; expected (n_budgets,) or "
                f"(n_budgets, {len(candidates)})")
        out = np.empty((budgets.shape[0], len(candidates)))
        for b in range(budgets.shape[0]):
            if shared:
                out[b] = self.rank(particles, candidates, budget=budgets[b],
                                   rng=np.random.default_rng(seed),
                                   exclude=exclude)
            else:
                for c, cand in enumerate(candidates):
                    out[b, c] = self.rank(
                        particles, [cand], budget=budgets[b, c],
                        rng=np.random.default_rng(seed), exclude=exclude)[0]
        return out

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
