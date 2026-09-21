"""Independent chains, pooled after the fact.

Several chains are run per dataset, differing in seed and initialisation and
**never exchanging information** -- the distinction from parallel tempering,
where rungs swap by design.  Post-burn-in samples pool into one posterior, and
the spread *between* ensembles is the convergence diagnostic: agreement is
evidence of convergence, disagreement reveals that individual chains remain
trapped.  See docs/model-specification.md Sec. 8.6.

Ensembles are independent, so they parallelise trivially.  The runner is built
for **one ensemble per task**: each task runs :meth:`EnsembleRunner.run_one`,
writes its trace, and a separate merge step pools any number of them.  Nothing
here ever needs every ensemble in one process, and a failed task is re-run
alone rather than restarting the set.

The ensemble count is a swept hyperparameter, not the inherited constant 5.
See docs/phase-4-plan.md Sec. 4, study 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .post.detection import BANDS

#: Parameters R-hat is computed for: the scalars with a stable identity across
#: trans-dimensional samples.  Deliberately excludes anything per-spin, where
#: label switching means there is no "spin 3" to take a variance of, and R_i,
#: which summarises an ensemble rather than varying within one.
RHAT_PARAMETERS = ("k", "lam", "n_stretch", "sigma")


def derive_seeds(root_seed, n_ensembles):
    """Per-ensemble seeds from one root seed. (n_ensembles,)

    Pure, so a run is reproducible from its config alone and ensemble j is
    reproducible without running the others -- which is what a job array needs.

    Extending the count must not renumber the ensembles already run: the first
    m entries of ``derive_seeds(root, n)`` are ``derive_seeds(root, m)`` for
    every m <= n.  Sweeping the ensemble count is otherwise not a sweep but a
    different experiment each time.
    """
    raise NotImplementedError


def spread_across_k(make_state, k_values):
    """An initialisation policy that starts ensembles at different k.

    Returns a callable ``(rng, index) -> State``, cycling ``k_values`` so that
    chains approach the truth from above and from below.

    This is the deliberate choice to *expose* the dimension multimodality of
    docs/test-plan.md Sec. 5.6 rather than hide it, and it is also what
    Gelman-Rubin requires: the statistic is meaningful only from overdispersed
    starts.  What policy to use on experimental data is a separate question,
    to be settled empirically.
    """
    raise NotImplementedError


def rhat(chains):
    """Gelman-Rubin potential scale reduction for one scalar parameter.

    ``chains`` is (M, N): M chains of N post-burn-in draws.  Compares the
    variance between chain means against the variance within chains,

        R = sqrt( [((N-1)/N) W + B/N] / W ),

    so that chains agreeing no more than they individually wander give R -> 1.

    Returns nan when the within-chain variance is zero, rather than dividing by
    it: a chain that never moved carries no information about mixing.

    **Reported, never gated.**  On this problem R-hat is structurally above 1 --
    measured at 2.39 on k for 4,000-step chains and *rising* to 2.68 at 20,000,
    while the pooled mode stayed correct at both.  A single chain cannot cross
    the dimension barrier, which is why ensembles exist rather than a defect in
    the run, and a conventional 1.01 gate would reject a correct posterior.
    See docs/phase-4-plan.md Sec. 6.
    """
    raise NotImplementedError


def merge_traces(traces):
    """Concatenate traces into one, in the order given.

    Raises rather than padding when the traces disagree on shape: a k_max or
    n_exp mismatch means they came from different models, and silently
    reconciling them would pool two posteriors that are not comparable.
    """
    raise NotImplementedError


def load_ensembles(directory, pattern="ensemble_*.npz"):
    """Load every saved trace in ``directory``, ordered by filename.

    The merge half of one-ensemble-per-task.  Any number of traces may be
    present, because the ensemble count is swept rather than fixed.
    """
    raise NotImplementedError


@dataclass
class Agreement:
    """Between-ensemble diagnostics.  Reports; never judges.

    There is deliberately no pass/fail field.  What licenses belief in a run is
    the stability of the *pooled* summary and the spread of modal k, read by a
    person against the calibration in docs/test-plan.md -- not a boolean
    computed here.
    """

    rhat: dict              # parameter name -> R-hat, for RHAT_PARAMETERS
    k_mode_spread: int      # max modal k across ensembles minus min
    max_delta_R: np.ndarray # largest pairwise |delta R_i| per coupling band
    n_ensembles: int


@dataclass
class EnsembleResult:
    """The output of a set of independent chains."""

    traces: list            # per-ensemble Trace, burn-in already discarded
    seeds: np.ndarray       # the seed each ensemble ran under
    init_name: str          # which initialisation policy produced the starts

    @property
    def pooled(self):
        """One Trace over every ensemble, burn-in already discarded."""
        raise NotImplementedError

    def agreement(self, summaries=None, bands=BANDS):
        """Between-ensemble diagnostics. (:class:`Agreement`)

        ``summaries`` is one :class:`~nuclear_spin_recovery.post.metrics.
        PosteriorSummary` per ensemble when detection spread is wanted; without
        it ``max_delta_R`` comes back empty, because R_i cannot be computed
        without a reference.
        """
        raise NotImplementedError

    def save(self, directory):
        """Write one compressed trace per ensemble, plus the run metadata."""
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        """Reconstruct a result from a directory written by :meth:`save`."""
        raise NotImplementedError


class EnsembleRunner:
    """Runs independent chains, one per task, and pools them afterwards."""

    def __init__(self, schedule, n_ensembles, n_steps, n_burn, init,
                 init_name="unnamed"):
        self.schedule = schedule
        self.n_ensembles = int(n_ensembles)
        self.n_steps = int(n_steps)
        self.n_burn = int(n_burn)
        self.init = init
        #: Recorded rather than inferred: the initialisation policy is a
        #: scientific choice and belongs in the run's provenance.
        self.init_name = str(init_name)
        if self.n_burn >= self.n_steps:
            raise ValueError(
                f"n_burn={self.n_burn} would leave nothing of n_steps="
                f"{self.n_steps}"
            )

    def seeds(self, root_seed):
        """The per-ensemble seeds this run would use. (n_ensembles,)"""
        raise NotImplementedError

    def run_one(self, index, target, root_seed):
        """Run ensemble ``index`` alone and return its **full** trace.

        Burn-in is not discarded here: a task writes what it sampled, and how
        much to discard is a decision made at pooling, against diagnostics the
        single task cannot see.
        """
        raise NotImplementedError

    def run(self, target, root_seed):
        """Run every ensemble in this process. (:class:`EnsembleResult`)

        Equivalent to calling :meth:`run_one` for each index and discarding
        burn-in; provided for small runs and for tests, not for the cluster.
        """
        raise NotImplementedError
