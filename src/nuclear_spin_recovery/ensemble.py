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
from pathlib import Path

import numpy as np

from .driver import HybridDriver
from .post.detection import BANDS
from .trace import Trace

#: Parameters R-hat is computed for: the scalars with a stable identity across
#: trans-dimensional samples.  Deliberately excludes anything per-spin, where
#: label switching means there is no "spin 3" to take a variance of, and R_i,
#: which summarises an ensemble rather than varying within one.
RHAT_PARAMETERS = ("k", "lam", "n_stretch", "sigma")


def derive_seeds(root_seed, n_ensembles):
    """Per-ensemble seeds from one root seed. (n_ensembles,)

    Pure, so a run is reproducible from its config alone and ensemble j is
    reproducible without running the others -- which is what a job array needs.

    Derived per index rather than by spawning, so prefix stability is
    structural: the first m entries of ``derive_seeds(root, n)`` are
    ``derive_seeds(root, m)`` for every m <= n, and stay so across numpy
    versions.  Sweeping the ensemble count is otherwise not a sweep but a
    different experiment at every point on the curve.
    """
    n = int(n_ensembles)
    if n < 1:
        raise ValueError(f"n_ensembles must be at least 1, got {n_ensembles}")
    return np.array(
        [np.random.SeedSequence([int(root_seed), i]).generate_state(
            1, dtype=np.uint64)[0] for i in range(n)],
        dtype=np.uint64,
    )


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
    k_values = tuple(int(k) for k in k_values)
    if not k_values:
        raise ValueError("k_values must not be empty")
    # The lattice size is a property of the state factory, not of this policy.
    # Asking it for an empty configuration is the cheapest way to learn it
    # without threading the table through a second argument.
    n_sites = make_state([]).n_sites

    def init(rng, index):
        k = k_values[int(index) % len(k_values)]
        return make_state(rng.choice(n_sites, size=k, replace=False))

    return init


def rhat(chains):
    """Gelman-Rubin potential scale reduction for one scalar parameter.

    ``chains`` is (M, N): M chains of N post-burn-in draws.  Compares the
    variance between chain means against the variance within chains,

        R = sqrt( [((N-1)/N) W + B/N] / W ),

    so that chains agreeing no more than they individually wander give R -> 1.

    Returns nan when the chains never moved, rather than dividing by a variance
    that is zero: a frozen parameter carries no information about mixing.

    The test is relative to the data scale, not ``within == 0``.  A chain held
    at a constant 3e-3 gives a ddof=1 variance of about 1e-37 rather than
    exactly zero -- cancellation residue, not movement -- and an exact
    comparison lets that through as R = sqrt((N-1)/N) = 0.994, which reads as
    *perfect convergence* for a parameter that never moved.  That is the worst
    failure mode available to a diagnostic, so the guard is scaled.

    **Reported, never gated.**  On this problem R-hat is structurally above 1 --
    measured at 2.39 on k for 4,000-step chains and *rising* to 2.68 at 20,000,
    while the pooled mode stayed correct at both.  A single chain cannot cross
    the dimension barrier, which is why ensembles exist rather than a defect in
    the run, and a conventional 1.01 gate would reject a correct posterior.
    See docs/phase-4-plan.md Sec. 6.
    """
    arr = np.asarray(chains, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_chains, n_draws), got shape {arr.shape}")
    m, n = arr.shape
    if m < 2:
        raise ValueError(f"R-hat needs at least 2 chains, got {m}")
    if n < 2:
        raise ValueError(f"R-hat needs at least 2 draws per chain, got {n}")
    within = arr.var(axis=1, ddof=1).mean()
    scale = float(np.abs(arr).max())
    if within <= np.finfo(float).eps * max(scale * scale, 1.0):
        return float("nan")
    means = arr.mean(axis=1)
    between = n / (m - 1) * ((means - means.mean()) ** 2).sum()
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def merge_traces(traces):
    """Concatenate traces into one, in the order given.

    Raises rather than padding when the traces disagree on shape: a k_max or
    n_exp mismatch means they came from different models, and silently
    reconciling them would pool two posteriors that are not comparable.
    """
    traces = list(traces)
    if not traces:
        raise ValueError("no traces to merge")
    first = traces[0]
    for i, other in enumerate(traces[1:], start=1):
        for attr in ("k_max", "n_exp", "n_sites"):
            if getattr(other, attr) != getattr(first, attr):
                raise ValueError(
                    f"trace {i} has {attr}={getattr(other, attr)}, trace 0 has "
                    f"{getattr(first, attr)}; they describe different models"
                )
    out = Trace(first.n_sites, first.k_max, first.n_exp)
    for store in ("_site_idx", "_k", "_lam", "_n_stretch", "_sigma",
                  "_dA_par", "_dA_perp", "_log_prob", "_algorithm"):
        merged = []
        for trace in traces:
            merged.extend(getattr(trace, store))
        setattr(out, store, merged)
    return out


def load_ensembles(directory, pattern="ensemble_*.npz"):
    """Load every saved trace in ``directory``, ordered by filename.

    The merge half of one-ensemble-per-task.  Any number of traces may be
    present, because the ensemble count is swept rather than fixed.
    """
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"no files matching {pattern!r} in {directory}")
    return [Trace.load(path) for path in paths]


def _rhat_across(traces, name):
    """R-hat for one trace field, worst case over experiments.

    nan for a single ensemble: R-hat compares chains against each other, so
    with one chain the question is undefined rather than failed.  Reporting it
    as nan keeps a one-ensemble result summarisable, which the ensemble-count
    sweep of docs/phase-4-plan.md Sec. 5.2 needs -- it starts at M = 1.
    """
    if len(traces) < 2:
        return float("nan")
    arrays = [np.asarray(getattr(trace, name), dtype=float) for trace in traces]
    if arrays[0].ndim == 1:
        return rhat(np.array(arrays))
    per_column = [rhat(np.array([a[:, c] for a in arrays]))
                  for c in range(arrays[0].shape[1])]
    # The worst-converging experiment is the honest summary; a mean would let a
    # well-mixed column hide one that never moved.
    return (float("nan") if all(np.isnan(v) for v in per_column)
            else float(np.nanmax(per_column)))


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

    #: Filename the run metadata is written under.  Deliberately not matching
    #: the ``ensemble_*.npz`` glob, so loading traces never picks it up.
    META = "run_meta.npz"

    @property
    def pooled(self):
        """One Trace over every ensemble, burn-in already discarded."""
        return merge_traces(self.traces)

    def agreement(self, summaries=None, bands=BANDS):
        """Between-ensemble diagnostics. (:class:`Agreement`)

        ``summaries`` is one :class:`~nuclear_spin_recovery.post.metrics.
        PosteriorSummary` per ensemble when detection spread is wanted; without
        it ``max_delta_R`` comes back empty, because R_i cannot be computed
        without a reference.
        """
        modes = [int(np.bincount(np.asarray(t.k)).argmax()) for t in self.traces]
        return Agreement(
            rhat={name: _rhat_across(self.traces, name)
                  for name in RHAT_PARAMETERS},
            k_mode_spread=int(max(modes) - min(modes)),
            max_delta_R=_delta_R(summaries, bands),
            n_ensembles=len(self.traces),
        )

    def save(self, directory):
        """Write one compressed trace per ensemble, plus the run metadata."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for i, trace in enumerate(self.traces):
            trace.save(path / f"ensemble_{i:03d}.npz")
        np.savez(path / self.META,
                 seeds=np.asarray(self.seeds),
                 init_name=np.array(self.init_name, dtype=np.str_))
        return path

    @classmethod
    def load(cls, directory):
        """Reconstruct a result from a directory written by :meth:`save`."""
        path = Path(directory)
        traces = load_ensembles(path)
        with np.load(path / cls.META, allow_pickle=False) as handle:
            seeds = np.asarray(handle["seeds"])
            init_name = str(handle["init_name"])
        return cls(traces=traces, seeds=seeds, init_name=init_name)


def _delta_R(summaries, bands):
    """Largest pairwise spread in R_i, per coupling band. (len(bands),)

    Taken per reference spin first and then maximised within the band, not
    averaged: one spin that half the ensembles find and half do not is the
    disagreement worth seeing, and averaging over its band would dilute it
    against spins every ensemble agrees on.
    """
    if summaries is None:
        return np.empty(0, dtype=float)
    summaries = list(summaries)
    if len(summaries) < 2:
        raise ValueError(
            f"detection spread needs at least 2 summaries, got {len(summaries)}")
    R = np.array([np.asarray(s.R_i, dtype=float) for s in summaries])
    if R.size == 0:
        return np.full(len(bands), np.nan)
    magnitude = np.asarray(summaries[0].magnitude, dtype=float)
    per_spin = R.max(axis=0) - R.min(axis=0)
    out = np.full(len(bands), np.nan)
    for b, (lo, hi) in enumerate(bands):
        sel = (magnitude >= lo) & (magnitude < hi)
        if sel.any():
            out[b] = float(per_spin[sel].max())
    return out


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
        return derive_seeds(root_seed, self.n_ensembles)

    def run_one(self, index, target, root_seed):
        """Run ensemble ``index`` alone and return its **full** trace.

        Burn-in is not discarded here: a task writes what it sampled, and how
        much to discard is a decision made at pooling, against diagnostics the
        single task cannot see.
        """
        index = int(index)
        if not 0 <= index < self.n_ensembles:
            raise IndexError(
                f"ensemble {index} outside [0, {self.n_ensembles})")
        rng = np.random.default_rng(int(self.seeds(root_seed)[index]))
        state = self.init(rng, index)
        trace = Trace(n_sites=state.n_sites, k_max=state.k_max,
                      n_exp=state.n_exp)
        HybridDriver(self.schedule).run(state, target, rng,
                                        n_total=self.n_steps, trace=trace)
        return trace

    def run(self, target, root_seed):
        """Run every ensemble in this process. (:class:`EnsembleResult`)

        Equivalent to calling :meth:`run_one` for each index and discarding
        burn-in; provided for small runs and for tests, not for the cluster.
        """
        traces = [self.run_one(i, target, root_seed).discard_burn_in(self.n_burn)
                  for i in range(self.n_ensembles)]
        return EnsembleResult(traces=traces, seeds=self.seeds(root_seed),
                              init_name=self.init_name)
