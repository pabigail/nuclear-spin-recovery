"""Run configuration: authored in TOML, recorded as resolved JSON.

A run is specified by a file, not by a Python literal.  That is what makes a
cluster run reproducible from its provenance alone, and it is what the
determinism property rests on: the same config and root seed must produce a
byte-identical trace, in this process or in a job-array task three days later.

**TOML in, JSON out.**  ``tomllib`` is stdlib from 3.11 so reading costs no
dependency; writing TOML is *not* stdlib and would add one.  The runner
therefore records the **resolved** configuration as JSON beside its results --
stdlib in both directions, diffable, and capturing the defaults that were
filled in rather than only what the author typed.

See docs/phase-4-plan.md, unit 4c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Algorithm names a schedule block may use.  A typo must not be silently
#: ignored: on a cluster the run would complete and quietly sample the wrong
#: model.
KNOWN_ALGORITHMS = ("sites", "lam", "n_stretch", "sigma", "offsets",
                    "rjmcmc", "tempering")

#: Perlmutter defaults, from the allocation.  Overridable per call.
PERLMUTTER_ACCOUNT = "m5305"
PERLMUTTER_WORKDIR = "/global/cfs/cdirs/m5305/pabigail/claude-rewrite"
#: `shared` rather than `regular`: one 25,000-step ensemble is 3.9 minutes on a
#: single core and a CPU node has 128, so an exclusive allocation would idle
#: 127 of them per array task.  See docs/phase-4-plan.md Sec. 3, unit 4c.
DEFAULT_QOS = "shared"


@dataclass
class RunConfig:
    """Everything needed to reproduce a run.

    Constructed from a file rather than assembled in code, so that the record
    written beside the results is the same object the run was driven by.
    """

    table: dict = field(default_factory=dict)
    experiment: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    likelihood: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    schedule: list = field(default_factory=list)
    ensemble: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path):
        """Read a configuration from a TOML file."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, raw):
        """Validate a raw mapping and fill in defaults.

        Unknown sections and unknown keys raise rather than being ignored.  A
        silently dropped key is the worst failure available here: the run
        completes, the record looks right, and the sampler used a default
        nobody chose.
        """
        raise NotImplementedError

    def to_dict(self):
        """The **resolved** configuration: what was typed, plus every default.

        Stable under repetition -- two calls give equal dicts, and the JSON
        written from them is byte-identical -- so a diff between two runs shows
        only what actually differed.
        """
        raise NotImplementedError

    def write_resolved(self, path):
        """Write :meth:`to_dict` as JSON beside the results."""
        raise NotImplementedError

    def build_table(self):
        """The SiteTable this configuration describes."""
        raise NotImplementedError

    def build_experiments(self):
        """The ExperimentSet this configuration describes, without data."""
        raise NotImplementedError

    def build_data(self, site_table, model):
        """The ExperimentSet with data attached.

        ``mode = "simulate"`` generates it from a truth configuration recorded
        in the config, so the truth is part of the provenance.  ``mode =
        "file"`` awaits the experimental reader of unit 4e.
        """
        raise NotImplementedError

    def build_schedule(self):
        """The Schedule this configuration describes, blocks in order."""
        raise NotImplementedError

    def build_runner(self, site_table):
        """The EnsembleRunner this configuration describes."""
        raise NotImplementedError


def submission_script(config, *, config_path, account=PERLMUTTER_ACCOUNT,
                      qos=DEFAULT_QOS, workdir=PERLMUTTER_WORKDIR,
                      time_limit="00:30:00", job_name="nsr-ensemble"):
    """Generate the sbatch text for a job array, one ensemble per task.

    Returns a string.  Nothing in this package submits anything -- generating
    the script and running it are deliberately separate steps, so that a test
    can assert on the text without a scheduler.
    """
    raise NotImplementedError


def run_ensemble(config, index, out_dir=None):
    """Run one ensemble and write its trace.  Returns the Trace.

    The entry point a job-array task calls, with ``index`` taken from
    ``SLURM_ARRAY_TASK_ID``.  Writes ``ensemble_{index:03d}.npz`` so that
    :func:`merge_run` can find it, and the **full** trace including burn-in --
    how much to discard is a pooling-time decision.
    """
    raise NotImplementedError


def merge_run(config, out_dir, allow_partial=False):
    """Pool every ensemble written under ``out_dir``. (:class:`EnsembleResult`)

    Raises when fewer traces are present than the configuration declares,
    unless ``allow_partial``.  A silently partial merge is how a failed array
    task becomes a published posterior: the pooled result looks healthy and is
    missing a third of its samples.
    """
    raise NotImplementedError
