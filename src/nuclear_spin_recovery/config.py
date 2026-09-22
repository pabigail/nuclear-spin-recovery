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

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .algorithms import RJMCMC, RWMH, BirthDeathKernel, ParallelTempering, ParameterBlock, Target
from .driver import Schedule, Step
from .ensemble import EnsembleResult, EnsembleRunner, derive_seeds, load_ensembles, spread_across_k
from .experiment import Experiment, ExperimentSet
from .forward import AnalyticCCE1, StretchedExponential
from .lattice import SiteTable
from .likelihood import GaussianL2
from .neighbors import NeighborIndex
from .proposals import ContinuousReflected, DiscreteLatticeWalk, GaussianOffset
from .simulate import simulate_dataset
from .state import State

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



#: Sentinel for a key the author must supply.
_REQUIRED = object()

#: Section schemas: key -> default, or _REQUIRED.  Every accepted key appears
#: here, so an unknown one is detectable rather than silently dropped.
_SECTIONS = {
    "table": {"path": _REQUIRED, "strong_thresh": _REQUIRED,
              "weak_thresh": _REQUIRED, "coupling_cutoff": None,
              "isotope": "13C"},
    "data": {"mode": "simulate", "truth_k": None, "truth_seed": None,
             "noise": 0.002, "path": None},
    "likelihood": {"sigma": _REQUIRED},
    "state": {"lam": _REQUIRED, "n_stretch": 1.0, "k_max": _REQUIRED},
    "ensemble": {"n_ensembles": _REQUIRED, "n_steps": _REQUIRED,
                 "n_burn": _REQUIRED, "root_seed": _REQUIRED,
                 "init_policy": "spread_across_k", "init_k": _REQUIRED},
    "output": {"dir": "runs"},
}

_EXPERIMENT = {"n_pulses": _REQUIRED, "b_z": _REQUIRED, "tau_start": _REQUIRED,
               "tau_stop": _REQUIRED, "tau_count": _REQUIRED}

#: Per-algorithm block keys, beyond the universal ``algorithm`` and ``n_steps``.
_BLOCK = {
    "sites": {"radius": 5.0},
    "lam": {"radius": 2e-4, "lower": 5e-4, "upper": 2e-2},
    "n_stretch": {"radius": 0.05, "lower": 0.5, "upper": 3.0},
    "sigma": {"radius": 2e-3, "lower": 1e-3, "upper": 1.0},
    "offsets": {"radius": 1.5, "scale": 4.0, "bound": None},
    "rjmcmc": {"k_max": _REQUIRED, "birth_prob": 0.5},
    "tempering": {"n_replicas": 6, "inner": "sites", "radius": 5.0},
}


def _resolve(section, raw, schema, where):
    """Fill defaults and reject anything the schema does not name."""
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [{where}]; "
            f"known keys are {sorted(schema)}"
        )
    out = {}
    for key, default in schema.items():
        if key in raw:
            out[key] = raw[key]
        elif default is _REQUIRED:
            raise ValueError(f"[{where}] is missing the required key {key!r}")
        else:
            out[key] = default
    return out


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
        with open(path, "rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, raw):
        """Validate a raw mapping and fill in defaults.

        Unknown sections and unknown keys raise rather than being ignored.  A
        silently dropped key is the worst failure available here: the run
        completes, the record looks right, and the sampler used a default
        nobody chose.
        """
        expected = set(_SECTIONS) | {"experiment", "schedule"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(
                f"unknown section(s) {sorted(unknown)}; "
                f"known sections are {sorted(expected)}"
            )
        sections = {name: _resolve(name, raw.get(name, {}), schema, name)
                    for name, schema in _SECTIONS.items()}

        experiments = [_resolve("experiment", e, _EXPERIMENT, "[[experiment]]")
                       for e in raw.get("experiment", [])]
        if not experiments:
            raise ValueError("at least one [[experiment]] is required")

        blocks = []
        for i, block in enumerate(raw.get("schedule", [])):
            name = block.get("algorithm")
            if name not in _BLOCK:
                raise ValueError(
                    f"[[schedule]] {i}: unknown algorithm {name!r}; "
                    f"known algorithms are {list(KNOWN_ALGORITHMS)}"
                )
            schema = {"algorithm": _REQUIRED, "n_steps": _REQUIRED, **_BLOCK[name]}
            resolved = _resolve("schedule", block, schema, f"[[schedule]] {i}")
            if int(resolved["n_steps"]) < 1:
                raise ValueError(
                    f"[[schedule]] {i}: n_steps must be at least 1, "
                    f"got {resolved['n_steps']}"
                )
            blocks.append(resolved)
        if not blocks:
            raise ValueError("at least one [[schedule]] block is required")

        ens = sections["ensemble"]
        if int(ens["n_burn"]) >= int(ens["n_steps"]):
            raise ValueError(
                f"n_burn={ens['n_burn']} would leave nothing of "
                f"n_steps={ens['n_steps']}"
            )
        if sections["data"]["mode"] not in ("simulate", "file"):
            raise ValueError(
                f"data.mode must be 'simulate' or 'file', "
                f"got {sections['data']['mode']!r}"
            )
        return cls(experiment=experiments, schedule=blocks, **sections)

    def to_dict(self):
        """The **resolved** configuration: what was typed, plus every default.

        Stable under repetition -- two calls give equal dicts, and the JSON
        written from them is byte-identical -- so a diff between two runs shows
        only what actually differed.
        """
        return {
            "table": dict(self.table),
            "experiment": [dict(e) for e in self.experiment],
            "data": dict(self.data),
            "likelihood": dict(self.likelihood),
            "state": dict(self.state),
            "schedule": [dict(b) for b in self.schedule],
            "ensemble": dict(self.ensemble),
            "output": dict(self.output),
        }

    def write_resolved(self, path):
        """Write :meth:`to_dict` as JSON beside the results."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    # -- building --------------------------------------------------------

    def build_table(self):
        """The SiteTable this configuration describes."""
        spec = self.table
        table = SiteTable.from_ivady_file(
            spec["path"], strong_thresh=spec["strong_thresh"],
            weak_thresh=spec["weak_thresh"], isotope=spec["isotope"])
        cutoff = spec["coupling_cutoff"]
        if cutoff is None:
            return table
        keep = np.hypot(table.a_par, table.a_perp) >= float(cutoff)
        return SiteTable(
            distance=table.distance[keep], positions=table.positions[keep],
            a_par=table.a_par[keep], a_perp=table.a_perp[keep],
            isotope=table.isotope[keep], gyro=table.gyro[keep])

    def build_experiments(self):
        """The ExperimentSet this configuration describes, without data."""
        return ExperimentSet([
            Experiment(
                tau=np.linspace(e["tau_start"], e["tau_stop"],
                                int(e["tau_count"])),
                n_pulses=int(e["n_pulses"]), b_z=float(e["b_z"]))
            for e in self.experiment
        ])

    def build_data(self, site_table, model):
        """The ExperimentSet with data attached.

        ``mode = "simulate"`` generates it from a truth configuration recorded
        in the config, so the truth is part of the provenance.  ``mode =
        "file"`` awaits the experimental reader of unit 4e.
        """
        spec = self.data
        if spec["mode"] == "file":
            raise NotImplementedError(
                "reading experimental data is unit 4e; until then use "
                "data.mode = 'simulate'"
            )
        if spec["truth_k"] is None or spec["truth_seed"] is None:
            raise ValueError(
                "data.mode = 'simulate' needs truth_k and truth_seed")
        rng = np.random.default_rng(int(spec["truth_seed"]))
        sites = rng.choice(len(site_table), int(spec["truth_k"]), replace=False)
        truth = self._state(site_table, sites, sigma=float(spec["noise"]))
        return simulate_dataset(
            truth, self.build_experiments(), site_table, model,
            sigma=float(spec["noise"]),
            rng=np.random.default_rng(int(spec["truth_seed"]) + 9000))

    def build_schedule(self, site_table=None):
        """The Schedule this configuration describes, blocks in order."""
        site_table = self.build_table() if site_table is None else site_table
        return Schedule([self._block(b, site_table) for b in self.schedule])

    def build_runner(self, site_table):
        """The EnsembleRunner this configuration describes."""
        ens = self.ensemble

        def make_state(sites):
            return self._state(site_table, sites)

        return EnsembleRunner(
            self.build_schedule(site_table),
            n_ensembles=int(ens["n_ensembles"]), n_steps=int(ens["n_steps"]),
            n_burn=int(ens["n_burn"]),
            init=spread_across_k(make_state, ens["init_k"]),
            init_name=str(ens["init_policy"]))

    # -- internals -------------------------------------------------------

    def _state(self, site_table, sites, sigma=None):
        """A single-replica State on ``site_table`` at the configured values."""
        n_exp = len(self.experiment)
        sigma = self.likelihood["sigma"] if sigma is None else sigma
        return State.from_sites(
            np.sort(np.asarray(list(sites), dtype=int)),
            n_sites=len(site_table), n_exp=n_exp,
            lam=np.full((1, n_exp), float(self.state["lam"])),
            n_stretch=np.full((1, n_exp), float(self.state["n_stretch"])),
            sigma=np.full((1, n_exp), float(sigma)),
            k_max=int(self.state["k_max"]))

    def _block(self, spec, site_table):
        """One Step from a resolved schedule block."""
        name, n_steps = spec["algorithm"], int(spec["n_steps"])
        if name == "sites":
            return Step(RWMH(ParameterBlock("sites"),
                             self._walk(spec["radius"], site_table)), n_steps)
        if name in ("lam", "n_stretch", "sigma"):
            return Step(RWMH(ParameterBlock(name),
                             ContinuousReflected(radius=spec["radius"],
                                                 lower=spec["lower"],
                                                 upper=spec["upper"])), n_steps)
        if name == "offsets":
            return Step(RWMH(ParameterBlock("offsets"),
                             GaussianOffset(radius=spec["radius"],
                                            scale=spec["scale"],
                                            bound=spec["bound"])), n_steps)
        if name == "rjmcmc":
            return Step(RJMCMC(ParameterBlock("sites"),
                               BirthDeathKernel(k_max=int(spec["k_max"]),
                                                birth_prob=spec["birth_prob"])),
                        n_steps)
        inner = Schedule([Step(RWMH(ParameterBlock("sites"),
                                    self._walk(spec["radius"], site_table)), 1)])
        return Step(ParallelTempering(inner,
                                      n_replicas=int(spec["n_replicas"])),
                    n_steps)

    @staticmethod
    def _walk(radius, site_table):
        return DiscreteLatticeWalk(NeighborIndex(site_table.positions,
                                                 radius=float(radius)))


def _build_model():
    """The forward model.

    Not yet configurable: the analytic CCE-1 form is the only one implemented,
    and a PyCCE backend is phase 5.  When there are two, this becomes a section.
    """
    return AnalyticCCE1(StretchedExponential())


def submission_script(config, *, config_path, account=PERLMUTTER_ACCOUNT,
                      qos=DEFAULT_QOS, workdir=PERLMUTTER_WORKDIR,
                      time_limit="00:30:00", job_name="nsr-ensemble"):
    """Generate the sbatch text for a job array, one ensemble per task.

    Returns a string.  Nothing in this package submits anything -- generating
    the script and running it are deliberately separate steps, so that a test
    can assert on the text without a scheduler.
    """
    last = int(config.ensemble["n_ensembles"]) - 1
    out_dir = config.output["dir"]
    return f"""#!/bin/bash
#SBATCH -A {account}
#SBATCH -C cpu
#SBATCH -q {qos}
#SBATCH --array=0-{last}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t {time_limit}
#SBATCH -D {workdir}
#SBATCH -J {job_name}
#SBATCH -o logs/%A_%a.out

# One ensemble per task. `shared` rather than `regular`: a single ensemble is
# minutes on one core, and a CPU node has 128 -- an exclusive allocation would
# idle 127 of them per task.
set -euo pipefail

srun -n 1 python scripts/run_ensemble.py \\
    --config {config_path} \\
    --ensemble "$SLURM_ARRAY_TASK_ID" \\
    --out {out_dir}/"$SLURM_ARRAY_JOB_ID"

# When the array completes, pool it:
#   python scripts/merge_ensembles.py --config {config_path} \\
#       --out {out_dir}/<jobid>
"""


def run_ensemble(config, index, out_dir=None):
    """Run one ensemble and write its trace.  Returns the Trace.

    The entry point a job-array task calls, with ``index`` taken from
    ``SLURM_ARRAY_TASK_ID``.  Writes ``ensemble_{index:03d}.npz`` so that
    :func:`merge_run` can find it, and the **full** trace including burn-in --
    how much to discard is a pooling-time decision.
    """
    site_table = config.build_table()
    model = _build_model()
    data = config.build_data(site_table, model)
    target = Target(data, model, GaussianL2(), site_table)
    trace = config.build_runner(site_table).run_one(
        int(index), target, config.ensemble["root_seed"])
    destination = Path(out_dir if out_dir is not None else config.output["dir"])
    destination.mkdir(parents=True, exist_ok=True)
    trace.save(destination / f"ensemble_{int(index):03d}.npz")
    return trace


def merge_run(config, out_dir, allow_partial=False):
    """Pool every ensemble written under ``out_dir``. (:class:`EnsembleResult`)

    Raises when fewer traces are present than the configuration declares,
    unless ``allow_partial``.  A silently partial merge is how a failed array
    task becomes a published posterior: the pooled result looks healthy and is
    missing a third of its samples.
    """
    traces = load_ensembles(out_dir)
    declared = int(config.ensemble["n_ensembles"])
    if len(traces) != declared and not allow_partial:
        raise ValueError(
            f"found {len(traces)} traces in {out_dir} but the configuration "
            f"declares {declared}; a partial merge hides a failed array task. "
            f"Pass allow_partial=True to pool them anyway."
        )
    n_burn = int(config.ensemble["n_burn"])
    return EnsembleResult(
        traces=[t.discard_burn_in(n_burn) for t in traces],
        seeds=derive_seeds(config.ensemble["root_seed"], len(traces)),
        init_name=str(config.ensemble["init_policy"]))
