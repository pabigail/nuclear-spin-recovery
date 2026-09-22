"""Run configuration: loading, resolution, determinism, and submission.

Three properties here are load-bearing for the cluster and nothing else in the
suite checks them:

- the same config and root seed give a byte-identical trace, in this process or
  in a job-array task days later;
- an unknown key raises rather than being ignored, because a silently dropped
  key produces a completed run that sampled a model nobody chose;
- a merge over fewer traces than the config declares raises, because a failed
  array task must not quietly become a published posterior.

Nothing in this file submits a job. docs/phase-4-plan.md unit 4c.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nuclear_spin_recovery import (
    DEFAULT_QOS,
    KNOWN_ALGORITHMS,
    PERLMUTTER_ACCOUNT,
    PERLMUTTER_WORKDIR,
    AnalyticCCE1,
    EnsembleRunner,
    ExperimentSet,
    RunConfig,
    Schedule,
    SiteTable,
    StretchedExponential,
    merge_run,
    run_ensemble,
    submission_script,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

MINIMAL = """
[table]
path = "{table}"
strong_thresh = 750.0
weak_thresh = 5.0
coupling_cutoff = 100.0

[[experiment]]
n_pulses = 16
b_z = 311.0
tau_start = 3.2e-5
tau_stop = 8.0e-3
tau_count = 40

[data]
mode = "simulate"
truth_k = 4
truth_seed = 7
noise = 0.002

[likelihood]
sigma = 0.02

[state]
lam = 3.0e-3
n_stretch = 1.0
k_max = 16

[[schedule]]
algorithm = "rjmcmc"
n_steps = 20
k_max = 16

[[schedule]]
algorithm = "sites"
n_steps = 20
radius = 6.0

[ensemble]
n_ensembles = 3
n_steps = 60
n_burn = 20
root_seed = 2026
init_policy = "spread_across_k"
init_k = [2, 5]

[output]
dir = "runs"
"""


@pytest.fixture
def toml_path(tmp_path, nv2_path):
    def build(text=None, replace=None):
        body = (text or MINIMAL).format(table=str(nv2_path))
        for old, new in (replace or {}).items():
            assert old in body, f"{old!r} is not in the template"
            body = body.replace(old, new)
        path = tmp_path / "run.toml"
        path.write_text(body)
        return path
    return build


@pytest.fixture
def config(toml_path):
    return RunConfig.from_toml(toml_path())


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


# --------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------


def test_from_toml_returns_a_config(config):
    assert isinstance(config, RunConfig)


def test_from_toml_reads_the_sections(config):
    assert config.ensemble["n_ensembles"] == 3
    assert config.table["strong_thresh"] == pytest.approx(750.0)
    assert len(config.schedule) == 2


def test_from_dict_matches_from_toml(config, toml_path):
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_unknown_section_raises(toml_path):
    """A typo'd section must not be ignored."""
    with pytest.raises((KeyError, ValueError)):
        RunConfig.from_toml(toml_path(replace={"[output]": "[outputs]"}))


def test_unknown_key_raises_naming_it(toml_path):
    """A silently dropped key is a run that sampled a model nobody chose."""
    with pytest.raises((KeyError, ValueError), match="n_ensembels|unknown"):
        RunConfig.from_toml(toml_path(replace={"n_ensembles": "n_ensembels"}))


def test_missing_required_key_raises(toml_path):
    with pytest.raises((KeyError, ValueError)):
        RunConfig.from_toml(toml_path(replace={"root_seed = 2026": ""}))


def test_burn_in_exceeding_the_budget_raises(toml_path):
    with pytest.raises(ValueError):
        RunConfig.from_toml(toml_path(replace={"n_burn = 20": "n_burn = 60"}))


def test_unknown_algorithm_raises_naming_the_known_ones(toml_path):
    with pytest.raises(ValueError) as excinfo:
        RunConfig.from_toml(toml_path(replace={'algorithm = "rjmcmc"':
                                               'algorithm = "rjmcm"'}))
    assert any(name in str(excinfo.value) for name in KNOWN_ALGORITHMS)


def test_block_with_no_steps_raises(toml_path):
    with pytest.raises(ValueError):
        RunConfig.from_toml(toml_path(replace={"n_steps = 20\nk_max = 16":
                                               "n_steps = 0\nk_max = 16"}))


# --------------------------------------------------------------------------
# the resolved record
# --------------------------------------------------------------------------


def test_to_dict_round_trips(config):
    assert RunConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_to_dict_is_stable_under_repetition(config):
    assert config.to_dict() == config.to_dict()


def test_resolved_record_includes_filled_defaults(config, toml_path):
    """The point of recording the resolved config, not the source file."""
    raw = toml_path().read_text()
    resolved = config.to_dict()
    assert "isotope" not in raw
    assert "isotope" in resolved["table"]


def test_write_resolved_is_valid_json(config, tmp_path):
    config.write_resolved(tmp_path / "resolved.json")
    with open(tmp_path / "resolved.json") as handle:
        assert json.load(handle) == config.to_dict()


def test_write_resolved_is_byte_identical_across_calls(config, tmp_path):
    """A diff between two runs must show only what differed."""
    config.write_resolved(tmp_path / "a.json")
    config.write_resolved(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_resolved_json_reloads_into_an_equal_config(config, tmp_path):
    config.write_resolved(tmp_path / "r.json")
    with open(tmp_path / "r.json") as handle:
        assert RunConfig.from_dict(json.load(handle)).to_dict() == config.to_dict()


# --------------------------------------------------------------------------
# building objects
# --------------------------------------------------------------------------


def test_build_table_applies_the_thresholds(config):
    table = config.build_table()
    assert isinstance(table, SiteTable)
    magnitude = np.hypot(table.a_par, table.a_perp)
    assert magnitude.min() >= config.table["coupling_cutoff"]


def test_build_experiments_matches_the_declaration(config):
    expset = config.build_experiments()
    assert isinstance(expset, ExperimentSet)
    assert expset.n_points == 40
    assert expset.experiments[0].n_pulses == 16


def test_build_schedule_preserves_block_order(config):
    schedule = config.build_schedule()
    assert isinstance(schedule, Schedule)
    assert [s.algorithm.label for s in schedule] == ["rjmcmc:sites", "rwmh:sites"]


def test_build_schedule_carries_block_lengths(config):
    assert [s.n_steps for s in config.build_schedule()] == [20, 20]


def test_build_runner_carries_the_ensemble_settings(config):
    runner = config.build_runner(config.build_table())
    assert isinstance(runner, EnsembleRunner)
    assert (runner.n_ensembles, runner.n_steps, runner.n_burn) == (3, 60, 20)
    assert runner.init_name == "spread_across_k"


def test_build_data_simulates_from_the_recorded_truth(config, model):
    """Simulated data puts the truth in the provenance, not in a script."""
    table = config.build_table()
    data = config.build_data(table, model)
    assert data.data_all.shape == (40,)


def test_experimental_data_is_not_supported_yet(toml_path, model):
    """Unit 4e adds the reader; until then this must say so, not guess."""
    cfg = RunConfig.from_toml(toml_path(replace={'mode = "simulate"':
                                                 'mode = "file"'}))
    with pytest.raises(NotImplementedError, match="4e|experimental"):
        cfg.build_data(cfg.build_table(), model)


# --------------------------------------------------------------------------
# determinism -- what the cluster rests on
# --------------------------------------------------------------------------


def test_same_config_and_seed_give_an_identical_trace(config, tmp_path):
    a = run_ensemble(config, 0, out_dir=tmp_path / "a")
    b = run_ensemble(config, 0, out_dir=tmp_path / "b")
    for field in ("site_idx", "k", "lam", "log_prob"):
        assert np.array_equal(np.asarray(getattr(a, field)),
                              np.asarray(getattr(b, field))), field


def test_a_different_root_seed_gives_a_different_trace(config, toml_path,
                                                       tmp_path):
    other = RunConfig.from_toml(toml_path(replace={"root_seed = 2026":
                                                   "root_seed = 1"}))
    a = run_ensemble(config, 0, out_dir=tmp_path / "a")
    b = run_ensemble(other, 0, out_dir=tmp_path / "b")
    assert not np.array_equal(np.asarray(a.k), np.asarray(b.k))


def test_run_ensemble_matches_the_runner_directly(config, model, tmp_path):
    """The config layer must not perturb the seeding it delegates to."""
    table = config.build_table()
    from nuclear_spin_recovery import GaussianL2, Target

    target = Target(config.build_data(table, model), model, GaussianL2(), table)
    direct = config.build_runner(table).run_one(1, target,
                                                config.ensemble["root_seed"])
    viaconfig = run_ensemble(config, 1, out_dir=tmp_path)
    assert np.array_equal(np.asarray(direct.k), np.asarray(viaconfig.k))


def test_ensembles_differ_from_one_another(config, tmp_path):
    a = run_ensemble(config, 0, out_dir=tmp_path)
    b = run_ensemble(config, 1, out_dir=tmp_path)
    assert not np.array_equal(np.asarray(a.site_idx), np.asarray(b.site_idx))


# --------------------------------------------------------------------------
# running and merging
# --------------------------------------------------------------------------


def test_run_ensemble_writes_a_named_trace(config, tmp_path):
    run_ensemble(config, 2, out_dir=tmp_path)
    assert (tmp_path / "ensemble_002.npz").exists()


def test_run_ensemble_keeps_burn_in(config, tmp_path):
    """A task writes what it sampled; trimming is a pooling-time decision."""
    assert len(run_ensemble(config, 0, out_dir=tmp_path)) == 60


def test_merge_run_pools_every_ensemble(config, tmp_path):
    for i in range(config.ensemble["n_ensembles"]):
        run_ensemble(config, i, out_dir=tmp_path)
    result = merge_run(config, tmp_path)
    assert len(result.traces) == 3
    assert len(result.pooled) == 3 * (60 - 20)


def test_merge_run_discards_burn_in(config, tmp_path):
    for i in range(3):
        run_ensemble(config, i, out_dir=tmp_path)
    assert all(len(t) == 40 for t in merge_run(config, tmp_path).traces)


def test_partial_merge_raises_by_default(config, tmp_path):
    """A failed array task must not quietly become a published posterior."""
    run_ensemble(config, 0, out_dir=tmp_path)
    run_ensemble(config, 1, out_dir=tmp_path)
    with pytest.raises(ValueError, match="2|3|partial"):
        merge_run(config, tmp_path)


def test_partial_merge_is_allowed_when_asked_for(config, tmp_path):
    run_ensemble(config, 0, out_dir=tmp_path)
    run_ensemble(config, 1, out_dir=tmp_path)
    assert len(merge_run(config, tmp_path, allow_partial=True).traces) == 2


def test_merge_records_the_init_policy(config, tmp_path):
    for i in range(3):
        run_ensemble(config, i, out_dir=tmp_path)
    assert merge_run(config, tmp_path).init_name == "spread_across_k"


# --------------------------------------------------------------------------
# the submission script -- generated, never submitted
# --------------------------------------------------------------------------


def test_script_requests_the_allocation(config, tmp_path):
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert f"-A {PERLMUTTER_ACCOUNT}" in text


def test_script_array_spans_the_ensembles(config, tmp_path):
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert "--array=0-2" in text


def test_a_single_ensemble_still_gives_a_valid_array(config, toml_path,
                                                     tmp_path):
    one = RunConfig.from_toml(toml_path(replace={"n_ensembles = 3":
                                                 "n_ensembles = 1"}))
    assert "--array=0-0" in submission_script(one,
                                              config_path=tmp_path / "r.toml")


def test_script_requests_one_core_per_task(config, tmp_path):
    """One ensemble is minutes on a single core; asking for a node idles 127."""
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert "-n 1" in text and "-c 1" in text


def test_script_uses_shared_not_exclusive_qos(config, tmp_path):
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert f"-q {DEFAULT_QOS}" in text
    assert "-q regular" not in text


def test_script_runs_out_of_the_project_directory(config, tmp_path):
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert PERLMUTTER_WORKDIR in text


def test_script_passes_the_array_index_through(config, tmp_path):
    """SLURM_ARRAY_TASK_ID is what makes a task an ensemble."""
    text = submission_script(config, config_path=tmp_path / "run.toml")
    assert "SLURM_ARRAY_TASK_ID" in text


def test_script_names_the_config(config, tmp_path):
    path = tmp_path / "myrun.toml"
    assert "myrun.toml" in submission_script(config, config_path=path)


def test_script_overrides_are_honoured(config, tmp_path):
    text = submission_script(config, config_path=tmp_path / "r.toml",
                             account="m9999", qos="debug",
                             time_limit="00:05:00")
    assert "-A m9999" in text and "-q debug" in text and "00:05:00" in text


def test_nothing_in_this_module_submits_anything():
    """Generating the script and running it are deliberately separate steps.

    Checked on the parsed tree rather than the text: config.py necessarily
    says "sbatch" in its docstrings, and a substring search would call that a
    submission. What matters is whether it can execute one.
    """
    import ast
    import inspect

    from nuclear_spin_recovery import config as config_module

    tree = ast.parse(inspect.getsource(config_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "subprocess" not in imported, "config.py imports subprocess"

    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    for forbidden in ("system", "popen", "spawnl", "check_call", "run"):
        assert forbidden not in called, f"config.py calls {forbidden}()"
