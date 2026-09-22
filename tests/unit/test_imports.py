"""Wiring checks: every module imports, and the public API is reachable.

These are the only tests expected to pass before phase 1 is implemented.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "nuclear_spin_recovery",
    "nuclear_spin_recovery.units",
    "nuclear_spin_recovery.lattice",
    "nuclear_spin_recovery.experiment",
    "nuclear_spin_recovery.state",
    "nuclear_spin_recovery.simulate",
    "nuclear_spin_recovery.forward",
    "nuclear_spin_recovery.forward.base",
    "nuclear_spin_recovery.forward.envelope",
    "nuclear_spin_recovery.forward.analytic",
    "nuclear_spin_recovery.likelihood",
    "nuclear_spin_recovery.likelihood.base",
    "nuclear_spin_recovery.likelihood.gaussian",
    "nuclear_spin_recovery.likelihood.wasserstein",
    "nuclear_spin_recovery.trace",
    "nuclear_spin_recovery.neighbors",
    "nuclear_spin_recovery.proposals",
    "nuclear_spin_recovery.algorithms",
    "nuclear_spin_recovery.algorithms.base",
    "nuclear_spin_recovery.algorithms.rwmh",
    "nuclear_spin_recovery.algorithms.rjmcmc",
    "nuclear_spin_recovery.algorithms.tempering",
    "nuclear_spin_recovery.driver",
    "nuclear_spin_recovery.post",
    "nuclear_spin_recovery.post.detection",
    "nuclear_spin_recovery.post.metrics",
    "nuclear_spin_recovery.post.residual",
    "nuclear_spin_recovery.post.plots",
    "nuclear_spin_recovery.ensemble",
    "nuclear_spin_recovery.config",
]

PUBLIC_NAMES = [
    "AnalyticCCE1",
    "Envelope",
    "Experiment",
    "ExperimentSet",
    "ForwardModel",
    "GaussianL2",
    "Likelihood",
    "SiteTable",
    "State",
    "StretchedExponential",
    "add_noise",
    "gyromagnetic_ratio",
    "read_hyperfine_table",
    "secular_components",
    "simulate_coherence",
    "simulate_dataset",
    "single_spin_modulation",
    "to_angular",
    "Algorithm",
    "ContinuousReflected",
    "DiscreteLatticeWalk",
    "NeighborIndex",
    "ParameterBlock",
    "Proposal",
    "RWMH",
    "Target",
    "Trace",
    "BirthDeathKernel",
    "GaussianOffset",
    "HybridDriver",
    "ParallelTempering",
    "RJMCMC",
    "Schedule",
    "Step",
    "geometric_ladder",
    "BANDS",
    "MATCH_TOL",
    "PosteriorSummary",
    "band_index",
    "by_band",
    "couplings",
    "detection_rate",
    "false_absence",
    "matches",
    "predictive_signals",
    "residual_distribution",
    "summarize",
    "RHAT_PARAMETERS",
    "Agreement",
    "EnsembleResult",
    "EnsembleRunner",
    "derive_seeds",
    "load_ensembles",
    "merge_traces",
    "rhat",
    "spread_across_k",
    "DEFAULT_QOS",
    "KNOWN_ALGORITHMS",
    "PERLMUTTER_ACCOUNT",
    "PERLMUTTER_WORKDIR",
    "RunConfig",
    "merge_run",
    "run_ensemble",
    "submission_script",
    "WassersteinL2",
    "signal_measure",
    "wasserstein_signal_distance",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_exported(name):
    pkg = importlib.import_module("nuclear_spin_recovery")
    assert hasattr(pkg, name), f"{name} missing from package namespace"


def test_all_is_sorted_and_complete():
    pkg = importlib.import_module("nuclear_spin_recovery")
    assert pkg.__all__ == sorted(pkg.__all__)
    for name in pkg.__all__:
        assert hasattr(pkg, name)


def test_abstract_bases_are_abstract():
    """The ABCs must refuse direct instantiation, or subclassing is pointless."""
    from nuclear_spin_recovery import Envelope, ForwardModel, Likelihood

    for cls in (ForwardModel, Likelihood, Envelope):
        with pytest.raises(TypeError):
            cls()


def test_concrete_classes_subclass_their_base():
    from nuclear_spin_recovery import (
        AnalyticCCE1,
        Envelope,
        ForwardModel,
        GaussianL2,
        Likelihood,
        StretchedExponential,
    )

    assert issubclass(AnalyticCCE1, ForwardModel)
    assert issubclass(GaussianL2, Likelihood)
    assert issubclass(StretchedExponential, Envelope)


def test_sampler_abstract_bases_are_abstract():
    """Phase 2 ABCs must refuse instantiation."""
    from nuclear_spin_recovery import Algorithm, Proposal

    for cls in (Algorithm, Proposal):
        with pytest.raises(TypeError):
            cls()


def test_sampler_concrete_classes_subclass_their_base():
    from nuclear_spin_recovery import (
        RWMH,
        Algorithm,
        ContinuousReflected,
        DiscreteLatticeWalk,
        Proposal,
    )

    assert issubclass(RWMH, Algorithm)
    assert issubclass(ContinuousReflected, Proposal)
    assert issubclass(DiscreteLatticeWalk, Proposal)


def test_rwmh_holds_its_block_and_proposal():
    """Constructor wiring across the phase 2 modules."""
    import numpy as np

    from nuclear_spin_recovery import (
        RWMH,
        DiscreteLatticeWalk,
        NeighborIndex,
        ParameterBlock,
    )

    block = ParameterBlock("sites")
    proposal = DiscreteLatticeWalk(NeighborIndex(np.zeros((3, 3)), radius=1.0))
    mover = RWMH(block, proposal)
    assert mover.block is block
    assert mover.proposal is proposal


def test_target_holds_its_components(tiny_site_table):
    import numpy as np

    from nuclear_spin_recovery import (
        AnalyticCCE1,
        Experiment,
        ExperimentSet,
        GaussianL2,
        StretchedExponential,
        Target,
    )

    eset = ExperimentSet([Experiment(tau=np.array([1e-3]), n_pulses=8, b_z=311.0)])
    model = AnalyticCCE1(StretchedExponential())
    lik = GaussianL2()
    target = Target(eset, model, lik, tiny_site_table)
    assert target.expset is eset
    assert target.model is model
    assert target.likelihood is lik
    assert target.site_table is tiny_site_table


def test_analytic_model_accepts_an_envelope():
    """Constructor wiring: the model holds the envelope it was given."""
    from nuclear_spin_recovery import AnalyticCCE1, StretchedExponential

    env = StretchedExponential()
    assert AnalyticCCE1(env).envelope is env


def test_phase3_classes_subclass_their_base():
    from nuclear_spin_recovery import (
        RJMCMC,
        Algorithm,
        ContinuousReflected,
        GaussianOffset,
        ParallelTempering,
        Proposal,
    )

    assert issubclass(RJMCMC, Algorithm)
    assert issubclass(ParallelTempering, Algorithm)
    assert issubclass(GaussianOffset, ContinuousReflected)
    assert issubclass(GaussianOffset, Proposal)


def test_driver_wiring_holds_its_schedule():
    """A Step holds an algorithm; a Schedule holds Steps; a driver holds one."""
    from nuclear_spin_recovery import (
        RWMH,
        ContinuousReflected,
        HybridDriver,
        ParameterBlock,
        Schedule,
        Step,
    )

    algo = RWMH(ParameterBlock("lam"), ContinuousReflected(0.1, 0.0, 1.0))
    step = Step(algo, 5)
    schedule = Schedule([step])
    driver = HybridDriver(schedule)
    assert step.algorithm is algo
    assert schedule.steps[0] is step
    assert driver.schedule is schedule


def test_tempering_holds_its_inner_schedule():
    """PT wraps a Schedule, not a single algorithm -- a rung advances several
    blocks before a swap."""
    from nuclear_spin_recovery import (
        RWMH,
        ContinuousReflected,
        ParallelTempering,
        ParameterBlock,
        Schedule,
        Step,
    )

    inner = Schedule([Step(RWMH(ParameterBlock("lam"),
                                ContinuousReflected(0.1, 0.0, 1.0)), 1)])
    ladder = ParallelTempering(inner, n_replicas=4)
    assert ladder.inner is inner
    assert ladder.n_replicas == 4


def test_rjmcmc_holds_its_kernel():
    from nuclear_spin_recovery import RJMCMC, BirthDeathKernel, ParameterBlock

    kernel = BirthDeathKernel(k_max=20)
    mover = RJMCMC(ParameterBlock("sites"), kernel)
    assert mover.kernel is kernel
    assert kernel.k_max == 20


def test_nv2_table_is_present(nv2_path):
    """The committed hyperfine table ships with the repo."""
    assert nv2_path.exists()
    assert nv2_path.stat().st_size > 1_000_000


def test_post_subpackage_reexports_its_names():
    """post/ is importable on its own, as the ensemble runner will use it."""
    from nuclear_spin_recovery import post

    assert post.__all__ == sorted(post.__all__)
    for name in post.__all__:
        assert hasattr(post, name), f"{name} missing from post namespace"


def test_importing_post_does_not_import_matplotlib():
    """A compute node writing summaries must not need a plotting stack.

    Run in a subprocess: matplotlib may already be in this interpreter because
    another test imported it, which would make an in-process check vacuous.
    """
    import os
    import pathlib
    import subprocess
    import sys

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = (
        "import sys; import nuclear_spin_recovery.post; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr or "matplotlib was imported"
