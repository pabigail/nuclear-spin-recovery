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
    "nuclear_spin_recovery.trace",
    "nuclear_spin_recovery.neighbors",
    "nuclear_spin_recovery.proposals",
    "nuclear_spin_recovery.algorithms",
    "nuclear_spin_recovery.algorithms.base",
    "nuclear_spin_recovery.algorithms.rwmh",
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
        Algorithm,
        ContinuousReflected,
        DiscreteLatticeWalk,
        Proposal,
        RWMH,
    )

    assert issubclass(RWMH, Algorithm)
    assert issubclass(ContinuousReflected, Proposal)
    assert issubclass(DiscreteLatticeWalk, Proposal)


def test_rwmh_holds_its_block_and_proposal():
    """Constructor wiring across the phase 2 modules."""
    import numpy as np

    from nuclear_spin_recovery import (
        DiscreteLatticeWalk,
        NeighborIndex,
        ParameterBlock,
        RWMH,
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


def test_nv2_table_is_present(nv2_path):
    """The committed hyperfine table ships with the repo."""
    assert nv2_path.exists()
    assert nv2_path.stat().st_size > 1_000_000
