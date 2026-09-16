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


def test_analytic_model_accepts_an_envelope():
    """Constructor wiring: the model holds the envelope it was given."""
    from nuclear_spin_recovery import AnalyticCCE1, StretchedExponential

    env = StretchedExponential()
    assert AnalyticCCE1(env).envelope is env


def test_nv2_table_is_present(nv2_path):
    """The committed hyperfine table ships with the repo."""
    assert nv2_path.exists()
    assert nv2_path.stat().st_size > 1_000_000
