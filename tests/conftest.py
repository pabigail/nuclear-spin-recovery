"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nuclear_spin_recovery import Experiment, ExperimentSet, SiteTable

REPO_ROOT = Path(__file__).resolve().parents[1]
NV2_PATH = REPO_ROOT / "nv-2.txt"


@pytest.fixture
def nv2_path():
    """Path to the committed NV hyperfine table."""
    return NV2_PATH


@pytest.fixture
def tiny_site_table():
    """A hand-built table with a known symmetry structure.

    Sites 0 and 1 share couplings exactly (a symmetry pair); site 2 differs
    in A_perp only; site 3 is far and weakly coupled.
    """
    return SiteTable(
        distance=np.array([1.5, 1.5, 2.5, 9.0]),
        positions=np.array(
            [[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 9.0]]
        ),
        a_par=np.array([120.0, 120.0, 120.0, 2.0]),
        a_perp=np.array([45.0, 45.0, 10.0, 1.0]),
        isotope=np.array(["13C", "13C", "13C", "13C"]),
        gyro=np.full(4, 6.7283),
    )


@pytest.fixture
def two_experiments():
    """An N=8 and an N=16 experiment with deliberately different grid lengths."""
    return ExperimentSet(
        [
            Experiment(tau=np.array([1e-3, 2e-3, 3e-3]), n_pulses=8, b_z=311.0),
            Experiment(
                tau=np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3]),
                n_pulses=16,
                b_z=311.0,
            ),
        ]
    )


@pytest.fixture
def single_experiment():
    return ExperimentSet(
        [Experiment(tau=np.linspace(1e-4, 8e-3, 50), n_pulses=16, b_z=311.0)]
    )
