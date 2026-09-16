"""Phase 1 gate: the forward model against a stored reference curve.

Marked slow: it loads the full 19,924-site table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    SiteTable,
    State,
    StretchedExponential,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "coherence_fig2a.npz"

pytestmark = pytest.mark.slow


def test_full_table_loads(nv2_path):
    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    assert 0 < len(table) < 19924


def test_reproduces_reference_curve(nv2_path):
    """Regression against a committed golden curve.

    TODO(phase-1): generate tests/fixtures/coherence_fig2a.npz from the
    reference implementation before implementing, so this test is a genuine
    regression rather than a restatement of the new code.
    """
    assert GOLDEN.exists(), f"golden fixture not generated: {GOLDEN}"
    ref = np.load(GOLDEN)

    table = SiteTable.from_ivady_file(nv2_path, strong_thresh=750.0, weak_thresh=5.0)
    eset = ExperimentSet(
        [Experiment(tau=ref["tau"], n_pulses=int(ref["n_pulses"]),
                    b_z=float(ref["b_z"]))]
    )
    st = State.from_sites(
        tuple(ref["sites"]),
        n_sites=len(table),
        n_exp=1,
        lam=np.array([[float(ref["lam"])]]),
        n_stretch=np.array([[float(ref["n_stretch"])]]),
        sigma=np.array([[0.1]]),
        k_max=64,
    )
    model = AnalyticCCE1(StretchedExponential())
    got = model.coherence(st, eset, table)[0]
    assert got == pytest.approx(ref["coherence"], rel=1e-10)
