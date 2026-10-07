"""Summaries of a relaxed run: where the couplings went, and against what.

Detection deliberately ignores offsets, so that a relaxed coupling drifting
past the match tolerance is not scored as a missed site.  These summaries are
the other half: the relaxed values themselves, the configuration to report
when the number of spins is free, and the comparison with an independent
measurement that says whether relaxing helped.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    State,
    Trace,
    compare_couplings,
    modal_configuration,
    relaxed_couplings,
)

K_MAX = 3


def make_trace(table, rows):
    """A trace from ``(sites, d_par, d_perp, log_prob, label)`` rows."""
    trace = Trace(n_sites=len(table), k_max=K_MAX, n_exp=1)
    for sites, d_par, d_perp, log_prob, label in rows:
        state = State.from_sites(
            sites, n_sites=len(table), n_exp=1, lam=np.array([[3e-3]]),
            n_stretch=np.ones((1, 1)), sigma=np.array([[0.1]]), k_max=K_MAX)
        state.dA_par[0, : len(sites)] = d_par
        state.dA_perp[0, : len(sites)] = d_perp
        trace.append(state, np.array([log_prob]), label)
    return trace


@pytest.fixture
def trace(tiny_site_table):
    return make_trace(tiny_site_table, [
        ((0,), [9.0], [9.0], -50.0, "a"),               # burn-in
        ((0, 2), [1.0, 4.0], [0.0, -1.0], -3.0, "a"),
        ((0, 2), [3.0, 4.0], [0.0, -3.0], -1.0, "b"),
        ((2, 0), [6.0, 2.0], [-2.0, 0.0], -2.0, "b"),   # same set, slots swapped
        ((0, 2, 3), [2.0, 4.0, 0.5], [0.0, -2.0, 0.0], -0.5, "a"),
    ])


# ------------------------------------------------------- relaxed couplings

def test_occupancy_is_the_share_of_draws(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert out.occupancy == pytest.approx([1.0, 0.0, 1.0, 0.25])


def test_relaxed_value_is_table_plus_mean_offset(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert out.d_par[0] == pytest.approx(np.mean([1.0, 3.0, 2.0, 2.0]))
    assert out.a_par[0] == pytest.approx(tiny_site_table.a_par[0] + 2.0)
    assert out.d_perp[2] == pytest.approx(np.mean([-1.0, -3.0, -2.0, -2.0]))
    assert out.a_perp[2] == pytest.approx(tiny_site_table.a_perp[2] - 2.0)


def test_offsets_follow_the_site_not_the_slot(trace, tiny_site_table):
    """The fourth draw holds site 2 in slot 0; its offset is still site 2's."""
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert out.d_par[2] == pytest.approx(np.mean([4.0, 4.0, 6.0, 4.0]))


def test_spread_is_over_occupied_draws(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert out.a_par_std[0] == pytest.approx(np.std([1.0, 3.0, 2.0, 2.0]))
    assert out.a_par_std[3] == pytest.approx(0.0)


def test_a_site_never_occupied_reports_nan(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert np.isnan(out.a_par[1]) and np.isnan(out.a_perp_std[1])


def test_burn_in_is_discarded(trace, tiny_site_table):
    kept = relaxed_couplings(trace, tiny_site_table, burn=1)
    everything = relaxed_couplings(trace, tiny_site_table)
    assert everything.d_par[0] > kept.d_par[0]


def test_occupied_applies_a_threshold(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    assert list(out.occupied()) == [0, 2]
    assert list(out.occupied(0.2)) == [0, 2, 3]


def test_discarding_everything_raises(trace, tiny_site_table):
    with pytest.raises(ValueError, match="nothing"):
        relaxed_couplings(trace, tiny_site_table, burn=5)


# ------------------------------------------------------ modal configuration

def test_modal_set_is_the_most_visited_not_the_best_fitting(trace):
    """The best-fitting draw has an extra spin, as it usually will."""
    modal = modal_configuration(trace, burn=1)
    assert modal.sites == (0, 2)
    assert modal.share == pytest.approx(0.75)


def test_modal_draw_is_the_best_on_that_set(trace):
    modal = modal_configuration(trace, burn=1)
    assert modal.step == 2
    assert list(modal.d_par) == [3.0, 4.0]
    assert list(modal.d_perp) == [0.0, -3.0]


def test_modal_offsets_are_in_site_order(tiny_site_table):
    trace = make_trace(tiny_site_table, [
        ((2, 0), [6.0, 2.0], [-2.0, 0.0], -1.0, "a")])
    modal = modal_configuration(trace)
    assert modal.sites == (0, 2)
    assert list(modal.d_par) == [2.0, 6.0]


# ---------------------------------------------------------------- compare

def test_relaxation_is_scored_against_the_measurement(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    measured = [(tiny_site_table.a_par[0] + 2.0, tiny_site_table.a_perp[0]),
                (tiny_site_table.a_par[2] + 4.5, tiny_site_table.a_perp[2] - 2.0)]
    table = compare_couplings(out, tiny_site_table, measured)
    assert list(table["site"]) == [0, 2]
    assert table["dft_error"] == pytest.approx([2.0, np.hypot(4.5, 2.0)])
    assert table["relaxed_error"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert table["relaxed"][0] == pytest.approx(measured[0])


def test_pairing_is_made_on_the_table_values(tiny_site_table):
    """Sites 0 and 2 share A_par and differ in A_perp by 35 kHz.  Site 2 has
    relaxed almost onto a measurement that belongs to site 0.  Pairing on the
    relaxed values would hand it that measurement and report a success."""
    trace = make_trace(tiny_site_table, [
        ((0, 2), [0.0, 0.0], [-20.0, 34.0], -1.0, "a")])
    out = relaxed_couplings(trace, tiny_site_table)
    measured = [(120.0, 45.0)]
    table = compare_couplings(out, tiny_site_table, measured)
    assert table["site"][0] == 0
    assert table["relaxed_error"][0] == pytest.approx(20.0)


def test_pairing_can_be_given(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    table = compare_couplings(out, tiny_site_table, [(120.0, 45.0)], sites=[2])
    assert table["site"][0] == 2
    assert table["dft"][0] == pytest.approx([120.0, 10.0])


def test_each_site_is_paired_once(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    table = compare_couplings(out, tiny_site_table,
                              [(120.0, 44.0), (120.0, 46.0)])
    assert sorted(table["site"]) == [0, 2]


def test_a_measurement_with_no_site_left_is_unpaired(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    table = compare_couplings(out, tiny_site_table,
                              [(120.0, 45.0), (120.0, 10.0), (60.0, 60.0)])
    assert sorted(table["site"]) == [-1, 0, 2]
    lost = int(np.flatnonzero(table["site"] == -1)[0])
    assert np.isnan(table["relaxed_error"][lost])
    assert np.isnan(table["dft"][lost]).all()


def test_measurements_must_be_pairs(trace, tiny_site_table):
    out = relaxed_couplings(trace, tiny_site_table, burn=1)
    with pytest.raises(ValueError, match="pairs"):
        compare_couplings(out, tiny_site_table, [(1.0, 2.0, 3.0)])
    with pytest.raises(ValueError, match="one site per measurement"):
        compare_couplings(out, tiny_site_table, [(1.0, 2.0)], sites=[0, 2])


# ------------------------------------------------------------------ stripes

def test_stripes_draw_one_band_per_run_of_an_algorithm(trace):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from nuclear_spin_recovery.post.plots import plot_algorithm_stripes

    ax = plot_algorithm_stripes(trace.log_prob, trace.algorithm)
    assert len(ax.patches) == 3                       # a a | b b | a
    assert sorted(ax.get_legend_handles_labels()[1]) == ["a", "b"]
    (line,) = ax.get_lines()
    assert list(line.get_ydata()) == pytest.approx([50.0, 3.0, 1.0, 2.0, 0.5])
    assert ax.get_yscale() == "log"


def test_stripes_can_show_a_window(trace):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from nuclear_spin_recovery.post.plots import plot_algorithm_stripes

    ax = plot_algorithm_stripes(trace.log_prob, trace.algorithm, first=1, last=4,
                                colours={"a": "tab:red"})
    assert len(ax.patches) == 2
    assert list(ax.get_lines()[0].get_xdata()) == [1, 2, 3]
    with pytest.raises(ValueError, match="empty window"):
        plot_algorithm_stripes(trace.log_prob, trace.algorithm, first=4, last=2)
