"""Posterior summaries: detection, residual, dimension, bands, diagnostics.

These tests encode the measurement errors that produced docs/test-plan.md, so
that the extraction of the metrics out of tests/theory/conftest.py cannot lose
them:

- detection computed over the posterior, never over one configuration;
- matched on couplings, never on site index;
- offsets read from the trace, never rebuilt as zero;
- each Trace array read once, not per step.

Spec Sec. 9.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    PosteriorSummary,
    State,
    StretchedExponential,
    Trace,
    simulate_dataset,
)
from nuclear_spin_recovery.post import (
    BANDS,
    MATCH_TOL,
    band_index,
    by_band,
    couplings,
    detection_rate,
    false_absence,
    matches,
    predictive_signals,
    residual_distribution,
    summarize,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


def make_state(sites, table, *, k_max=8, lam=3e-3, sigma=0.02, n_exp=1):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam),
        n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma),
        k_max=k_max,
    )


@pytest.fixture
def expset():
    return ExperimentSet(
        [Experiment(tau=np.linspace(1e-4, 8e-3, 40), n_pulses=16, b_z=311.0)]
    )


@pytest.fixture
def trace_of(tiny_site_table):
    """Build a Trace from an explicit list of configurations."""
    def build(configs, k_max=8, offsets=None):
        tr = Trace(n_sites=len(tiny_site_table), k_max=k_max, n_exp=1)
        for j, sites in enumerate(configs):
            st = make_state(sites, tiny_site_table, k_max=k_max)
            if offsets is not None:
                dpar, dperp = offsets[j]
                st.dA_par[0, : len(dpar)] = dpar
                st.dA_perp[0, : len(dperp)] = dperp
            tr.append(st, log_prob=-float(j))
        return tr
    return build


# --------------------------------------------------------------------------
# coupling matching -- the reason site index is never the unit
# --------------------------------------------------------------------------


def test_couplings_returns_one_pair_per_site(tiny_site_table):
    out = couplings(tiny_site_table, [0, 2])
    assert len(out) == 2


def test_couplings_includes_offsets_when_given(tiny_site_table):
    plain = couplings(tiny_site_table, [2])
    shifted = couplings(tiny_site_table, [2], dA_par=[5.0], dA_perp=[0.0])
    assert plain != shifted


def test_couplings_without_offsets_equals_zero_offsets(tiny_site_table):
    """The relaxed model must reduce exactly to the constrained one."""
    plain = couplings(tiny_site_table, [0, 2])
    zeroed = couplings(tiny_site_table, [0, 2], dA_par=[0.0, 0.0],
                       dA_perp=[0.0, 0.0])
    assert plain == zeroed


def test_matches_within_tolerance():
    assert matches((120.0, 45.0), [(120.05, 45.02)], tol=0.1)


def test_does_not_match_outside_tolerance():
    assert not matches((120.0, 45.0), [(120.5, 45.0)], tol=0.1)


def test_tolerance_applies_to_both_components():
    """A near miss on A_perp alone is still a miss."""
    assert not matches((120.0, 45.0), [(120.0, 45.5)], tol=0.1)


def test_default_tolerance_is_a_tenth_of_a_kilohertz():
    assert MATCH_TOL == pytest.approx(0.1)


# --------------------------------------------------------------------------
# detection rate -- over the posterior, never one configuration
# --------------------------------------------------------------------------


def test_detection_rate_is_one_when_always_present(tiny_site_table):
    ref = couplings(tiny_site_table, [0])
    samples = [couplings(tiny_site_table, [0, 2])] * 5
    assert detection_rate(samples, list(ref)) == pytest.approx([1.0])


def test_detection_rate_is_zero_when_never_present(tiny_site_table):
    ref = couplings(tiny_site_table, [3])
    samples = [couplings(tiny_site_table, [0, 2])] * 5
    assert detection_rate(samples, list(ref)) == pytest.approx([0.0])


def test_detection_rate_is_the_fraction_of_samples(tiny_site_table):
    ref = list(couplings(tiny_site_table, [2]))
    samples = [couplings(tiny_site_table, [0, 2])] * 3 + \
              [couplings(tiny_site_table, [0])] * 1
    assert detection_rate(samples, ref) == pytest.approx([0.75])


def test_symmetry_equivalent_site_scores_as_detected(tiny_site_table):
    """Sites 0 and 1 have identical couplings and are physically the same answer.

    Scoring by site index would call this a miss.  That is the error this whole
    module exists to prevent.
    """
    ref = list(couplings(tiny_site_table, [0]))
    samples = [couplings(tiny_site_table, [1])] * 4
    assert detection_rate(samples, ref) == pytest.approx([1.0])


def test_detection_rate_one_entry_per_reference_spin(tiny_site_table):
    ref = list(couplings(tiny_site_table, [0, 2, 3]))
    samples = [couplings(tiny_site_table, [0, 2])] * 2
    assert np.asarray(detection_rate(samples, ref)).shape == (3,)


def test_detection_rate_of_empty_reference_is_empty(tiny_site_table):
    samples = [couplings(tiny_site_table, [0])]
    assert np.asarray(detection_rate(samples, [])).size == 0


# --------------------------------------------------------------------------
# false absence -- FP of spec 9.2, explicitly not a false-positive rate
# --------------------------------------------------------------------------


def test_false_absence_is_zero_when_every_sample_is_the_mode(tiny_site_table):
    modal = couplings(tiny_site_table, [0, 2])
    assert false_absence([modal] * 5, modal) == pytest.approx(0.0)


def test_false_absence_is_one_when_modal_spins_never_recur(tiny_site_table):
    modal = couplings(tiny_site_table, [0])
    others = [couplings(tiny_site_table, [3])] * 4
    assert false_absence(others, modal) == pytest.approx(1.0)


def test_false_absence_of_an_empty_mode_is_zero(tiny_site_table):
    assert false_absence([couplings(tiny_site_table, [0])], set()) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# coupling-magnitude bands
# --------------------------------------------------------------------------


def test_bands_are_the_test_plan_bands():
    assert BANDS == ((5.0, 25.0), (25.0, 100.0), (100.0, 750.0))


def test_band_index_places_thirty_kilohertz_in_the_middle_band():
    assert band_index(30.0) == 1


def test_band_index_places_ten_kilohertz_in_the_lowest_band():
    assert band_index(10.0) == 0


def test_band_index_places_two_hundred_in_the_asserted_band():
    assert band_index(200.0) == 2


def test_band_lower_edge_is_inclusive():
    assert band_index(25.0) == 1


def test_magnitude_outside_every_band_returns_minus_one():
    assert band_index(1.0) == -1
    assert band_index(2000.0) == -1


def test_by_band_averages_within_each_band():
    values = np.array([0.0, 1.0, 0.5])
    magnitude = np.array([10.0, 200.0, 300.0])
    out = by_band(values, magnitude)
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.75)


def test_empty_band_is_nan_not_zero():
    """An empty band is missing data; averaging it as zero understates detection."""
    out = by_band(np.array([1.0]), np.array([200.0]))
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# predictive signals and residual
# --------------------------------------------------------------------------


def test_predictive_returns_one_signal_per_draw(tiny_site_table, expset, model):
    tr = Trace(n_sites=len(tiny_site_table), k_max=8, n_exp=1)
    for sites in ([0, 2], [1, 2], [0, 3]):
        tr.append(make_state(sites, tiny_site_table), log_prob=-1.0)
    out = predictive_signals(tr, expset, tiny_site_table, model)
    assert np.asarray(out).shape == (3, expset.n_points)


def test_predictive_respects_stride(tiny_site_table, expset, model):
    tr = Trace(n_sites=len(tiny_site_table), k_max=8, n_exp=1)
    for _ in range(10):
        tr.append(make_state([0, 2], tiny_site_table), log_prob=-1.0)
    out = predictive_signals(tr, expset, tiny_site_table, model, stride=5)
    assert np.asarray(out).shape[0] == 2


def test_offsets_reach_the_predictive_signal(tiny_site_table, expset, model, trace_of):
    """The bug this catches scored every relaxed run as if never relaxed.

    Rebuilding a configuration from site indices alone pins the offsets at
    zero, so a relaxed chain reads as a constrained one.
    """
    plain = trace_of([[0, 2]])
    shifted = trace_of([[0, 2]], offsets=[([40.0, 40.0], [20.0, 20.0])])
    a = predictive_signals(plain, expset, tiny_site_table, model)
    b = predictive_signals(shifted, expset, tiny_site_table, model)
    assert not np.allclose(a, b)


def test_residual_of_an_exact_match_is_zero():
    signal = np.array([0.9, 0.8, 0.7])
    out = residual_distribution(signal, signal[None, :], noise=0.01)
    assert out == pytest.approx([0.0])


def test_residual_is_reported_in_units_of_the_noise():
    obs = np.zeros(4)
    pred = np.full((1, 4), 0.02)
    assert residual_distribution(obs, pred, noise=0.01) == pytest.approx([2.0])


def test_residual_one_entry_per_draw():
    obs = np.zeros(3)
    pred = np.zeros((7, 3))
    assert np.asarray(residual_distribution(obs, pred, noise=0.1)).shape == (7,)


# --------------------------------------------------------------------------
# PosteriorSummary
# --------------------------------------------------------------------------


def make_summary(**kw):
    defaults = {
        "R_i": np.array([1.0, 0.5, 0.0]),
        "magnitude": np.array([200.0, 50.0, 10.0]),
        "residual": np.array([3.0, 1.0, 5.0]),
        "k_posterior": np.array([2, 3, 3, 3, 4]),
        "false_absence": 0.25,
        "predictive": np.zeros((3, 4)),
    }
    defaults.update(kw)
    return PosteriorSummary(**defaults)


def test_summary_R_averages_within_a_band():
    assert make_summary().R(100.0, 750.0) == pytest.approx(1.0)


def test_summary_R_of_an_empty_band_is_nan():
    assert np.isnan(make_summary().R(700.0, 750.0))


def test_summary_by_band_has_one_entry_per_band():
    assert np.asarray(make_summary().by_band()).shape == (len(BANDS),)


def test_median_residual():
    assert make_summary().median_residual == pytest.approx(3.0)


def test_best_residual_is_the_minimum():
    assert make_summary().best_residual == pytest.approx(1.0)


def test_k_mode_is_the_most_frequent_dimension():
    assert make_summary().k_mode == 3


def test_dimension_discrepancy_is_the_absolute_difference():
    assert make_summary().dimension_discrepancy(5) == 2


def test_dimension_discrepancy_is_zero_when_correct():
    assert make_summary().dimension_discrepancy(3) == 0


# --------------------------------------------------------------------------
# summarize -- the end-to-end entry point
# --------------------------------------------------------------------------


@pytest.fixture
def simulated(tiny_site_table, expset, model):
    truth = make_state([0, 2], tiny_site_table)
    data = simulate_dataset(truth, expset, tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    return truth, data


def test_summarize_returns_a_posterior_summary(simulated, tiny_site_table,
                                               model, trace_of):
    _truth, data = simulated
    out = summarize(trace_of([[0, 2], [1, 2]]), data, tiny_site_table, model,
                    reference=[0, 2], noise=0.002)
    assert isinstance(out, PosteriorSummary)


def test_summarize_discards_burn_in(simulated, tiny_site_table, model, trace_of):
    _truth, data = simulated
    tr = trace_of([[0, 2]] * 10)
    out = summarize(tr, data, tiny_site_table, model, reference=[0, 2],
                    burn=6, noise=0.002)
    assert out.k_posterior.size == 4


def test_summarize_detects_the_truth_it_was_given(simulated, tiny_site_table,
                                                  model, trace_of):
    _truth, data = simulated
    out = summarize(trace_of([[0, 2]] * 4), data, tiny_site_table, model,
                    reference=[0, 2], noise=0.002)
    assert out.R_i == pytest.approx([1.0, 1.0])


def test_summarize_residual_near_zero_for_the_true_configuration(
        simulated, tiny_site_table, model, trace_of):
    """Criterion A: the truth must fit its own noiseless data to about 1 sigma."""
    _truth, data = simulated
    out = summarize(trace_of([[0, 2]] * 3), data, tiny_site_table, model,
                    reference=[0, 2], noise=0.002)
    assert out.best_residual < 2.0


def test_summarize_without_reference_gives_empty_detection(
        simulated, tiny_site_table, model, trace_of):
    """Experimental data has no ground truth, so R_i is unavailable, not zero."""
    _truth, data = simulated
    out = summarize(trace_of([[0, 2]]), data, tiny_site_table, model,
                    reference=None, noise=0.002)
    assert np.asarray(out.R_i).size == 0


def test_summarize_magnitude_matches_the_reference_spins(
        simulated, tiny_site_table, model, trace_of):
    _truth, data = simulated
    out = summarize(trace_of([[0, 2]]), data, tiny_site_table, model,
                    reference=[0, 2], noise=0.002)
    expected = np.hypot(tiny_site_table.a_par[[0, 2]],
                        tiny_site_table.a_perp[[0, 2]])
    assert out.magnitude == pytest.approx(expected)


def test_summarize_rejects_a_state(simulated, tiny_site_table, model):
    """There is deliberately no entry point that summarises one configuration.

    An explicit type check, not duck-typing: a State that failed partway through
    on a missing attribute would raise AttributeError from somewhere deep in the
    call, which reads as a bug rather than as the refusal it is.
    """
    truth, data = simulated
    with pytest.raises(TypeError):
        summarize(truth, data, tiny_site_table, model, reference=[0, 2])


def test_summarize_reads_each_trace_array_once(simulated, tiny_site_table,
                                               model, trace_of):
    """Reading a Trace property per step is O(n^2); it cost 38 s of a 45 s suite.

    The guard is on access count, not on wall clock, so it cannot go flaky on a
    slow machine.
    """
    _truth, data = simulated
    tr = trace_of([[0, 2]] * 40)
    counts = {}

    class Counting:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            counts[name] = counts.get(name, 0) + 1
            return getattr(self._inner, name)

        def __len__(self):
            return len(self._inner)

    summarize(Counting(tr), data, tiny_site_table, model, reference=[0, 2],
              noise=0.002)
    hot = {k: v for k, v in counts.items()
           if k in ("site_idx", "k", "dA_par", "dA_perp", "lam", "log_prob")}
    assert hot, "summarize read no trace arrays at all"
    assert max(hot.values()) <= 2, f"trace array read repeatedly: {hot}"


# --------------------------------------------------------------------------
# plots -- structure, never pixels
# --------------------------------------------------------------------------


def test_plot_residual_returns_an_axes():
    from nuclear_spin_recovery.post import plots

    ax = plots.plot_residual(np.arange(5), np.linspace(5, 1, 5))
    assert hasattr(ax, "plot")


def test_plot_residual_marks_the_burn_in():
    from nuclear_spin_recovery.post import plots

    ax = plots.plot_residual(np.arange(10), np.ones(10), burn=4)
    assert len(ax.lines) + len(ax.collections) + len(ax.patches) >= 2


def test_plot_residual_draws_into_a_given_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nuclear_spin_recovery.post import plots

    _, ax = plt.subplots()
    assert plots.plot_residual(np.arange(3), np.ones(3), ax=ax) is ax
    plt.close("all")


def test_plot_dimension_returns_an_axes():
    from nuclear_spin_recovery.post import plots

    ax = plots.plot_dimension(np.array([2, 3, 3, 4]), k_true=3)
    assert hasattr(ax, "plot")


def test_plot_parameter_returns_an_axes():
    from nuclear_spin_recovery.post import plots

    ax = plots.plot_parameter(np.linspace(1e-3, 5e-3, 20), truth=3e-3)
    assert hasattr(ax, "plot")


def test_plot_detection_by_band_returns_an_axes():
    from nuclear_spin_recovery.post import plots

    ax = plots.plot_detection_by_band(make_summary())
    assert hasattr(ax, "plot")


def test_plots_do_not_compute_metrics():
    """Drawing and computing stay separate, or neither can be tested alone."""
    import inspect

    from nuclear_spin_recovery.post import metrics, plots

    source = inspect.getsource(plots)
    for name in ("detection_rate", "residual_distribution", "summarize"):
        assert f"{name}(" not in source, f"plots.py computes {name}"
    assert metrics is not None


def test_matplotlib_is_not_imported_at_module_scope():
    """A compute node needs post/ for summaries without a plotting stack."""
    import ast

    from nuclear_spin_recovery.post import plots

    with open(plots.__file__) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom)) or \
            "matplotlib" not in ast.dump(node), \
            "matplotlib imported at module scope in plots.py"
