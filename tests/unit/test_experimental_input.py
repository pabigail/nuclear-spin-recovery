"""Fitting and reading a posterior for data that was not simulated.

Everything the sampler needs already works on measured arrays — this file
covers the two places where the absence of ground truth changes what is
possible, and the two places where a measured trace can be silently wrong.

**The noise must be supplied.** Criterion A is the residual in units of the
noise, and it is the only criterion available without ground truth. The noise
cannot be recovered from a single dynamical-decoupling trace: at this sampling
density every estimator is floored by the modulation. Measured at a true sigma
of 0.002 — successive differences 0.062, second differences 0.016, decayed-tail
scatter 0.017. So the package requires it rather than guessing.

**The units must be stated.** Interpulse spacings are quoted in microseconds as
often as milliseconds. A default would accept a trace off by a thousand and
return a confident fit to the wrong physics.

Spec §9.1; docs/phase-4-plan.md unit 4e.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    TAU_UNITS,
    AnalyticCCE1,
    Experiment,
    ExperimentSet,
    State,
    StretchedExponential,
    Trace,
    coupling_posterior,
    simulate_dataset,
)
from nuclear_spin_recovery.post import couplings, summarize

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

TAU_US = np.linspace(0.05, 8.0, 40)          # a user's array, in microseconds


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def coherence(tiny_site_table, model):
    """Stand-in for a measurement: an array, with its provenance discarded."""
    truth = State.from_sites(
        [0, 2], n_sites=len(tiny_site_table), n_exp=1,
        lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
        sigma=np.array([[0.004]]), k_max=8)
    blank = ExperimentSet([Experiment(tau=TAU_US / 1000.0, n_pulses=16,
                                      b_z=311.0)])
    return simulate_dataset(truth, blank, tiny_site_table, model, sigma=0.004,
                            rng=np.random.default_rng(0)).data_all


@pytest.fixture
def measured(coherence):
    return ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                     n_pulses=16, b_z=311.0, sigma=0.004,
                                     tau_units="us")


# --------------------------------------------------------------------------
# building an experiment from arrays
# --------------------------------------------------------------------------


def test_from_arrays_builds_an_experiment_set(measured):
    assert isinstance(measured, ExperimentSet)
    assert measured.n_experiments == 1
    assert measured.n_points == TAU_US.size


def test_the_data_is_attached(measured, coherence):
    assert np.allclose(measured.data_all, coherence)


def test_microseconds_are_converted_to_milliseconds(measured):
    """The package works in ms; the user need not."""
    assert measured.experiments[0].tau == pytest.approx(TAU_US / 1000.0)


@pytest.mark.parametrize("unit", list(TAU_UNITS))
def test_every_accepted_unit_scales_correctly(coherence, unit):
    built = ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                      n_pulses=16, b_z=311.0, sigma=0.004,
                                      tau_units=unit)
    assert built.experiments[0].tau == pytest.approx(TAU_US * TAU_UNITS[unit])


def test_tau_units_has_no_default(coherence):
    """A default would silently accept a trace off by a thousand."""
    with pytest.raises(TypeError):
        ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                  n_pulses=16, b_z=311.0, sigma=0.004)


def test_an_unknown_unit_raises_naming_the_known_ones(coherence):
    with pytest.raises(ValueError) as excinfo:
        ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence, n_pulses=16,
                                  b_z=311.0, sigma=0.004, tau_units="minutes")
    assert any(u in str(excinfo.value) for u in TAU_UNITS)


def test_sigma_is_required(coherence):
    """Criterion A is meaningless without it, and it cannot be estimated."""
    with pytest.raises(TypeError):
        ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                  n_pulses=16, b_z=311.0, tau_units="us")


def test_a_non_positive_sigma_raises(coherence):
    for bad in (0.0, -0.001):
        with pytest.raises(ValueError):
            ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                      n_pulses=16, b_z=311.0, sigma=bad,
                                      tau_units="us")


def test_mismatched_lengths_raise(coherence):
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=TAU_US[:-1], coherence=coherence,
                                  n_pulses=16, b_z=311.0, sigma=0.004,
                                  tau_units="us")


def test_a_nan_in_the_data_raises(coherence):
    """Dropped points are common in real traces and must not reach a likelihood.

    A nan propagates to the log-likelihood, where every acceptance ratio
    becomes nan and the chain stops moving without ever reporting a failure.
    """
    spoiled = np.array(coherence, dtype=float)
    spoiled[5] = np.nan
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=TAU_US, coherence=spoiled, n_pulses=16,
                                  b_z=311.0, sigma=0.004, tau_units="us")


def test_a_nan_in_tau_raises(coherence):
    spoiled = np.array(TAU_US, dtype=float)
    spoiled[0] = np.nan
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=spoiled, coherence=coherence,
                                  n_pulses=16, b_z=311.0, sigma=0.004,
                                  tau_units="us")


def test_unsorted_tau_raises(coherence):
    """Out-of-order spacings mean the arrays were assembled wrongly."""
    shuffled = np.array(TAU_US, dtype=float)
    shuffled[[3, 9]] = shuffled[[9, 3]]
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=shuffled, coherence=coherence,
                                  n_pulses=16, b_z=311.0, sigma=0.004,
                                  tau_units="us")


def test_a_non_positive_tau_raises(coherence):
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=np.r_[0.0, TAU_US[1:]],
                                  coherence=coherence, n_pulses=16, b_z=311.0,
                                  sigma=0.004, tau_units="us")


# --------------------------------------------------------------------------
# several measured experiments
# --------------------------------------------------------------------------


def test_from_records_builds_several_experiments(coherence):
    built = ExperimentSet.from_records([
        {"tau": TAU_US, "coherence": coherence, "n_pulses": 8, "b_z": 311.0,
         "sigma": 0.004},
        {"tau": TAU_US[:20], "coherence": coherence[:20], "n_pulses": 16,
         "b_z": 311.0, "sigma": 0.006},
    ], tau_units="us")
    assert built.n_experiments == 2
    assert built.n_points == TAU_US.size + 20


def test_records_may_differ_in_pulse_number_and_noise(coherence):
    built = ExperimentSet.from_records([
        {"tau": TAU_US, "coherence": coherence, "n_pulses": 8, "b_z": 311.0,
         "sigma": 0.004},
        {"tau": TAU_US, "coherence": coherence, "n_pulses": 16, "b_z": 250.0,
         "sigma": 0.006},
    ], tau_units="us")
    assert [e.n_pulses for e in built.experiments] == [8, 16]
    assert [e.sigma for e in built.experiments] == [0.004, 0.006]


def test_an_empty_record_list_raises():
    with pytest.raises(ValueError):
        ExperimentSet.from_records([], tau_units="us")


def test_a_record_missing_a_key_raises_naming_it(coherence):
    with pytest.raises((KeyError, ValueError), match="sigma"):
        ExperimentSet.from_records(
            [{"tau": TAU_US, "coherence": coherence, "n_pulses": 8,
              "b_z": 311.0}], tau_units="us")


# --------------------------------------------------------------------------
# the posterior, read without ground truth
# --------------------------------------------------------------------------


@pytest.fixture
def trace_of(tiny_site_table):
    def build(configs):
        tr = Trace(n_sites=len(tiny_site_table), k_max=8, n_exp=1)
        for sites in configs:
            st = State.from_sites(
                np.sort(np.asarray(sites, dtype=int)),
                n_sites=len(tiny_site_table), n_exp=1,
                lam=np.array([[3e-3]]), n_stretch=np.array([[1.0]]),
                sigma=np.array([[0.02]]), k_max=8)
            tr.append(st, log_prob=-1.0)
        return tr
    return build


def test_coupling_posterior_ranks_by_frequency(tiny_site_table):
    """The reference-free reading: what is in there, and how often."""
    samples = ([couplings(tiny_site_table, [0, 2])] * 8
               + [couplings(tiny_site_table, [0, 3])] * 2)
    found, frequency = coupling_posterior(samples)
    assert len(found) == len(frequency)
    assert frequency[0] == pytest.approx(1.0)        # site 0 in every sample
    assert np.all(np.diff(frequency) <= 1e-12)       # sorted, decreasing


def test_coupling_posterior_reports_the_actual_couplings(tiny_site_table):
    samples = [couplings(tiny_site_table, [2])] * 5
    found, frequency = coupling_posterior(samples)
    assert np.asarray(found).shape == (1, 2)
    assert found[0] == pytest.approx([tiny_site_table.a_par[2],
                                      tiny_site_table.a_perp[2]])
    assert frequency[0] == pytest.approx(1.0)


def test_a_symmetry_orbit_is_reported_once(tiny_site_table):
    """Sites 0 and 1 are one physical answer, not two."""
    samples = ([couplings(tiny_site_table, [0])] * 5
               + [couplings(tiny_site_table, [1])] * 5)
    found, frequency = coupling_posterior(samples)
    assert len(found) == 1
    assert frequency[0] == pytest.approx(1.0)


def test_coupling_posterior_of_nothing_is_empty():
    found, frequency = coupling_posterior([])
    assert np.asarray(found).size == 0
    assert np.asarray(frequency).size == 0


def test_summarize_without_a_reference_still_scores_the_fit(measured,
                                                            tiny_site_table,
                                                            model, trace_of):
    """Criterion A survives the absence of ground truth; criterion B does not."""
    summary = summarize(trace_of([[0, 2]] * 4), measured, tiny_site_table,
                        model, reference=None, noise=0.004)
    assert np.asarray(summary.R_i).size == 0
    assert np.isfinite(summary.best_residual)
    assert summary.k_mode == 2


# --------------------------------------------------------------------------
# plotting a posterior with nothing to compare it to
# --------------------------------------------------------------------------


def test_detection_by_band_refuses_a_reference_free_summary(measured,
                                                            tiny_site_table,
                                                            model, trace_of):
    """Empty axes would read as 'asked and answered no'. It was never asked."""
    from nuclear_spin_recovery.post import plots

    summary = summarize(trace_of([[0, 2]]), measured, tiny_site_table, model,
                        reference=None, noise=0.004)
    with pytest.raises(ValueError, match="reference|coupling_posterior"):
        plots.plot_detection_by_band(summary)


def test_posterior_predictive_plot_returns_an_axes(measured, tiny_site_table,
                                                   model, trace_of):
    from nuclear_spin_recovery.post import plots

    summary = summarize(trace_of([[0, 2], [0, 3]]), measured, tiny_site_table,
                        model, reference=None, noise=0.004)
    ax = plots.plot_posterior_predictive(summary, TAU_US / 1000.0,
                                         measured.data_all)
    assert hasattr(ax, "plot")


def test_coupling_posterior_plot_returns_an_axes(tiny_site_table):
    from nuclear_spin_recovery.post import plots

    samples = [couplings(tiny_site_table, [0, 2])] * 3
    found, frequency = coupling_posterior(samples)
    assert hasattr(plots.plot_coupling_posterior(found, frequency), "plot")


def test_coupling_posterior_plot_truncates_to_the_top_entries(tiny_site_table):
    """A real posterior carries hundreds of couplings; the plot shows the few."""
    from nuclear_spin_recovery.post import plots

    samples = [couplings(tiny_site_table, [0, 2, 3])] * 3
    found, frequency = coupling_posterior(samples)
    ax = plots.plot_coupling_posterior(found, frequency, top=2)
    assert len(ax.patches) == 2


# --------------------------------------------------------------------------
# per-point measurement weights on a measured trace
# --------------------------------------------------------------------------


def test_from_arrays_accepts_a_weight(coherence):
    """The output of an adaptive design, or uneven repetition counts."""
    weight = np.linspace(1.0, 4.0, TAU_US.size)
    built = ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                      n_pulses=16, b_z=311.0, sigma=0.004,
                                      tau_units="us", weight=weight)
    assert built.weight_all == pytest.approx(weight)


def test_weight_is_optional(measured):
    assert measured.experiments[0].weight is None
    assert measured.weight_all == pytest.approx(np.ones(TAU_US.size))


def test_a_mismatched_weight_length_raises(coherence):
    with pytest.raises(ValueError):
        ExperimentSet.from_arrays(tau=TAU_US, coherence=coherence,
                                  n_pulses=16, b_z=311.0, sigma=0.004,
                                  tau_units="us", weight=np.ones(5))


def test_records_may_mix_weighted_and_unweighted(coherence):
    built = ExperimentSet.from_records([
        {"tau": TAU_US, "coherence": coherence, "n_pulses": 8, "b_z": 311.0,
         "sigma": 0.004, "weight": np.full(TAU_US.size, 3.0)},
        {"tau": TAU_US, "coherence": coherence, "n_pulses": 16, "b_z": 311.0,
         "sigma": 0.004},
    ], tau_units="us")
    out = built.weight_all
    assert out[:TAU_US.size] == pytest.approx(3.0)
    assert out[TAU_US.size:] == pytest.approx(1.0)
