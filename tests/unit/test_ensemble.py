"""Independent chains: seeds, pooling, persistence, and agreement.

Ensembles exist because a single chain cannot cross the dimension barrier
(docs/test-plan.md §5.6).  These tests therefore guard two things beyond the
arithmetic: that a run is reproducible from its config alone, which is what a
job array needs, and that the convergence diagnostic *reports* rather than
judges -- R-hat is structurally above 1 on this problem, and a conventional
gate would reject a correct posterior.

Spec §8.6; docs/phase-4-plan.md unit 4b.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    RHAT_PARAMETERS,
    RWMH,
    Agreement,
    AnalyticCCE1,
    ContinuousReflected,
    EnsembleResult,
    EnsembleRunner,
    Experiment,
    ExperimentSet,
    GaussianL2,
    ParameterBlock,
    PosteriorSummary,
    Schedule,
    State,
    Step,
    StretchedExponential,
    Target,
    Trace,
    derive_seeds,
    load_ensembles,
    merge_traces,
    rhat,
    simulate_dataset,
    spread_across_k,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

K_MAX = 8


def make_state(sites, table, *, lam=3e-3, sigma=0.02, k_max=K_MAX, n_exp=1):
    return State.from_sites(
        np.sort(np.asarray(list(sites), dtype=int)),
        n_sites=len(table), n_exp=n_exp,
        lam=np.full((1, n_exp), lam), n_stretch=np.ones((1, n_exp)),
        sigma=np.full((1, n_exp), sigma), k_max=k_max,
    )


@pytest.fixture
def expset():
    return ExperimentSet(
        [Experiment(tau=np.linspace(1e-4, 8e-3, 30), n_pulses=16, b_z=311.0)]
    )


@pytest.fixture
def target(tiny_site_table, expset):
    model = AnalyticCCE1(StretchedExponential())
    data = simulate_dataset(make_state([0, 2], tiny_site_table), expset,
                            tiny_site_table, model, sigma=0.002,
                            rng=np.random.default_rng(0))
    return Target(data, model, GaussianL2(), tiny_site_table)


@pytest.fixture
def schedule():
    return Schedule([
        Step(RWMH(ParameterBlock("lam"),
                  ContinuousReflected(radius=2e-4, lower=1e-4, upper=2e-2)), 5),
    ])


@pytest.fixture
def runner(schedule, tiny_site_table):
    def build(n_ensembles=4, n_steps=40, n_burn=10, k_values=(1, 3)):
        init = spread_across_k(
            lambda sites: make_state(sites, tiny_site_table), k_values)
        return EnsembleRunner(schedule, n_ensembles=n_ensembles,
                              n_steps=n_steps, n_burn=n_burn, init=init,
                              init_name="spread_across_k")
    return build


@pytest.fixture
def trace_of(tiny_site_table):
    """A Trace with recognisable content in every field."""
    def build(n=6, k_max=K_MAX, label="rwmh:lam", offsets=True):
        tr = Trace(n_sites=len(tiny_site_table), k_max=k_max, n_exp=1)
        for j in range(n):
            st = make_state([0, 2], tiny_site_table, lam=3e-3 + 1e-5 * j,
                            k_max=k_max)
            if offsets:
                st.dA_par[0, :2] = [1.5 * j, -0.5 * j]
                st.dA_perp[0, :2] = [0.25 * j, 2.0 * j]
            tr.append(st, log_prob=-float(j), algorithm=f"{label}:{j}")
        return tr
    return build


# --------------------------------------------------------------------------
# seed derivation -- what a job array depends on
# --------------------------------------------------------------------------


def test_seeds_are_reproducible_from_the_root():
    assert np.array_equal(derive_seeds(7, 5), derive_seeds(7, 5))


def test_different_roots_give_different_seeds():
    assert not np.array_equal(derive_seeds(7, 5), derive_seeds(8, 5))


def test_one_seed_per_ensemble():
    assert np.asarray(derive_seeds(7, 12)).shape == (12,)


def test_seeds_within_a_run_are_distinct():
    seeds = np.asarray(derive_seeds(7, 32))
    assert len(np.unique(seeds)) == seeds.size


def test_extending_the_count_does_not_renumber_existing_ensembles():
    """Sweeping the ensemble count must be a sweep, not a new experiment.

    docs/phase-4-plan.md §5.2 varies M over 1, 2, 5, 10, 20. If adding
    ensembles reshuffled the earlier seeds, every point on that curve would
    come from a different set of chains and the curve would mean nothing.
    """
    many = np.asarray(derive_seeds(7, 20))
    for m in (1, 2, 5, 10):
        assert np.array_equal(np.asarray(derive_seeds(7, m)), many[:m])


def test_runner_seeds_come_from_derive_seeds(runner):
    assert np.array_equal(runner().seeds(7), derive_seeds(7, 4))


# --------------------------------------------------------------------------
# initialisation policy -- a scientific choice, recorded
# --------------------------------------------------------------------------


def test_spread_across_k_uses_the_requested_dimensions(tiny_site_table):
    init = spread_across_k(lambda s: make_state(s, tiny_site_table), (1, 3))
    rng = np.random.default_rng(0)
    assert {int(init(rng, 0).k[0]), int(init(rng, 1).k[0])} == {1, 3}


def test_spread_across_k_cycles_when_there_are_more_ensembles(tiny_site_table):
    init = spread_across_k(lambda s: make_state(s, tiny_site_table), (1, 3))
    rng = np.random.default_rng(0)
    assert int(init(rng, 0).k[0]) == int(init(rng, 2).k[0])


def test_init_policy_name_is_recorded(runner, target):
    """Which policy was used is provenance, not an implementation detail."""
    assert runner().run(target, root_seed=1).init_name == "spread_across_k"


# --------------------------------------------------------------------------
# running -- one ensemble per task, then all of them
# --------------------------------------------------------------------------


def test_run_produces_one_trace_per_ensemble(runner, target):
    assert len(runner(n_ensembles=3).run(target, root_seed=1).traces) == 3


def test_run_discards_burn_in(runner, target):
    result = runner(n_ensembles=2, n_steps=40, n_burn=10).run(target, root_seed=1)
    assert all(len(t) == 30 for t in result.traces)


def test_run_one_keeps_every_step(runner, target):
    """A task writes what it sampled; how much to discard is decided later."""
    assert len(runner(n_steps=40, n_burn=10).run_one(0, target, root_seed=1)) == 40


def test_run_one_reproduces_that_ensemble_exactly(runner, target):
    """Ensemble j alone must equal ensemble j from the full run.

    This is the property the cluster design rests on: a job array runs each
    ensemble in a separate process, and the pooled result has to be the one a
    single process would have produced.
    """
    full = runner(n_ensembles=3, n_burn=10).run(target, root_seed=5)
    alone = runner(n_ensembles=3, n_burn=10).run_one(1, target, root_seed=5)
    assert np.array_equal(np.asarray(alone.lam)[10:], np.asarray(full.traces[1].lam))


def test_ensembles_do_not_share_state(runner, target):
    result = runner(n_ensembles=3).run(target, root_seed=1)
    assert result.traces[0] is not result.traces[1]


def test_ensembles_explore_differently(runner, target):
    """Different seeds must give different chains, or they are not independent."""
    result = runner(n_ensembles=2, n_steps=60, n_burn=10).run(target, root_seed=1)
    a, b = (np.asarray(t.lam) for t in result.traces)
    assert not np.array_equal(a, b)


def test_seeds_are_recorded_on_the_result(runner, target):
    result = runner(n_ensembles=3).run(target, root_seed=9)
    assert np.array_equal(result.seeds, derive_seeds(9, 3))


def test_burn_in_exceeding_the_budget_raises(schedule, tiny_site_table):
    init = spread_across_k(lambda s: make_state(s, tiny_site_table), (1,))
    with pytest.raises(ValueError):
        EnsembleRunner(schedule, n_ensembles=2, n_steps=10, n_burn=10, init=init)


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------


def test_pooled_length_is_ensembles_times_kept_steps(runner, target):
    result = runner(n_ensembles=4, n_steps=40, n_burn=10).run(target, root_seed=1)
    assert len(result.pooled) == 4 * 30


def test_pooled_contains_every_ensemble(runner, target):
    result = runner(n_ensembles=3, n_steps=40, n_burn=10).run(target, root_seed=1)
    pooled = np.asarray(result.pooled.lam)
    for t in result.traces:
        assert any(np.array_equal(pooled[i * 30:(i + 1) * 30], np.asarray(t.lam))
                   for i in range(3))


def test_merge_traces_concatenates_in_order(trace_of):
    merged = merge_traces([trace_of(n=3), trace_of(n=4)])
    assert len(merged) == 7


def test_merge_preserves_offsets(trace_of):
    merged = merge_traces([trace_of(n=3), trace_of(n=3)])
    assert np.asarray(merged.dA_par)[:3].tolist() == \
        np.asarray(trace_of(n=3).dA_par).tolist()


def test_merge_preserves_algorithm_labels(trace_of):
    merged = merge_traces([trace_of(n=2), trace_of(n=2)])
    assert list(merged.algorithm) == [f"rwmh:lam:{j}" for j in (0, 1, 0, 1)]


def test_merge_equals_the_pooled_property(runner, target):
    result = runner(n_ensembles=3, n_steps=40, n_burn=10).run(target, root_seed=1)
    assert np.array_equal(np.asarray(merge_traces(result.traces).lam),
                          np.asarray(result.pooled.lam))


def test_merging_mismatched_k_max_raises(trace_of):
    """Different k_max means a different model; pooling them is not comparable."""
    with pytest.raises(ValueError):
        merge_traces([trace_of(k_max=8), trace_of(k_max=16)])


def test_merging_nothing_raises(trace_of):
    with pytest.raises(ValueError):
        merge_traces([])


# --------------------------------------------------------------------------
# persistence -- the process boundary a job array crosses
# --------------------------------------------------------------------------


def test_trace_survives_a_save_load_round_trip(trace_of, tmp_path):
    original = trace_of(n=5)
    original.save(tmp_path / "e.npz")
    back = Trace.load(tmp_path / "e.npz")
    for field in ("site_idx", "k", "lam", "n_stretch", "sigma",
                  "dA_par", "dA_perp", "log_prob"):
        assert np.array_equal(np.asarray(getattr(back, field)),
                              np.asarray(getattr(original, field))), field


def test_round_trip_preserves_algorithm_labels(trace_of, tmp_path):
    """Strings, not numbers -- the field most likely to be lost by an npz."""
    original = trace_of(n=4)
    original.save(tmp_path / "e.npz")
    assert list(Trace.load(tmp_path / "e.npz").algorithm) == list(original.algorithm)


def test_round_trip_preserves_the_shape_metadata(trace_of, tmp_path):
    original = trace_of(n=3, k_max=16)
    original.save(tmp_path / "e.npz")
    back = Trace.load(tmp_path / "e.npz")
    assert (back.k_max, back.n_exp, back.n_sites) == \
        (original.k_max, original.n_exp, original.n_sites)


def test_round_trip_preserves_offsets(trace_of, tmp_path):
    """The field a previous bug silently zeroed; it must cross the boundary."""
    original = trace_of(n=4)
    original.save(tmp_path / "e.npz")
    assert np.array_equal(np.asarray(Trace.load(tmp_path / "e.npz").dA_perp),
                          np.asarray(original.dA_perp))


def test_saved_trace_can_still_be_appended_to(trace_of, tiny_site_table, tmp_path):
    original = trace_of(n=3)
    original.save(tmp_path / "e.npz")
    back = Trace.load(tmp_path / "e.npz")
    back.append(make_state([0, 2], tiny_site_table), log_prob=-1.0)
    assert len(back) == 4


def test_result_save_and_load_round_trips(runner, target, tmp_path):
    result = runner(n_ensembles=3).run(target, root_seed=2)
    result.save(tmp_path)
    back = EnsembleResult.load(tmp_path)
    assert len(back.traces) == 3
    assert back.init_name == result.init_name
    assert np.array_equal(back.seeds, result.seeds)


def test_load_ensembles_reads_every_saved_trace(runner, target, tmp_path):
    runner(n_ensembles=4).run(target, root_seed=2).save(tmp_path)
    assert len(load_ensembles(tmp_path)) == 4


def test_merging_loaded_traces_equals_pooling_in_process(runner, target, tmp_path):
    """The whole cluster story in one assertion."""
    result = runner(n_ensembles=3, n_steps=40, n_burn=10).run(target, root_seed=2)
    result.save(tmp_path)
    assert np.array_equal(
        np.asarray(merge_traces(load_ensembles(tmp_path)).lam),
        np.asarray(result.pooled.lam))


def test_loading_a_directory_with_no_traces_raises(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        load_ensembles(tmp_path)


# --------------------------------------------------------------------------
# Gelman-Rubin
# --------------------------------------------------------------------------


def test_rhat_of_agreeing_chains_is_about_one():
    rng = np.random.default_rng(0)
    chains = rng.normal(0.0, 1.0, size=(6, 400))
    assert rhat(chains) == pytest.approx(1.0, abs=0.05)


def test_rhat_rises_when_chains_disagree():
    rng = np.random.default_rng(0)
    chains = rng.normal(0.0, 1.0, size=(4, 200)) + np.array([[0.0], [8.0], [16.0], [24.0]])
    assert rhat(chains) > 2.0


def test_rhat_of_frozen_chains_is_nan_not_infinite():
    """A chain that never moved carries no information about mixing."""
    assert np.isnan(rhat(np.ones((4, 50))))


def test_rhat_needs_at_least_two_chains():
    with pytest.raises(ValueError):
        rhat(np.zeros((1, 50)))


def test_rhat_rejects_unequal_chain_lengths():
    with pytest.raises((ValueError, TypeError)):
        rhat([np.zeros(10), np.zeros(20)])


# --------------------------------------------------------------------------
# agreement -- reports, never judges
# --------------------------------------------------------------------------


def test_agreement_returns_an_agreement(runner, target):
    assert isinstance(runner().run(target, root_seed=1).agreement(), Agreement)


def test_agreement_reports_rhat_for_the_stable_scalars(runner, target):
    got = runner().run(target, root_seed=1).agreement().rhat
    assert set(got) == set(RHAT_PARAMETERS)


def test_rhat_is_not_computed_for_per_spin_quantities():
    """Label switching means there is no 'spin 3' common to two samples."""
    for name in ("site_idx", "dA_par", "dA_perp", "R_i"):
        assert name not in RHAT_PARAMETERS


def test_agreement_reports_the_spread_of_modal_k(runner, target):
    agreement = runner(n_ensembles=3).run(target, root_seed=1).agreement()
    assert agreement.k_mode_spread >= 0


def test_identical_ensembles_show_no_spread_in_k(trace_of):
    same = [trace_of(n=8) for _ in range(3)]
    result = EnsembleResult(traces=same, seeds=np.arange(3), init_name="fixed")
    assert result.agreement().k_mode_spread == 0


def test_agreement_records_the_ensemble_count(runner, target):
    assert runner(n_ensembles=3).run(target, root_seed=1).agreement().n_ensembles == 3


def test_delta_R_is_empty_without_summaries(runner, target):
    """R_i needs a reference; without one the question cannot be asked."""
    assert np.asarray(runner().run(target, root_seed=1)
                      .agreement().max_delta_R).size == 0


def test_delta_R_is_the_largest_pairwise_difference_per_band(runner, target):
    summaries = [
        PosteriorSummary(
            R_i=np.array([r, 1.0]), magnitude=np.array([200.0, 300.0]),
            residual=np.array([1.0]), k_posterior=np.array([2]),
            false_absence=0.0, predictive=np.zeros((1, 4)))
        for r in (0.2, 0.9, 0.5)
    ]
    out = runner().run(target, root_seed=1).agreement(summaries=summaries)
    assert out.max_delta_R[2] == pytest.approx(0.7)


def test_delta_R_is_nan_in_bands_with_no_reference_spins(runner, target):
    summaries = [
        PosteriorSummary(
            R_i=np.array([1.0]), magnitude=np.array([200.0]),
            residual=np.array([1.0]), k_posterior=np.array([2]),
            false_absence=0.0, predictive=np.zeros((1, 4)))
        for _ in range(2)
    ]
    out = runner().run(target, root_seed=1).agreement(summaries=summaries)
    assert np.isnan(out.max_delta_R[0]) and np.isnan(out.max_delta_R[1])


def test_agreement_reports_rather_than_judges():
    """R-hat is structurally above 1 here: measured 2.39 at 4,000 steps and
    2.68 at 20,000, while the pooled mode stayed correct at both. A pass/fail
    field would invite a gate that rejects correct posteriors, so there is
    deliberately none. See docs/phase-4-plan.md §6.
    """
    fields = set(Agreement.__dataclass_fields__)
    for verdict in ("passed", "ok", "converged", "valid", "verdict"):
        assert verdict not in fields
