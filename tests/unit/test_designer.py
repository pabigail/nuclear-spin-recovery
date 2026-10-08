"""The experiment designer: one experiment out, or the finding that none helps.

What is tested here is the procedure the designer carries out:

- **pulse number and delays are chosen together** -- every candidate is given
  its own best delays and judged on that design, so a candidate with one
  sharp feature among many dull points is not diluted out of contention;
- candidates are compared at **equal time**, and time is always charged:
  every design spends exactly the budget at ``2 N tau`` a repetition;
- **nothing to tell apart is an answer** -- below a gain threshold the result
  carries no experiment, and nothing is raised;
- a candidate's **envelope comes from the posterior**, from the measured
  experiment at the same pulse number and field, and a candidate at an
  unmeasured pulse number raises rather than guessing a decay;
- **noise is inherited** unless the candidate sets its own.

Exact relations are checked with the deterministic stand-ins of
``design_doubles``; the shipped Monte Carlo utility is used wherever the
result, not the arithmetic, is the point.

Scene: the tiny site table at N = 16.  Over tau in [1e-4, 8e-3] ms the three
particles disagree by up to 0.011; over [1e-6, 1e-5] ms they agree to within
1e-9 of the noise.

docs/phase-5-plan.md, unit 5d; notebooks/adaptive_design_prototype.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    CandidateDesign,
    DesignResult,
    ExpectedInformationGain,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    GaussianL2,
    InformationDensity,
    ParticleSet,
    SequenceDuration,
    State,
    StretchedExponential,
    simulate_dataset,
)

from .design_doubles import EvenSpread, TotalDensity

K_MAX = 8
B_Z = 311.0
SIGMA = 0.002
INFORMATIVE_TAU = np.linspace(1e-4, 8e-3, 20)
FLAT_TAU = np.linspace(1e-6, 1e-5, 20)


def unit_cost(experiment):
    """Every repetition costs the same; for tests of everything but cost."""
    return np.ones(len(experiment.tau))


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.fixture
def model():
    return AnalyticCCE1(StretchedExponential())


@pytest.fixture
def measured():
    """What the posterior was fitted to: N = 8 then N = 16."""
    return ExperimentSet([
        Experiment(tau=np.linspace(2e-4, 6e-3, 12), n_pulses=8, b_z=B_Z),
        Experiment(tau=np.linspace(2e-4, 6e-3, 12), n_pulses=16, b_z=B_Z),
    ])


def particles_of(configs, weight, *, sigma=(SIGMA, SIGMA), lam=(3e-3, 3e-3)):
    """A particle set over the two measured experiments."""
    n = len(configs)
    site_idx = np.full((n, K_MAX), -1, dtype=int)
    for i, c in enumerate(configs):
        site_idx[i, : len(c)] = c
    return ParticleSet(
        site_idx=site_idx, k=np.array([len(c) for c in configs]),
        weight=np.asarray(weight, float),
        dA_par=np.zeros((n, K_MAX)), dA_perp=np.zeros((n, K_MAX)),
        lam=np.tile(lam, (n, 1)), n_stretch=np.ones((n, 2)),
        sigma=np.tile(sigma, (n, 1)), n_sites=4, k_max=K_MAX)


@pytest.fixture
def particles():
    return particles_of([[0, 2], [0], [2]], [0.5, 0.3, 0.2])


@pytest.fixture
def collapsed():
    return particles_of([[0, 2]], [1.0])


def candidate(tau, n_pulses=16, **kw):
    return Experiment(tau=np.asarray(tau, float), n_pulses=n_pulses, b_z=B_Z,
                      **kw)


@pytest.fixture
def designer(tiny_site_table, model, measured):
    """Deterministic utility and unit cost, so exact comparisons hold."""
    return ExperimentDesigner(model, tiny_site_table, measured,
                              utility=TotalDensity(), cost=unit_cost)


@pytest.fixture
def default_designer(tiny_site_table, model, measured):
    """As shipped: expected information gain, density selector, 2 N tau."""
    return ExperimentDesigner(model, tiny_site_table, measured)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_defaults_are_the_shipped_procedure(default_designer):
    assert isinstance(default_designer.utility, ExpectedInformationGain)
    assert isinstance(default_designer.selector, InformationDensity)
    assert isinstance(default_designer.cost, SequenceDuration)
    assert default_designer.cost.overhead == 0.0
    assert default_designer.min_gain == pytest.approx(0.05)


def test_the_utility_must_be_a_design_utility(tiny_site_table, model, measured):
    with pytest.raises(TypeError):
        ExperimentDesigner(model, tiny_site_table, measured,
                           utility=InformationDensity())


def test_the_selector_must_be_a_point_selector(tiny_site_table, model, measured):
    with pytest.raises(TypeError):
        ExperimentDesigner(model, tiny_site_table, measured,
                           selector=TotalDensity())


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_the_threshold_must_be_a_non_negative_number(tiny_site_table, model,
                                                     measured, bad):
    with pytest.raises(ValueError, match="min_gain"):
        ExperimentDesigner(model, tiny_site_table, measured, min_gain=bad)


def test_particles_must_match_the_measured_experiments(designer):
    """Envelope columns are indexed by measured experiment; a posterior fitted
    to a different number of experiments cannot be read against this one."""
    n = 2
    wrong = ParticleSet(
        site_idx=np.full((n, K_MAX), -1), k=np.zeros(n, int), weight=[1, 1],
        dA_par=np.zeros((n, K_MAX)), dA_perp=np.zeros((n, K_MAX)),
        lam=np.full((n, 3), 3e-3), n_stretch=np.ones((n, 3)),
        sigma=np.full((n, 3), SIGMA), n_sites=4, k_max=K_MAX)
    with pytest.raises(ValueError):
        designer.rank(wrong, [candidate(INFORMATIVE_TAU)], budget=4.0, rng=rng())


@pytest.mark.parametrize("budget", [0.0, -1.0, np.nan])
def test_the_budget_must_be_positive(designer, particles, budget):
    with pytest.raises(ValueError, match="budget"):
        designer.rank(particles, [candidate(INFORMATIVE_TAU)], budget=budget,
                      rng=rng())


# --------------------------------------------------------------------------
# every candidate gets a design
# --------------------------------------------------------------------------


def test_one_design_per_candidate_in_the_order_given(designer, particles):
    cands = [candidate(INFORMATIVE_TAU), candidate(FLAT_TAU),
             candidate(INFORMATIVE_TAU[:5], n_pulses=8)]
    designs = designer.designs(particles, cands, budget=4.0, rng=rng())
    assert all(isinstance(d, CandidateDesign) for d in designs)
    assert [d.n_pulses for d in designs] == [16, 16, 8]
    assert designer.rank(particles, cands, budget=4.0, rng=rng()).shape == (3,)


def test_a_single_candidate_is_legal(designer, particles):
    scores = designer.rank(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng())
    assert scores.shape == (1,)


def test_an_empty_candidate_list_raises(designer, particles):
    with pytest.raises(ValueError):
        designer.rank(particles, [], budget=4.0, rng=rng())


def test_a_design_measures_a_subset_of_its_candidates_delays(designer,
                                                             particles):
    (d,) = designer.designs(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                            rng=rng())
    assert 0 < d.index.size < INFORMATIVE_TAU.size
    assert np.all(np.isin(d.tau, INFORMATIVE_TAU))
    assert np.all(np.diff(d.tau) > 0)
    assert d.weight.shape == d.tau.shape and np.all(d.weight > 0)
    assert d.signals.shape == (3, INFORMATIVE_TAU.size)
    assert d.density.shape == d.rate.shape == (INFORMATIVE_TAU.size,)


def test_a_design_sits_where_the_baths_disagree(designer, particles):
    """Delays of zero density are never measured; the densest one always is."""
    (d,) = designer.designs(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                            rng=rng())
    assert np.all(d.density[d.index] > 0)
    assert int(np.argmax(d.density)) in d.index


@pytest.mark.parametrize("which", ["designer", "default_designer"])
def test_rank_prefers_where_the_particles_disagree(which, particles, request):
    d = request.getfixturevalue(which)
    flat, informative = d.rank(
        particles, [candidate(FLAT_TAU), candidate(INFORMATIVE_TAU)],
        budget=4.0, rng=rng())
    assert informative > flat


# --------------------------------------------------------------------------
# pulse number and delays are chosen together
# --------------------------------------------------------------------------


def test_a_candidate_is_judged_on_its_best_delays_not_its_whole_grid(
        designer, particles):
    """The case that separates the joint choice from a two-stage one.

    ``peaked`` is two thousand delays on which the baths agree, plus the two
    best delays there are.  ``medium`` is eight ordinary ones.  Spread evenly
    over its grid, ``peaked`` wastes nearly all of its time and scores below
    ``medium``; given its own best delays it is far ahead.  A designer that
    picked the pulse number on the evenly measured grid and only then chose
    delays would take ``medium``.
    """
    (full,) = designer.designs(particles, [candidate(INFORMATIVE_TAU)],
                               budget=4.0, rng=rng())
    by_density = np.argsort(full.density)
    peaked = candidate(np.sort(np.concatenate(
        [np.linspace(1e-6, 1e-5, 2000), INFORMATIVE_TAU[by_density[-2:]]])))
    medium = candidate(np.sort(INFORMATIVE_TAU[by_density[8:16]]))

    result = designer.propose(particles, [peaked, medium], budget=4.0, rng=rng())
    a, b = result.designs

    def evenly(d):
        return float(np.sum(d.density / d.cost) * 4.0 / d.candidate.tau.size)

    assert evenly(a) < evenly(b), "the scene no longer separates the two rules"
    assert a.gain > b.gain
    assert a.chosen and not b.chosen
    assert np.all(np.isin(result.experiment.tau,
                          INFORMATIVE_TAU[by_density[-2:]]))


def test_the_same_delays_offered_twice_over_are_worth_the_same(designer,
                                                               particles):
    """Equal time: each copy of a delay gets half of it."""
    once = candidate(INFORMATIVE_TAU)
    twice = candidate(np.repeat(INFORMATIVE_TAU, 2))
    a, b = designer.rank(particles, [once, twice], budget=4.0, rng=rng())
    assert b == pytest.approx(a, rel=1e-12)


def test_a_larger_budget_scores_higher(designer, particles):
    cand = [candidate(INFORMATIVE_TAU)]
    small = designer.rank(particles, cand, budget=2.0, rng=rng())[0]
    large = designer.rank(particles, cand, budget=8.0, rng=rng())[0]
    assert large == pytest.approx(4 * small)


def test_candidates_share_their_random_numbers(default_designer, particles):
    """Two copies of one candidate must score identically, whatever the seed:
    a comparison between candidates must not be a comparison of their luck."""
    same = [candidate(INFORMATIVE_TAU), candidate(INFORMATIVE_TAU)]
    for seed in range(3):
        a, b = default_designer.rank(particles, same, budget=1e-3, rng=rng(seed))
        assert a == b


# --------------------------------------------------------------------------
# time is always charged
# --------------------------------------------------------------------------


def test_every_design_spends_exactly_the_budget(default_designer, particles):
    cands = [candidate(INFORMATIVE_TAU, n_pulses=8), candidate(INFORMATIVE_TAU)]
    for d in default_designer.designs(particles, cands, budget=6.5, rng=rng()):
        assert d.time.sum() == pytest.approx(6.5)
        np.testing.assert_allclose(
            d.cost, 2.0 * d.n_pulses * d.candidate.tau, rtol=1e-12)


def test_a_repetition_costs_two_n_tau_by_default(default_designer, particles):
    out = default_designer.propose(particles, [candidate(INFORMATIVE_TAU)],
                                   budget=6.5, rng=rng()).experiment
    spent = np.sum(out.weight * 2.0 * out.n_pulses * out.tau)
    assert spent == pytest.approx(6.5)


def test_a_slow_delay_is_repeated_less_for_the_same_time(tiny_site_table,
                                                         model, measured,
                                                         particles):
    """With every kept delay given equal time, the weight is inversely
    proportional to the delay: later means fewer repetitions and more noise."""
    d = ExperimentDesigner(model, tiny_site_table, measured,
                           utility=TotalDensity(), selector=EvenSpread(20))
    out = d.propose(particles, [candidate(INFORMATIVE_TAU)], budget=5.0,
                    rng=rng()).experiment
    np.testing.assert_allclose(out.weight * out.tau,
                               (out.weight * out.tau)[0], rtol=1e-12)
    assert np.all(np.diff(out.weight) < 0)


def test_a_custom_cost_is_honoured(tiny_site_table, model, measured, particles):
    d = ExperimentDesigner(model, tiny_site_table, measured,
                           utility=TotalDensity(), selector=EvenSpread(20),
                           cost=unit_cost)
    out = d.propose(particles, [candidate(INFORMATIVE_TAU)], budget=20.0,
                    rng=rng()).experiment
    np.testing.assert_allclose(out.weight, np.ones(20))


@pytest.mark.parametrize("bad", [lambda e: np.ones(3),
                                 lambda e: np.zeros(len(e.tau)),
                                 lambda e: np.full(len(e.tau), np.nan)])
def test_a_cost_must_be_positive_at_every_point(tiny_site_table, model,
                                                measured, particles, bad):
    d = ExperimentDesigner(model, tiny_site_table, measured, cost=bad)
    with pytest.raises(ValueError, match="cost"):
        d.rank(particles, [candidate(INFORMATIVE_TAU)], budget=4.0, rng=rng())


# --------------------------------------------------------------------------
# envelope and noise come from the posterior
# --------------------------------------------------------------------------


def test_a_candidate_takes_the_envelope_of_its_pulse_number(designer):
    """Particles whose sampled lambda differs between N = 8 and N = 16 must
    predict an N = 16 candidate with the N = 16 lambda."""
    fast_at_8 = particles_of([[0, 2], [0], [2]], [0.5, 0.3, 0.2],
                             lam=(1e-4, 3e-3))
    fast_at_16 = particles_of([[0, 2], [0], [2]], [0.5, 0.3, 0.2],
                              lam=(3e-3, 1e-4))
    cand = [candidate(INFORMATIVE_TAU, n_pulses=16)]
    slow = designer.rank(fast_at_8, cand, budget=4.0, rng=rng())[0]
    fast = designer.rank(fast_at_16, cand, budget=4.0, rng=rng())[0]
    # A lambda of 0.1 us kills the modulation across the window, and with it
    # the disagreement.
    assert fast < 0.01 * slow


def test_an_unmeasured_pulse_number_raises(designer, particles):
    with pytest.raises(ValueError, match="n_pulses|pulse"):
        designer.rank(particles, [candidate(INFORMATIVE_TAU, n_pulses=32)],
                      budget=4.0, rng=rng())


def test_an_unmeasured_field_raises(designer, particles):
    other = Experiment(tau=INFORMATIVE_TAU, n_pulses=16, b_z=400.0)
    with pytest.raises(ValueError, match="b_z|field"):
        designer.rank(particles, [other], budget=4.0, rng=rng())


def test_noise_is_inherited_from_the_posterior(designer):
    """Particle-weighted mean of the sampled sigma for the matched experiment:
    here 0.5 * 0.002 + 0.5 * 0.004 = 0.003 at N = 16."""
    mixed = particles_of([[0, 2], [0]], [0.5, 0.5], sigma=(0.5, SIGMA))
    mixed.sigma[1, 1] = 2 * SIGMA
    explicit = candidate(INFORMATIVE_TAU, sigma=0.003)
    inherited = candidate(INFORMATIVE_TAU)
    a, b = designer.rank(mixed, [explicit, inherited], budget=4.0, rng=rng())
    assert b == pytest.approx(a, rel=1e-12)


def test_a_candidate_may_set_its_own_noise(designer, particles):
    """Information density goes as 1 / sigma^2."""
    base = candidate(INFORMATIVE_TAU, sigma=SIGMA)
    noisier = candidate(INFORMATIVE_TAU, sigma=2 * SIGMA)
    a, b = designer.rank(particles, [base, noisier], budget=4.0, rng=rng())
    assert b == pytest.approx(a / 4)


# --------------------------------------------------------------------------
# propose: one experiment
# --------------------------------------------------------------------------


def test_propose_returns_one_experiment_on_the_best_candidate(default_designer,
                                                              particles):
    best = candidate(INFORMATIVE_TAU)
    result = default_designer.propose(particles, [candidate(FLAT_TAU), best],
                                      budget=4.0, rng=rng())
    assert isinstance(result, DesignResult)
    assert result.distinguishable
    out = result.experiment
    assert isinstance(out, Experiment)
    assert out.n_pulses == best.n_pulses and out.b_z == best.b_z
    assert np.all(np.isin(out.tau, best.tau))
    assert [d.chosen for d in result.designs] == [False, True]
    assert result.chosen is result.designs[1]
    assert result.gain == result.designs[1].gain


def test_proposed_tau_is_sorted(designer, particles):
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng()).experiment
    assert np.all(np.diff(out.tau) > 0)


def test_proposed_sigma_is_the_noise_designed_against(designer, particles):
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng()).experiment
    assert out.sigma == pytest.approx(SIGMA)
    assert out.data is None


def test_the_result_keeps_every_candidates_design(designer, particles):
    """The comparison behind the answer, without recomputing it."""
    cands = [candidate(INFORMATIVE_TAU), candidate(INFORMATIVE_TAU[:8])]
    result = designer.propose(particles, cands, budget=4.0, rng=rng())
    assert len(result.designs) == 2
    assert result.budget == 4.0
    np.testing.assert_array_equal(
        [d.gain for d in result.designs],
        designer.rank(particles, cands, budget=4.0, rng=rng()))


def test_the_result_prints_the_comparison_and_the_decision(default_designer,
                                                           particles):
    text = str(default_designer.propose(
        particles, [candidate(INFORMATIVE_TAU), candidate(FLAT_TAU)],
        budget=4.0, rng=rng()))
    assert "<- chosen" in text
    assert "Run CPMG-16 at" in text


def test_the_proposal_goes_straight_back_into_the_likelihood(
        default_designer, particles, tiny_site_table, model):
    """Simulate the proposed experiment from a truth and score it: the weights
    must reach both the noise and the likelihood without any conversion."""
    out = default_designer.propose(particles, [candidate(INFORMATIVE_TAU)],
                                   budget=4.0, rng=rng()).experiment
    truth = State.from_sites([0, 2], n_sites=4, n_exp=1,
                             lam=np.full((1, 1), 3e-3), n_stretch=np.ones((1, 1)),
                             sigma=np.full((1, 1), SIGMA), k_max=K_MAX)
    data = simulate_dataset(truth, ExperimentSet([out]), tiny_site_table, model,
                            sigma=SIGMA, rng=rng(1))
    np.testing.assert_array_equal(data.experiments[0].weight, out.weight)
    assert np.all(np.isfinite(
        GaussianL2().log_prob(truth, data, model, tiny_site_table)))


# --------------------------------------------------------------------------
# exclude
# --------------------------------------------------------------------------


def test_exclude_removes_points_already_measured(designer, particles):
    cand = candidate(INFORMATIVE_TAU)
    done = ExperimentSet([candidate(INFORMATIVE_TAU[10:])])
    out = designer.propose(particles, [cand], budget=4.0, rng=rng(),
                           exclude=done).experiment
    assert not np.any(np.isin(out.tau, INFORMATIVE_TAU[10:]))


def test_exclude_only_applies_to_the_same_pulse_number(designer, particles):
    cand = candidate(INFORMATIVE_TAU, n_pulses=16)
    at_8 = ExperimentSet([candidate(INFORMATIVE_TAU, n_pulses=8)])
    a = designer.propose(particles, [cand], budget=4.0, rng=rng()).experiment
    b = designer.propose(particles, [cand], budget=4.0, rng=rng(),
                         exclude=at_8).experiment
    np.testing.assert_array_equal(a.tau, b.tau)


def test_a_candidate_emptied_by_exclude_is_dropped(designer, particles):
    """The informative candidate is fully measured, so the proposal falls to
    the other -- which must still have something to offer."""
    other_window = candidate(np.linspace(1e-3, 6e-3, 15))
    done = ExperimentSet([candidate(INFORMATIVE_TAU)])
    result = designer.propose(particles,
                              [candidate(INFORMATIVE_TAU), other_window],
                              budget=4.0, rng=rng(), exclude=done)
    assert len(result.designs) == 1
    assert np.all(np.isin(result.experiment.tau, other_window.tau))


def test_excluding_every_point_raises(designer, particles):
    """Different from nothing being distinguishable: there is nothing left to
    offer at all, which is a mistake in the candidate list."""
    done = ExperimentSet([candidate(INFORMATIVE_TAU)])
    with pytest.raises(ValueError, match="already been measured"):
        designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                         rng=rng(), exclude=done)


# --------------------------------------------------------------------------
# nothing to tell apart is an answer
# --------------------------------------------------------------------------


def declined(result):
    return (result.experiment is None and not result.distinguishable
            and result.chosen is None
            and not any(d.chosen for d in result.designs))


def test_a_collapsed_posterior_gets_no_experiment(default_designer, collapsed):
    """One bath: whatever is measured, there is nothing to choose between."""
    result = default_designer.propose(collapsed, [candidate(INFORMATIVE_TAU)],
                                      budget=4.0, rng=rng())
    assert declined(result)
    assert result.gain == 0.0
    assert result.designs[0].index.size == 0


def test_candidates_on_which_the_baths_agree_get_no_experiment(
        default_designer, particles):
    """Several baths, but no candidate can separate them.  The gain estimate
    here is Monte Carlo noise around zero, of either sign, which is why the
    test is a threshold and not a comparison with zero."""
    result = default_designer.propose(
        particles, [candidate(FLAT_TAU), candidate(FLAT_TAU, n_pulses=8)],
        budget=4.0, rng=rng())
    assert declined(result)
    assert abs(result.gain) < 1e-3
    assert "No experiment on this list can tell the baths apart" in str(result)


def test_a_gain_under_the_threshold_gets_no_experiment(tiny_site_table, model,
                                                       measured, particles):
    """The same design is proposed or declined by the threshold alone."""
    cands = [candidate(INFORMATIVE_TAU)]

    def run(min_gain):
        d = ExperimentDesigner(model, tiny_site_table, measured,
                               min_gain=min_gain)
        return d.propose(particles, cands, budget=2e-3, rng=rng())

    gain = run(0.0).gain
    assert 0.05 < gain < 1.0, "the scene should gain something, but not much"
    assert run(0.9 * gain).distinguishable
    assert declined(run(1.1 * gain))
    assert run(1.1 * gain).gain == gain


def test_a_zero_threshold_still_declines_when_there_is_no_design(
        tiny_site_table, model, measured, collapsed):
    d = ExperimentDesigner(model, tiny_site_table, measured, min_gain=0.0)
    assert declined(d.propose(collapsed, [candidate(INFORMATIVE_TAU)],
                              budget=4.0, rng=rng()))


def test_declining_does_not_raise_for_any_candidate_mix(default_designer,
                                                        particles):
    """One candidate that helps among several that do not is still found."""
    result = default_designer.propose(
        particles, [candidate(FLAT_TAU), candidate(INFORMATIVE_TAU),
                    candidate(FLAT_TAU, n_pulses=8)], budget=4.0, rng=rng())
    assert [d.chosen for d in result.designs] == [False, True, False]


# --------------------------------------------------------------------------
# the envelope is a nuisance: design for the spins only
# --------------------------------------------------------------------------


def _with_lams(configs, weight, lam16):
    """Particles whose N = 16 lambda differs from particle to particle."""
    ps = particles_of(configs, weight)
    ps.lam[:, 1] = lam16
    return ps


def test_baths_that_differ_only_in_lambda_carry_nothing_to_learn(
        default_designer):
    """The same spins under two decay constants are one physical hypothesis.
    With the shared envelope the designer sees no disagreement at all."""
    ps = _with_lams([[0, 2], [0, 2]], [0.5, 0.5], [1e-3, 6e-3])
    result = default_designer.propose(ps, [candidate(INFORMATIVE_TAU)],
                                      budget=4.0, rng=rng())
    assert declined(result)
    assert np.all(result.designs[0].density == 0.0)


def test_per_particle_envelopes_would_pay_to_learn_lambda(
        tiny_site_table, model, measured):
    """The control: switched off, the same pair looks informative -- which
    is exactly the design value the default refuses to chase."""
    ps = _with_lams([[0, 2], [0, 2]], [0.5, 0.5], [1e-3, 6e-3])
    d = ExperimentDesigner(model, tiny_site_table, measured,
                           shared_envelope=False)
    assert d.propose(ps, [candidate(INFORMATIVE_TAU)], budget=4.0,
                     rng=rng()).distinguishable


def test_the_shared_envelope_is_the_weighted_mean(designer):
    """Different baths with different lambdas score exactly as the same baths
    all given the posterior-mean lambda."""
    w = [0.5, 0.3, 0.2]
    lams = np.array([2e-3, 3e-3, 5e-3])
    spread = _with_lams([[0, 2], [0], [2]], w, lams)
    mean = _with_lams([[0, 2], [0], [2]], w, np.full(3, np.dot(w, lams)))
    cands = [candidate(INFORMATIVE_TAU)]
    np.testing.assert_allclose(
        designer.rank(spread, cands, budget=4.0, rng=rng()),
        designer.rank(mean, cands, budget=4.0, rng=rng()), rtol=1e-12)


# --------------------------------------------------------------------------
# gain against budget
# --------------------------------------------------------------------------


def test_gain_curve_is_budgets_by_candidates(default_designer, particles):
    cands = [candidate(INFORMATIVE_TAU, n_pulses=8), candidate(INFORMATIVE_TAU)]
    curve = default_designer.gain_curve(particles, cands, [1e-5, 1e-4, 1e-3])
    assert curve.shape == (3, 2)
    assert np.all(np.diff(curve, axis=0) > 0), "more time must gain more"


def test_gain_curve_matches_rank_at_each_budget(default_designer, particles):
    cands = [candidate(INFORMATIVE_TAU, n_pulses=8), candidate(INFORMATIVE_TAU)]
    curve = default_designer.gain_curve(particles, cands, [1e-4], seed=3)
    np.testing.assert_array_equal(
        curve[0], default_designer.rank(particles, cands, budget=1e-4,
                                        rng=rng(3)))


def test_gain_curve_takes_a_budget_per_candidate(default_designer, particles):
    """Equal repetitions, not equal time: each candidate is given a multiple
    of its own experiment time."""
    cands = [candidate(INFORMATIVE_TAU, n_pulses=8), candidate(INFORMATIVE_TAU)]
    own = np.array([default_designer.cost_of(c).sum() for c in cands])
    assert own[1] == pytest.approx(2 * own[0])
    curve = default_designer.gain_curve(particles, cands, 1e-3 * own[None, :])
    for c, cand in enumerate(cands):
        alone = default_designer.rank(particles, [cand], budget=1e-3 * own[c],
                                      rng=rng(0))[0]
        assert curve[0, c] == alone
    with pytest.raises(ValueError, match="budgets"):
        default_designer.gain_curve(particles, cands, np.ones((2, 3)))
