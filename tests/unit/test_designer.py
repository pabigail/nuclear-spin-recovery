"""The experiment designer: rank candidates, propose the next experiment.

The designer is glue -- particles, a utility, a selector and a forward model
-- so most of what is tested here is the glue's contract: one score per
candidate, an Experiment that goes straight back into the sampler, points
already measured never proposed again, and a collapsed posterior reported
rather than ranked.

Three properties are the designer's own and carry the design:

- candidates are compared at **equal time**, so a candidate cannot win by
  having more points -- tested with a grid and the same grid doubled;
- a candidate's **envelope comes from the posterior**, from the measured
  experiment at the same pulse number and field, and a candidate at an
  unmeasured pulse number raises rather than guessing a decay;
- **noise is inherited** unless the candidate sets its own.

Scene: the tiny site table at N = 16.  Over tau in [1e-4, 8e-3] ms the three
particles disagree by up to 0.011; over [1e-6, 1e-5] ms they agree exactly,
spread 0.0.

docs/phase-5-plan.md, unit 5d.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    AnalyticCCE1,
    ExpectedInformationGain,
    Experiment,
    ExperimentDesigner,
    ExperimentSet,
    GaussianL2,
    InformationDensity,
    NothingToLearn,
    ParticleSet,
    PredictiveVariance,
    State,
    StretchedExponential,
    UniformThinning,
    simulate_dataset,
)

K_MAX = 8
B_Z = 311.0
SIGMA = 0.002
INFORMATIVE_TAU = np.linspace(1e-4, 8e-3, 20)
FLAT_TAU = np.linspace(1e-6, 1e-5, 20)


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
    """Deterministic parts, so exact comparisons hold."""
    return ExperimentDesigner(PredictiveVariance(), InformationDensity(), model,
                              tiny_site_table, measured)


@pytest.fixture
def eig_designer(tiny_site_table, model, measured):
    return ExperimentDesigner(ExpectedInformationGain(), InformationDensity(),
                              model, tiny_site_table, measured)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_the_utility_must_be_a_design_utility(tiny_site_table, model, measured):
    with pytest.raises(TypeError):
        ExperimentDesigner(InformationDensity(), InformationDensity(), model,
                           tiny_site_table, measured)


def test_the_selector_must_be_a_point_selector(tiny_site_table, model, measured):
    with pytest.raises(TypeError):
        ExperimentDesigner(PredictiveVariance(), PredictiveVariance(), model,
                           tiny_site_table, measured)


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


# --------------------------------------------------------------------------
# rank
# --------------------------------------------------------------------------


def test_rank_returns_one_score_per_candidate(designer, particles):
    cands = [candidate(INFORMATIVE_TAU), candidate(FLAT_TAU),
             candidate(INFORMATIVE_TAU[:5], n_pulses=8)]
    scores = designer.rank(particles, cands, budget=4.0, rng=rng())
    assert np.asarray(scores).shape == (3,)


def test_a_single_candidate_is_legal(designer, particles):
    scores = designer.rank(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng())
    assert np.asarray(scores).shape == (1,)


def test_an_empty_candidate_list_raises(designer, particles):
    with pytest.raises(ValueError):
        designer.rank(particles, [], budget=4.0, rng=rng())


@pytest.mark.parametrize("which", ["designer", "eig_designer"])
def test_rank_prefers_where_the_particles_disagree(which, particles, request):
    d = request.getfixturevalue(which)
    flat, informative = d.rank(
        particles, [candidate(FLAT_TAU), candidate(INFORMATIVE_TAU)],
        budget=4.0, rng=rng())
    assert informative > flat


def test_candidates_are_compared_at_equal_time(designer, particles):
    """The same grid measured twice over, at the same total time, is worth
    exactly the same: each point gets half the time, so twice the variance."""
    once = candidate(INFORMATIVE_TAU)
    twice = candidate(np.repeat(INFORMATIVE_TAU, 2))
    a, b = designer.rank(particles, [once, twice], budget=4.0, rng=rng())
    assert b == pytest.approx(a, rel=1e-12)


def test_a_larger_budget_scores_higher(designer, particles):
    cand = [candidate(INFORMATIVE_TAU)]
    small = designer.rank(particles, cand, budget=2.0, rng=rng())[0]
    large = designer.rank(particles, cand, budget=8.0, rng=rng())[0]
    assert large == pytest.approx(4 * small)


def test_a_candidate_takes_the_envelope_of_its_pulse_number(
        tiny_site_table, model, measured):
    """Particles whose sampled lambda differs between N = 8 and N = 16 must
    predict an N = 16 candidate with the N = 16 lambda."""
    fast_at_8 = particles_of([[0, 2], [0], [2]], [0.5, 0.3, 0.2],
                             lam=(1e-4, 3e-3))
    fast_at_16 = particles_of([[0, 2], [0], [2]], [0.5, 0.3, 0.2],
                              lam=(3e-3, 1e-4))
    d = ExperimentDesigner(PredictiveVariance(), InformationDensity(), model,
                           tiny_site_table, measured)
    cand = [candidate(INFORMATIVE_TAU, n_pulses=16)]
    slow = d.rank(fast_at_8, cand, budget=4.0, rng=rng())[0]
    fast = d.rank(fast_at_16, cand, budget=4.0, rng=rng())[0]
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
    """Predictive variance goes as 1 / sigma^2."""
    base = candidate(INFORMATIVE_TAU, sigma=SIGMA)
    noisier = candidate(INFORMATIVE_TAU, sigma=2 * SIGMA)
    a, b = designer.rank(particles, [base, noisier], budget=4.0, rng=rng())
    assert b == pytest.approx(a / 4)


def test_rank_on_a_collapsed_posterior_raises(designer, collapsed):
    with pytest.raises(NothingToLearn):
        designer.rank(collapsed, [candidate(INFORMATIVE_TAU)], budget=4.0,
                      rng=rng())


@pytest.mark.parametrize("which", ["designer", "eig_designer"])
def test_rank_where_every_candidate_is_flat_raises(which, particles, request):
    """Several particles, but no candidate can separate them.  Decided on the
    information density, not the utility's score: over FLAT_TAU predictive
    variance is 1.8e-17 but the EIG estimate is -2.7e-10, Monte Carlo noise
    larger than any sensible zero threshold, and of either sign."""
    d = request.getfixturevalue(which)
    with pytest.raises(NothingToLearn):
        d.rank(particles, [candidate(FLAT_TAU)], budget=4.0, rng=rng())


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------


def test_propose_returns_an_experiment_on_the_best_candidate(designer,
                                                             particles):
    best = candidate(INFORMATIVE_TAU)
    out = designer.propose(particles, [candidate(FLAT_TAU), best], budget=4.0,
                           rng=rng())
    assert isinstance(out, Experiment)
    assert out.n_pulses == best.n_pulses and out.b_z == best.b_z
    assert np.all(np.isin(out.tau, best.tau))


def test_proposed_tau_is_sorted(designer, particles):
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng())
    assert np.all(np.diff(out.tau) > 0)


def test_proposed_weights_spend_the_budget(designer, particles):
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=6.5,
                           rng=rng())
    assert out.weight.shape == out.tau.shape
    assert out.weight.sum() == pytest.approx(6.5)


def test_proposed_sigma_is_the_noise_designed_against(designer, particles):
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng())
    assert out.sigma == pytest.approx(SIGMA)
    assert out.data is None


def test_exclude_removes_points_already_measured(designer, particles):
    cand = candidate(INFORMATIVE_TAU)
    done = ExperimentSet([candidate(INFORMATIVE_TAU[10:])])
    out = designer.propose(particles, [cand], budget=4.0, rng=rng(),
                           exclude=done)
    assert not np.any(np.isin(out.tau, INFORMATIVE_TAU[10:]))


def test_exclude_only_applies_to_the_same_pulse_number(designer, particles):
    cand = candidate(INFORMATIVE_TAU, n_pulses=16)
    at_8 = ExperimentSet([candidate(INFORMATIVE_TAU, n_pulses=8)])
    a = designer.propose(particles, [cand], budget=4.0, rng=rng())
    b = designer.propose(particles, [cand], budget=4.0, rng=rng(), exclude=at_8)
    np.testing.assert_array_equal(a.tau, b.tau)


def test_a_candidate_emptied_by_exclude_is_dropped(designer, particles):
    """The informative candidate is fully measured, so the proposal falls to
    the other -- which must still have something to offer."""
    other_window = candidate(np.linspace(1e-3, 6e-3, 15))
    done = ExperimentSet([candidate(INFORMATIVE_TAU)])
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU), other_window],
                           budget=4.0, rng=rng(), exclude=done)
    assert np.all(np.isin(out.tau, other_window.tau))


def test_excluding_every_point_raises(designer, particles):
    done = ExperimentSet([candidate(INFORMATIVE_TAU)])
    with pytest.raises(ValueError):
        designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                         rng=rng(), exclude=done)


def test_propose_on_a_collapsed_posterior_raises(designer, collapsed):
    with pytest.raises(NothingToLearn):
        designer.propose(collapsed, [candidate(INFORMATIVE_TAU)], budget=4.0,
                         rng=rng())


def test_the_uniform_control_works_through_the_designer(
        tiny_site_table, model, measured, particles):
    d = ExperimentDesigner(PredictiveVariance(), UniformThinning(5), model,
                           tiny_site_table, measured)
    out = d.propose(particles, [candidate(INFORMATIVE_TAU)], budget=5.0,
                    rng=rng())
    np.testing.assert_allclose(out.weight, np.ones(5))


def test_the_proposal_goes_straight_back_into_the_likelihood(
        designer, particles, tiny_site_table, model):
    """Simulate the proposed experiment from a truth and score it: the weights
    must reach both the noise and the likelihood without any conversion."""
    out = designer.propose(particles, [candidate(INFORMATIVE_TAU)], budget=4.0,
                           rng=rng())
    truth = State.from_sites([0, 2], n_sites=4, n_exp=1,
                             lam=np.full((1, 1), 3e-3), n_stretch=np.ones((1, 1)),
                             sigma=np.full((1, 1), SIGMA), k_max=K_MAX)
    data = simulate_dataset(truth, ExperimentSet([out]), tiny_site_table, model,
                            sigma=SIGMA, rng=rng(1))
    np.testing.assert_array_equal(data.experiments[0].weight, out.weight)
    assert np.all(np.isfinite(
        GaussianL2().log_prob(truth, data, model, tiny_site_table)))
