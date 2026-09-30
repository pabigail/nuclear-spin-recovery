"""Design utilities: expected information gain and predictive variance.

Both take arrays only, so each can be tested against a case whose answer is
known without a forward model: identical predictions carry no information,
well-separated ones carry the prior entropy, and noise divides everything.

The EIG tests are the delicate ones, because the estimator is Monte Carlo and
most of its properties hold in expectation rather than per call.  Each
statistical threshold below was measured on a reference implementation before
the test was written, and the measurement is quoted where it is used:

  ranking under common random numbers   40/40 seeds correct; independent 18/40
  monotone in separation, 64 draws      36/200 seeds non-monotone
  monotone in separation, 1024 draws    0/40 seeds non-monotone
  permutation, 4096 draws               max relative difference 1.9%
  entropy bound, skewed w, 64 draws     one estimate 1.04 against H = 0.80

docs/phase-5-plan.md, unit 5b.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    DesignUtility,
    ExpectedInformationGain,
    PredictiveVariance,
    information_density,
)


def rng(seed=0):
    return np.random.default_rng(seed)


def two_particles(delta, n=10):
    """Particle 0 predicts zero everywhere; particle 1 is offset by ``delta``."""
    P = np.zeros((2, n))
    P[1] = delta
    return P


UNIFORM2 = np.array([0.5, 0.5])
BOTH = [ExpectedInformationGain, PredictiveVariance]


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------


def test_design_utility_is_abstract():
    with pytest.raises(TypeError):
        DesignUtility()


def test_both_utilities_are_design_utilities():
    assert issubclass(ExpectedInformationGain, DesignUtility)
    assert issubclass(PredictiveVariance, DesignUtility)


@pytest.mark.parametrize("utility_cls", BOTH)
def test_score_is_score_many_of_one(utility_cls):
    utility = utility_cls()
    P = two_particles(0.3)
    one = utility.score(P, UNIFORM2, 1.0, rng(3))
    many = utility.score_many([P], UNIFORM2, [1.0], rng(3))
    assert one == pytest.approx(many[0], rel=0, abs=0)


@pytest.mark.parametrize("utility_cls", BOTH)
def test_score_many_returns_one_score_per_candidate(utility_cls):
    """Candidates may have different point counts."""
    utility = utility_cls()
    cands = [two_particles(0.3, n=4), two_particles(0.3, n=9),
             two_particles(0.3, n=1)]
    out = utility.score_many(cands, UNIFORM2, [1.0] * 3, rng())
    assert np.asarray(out).shape == (3,)


@pytest.mark.parametrize("utility_cls", BOTH)
def test_a_weight_per_particle_is_required(utility_cls):
    utility = utility_cls()
    with pytest.raises(ValueError):
        utility.score(two_particles(0.3), np.array([0.2, 0.3, 0.5]), 1.0, rng())


@pytest.mark.parametrize("utility_cls", BOTH)
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_non_positive_or_nan_noise_raises(utility_cls, bad):
    utility = utility_cls()
    noise = np.ones(10)
    noise[4] = bad
    with pytest.raises(ValueError):
        utility.score(two_particles(0.3), UNIFORM2, noise, rng())


@pytest.mark.parametrize("utility_cls", BOTH)
def test_scalar_noise_broadcasts(utility_cls):
    utility = utility_cls()
    P = two_particles(0.3)
    assert utility.score(P, UNIFORM2, 0.5, rng(1)) == pytest.approx(
        utility.score(P, UNIFORM2, np.full(10, 0.5), rng(1)), rel=0, abs=0)


# --------------------------------------------------------------------------
# information density
# --------------------------------------------------------------------------


def test_density_has_one_value_per_point():
    assert information_density(two_particles(0.3), UNIFORM2, 1.0).shape == (10,)


def test_density_is_zero_where_particles_agree_and_positive_where_not():
    P = np.zeros((2, 8))
    P[1, 4:] = 0.5
    dens = information_density(P, UNIFORM2, 1.0)
    np.testing.assert_array_equal(dens[:4], 0.0)
    assert np.all(dens[4:] > 0)


def test_density_is_exactly_zero_for_identical_predictions():
    """Exactly, because selectors test for zero.  Centring on the weighted
    mean left rounding residue whenever the normalised weights missed 1 by an
    ulp: nonzero on 869 of 1000 random rows at weights 0.5/0.3/0.2."""
    w = np.array([0.5, 0.3, 0.2])
    gen = rng(0)
    for _ in range(200):
        P = np.tile(gen.normal(0.5, 0.3, 20), (3, 1))
        assert np.all(information_density(P, w, 0.05) == 0.0)


def test_density_matches_a_hand_computation():
    """Predictions 0 and 2 at equal weight: variance 1; noise 0.5: density 4."""
    P = np.array([[0.0], [2.0]])
    assert information_density(P, UNIFORM2, 0.5)[0] == pytest.approx(4.0)


def test_density_uses_the_weights():
    """Weights 0.9/0.1 on 0 and 1: variance 0.09, not the unweighted 0.25."""
    P = np.array([[0.0], [1.0]])
    assert information_density(P, np.array([0.9, 0.1]), 1.0)[0] == (
        pytest.approx(0.09))


def test_density_scales_as_inverse_noise_variance():
    P = two_particles(0.3)
    np.testing.assert_allclose(information_density(P, UNIFORM2, 2.0),
                               information_density(P, UNIFORM2, 1.0) / 4)


def test_density_is_zero_at_an_unmeasured_point():
    noise = np.ones(10)
    noise[3] = np.inf
    dens = information_density(two_particles(0.3), UNIFORM2, noise)
    assert dens[3] == 0.0
    assert np.all(dens[np.arange(10) != 3] > 0)


# --------------------------------------------------------------------------
# predictive variance
# --------------------------------------------------------------------------


def test_predictive_variance_is_zero_for_identical_predictions():
    P = np.tile(np.linspace(0.2, 0.9, 10), (4, 1))
    assert PredictiveVariance().score(P, np.full(4, 0.25), 1.0, rng()) == 0.0


def test_predictive_variance_of_one_particle_is_zero():
    P = np.linspace(0.2, 0.9, 10)[None, :]
    assert PredictiveVariance().score(P, np.array([1.0]), 1.0, rng()) == 0.0


def test_predictive_variance_is_the_summed_density():
    P = rng(2).normal(size=(5, 12))
    w = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    assert PredictiveVariance().score(P, w, 0.3, rng()) == pytest.approx(
        information_density(P, w, 0.3).sum())


def test_predictive_variance_scales_as_inverse_noise_variance():
    P = two_particles(0.3)
    pv = PredictiveVariance()
    assert pv.score(P, UNIFORM2, 2.0, rng()) == pytest.approx(
        pv.score(P, UNIFORM2, 1.0, rng()) / 4)


def test_predictive_variance_ignores_the_rng():
    P = rng(2).normal(size=(5, 12))
    w = np.full(5, 0.2)
    pv = PredictiveVariance()
    assert pv.score(P, w, 1.0, rng(0)) == pv.score(P, w, 1.0, rng(99))


def test_predictive_variance_is_invariant_to_permuting_particles():
    P = rng(2).normal(size=(5, 12))
    w = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    perm = [3, 0, 4, 1, 2]
    pv = PredictiveVariance()
    assert pv.score(P, w, 1.0, rng()) == pytest.approx(
        pv.score(P[perm], w[perm], 1.0, rng()))


# --------------------------------------------------------------------------
# expected information gain: known answers
# --------------------------------------------------------------------------


def test_n_draws_must_be_positive():
    with pytest.raises(ValueError):
        ExpectedInformationGain(n_draws=0)


def test_eig_is_zero_for_identical_predictions():
    """Nothing is learnable.  Exactly zero, not approximately: every particle
    has the same likelihood, so each draw contributes log 1."""
    P = np.tile(np.linspace(0.2, 0.9, 10), (4, 1))
    assert ExpectedInformationGain().score(P, np.full(4, 0.25), 0.1, rng()) == (
        pytest.approx(0.0, abs=1e-12))


def test_eig_of_one_particle_is_zero():
    P = np.linspace(0.2, 0.9, 10)[None, :]
    assert ExpectedInformationGain().score(P, np.array([1.0]), 0.1, rng()) == (
        pytest.approx(0.0, abs=1e-12))


def test_eig_of_two_separated_particles_is_log_two():
    """Separated by a hundred sigma, every draw identifies its particle: one
    bit, in nats."""
    assert ExpectedInformationGain().score(
        two_particles(100.0), UNIFORM2, 1.0, rng()) == pytest.approx(np.log(2))


def test_eig_with_uniform_weights_never_exceeds_log_K():
    """Per draw the term is at most -log w_k = log K, so this bound is exact,
    not statistical."""
    P = np.array([np.full(5, x) for x in (0.0, 1.0, 2.0, 30.0)])
    w = np.full(4, 0.25)
    eig = ExpectedInformationGain()
    for seed in range(50):
        assert eig.score(P, w, 1.0, rng(seed)) <= np.log(4) + 1e-12


def test_eig_converges_on_the_prior_entropy_for_separated_particles():
    """With skewed weights the bound holds only in expectation; one estimate
    at 64 draws was measured at 1.04 against 0.80.  At 4096 it is within
    0.02 on every seed measured."""
    w = np.array([0.7, 0.2, 0.1])
    P = np.array([np.zeros(5), np.full(5, 50.0), np.full(5, 100.0)])
    entropy = -np.sum(w * np.log(w))
    eig = ExpectedInformationGain(n_draws=4096)
    for seed in range(5):
        assert eig.score(P, w, 1.0, rng(seed)) == pytest.approx(entropy, abs=0.05)


def test_eig_is_deterministic_given_the_seed():
    P = rng(2).normal(size=(5, 12))
    w = np.full(5, 0.2)
    eig = ExpectedInformationGain()
    assert eig.score(P, w, 0.3, rng(7)) == eig.score(P, w, 0.3, rng(7))


def test_eig_is_invariant_to_permuting_particles():
    """In distribution, not per call: a permutation changes which particle a
    given random number selects.  Measured maximum 1.9% at 4096 draws."""
    P = rng(1).normal(0, 0.3, (5, 12))
    w = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
    perm = [3, 0, 4, 1, 2]
    eig = ExpectedInformationGain(n_draws=4096)
    assert eig.score(P, w, 0.2, rng(0)) == pytest.approx(
        eig.score(P[perm], w[perm], 0.2, rng(1)), rel=0.05)


def test_an_unmeasured_point_is_the_same_as_no_point():
    """Infinite noise on the last point equals a candidate without it.  Exact,
    because common random numbers truncate the noise draws per candidate, so
    the remaining points see the same draws."""
    P = rng(4).normal(0, 0.3, (3, 8))
    w = np.array([0.5, 0.3, 0.2])
    noise = np.full(8, 0.2)
    noise[-1] = np.inf
    eig = ExpectedInformationGain()
    full, cut = eig.score_many([P, P[:, :-1]], w, [noise, 0.2], rng(5))
    assert np.isfinite(full)
    assert full == pytest.approx(cut, rel=1e-12)


# --------------------------------------------------------------------------
# expected information gain: comparisons between candidates
# --------------------------------------------------------------------------

SEPARATIONS = [0.05, 0.1, 0.2, 0.4, 0.8]


def test_eig_rises_with_separation():
    """At 64 draws this fails on 36 of 200 seeds -- the estimator's noise is
    comparable to the gaps -- and at 1024 on none of 40."""
    eig = ExpectedInformationGain(n_draws=1024)
    cands = [two_particles(s) for s in SEPARATIONS]
    for seed in range(10):
        scores = eig.score_many(cands, UNIFORM2, [1.0] * len(cands), rng(seed))
        assert np.all(np.diff(scores) > 0), scores


def test_common_random_numbers_keep_a_close_ranking_stable():
    """Candidates 0.012 nats apart (0.101 against 0.089 at 200k draws).
    Measured on the reference: shared draws rank them correctly on 40 of 40
    seeds, independent draws on 18."""
    better, worse = two_particles(0.30), two_particles(0.28)
    shared = ExpectedInformationGain(n_draws=64, common_random=True)
    for seed in range(40):
        a, b = shared.score_many([better, worse], UNIFORM2, [1.0, 1.0], rng(seed))
        assert a > b, f"seed {seed}: {a} <= {b}"


def test_independent_draws_do_flip_that_ranking():
    """The control: without it the test above could pass because the
    candidates are easy to tell apart, not because the draws are shared."""
    better, worse = two_particles(0.30), two_particles(0.28)
    independent = ExpectedInformationGain(n_draws=64, common_random=False)
    flips = sum(
        np.subtract(*independent.score_many([better, worse], UNIFORM2,
                                            [1.0, 1.0], rng(seed))) <= 0
        for seed in range(40))
    assert flips >= 5


@pytest.mark.parametrize("utility_cls", BOTH)
def test_a_zero_weight_particle_changes_nothing(utility_cls, recwarn):
    """Legal, silent, and the same as leaving the particle out.  Appended last
    so the particle draws, which invert the cumulative weights, are unchanged
    too."""
    utility = utility_cls()
    P = rng(1).normal(0, 0.3, (3, 12))
    w = np.array([0.5, 0.3, 0.2])
    padded = np.vstack([P, rng(2).normal(0, 0.3, (2, 12))])
    without = utility.score(P, w, 0.2, rng(4))
    with_zero = utility.score(padded, np.r_[w, 0.0, 0.0], 0.2, rng(4))
    assert with_zero == pytest.approx(without, rel=1e-12)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
